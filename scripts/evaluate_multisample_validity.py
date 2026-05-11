#!/usr/bin/env python
"""Evaluate AdsorbGen checkpoints with repeated samples per input system.

This is a thin analysis wrapper around ``revalidate_pristine.py``.  It reports
both attempt-level strict validity rates and system-level valid@K, where a
system is counted valid if any of its K generated attempts is valid.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from adsorbgen.dataset import PlacementPriorDataset, collate_displacement
from adsorbgen.eval import compute_anomaly_metrics, compute_displacement_metrics
from adsorbgen.flow import FlowConfig

from revalidate_pristine import _generate_records, _load_module


CKPT_RE = re.compile(r"ckpt_epochepoch=(\d+)\.ckpt$")


def _epoch_from_path(path: Path) -> int:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def _rates_from_per_sample(per_sample: list[dict]) -> dict:
    n = len(per_sample)
    if n == 0:
        return {
            "n": 0,
            "valid_rate_strict": None,
            "any_anomaly_rate": None,
            "overlap_rate": None,
            "dissoc_rate": None,
            "desorbed_rate": None,
            "intercalated_rate": None,
            "surf_changed_rate": None,
        }

    def frac(key: str) -> float:
        return sum(1 for row in per_sample if row.get(key) is True) / n

    any_anom = frac("is_any_anomaly")
    return {
        "n": n,
        "valid_rate_strict": 1.0 - any_anom,
        "any_anomaly_rate": any_anom,
        "overlap_rate": frac("has_overlap"),
        "dissoc_rate": frac("has_dissoc"),
        "desorbed_rate": frac("has_desorbed"),
        "intercalated_rate": frac("has_intercalated"),
        "surf_changed_rate": frac("has_surf_changed"),
    }


def _valid_at_k(per_sample: list[dict], base_n: int, k: int) -> dict:
    groups = [[] for _ in range(base_n)]
    for idx, row in enumerate(per_sample):
        groups[idx % base_n].append(row)
    groups = [g[:k] for g in groups if g]

    any_valid = []
    all_invalid_rows = []
    for g in groups:
        valid_flags = [row.get("valid_strict") is True for row in g]
        ok = any(valid_flags)
        any_valid.append(ok)
        if not ok:
            all_invalid_rows.extend(g)

    out = {
        "n_systems": len(groups),
        "k": k,
        "valid_at_k": sum(any_valid) / max(len(any_valid), 1),
        "all_attempts_invalid_rate": 1.0 - sum(any_valid) / max(len(any_valid), 1),
    }
    invalid_rates = _rates_from_per_sample(all_invalid_rows)
    for key, value in invalid_rates.items():
        if key != "n":
            out[f"failed_system_attempts_{key}"] = value
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--val-lmdb", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="")
    p.add_argument("--max-val-samples", type=int, default=1000)
    p.add_argument("--sample-repeats", type=int, default=1)
    p.add_argument("--sample-eval-steps", type=int, default=20)
    p.add_argument("--prediction-type", choices=["x1", "v"], default="x1")
    p.add_argument("--prior-mode", type=str, default="random_heuristic")
    p.add_argument("--translation-std", type=float, default=0.0)
    p.add_argument("--interstitial-gap", type=float, default=0.1)
    p.add_argument("--pristine-slabs", type=Path, required=True)
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--refine-final", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    with open(run_dir / "args.json") as f:
        args_blob = json.load(f)

    device = torch.device(args.device)
    model, model_cfg = _load_module(args.ckpt_path.resolve(), args_blob, device)
    flow_cfg = FlowConfig(eps=1e-5, prediction_type=args.prediction_type)

    base_ds = PlacementPriorDataset(
        args.val_lmdb,
        max_samples=args.max_val_samples,
        training_aug=False,
        translation_std=args.translation_std,
        prior_mode=args.prior_mode,
        interstitial_gap=args.interstitial_gap,
        provide_ads_ref_pos=bool(args_blob["model_config"].get("use_ads_ref_pos", False)),
        skip_anomaly=True,
    )
    val_ds = torch.utils.data.ConcatDataset([base_ds] * args.sample_repeats)
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_displacement,
    )

    t0 = time.time()
    target_n = len(base_ds) * int(args.sample_repeats)
    records = _generate_records(
        model,
        model_cfg,
        loader,
        flow_cfg,
        sample_eval_steps=args.sample_eval_steps,
        max_samples=target_n,
        device=device,
        refine_final=args.refine_final,
    )
    disp = compute_displacement_metrics(records)
    strict = compute_anomaly_metrics(
        records,
        pristine_slabs=args.pristine_slabs,
        pristine_sid_index=args.pristine_sid_index,
    )
    per_sample = strict["per_sample"]
    out = {
        "run_dir": str(run_dir),
        "ckpt": str(args.ckpt_path.resolve()),
        "epoch": _epoch_from_path(args.ckpt_path),
        "dataset_name": args.dataset_name or Path(args.val_lmdb).stem,
        "val_lmdb": args.val_lmdb,
        "max_val_samples": len(base_ds),
        "sample_repeats": int(args.sample_repeats),
        "sample_eval_steps": int(args.sample_eval_steps),
        "refine_final": bool(args.refine_final),
        "attempt_level": {
            **strict["aggregate"],
            "displacement_mae_A": disp["aggregate"]["displacement_mae_A"],
            "displacement_rmse_A": disp["aggregate"]["displacement_rmse_A"],
        },
        "system_level": _valid_at_k(per_sample, len(base_ds), int(args.sample_repeats)),
        "elapsed_s": time.time() - t0,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(
        f"[done] {args.dataset_name or args.val_lmdb} ep={out['epoch']} "
        f"K={args.sample_repeats} attempt_valid={out['attempt_level']['valid_rate_strict']:.3f} "
        f"valid@K={out['system_level']['valid_at_k']:.3f} "
        f"elapsed={out['elapsed_s']:.1f}s"
    )


if __name__ == "__main__":
    main()
