#!/usr/bin/env python
"""Diagnose which atom-pair classes cause AdsorbGen overlap flags.

This reuses the same checkpoint sampling path as ``revalidate_pristine.py`` but
adds a breakdown of the minimum MIC distance by tag-pair class:

  tag 0: subsurface/bulk slab atoms
  tag 1: surface slab atoms
  tag 2: adsorbate atoms

The existing strict overlap metric is any pair with distance < 0.5 A across all
real atoms. This script reports which tag-pair classes account for those hits,
so auxiliary pair-distance losses can be weighted from evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from adsorbgen.eval import OVERLAP_MIN_DIST_A, compute_anomaly_metrics

# Reuse the already-tested revalidation plumbing rather than duplicating model
# reconstruction and sampling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from revalidate_pristine import (  # noqa: E402
    _build_cfg_from_blob,
    _generate_records,
    _load_module,
)
from adsorbgen.dataset import PlacementPriorDataset, collate_displacement  # noqa: E402
from adsorbgen.flow import FlowConfig  # noqa: E402


TAG_NAME = {
    0: "bulk",
    1: "surface",
    2: "ads",
}


def _pair_name(a: int, b: int) -> str:
    aa = TAG_NAME.get(int(a), f"tag{int(a)}")
    bb = TAG_NAME.get(int(b), f"tag{int(b)}")
    return "--".join(sorted((aa, bb)))


def _pair_breakdown(record: dict, threshold: float) -> dict:
    pos = record["pos_pred"]
    if pos.dim() == 3:
        pos = pos[0]
    tags = record["tags"].numpy()
    cell = record["cell"].numpy()
    z = record["atomic_numbers"].numpy()

    atoms = Atoms(numbers=z, positions=pos.numpy(), cell=cell, pbc=True)
    d = atoms.get_all_distances(mic=True)
    n = len(atoms)

    by_type_min: dict[str, float] = defaultdict(lambda: math.inf)
    by_type_under = Counter()
    global_min = math.inf
    global_type = None
    global_pair = None

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(d[i, j])
            name = _pair_name(int(tags[i]), int(tags[j]))
            if dist < by_type_min[name]:
                by_type_min[name] = dist
            if dist < threshold:
                by_type_under[name] += 1
            if dist < global_min:
                global_min = dist
                global_type = name
                global_pair = [int(i), int(j)]

    return {
        "sid": record.get("sid"),
        "has_overlap": bool(global_min < threshold),
        "global_min_distance_A": global_min,
        "global_min_pair_type": global_type,
        "global_min_pair": global_pair,
        "min_distance_by_pair_type_A": dict(sorted(by_type_min.items())),
        "n_pairs_under_threshold_by_pair_type": dict(sorted(by_type_under.items())),
    }


def _summarize(rows: list[dict], anomaly_per_sample: list[dict], threshold: float) -> dict:
    n = len(rows)
    overlap_rows = [r for r in rows if r["has_overlap"]]
    overlap_n = len(overlap_rows)

    global_type_counts = Counter(
        r["global_min_pair_type"] for r in overlap_rows if r["global_min_pair_type"] is not None
    )
    any_under_counts = Counter()
    under_pair_counts = Counter()
    min_by_type = defaultdict(list)

    for r in rows:
        for name, value in r["min_distance_by_pair_type_A"].items():
            if math.isfinite(value):
                min_by_type[name].append(float(value))
        if r["has_overlap"]:
            for name, cnt in r["n_pairs_under_threshold_by_pair_type"].items():
                any_under_counts[name] += 1
                under_pair_counts[name] += int(cnt)

    overlap_and_anom = {
        "overlap_and_dissoc": 0,
        "overlap_and_desorbed": 0,
        "overlap_and_intercalated": 0,
        "overlap_and_surf_changed": 0,
    }
    by_sid = {r.get("sid"): r for r in anomaly_per_sample}
    for r in overlap_rows:
        a = by_sid.get(r.get("sid"), {})
        overlap_and_anom["overlap_and_dissoc"] += int(bool(a.get("has_dissoc")))
        overlap_and_anom["overlap_and_desorbed"] += int(bool(a.get("has_desorbed")))
        overlap_and_anom["overlap_and_intercalated"] += int(bool(a.get("has_intercalated")))
        overlap_and_anom["overlap_and_surf_changed"] += int(bool(a.get("has_surf_changed")))

    min_stats = {}
    for name, values in sorted(min_by_type.items()):
        arr = np.asarray(values, dtype=float)
        min_stats[name] = {
            "n": int(arr.size),
            "min": float(arr.min()) if arr.size else None,
            "p01": float(np.quantile(arr, 0.01)) if arr.size else None,
            "p05": float(np.quantile(arr, 0.05)) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
        }

    return {
        "n_samples": n,
        "overlap_threshold_A": threshold,
        "overlap_count": overlap_n,
        "overlap_rate": overlap_n / max(n, 1),
        "global_min_pair_type_counts_on_overlap_samples": dict(sorted(global_type_counts.items())),
        "samples_with_any_under_threshold_by_pair_type": dict(sorted(any_under_counts.items())),
        "pairs_under_threshold_by_pair_type": dict(sorted(under_pair_counts.items())),
        **overlap_and_anom,
        "min_distance_stats_by_pair_type_A": min_stats,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ckpt-path", type=Path, required=True)
    p.add_argument("--val-lmdb", type=str, required=True)
    p.add_argument("--max-val-samples", type=int, default=200)
    p.add_argument("--val-replicate", type=int, default=1)
    p.add_argument("--sample-eval-steps", type=int, default=50)
    p.add_argument("--prediction-type", type=str, default="x1", choices=["x1", "v"])
    p.add_argument("--prior-mode", type=str, default="random_heuristic")
    p.add_argument("--translation-std", type=float, default=0.0)
    p.add_argument("--interstitial-gap", type=float, default=0.1)
    p.add_argument("--refine-final", action="store_true")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--overlap-threshold", type=float, default=OVERLAP_MIN_DIST_A)
    p.add_argument("--pristine-slabs", type=Path, default=None)
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    with open(run_dir / "args.json") as f:
        args_blob = json.load(f)

    device = torch.device(args.device)
    model, model_cfg = _load_module(args.ckpt_path.resolve(), args_blob, device)
    _ = _build_cfg_from_blob(args_blob)  # Fail early if config is malformed.
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
    val_ds = torch.utils.data.ConcatDataset([base_ds] * args.val_replicate)
    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_displacement,
    )
    target_n = args.max_val_samples * args.val_replicate
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

    rows = [_pair_breakdown(r, args.overlap_threshold) for r in records]
    strict = compute_anomaly_metrics(
        records,
        pristine_slabs=args.pristine_slabs,
        pristine_sid_index=args.pristine_sid_index,
    )
    summary = _summarize(rows, strict["per_sample"], args.overlap_threshold)

    out = {
        "run_dir": str(run_dir),
        "ckpt_path": str(args.ckpt_path.resolve()),
        "val_lmdb": args.val_lmdb,
        "max_val_samples": args.max_val_samples,
        "val_replicate": args.val_replicate,
        "sample_eval_steps": args.sample_eval_steps,
        "refine_final": bool(args.refine_final),
        "strict_aggregate": strict["aggregate"],
        "overlap_pair_summary": summary,
        "per_sample": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[diagnose] wrote {args.out}")


if __name__ == "__main__":
    main()
