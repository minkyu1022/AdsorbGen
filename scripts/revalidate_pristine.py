#!/usr/bin/env python
"""Re-validate every saved checkpoint in a run dir under the pristine
surface_changed metric, producing a per-epoch JSON of sample_eval/dense/* rates.

Used to back-fill metrics for runs that finished BEFORE eval.py started using
pristine relaxed slabs as the surface_changed reference. Reuses the in-process
flow + euler sampler + eval pipeline so numbers are directly comparable to
sample_eval/dense/* logged during training.

Usage:
    PYTHONPATH=AdsorbGen python scripts/revalidate_pristine.py \
        --run-dir runs/overfit_100 \
        --val-lmdb data/processed/is2res_train.lmdb \
        --max-val-samples 100 --val-replicate 10 --sample-eval-steps 20 \
        --pristine-slabs results/pristine_slabs/is2res.pkl \
        --out runs/overfit_100/revalidated_metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from adsorbgen.dataset import (
    PlacementPriorDataset,
    PreprocessedDisplacementDataset,
    collate_displacement,
)
from adsorbgen.eval import (
    compute_anomaly_metrics, compute_displacement_metrics,
)
from adsorbgen.flow import FlowConfig, euler_sample
from adsorbgen.model import DiTDenoiserConfig
from adsorbgen.model_factory import build_model

try:
    from adsorbgen.model_v2 import DiTDenoiserV2Config
except Exception:
    DiTDenoiserV2Config = None


CKPT_RE = re.compile(r"ckpt_epochepoch=(\d+)\.ckpt$")


def _ckpt_paths(run_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in sorted(run_dir.glob("ckpt_epochepoch=*.ckpt")):
        m = CKPT_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _build_cfg_from_blob(args_blob: dict):
    """Reconstruct DiTDenoiserConfig (or V2) from saved args.json."""
    arch = args_blob.get("arch", "v1")
    blob = dict(args_blob["model_config"])
    if arch == "v1":
        return DiTDenoiserConfig(**blob)
    if DiTDenoiserV2Config is None:
        raise RuntimeError("DiTDenoiserV2Config not importable in this env")
    return DiTDenoiserV2Config(**blob)


def _load_module(ckpt_path: Path, args_blob: dict, device: torch.device,
                 strict_load: bool = True):
    """Build the model from args.json, then load weights from a Lightning ckpt."""
    model_cfg = _build_cfg_from_blob(args_blob)
    model = build_model(model_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict") or state
    # Strip the Lightning module prefix "model."
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=strict_load)
    if unexpected:
        print(f"  [warn] {ckpt_path.name} unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
    if missing:
        print(f"  [warn] {ckpt_path.name} missing keys: {len(missing)} (first 5: {missing[:5]})")
    model.to(device).eval()
    return model, model_cfg


@torch.no_grad()
def _generate_records(model, model_cfg, loader, flow_cfg: FlowConfig,
                      sample_eval_steps: int, max_samples: int,
                      device: torch.device, refine_final: bool = False) -> list[dict]:
    """Run euler_sample over the loader and build evaluation records."""
    use_self_cond = bool(getattr(model_cfg, "use_self_cond", False))
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    records: list[dict] = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        B = batch["pos"].shape[0]
        state = {"prev_pred": None}

        def model_forward(x_t, t):
            extra = {}
            if use_self_cond:
                extra["prev_pred"] = state["prev_pred"]
            if use_ads_ref:
                extra["ads_ref_pos"] = batch["ads_ref_pos"]
            out = model(
                pos=batch["pos"], x_t=x_t, t=t,
                atomic_numbers=batch["atomic_numbers"], tags=batch["tags"],
                movable_mask=batch["movable_mask"], pad_mask=batch["pad_mask"],
                cell=batch["cell"],
                **extra,
            )
            if use_self_cond:
                state["prev_pred"] = out.detach()
            return out

        x_out = euler_sample(
            model_forward, batch["pos"],
            batch["movable_mask"], batch["pad_mask"], flow_cfg,
            num_steps=sample_eval_steps,
            refine_final=refine_final,
        )
        for i in range(B):
            n = int(batch["pad_mask"][i].sum().item())
            cell_i = batch["cell"][i].cpu()
            if cell_i.dim() == 3:
                cell_i = cell_i[0]
            sid_i = int(batch["sid"][i].item()) if "sid" in batch else -1
            system_key_i = batch.get("system_key", [None] * B)[i]
            config_key_i = batch.get("config_key", [None] * B)[i]
            records.append({
                "sid": sid_i,
                "system_key": system_key_i,
                "config_key": config_key_i,
                "pos_pred": x_out[i, :n].cpu(),
                "pos_gt": batch["pos_relaxed"][i, :n].cpu(),
                "pos_ref": batch["pos"][i, :n].cpu(),
                "movable_mask": batch["movable_mask"][i, :n].cpu(),
                "atomic_numbers": batch["atomic_numbers"][i, :n].cpu(),
                "tags": batch["tags"][i, :n].cpu(),
                "cell": cell_i,
            })
            if max_samples > 0 and len(records) >= max_samples:
                return records
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Run output directory (contains args.json and ckpt_epoch*.ckpt).")
    p.add_argument("--val-lmdb", type=str, required=True)
    p.add_argument("--max-val-samples", type=int, default=100)
    p.add_argument("--val-replicate", type=int, default=10)
    p.add_argument("--sample-eval-steps", type=int, default=20)
    p.add_argument("--prediction-type", type=str, default="x1", choices=["x1", "v"],
                   help="x1 (default) for old runs; v for v-pred runs.")
    p.add_argument("--prior-mode", type=str, default="random_heuristic")
    p.add_argument("--dataset-mode", choices=["placement", "lmdb_pos"], default="placement",
                   help="placement: fresh PlacementPriorDataset x0 (default). "
                        "lmdb_pos: stored LMDB initial pos as x0.")
    p.add_argument("--subset-seed", type=int, default=None,
                   help="If set, sample max-val-samples systems uniformly from "
                        "the clean dataset instead of taking the first N. Use the "
                        "same seed to compare placement vs lmdb_pos on the same systems.")
    p.add_argument("--translation-std", type=float, default=0.0)
    p.add_argument("--interstitial-gap", type=float, default=0.1)
    p.add_argument("--pristine-slabs", type=Path, default=None,
                   help="Optional pristine relaxed-slab pkl. Omit to force the "
                        "pos_gt[tag!=2] fallback for every split.")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-ckpts", type=int, default=None,
                   help="Cap the number of ckpts processed (smoke).")
    p.add_argument("--ckpt-stride", type=int, default=1,
                   help="Sample every Nth ckpt (default 1 = all). The last "
                        "ckpt is always included regardless of stride so the "
                        "final-epoch number is captured.")
    p.add_argument("--ckpt-path", type=Path, default=None,
                   help="Validate ONLY this specific ckpt (skips run-dir iteration). "
                        "Epoch is parsed from filename or set to -1 if unparseable.")
    p.add_argument("--refine-final", action="store_true",
                   help="One extra forward at t=1-eps; for x1-mode uses model "
                        "output as x_1, for v-mode uses x_0 + pred. Standard "
                        "practice for x1-pred sampling.")
    p.add_argument("--allow-nonstrict-load", action="store_true",
                   help="Load checkpoint with strict=False for legacy/debug runs. "
                        "Default is strict=True so config/checkpoint mismatches fail fast.")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    args_json = run_dir / "args.json"
    with open(args_json) as f:
        args_blob = json.load(f)

    flow_cfg = FlowConfig(eps=1e-5, prediction_type=args.prediction_type)
    device = torch.device(args.device)

    # Build val dataset once. In placement mode each access draws a fresh
    # fairchem placement; val_replicate therefore estimates placement variance.
    # In lmdb_pos mode x0 is the stored LMDB initial structure; replication is
    # deterministic and mainly kept for API symmetry.
    provide_ads_ref = bool(args_blob["model_config"].get("use_ads_ref_pos", False))
    dataset_cls = PlacementPriorDataset if args.dataset_mode == "placement" else PreprocessedDisplacementDataset
    dataset_kwargs = dict(
        lmdb_path=args.val_lmdb,
        max_samples=None if args.subset_seed is not None else args.max_val_samples,
        training_aug=False,
        translation_std=args.translation_std,
        provide_ads_ref_pos=provide_ads_ref,
        skip_anomaly=True,
    )
    if args.dataset_mode == "placement":
        dataset_kwargs.update(
            prior_mode=args.prior_mode,
            interstitial_gap=args.interstitial_gap,
        )
    base_ds = dataset_cls(**dataset_kwargs)
    if args.subset_seed is not None:
        import random
        n_take = min(int(args.max_val_samples), len(base_ds))
        rng = random.Random(int(args.subset_seed))
        indices = rng.sample(range(len(base_ds)), n_take)
        base_ds = torch.utils.data.Subset(base_ds, indices)
    val_ds = torch.utils.data.ConcatDataset([base_ds] * args.val_replicate)
    loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_displacement,
    )
    target_n = len(base_ds) * args.val_replicate

    if args.ckpt_path is not None:
        m = CKPT_RE.search(args.ckpt_path.name)
        epoch = int(m.group(1)) if m else -1
        ckpts = [(epoch, args.ckpt_path.resolve())]
    else:
        all_ckpts = _ckpt_paths(run_dir)
        if args.ckpt_stride > 1 and len(all_ckpts) > 1:
            strided = all_ckpts[:: args.ckpt_stride]
            # Always include the last ckpt so the final-epoch number is captured.
            if all_ckpts[-1] not in strided:
                strided.append(all_ckpts[-1])
            ckpts = strided
        else:
            ckpts = all_ckpts
        if args.max_ckpts is not None:
            ckpts = ckpts[: args.max_ckpts]
    print(f"[revalidate] {len(ckpts)} ckpts (refine_final={args.refine_final}, "
          f"steps={args.sample_eval_steps}, stride={args.ckpt_stride}, "
          f"dataset_mode={args.dataset_mode}, subset_seed={args.subset_seed}), "
          f"target_n={target_n}")

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for epoch, ckpt in ckpts:
        t0 = time.time()
        model, model_cfg = _load_module(
            ckpt, args_blob, device,
            strict_load=not args.allow_nonstrict_load,
        )
        recs = _generate_records(
            model, model_cfg, loader, flow_cfg,
            sample_eval_steps=args.sample_eval_steps,
            max_samples=target_n, device=device,
            refine_final=args.refine_final,
        )
        disp = compute_displacement_metrics(recs)
        strict = compute_anomaly_metrics(
            recs,
            pristine_slabs=args.pristine_slabs,
            pristine_sid_index=args.pristine_sid_index,
        )
        agg = {**disp["aggregate"], **strict["aggregate"]}
        row = {
            "epoch": epoch,
            "ckpt": str(ckpt),
            "dataset_mode": args.dataset_mode,
            "subset_seed": args.subset_seed,
            "n_samples": agg["n_samples"],
            "displacement_mae_A": agg["displacement_mae_A"],
            "displacement_rmse_A": agg["displacement_rmse_A"],
            "valid_rate_strict": agg["valid_rate_strict"],
            "any_anomaly_rate": agg["any_anomaly_rate"],
            "overlap_rate": agg["overlap_rate"],
            "dissoc_rate": agg["dissoc_rate"],
            "desorbed_rate": agg["desorbed_rate"],
            "intercalated_rate": agg["intercalated_rate"],
            "surf_changed_rate": agg["surf_changed_rate"],
            "n_errors": agg.get("n_errors", 0),
            "elapsed_s": time.time() - t0,
        }
        rows.append(row)
        # Free GPU memory before next ckpt.
        del model
        torch.cuda.empty_cache()
        print(
            f"[ep {epoch:>3}] mae={row['displacement_mae_A']:.4f} "
            f"valid={row['valid_rate_strict']:.3f} "
            f"surf_chg={row['surf_changed_rate']:.3f} "
            f"({row['elapsed_s']:.1f}s)"
        )
        # Incremental save so partial progress is recoverable.
        with open(out_path, "w") as f:
            json.dump(rows, f, indent=2)

    print(f"[revalidate] wrote {out_path}")


if __name__ == "__main__":
    main()
