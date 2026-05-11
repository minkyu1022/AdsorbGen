#!/usr/bin/env python
"""Standalone replay eval on a single checkpoint.

Loads a Lightning ckpt (AdsorbGenModule), runs inference + batched UMA relax
(via fast_dynamics + nvalchemi FIRE) + anomaly filter + energy check, and
writes:
    - ReplayBuffer pkl (per shard)
    - viz artifacts (replay_viz/ep{TAG}/) for the web UI (per shard)
    - metrics json (per shard)

fast_dynamics already batches FIRE across many systems on ONE GPU. To use
multiple GPUs we shard the candidate pool across processes (each process owns
one GPU, loads its own UMAWrapper). Use --shard-idx / --num-shards. Each shard
MUST get its own --viz-root, --buffer-path, --metrics-path. Merge with
scripts/merge_replay_shards.py.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
if (_REPO / "adsorbgen").is_dir():
    sys.path.insert(0, str(_REPO))
else:
    sys.path.insert(0, str(_REPO / "AdsorbGen"))

from adsorbgen.dataset import PreprocessedDisplacementDataset  # noqa: E402
from adsorbgen.eval import load_pristine_context  # noqa: E402
from adsorbgen.eval_replay import ReplayEvalConfig, run_replay_eval  # noqa: E402
from adsorbgen.flow import FlowConfig  # noqa: E402
from adsorbgen.model import DiTDenoiserConfig  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.replay import ReplayBuffer  # noqa: E402


def load_model_from_ckpt(ckpt_path: Path, device: torch.device):
    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    model_cfg = hp["model_cfg"]
    flow_cfg = hp["flow_cfg"]
    model = build_model(model_cfg)
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    model.to(device).eval()
    return model, flow_cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str)
    p.add_argument("--gt-index", required=True, type=str)
    p.add_argument("--train-lmdb", required=True, type=str)
    p.add_argument("--viz-root", required=True, type=str,
                   help="per-shard dir; will create replay_viz/ep{TAG}/ inside")
    p.add_argument("--buffer-path", required=True, type=str)
    p.add_argument("--metrics-path", required=True, type=str)
    p.add_argument("--epoch-tag", type=int, default=30,
                   help="viz dir is ep{TAG} (run_replay_eval uses ep+1)")
    p.add_argument("--num-systems", type=int, default=500,
                   help="TOTAL systems across all shards. Each shard handles ~num_systems/num_shards.")
    p.add_argument("--num-placements", type=int, default=3)
    p.add_argument("--flow-steps", type=int, default=50)
    p.add_argument("--prior-mode", choices=["random", "heuristic", "random_heuristic"],
                   default="random_heuristic")
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-fmax", type=float, default=0.05)
    p.add_argument("--uma-max-steps", type=int, default=100)
    p.add_argument("--uma-atom-budget", type=int, default=4000)
    p.add_argument("--flow-batch-size", type=int, default=32)
    p.add_argument("--success-margin", type=float, default=0.05)
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    p.add_argument("--viz-capture-n", type=int, default=8,
                   help="per-shard viz target count. With 4 shards × 8 = 32 winners total.")
    p.add_argument("--shard-idx", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--pristine-slabs", type=str, default="",
                   help="path to pristine relaxed slab pkl. When set, anomaly "
                        "check's surf_changed axis uses pristine slab as the "
                        "reference (matches validation behavior). Empty string "
                        "→ fallback to pos_gt[slab_mask].")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=str, default="",
                   help="path to pristine sid/system_key→slab_key index pkl. Defaults to "
                        "<pristine-slabs>.sid_index.pkl or .system_index.pkl when "
                        "--pristine-slabs is set.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for UMA relax")
    device = torch.device("cuda")

    print(f"[replay] shard {args.shard_idx}/{args.num_shards}  ckpt: {args.ckpt}")
    print(f"[replay] viz_root: {args.viz_root}")
    t0 = time.time()

    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)
    print(f"[replay] model loaded ({time.time()-t0:.1f}s)")

    with open(args.gt_index, "rb") as f:
        gt_index_by_sid = pickle.load(f)
    print(f"[replay] GT index: {len(gt_index_by_sid)} sids")

    # Install pristine slab context so _passes_anomaly's surf_changed axis uses
    # the pristine relaxed slab (validation-equivalent). Without this, the
    # check falls back to pos_gt[slab_mask].
    if args.pristine_slabs:
        prist_pkl = Path(args.pristine_slabs)
        prist_idx = Path(args.pristine_sid_index) if args.pristine_sid_index else None
        ctx = load_pristine_context(prist_pkl, prist_idx)
        db_n = len(ctx["db"]) if ctx and ctx.get("db") is not None else 0
        sid_n = len(ctx["sid_to_key"]) if ctx and ctx.get("sid_to_key") is not None else 0
        system_n = len(ctx["system_to_key"]) if ctx and ctx.get("system_to_key") is not None else 0
        print(f"[replay] pristine slabs loaded: db={db_n}, sid_index={sid_n}, system_index={system_n}")
    else:
        print("[replay] pristine slabs NOT loaded (anomaly surf_changed → pos_gt fallback)")

    dataset = PreprocessedDisplacementDataset(args.train_lmdb, max_samples=None)
    n_total = len(dataset)
    print(f"[replay] dataset: {n_total} samples")

    # Same seed across shards → identical full list → deterministic non-overlap stride
    rng = np.random.default_rng(seed=args.epoch_tag)
    full_indices = rng.choice(n_total, size=min(args.num_systems, n_total), replace=False)
    sys_indices = full_indices[args.shard_idx::args.num_shards]
    print(f"[replay] shard sys_indices: {len(sys_indices)} / {len(full_indices)} total")

    cfg = ReplayEvalConfig(
        prior_mode=args.prior_mode,
        num_systems=int(len(sys_indices)),
        num_placements=args.num_placements,
        flow_steps=args.flow_steps,
        uma_model=args.uma_model,
        uma_fmax=args.uma_fmax,
        uma_max_steps=args.uma_max_steps,
        overlap_threshold=args.overlap_threshold,
        success_margin=args.success_margin,
        device="cuda",
        flow_batch_size=args.flow_batch_size,
        uma_atom_budget=args.uma_atom_budget,
        viz_capture_n=args.viz_capture_n,
        viz_root=args.viz_root,
    )

    buffer = ReplayBuffer(mode="append", per_system_cap=10, global_cap=1_070_000)

    metrics = run_replay_eval(
        model=model,
        dataset=dataset,
        gt_index_by_sid=gt_index_by_sid,
        buffer=buffer,
        cfg=cfg,
        flow_cfg=flow_cfg,
        epoch=args.epoch_tag - 1,
        sys_indices_override=sys_indices,
    )
    metrics["shard_idx"] = args.shard_idx
    metrics["num_shards"] = args.num_shards
    metrics["sys_indices_len"] = int(len(sys_indices))
    metrics["wall_sec"] = float(time.time() - t0)

    Path(args.buffer_path).parent.mkdir(parents=True, exist_ok=True)
    buffer.save(Path(args.buffer_path))
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[replay] shard {args.shard_idx} metrics: {metrics}")
    print(f"[replay] wall: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
