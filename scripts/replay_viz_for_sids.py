#!/usr/bin/env python
"""Re-run replay flow + UMA for specific sids and capture viz.

Use when buffer.pkl contains successful entries whose viz wasn't captured by
the random viz_capture_n selection during the main replay run.

Outputs sys_<NAME>_<idx>/ folders directly into the given --viz-root/ep{TAG}/
and updates _index.json to include them.
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
    model = build_model(hp["model_cfg"])
    sd = ck["state_dict"]
    stripped = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    model.load_state_dict(stripped, strict=False)
    model.to(device).eval()
    return model, hp["flow_cfg"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--gt-index", required=True)
    p.add_argument("--train-lmdb", required=True)
    p.add_argument("--target-sids", required=True,
                   help="comma-separated sids to re-run viz for")
    p.add_argument("--viz-root", required=True,
                   help="dir containing ep{TAG}/; new sys_* folders are appended")
    p.add_argument("--epoch-tag", type=int, default=30)
    p.add_argument("--num-placements", type=int, default=3)
    p.add_argument("--flow-steps", type=int, default=50)
    p.add_argument("--prior-mode", default="random_heuristic")
    p.add_argument("--uma-model", default="uma-s-1p1")
    p.add_argument("--uma-fmax", type=float, default=0.05)
    p.add_argument("--uma-max-steps", type=int, default=300)
    p.add_argument("--uma-atom-budget", type=int, default=4000)
    p.add_argument("--flow-batch-size", type=int, default=32)
    p.add_argument("--success-margin", type=float, default=0.05)
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    p.add_argument("--pristine-slabs", type=str, default="")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=str, default="")
    p.add_argument("--name-prefix", default="success",
                   help="prefix for new sys_* folder names (e.g. sys_<prefix>_g###)")
    args = p.parse_args()

    target_sids = sorted({int(s) for s in args.target_sids.split(",") if s.strip()})
    print(f"[viz-redo] target sids: {target_sids}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    t0 = time.time()
    model, flow_cfg = load_model_from_ckpt(Path(args.ckpt), device)

    with open(args.gt_index, "rb") as f:
        gt_index_by_sid = pickle.load(f)

    if args.pristine_slabs:
        prist_pkl = Path(args.pristine_slabs)
        prist_idx = Path(args.pristine_sid_index) if args.pristine_sid_index else None
        ctx = load_pristine_context(prist_pkl, prist_idx)
        print(f"[viz-redo] pristine loaded: db={len(ctx['db'])}")

    dataset = PreprocessedDisplacementDataset(args.train_lmdb, max_samples=None)
    print(f"[viz-redo] dataset: {len(dataset)} samples; scanning for target sids...")

    sid_to_idx: dict = {}
    for i in range(len(dataset)):
        s = int(dataset[i]["sid"].item())
        if s in set(target_sids):
            sid_to_idx[s] = i
            if len(sid_to_idx) == len(target_sids):
                break
    print(f"[viz-redo] sid → idx: {sid_to_idx}")
    missing = set(target_sids) - set(sid_to_idx)
    if missing:
        print(f"[viz-redo] WARN missing sids: {missing}")

    sys_indices = np.array(sorted(sid_to_idx.values()), dtype=np.int64)
    print(f"[viz-redo] sys_indices: {sys_indices}")

    # Use a temp viz root so we don't clobber the existing ep30/
    tmp_viz_root = Path(args.viz_root).parent / f"_viz_redo_{args.name_prefix}"
    tmp_viz_root.mkdir(parents=True, exist_ok=True)

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
        viz_capture_n=len(sys_indices),     # capture ALL target sids
        viz_root=str(tmp_viz_root),
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
    print(f"[viz-redo] metrics: {metrics}")
    print(f"[viz-redo] wall: {time.time()-t0:.1f}s")

    # Move sys_* folders from tmp_viz_root/ep{TAG}/ into the canonical viz_root/ep{TAG}/
    # with a unique prefix to avoid clobbering existing folders. Then update _index.json.
    src_ep = tmp_viz_root / f"ep{args.epoch_tag}"
    dst_ep = Path(args.viz_root) / f"ep{args.epoch_tag}"
    if not dst_ep.exists():
        raise SystemExit(f"target viz dir {dst_ep} does not exist")
    if not src_ep.exists():
        raise SystemExit(f"redo viz dir {src_ep} not created (run_replay_eval did not produce viz)")

    moved = []
    for sys_dir in sorted(src_ep.iterdir()):
        if sys_dir.is_dir() and sys_dir.name.startswith("sys_"):
            new_name = f"sys_{args.name_prefix}_{sys_dir.name[4:]}"
            target = dst_ep / new_name
            if target.exists():
                import shutil; shutil.rmtree(target)
            sys_dir.rename(target)
            moved.append(new_name)
    print(f"[viz-redo] moved {len(moved)} sys_* folders → {dst_ep}: {moved}")

    # Append entries from redo's _index.json to the final _index.json
    src_idx = src_ep / "_index.json"
    dst_idx = dst_ep / "_index.json"
    if src_idx.exists():
        with open(src_idx) as f:
            redo_idx = json.load(f)
    else:
        redo_idx = {"systems": []}
    with open(dst_idx) as f:
        final_idx = json.load(f)

    for new_name, e in zip(moved, redo_idx.get("systems", [])):
        e["sys_dir_name"] = new_name
        e["shard_idx"] = -1
        e["redo"] = True
        final_idx["systems"].append(e)
    final_idx["n_systems"] = len(final_idx["systems"])
    with open(dst_idx, "w") as f:
        json.dump(final_idx, f, indent=2)
    print(f"[viz-redo] _index.json updated: {final_idx['n_systems']} systems total")

    # Cleanup
    import shutil; shutil.rmtree(tmp_viz_root, ignore_errors=True)
    print(f"[viz-redo] cleanup done. dst: {dst_ep}")


if __name__ == "__main__":
    main()
