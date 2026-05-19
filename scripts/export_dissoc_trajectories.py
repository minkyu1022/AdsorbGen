#!/usr/bin/env python
"""Convert an adsorbgen.inference dump into per-sample ASE .traj files,
labelled with dissoc/overlap flags so the trajectory viewer can pick
dissociation cases.

Usage:
    python export_dissoc_trajectories.py <run_dir>

Expects <run_dir>/records.pt produced with --save-trajectories N. Writes:
    <run_dir>/trajectories/sample_<idx>_sid_<sid>_flow.traj
    <run_dir>/trajectories/sample_<idx>_sid_<sid>_flow.xyz
    <run_dir>/trajectories/_index.json   # list of dicts, one per saved sample
    <run_dir>/anomaly_summary.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write

sys.path.insert(0, "/home/irteam/AdsorbGen")
from adsorbgen.eval import _score_record_anomaly, _pos_first_placement  # noqa: E402


def _atoms_from(positions, numbers, cell, tags) -> Atoms:
    a = Atoms(
        numbers=np.asarray(numbers, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=True,
    )
    a.set_tags(np.asarray(tags, dtype=np.int64).tolist())
    return a


def export_run(run_dir: Path, only_with_dissoc_or_overlap: bool = True, max_export: int = 20):
    dump = torch.load(run_dir / "records.pt", weights_only=False, map_location="cpu")
    records = dump["records"] if isinstance(dump, dict) else dump
    print(f"[load] {len(records)} records from {run_dir/'records.pt'}")

    out_dir = run_dir / "trajectories"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = []
    counts = {"dissoc": 0, "overlap": 0, "valid": 0, "other": 0, "total": 0, "saved": 0}

    for rec in records:
        counts["total"] += 1
        flags = _score_record_anomaly(rec)
        if flags["valid_strict"]:
            counts["valid"] += 1
        if flags.get("has_dissoc"):
            counts["dissoc"] += 1
        if flags.get("has_overlap"):
            counts["overlap"] += 1

        # Only export the saved trajectories (x_trajectory present)
        if "x_trajectory" not in rec:
            continue
        if only_with_dissoc_or_overlap and not (flags.get("has_dissoc") or flags.get("has_overlap")):
            continue
        if counts["saved"] >= max_export:
            continue

        traj_tensor = rec["x_trajectory"]
        if traj_tensor.dim() == 4:
            traj_tensor = traj_tensor[:, 0]  # (T, N, 3) — first placement of FK group
        positions_list = traj_tensor.numpy()

        numbers = rec["atomic_numbers"].numpy()
        tags = rec["tags"].numpy()
        cell = rec["cell"].numpy()

        frames = [_atoms_from(p, numbers, cell, tags) for p in positions_list]
        # Final predicted pose at the end (already last frame of trajectory)

        sample_idx = counts["saved"]
        sid = int(rec.get("sid", -1)) if not isinstance(rec.get("sid"), torch.Tensor) else int(rec["sid"].item())
        stem = f"sample_{sample_idx:03d}_sid_{sid}_flow"
        traj_path = out_dir / f"{stem}.traj"
        xyz_path = out_dir / f"{stem}.xyz"
        ase_write(str(traj_path), frames)
        ase_write(str(xyz_path), frames)

        index.append({
            "sample_index": sample_idx,
            "record_index": counts["total"] - 1,
            "sid": sid,
            "system_key": rec.get("system_key"),
            "config_key": rec.get("config_key"),
            "has_dissoc": bool(flags.get("has_dissoc")),
            "has_overlap": bool(flags.get("has_overlap")),
            "has_desorbed": bool(flags.get("has_desorbed")),
            "has_intercalated": bool(flags.get("has_intercalated")),
            "has_surf_changed": bool(flags.get("has_surf_changed")),
            "valid_strict": bool(flags.get("valid_strict")),
            "min_pair_distance_A": float(flags.get("min_pair_distance_A", float("nan"))),
            "n_frames": len(frames),
            "traj": str(traj_path),
            "xyz": str(xyz_path),
        })
        counts["saved"] += 1

    (out_dir / "_index.json").write_text(json.dumps(index, indent=2))
    (run_dir / "anomaly_summary.json").write_text(json.dumps({
        "total": counts["total"],
        "valid_strict_rate": counts["valid"] / max(counts["total"], 1),
        "dissoc_rate": counts["dissoc"] / max(counts["total"], 1),
        "overlap_rate": counts["overlap"] / max(counts["total"], 1),
        "n_exported": counts["saved"],
    }, indent=2))
    print(f"[done] exported {counts['saved']} trajectories (dissoc/overlap preferred) to {out_dir}")
    print(f"       overall: total={counts['total']} valid={counts['valid']} dissoc={counts['dissoc']} overlap={counts['overlap']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--max-export", type=int, default=20)
    p.add_argument("--all", action="store_true", help="export every saved trajectory, not just dissoc/overlap")
    args = p.parse_args()
    export_run(Path(args.run_dir), only_with_dissoc_or_overlap=not args.all, max_export=args.max_export)
