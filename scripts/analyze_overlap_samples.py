#!/usr/bin/env python
"""Summarize overlap pair types from an AdsorbGen inference dump.

The inference dump is the ``torch.save`` output from ``adsorbgen.inference``.
This script reports which pair class is responsible for MIC distances below the
strict overlap threshold and optionally exports saved flow trajectories as ASE
trajectory files for visualization.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write


TAG_NAME = {0: "bulk", 1: "surface", 2: "ads"}


def _as_first_placement(x: torch.Tensor) -> torch.Tensor:
    return x[0] if isinstance(x, torch.Tensor) and x.dim() == 3 else x


def _pair_type(tag_i: int, tag_j: int) -> str:
    a, b = int(tag_i), int(tag_j)
    if a == 2 and b == 2:
        return "ads-ads"
    if (a == 2 and b == 1) or (a == 1 and b == 2):
        return "ads-surface"
    if (a == 2 and b == 0) or (a == 0 and b == 2):
        return "ads-bulk"
    if a != 2 and b != 2:
        return "slab-slab"
    return "--".join(sorted((TAG_NAME.get(a, f"tag{a}"), TAG_NAME.get(b, f"tag{b}"))))


def _coarse_pair_type(pair_type: str) -> str:
    if pair_type == "ads-bulk":
        return "ads-surface"
    return pair_type


def _atoms(numbers, positions, cell, tags) -> Atoms:
    atoms = Atoms(
        numbers=np.asarray(numbers, dtype=np.int64),
        positions=np.asarray(positions, dtype=np.float64),
        cell=np.asarray(cell, dtype=np.float64),
        pbc=True,
    )
    atoms.set_tags(np.asarray(tags, dtype=np.int64).tolist())
    return atoms


def pair_breakdown(record: dict, threshold: float) -> dict:
    pos = _as_first_placement(record["pos_pred"]).numpy()
    tags = record["tags"].numpy().astype(np.int64)
    numbers = record["atomic_numbers"].numpy().astype(np.int64)
    cell = record["cell"].numpy()

    atoms = _atoms(numbers, pos, cell, tags)
    d = atoms.get_all_distances(mic=True)
    by_type_min: dict[str, float] = defaultdict(lambda: math.inf)
    under_counts = Counter()
    under_counts_coarse = Counter()
    global_min = math.inf
    global_type = None
    global_pair = None

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = float(d[i, j])
            typ = _pair_type(tags[i], tags[j])
            by_type_min[typ] = min(by_type_min[typ], dist)
            if dist < threshold:
                under_counts[typ] += 1
                under_counts_coarse[_coarse_pair_type(typ)] += 1
            if dist < global_min:
                global_min = dist
                global_type = typ
                global_pair = [int(i), int(j)]

    return {
        "sid": int(record.get("sid", -1)),
        "system_key": record.get("system_key"),
        "config_key": record.get("config_key"),
        "has_overlap": bool(global_min < threshold),
        "global_min_distance_A": global_min,
        "global_min_pair_type": global_type,
        "global_min_pair_type_coarse": _coarse_pair_type(global_type) if global_type else None,
        "global_min_pair": global_pair,
        "n_pairs_under_threshold_by_pair_type": dict(sorted(under_counts.items())),
        "n_pairs_under_threshold_by_pair_type_coarse": dict(sorted(under_counts_coarse.items())),
        "min_distance_by_pair_type_A": dict(sorted(by_type_min.items())),
    }


def summarize(rows: list[dict], threshold: float) -> dict:
    n = len(rows)
    overlap_rows = [r for r in rows if r["has_overlap"]]
    global_min_counts = Counter(r["global_min_pair_type"] for r in overlap_rows)
    global_min_counts_coarse = Counter(r["global_min_pair_type_coarse"] for r in overlap_rows)
    sample_under_counts = Counter()
    sample_under_counts_coarse = Counter()
    pair_under_counts = Counter()
    pair_under_counts_coarse = Counter()
    min_by_type = defaultdict(list)

    for r in rows:
        for typ, value in r["min_distance_by_pair_type_A"].items():
            if math.isfinite(value):
                min_by_type[typ].append(value)
        if not r["has_overlap"]:
            continue
        for typ, count in r["n_pairs_under_threshold_by_pair_type"].items():
            sample_under_counts[typ] += 1
            pair_under_counts[typ] += int(count)
        for typ, count in r["n_pairs_under_threshold_by_pair_type_coarse"].items():
            sample_under_counts_coarse[typ] += 1
            pair_under_counts_coarse[typ] += int(count)

    min_stats = {}
    for typ, values in sorted(min_by_type.items()):
        arr = np.asarray(values, dtype=np.float64)
        min_stats[typ] = {
            "n": int(arr.size),
            "min": float(arr.min()),
            "p01": float(np.quantile(arr, 0.01)),
            "p05": float(np.quantile(arr, 0.05)),
            "median": float(np.median(arr)),
        }

    return {
        "n_samples": n,
        "overlap_threshold_A": threshold,
        "overlap_count": len(overlap_rows),
        "overlap_rate": len(overlap_rows) / max(n, 1),
        "global_min_pair_type_counts_on_overlap_samples": dict(sorted(global_min_counts.items())),
        "global_min_pair_type_counts_coarse_on_overlap_samples": dict(sorted(global_min_counts_coarse.items())),
        "samples_with_any_under_threshold_by_pair_type": dict(sorted(sample_under_counts.items())),
        "samples_with_any_under_threshold_by_pair_type_coarse": dict(sorted(sample_under_counts_coarse.items())),
        "pairs_under_threshold_by_pair_type": dict(sorted(pair_under_counts.items())),
        "pairs_under_threshold_by_pair_type_coarse": dict(sorted(pair_under_counts_coarse.items())),
        "min_distance_stats_by_pair_type_A": min_stats,
    }


def export_trajectories(
    records: list[dict],
    traj_dir: Path,
    rows: list[dict],
    *,
    only_overlap: bool = False,
    max_exports: int | None = None,
) -> list[dict]:
    traj_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for idx, rec in enumerate(records):
        if "x_trajectory" not in rec:
            continue
        row = rows[idx]
        if only_overlap and not row["has_overlap"]:
            continue
        traj = rec["x_trajectory"]
        if traj.dim() == 4:
            traj = traj[:, 0]
        numbers = rec["atomic_numbers"].numpy().astype(np.int64)
        tags = rec["tags"].numpy().astype(np.int64)
        cell = rec["cell"].numpy()
        frames = [_atoms(numbers, frame.numpy(), cell, tags) for frame in traj]
        stem = f"sample_{idx:03d}_sid_{int(rec.get('sid', -1))}"
        traj_path = traj_dir / f"{stem}_flow.traj"
        xyz_path = traj_dir / f"{stem}_flow.xyz"
        ase_write(str(traj_path), frames)
        ase_write(str(xyz_path), frames, format="extxyz")
        entries.append({
            "sample_index": idx,
            "sid": int(rec.get("sid", -1)),
            "system_key": rec.get("system_key"),
            "config_key": rec.get("config_key"),
            "has_overlap": bool(row["has_overlap"]),
            "global_min_distance_A": row["global_min_distance_A"],
            "global_min_pair_type": row["global_min_pair_type"],
            "global_min_pair": row["global_min_pair"],
            "n_frames": len(frames),
            "traj": str(traj_path),
            "xyz": str(xyz_path),
        })
        if max_exports is not None and len(entries) >= max_exports:
            break
    with open(traj_dir / "_index.json", "w") as f:
        json.dump(entries, f, indent=2)
    return entries


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--traj-dir", type=Path, default=None)
    p.add_argument("--only-overlap-trajectories", action="store_true")
    p.add_argument("--max-trajectory-exports", type=int, default=None)
    p.add_argument("--overlap-threshold", type=float, default=0.5)
    args = p.parse_args()

    payload = torch.load(args.samples, map_location="cpu", weights_only=False)
    records = payload["records"]
    rows = [pair_breakdown(r, args.overlap_threshold) for r in records]
    summary = summarize(rows, args.overlap_threshold)
    traj_entries = (
        export_trajectories(
            records,
            args.traj_dir,
            rows,
            only_overlap=args.only_overlap_trajectories,
            max_exports=args.max_trajectory_exports,
        )
        if args.traj_dir else []
    )
    out = {
        "samples": str(args.samples),
        "meta": payload.get("meta", {}),
        "summary": summary,
        "trajectory_exports": traj_entries,
        "per_sample": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    if traj_entries:
        print(f"[traj] wrote {len(traj_entries)} trajectories -> {args.traj_dir}")
    print(f"[done] wrote {args.out_json}")


if __name__ == "__main__":
    main()
