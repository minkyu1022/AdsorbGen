"""Compute displacement statistics on a preprocessed IS2RES LMDB.

Iterates the LMDB, computes per-axis stats of delta_1 over movable atoms
(tags in {1, 2} and fixed == 0), and writes a JSON summary used to pick the
prior sigma for flow matching.

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.compute_stats \
        --lmdb data/processed/is2res_train.lmdb \
        --out  data/processed/is2res_disp_stats.json \
        --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.dataset import PreprocessedDisplacementDataset  # noqa: E402
from adsorbgen.flow import minimum_image  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-samples", type=int, default=5000)
    args = p.parse_args()

    ds = PreprocessedDisplacementDataset(args.lmdb, unconditional=True, max_samples=args.max_samples)
    print(f"Loaded {len(ds)} samples from {args.lmdb}", flush=True)

    per_axis_sq = np.zeros(3, dtype=np.float64)
    per_axis_sum = np.zeros(3, dtype=np.float64)
    count = 0
    max_norm = 0.0
    norm_hist = []

    for i in range(len(ds)):
        s = ds[i]
        pos = s["pos"].unsqueeze(0)
        pos_rel = s["pos_relaxed"].unsqueeze(0)
        cell = s["cell"].unsqueeze(0)
        tags = s["tags"]
        fixed = s["fixed"]
        movable = ((tags == 1) | (tags == 2)) & (fixed == 0)
        if not movable.any():
            continue
        delta = minimum_image(pos_rel - pos, cell).squeeze(0)
        delta = delta[movable]
        delta_np = delta.numpy()
        per_axis_sq += (delta_np ** 2).sum(axis=0)
        per_axis_sum += delta_np.sum(axis=0)
        count += delta_np.shape[0]
        norm = np.linalg.norm(delta_np, axis=1)
        max_norm = max(max_norm, float(norm.max()))
        norm_hist.append(norm)

        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(ds)}  movable atoms so far: {count}", flush=True)

    mean = (per_axis_sum / max(count, 1)).tolist()
    var = (per_axis_sq / max(count, 1)) - np.array(mean) ** 2
    std = np.sqrt(np.clip(var, 0.0, None)).tolist()
    global_std = float(math.sqrt(max(np.mean(var), 0.0)))

    norms_all = np.concatenate(norm_hist) if norm_hist else np.array([])
    pct = np.percentile(norms_all, [50, 90, 95, 99, 99.9]).tolist() if norms_all.size else []

    out = {
        "lmdb": args.lmdb,
        "num_samples_used": min(len(ds), args.max_samples),
        "num_movable_atoms": count,
        "per_axis_mean_angstrom": mean,
        "per_axis_std_angstrom": std,
        "global_std_angstrom": global_std,
        "max_norm_angstrom": max_norm,
        "norm_percentiles_angstrom": {
            "p50": pct[0] if pct else None,
            "p90": pct[1] if pct else None,
            "p95": pct[2] if pct else None,
            "p99": pct[3] if pct else None,
            "p99.9": pct[4] if pct else None,
        },
        "recommended_sigma": float(round(global_std, 3)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
