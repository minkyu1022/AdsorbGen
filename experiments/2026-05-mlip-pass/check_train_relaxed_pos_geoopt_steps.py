#!/usr/bin/env python3
"""Check UMA geo-opt steps when starting from train LMDB pos_relaxed."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
GEOOPT = REPO / "geoopt"
if str(GEOOPT) not in sys.path:
    sys.path.insert(0, str(GEOOPT))

from geoopt import load_uma, run_optimizer  # noqa: E402


def read_entry(env: lmdb.Environment, idx: int):
    with env.begin(write=False) as txn:
        raw = txn.get(str(idx).encode())
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def to_numpy(x, dtype=None) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    return arr.astype(dtype) if dtype is not None else arr


def stats(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/home/irteam/data/processed_ID/is2res_train.lmdb")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=4096)
    ap.add_argument("--maxstep", type=float, default=0.04)
    ap.add_argument("--lbfgs-memory", type=int, default=50)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-check-interval", type=int, default=10)
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False, readahead=False)
    n_total = env.stat()["entries"]
    rng = random.Random(args.seed)
    indices = rng.sample(range(n_total), min(int(args.num_samples), n_total))

    jobs = []
    for global_i, idx in enumerate(indices):
        entry = read_entry(env, idx)
        pos_relaxed = to_numpy(entry["pos_relaxed"], np.float64)
        numbers = to_numpy(entry.get("atomic_numbers", entry.get("z")), np.int64)
        tags = to_numpy(entry["tags"], np.int64)
        fixed = to_numpy(entry.get("fixed", tags == 0), np.int64)
        cell = to_numpy(entry["cell"], np.float64)
        if cell.ndim == 3:
            cell = cell[0]
        jobs.append(
            {
                "global_i": int(global_i),
                "raw_idx": int(idx),
                "relax_input": {
                    "numbers": numbers,
                    "tags": tags,
                    "fixed": fixed,
                    "cell": cell,
                    "pos_pred": pos_relaxed,
                },
            }
        )

    t0 = time.time()
    uma = load_uma(str(args.uma_model), str(args.uma_task), device)
    opt_args = SimpleNamespace(**vars(args))
    results = run_optimizer(jobs, uma, opt_args, device, "lbfgs", serial=False)
    elapsed = time.time() - t0

    rows = []
    for job, result in zip(jobs, sorted(results, key=lambda r: int(r["global_i"]))):
        rows.append(
            {
                "global_i": int(job["global_i"]),
                "raw_idx": int(job["raw_idx"]),
                "n_atoms": int(len(job["relax_input"]["numbers"])),
                "converged": bool(result["converged"]),
                "n_steps": int(result["n_steps"]),
                "fmax": float(result["fmax"]),
                "E_sys": float(result["E_sys"]),
                "error": result.get("error"),
            }
        )

    conv = [r for r in rows if r["converged"]]
    summary = {
        "lmdb": str(args.lmdb),
        "num_samples": len(rows),
        "elapsed_sec": float(elapsed),
        "settings": {
            "uma_model": args.uma_model,
            "uma_task": args.uma_task,
            "fmax": args.fmax,
            "max_steps": args.max_steps,
            "max_atoms": args.max_atoms,
            "lbfgs_streaming": bool(args.lbfgs_streaming),
            "lbfgs_check_interval": args.lbfgs_check_interval,
        },
        "converged": len(conv),
        "converged_rate": len(conv) / max(len(rows), 1),
        "all_n_steps": stats([r["n_steps"] for r in rows]),
        "converged_n_steps": stats([r["n_steps"] for r in conv]),
        "initial_or_final_fmax": stats([r["fmax"] for r in rows]),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
