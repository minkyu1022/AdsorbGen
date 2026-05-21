#!/usr/bin/env python
"""Merge OOD-50 MLIP Pass@k shard outputs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    if k > n:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/irteam/data/replay/mlip_pass_lbfgs_ood50")
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 5, 10])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    missing = []
    for shard in range(args.num_shards):
        p = out_dir / f"shard_{shard}.pkl"
        if not p.exists():
            missing.append(str(p))
            continue
        with p.open("rb") as f:
            rows.extend(pickle.load(f))
    if missing:
        raise FileNotFoundError(f"missing shard outputs: {missing[:3]}")

    by_system = defaultdict(list)
    for r in rows:
        by_system[str(r["system_key"])].append(r)

    per_system = {}
    for sk, rs in sorted(by_system.items()):
        n = len(rs)
        c = sum(1 for r in rs if r.get("success"))
        per_system[sk] = {
            "n": n,
            "expected_n": args.num_samples,
            "c_success": c,
            "n_valid": sum(1 for r in rs if r.get("valid")),
            "n_converged": sum(1 for r in rs if r.get("converged")),
            "status_counts": dict(Counter(str(r.get("status")) for r in rs)),
            **{f"pass@{k}": pass_at_k(n, c, k) for k in args.k},
        }

    n_rows = max(len(rows), 1)
    n_systems = max(len(per_system), 1)
    summary = {
        "systems": len(per_system),
        "candidates": len(rows),
        "expected_candidates": len(per_system) * args.num_samples,
        "complete_systems": sum(1 for v in per_system.values() if v["n"] == args.num_samples),
        "converged_rate": sum(1 for r in rows if r.get("converged")) / n_rows,
        "valid_rate": sum(1 for r in rows if r.get("valid")) / n_rows,
        "success_sample_rate": sum(1 for r in rows if r.get("success")) / n_rows,
        "status_counts": dict(Counter(str(r.get("status")) for r in rows)),
        **{
            f"mlip_pass@{k}": sum(v[f"pass@{k}"] for v in per_system.values()) / n_systems
            for k in args.k
        },
        "per_system": per_system,
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
