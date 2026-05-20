#!/usr/bin/env python
"""Aggregate the per-shard cycle JSON files from the 5000×10 one-off replay
and print the report the user asked for:

  - strict success criterion (E_pred < E_gt, with E_gt = E_sys_min per system)
      * rate1 = #systems with ≥1 success / #systems (target 5000)
      * rate2 = #success candidates / #candidates (target 50000)
  - relaxed criteria at +0.1, +0.2, +0.3 eV
      * #success candidates and #distinct success systems

Reads ``{stream_dir}/logs/cycle_{CYCLE:06d}_shard*.json`` and sums the per-shard
counters. Each shard owns a disjoint slice of unique systems, so summing the
``n_success_systems`` / ``n_relaxed_success_systems_*`` integers across shards
yields the true union count (no double-count).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stream-dir", required=True)
    p.add_argument("--cycle", type=int, default=0)
    args = p.parse_args()

    log_dir = Path(args.stream_dir) / "logs"
    shards = sorted(log_dir.glob(f"cycle_{args.cycle:06d}_shard*.json"))
    if not shards:
        raise SystemExit(f"no shard JSONs found under {log_dir}")

    agg = {
        "candidates": 0,
        "systems_evaluated": 0,
        "n_success": 0,
        "n_success_systems": 0,
        "n_relaxed_success_plus_0p1": 0,
        "n_relaxed_success_plus_0p2": 0,
        "n_relaxed_success_plus_0p3": 0,
        "n_relaxed_success_systems_plus_0p1": 0,
        "n_relaxed_success_systems_plus_0p2": 0,
        "n_relaxed_success_systems_plus_0p3": 0,
    }
    valid_weighted = 0.0
    unconv_weighted = 0.0
    elapsed_max = 0.0
    for sp in shards:
        d = json.loads(sp.read_text())
        for k in agg:
            agg[k] += int(d.get(k, 0))
        c = int(d.get("candidates", 0))
        valid_weighted += float(d.get("valid_rate", 0.0)) * c
        unconv_weighted += float(d.get("uma_unconverged_rate", 0.0)) * c
        elapsed_max = max(elapsed_max, float(d.get("elapsed_sec", 0.0)))

    N_sys = agg["systems_evaluated"]
    N_cand = agg["candidates"]
    n_str_cand = agg["n_success"]
    n_str_sys = agg["n_success_systems"]

    def pct(num: int, den: int) -> str:
        return f"{(100.0 * num / den if den else 0.0):.2f}%"

    print(f"\n=== 5000×10 replay report — cycle {args.cycle}, {len(shards)} shards ===")
    print(f"systems evaluated : {N_sys}")
    print(f"candidates        : {N_cand}")
    print(f"wall clock        : {elapsed_max/60:.1f} min (max shard)")
    print(f"valid rate        : {valid_weighted / max(N_cand, 1) * 100:.2f}%  "
          f"(uma_unconverged {unconv_weighted / max(N_cand, 1) * 100:.2f}%)")
    print()
    print("STRICT (E_pred < E_gt = E_sys_min)")
    print(f"  candidates succeeded: {n_str_cand} / {N_cand}  -> {pct(n_str_cand, N_cand)}")
    print(f"  systems with ≥1 succ: {n_str_sys} / {N_sys}  -> {pct(n_str_sys, N_sys)}")
    print()
    print("RELAXED criteria (E_pred < E_gt + Δ)")
    print(f"  {'Δ':>5}  {'success candidates':>20}  {'success systems':>17}")
    for tol_key, tol in (("0p1", 0.1), ("0p2", 0.2), ("0p3", 0.3)):
        nc = agg[f"n_relaxed_success_plus_{tol_key}"]
        ns = agg[f"n_relaxed_success_systems_plus_{tol_key}"]
        print(f"  +{tol:.1f}  {nc:>10} ({pct(nc, N_cand):>7})  "
              f"{ns:>10} ({pct(ns, N_sys):>7})")
    print()


if __name__ == "__main__":
    main()
