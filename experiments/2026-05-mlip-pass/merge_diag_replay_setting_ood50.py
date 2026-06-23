#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    if k > n:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


def stats(vals):
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 5, 10])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rows = []
    shard_summaries = []
    for p in sorted(out_dir.glob("shard_*.pkl")):
        with p.open("rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "rows" in payload:
            rows.extend(payload["rows"])
            shard_summaries.append(payload.get("summary", {}))
        elif isinstance(payload, list):
            rows.extend(payload)

    rows.sort(key=lambda r: int(r["global_i"]))
    n = max(len(rows), 1)
    conv = [r for r in rows if r.get("converged")]
    unconv = [r for r in rows if not r.get("converged")]
    valid = [r for r in rows if r.get("valid")]
    success = [r for r in valid if r.get("success")]
    systems = sorted({str(r["system_key"]) for r in rows})

    by_system = defaultdict(list)
    for r in rows:
        by_system[str(r["system_key"])].append(r)
    per_system = {}
    for sk in systems:
        rs = by_system.get(sk, [])
        valid_rs = [r for r in rs if r.get("valid")]
        n_valid = len(valid_rs)
        c_success = sum(1 for r in valid_rs if r.get("success"))
        per_system[sk] = {
            "n": len(rs),
            "expected_n": int(args.num_samples),
            "n_valid": n_valid,
            "n_converged": sum(1 for r in rs if r.get("converged")),
            "c_success": c_success,
            "status_counts": dict(Counter(str(r.get("status")) for r in rs)),
            **{f"pass@{k}": pass_at_k(int(args.num_samples), c_success, k) for k in args.k},
            **{f"valid_pass@{k}": pass_at_k(n_valid, c_success, k) for k in args.k},
        }

    def best_by(key: str, reverse: bool = False) -> dict[str, object]:
        selected = []
        for sk, rs in by_system.items():
            cand = [r for r in rs if r.get("valid") and math.isfinite(float(r.get(key, float("nan"))))]
            if not cand:
                continue
            selected.append(sorted(cand, key=lambda r: float(r[key]), reverse=reverse)[0])
        return {
            "systems": len(selected),
            "sp_delta_E_sys": stats([r["sp_delta_E_sys"] for r in selected]),
            "final_delta_E_sys": stats([r["final_delta_E_sys"] for r in selected]),
            "n_steps": stats([r["n_steps"] for r in selected]),
        }

    def max_stage(stage: str) -> float | None:
        vals = []
        for s in shard_summaries:
            v = (s.get("stage_elapsed_sec") or {}).get(stage)
            if v is not None and math.isfinite(float(v)):
                vals.append(float(v))
        return max(vals) if vals else None

    total_wall_sec = max_stage("total")
    generate_wall_sec = max_stage("generate_or_load")
    sp_wall_sec = max_stage("single_point")
    relax_wall_sec = max_stage("relax")

    summary = {
        "out_dir": str(out_dir),
        "label": shard_summaries[0].get("label") if shard_summaries else out_dir.name,
        "mode": shard_summaries[0].get("mode") if shard_summaries else None,
        "shards": len(shard_summaries),
        "systems": len(systems),
        "candidates": len(rows),
        "converged": len(conv),
        "unconverged": len(unconv),
        "converged_rate": len(conv) / n,
        "valid": len(valid),
        "valid_rate": len(valid) / n,
        "valid_success": len(success),
        "valid_success_rate": len(success) / max(len(valid), 1),
        "status_counts": dict(Counter(str(r.get("status")) for r in rows)),
        "sp_delta_E_sys": stats([r["sp_delta_E_sys"] for r in rows]),
        "valid_sp_delta_E_sys": stats([r["sp_delta_E_sys"] for r in valid]),
        "sp_abs_delta_E_sys": stats([abs(r["sp_delta_E_sys"]) for r in rows]),
        "valid_sp_abs_delta_E_sys": stats([abs(r["sp_delta_E_sys"]) for r in valid]),
        "all_n_steps": stats([r["n_steps"] for r in rows]),
        "converged_n_steps": stats([r["n_steps"] for r in conv]),
        "valid_n_steps": stats([r["n_steps"] for r in valid]),
        "unconverged_n_steps": stats([r["n_steps"] for r in unconv]),
        "final_delta_E_sys": stats([r["final_delta_E_sys"] for r in rows]),
        "valid_final_delta_E_sys": stats([r["final_delta_E_sys"] for r in valid]),
        "final_abs_delta_E_sys": stats([abs(r["final_delta_E_sys"]) for r in rows]),
        "valid_final_abs_delta_E_sys": stats([abs(r["final_delta_E_sys"]) for r in valid]),
        "valid_success_systems": sum(1 for v in per_system.values() if v["c_success"] > 0),
        **{
            f"mlip_pass@{k}": sum(v[f"pass@{k}"] for v in per_system.values()) / max(len(per_system), 1)
            for k in args.k
        },
        **{
            f"valid_mlip_pass@{k}": sum(v[f"valid_pass@{k}"] for v in per_system.values()) / max(len(per_system), 1)
            for k in args.k
        },
        "valid_best_by_pre_gap": best_by("sp_delta_E_sys"),
        "valid_best_by_post_gap": best_by("final_delta_E_sys"),
        "valid_best_by_steps": best_by("n_steps"),
        "walltime_sec": {
            "pre_relax_generate_or_load": generate_wall_sec,
            "single_point": sp_wall_sec,
            "relax": relax_wall_sec,
            "total": total_wall_sec,
        },
        "throughput_8gpu": {
            "pre_relax_candidates_per_sec": len(rows) / generate_wall_sec if generate_wall_sec and generate_wall_sec > 0 else None,
            "single_point_candidates_per_sec": len(rows) / sp_wall_sec if sp_wall_sec and sp_wall_sec > 0 else None,
            "post_relax_candidates_per_sec": len(rows) / total_wall_sec if total_wall_sec and total_wall_sec > 0 else None,
            "valid_post_relax_candidates_per_sec": len(valid) / total_wall_sec if total_wall_sec and total_wall_sec > 0 else None,
        },
        "per_system": per_system,
        "settings": shard_summaries[0].get("settings", {}) if shard_summaries else {},
    }
    with (out_dir / "all_rows.pkl").open("wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
