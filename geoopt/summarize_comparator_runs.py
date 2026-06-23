#!/usr/bin/env python
"""Summarize H200 sweep or ASE-vs-custom comparator shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsons(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        with path.open() as f:
            d = json.load(f)
        d["_path"] = str(path)
        out.append(d)
    return out


def pct(vals: list[float], q: float) -> float | None:
    arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else None


def summarize_compare(files: list[Path]) -> dict:
    docs = load_jsons(files)
    rows = [r for d in docs for r in d.get("rows", [])]
    finite_de = [abs(float(r["dE"])) for r in rows if np.isfinite(float(r["dE"]))]
    both = [
        r for r in rows
        if r.get("ase_converged") and r.get("custom_converged") and np.isfinite(float(r["dE"]))
    ]
    both_de = [abs(float(r["dE"])) for r in both]
    timing = [d.get("timing", {}) for d in docs]
    return {
        "files": len(files),
        "n": len(rows),
        "ase_converged": sum(bool(r.get("ase_converged")) for r in rows),
        "custom_converged": sum(bool(r.get("custom_converged")) for r in rows),
        "converged_agreement": sum(bool(r.get("ase_converged")) == bool(r.get("custom_converged")) for r in rows),
        "ase_converged_custom_not": sum(bool(r.get("ase_converged")) and not bool(r.get("custom_converged")) for r in rows),
        "custom_converged_ase_not": sum(bool(r.get("custom_converged")) and not bool(r.get("ase_converged")) for r in rows),
        "valid_agreement": sum(bool(r.get("ase_valid")) == bool(r.get("custom_valid")) for r in rows),
        "success_flips": sum(bool(r.get("ase_success")) != bool(r.get("custom_success")) for r in rows),
        "window_flips": sum(bool(r.get("ase_window")) != bool(r.get("custom_window")) for r in rows),
        "both_converged": len(both),
        "abs_dE_median": pct(finite_de, 50),
        "abs_dE_p95": pct(finite_de, 95),
        "abs_dE_max": max(finite_de) if finite_de else None,
        "both_converged_abs_dE_median": pct(both_de, 50),
        "both_converged_abs_dE_p95": pct(both_de, 95),
        "both_converged_abs_dE_max": max(both_de) if both_de else None,
        "ase_elapsed_sec_max": max((float(t.get("ase_elapsed_sec", 0.0)) for t in timing), default=0.0),
        "custom_elapsed_sec_max": max((float(t.get("custom_elapsed_sec", 0.0)) for t in timing), default=0.0),
        "ase_candidates_per_sec_sum": sum(float(t.get("ase_candidates_per_sec") or 0.0) for t in timing),
        "custom_candidates_per_sec_sum": sum(float(t.get("custom_candidates_per_sec") or 0.0) for t in timing),
    }


def summarize_ase_step_delta(files_a: list[Path], files_b: list[Path]) -> dict:
    docs_a = load_jsons(files_a)
    docs_b = load_jsons(files_b)
    def frozen(x):
        if isinstance(x, list):
            return tuple(frozen(v) for v in x)
        return x

    def key(r: dict) -> tuple:
        return (frozen(r.get("system_key", [])), int(r.get("sample_i", r.get("global_i", -1))))

    rows_a = {key(r): r for d in docs_a for r in d.get("ase_rows", [])}
    rows_b = {key(r): r for d in docs_b for r in d.get("ase_rows", [])}
    keys = sorted(set(rows_a) & set(rows_b))
    rows = []
    for k in keys:
        a, b = rows_a[k], rows_b[k]
        d_e = float(b["E_sys"] - a["E_sys"])
        rows.append(
            {
                "global_i": k,
                "a_converged": bool(a["converged"]),
                "b_converged": bool(b["converged"]),
                "a_success": bool(a["success"]),
                "b_success": bool(b["success"]),
                "a_window": bool(a["window"]),
                "b_window": bool(b["window"]),
                "dE": d_e,
            }
        )
    finite_de = [abs(r["dE"]) for r in rows if np.isfinite(r["dE"])]
    return {
        "n": len(rows),
        "ase_converged_agreement": sum(r["a_converged"] == r["b_converged"] for r in rows),
        "ase_success_flips": sum(r["a_success"] != r["b_success"] for r in rows),
        "ase_window_flips": sum(r["a_window"] != r["b_window"] for r in rows),
        "ase_abs_dE_median": pct(finite_de, 50),
        "ase_abs_dE_p95": pct(finite_de, 95),
        "ase_abs_dE_max": max(finite_de) if finite_de else None,
    }


def summarize_sweep(files: list[Path]) -> list[dict]:
    rows = []
    for d in load_jsons(files):
        s = d.get("settings", {})
        t = d.get("algorithms", {}).get("lbfgs", {}).get("throughput", {})
        rows.append(
            {
                "path": d["_path"],
                "max_atoms": int(s.get("max_atoms", 0)),
                "flow_batch_size": int(s.get("flow_batch_size", 0)),
                "max_steps": int(s.get("max_steps", 0)),
                "num_jobs": int(d.get("num_jobs", 0)),
                "elapsed_sec": float(t.get("elapsed_sec", 0.0)),
                "candidates_per_sec": float(t.get("candidates_per_sec") or 0.0),
                "converged_rate": float(t.get("converged_rate") or 0.0),
            }
        )
    return sorted(rows, key=lambda r: (r["max_atoms"], r["flow_batch_size"], r["path"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compare", "sweep", "ase-step-delta"], required=True)
    ap.add_argument("--files", nargs="+", type=Path, required=True)
    ap.add_argument("--files-b", nargs="*", type=Path, default=[])
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    if args.mode == "compare":
        out = summarize_compare(args.files)
    elif args.mode == "sweep":
        out = summarize_sweep(args.files)
    else:
        out = summarize_ase_step_delta(args.files, args.files_b)

    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
