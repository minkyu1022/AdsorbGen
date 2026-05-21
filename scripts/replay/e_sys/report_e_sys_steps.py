#!/usr/bin/env python
"""Report convergence-step statistics for recomputed E_sys records."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": int(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(np.max(values)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--e-sys", default="/home/irteam/data/replay/E_sys.pkl")
    p.add_argument("--out", default="")
    args = p.parse_args()

    with open(args.e_sys, "rb") as f:
        e_sys = pickle.load(f)

    conv_steps = []
    unconv_steps = []
    fmax_conv = []
    fmax_unconv = []
    atoms_conv = []
    atoms_unconv = []
    for rec in e_sys.values():
        steps = rec.get("n_steps")
        if steps is None:
            continue
        if bool(rec.get("converged", False)):
            conv_steps.append(int(steps))
            fmax_conv.append(float(rec.get("fmax", np.nan)))
            atoms_conv.append(int(rec.get("n_atoms", 0)))
        else:
            unconv_steps.append(int(steps))
            fmax_unconv.append(float(rec.get("fmax", np.nan)))
            atoms_unconv.append(int(rec.get("n_atoms", 0)))

    conv_steps_a = np.asarray(conv_steps, dtype=np.int64)
    unconv_steps_a = np.asarray(unconv_steps, dtype=np.int64)
    total = len(e_sys)
    report = {
        "e_sys": str(args.e_sys),
        "total_records": int(total),
        "converged_records": int(conv_steps_a.size),
        "unconverged_records": int(unconv_steps_a.size),
        "converged_rate": float(conv_steps_a.size / max(total, 1)),
        "converged_n_steps": _stats(conv_steps_a),
        "unconverged_n_steps": _stats(unconv_steps_a),
        "converged_fmax": _stats(np.asarray(fmax_conv, dtype=np.float64)),
        "unconverged_fmax": _stats(np.asarray(fmax_unconv, dtype=np.float64)),
        "converged_n_atoms": _stats(np.asarray(atoms_conv, dtype=np.int64)),
        "unconverged_n_atoms": _stats(np.asarray(atoms_unconv, dtype=np.int64)),
    }

    out = Path(args.out) if args.out else Path(args.e_sys).with_name("E_sys_step_stats.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[steps] wrote {out}")


if __name__ == "__main__":
    main()
