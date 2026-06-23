#!/usr/bin/env python
"""Append one self-improvement loop metrics row and keep initial references.

The replay worker writes ``rows_shard*.pkl`` with one row per candidate.  This
script recomputes loop-level metrics from those raw rows instead of trusting the
worker's older ``success`` definition, because the current buffer policy is a
per-system moving window rather than strict replacement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _key(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _finite(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: list[float], prefix: str) -> dict[str, float | int | None]:
    arr = _finite(values)
    if arr.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_best": None,
            f"{prefix}_p10": None,
            f"{prefix}_p90": None,
        }
    return {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_best": float(np.min(arr)),
        f"{prefix}_p10": float(np.quantile(arr, 0.10)),
        f"{prefix}_p90": float(np.quantile(arr, 0.90)),
    }


def _load_rows(replay_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(replay_dir.glob("rows_shard*.pkl")):
        with path.open("rb") as f:
            rows.extend(pickle.load(f))
    if not rows:
        raise SystemExit(f"no rows_shard*.pkl found under {replay_dir}")
    return rows


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"missing JSON: {path}")
    return json.loads(path.read_text())


def _extract_pass_summary(summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not summary:
        return out
    mapping = {
        "valid_mlip_pass@1": "ood50_mlip_pass@1",
        "valid_mlip_pass@2": "ood50_mlip_pass@2",
        "valid_mlip_pass@5": "ood50_mlip_pass@5",
        "valid_mlip_pass@10": "ood50_mlip_pass@10",
        "mlip_pass@1": "ood50_mlip_pass@1",
        "mlip_pass@2": "ood50_mlip_pass@2",
        "mlip_pass@5": "ood50_mlip_pass@5",
        "mlip_pass@10": "ood50_mlip_pass@10",
        "valid_rate": "ood50_valid_rate",
        "converged_rate": "ood50_converged_rate",
    }
    for src, dst in mapping.items():
        if src in summary:
            out[dst] = summary[src]
    nested = {
        "valid_sp_delta_E_sys": "ood50_pre_gap",
        "valid_final_delta_E_sys": "ood50_post_gap",
        "valid_n_steps": "ood50_relax_steps",
    }
    for src, dst in nested.items():
        val = summary.get(src)
        if isinstance(val, dict):
            for k in ("mean", "median", "min", "best", "p10", "p90"):
                if k in val:
                    name = "best" if k == "min" else k
                    out[f"{dst}_{name}"] = val[k]
    throughput = summary.get("throughput_8gpu")
    if isinstance(throughput, dict):
        for k, v in throughput.items():
            if isinstance(v, (int, float)):
                out[f"ood50_throughput_{k}"] = v
    return out


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-root", required=True)
    p.add_argument("--loop-idx", type=int, required=True)
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--train-run", default=None)
    p.add_argument("--train-ckpt", default=None)
    p.add_argument("--pass-summary", default=None)
    p.add_argument("--window-ev", type=float, default=0.05)
    p.add_argument("--new-best-tol-ev", type=float, default=1e-4)
    p.add_argument("--initial-ref-json", default=None)
    args = p.parse_args()

    metrics_root = Path(args.metrics_root)
    initial_ref_path = Path(args.initial_ref_json) if args.initial_ref_json else metrics_root / "initial_refs.json"
    rows_path = metrics_root / "loop_metrics.jsonl"
    csv_path = metrics_root / "loop_metrics.csv"

    replay_dir = Path(args.replay_dir)
    rows = _load_rows(replay_dir)
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[_key(row["system_key"])].append(row)

    current_refs = {
        sk: min(float(r["E_sys_ref"]) for r in rs if math.isfinite(float(r["E_sys_ref"])))
        for sk, rs in by_system.items()
    }
    if initial_ref_path.is_file():
        initial_refs = {str(k): float(v) for k, v in json.loads(initial_ref_path.read_text()).items()}
    else:
        initial_refs = dict(current_refs)
        initial_ref_path.parent.mkdir(parents=True, exist_ok=True)
        initial_ref_path.write_text(json.dumps(initial_refs, indent=2, sort_keys=True), encoding="utf-8")

    total = len(rows)
    finite_rows = [r for r in rows if math.isfinite(float(r.get("E_sys", float("nan"))))]
    converged_rows = [r for r in rows if bool(r.get("converged"))]
    geom_valid_rows = [r for r in rows if bool(r.get("geom_valid", r.get("valid", False)))]
    buffer_eligible_rows = [
        r for r in rows
        if bool(r.get("geom_valid", r.get("valid", False)))
        and bool(r.get("converged"))
        and math.isfinite(float(r.get("E_sys", float("nan"))))
    ]

    post_gap_initial = []
    post_gap_current = []
    eligible_gap_initial = []
    eligible_gap_current = []
    steps_all = []
    steps_eligible = []
    accepted_rows = []
    systems_new_best = set()
    systems_accepted = set()

    for row in rows:
        sk = _key(row["system_key"])
        e = float(row.get("E_sys", float("nan")))
        if not math.isfinite(e):
            continue
        init_ref = initial_refs.get(sk)
        cur_ref = current_refs.get(sk)
        if init_ref is not None:
            post_gap_initial.append(e - float(init_ref))
        if cur_ref is not None:
            gap_cur = e - float(cur_ref)
            post_gap_current.append(gap_cur)
            if gap_cur < -float(args.new_best_tol_ev):
                systems_new_best.add(sk)
        steps_all.append(float(row.get("n_steps", float("nan"))))
        if row in buffer_eligible_rows:
            if init_ref is not None:
                eligible_gap_initial.append(e - float(init_ref))
            if cur_ref is not None:
                gap_cur = e - float(cur_ref)
                eligible_gap_current.append(gap_cur)
                if gap_cur <= float(args.window_ev) + 1e-12:
                    accepted_rows.append(row)
                    systems_accepted.add(sk)
            steps_eligible.append(float(row.get("n_steps", float("nan"))))

    elapsed = 0.0
    for path in sorted(replay_dir.glob("shard_*.json")):
        data = _load_json(path)
        elapsed = max(elapsed, float(data.get("elapsed_sec", 0.0)))

    n_systems = len(by_system)
    metric: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "loop_idx": int(args.loop_idx),
        "train_run": args.train_run,
        "train_ckpt": args.train_ckpt,
        "replay_dir": str(replay_dir),
        "window_ev": float(args.window_ev),
        "new_best_tol_ev": float(args.new_best_tol_ev),
        "n_systems": n_systems,
        "n_candidates": total,
        "finite_energy_rate": len(finite_rows) / max(total, 1),
        "converged_rate": len(converged_rows) / max(total, 1),
        "geom_valid_rate": len(geom_valid_rows) / max(total, 1),
        "buffer_eligible_rate": len(buffer_eligible_rows) / max(total, 1),
        "accepted_count": len(accepted_rows),
        "accepted_rate": len(accepted_rows) / max(total, 1),
        "systems_with_accept_count": len(systems_accepted),
        "systems_with_accept_rate": len(systems_accepted) / max(n_systems, 1),
        "new_best_system_count": len(systems_new_best),
        "new_best_system_rate": len(systems_new_best) / max(n_systems, 1),
        "elapsed_sec": elapsed if elapsed > 0 else None,
        "throughput_candidates_per_sec": total / elapsed if elapsed > 0 else None,
    }
    metric.update(_stats(post_gap_initial, "post_gap_to_initial_eV"))
    metric.update(_stats(post_gap_current, "post_gap_to_current_eV"))
    metric.update(_stats(eligible_gap_initial, "eligible_post_gap_to_initial_eV"))
    metric.update(_stats(eligible_gap_current, "eligible_post_gap_to_current_eV"))
    metric.update(_stats(steps_all, "relax_steps_all"))
    metric.update(_stats(steps_eligible, "relax_steps_eligible"))
    metric.update(_extract_pass_summary(_load_json(Path(args.pass_summary)) if args.pass_summary else {}))

    _write_jsonl(rows_path, metric)
    all_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    _write_csv(csv_path, all_rows)
    latest_path = metrics_root / "latest_loop_metrics.json"
    latest_path.write_text(json.dumps(metric, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metric, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
