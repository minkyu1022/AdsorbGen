#!/usr/bin/env python
"""Monitor a second self-improvement replay and compare it to the first run."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def frozen_key(x: Any):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def load_rows(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("rows_shard*.pkl")):
        with path.open("rb") as f:
            rows.extend(pickle.load(f))
    return rows


def load_successes(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(run_dir.glob("success_shard*.pkl")):
        with path.open("rb") as f:
            rows.extend(pickle.load(f))
    return rows


def best_by_system(successes: list[dict]) -> dict:
    best = {}
    for row in successes:
        sk = frozen_key(row["system_key"])
        cur = best.get(sk)
        if cur is None or float(row["E_sys"]) < float(cur["E_sys"]):
            best[sk] = row
    return best


def aggregate(rows: list[dict], successes: list[dict], n_systems: int) -> dict[str, Any]:
    n = max(len(rows), 1)
    best = best_by_system(successes)
    imps = np.asarray([float(r["improvement"]) for r in successes], dtype=np.float64)
    conv = sum(1 for r in rows if r.get("converged"))
    valid = sum(1 for r in rows if r.get("valid"))
    succ = sum(1 for r in rows if r.get("success"))
    out: dict[str, Any] = {
        "systems": int(n_systems),
        "candidates": int(len(rows)),
        "converged": int(conv),
        "valid": int(valid),
        "success": int(succ),
        "systems_with_success": int(len(best)),
        "converged_rate": float(conv / n),
        "valid_rate": float(valid / n),
        "success_sample_rate": float(succ / n),
        "success_among_converged": float(succ / max(conv, 1)),
        "success_among_valid": float(succ / max(valid, 1)),
        "success_system_rate": float(len(best) / max(n_systems, 1)),
    }
    if imps.size:
        out.update({
            "improvement_mean": float(imps.mean()),
            "improvement_p50": float(np.quantile(imps, 0.50)),
            "improvement_p90": float(np.quantile(imps, 0.90)),
            "improvement_max": float(imps.max()),
        })
    else:
        out.update({
            "improvement_mean": None,
            "improvement_p50": None,
            "improvement_p90": None,
            "improvement_max": None,
        })
    return out


def summarize_progress(run_dir: Path) -> dict[str, Any]:
    progress = []
    for path in sorted((run_dir / "logs").glob("progress_shard*.json")):
        try:
            progress.append(json.loads(path.read_text()))
        except Exception:
            pass
    if not progress:
        return {"progress_files": 0}
    candidates = sum(int(p.get("candidates", 0)) for p in progress)
    target_candidates = max(int(p.get("target_candidates", 0)) for p in progress)
    return {
        "progress_files": len(progress),
        "candidates": candidates,
        "target_candidates": target_candidates,
        "candidate_rate": candidates / max(target_candidates, 1),
        "converged": sum(int(p.get("converged", 0)) for p in progress),
        "valid": sum(int(p.get("valid", 0)) for p in progress),
        "success": sum(int(p.get("success", 0)) for p in progress),
        "elapsed_sec_max": max(float(p.get("elapsed_sec", 0.0)) for p in progress),
        "updated_at_max": max(str(p.get("updated_at", "")) for p in progress),
    }


def write_report(first_dir: Path, second_dir: Path, out_md: Path, out_json: Path) -> None:
    selected = json.loads((second_dir / "selected_systems.json").read_text())
    systems = selected["systems"]
    n_systems = len(systems)
    original_ref = {frozen_key(r["system_key"]): float(r.get("E_sys_ref_original", r["E_sys_ref"])) for r in systems}
    second_ref = {frozen_key(r["system_key"]): float(r["E_sys_ref"]) for r in systems}

    first_rows = load_rows(first_dir)
    second_rows = load_rows(second_dir)
    first_successes = load_successes(first_dir)
    second_successes = load_successes(second_dir)
    first_best = best_by_system(first_successes)
    second_best = best_by_system(second_successes)

    first_agg = aggregate(first_rows, first_successes, n_systems)
    second_agg = aggregate(second_rows, second_successes, n_systems)

    overlap_systems = sorted(set(first_best) & set(second_best), key=str)
    second_new_systems = sorted(set(second_best) - set(first_best), key=str)
    first_only_systems = sorted(set(first_best) - set(second_best), key=str)
    second_better_than_first = []
    best_overall = {}
    for sk in set(first_best) | set(second_best):
        f = first_best.get(sk)
        s = second_best.get(sk)
        if f is not None and s is not None and float(s["E_sys"]) < float(f["E_sys"]):
            second_better_than_first.append(sk)
        best = s if f is None else f
        if s is not None and float(s["E_sys"]) < float(best["E_sys"]):
            best = s
        best_overall[sk] = best

    second_vs_original_imps = []
    second_vs_first_ref_imps = []
    incremental_gains = []
    for sk, row in second_best.items():
        second_vs_original_imps.append(original_ref[sk] - float(row["E_sys"]))
        second_vs_first_ref_imps.append(second_ref[sk] - float(row["E_sys"]))
        if sk in first_best:
            incremental_gains.append(float(first_best[sk]["E_sys"]) - float(row["E_sys"]))

    overall_original_imps = [
        original_ref[sk] - float(row["E_sys"])
        for sk, row in best_overall.items()
        if sk in original_ref
    ]

    def stats(vals: list[float]) -> dict[str, Any]:
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            return {"n": 0, "mean": None, "p50": None, "p90": None, "max": None}
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(arr.max()),
        }

    payload = {
        "first_dir": str(first_dir),
        "second_dir": str(second_dir),
        "reference_mode_second": selected.get("reference_mode"),
        "num_refs_updated_from_first_replay": selected.get("num_refs_updated_from_first_replay"),
        "first": first_agg,
        "second": second_agg,
        "comparison": {
            "first_systems_with_success": len(first_best),
            "second_systems_with_success": len(second_best),
            "overlap_systems_with_success": len(overlap_systems),
            "first_only_systems_with_success": len(first_only_systems),
            "second_new_systems_with_success": len(second_new_systems),
            "second_better_than_first_on_overlap": len(second_better_than_first),
            "best_overall_systems_after_two_replays": len(best_overall),
            "second_success_improvement_vs_original": stats(second_vs_original_imps),
            "second_success_increment_vs_second_reference": stats(second_vs_first_ref_imps),
            "second_incremental_gain_vs_first_best_on_overlap": stats(incremental_gains),
            "best_overall_improvement_vs_original": stats(overall_original_imps),
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def pct(x: float | None) -> str:
        return "NA" if x is None else f"{100.0 * x:.2f}%"

    def ev(x: float | None) -> str:
        return "NA" if x is None else f"{x:.4f} eV"

    c = payload["comparison"]
    lines = [
        "# Second Replay Comparison",
        "",
        f"- First replay: `{first_dir}`",
        f"- Second replay: `{second_dir}`",
        f"- Second replay reference mode: `{payload['reference_mode_second']}`",
        f"- References updated from first replay: `{payload['num_refs_updated_from_first_replay']}` systems",
        "",
        "## Aggregate Rates",
        "",
        "| Metric | First replay | Second replay |",
        "|---|---:|---:|",
        f"| Candidates | {first_agg['candidates']} | {second_agg['candidates']} |",
        f"| Converged | {first_agg['converged']} ({pct(first_agg['converged_rate'])}) | {second_agg['converged']} ({pct(second_agg['converged_rate'])}) |",
        f"| Valid | {first_agg['valid']} ({pct(first_agg['valid_rate'])}) | {second_agg['valid']} ({pct(second_agg['valid_rate'])}) |",
        f"| Success samples | {first_agg['success']} ({pct(first_agg['success_sample_rate'])}) | {second_agg['success']} ({pct(second_agg['success_sample_rate'])}) |",
        f"| Success / converged | {pct(first_agg['success_among_converged'])} | {pct(second_agg['success_among_converged'])} |",
        f"| Success / valid | {pct(first_agg['success_among_valid'])} | {pct(second_agg['success_among_valid'])} |",
        f"| Systems with >=1 success | {first_agg['systems_with_success']} ({pct(first_agg['success_system_rate'])}) | {second_agg['systems_with_success']} ({pct(second_agg['success_system_rate'])}) |",
        "",
        "## Energy Improvements",
        "",
        "| Metric | First replay | Second replay |",
        "|---|---:|---:|",
        f"| Success improvement mean | {ev(first_agg['improvement_mean'])} | {ev(second_agg['improvement_mean'])} |",
        f"| Success improvement p50 | {ev(first_agg['improvement_p50'])} | {ev(second_agg['improvement_p50'])} |",
        f"| Success improvement p90 | {ev(first_agg['improvement_p90'])} | {ev(second_agg['improvement_p90'])} |",
        f"| Success improvement max | {ev(first_agg['improvement_max'])} | {ev(second_agg['improvement_max'])} |",
        "",
        "## Cross-Replay System Comparison",
        "",
        f"- Overlap systems with success in both: `{c['overlap_systems_with_success']}`",
        f"- Systems successful only in first replay: `{c['first_only_systems_with_success']}`",
        f"- Systems newly successful in second replay: `{c['second_new_systems_with_success']}`",
        f"- Second replay beats first replay best on overlap: `{c['second_better_than_first_on_overlap']}`",
        f"- Best-over-two-replays systems: `{c['best_overall_systems_after_two_replays']}`",
        f"- Second success improvement vs original mean: {ev(c['second_success_improvement_vs_original']['mean'])}",
        f"- Incremental second gain vs first/current reference mean: {ev(c['second_success_increment_vs_second_reference']['mean'])}",
        f"- Best-over-two improvement vs original mean: {ev(c['best_overall_improvement_vs_original']['mean'])}",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--first-dir", required=True)
    p.add_argument("--second-dir", required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--interval-sec", type=int, default=600)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    first_dir = Path(args.first_dir)
    second_dir = Path(args.second_dir)
    log_path = second_dir / "logs" / "second_replay_monitor.log"
    report_md = second_dir / "second_vs_first_report.md"
    report_json = second_dir / "second_vs_first_report.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        progress = summarize_progress(second_dir)
        shard_jsons = sorted(second_dir.glob("shard_*.json"))
        line = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "done_shards": len(shard_jsons),
            **progress,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")

        if len(shard_jsons) >= args.num_shards:
            write_report(first_dir, second_dir, report_md, report_json)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "status": "report_written",
                    "report_md": str(report_md),
                    "report_json": str(report_json),
                }, sort_keys=True) + "\n")
            break
        if args.once:
            break
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
