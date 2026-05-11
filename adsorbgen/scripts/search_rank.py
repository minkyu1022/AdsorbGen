"""Aggregate per-variant ``search_metrics.json`` rows into a leaderboard.

Scans ``--runs-root`` for every subdirectory containing a
``search_metrics.json`` file and prints a table sorted by the chosen metric
(``valid_rate_strict`` descending by default, tie-broken by ``displacement_mae_A``
ascending). Also writes the full table to ``<runs-root>/leaderboard.json``.

Usage:

    PYTHONPATH=AdsorbGen python -m adsorbgen.scripts.search_rank \
        --runs-root runs --baseline v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--baseline", default="v2",
                   help="variant name used as the delta reference in the table")
    p.add_argument("--sort-by", default="valid_rate_strict",
                   choices=["valid_rate_strict", "displacement_mae_A"])
    args = p.parse_args()

    root = Path(args.runs_root)
    rows: list[dict] = []
    for row_path in sorted(root.glob("*/search_metrics.json")):
        with open(row_path) as f:
            rows.append(json.load(f))
    if not rows:
        print(f"[rank] no search_metrics.json files under {root}/")
        return

    baseline = next((r for r in rows if r["variant"] == args.baseline), None)

    def sort_key(r):
        return (
            -r.get("valid_rate_strict", 0.0),
            r.get("displacement_mae_A", float("inf")),
        ) if args.sort_by == "valid_rate_strict" else (
            r.get("displacement_mae_A", float("inf")),
            -r.get("valid_rate_strict", 0.0),
        )

    rows.sort(key=sort_key)

    header = f"{'rank':<5}{'variant':<24}{'epochs':>7}{'n':>6}{'valid':>8}{'mae(Å)':>10}{'Δvalid':>10}{'Δmae':>10}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        d_valid = (r["valid_rate_strict"] - baseline["valid_rate_strict"]) if baseline else 0.0
        d_mae = (r["displacement_mae_A"] - baseline["displacement_mae_A"]) if baseline else 0.0
        marker = " *" if baseline and r["variant"] == args.baseline else "  "
        print(
            f"{i:<5}{r['variant']:<24}{r['epochs']:>7}{r['n_samples']:>6}"
            f"{r['valid_rate_strict']:>8.3f}{r['displacement_mae_A']:>10.4f}"
            f"{d_valid:>+10.3f}{d_mae:>+10.4f}{marker}"
        )

    out = {"baseline": args.baseline, "sort_by": args.sort_by, "rows": rows}
    (root / "leaderboard.json").write_text(json.dumps(out, indent=2))
    print(f"\n[rank] wrote {root/'leaderboard.json'}")


if __name__ == "__main__":
    main()
