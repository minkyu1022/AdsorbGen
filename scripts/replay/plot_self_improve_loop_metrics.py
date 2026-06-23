#!/usr/bin/env python
"""Plot trends from ``record_self_improve_loop_metrics.py`` JSONL output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r.get("loop_idx", 0)))
    return rows


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for row in rows:
        val = row.get(key)
        if isinstance(val, (int, float)):
            xs.append(int(row.get("loop_idx", 0)))
            ys.append(float(val))
    return xs, ys


def _plot_group(rows: list[dict], out: Path, title: str, keys: list[tuple[str, str]], ylabel: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=160)
    plotted = 0
    for key, label in keys:
        xs, ys = _series(rows, key)
        if ys:
            ax.plot(xs, ys, marker="o", linewidth=1.8, label=label)
            plotted += 1
    if plotted == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("self-improvement loop")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-jsonl", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    rows = _load_rows(Path(args.metrics_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_group(
        rows,
        out_dir / "energy_gap_trends.png",
        "Energy Gap Trends",
        [
            ("eligible_post_gap_to_initial_eV_mean", "eligible mean vs initial"),
            ("eligible_post_gap_to_initial_eV_best", "eligible best vs initial"),
            ("eligible_post_gap_to_current_eV_mean", "eligible mean vs current"),
            ("eligible_post_gap_to_current_eV_best", "eligible best vs current"),
        ],
        "E_post - reference (eV)",
    )
    _plot_group(
        rows,
        out_dir / "buffer_trends.png",
        "Buffer / New Best Trends",
        [
            ("accepted_rate", "accepted candidates"),
            ("systems_with_accept_rate", "systems with accepted"),
            ("new_best_system_rate", "systems with new best"),
        ],
        "rate",
    )
    _plot_group(
        rows,
        out_dir / "validity_steps_throughput.png",
        "Validity, Steps, Throughput",
        [
            ("geom_valid_rate", "geom valid"),
            ("converged_rate", "converged"),
            ("buffer_eligible_rate", "buffer eligible"),
            ("relax_steps_eligible_mean", "eligible steps mean"),
            ("throughput_candidates_per_sec", "candidates/sec"),
        ],
        "mixed units",
    )
    _plot_group(
        rows,
        out_dir / "ood50_pass_trends.png",
        "OOD50 Paper-Style Pass@k Trends",
        [
            ("ood50_mlip_pass@1", "pass@1"),
            ("ood50_mlip_pass@2", "pass@2"),
            ("ood50_mlip_pass@5", "pass@5"),
            ("ood50_mlip_pass@10", "pass@10"),
        ],
        "pass@k",
    )
    (out_dir / "plot_manifest.json").write_text(
        json.dumps({
            "source": str(Path(args.metrics_jsonl)),
            "n_loops": len(rows),
            "plots": [
                "energy_gap_trends.png",
                "buffer_trends.png",
                "validity_steps_throughput.png",
                "ood50_pass_trends.png",
            ],
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
