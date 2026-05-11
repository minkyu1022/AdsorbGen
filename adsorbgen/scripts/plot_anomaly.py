"""Plots for fairchem DetectTrajAnomaly-based validity metrics.

Assumes ``rescore_anomaly.py`` has already written
``runs/<variant>/search_metrics_anomaly.json`` and (optionally)
``runs/<variant>/epoch_scan/ep{e}_metrics_anomaly.json``.

Outputs (under ``runs/plots_anomaly/``):
  1. anomaly_leaderboard.png
     - Top: strict valid_rate per variant, sorted
     - Bottom: per-anomaly rate breakdown (grouped bars)
  2. epoch_anomaly_{group}.png   (v1-to-v2 and v2-variants groups)
     - Multi-panel: strict valid, dissoc, surf_changed, desorbed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/minkyu/Cat-bench/runs")
OUT = ROOT / "plots_anomaly"

# -- same grouping as runs/epoch_curves/aggregate_and_plot.py -----------------
GROUP_V1_TO_V2 = [
    ("v1-retrained",    "v1 retrained",          "C0"),
    ("v1-no-enrich",    "v1 - outer enrich",     "C7"),
    ("v1-dec-pair",     "v1 + dec pair bias",    "C1"),
    ("v1-wide",         "v1 wide (uniform 512)", "C5"),
    ("v1-wide-no-gate", "v1 wide - ReLU gates",  "C2"),
    ("v6-depth-20",     "single trunk d=20",     "C3"),
    ("v2",              "v2 (single trunk d=13)","C4"),
]
GROUP_V2_VARIANTS = [
    ("v2",              "v2 baseline",     "C4"),
    ("v3-no-ads-pair",  "-ads pair",       "C1"),
    ("v4-gaussian-dist","gaussian dist",   "C2"),
    ("v8-ads-surf-pair","+ads-surf pair",  "C3"),
    ("v9-dynamic-pair", "dynamic pair",    "C0"),
    ("v10-self-cond",   "+self-cond",      "C5"),
    ("v11-cross-attn",  "cross-attn 2-str","C6"),
]

ANOMALY_KEYS = [
    ("dissoc_rate",       "dissociation",     "#d62728"),
    ("surf_changed_rate", "surface changed",  "#ff7f0e"),
    ("desorbed_rate",     "desorbed",         "#2ca02c"),
    ("intercalated_rate", "intercalated",     "#9467bd"),
    ("overlap_rate",      "overlap (<0.5A)",  "#8c564b"),
]


def _load_agg(path: Path) -> Dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["aggregate"]


# ---------------------------------------------------------------------------
# Plot 1: leaderboard
# ---------------------------------------------------------------------------


def plot_leaderboard(out_path: Path) -> None:
    rows: List[Tuple[str, Dict]] = []
    for vdir in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("v")):
        agg = _load_agg(vdir / "search_metrics_anomaly.json")
        if agg is None:
            # v1 is the only one whose rescored file was named *_oc20dense
            agg = _load_agg(vdir / "search_metrics_anomaly_oc20dense.json")
        if agg is None:
            continue
        rows.append((vdir.name, agg))

    rows.sort(key=lambda r: r[1]["valid_rate_strict"], reverse=True)
    names = [r[0] for r in rows]
    valids = [r[1]["valid_rate_strict"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 0.7 * len(rows)), 8.5),
                             gridspec_kw={"height_ratios": [1, 1.2]}, sharex=True)

    # top: strict valid_rate
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))
    bars = ax.bar(names, valids, color=colors)
    for b, v in zip(bars, valids):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("strict valid_rate\n(1 - any(dissoc|surf|desorb|inter|overlap))")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("fairchem DetectTrajAnomaly leaderboard  (512 samples / oc20dense_val)")

    # bottom: per-anomaly rates (grouped)
    ax = axes[1]
    x = np.arange(len(rows))
    width = 0.16
    for i, (k, label, color) in enumerate(ANOMALY_KEYS):
        vals = [r[1][k] for r in rows]
        ax.bar(x + (i - 2) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("anomaly rate")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncol=5, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: epoch curves by group
# ---------------------------------------------------------------------------


def _epoch_series(variant: str, metric: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    vdir = ROOT / variant / "epoch_scan"
    if not vdir.exists():
        return xs, ys
    for path in sorted(vdir.glob("ep*_metrics_anomaly.json")):
        try:
            e = int(path.stem.split("ep")[1].split("_")[0])
        except ValueError:
            continue
        agg = _load_agg(path)
        if agg is None:
            continue
        xs.append(e)
        ys.append(agg[metric])
    return xs, ys


def plot_epoch_group(group: List[Tuple[str, str, str]], title: str, out_path: Path) -> None:
    metrics = [
        ("valid_rate_strict", "strict valid_rate"),
        ("dissoc_rate",       "dissociation"),
        ("surf_changed_rate", "surface changed"),
        ("desorbed_rate",     "desorbed"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.2), sharex=True)
    any_data = False
    for ax, (mkey, mlabel) in zip(axes, metrics):
        for variant, vlabel, color in group:
            xs, ys = _epoch_series(variant, mkey)
            if not xs:
                continue
            any_data = True
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4.5,
                    label=vlabel, color=color)
        ax.set_xlabel("epoch")
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if not any_data:
        plt.close(fig)
        print(f"[plot] skip {out_path} (no epoch_scan anomaly data yet)")
        return
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-epoch", action="store_true",
                    help="skip epoch-curve plots (useful before epoch rescoring finishes)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    plot_leaderboard(OUT / "anomaly_leaderboard.png")
    if not args.skip_epoch:
        plot_epoch_group(GROUP_V1_TO_V2, "v1 -> v2 ablation ladder (strict validity)",
                         OUT / "epoch_anomaly_v1_to_v2.png")
        plot_epoch_group(GROUP_V2_VARIANTS, "v2 variants (strict validity)",
                         OUT / "epoch_anomaly_v2_variants.png")


if __name__ == "__main__":
    main()
