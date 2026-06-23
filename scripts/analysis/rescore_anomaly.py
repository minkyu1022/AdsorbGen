"""Re-score saved inference dumps with fairchem DetectTrajAnomaly metrics.

Walks ``runs/<variant>/search_samples*.pt`` (and optionally
``runs/<variant>/epoch_scan/ep*_samples.pt``), applies
``adsorbgen.evaluation.metrics.compute_anomaly_metrics`` with the user-defined protocol
(init=pos_ref, final=pos_pred, final_slab=bare/pristine relaxed slab, default cutoffs),
and writes ``search_metrics_anomaly.json`` / ``ep{e}_metrics_anomaly.json``
next to each source file.

Usage:
    PYTHONPATH=AdsorbGen python scripts/analysis/rescore_anomaly.py \
        --runs-root runs --workers 16 \
        --pristine-slabs /path/to/bare_slabs.pkl \
        --pristine-index /path/to/sid_index.pkl \
        [--epoch-scan] [--variants v2,v11-cross-attn]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import torch

from adsorbgen.evaluation.metrics import compute_anomaly_metrics


def _discover_final(runs_root: Path, variants: List[str] | None) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        name = d.name
        if variants and name not in variants:
            continue
        for fname in ("search_samples.pt", "search_samples_oc20dense.pt"):
            src = d / fname
            if src.exists():
                out_name = (
                    "search_metrics_anomaly.json"
                    if fname == "search_samples.pt"
                    else "search_metrics_anomaly_oc20dense.json"
                )
                out.append((name, src, d / out_name))
    return out


def _discover_epoch_scan(runs_root: Path, variants: List[str] | None) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    for d in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        name = d.name
        if variants and name not in variants:
            continue
        es = d / "epoch_scan"
        if not es.exists():
            continue
        for src in sorted(es.glob("ep*_samples.pt")):
            stem = src.stem  # "ep3_samples"
            ep_tag = stem.replace("_samples", "")  # "ep3"
            out.append((name, src, es / f"{ep_tag}_metrics_anomaly.json"))
    return out


def _run_one(
    label: str,
    src: Path,
    dst: Path,
    workers: int,
    pristine_slabs: str,
    pristine_index: str,
) -> dict:
    t0 = time.time()
    blob = torch.load(src, map_location="cpu", weights_only=False)
    records = blob["records"]
    metrics = compute_anomaly_metrics(
        records,
        workers=workers,
        pristine_slabs=pristine_slabs,
        pristine_sid_index=pristine_index,
    )
    metrics["meta"] = blob.get("meta", {})
    metrics["meta"]["source"] = str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        json.dump(metrics, f, indent=2)
    agg = metrics["aggregate"]
    dt = time.time() - t0
    print(
        f"[rescore] {label:28s} n={agg['n_samples']:4d} "
        f"valid={agg['valid_rate_strict']:.3f} "
        f"over={agg['overlap_rate']:.3f} "
        f"diss={agg['dissoc_rate']:.3f} "
        f"deso={agg['desorbed_rate']:.3f} "
        f"inter={agg['intercalated_rate']:.3f} "
        f"surfC={agg['surf_changed_rate']:.3f} "
        f"err={agg['n_errors']} t={dt:.1f}s -> {dst.name}",
        flush=True,
    )
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("runs"))
    ap.add_argument("--variants", type=str, default=None,
                    help="comma-separated subset; default = all")
    ap.add_argument("--epoch-scan", action="store_true",
                    help="also process epoch_scan dumps (10x more work)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only-missing", action="store_true",
                    help="skip source files whose output already exists")
    ap.add_argument("--pristine-slabs", type=str, required=True,
                    help="bare/pristine relaxed-slab pkl used for surface-change checks")
    ap.add_argument("--pristine-index", "--pristine-sid-index",
                    dest="pristine_index", type=str, required=True,
                    help="sid/system_key -> bare slab key index pkl")
    args = ap.parse_args()

    variants = [s.strip() for s in args.variants.split(",")] if args.variants else None

    tasks: List[Tuple[str, Path, Path]] = _discover_final(args.runs_root, variants)
    if args.epoch_scan:
        tasks += _discover_epoch_scan(args.runs_root, variants)

    if args.only_missing:
        tasks = [(n, s, d) for (n, s, d) in tasks if not d.exists()]

    print(f"[rescore] {len(tasks)} files to process, workers={args.workers}", flush=True)
    for name, src, dst in tasks:
        label = f"{name}/{src.name}"
        _run_one(
            label,
            src,
            dst,
            workers=args.workers,
            pristine_slabs=args.pristine_slabs,
            pristine_index=args.pristine_index,
        )


if __name__ == "__main__":
    main()
