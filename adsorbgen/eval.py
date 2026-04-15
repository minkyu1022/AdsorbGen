"""Geometric evaluation metrics for flow-matching generated structures.

Reads a samples `.pt` file written by ``adsorbgen.inference`` and computes:

    - displacement_mae   MAE of pred vs ground-truth positions (movable atoms)
    - displacement_rmse  RMSE ditto
    - atom_overlap_rate  fraction of samples with any pair of atoms closer
                         than ``overlap_factor * (r_cov_i + r_cov_j)``
    - dissociation_rate  fraction of samples where at least one adsorbate
                         atom has no surface/adsorbate neighbor within
                         ``dissoc_max_dist`` of its predicted position
    - valid_rate         1 - (overlap OR dissociation)
    - per-sample records with all the sub-metrics for post-hoc inspection

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.eval \
        --samples runs/fm_ft/samples.pt \
        --out     runs/fm_ft/metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


_COVALENT_RADII_PM = {
    1: 31, 2: 28, 3: 128, 4: 96, 5: 84, 6: 73, 7: 71, 8: 66, 9: 57,
    10: 58, 11: 166, 12: 141, 13: 121, 14: 111, 15: 107, 16: 105, 17: 102,
    18: 106, 19: 203, 20: 176, 21: 170, 22: 160, 23: 153, 24: 139, 25: 150,
    26: 142, 27: 138, 28: 124, 29: 132, 30: 122, 31: 122, 32: 120, 33: 119,
    34: 120, 35: 120, 36: 116, 37: 220, 38: 195, 39: 190, 40: 175, 41: 164,
    42: 154, 43: 147, 44: 146, 45: 142, 46: 139, 47: 145, 48: 144, 49: 142,
    50: 139, 51: 139, 52: 138, 53: 139, 54: 140, 55: 244, 56: 215, 57: 207,
    72: 175, 73: 170, 74: 162, 75: 151, 76: 144, 77: 141, 78: 136, 79: 136,
    80: 132, 81: 145, 82: 146, 83: 148,
}


def covalent_radii(atomic_numbers: torch.Tensor, default_pm: float = 150.0) -> torch.Tensor:
    """Lookup covalent radii in Angstroms for an atomic-number tensor."""
    out = torch.full(atomic_numbers.shape, default_pm, dtype=torch.float32)
    for z, rpm in _COVALENT_RADII_PM.items():
        out = torch.where(atomic_numbers == z, torch.tensor(float(rpm)), out)
    return out * 0.01  # pm -> Å


def _pairwise_min_fraction(
    pos: torch.Tensor,
    atomic_numbers: torch.Tensor,
    pad_mask: torch.Tensor,
) -> float:
    """Return min_{i<j} (d_ij / (r_cov_i + r_cov_j)) for real atoms only.

    Values < 1 mean atoms are closer than their covalent-radii sum; we flag
    overlap at ``ratio < overlap_factor``.
    """
    n = int(pad_mask.sum().item())
    if n < 2:
        return float("inf")
    p = pos[:n]
    z = atomic_numbers[:n]
    rcov = covalent_radii(z).to(p.dtype)
    diff = p.unsqueeze(0) - p.unsqueeze(1)
    d = diff.norm(dim=-1)
    d.fill_diagonal_(float("inf"))
    denom = rcov.unsqueeze(0) + rcov.unsqueeze(1)
    denom = denom.clamp_min(1e-6)
    ratio = d / denom
    return float(ratio.min().item())


def _adsorbate_connectivity_min(
    pos: torch.Tensor,
    tags: torch.Tensor,
    pad_mask: torch.Tensor,
) -> float:
    """Min distance from any adsorbate atom to the nearest surface/adsorbate atom.

    High values mean the adsorbate has drifted away from the slab — a
    crude "dissociation / desorption" proxy.
    """
    n = int(pad_mask.sum().item())
    p = pos[:n]
    t = tags[:n]
    ads = t == 2
    nbr = (t == 1) | (t == 2)
    if not ads.any() or nbr.sum().item() < 2:
        return float("inf")
    p_ads = p[ads]
    p_nbr = p[nbr]
    d = (p_ads.unsqueeze(1) - p_nbr.unsqueeze(0)).norm(dim=-1)
    d_masked = d.clone()
    # Remove self-distance for adsorbate atoms that appear in both sets.
    ads_to_nbr_self = torch.where(ads)[0].unsqueeze(1) == torch.where(nbr)[0].unsqueeze(0)
    d_masked[ads_to_nbr_self] = float("inf")
    min_per_ads = d_masked.min(dim=1).values
    return float(min_per_ads.max().item())


def compute_metrics(
    records: List[Dict],
    overlap_factor: float = 0.6,
    dissoc_max_dist: float = 3.0,
) -> Dict:
    """Aggregate metrics over a list of inference records."""
    per_sample = []
    disp_err_sum = 0.0
    disp_err_sq_sum = 0.0
    disp_err_count = 0
    overlap_count = 0
    dissoc_count = 0

    for r in records:
        pos_pred = r["pos_pred"]
        pos_gt = r["pos_gt"]
        mov = r["movable_mask"]
        pad = torch.ones_like(mov)  # pos_pred/pos_gt already truncated to real atoms

        # Displacement error on movable atoms.
        mov_idx = mov.bool()
        if mov_idx.any():
            diff = (pos_pred[mov_idx] - pos_gt[mov_idx]).norm(dim=-1)
            disp_err_sum += float(diff.sum().item())
            disp_err_sq_sum += float((diff ** 2).sum().item())
            disp_err_count += int(diff.numel())
            mae_sample = float(diff.mean().item())
            rmse_sample = float(diff.pow(2).mean().sqrt().item())
        else:
            mae_sample = float("nan")
            rmse_sample = float("nan")

        # Validity — overlap and dissociation on the predicted structure.
        overlap_ratio = _pairwise_min_fraction(pos_pred, r["atomic_numbers"], pad)
        has_overlap = overlap_ratio < overlap_factor
        overlap_count += int(has_overlap)

        nbr_dist = _adsorbate_connectivity_min(pos_pred, r["tags"], pad)
        has_dissoc = nbr_dist > dissoc_max_dist
        dissoc_count += int(has_dissoc)

        per_sample.append({
            "sid": r.get("sid"),
            "delta_e_cond": r.get("delta_e"),
            "mae": mae_sample,
            "rmse": rmse_sample,
            "overlap_ratio": overlap_ratio,
            "has_overlap": bool(has_overlap),
            "ads_neighbor_dist": nbr_dist,
            "has_dissoc": bool(has_dissoc),
            "valid": bool(not has_overlap and not has_dissoc),
        })

    N = len(records)
    aggregate = {
        "n_samples": N,
        "displacement_mae_A": (disp_err_sum / max(disp_err_count, 1)) if disp_err_count else float("nan"),
        "displacement_rmse_A": (
            float(np.sqrt(disp_err_sq_sum / max(disp_err_count, 1))) if disp_err_count else float("nan")
        ),
        "overlap_rate": overlap_count / max(N, 1),
        "dissociation_rate": dissoc_count / max(N, 1),
        "valid_rate": (N - overlap_count - dissoc_count + _both(per_sample)) / max(N, 1),
        "overlap_factor": overlap_factor,
        "dissoc_max_dist_A": dissoc_max_dist,
    }
    return {"aggregate": aggregate, "per_sample": per_sample}


def _both(per_sample: List[Dict]) -> int:
    return sum(1 for p in per_sample if p["has_overlap"] and p["has_dissoc"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--overlap-factor", type=float, default=0.6)
    p.add_argument("--dissoc-max-dist", type=float, default=3.0)
    args = p.parse_args()

    blob = torch.load(args.samples, weights_only=False)
    records = blob["records"]
    meta = blob.get("meta", {})

    metrics = compute_metrics(
        records,
        overlap_factor=args.overlap_factor,
        dissoc_max_dist=args.dissoc_max_dist,
    )
    metrics["meta"] = meta

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    agg = metrics["aggregate"]
    print(
        f"[eval] n={agg['n_samples']} "
        f"mae={agg['displacement_mae_A']:.4f} "
        f"rmse={agg['displacement_rmse_A']:.4f} "
        f"overlap={agg['overlap_rate']:.3f} "
        f"dissoc={agg['dissociation_rate']:.3f} "
        f"valid={agg['valid_rate']:.3f}",
        flush=True,
    )
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
