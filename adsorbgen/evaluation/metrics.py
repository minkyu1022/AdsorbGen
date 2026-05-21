"""Geometric evaluation metrics for flow-matching generated structures.

Reads a samples `.pt` file written by ``adsorbgen.inference`` and computes:

    - displacement_mae    MAE of pred vs ground-truth positions (movable atoms)
    - displacement_rmse   RMSE ditto
    - 6-axis strict validity via ``compute_anomaly_metrics`` using fairchem's
      ``DetectTrajAnomaly`` (``fairchem.data.oc.utils.flag_anomaly``) plus an
      absolute-distance overlap (any pair MIC distance < 0.5 A).

Mapping from our record fields to DetectTrajAnomaly inputs:

    init_atoms        = pos_ref   (unrelaxed initial placement, IS)
    final_atoms       = pos_pred  (model prediction of the relaxed adslab)
    final_slab_atoms  = pristine relaxed slab (no adsorbate), looked up via
                        sid/system_key -> slab_key -> pristine pkl. Falls back to
                        ``pos_gt`` restricted to tag != 2 when the pristine
                        DB is not provided or the sid is missing — used so
                        runs from before the pristine extraction still work.

Cutoffs are left at DetectTrajAnomaly defaults (surface_change=1.5,
desorption=1.5). ``valid_rate_strict = 1 - any(4 flags OR overlap)``.

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.evaluation.metrics \
        --samples runs/fm_ft/samples.pt \
        --out     runs/fm_ft/metrics.json \
        --pristine-slabs results/pristine_slabs/is2res.pkl \
        --pristine-index results/pristine_slabs/is2res.sid_index.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# Module-level handle so multiprocessing fork workers inherit the loaded
# pristine DB without needing to pickle it across the pool boundary.
_PRISTINE_CTX: dict = {"db": None, "sid_to_key": None, "system_to_key": None}

# Lazy-loaded adsorbates.pkl for canonical-bond-graph dissoc reference.
_ADS_DB_CTX: dict = {"db": None, "path": "/home/irteam/data/pkls/adsorbates.pkl"}


def _load_ads_db():
    if _ADS_DB_CTX["db"] is None:
        try:
            with open(_ADS_DB_CTX["path"], "rb") as f:
                _ADS_DB_CTX["db"] = pickle.load(f)
        except Exception:
            _ADS_DB_CTX["db"] = {}
    return _ADS_DB_CTX["db"]


def _split_pristine_index(index: dict) -> tuple[dict, dict]:
    sid_to_key = {}
    system_to_key = {}
    for k, v in index.items():
        if isinstance(k, (int, np.integer)):
            sid_to_key[int(k)] = v
            continue
        if isinstance(k, str) and k.lstrip("-").isdigit():
            sid_to_key[int(k)] = v
        else:
            system_to_key[str(k)] = v
    return sid_to_key, system_to_key


def load_pristine_context(pristine_pkl: Optional[Path],
                          sid_index_pkl: Optional[Path]) -> Optional[dict]:
    """Load pristine slab DB + optional sid/system indexes into module ctx.

    Returns the loaded context (or None if no pristine path was given).
    """
    if pristine_pkl is None:
        _PRISTINE_CTX["db"] = None
        _PRISTINE_CTX["sid_to_key"] = None
        _PRISTINE_CTX["system_to_key"] = None
        return None
    pristine_pkl = Path(pristine_pkl)
    if sid_index_pkl is None:
        sid_candidate = pristine_pkl.with_suffix(".sid_index.pkl")
        system_candidate = pristine_pkl.with_suffix(".system_index.pkl")
        if sid_candidate.exists():
            sid_index_pkl = sid_candidate
        elif system_candidate.exists():
            sid_index_pkl = system_candidate
    with open(pristine_pkl, "rb") as f:
        db = pickle.load(f)
    sid_to_key = {}
    system_to_key = {}
    sid_index_pkl = Path(sid_index_pkl) if sid_index_pkl is not None else None
    if sid_index_pkl is not None and sid_index_pkl.exists():
        with open(sid_index_pkl, "rb") as f:
            sid_to_key, system_to_key = _split_pristine_index(pickle.load(f))
    _PRISTINE_CTX["db"] = db
    _PRISTINE_CTX["sid_to_key"] = sid_to_key
    _PRISTINE_CTX["system_to_key"] = system_to_key
    return _PRISTINE_CTX


def _pristine_record_pos(rec) -> Optional[np.ndarray]:
    if rec is None:
        return None
    if isinstance(rec, dict):
        rec = rec.get("pos")
    if rec is None:
        return None
    return np.asarray(rec, dtype=np.float64)


def _lookup_pristine_pos(sid: Optional[int], system_key: Optional[str] = None) -> Optional[np.ndarray]:
    db = _PRISTINE_CTX["db"]
    sid_to_key = _PRISTINE_CTX["sid_to_key"]
    system_to_key = _PRISTINE_CTX["system_to_key"]
    if db is None:
        return None
    if sid_to_key is not None and sid is not None and sid >= 0:
        key = sid_to_key.get(int(sid))
        pos = _pristine_record_pos(db.get(key)) if key is not None else None
        if pos is not None:
            return pos
    if system_to_key is not None and system_key:
        key = system_to_key.get(str(system_key))
        pos = _pristine_record_pos(db.get(key)) if key is not None else None
        if pos is not None:
            return pos
    if system_key:
        return _pristine_record_pos(db.get(str(system_key)))
    return None


def compute_displacement_metrics(records: List[Dict]) -> Dict:
    """Aggregate MAE/RMSE of pred vs ground-truth on movable atoms."""
    per_sample = []
    disp_err_sum = 0.0
    disp_err_sq_sum = 0.0
    disp_err_count = 0

    for r in records:
        pos_pred = r["pos_pred"]
        pos_gt = r["pos_gt"]
        mov_idx = r["movable_mask"].bool()
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
        per_sample.append({
            "sid": r.get("sid"),
            "delta_e_cond": r.get("delta_e"),
            "mae": mae_sample,
            "rmse": rmse_sample,
        })

    N = len(records)
    aggregate = {
        "n_samples": N,
        "displacement_mae_A": (disp_err_sum / max(disp_err_count, 1)) if disp_err_count else float("nan"),
        "displacement_rmse_A": (
            float(np.sqrt(disp_err_sq_sum / max(disp_err_count, 1))) if disp_err_count else float("nan")
        ),
        # Raw sums for DDP all_reduce aggregation across ranks:
        "displacement_err_sum": float(disp_err_sum),
        "displacement_err_sq_sum": float(disp_err_sq_sum),
        "displacement_err_count": int(disp_err_count),
    }
    return {"aggregate": aggregate, "per_sample": per_sample}


# ---------------------------------------------------------------------------
# 5-axis validity metrics (absolute-distance overlap + fairchem DetectTrajAnomaly)
# ---------------------------------------------------------------------------
#
#   Axis              | What                                         | Scope
#   ------------------+----------------------------------------------+----------------
#   overlap           | any MIC pair distance < OVERLAP_MIN_DIST_A   | all atom pairs
#   dissoc            | is_adsorbate_dissociated (init vs final)     | ads topology
#   desorbed          | is_adsorbate_desorbed                         | ads-slab bonds
#   intercalated      | is_adsorbate_intercalated                     | ads-bulk bonds
#   surf_changed      | has_surface_changed vs final_slab_atoms       | slab topology
#
# ads-surf overlap axis via fairchem there_is_overlap was considered and
# dropped: that check uses a 1.0x covalent-radii threshold which is meant
# for unrelaxed placement, and falsely flags normal chemisorption bond
# lengths on relaxed structures (GT hit rate ~= pred hit rate ~= 70%).

OVERLAP_MIN_DIST_A = 0.5  # absolute, covalent-radii agnostic, all pairs

_VALIDITY_FLAGS = (
    "overlap",
    "dissoc",
    "desorbed",
    "intercalated",
    "surf_changed",
)


def _pos_first_placement(pos: torch.Tensor) -> torch.Tensor:
    """Return (N, 3). If pos has a leading K dimension, take placement 0."""
    return pos[0] if pos.dim() == 3 else pos


def _build_atoms_triplet(record: Dict):
    """Build (init_atoms, final_atoms, final_slab_atoms, tags_np) per protocol."""
    from ase import Atoms

    pos_ref = _pos_first_placement(record["pos_ref"])
    pos_pred = _pos_first_placement(record["pos_pred"])
    pos_gt = record["pos_gt"]
    z = record["atomic_numbers"]
    tags = record["tags"]
    cell = record["cell"]

    z_np = z.numpy()
    cell_np = cell.numpy()
    tags_np = tags.numpy()

    init_atoms = Atoms(
        numbers=z_np, positions=pos_ref.numpy(), cell=cell_np, pbc=True
    )
    init_atoms.set_tags(tags_np)

    final_atoms = Atoms(
        numbers=z_np, positions=pos_pred.numpy(), cell=cell_np, pbc=True
    )
    final_atoms.set_tags(tags_np)

    slab_mask = tags_np != 2

    pristine_pos = _lookup_pristine_pos(record.get("sid"), record.get("system_key"))
    if pristine_pos is not None and pristine_pos.shape[0] == int(slab_mask.sum()):
        slab_positions = pristine_pos
    else:
        # Fallback: pos_gt[slab_mask]. Used when the pristine DB isn't loaded
        # or the sid wasn't indexed (e.g. older sample dumps).
        slab_positions = pos_gt.numpy()[slab_mask]

    final_slab_atoms = Atoms(
        numbers=z_np[slab_mask],
        positions=slab_positions,
        cell=cell_np,
        pbc=True,
    )
    final_slab_atoms.set_tags(tags_np[slab_mask])

    return init_atoms, final_atoms, final_slab_atoms, tags_np


def _min_pair_distance_mic(final_atoms) -> float:
    """Min MIC distance among all atom pairs (ads-ads, ads-slab, slab-slab)."""
    import numpy as np

    if len(final_atoms) < 2:
        return float("inf")
    d = final_atoms.get_all_distances(mic=True)
    if d.size == 0:
        return float("inf")
    m = d.copy()
    np.fill_diagonal(m, np.inf)
    return float(m.min())


def _score_record_anomaly(record: Dict) -> Dict:
    """Score a single record against the 6-axis validity scheme.

    Top-level function so multiprocessing can pickle it.
    """
    from fairchem.data.oc.utils import DetectTrajAnomaly

    pos_pred = _pos_first_placement(record["pos_pred"])

    result: Dict = {"sid": record.get("sid"), "error": None}
    for k in _VALIDITY_FLAGS:
        result[f"has_{k}"] = None

    if not torch.isfinite(pos_pred).all():
        result["error"] = "nonfinite_pred"
        for k in _VALIDITY_FLAGS:
            result[f"has_{k}"] = True
        result["is_any_anomaly"] = True
        result["valid_strict"] = False
        return result

    init_atoms, final_atoms, final_slab_atoms, tags_np = _build_atoms_triplet(record)

    pair_min = _min_pair_distance_mic(final_atoms)
    result["has_overlap"] = bool(pair_min < OVERLAP_MIN_DIST_A)
    result["min_pair_distance_A"] = pair_min

    # Build a canonical-reference init_atoms for the dissoc bond-graph check.
    # x_0 from a non-rigid prior (e.g. CatFlow Gaussian rel-pos) does not have
    # a valid molecular bond graph, so the init-vs-final comparison was a
    # massive source of false positives. When ads_id is available, replace ads
    # positions in init with the gas-phase canonical geometry from
    # adsorbates.pkl; this leaves the comparison meaningful regardless of
    # prior. Falls back silently if ads_id absent or atom order mismatched.
    ads_id = int(record.get("ads_id", -1)) if record.get("ads_id") is not None else -1
    init_atoms_diss = init_atoms
    if ads_id >= 0:
        db = _load_ads_db()
        entry = db.get(ads_id)
        if entry is not None:
            canon_atoms = entry[0]
            canon_z = np.asarray(canon_atoms.get_atomic_numbers(), dtype=np.int64)
            ads_mask = (tags_np == 2)
            rec_ads_z = np.asarray(record["atomic_numbers"].numpy(), dtype=np.int64)[ads_mask]
            if canon_z.size == rec_ads_z.size and np.array_equal(canon_z, rec_ads_z):
                from ase import Atoms as _Atoms
                init_pos = init_atoms.get_positions().copy()
                init_pos[ads_mask] = canon_atoms.get_positions()
                init_atoms_diss = _Atoms(
                    numbers=init_atoms.get_atomic_numbers(),
                    positions=init_pos,
                    cell=init_atoms.cell,
                    pbc=init_atoms.pbc,
                )
                init_atoms_diss.set_tags(tags_np.tolist())

    det = DetectTrajAnomaly(
        init_atoms=init_atoms_diss,
        final_atoms=final_atoms,
        atoms_tag=tags_np.tolist(),
        final_slab_atoms=final_slab_atoms,
    )
    result["has_dissoc"] = bool(det.is_adsorbate_dissociated())
    result["has_desorbed"] = bool(det.is_adsorbate_desorbed())
    result["has_intercalated"] = bool(det.is_adsorbate_intercalated())
    result["has_surf_changed"] = bool(det.has_surface_changed())

    flags_bool = [result[f"has_{k}"] for k in _VALIDITY_FLAGS]
    is_any = any(bool(f) for f in flags_bool)
    result["is_any_anomaly"] = is_any
    result["valid_strict"] = not is_any
    return result


def compute_anomaly_metrics(
    records: List[Dict],
    workers: int = 1,
    pristine_slabs: Optional[Path] = None,
    pristine_sid_index: Optional[Path] = None,
) -> Dict:
    """6-axis validity metrics. See module docstring for the protocol."""
    pristine_loaded = load_pristine_context(pristine_slabs, pristine_sid_index)
    if workers and workers > 1:
        import multiprocessing as mp

        # 'fork' inherits the module globals (incl. _PRISTINE_CTX) into workers
        # without re-pickling the (potentially huge) pristine DB.
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            per_sample = list(
                pool.imap(_score_record_anomaly, records, chunksize=16)
            )
    else:
        per_sample = [_score_record_anomaly(r) for r in records]

    N = len(records)
    counts = {k: 0 for k in _VALIDITY_FLAGS}
    any_count = 0
    err_count = 0
    for p in per_sample:
        if p.get("error"):
            err_count += 1
        for k in _VALIDITY_FLAGS:
            if p[f"has_{k}"] is True:
                counts[k] += 1
        if p["is_any_anomaly"]:
            any_count += 1

    aggregate = {
        "n_samples": N,
        "overlap_rate": counts["overlap"] / max(N, 1),
        "dissoc_rate": counts["dissoc"] / max(N, 1),
        "desorbed_rate": counts["desorbed"] / max(N, 1),
        "intercalated_rate": counts["intercalated"] / max(N, 1),
        "surf_changed_rate": counts["surf_changed"] / max(N, 1),
        "any_anomaly_rate": any_count / max(N, 1),
        "valid_rate_strict": (N - any_count) / max(N, 1),
        "n_errors": err_count,
        "overlap_min_distance_A": OVERLAP_MIN_DIST_A,
        "fairchem_surface_change_cutoff_mult": 1.5,
        "fairchem_desorption_cutoff_mult": 1.5,
        "protocol": {
            "overlap": "min MIC distance among all atom pairs < 0.5A",
            "dissoc": "DetectTrajAnomaly.is_adsorbate_dissociated (init=pos_ref, final=pos_pred)",
            "desorbed": "DetectTrajAnomaly.is_adsorbate_desorbed (final=pos_pred)",
            "intercalated": "DetectTrajAnomaly.is_adsorbate_intercalated (final=pos_pred)",
            "surf_changed": (
                "DetectTrajAnomaly.has_surface_changed (final_slab=pristine "
                "relaxed slab via sid/system_key index, fallback pos_gt[tag!=2] if missing)"
                if pristine_loaded is not None
                else "DetectTrajAnomaly.has_surface_changed (final_slab=pos_gt[tag!=2])"
            ),
        },
        "pristine_slab_reference": pristine_loaded is not None,
    }
    return {"aggregate": aggregate, "per_sample": per_sample}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1,
                   help="parallel workers for compute_anomaly_metrics")
    p.add_argument("--pristine-slabs", type=Path, default=None,
                   help="pristine relaxed-slab pkl (results/pristine_slabs/is2res.pkl). "
                        "When provided, has_surface_changed uses the pristine slab as "
                        "the relaxed-slab reference instead of pos_gt[tag!=2].")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=Path, default=None,
                   help="sid/system_key -> slab_key index pkl. Defaults to "
                        "<--pristine-slabs>.sid_index.pkl or "
                        "<--pristine-slabs>.system_index.pkl")
    args = p.parse_args()

    blob = torch.load(args.samples, weights_only=False)
    records = blob["records"]
    meta = blob.get("meta", {})

    disp = compute_displacement_metrics(records)
    strict = compute_anomaly_metrics(
        records, workers=args.workers,
        pristine_slabs=args.pristine_slabs,
        pristine_sid_index=args.pristine_sid_index,
    )
    out_blob = {
        "aggregate": {**disp["aggregate"], **strict["aggregate"]},
        "per_sample_displacement": disp["per_sample"],
        "per_sample_strict": strict["per_sample"],
        "meta": meta,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_blob, f, indent=2)
    agg = out_blob["aggregate"]
    print(
        f"[eval] n={agg['n_samples']} "
        f"mae={agg['displacement_mae_A']:.4f} "
        f"rmse={agg['displacement_rmse_A']:.4f} "
        f"valid_strict={agg['valid_rate_strict']:.3f} "
        f"any_anomaly={agg['any_anomaly_rate']:.3f}",
        flush=True,
    )
    print(f"[eval] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
