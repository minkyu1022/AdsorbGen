"""OC20-canonical anomaly audit of preprocessed LMDB ground truth.

Replaces the earlier ratio/threshold heuristics with the conventions used by
FAIR-chem / AdsorbML and DiffCSP:

* **Dissociation** — ``fairchem.data.oc.utils.DetectTrajAnomaly
  .is_adsorbate_dissociated``: compare the connectivity graph of the
  adsorbate-only subsystem between the initial (``pos``) and final
  (``pos_relaxed``) structure using ASE's ``natural_cutoffs(mult=1.0)``.
  Any change in bond topology is flagged.
* **Desorption** — ``is_adsorbate_desorbed``: full-system connectivity with
  ``mult=1.5`` (PBC-aware via ASE ``NeighborList``); flag if *no* adsorbate
  atom has a surface neighbor.
* **Surface change** — ``has_surface_changed``: detects reconstructed slabs.
* **Intercalation** — ``is_adsorbate_intercalated``: adsorbate atoms buried
  inside the slab.
* **Overlap** — hard 0.5 Å minimum pair distance on the MIC-aware distance
  matrix (``atoms.get_all_distances(mic=True)``). This is the convention used
  by DiffCSP/CDVAE/OMatG for structural validity of generated crystals.
* **Initial-already-broken** — compare the adsorbate-only connectivity of
  ``pos`` against the canonical reference from ``adsorbates.pkl`` (indexed by
  ``ads_id``). Catches samples whose *starting* structure is already
  fragmented.
* **OC20 official anomaly** — for IS2RES we join on ``f"random{sid}"`` into
  ``oc20_metadata.pkl`` and read the pre-computed ``anomaly`` integer.
  Per OC20 documentation:
      0 = no anomaly
      1 = adsorbate dissociation
      2 = adsorbate desorption
      3 = surface reconstruction
      4 = incorrect CHCOH placement (CHCO with a stray non-interacting H)
  Used as the gold label to cross-check our reproduced checks. Label 4 is
  CHCOH-specific and does not map cleanly onto any DetectTrajAnomaly method;
  we accept misses on it.

Output per LMDB includes aggregate rates, a confusion matrix against the
official label (when available), and stratification of ``our_any_anomaly``
rate by ``|delta_max|`` and ``|delta_e|`` bins.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from fairchem.data.oc.utils import DetectTrajAnomaly

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.flow import minimum_image  # noqa: E402


_OVERLAP_A = 0.5            # DiffCSP convention
_DELTA_BINS = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, float("inf")]
_ENERGY_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, float("inf")]


def _load_metadata(path: Path) -> Dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _build_reference_connectivity(adsorbates_pkl: Path) -> Dict[int, np.ndarray]:
    """Return ``{ads_id: connectivity_matrix}`` for all reference adsorbates.

    Uses the same ``NeighborList(natural_cutoffs(mult=1.0))`` construction as
    ``DetectTrajAnomaly._get_connectivity`` so the comparison is apples-to-
    apples when we later test ``init adsorbate`` connectivity against it.
    """
    with open(adsorbates_pkl, "rb") as f:
        data = pickle.load(f)
    out: Dict[int, np.ndarray] = {}
    for ads_id, (atoms, _symbol, _bind_idx, _rxn) in data.items():
        out[int(ads_id)] = _ase_connectivity(atoms, mult=1.0)
    return out


def _ase_connectivity(atoms: Atoms, mult: float = 1.0) -> np.ndarray:
    cutoff = natural_cutoffs(atoms, mult=mult)
    nl = NeighborList(cutoff, self_interaction=False, bothways=True)
    nl.update(atoms)
    from ase.neighborlist import get_connectivity_matrix
    return get_connectivity_matrix(nl.nl).toarray()


def _build_ase(pos: np.ndarray, z: np.ndarray, cell: np.ndarray, tags: np.ndarray) -> Atoms:
    a = Atoms(numbers=z, positions=pos, cell=cell, pbc=True)
    a.set_tags(tags)
    return a


def _min_mic_distance(atoms: Atoms) -> float:
    """Smallest off-diagonal entry of the MIC-aware distance matrix."""
    d = atoms.get_all_distances(mic=True)
    n = d.shape[0]
    if n < 2:
        return float("inf")
    m = d.copy()
    np.fill_diagonal(m, np.inf)
    return float(m.min())


def _init_broken(init_atoms: Atoms, tags: np.ndarray, ref_conn: Optional[np.ndarray]) -> Optional[bool]:
    """True iff initial adsorbate connectivity differs from reference.

    Returns None when no reference is available (unknown ads_id).
    """
    if ref_conn is None:
        return None
    adsorbate_idx = [i for i, t in enumerate(tags) if int(t) == 2]
    if len(adsorbate_idx) != ref_conn.shape[0]:
        # Atom count mismatch — reference doesn't correspond to this sample.
        return None
    sub = init_atoms[adsorbate_idx]
    init_conn = _ase_connectivity(sub, mult=1.0)
    return not np.array_equal(init_conn, ref_conn)


def _delta_max(pos: np.ndarray, pos_rel: np.ndarray, cell: np.ndarray, movable: np.ndarray) -> float:
    delta = torch.from_numpy(pos_rel - pos).to(torch.float32).unsqueeze(0)
    cell_t = torch.from_numpy(cell).to(torch.float32).unsqueeze(0)
    delta = minimum_image(delta, cell_t).squeeze(0).numpy()
    mov = movable.astype(bool)
    if not mov.any():
        return 0.0
    return float(np.linalg.norm(delta[mov], axis=-1).max())


def _bin_index(value: float, edges: List[float]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return len(edges) - 2


def _label(edges: List[float], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    hi_s = "inf" if hi == float("inf") else f"{hi:g}"
    return f"[{lo:g},{hi_s})"


def _empty_bin() -> Dict:
    return {"n": 0, "any_anom": 0, "overlap": 0, "dissoc": 0, "desorb": 0, "surf": 0, "inter": 0}


def _iter_lmdb(path: str, max_samples: Optional[int]):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(b"length")
        n = int(pickle.loads(raw)) if raw is not None else txn.stat()["entries"]
        if max_samples is not None:
            n = min(n, max_samples)
        for idx in range(n):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                continue
            yield idx, n, pickle.loads(raw)
    env.close()


_WORKER_STATE: Dict = {}


def _worker_init(oc20_md, oc20dense_md, ref_conn_by_ads_id, ads_symbol_to_id):
    torch.set_num_threads(1)
    _WORKER_STATE["oc20_md"] = oc20_md
    _WORKER_STATE["oc20dense_md"] = oc20dense_md
    _WORKER_STATE["ref"] = ref_conn_by_ads_id
    _WORKER_STATE["sym"] = ads_symbol_to_id


def _process_sample(args: Tuple[int, dict]) -> dict:
    idx, entry = args
    oc20_md = _WORKER_STATE.get("oc20_md")
    oc20dense_md = _WORKER_STATE.get("oc20dense_md")
    ref_conn_by_ads_id = _WORKER_STATE.get("ref", {})
    ads_symbol_to_id = _WORKER_STATE.get("sym", {})

    pos = np.asarray(entry["pos"], dtype=np.float64)
    pos_rel = np.asarray(entry["pos_relaxed"], dtype=np.float64)
    cell = np.asarray(entry["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell.squeeze(0)
    tags = np.asarray(entry["tags"], dtype=np.int64)
    fixed = np.asarray(entry["fixed"], dtype=np.int64)
    z = np.asarray(entry["atomic_numbers"], dtype=np.int64)
    movable = ((tags == 1) | (tags == 2)) & (fixed == 0)

    init_atoms = _build_ase(pos, z, cell, tags)
    final_atoms = _build_ase(pos_rel, z, cell, tags)

    detector = DetectTrajAnomaly(init_atoms, final_atoms, atoms_tag=tags.tolist())
    is_diss = bool(detector.is_adsorbate_dissociated())
    is_deso = bool(detector.is_adsorbate_desorbed())
    is_surf = bool(detector.has_surface_changed())
    is_inter = bool(detector.is_adsorbate_intercalated())

    min_d = _min_mic_distance(final_atoms)
    is_over = min_d < _OVERLAP_A

    sid = entry.get("sid", None)
    off_label: Optional[int] = None
    ads_id: Optional[int] = None
    if oc20_md is not None and sid is not None and int(sid) >= 0:
        md = oc20_md.get(f"random{int(sid)}")
        if md is not None:
            off_label = int(md.get("anomaly", 0))
            ads_id = int(md["ads_id"])
    if ads_id is None and oc20dense_md is not None:
        key = entry.get("system_key")
        cfg = entry.get("config_key")
        if key is not None and cfg is not None:
            md = oc20dense_md.get((str(key), str(cfg)))
            if md is not None and "adsorbate" in md:
                ads_id = ads_symbol_to_id.get(md["adsorbate"])

    ref_conn = ref_conn_by_ads_id.get(ads_id) if ads_id is not None else None
    init_br = _init_broken(init_atoms, tags, ref_conn)

    delta_e = float(abs(entry.get("delta_e", 0.0)))
    delta_max = _delta_max(pos, pos_rel, cell, movable)

    return {
        "idx": idx,
        "sid": None if sid is None else int(sid),
        "system_key": entry.get("system_key"),
        "is_diss": is_diss,
        "is_deso": is_deso,
        "is_surf": is_surf,
        "is_inter": is_inter,
        "is_over": is_over,
        "min_mic_dist": min_d,
        "init_br": init_br,
        "off_label": off_label,
        "delta_max": delta_max,
        "delta_e": delta_e,
    }


def _iter_entries(path: str, max_samples: Optional[int]):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(b"length")
        n = int(pickle.loads(raw)) if raw is not None else txn.stat()["entries"]
        if max_samples is not None:
            n = min(n, max_samples)
        for idx in range(n):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                continue
            yield idx, pickle.loads(raw)
    env.close()


def run(
    path: str,
    max_samples: Optional[int],
    log_every: int,
    oc20_md: Optional[Dict],
    oc20dense_md: Optional[Dict],
    ref_conn_by_ads_id: Dict[int, np.ndarray],
    ads_symbol_to_id: Dict[str, int],
    num_workers: int = 1,
) -> Dict:
    torch.set_num_threads(1)

    n_total = 0
    dissoc_n = 0
    desorb_n = 0
    surf_n = 0
    inter_n = 0
    overlap_n = 0
    init_broken_n = 0
    init_broken_known = 0
    any_anom_n = 0
    official_n = 0                 # samples with OC20 official label available
    official_anom_n = 0
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_anom_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    d_bins = [_empty_bin() for _ in range(len(_DELTA_BINS) - 1)]
    e_bins = [_empty_bin() for _ in range(len(_ENERGY_BINS) - 1)]
    deltas: List[float] = []
    energies: List[float] = []
    first_fail: List[Dict] = []

    t0 = time.time()

    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(b"length")
        total = int(pickle.loads(raw)) if raw is not None else txn.stat()["entries"]
        if max_samples is not None:
            total = min(total, max_samples)
    env.close()

    if num_workers > 1:
        pool = mp.get_context("fork").Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(oc20_md, oc20dense_md, ref_conn_by_ads_id, ads_symbol_to_id),
        )
        result_iter = pool.imap_unordered(
            _process_sample, _iter_entries(path, max_samples), chunksize=32
        )
    else:
        _worker_init(oc20_md, oc20dense_md, ref_conn_by_ads_id, ads_symbol_to_id)
        pool = None
        result_iter = (_process_sample(args) for args in _iter_entries(path, max_samples))

    for r in result_iter:
        is_diss = r["is_diss"]
        is_deso = r["is_deso"]
        is_surf = r["is_surf"]
        is_inter = r["is_inter"]
        is_over = r["is_over"]
        min_d = r["min_mic_dist"]
        init_br = r["init_br"]
        off_label = r["off_label"]
        delta_max = r["delta_max"]
        delta_e = r["delta_e"]

        is_any = is_diss or is_deso or is_surf or is_inter or is_over

        n_total += 1
        dissoc_n += int(is_diss)
        desorb_n += int(is_deso)
        surf_n += int(is_surf)
        inter_n += int(is_inter)
        overlap_n += int(is_over)
        any_anom_n += int(is_any)
        if init_br is not None:
            init_broken_known += 1
            init_broken_n += int(init_br)
        if off_label is not None:
            official_n += 1
            per_anom_type[off_label] = per_anom_type.get(off_label, 0) + 1
            if off_label != 0:
                official_anom_n += 1
            off_is_anom = off_label != 0
            if off_is_anom and is_any:
                confusion["tp"] += 1
            elif off_is_anom and not is_any:
                confusion["fn"] += 1
            elif (not off_is_anom) and is_any:
                confusion["fp"] += 1
            else:
                confusion["tn"] += 1
        deltas.append(delta_max)
        energies.append(delta_e)

        di = _bin_index(delta_max, _DELTA_BINS)
        ei = _bin_index(delta_e, _ENERGY_BINS)
        for bucket, bidx in ((d_bins, di), (e_bins, ei)):
            b = bucket[bidx]
            b["n"] += 1
            b["any_anom"] += int(is_any)
            b["overlap"] += int(is_over)
            b["dissoc"] += int(is_diss)
            b["desorb"] += int(is_deso)
            b["surf"] += int(is_surf)
            b["inter"] += int(is_inter)

        if is_any and len(first_fail) < 20:
            first_fail.append({
                "idx": r["idx"],
                "sid": r["sid"],
                "system_key": r["system_key"],
                "dissoc": is_diss,
                "desorb": is_deso,
                "surf": is_surf,
                "inter": is_inter,
                "overlap": is_over,
                "min_mic_dist": min_d,
                "init_broken": init_br,
                "official_label": off_label,
                "delta_max": delta_max,
                "delta_e": delta_e,
            })

        if log_every and n_total % log_every == 0:
            elapsed = time.time() - t0
            rate = n_total / max(elapsed, 1e-6)
            eta = (total - n_total) / max(rate, 1e-6)
            print(
                f"[probe] {path} {n_total}/{total} "
                f"any={any_anom_n / n_total:.4f} "
                f"diss={dissoc_n / n_total:.4f} "
                f"deso={desorb_n / n_total:.4f} "
                f"surf={surf_n / n_total:.4f} "
                f"inter={inter_n / n_total:.4f} "
                f"over={overlap_n / n_total:.4f} "
                f"rate={rate:.0f}/s eta={eta:.0f}s",
                flush=True,
            )

    if pool is not None:
        pool.close()
        pool.join()

    def _rates(bins, edges):
        return [
            {
                "bin": _label(edges, i),
                "n": b["n"],
                "any_anom_rate": (b["any_anom"] / b["n"]) if b["n"] else 0.0,
                "dissoc_rate": (b["dissoc"] / b["n"]) if b["n"] else 0.0,
                "desorb_rate": (b["desorb"] / b["n"]) if b["n"] else 0.0,
                "surf_rate": (b["surf"] / b["n"]) if b["n"] else 0.0,
                "inter_rate": (b["inter"] / b["n"]) if b["n"] else 0.0,
                "overlap_rate": (b["overlap"] / b["n"]) if b["n"] else 0.0,
            }
            for i, b in enumerate(bins)
        ]

    deltas_arr = np.asarray(deltas, dtype=np.float64)
    energies_arr = np.asarray(energies, dtype=np.float64)

    return {
        "lmdb": path,
        "n_samples": n_total,
        "elapsed_sec": time.time() - t0,
        "aggregate": {
            "dissoc_rate": dissoc_n / max(n_total, 1),
            "desorb_rate": desorb_n / max(n_total, 1),
            "surf_rate": surf_n / max(n_total, 1),
            "inter_rate": inter_n / max(n_total, 1),
            "overlap_rate": overlap_n / max(n_total, 1),
            "any_anom_rate": any_anom_n / max(n_total, 1),
            "init_broken_rate": (init_broken_n / init_broken_known) if init_broken_known else None,
            "init_broken_n": init_broken_n,
            "init_broken_known": init_broken_known,
            "overlap_A_threshold": _OVERLAP_A,
        },
        "official_label": {
            "n_with_label": official_n,
            "anomaly_rate_official": (official_anom_n / official_n) if official_n else None,
            "per_type_counts": per_anom_type,
            "confusion_vs_ours": confusion,
        },
        "delta_max_stats_A": {
            "p50": float(np.percentile(deltas_arr, 50)) if deltas_arr.size else 0.0,
            "p90": float(np.percentile(deltas_arr, 90)) if deltas_arr.size else 0.0,
            "p99": float(np.percentile(deltas_arr, 99)) if deltas_arr.size else 0.0,
            "p99.9": float(np.percentile(deltas_arr, 99.9)) if deltas_arr.size else 0.0,
            "max": float(deltas_arr.max()) if deltas_arr.size else 0.0,
        },
        "delta_e_stats_eV": {
            "p50": float(np.percentile(energies_arr, 50)) if energies_arr.size else 0.0,
            "p90": float(np.percentile(energies_arr, 90)) if energies_arr.size else 0.0,
            "p99": float(np.percentile(energies_arr, 99)) if energies_arr.size else 0.0,
            "p99.9": float(np.percentile(energies_arr, 99.9)) if energies_arr.size else 0.0,
            "max": float(energies_arr.max()) if energies_arr.size else 0.0,
        },
        "by_delta_max": _rates(d_bins, _DELTA_BINS),
        "by_delta_e": _rates(e_bins, _ENERGY_BINS),
        "failing_samples_head": first_fail,
    }


def _build_oc20dense_index(mapping_path: Path) -> Dict:
    with open(mapping_path, "rb") as f:
        m = pickle.load(f)
    out: Dict = {}
    for v in m.values():
        sys_id = str(v.get("system_id"))
        cfg_id = str(v.get("config_id"))
        if sys_id is None or cfg_id is None:
            continue
        out[(sys_id, cfg_id)] = v
    return out


def _build_ads_symbol_map(adsorbates_pkl: Path) -> Dict[str, int]:
    with open(adsorbates_pkl, "rb") as f:
        data = pickle.load(f)
    out: Dict[str, int] = {}
    for ads_id, (_atoms, symbol, _bind_idx, _rxn) in data.items():
        out[str(symbol)] = int(ads_id)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", action="append", required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10000)
    ap.add_argument("--oc20-metadata", type=str,
                    default="data/pkls/oc20_metadata.pkl")
    ap.add_argument("--oc20dense-mapping", type=str,
                    default="data/oc20dense/oc20dense_mapping.pkl")
    ap.add_argument("--adsorbates", type=str,
                    default="data/pkls/adsorbates.pkl")
    ap.add_argument("--num-workers", type=int, default=1)
    args = ap.parse_args()

    print(f"[probe] loading references ...", flush=True)
    oc20_md = _load_metadata(Path(args.oc20_metadata)) if Path(args.oc20_metadata).exists() else None
    oc20dense_md = _build_oc20dense_index(Path(args.oc20dense_mapping)) if Path(args.oc20dense_mapping).exists() else None
    ref_conn_by_ads_id = _build_reference_connectivity(Path(args.adsorbates))
    ads_symbol_to_id = _build_ads_symbol_map(Path(args.adsorbates))
    print(f"[probe] oc20_md entries: {len(oc20_md) if oc20_md else 0}", flush=True)
    print(f"[probe] oc20dense idx:   {len(oc20dense_md) if oc20dense_md else 0}", flush=True)
    print(f"[probe] ads references:  {len(ref_conn_by_ads_id)}", flush=True)

    reports = []
    for path in args.lmdb:
        print(f"[probe] === start {path} ===", flush=True)
        rep = run(
            path,
            args.max_samples,
            args.log_every,
            oc20_md,
            oc20dense_md,
            ref_conn_by_ads_id,
            ads_symbol_to_id,
            num_workers=args.num_workers,
        )
        reports.append(rep)
        a = rep["aggregate"]
        print(
            f"[probe] === done {path}: n={rep['n_samples']} "
            f"any={a['any_anom_rate']:.4f} "
            f"diss={a['dissoc_rate']:.4f} "
            f"deso={a['desorb_rate']:.4f} "
            f"surf={a['surf_rate']:.4f} "
            f"inter={a['inter_rate']:.4f} "
            f"over={a['overlap_rate']:.4f} "
            f"time={rep['elapsed_sec']:.1f}s ===",
            flush=True,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"reports": reports}, f, indent=2)

    print(f"[probe] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
