"""Inspect the 'ours-only' anomaly samples (fp against OC20 official label).

Re-scans IS2RES val and reports:
  - per-type confusion: for each official label in {0,1,2,3,4}, how many we flag vs miss
  - which of our 5 checks (diss/deso/surf/inter/over) fires on each fp / tp
  - delta_max distribution for fp vs tp vs tn
  - dumps N random fp samples as ASE .traj files (init + final) so we can
    visualize whether they are actually broken structures or just over-strict
    heuristic firings
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.io import write as ase_write
from fairchem.data.oc.utils import DetectTrajAnomaly


_OVERLAP_A = 0.5


def _build_ase(pos, z, cell, tags):
    a = Atoms(numbers=z, positions=pos, cell=cell, pbc=True)
    a.set_tags(tags)
    return a


def _min_mic_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    if d.shape[0] < 2:
        return float("inf")
    m = d.copy()
    np.fill_diagonal(m, np.inf)
    return float(m.min())


_WORKER_MD = {}


def _worker_init(md):
    torch.set_num_threads(1)
    _WORKER_MD["md"] = md


def _process(args):
    idx, entry = args
    md = _WORKER_MD["md"]
    pos = np.asarray(entry["pos"], dtype=np.float64)
    pos_rel = np.asarray(entry["pos_relaxed"], dtype=np.float64)
    cell = np.asarray(entry["cell"], dtype=np.float64)
    if cell.ndim == 3:
        cell = cell.squeeze(0)
    tags = np.asarray(entry["tags"], dtype=np.int64)
    z = np.asarray(entry["atomic_numbers"], dtype=np.int64)
    fixed = np.asarray(entry["fixed"], dtype=np.int64)
    movable = ((tags == 1) | (tags == 2)) & (fixed == 0)

    init_atoms = _build_ase(pos, z, cell, tags)
    final_atoms = _build_ase(pos_rel, z, cell, tags)
    det = DetectTrajAnomaly(init_atoms, final_atoms, atoms_tag=tags.tolist())
    flags = {
        "diss": bool(det.is_adsorbate_dissociated()),
        "deso": bool(det.is_adsorbate_desorbed()),
        "surf": bool(det.has_surface_changed()),
        "inter": bool(det.is_adsorbate_intercalated()),
        "over": _min_mic_distance(final_atoms) < _OVERLAP_A,
    }
    sid = int(entry.get("sid", -1))
    label = int(md.get(f"random{sid}", {}).get("anomaly", 0)) if sid >= 0 else 0
    diff = pos_rel - pos
    delta_max = float(np.linalg.norm(diff[movable], axis=-1).max()) if movable.any() else 0.0
    return {
        "idx": idx, "sid": sid, "label": label,
        "flags": flags, "delta_max": delta_max,
        "entry": entry,
    }


def _iter_entries(path, n):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        for idx in range(n):
            raw = txn.get(str(idx).encode("ascii"))
            if raw is None:
                continue
            yield idx, pickle.loads(raw)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="data/processed/is2res_val.lmdb")
    ap.add_argument("--metadata", default="data/pkls/oc20_metadata.pkl")
    ap.add_argument("--out", default="runs/probe_v3/fp_inspect")
    ap.add_argument("--n-dump", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print("[inspect] loading oc20_metadata ...", flush=True)
    with open(args.metadata, "rb") as f:
        md = pickle.load(f)

    env = lmdb.open(args.lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        raw = txn.get(b"length")
        n = int(pickle.loads(raw)) if raw is not None else txn.stat()["entries"]
    if args.max_samples:
        n = min(n, args.max_samples)

    # Per-official-label tally: {label: {"flagged": k, "not_flagged": k}}
    per_label = defaultdict(lambda: {"flagged": 0, "not_flagged": 0})
    # Which checks fire for each category
    fp_check_cnt = Counter()
    tp_check_cnt = Counter()
    fn_check_cnt = Counter()

    fp_records = []  # (idx, sid, flags, delta_max, label, entry)
    fn_records = []
    delta_buckets = {"fp": [], "tp": [], "tn": [], "fn": []}

    print(f"[inspect] scanning {n} samples with {args.num_workers} workers ...", flush=True)
    pool = mp.get_context("fork").Pool(
        processes=args.num_workers, initializer=_worker_init, initargs=(md,)
    )
    seen = 0
    for r in pool.imap_unordered(_process, _iter_entries(args.lmdb, n), chunksize=32):
        seen += 1
        label = r["label"]
        flags = r["flags"]
        is_any = any(flags.values())
        delta_max = r["delta_max"]
        off_is_anom = label != 0
        fired = tuple(k for k, v in flags.items() if v)

        if off_is_anom:
            per_label[label]["flagged" if is_any else "not_flagged"] += 1
            if is_any:
                tp_check_cnt[fired] += 1
                delta_buckets["tp"].append(delta_max)
            else:
                fn_check_cnt[(label,)] += 1
                delta_buckets["fn"].append(delta_max)
                fn_records.append({
                    "idx": r["idx"], "sid": r["sid"], "label": label,
                    "flags": flags, "delta_max": delta_max,
                })
        else:
            if is_any:
                fp_check_cnt[fired] += 1
                delta_buckets["fp"].append(delta_max)
                fp_records.append({
                    "idx": r["idx"], "sid": r["sid"], "label": 0,
                    "flags": flags, "delta_max": delta_max,
                    "entry": r["entry"],
                })
            else:
                delta_buckets["tn"].append(delta_max)
        if seen % 5000 == 0:
            print(f"[inspect] {seen}/{n}", flush=True)
    pool.close()
    pool.join()
    env.close()

    print()
    print("=== per-official-label breakdown (recall) ===")
    names = {0: "clean", 1: "dissoc", 2: "desorb", 3: "surf-recon", 4: "CHCOH"}
    for lbl in sorted(per_label):
        p = per_label[lbl]
        tot = p["flagged"] + p["not_flagged"]
        rec = p["flagged"] / tot if tot else 0.0
        print(f"  label {lbl} ({names[lbl]:<10}): total={tot:>6} flagged={p['flagged']:>6} missed={p['not_flagged']:>6} recall={rec:.3f}")

    print()
    print("=== fp 'which check fired' top-10 ===")
    for fired, cnt in fp_check_cnt.most_common(10):
        print(f"  {'+'.join(fired):<25} {cnt}")

    print()
    print("=== tp 'which check fired' top-10 ===")
    for fired, cnt in tp_check_cnt.most_common(10):
        print(f"  {'+'.join(fired):<25} {cnt}")

    print()
    print("=== delta_max distributions (p50/p90/p99/max) ===")
    for k in ("tn", "fp", "tp", "fn"):
        arr = np.asarray(delta_buckets[k])
        if arr.size == 0:
            print(f"  {k}: empty")
            continue
        print(f"  {k:<4} n={arr.size:>6} p50={np.percentile(arr, 50):.2f} p90={np.percentile(arr, 90):.2f} p99={np.percentile(arr, 99):.2f} max={arr.max():.2f}")

    # Dump N random fp samples as xyz for visual inspection
    print()
    print(f"=== dumping {args.n_dump} random fp samples to {out_dir} ===")
    n_dump = min(args.n_dump, len(fp_records))
    picked = rng.sample(fp_records, n_dump)
    for r in picked:
        entry = r["entry"]
        pos = np.asarray(entry["pos"], dtype=np.float64)
        pos_rel = np.asarray(entry["pos_relaxed"], dtype=np.float64)
        cell = np.asarray(entry["cell"], dtype=np.float64)
        if cell.ndim == 3:
            cell = cell.squeeze(0)
        tags = np.asarray(entry["tags"], dtype=np.int64)
        z = np.asarray(entry["atomic_numbers"], dtype=np.int64)
        init = _build_ase(pos, z, cell, tags)
        final = _build_ase(pos_rel, z, cell, tags)
        fired = "_".join(k for k, v in r["flags"].items() if v) or "none"
        base = out_dir / f"fp_sid{r['sid']}_idx{r['idx']}_{fired}_dmax{r['delta_max']:.1f}"
        ase_write(str(base) + "_init.xyz", init)
        ase_write(str(base) + "_final.xyz", final)
    print(f"[inspect] wrote {n_dump*2} files to {out_dir}")


if __name__ == "__main__":
    main()
