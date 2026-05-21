#!/usr/bin/env python
"""Build derived training LMDBs whose x1 target is the UMA-relaxed structure.

Original LMDBs are never modified. Each output LMDB copies the original entries
and replaces ``pos_relaxed`` from ``E_sys.pkl`` by sid.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
from tqdm.auto import tqdm


def _copy_special_keys(src_txn, dst_txn) -> None:
    for key in (b"length", b"anomaly_mask"):
        value = src_txn.get(key)
        if value is not None:
            dst_txn.put(key, value)


def convert_one(src_path: Path, dst_path: Path, e_sys: dict, only_converged: bool) -> dict:
    src_env = lmdb.open(str(src_path), subdir=False, readonly=True, lock=False, readahead=False)
    with src_env.begin() as src_txn:
        n = int(pickle.loads(src_txn.get(b"length")))
    map_size = max(src_path.stat().st_size * 3, 1 << 30)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    dst_env = lmdb.open(str(dst_path), subdir=False, map_size=map_size)

    replaced = 0
    kept_missing = 0
    kept_unconverged = 0
    kept_shape_mismatch = 0

    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        _copy_special_keys(src_txn, dst_txn)
        for i in tqdm(range(n), desc=src_path.name, unit="row", dynamic_ncols=True):
            raw = src_txn.get(str(i).encode("ascii"))
            if raw is None:
                continue
            entry = pickle.loads(raw)
            sid = int(entry.get("sid", -1))
            rec = e_sys.get(sid)
            if rec is None:
                kept_missing += 1
            elif only_converged and not bool(rec.get("converged", False)):
                kept_unconverged += 1
            else:
                pos = np.asarray(rec.get("pos_relaxed"), dtype=np.float32)
                if pos.shape != np.asarray(entry["pos_relaxed"]).shape:
                    kept_shape_mismatch += 1
                else:
                    entry["pos_relaxed"] = pos
                    entry["mlip_e_total"] = float(rec.get("e_total", np.nan))
                    entry["mlip_fmax"] = float(rec.get("fmax", np.nan))
                    entry["mlip_converged"] = bool(rec.get("converged", False))
                    entry["mlip_relaxed_source"] = "uma-s-1p1/oc20/E_sys.pkl"
                    replaced += 1
            dst_txn.put(str(i).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))

    src_env.close()
    dst_env.sync()
    dst_env.close()
    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "rows": n,
        "replaced": replaced,
        "kept_missing": kept_missing,
        "kept_unconverged": kept_unconverged,
        "kept_shape_mismatch": kept_shape_mismatch,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--e-sys", default="/home/irteam/data/replay/E_sys.pkl")
    p.add_argument("--out-dir", default="/home/irteam/data/processed_mlip_oc20")
    p.add_argument("--only-converged", action="store_true",
                   help="replace pos_relaxed only for converged E_sys records")
    p.add_argument("--lmdbs", nargs="+", default=[
        "/home/irteam/data/processed/is2res_train.lmdb",
        "/home/irteam/data/processed/is2res_val.lmdb",
        "/home/irteam/data/processed/is2res_val_ood_ads.lmdb",
        "/home/irteam/data/processed/is2res_val_ood_cat.lmdb",
        "/home/irteam/data/processed/is2res_val_ood_both.lmdb",
    ])
    args = p.parse_args()

    with open(args.e_sys, "rb") as f:
        e_sys = pickle.load(f)

    out_dir = Path(args.out_dir)
    summaries = []
    for src in args.lmdbs:
        src_path = Path(src)
        dst_path = out_dir / src_path.name
        summaries.append(convert_one(src_path, dst_path, e_sys, args.only_converged))

    summary_path = out_dir / "build_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(summaries, f)
    print("Wrote derived LMDBs:")
    for s in summaries:
        print(s)
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
