#!/usr/bin/env python
"""Create fixed small train LMDBs for Goal-1 overfit diagnostics."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import lmdb
import numpy as np


def lmdb_len(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def read_entry(env: lmdb.Environment, idx: int) -> dict[str, Any]:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def clean_mask(env: lmdb.Environment, n: int) -> np.ndarray:
    with env.begin() as txn:
        raw = txn.get(b"anomaly_mask")
    if raw is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(pickle.loads(raw), dtype=np.int8)[:n]
    return mask == 0


def system_key(entry: dict[str, Any], lmdb_id: int, raw_idx: int) -> str:
    if "system_key" in entry:
        return str(entry["system_key"])
    sid = int(entry.get("sid", -1))
    return f"sid:{sid}" if sid >= 0 else f"lmdb{lmdb_id}:idx:{raw_idx}"


def build_selection(paths: list[str], need: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lmdb_id, path in enumerate(paths):
        env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
        n = lmdb_len(env)
        cmask = clean_mask(env, n)
        order = rng.permutation(np.where(cmask)[0])
        for raw_idx in order.tolist():
            entry = read_entry(env, int(raw_idx))
            key = system_key(entry, lmdb_id, int(raw_idx))
            if key in seen:
                continue
            if "pos_relaxed" not in entry or "mlip_e_total" not in entry:
                continue
            seen.add(key)
            candidates.append(
                {
                    "lmdb_id": int(lmdb_id),
                    "lmdb": str(path),
                    "raw_idx": int(raw_idx),
                    "sid": int(entry.get("sid", -1)),
                    "ads_id": int(entry.get("ads_id", -1)),
                    "system_key": key,
                    "n_atoms": int(np.asarray(entry["atomic_numbers"]).shape[0]),
                    "mlip_e_total": float(entry["mlip_e_total"]),
                    "mlip_fmax": float(entry.get("mlip_fmax", np.nan)),
                }
            )
            if len(candidates) >= need:
                env.close()
                return candidates
        env.close()
    raise RuntimeError(f"selected only {len(candidates)} unique clean systems, need {need}")


def write_subset(rows: list[dict[str, Any]], out_lmdb: Path) -> None:
    if out_lmdb.exists():
        if out_lmdb.is_dir():
            shutil.rmtree(out_lmdb)
        else:
            out_lmdb.unlink()
    lock_path = Path(str(out_lmdb) + "-lock")
    if lock_path.exists():
        lock_path.unlink()
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    src_envs: dict[str, lmdb.Environment] = {}
    try:
        map_size = max(1 << 30, len(rows) * 2_000_000)
        env_out = lmdb.open(str(out_lmdb), subdir=False, map_size=map_size)
        with env_out.begin(write=True) as txn_out:
            txn_out.put(b"length", pickle.dumps(len(rows), protocol=pickle.HIGHEST_PROTOCOL))
            txn_out.put(
                b"anomaly_mask",
                pickle.dumps(np.zeros(len(rows), dtype=np.int8), protocol=pickle.HIGHEST_PROTOCOL),
            )
            for out_i, row in enumerate(rows):
                path = str(row["lmdb"])
                if path not in src_envs:
                    src_envs[path] = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
                entry = read_entry(src_envs[path], int(row["raw_idx"]))
                entry = dict(entry)
                entry["goal1_source_lmdb"] = path
                entry["goal1_source_raw_idx"] = int(row["raw_idx"])
                entry["goal1_system_key"] = str(row["system_key"])
                txn_out.put(
                    str(out_i).encode("ascii"),
                    pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL),
                )
        env_out.sync()
        env_out.close()
    finally:
        for env in src_envs.values():
            env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-lmdb", nargs="+", default=[
        "/home1/irteam/data/processed_ID/is2res_train.lmdb",
        "/home1/irteam/data/processed_ID/is2res_val.lmdb",
    ])
    ap.add_argument("--out-dir", default="/home1/irteam/data/replay/goal1_overfit_subsets_20260615")
    ap.add_argument("--sizes", default="10,100,1000")
    ap.add_argument("--seed", type=int, default=20260615)
    args = ap.parse_args()

    sizes = sorted({int(x) for x in str(args.sizes).split(",") if x.strip()})
    if not sizes:
        raise ValueError("no sizes requested")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_selection([str(p) for p in args.train_lmdb], max(sizes), int(args.seed))
    manifest = {
        "seed": int(args.seed),
        "train_lmdb": [str(p) for p in args.train_lmdb],
        "sizes": sizes,
        "rows": rows,
        "subsets": {},
    }
    for size in sizes:
        subdir = out_dir / f"n{size}"
        lmdb_path = subdir / "train.lmdb"
        write_subset(rows[:size], lmdb_path)
        payload = {
            "size": int(size),
            "lmdb": str(lmdb_path),
            "rows": rows[:size],
        }
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
        manifest["subsets"][str(size)] = str(lmdb_path)

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
