#!/usr/bin/env python
"""Build a one-best-per-system LMDB with self-improvement replacements.

For each eligible train/val-ID system, keep exactly one configuration:
the lowest-energy original configuration by ``gt_index_by_sid_oc20`` unless a
self-improvement success exists for that system, in which case the lowest
success relaxed structure replaces the target ``pos_relaxed``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import lmdb
import numpy as np


def frozen_key(x):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def lmdb_length(txn) -> int:
    raw = txn.get(b"length")
    return int(pickle.loads(raw)) if raw is not None else int(txn.stat()["entries"])


def read_entry(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(f"missing LMDB row {idx}")
    return pickle.loads(raw)


def load_best_successes(replay_dir: Path) -> dict:
    best = {}
    for path in sorted(replay_dir.glob("success_shard*.pkl")):
        with path.open("rb") as f:
            rows = pickle.load(f)
        for row in rows:
            sk = frozen_key(row["system_key"])
            cur = best.get(sk)
            if cur is None or float(row["E_sys"]) < float(cur["E_sys"]):
                best[sk] = row
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-lmdb", nargs="+", required=True)
    p.add_argument("--gt-index", default="/home/irteam/data/replay/gt_index_by_sid_oc20.pkl")
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--out-lmdb", required=True)
    p.add_argument("--map-size-gb", type=float, default=16.0)
    args = p.parse_args()

    out_lmdb = Path(args.out_lmdb)
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    for pth in [out_lmdb, Path(str(out_lmdb) + "-lock")]:
        if pth.exists():
            pth.unlink()

    with open(args.gt_index, "rb") as f:
        gt_index = pickle.load(f)

    src_envs = [
        lmdb.open(p, subdir=False, readonly=True, lock=False, readahead=False)
        for p in args.train_lmdb
    ]

    # Lowest original row per system using per-row E_sys_own.
    best_original = {}
    scan_counts = []
    for lid, (path, env) in enumerate(zip(args.train_lmdb, src_envs)):
        kept = 0
        with env.begin() as txn:
            n = lmdb_length(txn)
            mask_raw = txn.get(b"anomaly_mask")
            mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n] if mask_raw else None
            for raw_idx in range(n):
                if mask is not None and int(mask[raw_idx]) != 0:
                    continue
                raw = txn.get(str(raw_idx).encode("ascii"))
                if raw is None:
                    continue
                entry = pickle.loads(raw)
                sid = int(entry.get("sid", -1))
                gi = gt_index.get(sid)
                if not (
                    isinstance(gi, dict)
                    and gi.get("eligible")
                    and gi.get("system_key") is not None
                    and gi.get("E_sys_own") is not None
                ):
                    continue
                sk = frozen_key(gi["system_key"])
                e = float(gi["E_sys_own"])
                cur = best_original.get(sk)
                if cur is None or e < float(cur["E_sys_own"]):
                    best_original[sk] = {
                        "lmdb_id": lid,
                        "raw_idx": int(raw_idx),
                        "sid": sid,
                        "system_key": gi["system_key"],
                        "E_sys_own": e,
                        "E_sys_min": gi.get("E_sys_min"),
                    }
                kept += 1
        scan_counts.append({"path": path, "eligible_rows": kept})

    best_success = load_best_successes(Path(args.replay_dir))

    out_env = lmdb.open(
        str(out_lmdb),
        subdir=False,
        map_size=int(args.map_size_gb * (1024 ** 3)),
        meminit=False,
    )
    anomaly_mask = np.zeros(len(best_original), dtype=np.int8)
    n_replaced = 0
    replacement_manifest = []
    with out_env.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(len(best_original)))
        for out_i, sk in enumerate(sorted(best_original, key=str)):
            rec = best_original[sk]
            repl = best_success.get(sk)
            if repl is not None:
                src_env = src_envs[int(repl["lmdb_id"])]
                entry = dict(read_entry(src_env, int(repl["raw_idx"])))
                entry["pos_relaxed"] = np.asarray(repl["pos_relaxed"], dtype=np.float32)
                entry["y_relaxed"] = float(repl["E_sys"])
                entry["mlip_e_total"] = float(repl["E_sys"])
                entry["mlip_fmax"] = float(repl["fmax"])
                entry["mlip_converged"] = bool(repl["converged"])
                entry["self_improve_replaced"] = True
                entry["self_improve_E_sys_ref"] = float(repl["E_sys_ref"])
                entry["self_improve_E_sys"] = float(repl["E_sys"])
                entry["self_improve_improvement"] = float(repl["improvement"])
                entry["self_improve_source_raw_idx"] = int(repl["raw_idx"])
                entry["self_improve_source_lmdb_id"] = int(repl["lmdb_id"])
                n_replaced += 1
                replacement_manifest.append({
                    "out_idx": out_i,
                    "system_key": repl["system_key"],
                    "sid": int(repl["sid"]),
                    "lmdb_id": int(repl["lmdb_id"]),
                    "raw_idx": int(repl["raw_idx"]),
                    "E_sys_ref": float(repl["E_sys_ref"]),
                    "E_sys": float(repl["E_sys"]),
                    "improvement": float(repl["improvement"]),
                })
            else:
                src_env = src_envs[int(rec["lmdb_id"])]
                entry = dict(read_entry(src_env, int(rec["raw_idx"])))
                entry["self_improve_replaced"] = False
                entry["self_improve_E_sys_ref"] = float(rec["E_sys_own"])
            txn.put(str(out_i).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(b"anomaly_mask", pickle.dumps(anomaly_mask, protocol=pickle.HIGHEST_PROTOCOL))
    out_env.sync()
    out_env.close()
    for env in src_envs:
        env.close()

    summary = {
        "out_lmdb": str(out_lmdb),
        "n_systems": len(best_original),
        "n_replaced_by_self_improve": n_replaced,
        "source_train_lmdb": args.train_lmdb,
        "replay_dir": str(args.replay_dir),
        "scan_counts": scan_counts,
    }
    out_lmdb.with_suffix(".report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out_lmdb.with_suffix(".replacements.json").write_text(
        json.dumps(replacement_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
