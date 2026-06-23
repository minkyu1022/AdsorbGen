#!/usr/bin/env python
"""Materialize OC20-dense LMDB and per-system minima from UMA relabel shards."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
from tqdm.auto import tqdm


def _open_lmdb(path: str, readonly: bool, map_size: int | None = None) -> lmdb.Environment:
    kwargs = {"subdir": False}
    if readonly:
        kwargs.update({"readonly": True, "lock": False, "readahead": False})
    else:
        kwargs.update({"map_size": int(map_size or (1 << 40))})
    try:
        return lmdb.open(path, **kwargs)
    except lmdb.Error:
        kwargs.pop("subdir", None)
        return lmdb.open(path, **kwargs)


def _read_length(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def _load_shards(shard_dir: Path, num_shards: int, require_all: bool) -> dict[int, dict]:
    records: dict[int, dict] = {}
    missing = []
    for shard in range(num_shards):
        path = shard_dir / f"oc20dense_lmdb_shard{shard}.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with path.open("rb") as f:
            part = pickle.load(f)
        for key, rec in part.items():
            records[int(key)] = rec
        print(f"loaded shard {shard}: {len(part)}")
    if missing and require_all:
        raise FileNotFoundError("missing shard files:\n" + "\n".join(missing))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lmdb", default="/home1/irteam/data/processed_old/oc20dense.lmdb")
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--require-all-shards", action="store_true")
    ap.add_argument("--only-converged", action="store_true",
                    help="Only replace y_relaxed/pos_relaxed for converged records.")
    ap.add_argument("--map-size-gb", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shard_dir)
    records = _load_shards(shard_dir, args.num_shards, args.require_all_shards)

    src = _open_lmdb(args.source_lmdb, readonly=True)
    n = _read_length(src)
    dst_lmdb = out_dir / "oc20dense.lmdb"
    if dst_lmdb.exists():
        dst_lmdb.unlink()
    dst = _open_lmdb(str(dst_lmdb), readonly=False, map_size=args.map_size_gb * (1 << 30))

    replaced = 0
    kept_missing = 0
    kept_unconverged = 0
    shape_mismatch = 0
    by_system: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    relabel_records: dict[int, dict] = {}

    with src.begin() as rtxn, dst.begin(write=True) as wtxn:
        for meta_key in (b"length", b"anomaly_mask"):
            val = rtxn.get(meta_key)
            if val is not None:
                wtxn.put(meta_key, val)
        for idx in tqdm(range(n), desc="materialize oc20dense", dynamic_ncols=True):
            raw = rtxn.get(str(idx).encode("ascii"))
            if raw is None:
                continue
            entry = pickle.loads(raw)
            rec = records.get(idx)
            if rec is None:
                kept_missing += 1
            elif args.only_converged and not bool(rec.get("converged", False)):
                kept_unconverged += 1
            else:
                pos = np.asarray(rec["pos_relaxed"], dtype=np.float32)
                if pos.shape != np.asarray(entry["pos_relaxed"]).shape:
                    shape_mismatch += 1
                else:
                    entry["pos_relaxed"] = pos
                    entry["y_relaxed"] = float(rec["y_relaxed"])
                    entry["mlip_e_total"] = float(rec["e_total"])
                    entry["mlip_fmax"] = float(rec.get("fmax", np.nan))
                    entry["mlip_converged"] = bool(rec.get("converged", False))
                    entry["mlip_relaxed_source"] = "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed"
                    replaced += 1
                    relabel_records[idx] = rec
            wtxn.put(str(idx).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
            if rec is not None:
                by_system[str(entry.get("system_key"))].append((idx, rec))

    src.close()
    dst.sync()
    dst.close()

    global_min = {}
    for system_key, rows in by_system.items():
        converged = [(idx, r) for idx, r in rows if bool(r.get("converged", False)) and np.isfinite(r.get("e_total", np.nan))]
        pool = converged if converged else [(idx, r) for idx, r in rows if np.isfinite(r.get("e_total", np.nan))]
        if not pool:
            continue
        idx_min, rec_min = min(pool, key=lambda x: float(x[1]["e_total"]))
        e_vals = [float(r["e_total"]) for _, r in converged]
        global_min[system_key] = {
            "system_key": system_key,
            "idx_min": int(idx_min),
            "config_key_min": rec_min.get("config_key"),
            "E_sys_min": float(rec_min["e_total"]),
            "E_sys_mean_converged": float(np.mean(e_vals)) if e_vals else None,
            "n_configs": len(rows),
            "n_converged": len(converged),
            "all_converged": len(converged) == len(rows),
            "used_unconverged_min": not bool(converged),
            "fmax_min": float(rec_min.get("fmax", np.nan)),
            "n_steps_min": int(rec_min.get("n_steps", -1)),
            "pos_relaxed": np.asarray(rec_min["pos_relaxed"], dtype=np.float32),
            "source": "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed",
        }

    records_path = out_dir / "oc20dense_mlip_relax.pkl"
    min_path = out_dir / "oc20dense_mlip_global_min_by_system.pkl"
    summary_path = out_dir / "oc20dense_mlip_relax_summary.json"
    with records_path.open("wb") as f:
        pickle.dump(relabel_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    with min_path.open("wb") as f:
        pickle.dump(global_min, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary = {
        "source": "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed",
        "source_lmdb": args.source_lmdb,
        "shard_dir": str(shard_dir),
        "records_in_source": n,
        "records_in_shards": len(records),
        "replaced": replaced,
        "kept_missing": kept_missing,
        "kept_unconverged": kept_unconverged,
        "shape_mismatch": shape_mismatch,
        "systems": len(global_min),
        "converged_records": sum(1 for r in records.values() if r.get("converged")),
        "converged_rate": sum(1 for r in records.values() if r.get("converged")) / max(len(records), 1),
        "systems_all_converged": sum(1 for r in global_min.values() if r["all_converged"]),
        "systems_with_no_converged_config": sum(1 for r in global_min.values() if r["used_unconverged_min"]),
        "lmdb": str(dst_lmdb),
        "full_records": str(records_path),
        "global_min_by_system": str(min_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
