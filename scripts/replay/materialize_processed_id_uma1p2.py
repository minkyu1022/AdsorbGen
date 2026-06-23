#!/usr/bin/env python
"""Build processed_ID UMA-s-1p2 LMDBs from batched relabel shards."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
from tqdm.auto import tqdm


def _open(path: str, readonly: bool, map_size: int | None = None):
    kw = {"subdir": False}
    if readonly:
        kw.update({"readonly": True, "lock": False, "readahead": False})
    else:
        kw.update({"map_size": int(map_size or (1 << 40))})
    return lmdb.open(path, **kw)


def _length(env) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        return int(pickle.loads(raw)) if raw is not None else int(txn.stat()["entries"])


def _load_relabel_by_sid(shard_dirs: list[Path], num_shards: int, require_all: bool) -> dict[int, dict]:
    out = {}
    missing = []
    for shard_dir in shard_dirs:
        for shard in range(num_shards):
            path = shard_dir / f"id_lmdb_shard{shard}.pkl"
            if not path.exists():
                missing.append(str(path))
                continue
            with path.open("rb") as f:
                part = pickle.load(f)
            for _, rec in part.items():
                if rec.get("sid") is not None:
                    out[int(rec["sid"])] = rec
            print(f"loaded {shard_dir.name} shard {shard}: {len(part)}")
    if missing and require_all:
        raise FileNotFoundError("missing shard files:\n" + "\n".join(missing))
    return out


def convert(src_path: Path, dst_path: Path, relabel: dict[int, dict], only_converged: bool, map_size_gb: int) -> dict:
    src = _open(str(src_path), readonly=True)
    n = _length(src)
    if dst_path.exists():
        dst_path.unlink()
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst = _open(str(dst_path), readonly=False, map_size=map_size_gb * (1 << 30))
    replaced = kept_missing = kept_unconverged = shape_mismatch = 0
    with src.begin() as rtxn, dst.begin(write=True) as wtxn:
        for meta_key in (b"length", b"anomaly_mask"):
            val = rtxn.get(meta_key)
            if val is not None:
                wtxn.put(meta_key, val)
        for idx in tqdm(range(n), desc=src_path.name, dynamic_ncols=True):
            raw = rtxn.get(str(idx).encode("ascii"))
            if raw is None:
                continue
            entry = pickle.loads(raw)
            sid = int(entry.get("sid", -1))
            rec = relabel.get(sid)
            if rec is None:
                kept_missing += 1
            elif only_converged and not bool(rec.get("converged", False)):
                kept_unconverged += 1
            else:
                pos = np.asarray(rec["pos_relaxed"], dtype=np.float32)
                if pos.shape != np.asarray(entry["pos_relaxed"]).shape:
                    shape_mismatch += 1
                else:
                    entry["pos_relaxed"] = pos
                    entry["mlip_e_total"] = float(rec["e_total"])
                    entry["mlip_fmax"] = float(rec.get("fmax", np.nan))
                    entry["mlip_converged"] = bool(rec.get("converged", False))
                    entry["mlip_relaxed_source"] = "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed"
                    replaced += 1
            wtxn.put(str(idx).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
    src.close()
    dst.sync()
    dst.close()
    return {
        "src": str(src_path),
        "dst": str(dst_path),
        "rows": n,
        "replaced": replaced,
        "kept_missing": kept_missing,
        "kept_unconverged": kept_unconverged,
        "shape_mismatch": shape_mismatch,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", default="")
    ap.add_argument("--shard-dirs", nargs="+", default=[])
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lmdbs", nargs="+", default=[
        "/home1/irteam/data/processed_ID/is2res_train.lmdb",
        "/home1/irteam/data/processed_ID/is2res_val.lmdb",
    ])
    ap.add_argument("--only-converged", action="store_true")
    ap.add_argument("--require-all-shards", action="store_true")
    ap.add_argument("--map-size-gb", type=int, default=512)
    args = ap.parse_args()

    shard_dirs = [Path(p) for p in args.shard_dirs]
    if args.shard_dir:
        shard_dirs.append(Path(args.shard_dir))
    if not shard_dirs:
        raise ValueError("--shard-dir or --shard-dirs is required")
    relabel = _load_relabel_by_sid(shard_dirs, args.num_shards, args.require_all_shards)
    out_dir = Path(args.out_dir)
    summaries = []
    for src in args.lmdbs:
        src_path = Path(src)
        summaries.append(convert(src_path, out_dir / src_path.name, relabel, args.only_converged, args.map_size_gb))
    summary = {
        "source": "uma-s-1p2/oc20/custom_batched_LBFGS/from_stored_pos_relaxed",
        "shard_dirs": [str(p) for p in shard_dirs],
        "relabel_sid_records": len(relabel),
        "outputs": summaries,
    }
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
