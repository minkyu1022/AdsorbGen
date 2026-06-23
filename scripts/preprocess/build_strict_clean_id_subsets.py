#!/usr/bin/env python
"""Build small ID train/eval LMDBs from target-geometry strict-valid samples."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch

from adsorbgen.evaluation.metrics import (
    _VALIDITY_FLAGS,
    _score_record_anomaly,
    load_pristine_context,
)


def _read_entry(env: lmdb.Environment, idx: int) -> dict[str, Any]:
    with env.begin() as txn:
        raw = txn.get(str(idx).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def _record_for_target(entry: dict[str, Any]) -> dict[str, Any]:
    pos = np.asarray(entry["pos"], dtype=np.float32)
    pos_rel = np.asarray(entry["pos_relaxed"], dtype=np.float32)
    z = np.asarray(entry["atomic_numbers"], dtype=np.int64)
    tags = np.asarray(entry["tags"], dtype=np.int64)
    cell = np.asarray(entry["cell"], dtype=np.float32)
    if cell.ndim == 3:
        cell = cell[0]
    return {
        "sid": int(entry.get("sid", -1)) if entry.get("sid") is not None else -1,
        "system_key": str(entry.get("system_key")) if entry.get("system_key") is not None else None,
        "ads_id": int(entry.get("ads_id", -1)) if entry.get("ads_id") is not None else -1,
        "pos_ref": torch.from_numpy(pos),
        "pos_pred": torch.from_numpy(pos_rel),
        "pos_gt": torch.from_numpy(pos_rel),
        "atomic_numbers": torch.from_numpy(z),
        "tags": torch.from_numpy(tags),
        "cell": torch.from_numpy(cell),
    }


def _score_one(payload: tuple[str, int, dict[str, Any]]) -> tuple[str, int, dict[str, Any]]:
    split, idx, entry = payload
    return split, idx, _score_record_anomaly(_record_for_target(entry))


def _copy_entry(entry: dict[str, Any], *, source_split: str, source_idx: int) -> dict[str, Any]:
    out = dict(entry)
    out["source_split"] = source_split
    out["source_idx"] = int(source_idx)
    out["strict_target_valid"] = True
    out["strict_target_filter"] = "bare_slab_uma_s_1p2_no_pos_gt_fallback"
    return out


def _write_lmdb(path: Path, rows: list[dict[str, Any]], map_size_gb: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    env = lmdb.open(str(path), subdir=False, map_size=map_size_gb * (1 << 30))
    with env.begin(write=True) as txn:
        for i, row in enumerate(rows):
            txn.put(str(i).encode("ascii"), pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(b"length", pickle.dumps(len(rows), protocol=pickle.HIGHEST_PROTOCOL))
        txn.put(
            b"anomaly_mask",
            pickle.dumps(np.zeros(len(rows), dtype=np.int8), protocol=pickle.HIGHEST_PROTOCOL),
        )
    env.sync()
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-lmdb", required=True)
    ap.add_argument("--val-lmdb", required=True)
    ap.add_argument("--pristine-slabs", required=True)
    ap.add_argument("--pristine-index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-size", type=int, default=30000)
    ap.add_argument("--eval-size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260622)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--map-size-gb", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    train_path = Path(args.train_lmdb)
    val_path = Path(args.val_lmdb)
    out_dir = Path(args.out_dir)
    out_train = out_dir / f"id_strict_clean_train{args.train_size}_seed{args.seed}.lmdb"
    out_eval = out_dir / f"id_strict_clean_eval{args.eval_size}_seed{args.seed}.lmdb"
    report_path = out_dir / f"id_strict_clean_train{args.train_size}_eval{args.eval_size}_seed{args.seed}.report.json"

    for p in (train_path, val_path, Path(args.pristine_slabs), Path(args.pristine_index)):
        if not p.exists():
            raise FileNotFoundError(p)
    if out_train.exists() or out_eval.exists() or report_path.exists():
        raise FileExistsError("output already exists; choose a new --out-dir or seed")

    train_env = lmdb.open(str(train_path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=512)
    val_env = lmdb.open(str(val_path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=512)
    envs = {"train": train_env, "val": val_env}
    with train_env.begin() as txn:
        n_train = int(pickle.loads(txn.get(b"length")))
        mask_train = np.asarray(pickle.loads(txn.get(b"anomaly_mask")), dtype=np.int8)[:n_train]
    with val_env.begin() as txn:
        n_val = int(pickle.loads(txn.get(b"length")))
        mask_val = np.asarray(pickle.loads(txn.get(b"anomaly_mask")), dtype=np.int8)[:n_val]

    candidates = np.array(
        [("train", i) for i in range(n_train)] + [("val", i) for i in range(n_val)],
        dtype=object,
    )
    rng = np.random.default_rng(args.seed)
    rng.shuffle(candidates)

    load_pristine_context(Path(args.pristine_slabs), Path(args.pristine_index))
    ctx = get_context("fork")
    pool = ctx.Pool(args.workers)

    need_total = args.train_size + args.eval_size
    accepted_meta: list[dict[str, Any]] = []
    accepted_entries: list[dict[str, Any]] = []
    scanned = 0
    strict_invalid_counts = Counter()
    original_mask_counts_scanned = Counter()
    original_mask_counts_accepted = Counter()
    source_counts_accepted = Counter()
    source_counts_scanned = Counter()

    try:
        for start in range(0, len(candidates), args.chunk_size):
            chunk_refs = candidates[start : start + args.chunk_size]
            payloads = []
            entries = []
            for split, idx_obj in chunk_refs:
                idx = int(idx_obj)
                entry = _read_entry(envs[str(split)], idx)
                payloads.append((str(split), idx, entry))
                entries.append(entry)
                source_counts_scanned[str(split)] += 1
                mask_arr = mask_train if split == "train" else mask_val
                original_mask_counts_scanned[int(mask_arr[idx])] += 1

            for (split, idx, res), entry in zip(pool.imap(_score_one, payloads, chunksize=16), entries):
                scanned += 1
                if res.get("valid_strict"):
                    if len(accepted_entries) < need_total:
                        accepted_entries.append(_copy_entry(entry, source_split=split, source_idx=idx))
                        mask_arr = mask_train if split == "train" else mask_val
                        original_mask_counts_accepted[int(mask_arr[idx])] += 1
                        source_counts_accepted[split] += 1
                        accepted_meta.append({
                            "new_idx": len(accepted_entries) - 1,
                            "source_split": split,
                            "source_idx": idx,
                            "sid": int(entry.get("sid", -1)) if entry.get("sid") is not None else -1,
                            "ads_id": int(entry.get("ads_id", -1)) if entry.get("ads_id") is not None else -1,
                            "original_anomaly": int(mask_arr[idx]),
                        })
                else:
                    for flag in _VALIDITY_FLAGS:
                        if res.get(f"has_{flag}") is True:
                            strict_invalid_counts[flag] += 1
                            break
                if len(accepted_entries) >= need_total:
                    break
            print(
                f"[subset] scanned={scanned} accepted={len(accepted_entries)}/{need_total} "
                f"source={dict(source_counts_accepted)}",
                flush=True,
            )
            if len(accepted_entries) >= need_total:
                break
    finally:
        pool.close()
        pool.join()
        train_env.close()
        val_env.close()

    if len(accepted_entries) < need_total:
        raise RuntimeError(f"only found {len(accepted_entries)} strict-valid samples; need {need_total}")

    eval_rows = accepted_entries[: args.eval_size]
    train_rows = accepted_entries[args.eval_size : args.eval_size + args.train_size]
    eval_meta = accepted_meta[: args.eval_size]
    train_meta = accepted_meta[args.eval_size : args.eval_size + args.train_size]

    _write_lmdb(out_eval, eval_rows, args.map_size_gb)
    _write_lmdb(out_train, train_rows, args.map_size_gb)

    report = {
        "created_at_unix": time.time(),
        "elapsed_sec": time.time() - t0,
        "seed": args.seed,
        "train_lmdb": str(train_path),
        "val_lmdb": str(val_path),
        "pristine_slabs": str(Path(args.pristine_slabs)),
        "pristine_index": str(Path(args.pristine_index)),
        "source_lengths": {"train": n_train, "val": n_val},
        "source_reports": {
            "train": str(train_path.with_suffix(".report.json")),
            "val": str(val_path.with_suffix(".report.json")),
        },
        "source_geometry_preprocessing_required": {
            "unwrap_adsorbate": True,
            "center_mode": "relaxed_all",
            "note": "Verified from source .report.json before building.",
        },
        "strict_filter": {
            "pos_pred": "pos_relaxed",
            "pos_gt": "pos_relaxed",
            "surface_change_reference": "bare/pristine relaxed slab, hard error if missing",
            "fallback_to_pos_gt": False,
        },
        "requested": {"train": args.train_size, "eval": args.eval_size},
        "written": {"train": len(train_rows), "eval": len(eval_rows)},
        "out_train_lmdb": str(out_train),
        "out_eval_lmdb": str(out_eval),
        "scanned": scanned,
        "accepted_total": len(accepted_entries),
        "source_counts_scanned": dict(source_counts_scanned),
        "source_counts_accepted_total": dict(source_counts_accepted),
        "original_mask_counts_scanned": dict(original_mask_counts_scanned),
        "original_mask_counts_accepted_total": dict(original_mask_counts_accepted),
        "strict_invalid_counts_while_scanning": dict(strict_invalid_counts),
        "train_manifest": train_meta,
        "eval_manifest": eval_meta,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "elapsed_sec",
        "scanned",
        "accepted_total",
        "written",
        "source_counts_accepted_total",
        "original_mask_counts_accepted_total",
        "out_train_lmdb",
        "out_eval_lmdb",
    )}, indent=2, sort_keys=True))
    print(f"REPORT {report_path}")


if __name__ == "__main__":
    main()
