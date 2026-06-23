#!/usr/bin/env python
"""Batched UMA LBFGS relabeling from stored relaxed positions.

This is for rebuilding data references when the UMA version changes.  It reads
existing processed LMDB/cache records, starts from their stored relaxed
coordinates, and writes shard pickles containing the new energy/geometry.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import lmdb
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
GEOOPT = REPO / "geoopt"
for p in (REPO, GEOOPT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from geoopt import load_uma, run_optimizer, summarize_throughput  # noqa: E402


def _open_lmdb(path: str) -> lmdb.Environment:
    try:
        return lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False, max_readers=64)
    except lmdb.Error:
        return lmdb.open(path, readonly=True, lock=False, readahead=False, max_readers=64)


def _lmdb_len(env: lmdb.Environment) -> int:
    with env.begin() as txn:
        raw = txn.get(b"length")
        if raw is not None:
            return int(pickle.loads(raw))
        return int(txn.stat()["entries"])


def _lmdb_indices(env: lmdb.Environment, *, skip_anomaly_mask: bool) -> list[int]:
    with env.begin() as txn:
        n = _lmdb_len(env)
        mask_raw = txn.get(b"anomaly_mask")
    if not skip_anomaly_mask or mask_raw is None:
        return list(range(n))
    mask = np.asarray(pickle.loads(mask_raw), dtype=np.int8)[:n]
    return np.where(mask == 0)[0].astype(int).tolist()


def _read_lmdb(env: lmdb.Environment, idx: int) -> dict:
    with env.begin() as txn:
        raw = txn.get(str(int(idx)).encode("ascii"))
    if raw is None:
        raise KeyError(idx)
    return pickle.loads(raw)


def _cell(rec: dict) -> np.ndarray:
    cell = np.asarray(rec["cell"], dtype=np.float32)
    if cell.ndim == 3:
        cell = cell[0]
    return cell.reshape(3, 3)


def _fixed(rec: dict, tags: np.ndarray) -> np.ndarray:
    fixed = np.asarray(rec.get("fixed", np.zeros_like(tags)), dtype=np.int64)
    if not fixed.astype(bool).any():
        fixed = (tags == 0).astype(np.int64)
    return fixed.astype(np.int64)


def _job_from_record(global_i: int, rec: dict, *, source_kind: str, source_id: str, key: Any) -> dict:
    tags = np.asarray(rec.get("tags", np.zeros(len(rec["atomic_numbers"]))), dtype=np.int64)
    fixed = _fixed(rec, tags)
    start_key = "pos_relaxed" if "pos_relaxed" in rec else "pos"
    pos = np.asarray(rec[start_key], dtype=np.float64)
    return {
        "global_i": int(global_i),
        "source_kind": source_kind,
        "source_id": source_id,
        "source_key": key,
        "sid": rec.get("sid"),
        "system_key": rec.get("system_key"),
        "config_key": rec.get("config_key"),
        "ads_id": rec.get("ads_id"),
        "relax_input": {
            "numbers": np.asarray(rec["atomic_numbers"], dtype=np.int64),
            "tags": tags,
            "fixed": fixed,
            "cell": _cell(rec),
            "pos_pred": pos,
        },
    }


def _result_record(job: dict, result: dict, args: argparse.Namespace) -> dict:
    return {
        "source_kind": job["source_kind"],
        "source_id": job["source_id"],
        "source_key": job["source_key"],
        "sid": job.get("sid"),
        "system_key": job.get("system_key"),
        "config_key": job.get("config_key"),
        "ads_id": job.get("ads_id"),
        "e_total": float(result.get("E_sys", np.nan)),
        "y_relaxed": float(result.get("E_sys", np.nan)),
        "converged": bool(result.get("converged", False)),
        "fmax": float(result.get("fmax", np.nan)),
        "n_steps": int(result.get("n_steps", 0) or 0),
        "n_atoms": int(result.get("n_atoms", 0) or 0),
        "pos_relaxed": np.asarray(result["pos_relaxed"], dtype=np.float32),
        "error": result.get("error"),
        "relaxer": "custom_batched_LBFGS",
        "uma_model": args.uma_model,
        "uma_task": args.uma_task,
        "start_source": "stored_pos_relaxed",
        "fmax_target": float(args.fmax),
        "max_steps": int(args.max_steps),
        "lbfgs_maxstep": float(args.maxstep),
        "lbfgs_memory": int(args.lbfgs_memory),
    }


def _iter_lmdb_jobs(args: argparse.Namespace):
    env = _open_lmdb(args.input)
    indices_all = _lmdb_indices(env, skip_anomaly_mask=bool(args.skip_anomaly_mask))
    indices = indices_all[int(args.shard_idx) :: int(args.num_shards)]
    if args.limit > 0:
        indices = indices[: int(args.limit)]
    for idx in indices:
        rec = _read_lmdb(env, idx)
        yield idx, _job_from_record(idx, rec, source_kind=args.source_kind, source_id=args.input, key=int(idx))
    env.close()


def _iter_bare_slab_jobs(args: argparse.Namespace):
    with open(args.input, "rb") as f:
        db = pickle.load(f)
    keys = sorted(db.keys(), key=repr)
    keys = keys[int(args.shard_idx) :: int(args.num_shards)]
    if args.limit > 0:
        keys = keys[: int(args.limit)]
    for global_i, key in enumerate(keys):
        rec = db[key]
        yield key, _job_from_record(global_i, rec, source_kind=args.source_kind, source_id=args.input, key=key)


def _runtime_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        fmax=float(args.fmax),
        max_steps=int(args.max_steps),
        max_atoms=int(args.max_atoms),
        maxstep=float(args.maxstep),
        lbfgs_memory=int(args.lbfgs_memory),
        lbfgs_damping=float(args.lbfgs_damping),
        lbfgs_alpha=float(args.lbfgs_alpha),
        lbfgs_history_dtype=str(args.lbfgs_history_dtype),
        lbfgs_position_dtype=str(args.lbfgs_position_dtype),
        lbfgs_curvature_guard=str(args.lbfgs_curvature_guard),
        lbfgs_gpu_history_guard=bool(args.lbfgs_gpu_history_guard),
        lbfgs_keep_survivors_on_gpu=bool(args.lbfgs_keep_survivors_on_gpu),
        lbfgs_streaming=bool(args.lbfgs_streaming),
        lbfgs_check_interval=int(args.lbfgs_check_interval),
        lbfgs_stream_sort=bool(args.lbfgs_stream_sort),
        fire_dt=0.1,
        fire_dt_max=1.0,
        cg_step_size=0.04,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-kind", choices=["id_lmdb", "oc20dense_lmdb", "bare_slab_pkl"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-idx", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--records-per-chunk", type=int, default=4096)
    ap.add_argument("--skip-anomaly-mask", action="store_true")
    ap.add_argument("--uma-model", default="uma-s-1p2")
    ap.add_argument("--uma-task", default="oc20")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--max-atoms", type=int, default=65536)
    ap.add_argument("--maxstep", type=float, default=0.2)
    ap.add_argument("--lbfgs-memory", type=int, default=100)
    ap.add_argument("--lbfgs-damping", type=float, default=1.0)
    ap.add_argument("--lbfgs-alpha", type=float, default=70.0)
    ap.add_argument("--lbfgs-history-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-position-dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--lbfgs-curvature-guard", choices=["abs", "positive", "ase"], default="abs")
    ap.add_argument("--lbfgs-gpu-history-guard", action="store_true")
    ap.add_argument("--lbfgs-keep-survivors-on-gpu", action="store_true")
    ap.add_argument("--lbfgs-streaming", action="store_true")
    ap.add_argument("--lbfgs-check-interval", type=int, default=10)
    ap.add_argument("--lbfgs-stream-sort", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.source_kind}_shard{args.shard_idx}.pkl"
    summary_path = out_dir / f"{args.source_kind}_shard{args.shard_idx}.summary.pkl"

    results: dict[Any, dict] = {}
    if args.resume and out_path.exists():
        with out_path.open("rb") as f:
            results = pickle.load(f)
        print(f"[relabel {args.shard_idx}] resume loaded {len(results)} records", flush=True)

    iterator = _iter_bare_slab_jobs(args) if args.source_kind == "bare_slab_pkl" else _iter_lmdb_jobs(args)
    uma = load_uma(args.uma_model, args.uma_task, device)
    opt_args = _runtime_args(args)

    t0 = time.time()
    chunk: list[tuple[Any, dict]] = []
    processed = 0
    conv = 0

    def flush_chunk(chunk_rows: list[tuple[Any, dict]]) -> None:
        nonlocal processed, conv
        if not chunk_rows:
            return
        keys = [k for k, _ in chunk_rows]
        jobs = [j for _, j in chunk_rows]
        t_relax = time.time()
        out = run_optimizer(jobs, uma, opt_args, device, "lbfgs", serial=False)
        elapsed = time.time() - t_relax
        by_global = {int(r["global_i"]): r for r in out}
        for key, job in chunk_rows:
            rec = _result_record(job, by_global[int(job["global_i"])], args)
            results[key] = rec
        processed += len(chunk_rows)
        conv = sum(1 for r in results.values() if r.get("converged"))
        rate = processed / max(time.time() - t0, 1e-6)
        print(
            f"[relabel {args.shard_idx}] processed={processed} stored={len(results)} "
            f"conv={conv} chunk={len(chunk_rows)} relax_rate={summarize_throughput(out, elapsed)} total_rate={rate:.3f}/s",
            flush=True,
        )
        with out_path.open("wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    for key, job in iterator:
        if key in results:
            continue
        chunk.append((key, job))
        if len(chunk) >= int(args.records_per_chunk):
            flush_chunk(chunk)
            chunk = []
    flush_chunk(chunk)

    elapsed = time.time() - t0
    summary = {
        "source_kind": args.source_kind,
        "input": args.input,
        "shard_idx": int(args.shard_idx),
        "num_shards": int(args.num_shards),
        "records": len(results),
        "converged": conv,
        "converged_rate": conv / max(len(results), 1),
        "elapsed_sec": elapsed,
        "throughput_records_per_sec": len(results) / max(elapsed, 1e-6),
        "settings": vars(args),
    }
    with summary_path.open("wb") as f:
        pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
