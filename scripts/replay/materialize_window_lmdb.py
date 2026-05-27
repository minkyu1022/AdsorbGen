#!/usr/bin/env python
"""Build a moving-window self-improvement LMDB.

For each system, collect candidates from the current training LMDB(s) plus the
latest replay outputs, find the current minimum E_sys, and keep every candidate
within ``min(E_sys) + window_ev``.  This generalizes the old one-best materializer:
success reporting stays unchanged, but training can see multiple low-energy
configurations per system.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np


def frozen_key(x: Any):
    if isinstance(x, (list, tuple)):
        return tuple(frozen_key(v) for v in x)
    return x


def jsonable_key(x: Any):
    if isinstance(x, tuple):
        return [jsonable_key(v) for v in x]
    if isinstance(x, list):
        return [jsonable_key(v) for v in x]
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


def energy_from_entry(entry: dict, gt_info: dict | None) -> float | None:
    for key in (
        "self_improve_window_E_sys",
        "self_improve_E_sys",
        "mlip_e_total",
    ):
        if key in entry and entry[key] is not None:
            try:
                e = float(entry[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(e):
                return e
    if isinstance(gt_info, dict) and gt_info.get("E_sys_own") is not None:
        e = float(gt_info["E_sys_own"])
        if np.isfinite(e):
            return e
    return None


def load_replay_candidates(replay_dir: Path) -> tuple[list[dict], str]:
    candidate_paths = sorted(replay_dir.glob("candidate_shard*.pkl"))
    source = "candidate_shard*.pkl"
    if not candidate_paths:
        candidate_paths = sorted(replay_dir.glob("success_shard*.pkl"))
        source = "success_shard*.pkl"
    rows: list[dict] = []
    for path in candidate_paths:
        with path.open("rb") as f:
            rows.extend(pickle.load(f))
    return rows, source


def make_replay_entry(row: dict, src_envs: list[lmdb.Environment]) -> dict:
    src_env = src_envs[int(row["lmdb_id"])]
    entry = dict(read_entry(src_env, int(row["raw_idx"])))
    e_sys = float(row["E_sys"])
    entry["pos_relaxed"] = np.asarray(row["pos_relaxed"], dtype=np.float32)
    entry["y_relaxed"] = e_sys
    entry["mlip_e_total"] = e_sys
    entry["mlip_fmax"] = float(row["fmax"])
    entry["mlip_converged"] = bool(row["converged"])
    entry["self_improve_window_source"] = "replay"
    entry["self_improve_window_E_sys"] = e_sys
    entry["self_improve_window_system_key"] = jsonable_key(row["system_key"])
    entry["self_improve_window_sample_i"] = int(row.get("sample_i", -1))
    entry["self_improve_window_global_i"] = int(row.get("global_i", -1))
    entry["self_improve_window_status"] = row.get("status")
    entry["self_improve_window_valid"] = bool(row.get("valid", False))
    entry["self_improve_window_success"] = bool(row.get("success", False))
    entry["self_improve_E_sys_ref"] = float(row["E_sys_ref"])
    entry["self_improve_E_sys"] = e_sys
    entry["self_improve_improvement"] = float(row["improvement"])
    entry["self_improve_source_raw_idx"] = int(row["raw_idx"])
    entry["self_improve_source_lmdb_id"] = int(row["lmdb_id"])
    return entry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-lmdb", nargs="+", required=True)
    p.add_argument("--gt-index", default="/home/irteam/data/replay/gt_index_by_sid_oc20_lbfgs.pkl")
    p.add_argument("--replay-dir", required=True)
    p.add_argument("--out-lmdb", required=True)
    p.add_argument("--window-ev", type=float, default=0.1)
    p.add_argument("--map-size-gb", type=float, default=64.0)
    args = p.parse_args()

    out_lmdb = Path(args.out_lmdb)
    out_lmdb.parent.mkdir(parents=True, exist_ok=True)
    for pth in [out_lmdb, Path(str(out_lmdb) + "-lock")]:
        if pth.exists():
            pth.unlink()

    with open(args.gt_index, "rb") as f:
        gt_index = pickle.load(f)

    src_envs = [
        lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
        for path in args.train_lmdb
    ]

    grouped: dict[tuple, list[dict]] = {}
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
                gt_info = gt_index.get(sid)
                if isinstance(gt_info, dict) and gt_info.get("system_key") is not None:
                    sk = frozen_key(gt_info["system_key"])
                elif entry.get("self_improve_window_system_key") is not None:
                    sk = frozen_key(entry["self_improve_window_system_key"])
                else:
                    continue
                e_sys = energy_from_entry(entry, gt_info)
                if e_sys is None:
                    continue
                grouped.setdefault(sk, []).append({
                    "kind": "train",
                    "E_sys": float(e_sys),
                    "lmdb_id": int(lid),
                    "raw_idx": int(raw_idx),
                    "entry": entry,
                    "sid": sid,
                })
                kept += 1
        scan_counts.append({"path": path, "eligible_rows": kept})

    replay_rows, replay_source = load_replay_candidates(Path(args.replay_dir))
    replay_loaded = 0
    replay_usable = 0
    for row in replay_rows:
        replay_loaded += 1
        if not row.get("valid", False) or not row.get("converged", False):
            continue
        e_sys = float(row.get("E_sys", float("nan")))
        if not np.isfinite(e_sys):
            continue
        sk = frozen_key(row["system_key"])
        grouped.setdefault(sk, []).append({
            "kind": "replay",
            "E_sys": e_sys,
            "row": row,
            "sid": int(row.get("sid", -1)),
        })
        replay_usable += 1

    selected: list[dict] = []
    per_system_counts = []
    for sk in sorted(grouped, key=str):
        pool = grouped[sk]
        e_min = min(float(c["E_sys"]) for c in pool)
        threshold = e_min + float(args.window_ev)
        keep = [c for c in pool if float(c["E_sys"]) <= threshold + 1e-12]
        keep.sort(key=lambda c: (float(c["E_sys"]), c["kind"], int(c.get("raw_idx", -1))))
        for c in keep:
            c["system_key"] = sk
            c["window_min_E_sys"] = e_min
            c["window_threshold_E_sys"] = threshold
            selected.append(c)
        per_system_counts.append({
            "system_key": jsonable_key(sk),
            "n_pool": len(pool),
            "n_kept": len(keep),
            "n_train_kept": sum(1 for c in keep if c["kind"] == "train"),
            "n_replay_kept": sum(1 for c in keep if c["kind"] == "replay"),
            "min_E_sys": e_min,
            "threshold_E_sys": threshold,
        })

    out_env = lmdb.open(
        str(out_lmdb),
        subdir=False,
        map_size=int(args.map_size_gb * (1024 ** 3)),
        meminit=False,
    )
    anomaly_mask = np.zeros(len(selected), dtype=np.int8)
    replay_kept = 0
    train_kept = 0
    manifest = []
    with out_env.begin(write=True) as txn:
        txn.put(b"length", pickle.dumps(len(selected)))
        for out_i, cand in enumerate(selected):
            if cand["kind"] == "replay":
                entry = make_replay_entry(cand["row"], src_envs)
                replay_kept += 1
            else:
                entry = dict(cand["entry"])
                train_kept += 1
                entry["self_improve_window_source"] = "train_lmdb"
                entry["self_improve_window_E_sys"] = float(cand["E_sys"])
                entry["self_improve_window_system_key"] = jsonable_key(cand["system_key"])
            entry["self_improve_window_ev"] = float(args.window_ev)
            entry["self_improve_window_min_E_sys"] = float(cand["window_min_E_sys"])
            entry["self_improve_window_threshold_E_sys"] = float(cand["window_threshold_E_sys"])
            entry["self_improve_window_rank_energy"] = int(out_i)
            txn.put(str(out_i).encode("ascii"), pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL))
            manifest.append({
                "out_idx": out_i,
                "kind": cand["kind"],
                "sid": int(cand.get("sid", -1)),
                "system_key": jsonable_key(cand["system_key"]),
                "E_sys": float(cand["E_sys"]),
                "window_min_E_sys": float(cand["window_min_E_sys"]),
                "window_threshold_E_sys": float(cand["window_threshold_E_sys"]),
            })
        txn.put(b"anomaly_mask", pickle.dumps(anomaly_mask, protocol=pickle.HIGHEST_PROTOCOL))
    out_env.sync()
    out_env.close()
    for env in src_envs:
        env.close()

    systems_with_replay = sum(1 for r in per_system_counts if r["n_replay_kept"] > 0)
    summary = {
        "out_lmdb": str(out_lmdb),
        "policy": "per-system moving window",
        "window_ev": float(args.window_ev),
        "n_systems": len(grouped),
        "n_rows": len(selected),
        "n_train_rows_kept": train_kept,
        "n_replay_rows_kept": replay_kept,
        "systems_with_replay_rows_kept": systems_with_replay,
        "source_train_lmdb": args.train_lmdb,
        "gt_index": args.gt_index,
        "replay_dir": str(args.replay_dir),
        "replay_candidate_source": replay_source,
        "replay_rows_loaded": replay_loaded,
        "replay_rows_usable": replay_usable,
        "scan_counts": scan_counts,
        "mean_rows_per_system": len(selected) / max(len(grouped), 1),
    }
    out_lmdb.with_suffix(".report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out_lmdb.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    out_lmdb.with_suffix(".per_system_counts.json").write_text(
        json.dumps(per_system_counts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
