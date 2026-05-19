#!/usr/bin/env python
"""Merge replay-eval shard outputs.

Combines per-shard ReplayBuffer files, metrics JSON, and optional replay-viz
directories produced by scripts/run_replay_4gpu.sh.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from adsorbgen.replay import ReplayBuffer
from adsorbgen.replay_viz import write_index


SUM_KEYS = {
    "systems_evaluated",
    "candidates",
    "n_success",
    "n_added_to_buffer",
    "sys_indices_len",
}

RATE_KEYS = {
    "valid_rate",
    "dissoc_rate",
    "desorbed_rate",
    "surf_changed_rate",
    "intercalated_rate",
    "overlap_rate",
    "uma_unconverged_rate",
}


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _copy_viz(shard_root: Path, final_viz_root: Path, epoch_tag: int) -> list[dict[str, Any]]:
    final_ep = final_viz_root / f"ep{epoch_tag}"
    if final_ep.exists():
        shutil.rmtree(final_ep)
    final_ep.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    out_idx = 0
    for shard_dir in sorted(shard_root.glob("shard_*")):
        shard_ep = shard_dir / "viz" / f"ep{epoch_tag}"
        index_path = shard_ep / "_index.json"
        if not index_path.exists():
            continue
        index = _load_json(index_path)
        by_dir = {
            f"sys_{int(e.get('global_idx', i)):03d}": e
            for i, e in enumerate(index.get("systems", []))
        }
        for sys_dir in sorted(p for p in shard_ep.glob("sys_*") if p.is_dir()):
            dest_name = f"sys_{out_idx:03d}"
            shutil.copytree(sys_dir, final_ep / dest_name)
            meta = by_dir.get(sys_dir.name)
            if meta is None and (sys_dir / "meta.json").exists():
                meta = _load_json(sys_dir / "meta.json")
            if meta is None:
                meta = {}
            meta = dict(meta)
            meta["merged_dir"] = dest_name
            meta["source_shard"] = shard_dir.name
            entries.append(meta)
            out_idx += 1

    write_index(final_ep, entries)
    return entries


def _merge_buffers(shard_root: Path, final_buffer: Path) -> ReplayBuffer:
    merged = ReplayBuffer(mode="append", per_system_cap=10, global_cap=1_070_000)
    for buf_path in sorted(shard_root.glob("shard_*/buffer.pkl")):
        buf = ReplayBuffer.load(buf_path)
        for entry in buf._entries:
            merged.add(entry)
    merged.save(final_buffer)
    return merged


def _merge_metrics(shard_root: Path, final_metrics: Path, buffer: ReplayBuffer,
                   viz_entries: list[dict[str, Any]]) -> dict[str, Any]:
    shard_metrics = []
    for metrics_path in sorted(shard_root.glob("shard_*/metrics.json")):
        m = _load_json(metrics_path)
        m["shard_name"] = metrics_path.parent.name
        shard_metrics.append(m)

    merged: dict[str, Any] = {
        "num_shards": len(shard_metrics),
        "buffer_size": len(buffer),
        "buffer_n_systems": buffer.n_systems(),
        "viz_systems": len(viz_entries),
        "shards": shard_metrics,
    }
    if not shard_metrics:
        final_metrics.parent.mkdir(parents=True, exist_ok=True)
        with open(final_metrics, "w") as f:
            json.dump(merged, f, indent=2)
        return merged

    for key in SUM_KEYS:
        merged[key] = sum(int(m.get(key, 0)) for m in shard_metrics)

    candidates = max(int(merged.get("candidates", 0)), 1)
    for key in RATE_KEYS:
        merged[key] = sum(
            float(m.get(key, 0.0)) * int(m.get("candidates", 0))
            for m in shard_metrics
        ) / candidates

    merged["elapsed_sec_max"] = max(float(m.get("elapsed_sec", 0.0)) for m in shard_metrics)
    merged["wall_sec_max"] = max(float(m.get("wall_sec", 0.0)) for m in shard_metrics)

    final_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(final_metrics, "w") as f:
        json.dump(merged, f, indent=2)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--final-viz-root", required=True)
    parser.add_argument("--final-buffer", required=True)
    parser.add_argument("--final-metrics", required=True)
    parser.add_argument("--epoch-tag", type=int, required=True)
    args = parser.parse_args()

    shard_root = Path(args.shard_root)
    buffer = _merge_buffers(shard_root, Path(args.final_buffer))
    viz_entries = _copy_viz(shard_root, Path(args.final_viz_root), args.epoch_tag)
    metrics = _merge_metrics(shard_root, Path(args.final_metrics), buffer, viz_entries)

    print(
        "[merge] shards={num_shards} buffer={buffer_size} "
        "systems={buffer_n_systems} viz={viz_systems}".format(**metrics)
    )


if __name__ == "__main__":
    main()
