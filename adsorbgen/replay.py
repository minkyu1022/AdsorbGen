"""Replay buffer for expert-iteration flow matching.

Stores model-discovered configurations that (a) UMA-relax to a lower E_sys
than the GT reference for their system AND (b) pass anomaly filters.
These become new x_1 targets for subsequent training iterations.

Entry schema (``ReplayEntry``):
    system_key:          tuple                 physical system identifier
    sid:                 int                   original LMDB sid (for traceability)
    ads_id:              int
    pos_relaxed:         np.ndarray (N,3)      predicted relaxed coords (new target)
    tags:                np.ndarray (N,)
    atomic_numbers:      np.ndarray (N,)
    fixed:               np.ndarray (N,)
    cell:                np.ndarray (3,3)
    E_sys_pred:          float                 UMA E after our relax
    E_sys_gt:            float                 GT reference (phase3 system min)
    improvement:         float                 E_sys_gt - E_sys_pred   (> margin)
    epoch_added:         int
    source_placement:    str                   "random"/"heuristic"/"random_heuristic"

Buffer enforces per-system cap (keep top-K by lowest E_sys_pred within system)
and global cap (evict entries with highest E_sys_pred globally). Sampling is
weighted by ``improvement`` by default.
"""
from __future__ import annotations

import fcntl
import json
import os
import pickle
import random
import time
from bisect import insort
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class ReplayEntry:
    system_key: tuple
    sid: int
    ads_id: int
    pos_relaxed: np.ndarray
    tags: np.ndarray
    atomic_numbers: np.ndarray
    fixed: np.ndarray
    cell: np.ndarray
    E_sys_pred: float
    E_sys_gt: float
    improvement: float
    epoch_added: int
    source_placement: str

    def to_dict(self) -> dict:
        return {
            "system_key": self.system_key, "sid": int(self.sid),
            "ads_id": int(self.ads_id),
            "pos_relaxed": np.asarray(self.pos_relaxed, dtype=np.float32),
            "tags": np.asarray(self.tags, dtype=np.int64),
            "atomic_numbers": np.asarray(self.atomic_numbers, dtype=np.int64),
            "fixed": np.asarray(self.fixed, dtype=np.int64),
            "cell": np.asarray(self.cell, dtype=np.float32),
            "E_sys_pred": float(self.E_sys_pred),
            "E_sys_gt": float(self.E_sys_gt),
            "improvement": float(self.improvement),
            "epoch_added": int(self.epoch_added),
            "source_placement": str(self.source_placement),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReplayEntry":
        return cls(**d)


class ReplayBuffer:
    """Append-mode / replace-mode replay buffer with two-level eviction.

    Args:
        mode: "append" keeps up to per_system_cap entries per system; "replace"
            keeps exactly 1 (best) per system.
        per_system_cap: max entries per (system_key) in append mode.
        global_cap: max total entries (across all systems). On overflow, evict
            the globally highest E_sys_pred entry.
        weight_mode: "improvement" (default, sample ∝ improvement),
            "uniform" (all equal), or "recency" (linearly up-weight newer).
    """

    def __init__(
        self,
        mode: str = "append",
        per_system_cap: int = 10,
        global_cap: int = 1_070_000,
        weight_mode: str = "improvement",
    ):
        if mode not in ("append", "replace"):
            raise ValueError(f"mode must be 'append' or 'replace', got {mode!r}")
        if weight_mode not in ("improvement", "uniform", "recency"):
            raise ValueError(f"weight_mode invalid: {weight_mode!r}")
        self.mode = mode
        self.per_system_cap = int(per_system_cap)
        self.global_cap = int(global_cap)
        self.weight_mode = weight_mode

        # Flat list of entries. Per-system index for O(1) per-system ops.
        self._entries: List[ReplayEntry] = []
        self._by_system: Dict[tuple, List[int]] = {}  # sys_key -> list of indices

    def __len__(self) -> int:
        return len(self._entries)

    def n_systems(self) -> int:
        return len(self._by_system)

    def iter_entries(self):
        return iter(self._entries)

    def add(self, entry: ReplayEntry) -> bool:
        """Insert one entry, honoring per-system and global caps.

        Returns True if inserted (or replaced); False if rejected.
        """
        sk = entry.system_key
        idx_list = self._by_system.get(sk, [])

        if self.mode == "replace":
            if idx_list:
                cur_i = idx_list[0]
                cur = self._entries[cur_i]
                if entry.E_sys_pred < cur.E_sys_pred:
                    self._entries[cur_i] = entry
                    return True
                return False
            # fall through: first entry for this system
            self._entries.append(entry)
            self._by_system[sk] = [len(self._entries) - 1]
            return True

        # append mode
        if len(idx_list) < self.per_system_cap:
            self._entries.append(entry)
            idx_list = self._by_system.setdefault(sk, [])
            idx_list.append(len(self._entries) - 1)
            self._enforce_global_cap()
            return True

        # per-system cap reached: replace worst in this system if new is better
        worst_i = max(idx_list, key=lambda i: self._entries[i].E_sys_pred)
        if entry.E_sys_pred < self._entries[worst_i].E_sys_pred:
            self._entries[worst_i] = entry
            return True
        return False

    def _enforce_global_cap(self):
        while len(self._entries) > self.global_cap:
            # evict globally highest E_sys_pred
            worst_i = max(range(len(self._entries)),
                          key=lambda i: self._entries[i].E_sys_pred)
            self._remove_index(worst_i)

    def _remove_index(self, i: int):
        sk = self._entries[i].system_key
        # swap-remove
        last = len(self._entries) - 1
        if i != last:
            self._entries[i] = self._entries[last]
            moved_sk = self._entries[i].system_key
            # update index in _by_system for the swapped entry
            self._by_system[moved_sk] = [
                j if j != last else i for j in self._by_system[moved_sk]
            ]
        self._entries.pop()
        # drop i from its system list
        ent_list = self._by_system[sk]
        ent_list.remove(i) if i in ent_list else ent_list.remove(last)
        if not ent_list:
            del self._by_system[sk]

    def sample(self, rng: Optional[random.Random] = None) -> ReplayEntry:
        if not self._entries:
            raise RuntimeError("sample from empty buffer")
        rng = rng or random
        if self.weight_mode == "uniform":
            return rng.choice(self._entries)
        if self.weight_mode == "recency":
            weights = [e.epoch_added + 1 for e in self._entries]
        else:  # improvement
            weights = [max(e.improvement, 1e-6) for e in self._entries]
        return rng.choices(self._entries, weights=weights, k=1)[0]

    def stats(self) -> dict:
        if not self._entries:
            return {"size": 0, "n_systems": 0}
        imps = np.array([e.improvement for e in self._entries])
        return {
            "size": len(self._entries),
            "n_systems": len(self._by_system),
            "improvement_mean": float(imps.mean()),
            "improvement_p50": float(np.median(imps)),
            "improvement_p90": float(np.quantile(imps, 0.9)),
            "per_system_saturation": sum(
                1 for v in self._by_system.values() if len(v) >= self.per_system_cap
            ) / max(len(self._by_system), 1),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mode": self.mode,
            "per_system_cap": self.per_system_cap,
            "global_cap": self.global_cap,
            "weight_mode": self.weight_mode,
            "entries": [e.to_dict() for e in self._entries],
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "ReplayBuffer":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        buf = cls(
            mode=payload.get("mode", "append"),
            per_system_cap=payload.get("per_system_cap", 10),
            global_cap=payload.get("global_cap", 1_070_000),
            weight_mode=payload.get("weight_mode", "improvement"),
        )
        for d in payload["entries"]:
            e = ReplayEntry.from_dict(d)
            sk = e.system_key
            idx = len(buf._entries)
            buf._entries.append(e)
            buf._by_system.setdefault(sk, []).append(idx)
        return buf


# ---------------------------------------------------------------------------
# Append-only replay stream (writer + reader)
#
# Contract: replay daemon processes write `ReplayEntry` records into a
# directory tree, one subdir per shard. Training reads them incrementally,
# tracking how many manifest lines it has already consumed per shard.
#
# Layout:
#     {root}/
#       shard_{shard_id}/
#         chunk_00000.pkl    list[ReplayEntry]
#         chunk_00001.pkl
#         manifest.jsonl     one line per flushed chunk
#
# The manifest line is the single source of truth that a chunk has been
# fully written and is safe to read. Chunk pkl files are produced via
# atomic rename (write to .tmp, then os.replace).
# ---------------------------------------------------------------------------


def _atomic_pickle_write(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ReplayStreamWriter:
    """One writer per shard. Append-only with chunked flushes."""

    def __init__(self, root: Path, shard_id: int, chunk_size: int = 64):
        self.root = Path(root) / f"shard_{int(shard_id)}"
        self.shard_id = int(shard_id)
        self.chunk_size = int(chunk_size)
        self.root.mkdir(parents=True, exist_ok=True)
        self._buffer: List[ReplayEntry] = []
        self._next_chunk_idx = self._find_next_chunk_idx()

    def _find_next_chunk_idx(self) -> int:
        existing = sorted(self.root.glob("chunk_*.pkl"))
        if not existing:
            return 0
        try:
            return int(existing[-1].stem.split("_")[1]) + 1
        except (IndexError, ValueError):
            return 0

    def append(self, entry: ReplayEntry) -> None:
        self._buffer.append(entry)
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def extend(self, entries: Iterable[ReplayEntry]) -> None:
        for e in entries:
            self.append(e)

    def flush(self) -> Optional[int]:
        """Write current buffer as a new chunk. Returns chunk index or None."""
        if not self._buffer:
            return None
        idx = self._next_chunk_idx
        chunk_path = self.root / f"chunk_{idx:05d}.pkl"
        payload = [e.to_dict() for e in self._buffer]
        _atomic_pickle_write(chunk_path, payload)

        manifest_path = self.root / "manifest.jsonl"
        line = json.dumps({
            "chunk": idx,
            "n_entries": len(self._buffer),
            "mtime": time.time(),
            "shard_id": self.shard_id,
        }) + "\n"
        with open(manifest_path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                if end > 0:
                    f.seek(end - 1)
                    if f.read(1) != "\n":
                        # Previous writer may have died mid-line. Separate the
                        # corrupted record so this valid record remains readable.
                        f.write("\n")
                    f.seek(0, os.SEEK_END)
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._buffer.clear()
        self._next_chunk_idx += 1
        return idx


class ReplayStreamReader:
    """Incremental multi-shard reader.

    Each call to ``load_new_chunks`` returns ReplayEntry instances from
    manifest lines that were not seen on previous calls. Safe to call
    repeatedly; idempotent within a process lifetime.
    """

    def __init__(self, root: Path, shard_ids: Optional[List[int]] = None):
        self.root = Path(root)
        self.shard_ids: List[int] = (
            sorted(int(s) for s in shard_ids) if shard_ids is not None
            else self._discover_shard_ids()
        )
        # Number of manifest lines already consumed per shard.
        self._consumed: Dict[int, int] = {sid: 0 for sid in self.shard_ids}

    def _discover_shard_ids(self) -> List[int]:
        if not self.root.exists():
            return []
        ids: List[int] = []
        for p in sorted(self.root.glob("shard_*")):
            if p.is_dir():
                try:
                    ids.append(int(p.name.split("_")[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(ids)

    def refresh_shards(self) -> None:
        """Pick up newly-created shard directories (daemon may add later)."""
        for sid in self._discover_shard_ids():
            self._consumed.setdefault(sid, 0)
        # keep ordering deterministic
        self.shard_ids = sorted(self._consumed.keys())

    def load_new_chunks(self) -> List[ReplayEntry]:
        """Return ReplayEntry from manifest lines newer than last consumed."""
        self.refresh_shards()
        out: List[ReplayEntry] = []
        for sid in self.shard_ids:
            shard_root = self.root / f"shard_{sid}"
            manifest = shard_root / "manifest.jsonl"
            if not manifest.exists():
                continue
            with open(manifest, "r") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    lines = f.readlines()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            consumed = self._consumed.get(sid, 0)
            for raw in lines[consumed:]:
                if not raw.endswith("\n"):
                    # line is still being written; pick up next call
                    break
                stripped = raw.strip()
                if not stripped:
                    consumed += 1
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    # newline-terminated but corrupted (e.g. writer crashed
                    # mid-line then recovered): skip permanently and advance.
                    consumed += 1
                    continue
                chunk_path = shard_root / f"chunk_{int(rec['chunk']):05d}.pkl"
                if not chunk_path.exists():
                    break
                try:
                    with open(chunk_path, "rb") as cf:
                        payload = pickle.load(cf)
                except (EOFError, pickle.UnpicklingError):
                    # chunk pkl mid-write race; bail and retry next call
                    break
                for d in payload:
                    out.append(ReplayEntry.from_dict(d))
                consumed += 1
            self._consumed[sid] = consumed
        return out

    def n_consumed(self) -> Dict[int, int]:
        return dict(self._consumed)
