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

import pickle
import random
from bisect import insort
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
