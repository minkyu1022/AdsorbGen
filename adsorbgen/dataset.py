"""Displacement datasets for flow matching training.

Loads from a *preprocessed* LMDB whose values are pickled dicts of numpy
arrays. Raw OC20 IS2RES LMDBs contain old PyG `Data` objects that fail under
modern torch_geometric, so we preprocess them once into the clean format via
`scripts/preprocess_is2res.py` / `preprocess_oc20dense.py`.

Preprocessed LMDB schema (per key, str(int) key 0..N-1):
    {
        "pos":            np.float32 (N, 3),     # x_ref Cartesian (Å), already centered
        "pos_relaxed":    np.float32 (N, 3),     # x_relax Cartesian (Å), same shift
        "cell":           np.float32 (3, 3),
        "tags":           np.int64   (N,)        # 0=bulk, 1=surface, 2=adsorbate
        "fixed":          np.int64   (N,),
        "atomic_numbers": np.int64   (N,),
        "sid":            int,
        "ads_id":         int,                   # canonical adsorbate id
        "y_init":         float,
        "y_relaxed":      float,
    }
"""

from __future__ import annotations

import pickle
from typing import Dict, List, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


def _to_tensor(v, dtype=None) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        t = v
    elif isinstance(v, np.ndarray):
        t = torch.from_numpy(v)
    else:
        t = torch.tensor(v)
    if dtype is not None:
        t = t.to(dtype)
    return t


class PreprocessedDisplacementDataset(Dataset):
    """Reads a preprocessed displacement LMDB.

    Args:
        lmdb_path: path to preprocessed `.lmdb`.
        max_samples: optional cap.
        recenter: if True, translate pos/pos_relaxed so that the mean of real
            atoms of ``pos`` is at the origin. Always safe — delta1 is shift-
            invariant because the same shift is applied to both tensors.
        training_aug: if True, add a random translation ``N(0, translation_std^2)``
            drawn independently per sample to both ``pos`` and ``pos_relaxed``.
            Teaches the non-equivariant model that outputs should be invariant
            to global shifts. No random rotation — the slab z-axis is
            physically meaningful.
        translation_std: stddev (Å) of training translation augmentation.
    """

    def __init__(
        self,
        lmdb_path: str,
        max_samples: Optional[int] = None,
        recenter: bool = True,
        training_aug: bool = False,
        translation_std: float = 0.5,
        skip_anomaly: bool = True,
    ):
        self.lmdb_path = str(lmdb_path)
        self.recenter = bool(recenter)
        self.training_aug = bool(training_aug)
        self.translation_std = float(translation_std)
        self.skip_anomaly = bool(skip_anomaly)
        self._env = None

        env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            n_total = txn.stat()["entries"]
            raw = txn.get(b"length")
            if raw is not None:
                try:
                    n_total = int(pickle.loads(raw))
                except Exception:
                    pass
            mask_raw = txn.get(b"anomaly_mask")
        env.close()

        self._idx_map: Optional[np.ndarray] = None
        if self.skip_anomaly and mask_raw is not None:
            mask = pickle.loads(mask_raw)
            mask = np.asarray(mask, dtype=np.int8)[:n_total]
            self._idx_map = np.where(mask == 0)[0].astype(np.int64)
            n_clean = int(self._idx_map.size)
            print(
                f"[dataset] {self.lmdb_path}: skip_anomaly=True "
                f"total={n_total} clean={n_clean} filtered={n_total - n_clean}",
                flush=True,
            )
            self.n = n_clean if max_samples is None else min(n_clean, max_samples)
            if max_samples is not None:
                self._idx_map = self._idx_map[:self.n]
        else:
            self.n = n_total if max_samples is None else min(n_total, max_samples)

    def __len__(self) -> int:
        return self.n

    def _ensure_env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
            )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_env()
        real_idx = int(self._idx_map[idx]) if self._idx_map is not None else int(idx)
        with self._env.begin() as txn:
            raw = txn.get(str(real_idx).encode("ascii"))
        if raw is None:
            raise IndexError(f"Key {real_idx} not found in {self.lmdb_path}")
        entry = pickle.loads(raw)

        pos = _to_tensor(entry["pos"], dtype=torch.float32)
        pos_rel = _to_tensor(entry["pos_relaxed"], dtype=torch.float32)

        if self.recenter:
            shift = -pos.mean(dim=0, keepdim=True)
            pos = pos + shift
            pos_rel = pos_rel + shift
        if self.training_aug and self.translation_std > 0.0:
            trans = torch.randn(1, 3) * self.translation_std
            pos = pos + trans
            pos_rel = pos_rel + trans

        sample = {
            "pos": pos,
            "pos_relaxed": pos_rel,
            "cell": _to_tensor(entry["cell"], dtype=torch.float32),
            "tags": _to_tensor(entry["tags"], dtype=torch.long),
            "fixed": _to_tensor(entry["fixed"], dtype=torch.long),
            "atomic_numbers": _to_tensor(entry["atomic_numbers"], dtype=torch.long),
            "sid": torch.tensor(int(entry.get("sid", -1)), dtype=torch.long),
            "ads_id": torch.tensor(int(entry.get("ads_id", -1)), dtype=torch.long),
            "y_relaxed": torch.tensor(float(entry.get("y_relaxed", 0.0)), dtype=torch.float32),
        }
        if sample["cell"].dim() == 3:
            sample["cell"] = sample["cell"].squeeze(0)
        return sample


def collate_displacement(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad variable-length atomic systems into a dense batch.

    Returns a dict with shapes:
        pos:            (B, N_max, 3)
        pos_relaxed:    (B, N_max, 3)
        cell:           (B, 3, 3)
        tags:           (B, N_max)
        fixed:          (B, N_max)         padding = 1 (treated as fixed)
        atomic_numbers: (B, N_max)
        pad_mask:       (B, N_max) bool    True = real atom
        movable_mask:   (B, N_max) bool    (tags in {1,2}) & (fixed==0) & pad
        sid:            (B,)
        ads_id:         (B,)
        y_relaxed:      (B,)
    """
    B = len(samples)
    N_max = max(int(s["pos"].shape[0]) for s in samples)

    def _empty(shape, dtype):
        return torch.zeros(shape, dtype=dtype)

    pos = _empty((B, N_max, 3), torch.float32)
    pos_rel = _empty((B, N_max, 3), torch.float32)
    cell = _empty((B, 3, 3), torch.float32)
    tags = _empty((B, N_max), torch.long)
    fixed = torch.ones((B, N_max), dtype=torch.long)
    atomic_numbers = _empty((B, N_max), torch.long)
    pad_mask = torch.zeros((B, N_max), dtype=torch.bool)
    sid = _empty((B,), torch.long)
    ads_id = _empty((B,), torch.long)
    y_relaxed = _empty((B,), torch.float32)

    for i, s in enumerate(samples):
        n = int(s["pos"].shape[0])
        pos[i, :n] = s["pos"]
        pos_rel[i, :n] = s["pos_relaxed"]
        cell[i] = s["cell"]
        tags[i, :n] = s["tags"]
        fixed[i, :n] = s["fixed"]
        atomic_numbers[i, :n] = s["atomic_numbers"]
        pad_mask[i, :n] = True
        sid[i] = s["sid"]
        ads_id[i] = s["ads_id"]
        y_relaxed[i] = s["y_relaxed"]

    movable_mask = ((tags == 1) | (tags == 2)) & (fixed == 0) & pad_mask

    return {
        "pos": pos,
        "pos_relaxed": pos_rel,
        "cell": cell,
        "tags": tags,
        "fixed": fixed,
        "atomic_numbers": atomic_numbers,
        "pad_mask": pad_mask,
        "movable_mask": movable_mask,
        "sid": sid,
        "ads_id": ads_id,
        "y_relaxed": y_relaxed,
    }
