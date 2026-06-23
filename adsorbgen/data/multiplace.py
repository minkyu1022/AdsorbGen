"""K-placement wrapper dataset for inference.

Given a preprocessed displacement LMDB where each entry carries the canonical
``ads_id``, generate K fresh ``random_site_heuristic_placement`` starts of the
canonical adsorbate on the same slab via fairchem's ``AdsorbateSlabConfig``.

The wrapper exposes ``len(base) * K`` samples so the existing DataLoader /
collate pipeline is reused unchanged. Per-worker state caches the K placements
for the current base index so sequential reads within a K-block cost one
fairchem call, not K.

Placements share everything (tags, numbers, fixed, pos_relaxed, cell) except
``pos``, which is the re-placed initial structure. Downstream inference reshapes
``(B*K, ...)`` back to ``(B, K, ...)`` when writing records.
"""

from __future__ import annotations

import pickle
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from adsorbgen.data.dataset import (
    DEFAULT_ADSORBATES_PKL,
    PreprocessedDisplacementDataset,
    load_gaussian_ads_stats,
    load_pristine_slab_db,
    lookup_pristine_slab_pos,
)

_PRIOR_MODE_MAP = {
    "random":            "random",
    "heuristic":         "heuristic",
    "random_heuristic":  "random_site_heuristic_placement",
}

_CUSTOM_PRIOR_MODES = {
    "gaussian_ads_train_stats",
}


class MultiPlacementDataset(Dataset):
    """Wrap a preprocessed displacement LMDB with K adsorbate re-placements.

    Args:
        lmdb_path: preprocessed displacement LMDB with ``ads_id`` field.
        num_placements: K placements per base system.
        adsorbates_pkl_path: path to fairchem's ``adsorbates.pkl``.
        max_samples: optional cap on base systems.
    """

    def __init__(
        self,
        lmdb_path: str,
        num_placements: int,
        adsorbates_pkl_path: str = DEFAULT_ADSORBATES_PKL,
        max_samples: Optional[int] = None,
        unique_by_system_key: bool = False,
        prior_mode: str = "random_heuristic",
        interstitial_gap: float = 0.1,
        provide_ads_ref_pos: bool = False,
        slab_source: str = "initial",
        pristine_slabs: str = "",
        pristine_index: str = "",
        gaussian_ads_stats: str = "",
        gaussian_ads_std_scale: float = 1.0,
    ):
        if slab_source not in ("initial", "pristine_relaxed"):
            raise ValueError("slab_source must be 'initial' or 'pristine_relaxed'")
        if slab_source == "pristine_relaxed" and not pristine_slabs:
            raise ValueError("slab_source='pristine_relaxed' requires pristine_slabs")
        self.slab_source = str(slab_source)
        self.pristine_slabs = str(pristine_slabs or "")
        self.pristine_index = str(pristine_index or "")
        self.base = PreprocessedDisplacementDataset(
            lmdb_path,
            max_samples=max_samples,
            unique_by_system_key=unique_by_system_key,
            recenter=(self.slab_source == "initial"),
            training_aug=False,
            provide_ads_ref_pos=provide_ads_ref_pos,
            adsorbates_pkl=adsorbates_pkl_path,
        )
        self.K = int(num_placements)
        assert self.K >= 1
        self._ads_pkl_path = str(adsorbates_pkl_path)
        self._ads_db: Optional[dict] = None
        self._cache_base_i: int = -1
        self._cache_placements: Optional[List[dict]] = None
        if prior_mode not in _PRIOR_MODE_MAP and prior_mode not in _CUSTOM_PRIOR_MODES:
            raise ValueError(
                f"prior_mode must be one of "
                f"{list(_PRIOR_MODE_MAP) + sorted(_CUSTOM_PRIOR_MODES)}, got {prior_mode!r}"
            )
        self.prior_mode = prior_mode
        self._fairchem_mode = _PRIOR_MODE_MAP.get(prior_mode)
        self.interstitial_gap = float(interstitial_gap)
        self.gaussian_ads_stats = str(gaussian_ads_stats or "")
        self.gaussian_ads_std_scale = float(gaussian_ads_std_scale)
        if self.prior_mode == "gaussian_ads_train_stats":
            load_gaussian_ads_stats(self.gaussian_ads_stats)
        self._pristine_db = None
        self._pristine_sid_to_key = None
        self._pristine_system_to_key = None

    def __len__(self) -> int:
        return len(self.base) * self.K

    def _ensure_db(self) -> None:
        if self._ads_db is None:
            with open(self._ads_pkl_path, "rb") as f:
                self._ads_db = pickle.load(f)

    def _ensure_pristine_db(self) -> None:
        if self.slab_source != "pristine_relaxed":
            return
        if self._pristine_db is None:
            db, sid_to_key, system_to_key = load_pristine_slab_db(
                self.pristine_slabs, self.pristine_index,
            )
            self._pristine_db = db
            self._pristine_sid_to_key = sid_to_key
            self._pristine_system_to_key = system_to_key

    def _generate(self, base_i: int) -> List[dict]:
        from ase import Atoms
        from ase.constraints import FixAtoms
        from fairchem.data.oc.core.adsorbate import Adsorbate
        from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig
        from fairchem.data.oc.core.slab import Slab

        self._ensure_db()
        sample = self.base[base_i]
        ads_id = int(sample["ads_id"].item())
        if ads_id < 0:
            raise RuntimeError(
                f"sample {base_i}: ads_id=-1 — cannot run multi-placement without canonical adsorbate"
            )

        pos = sample["pos"].numpy().astype(np.float64)
        cell = sample["cell"].numpy().astype(np.float64)
        tags = sample["tags"].numpy().astype(np.int64)
        fixed = sample["fixed"].numpy().astype(np.int64)
        atomic_numbers = sample["atomic_numbers"].numpy().astype(np.int64)

        slab_idx = np.where(tags != 2)[0]
        ads_idx = np.where(tags == 2)[0]
        if ads_idx.size == 0:
            raise RuntimeError(f"sample {base_i}: no tag==2 adsorbate atoms")
        if self.slab_source == "pristine_relaxed":
            self._ensure_pristine_db()
            pristine = lookup_pristine_slab_pos(
                self._pristine_db or {},
                self._pristine_sid_to_key or {},
                self._pristine_system_to_key or {},
                int(sample["sid"].item()) if "sid" in sample else -1,
                sample.get("system_key", None),
            )
            if pristine is None or pristine.shape[0] != slab_idx.size:
                got = None if pristine is None else pristine.shape
                raise RuntimeError(
                    f"sample {base_i}: pristine slab lookup failed "
                    f"expected_slab_atoms={slab_idx.size} got={got}"
                )
            pos[slab_idx] = pristine.astype(np.float64, copy=False)

        if self.prior_mode == "gaussian_ads_train_stats":
            mean, std = load_gaussian_ads_stats(self.gaussian_ads_stats)
            placements = []
            for _ in range(self.K):
                pos_out = pos.astype(np.float32).copy()
                sample_ads = np.random.randn(ads_idx.size, 3).astype(np.float32)
                pos_out[ads_idx] = sample_ads * (
                    std[None, :] * np.float32(self.gaussian_ads_std_scale)
                ) + mean[None, :]
                new_sample = dict(sample)
                if self.slab_source == "pristine_relaxed":
                    pos_rel = sample["pos_relaxed"].numpy().astype(np.float32).copy()
                    shift = -pos_out.mean(axis=0, keepdims=True)
                    pos_out = pos_out + shift
                    pos_rel = pos_rel + shift
                    new_sample["pos_relaxed"] = torch.from_numpy(pos_rel)
                new_sample["pos"] = torch.from_numpy(pos_out)
                placements.append(new_sample)
            return placements

        canonical_atoms = self._ads_db[ads_id][0]
        canonical_nums = list(canonical_atoms.get_atomic_numbers())
        if list(atomic_numbers[ads_idx]) != canonical_nums:
            raise RuntimeError(
                f"sample {base_i}: adsorbate atomic-number mismatch with canonical "
                f"ads_id={ads_id}: preprocessed={list(atomic_numbers[ads_idx])} "
                f"canonical={canonical_nums}"
            )

        slab_atoms = Atoms(
            numbers=atomic_numbers[slab_idx],
            positions=pos[slab_idx],
            cell=cell,
            pbc=[True, True, False],
            tags=tags[slab_idx].tolist(),
        )
        slab_fixed_indices = np.where(fixed[slab_idx] == 1)[0].tolist()
        if slab_fixed_indices:
            slab_atoms.set_constraint(FixAtoms(indices=slab_fixed_indices))
        else:
            slab_atoms.set_constraint(FixAtoms(indices=[0]))

        slab = Slab(slab_atoms=slab_atoms, min_ab=0.0)
        ads = Adsorbate(adsorbate_id_from_db=ads_id, adsorbate_db=self._ads_db)

        cfg = AdsorbateSlabConfig(
            slab=slab,
            adsorbate=ads,
            num_sites=self.K,
            num_augmentations_per_site=1,
            interstitial_gap=self.interstitial_gap,
            mode=self._fairchem_mode,
        )

        n_slab = int(slab_idx.size)
        placements: List[dict] = []
        for new_atoms in cfg.atoms_list[: self.K]:
            new_pos = new_atoms.get_positions().astype(np.float32)
            pos_out = pos.astype(np.float32).copy()
            pos_out[slab_idx] = new_pos[:n_slab]
            pos_out[ads_idx] = new_pos[n_slab:]
            new_sample = dict(sample)
            if self.slab_source == "pristine_relaxed":
                pos_rel = sample["pos_relaxed"].numpy().astype(np.float32).copy()
                shift = -pos_out.mean(axis=0, keepdims=True)
                pos_out = pos_out + shift
                pos_rel = pos_rel + shift
                new_sample["pos_relaxed"] = torch.from_numpy(pos_rel)
            new_sample["pos"] = torch.from_numpy(pos_out)
            placements.append(new_sample)

        # Fairchem may return fewer than K placements if Delaunay yields few
        # unique sites. Pad by repeating the last one.
        if not placements:
            raise RuntimeError(f"sample {base_i}: fairchem returned zero placements")
        while len(placements) < self.K:
            placements.append(placements[-1])

        return placements

    def __getitem__(self, idx: int) -> dict:
        base_i, j = divmod(int(idx), self.K)
        if base_i != self._cache_base_i or self._cache_placements is None:
            self._cache_placements = self._generate(base_i)
            self._cache_base_i = base_i
        return self._cache_placements[j]
