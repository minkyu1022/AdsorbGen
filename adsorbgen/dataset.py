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

import os
import pickle
from pathlib import Path
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


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADSORBATES_PKL = os.environ.get(
    "ADSORBATES_PKL",
    str(_PROJECT_ROOT / "data" / "pkls" / "adsorbates.pkl"),
)

# Process-wide cache for fairchem adsorbates.pkl reference geometries.
# Each entry: {ads_id: (numbers (M,) int64, centered_pos (M, 3) float32)}.
_ADS_DB_CACHE: Optional[dict] = None


def load_ads_ref_db(adsorbates_pkl: str) -> dict:
    """Load fairchem adsorbates.pkl and return {ads_id: (numbers, centered_pos)}.

    Centering: positions are translated so the molecule's atom-mean is at the
    origin (CatFlow convention for the reference-geometry feature).
    """
    global _ADS_DB_CACHE
    if _ADS_DB_CACHE is not None:
        return _ADS_DB_CACHE
    with open(adsorbates_pkl, "rb") as f:
        db = pickle.load(f)
    out: dict = {}
    for k, v in db.items():
        atoms = v[0]
        nums = atoms.get_atomic_numbers().astype(np.int64)
        pos = atoms.get_positions().astype(np.float32)
        pos = pos - pos.mean(axis=0, keepdims=True)
        out[int(k)] = (nums, pos)
    _ADS_DB_CACHE = out
    return out


def _ads_ref_pos_for_sample(
    atomic_numbers: np.ndarray,
    tags: np.ndarray,
    ads_id: int,
    ads_db: dict,
    n_atoms: int,
) -> np.ndarray:
    """Build per-atom (n_atoms, 3) reference geometry channel.

    For each tag==2 (adsorbate) atom, fill with a reference position from
    ``ads_db[ads_id]``. Non-adsorbate atoms get 0.

    The LMDB sample and the reference may store adsorbate atoms in different
    orders (e.g. sample [H,H,C,C,O] vs ref [C,C,H,H,O]) but they share the
    same atomic-number multiset. We pair them up by element via a greedy
    consume-in-order scheme: for each sample ads atom (in sample order), pop
    the first reference atom with the same atomic number. This yields a
    deterministic permutation; same-element atoms are interchangeable so
    the molecule's geometry is preserved up to within-element relabelling.

    Asserts the atomic-number multisets match — fail-loud on any mismatch.
    """
    out = np.zeros((n_atoms, 3), dtype=np.float32)
    if ads_id < 0 or ads_id not in ads_db:
        return out
    ref_nums, ref_pos = ads_db[ads_id]
    ads_idx = np.where(tags == 2)[0]
    if len(ads_idx) != len(ref_pos):
        raise AssertionError(
            f"ads_id={ads_id}: sample has {len(ads_idx)} ads atoms but "
            f"reference has {len(ref_pos)}."
        )
    sample_nums = atomic_numbers[ads_idx]
    if sorted(sample_nums.tolist()) != sorted(ref_nums.tolist()):
        raise AssertionError(
            f"ads_id={ads_id}: ads atomic-number multiset mismatch — "
            f"sample={sample_nums.tolist()} ref={ref_nums.tolist()}"
        )
    # Build per-element queue of available reference indices.
    from collections import defaultdict
    queues: dict = defaultdict(list)
    for j, z in enumerate(ref_nums.tolist()):
        queues[int(z)].append(j)
    for sample_local, full_idx in enumerate(ads_idx.tolist()):
        z = int(sample_nums[sample_local])
        ref_j = queues[z].pop(0)
        out[full_idx] = ref_pos[ref_j]
    return out


class PreprocessedDisplacementDataset(Dataset):
    """Reads a preprocessed displacement LMDB.

    If ``provide_ads_ref_pos=True`` and ``adsorbates_pkl`` is given, each
    sample also gets ``ads_ref_pos`` (N, 3) holding the centered reference
    molecular geometry on tag==2 atoms (zeros elsewhere). Used by models
    with the CatFlow-style ads-reference embedding (cfg.use_ads_ref_pos).

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
        provide_ads_ref_pos: bool = False,
        adsorbates_pkl: str = DEFAULT_ADSORBATES_PKL,
    ):
        self.lmdb_path = str(lmdb_path)
        self.recenter = bool(recenter)
        self.training_aug = bool(training_aug)
        self.translation_std = float(translation_std)
        self.skip_anomaly = bool(skip_anomaly)
        self.provide_ads_ref_pos = bool(provide_ads_ref_pos)
        self.adsorbates_pkl = str(adsorbates_pkl) if provide_ads_ref_pos else None
        self._env = None
        if self.provide_ads_ref_pos:
            # warm up the cache once so per-worker loads are cheap
            load_ads_ref_db(self.adsorbates_pkl)

        env = lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False)
        with env.begin() as txn:
            n_total = txn.stat()["entries"]
            raw = txn.get(b"length")
            if raw is not None:
                n_total = int(pickle.loads(raw))
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
        if "system_key" in entry:
            sample["system_key"] = str(entry["system_key"])
        if "config_key" in entry:
            sample["config_key"] = str(entry["config_key"])
        if sample["cell"].dim() == 3:
            sample["cell"] = sample["cell"].squeeze(0)
        if self.provide_ads_ref_pos:
            ads_db = load_ads_ref_db(self.adsorbates_pkl)
            tags_np = np.asarray(entry["tags"]).astype(np.int64)
            nums_np = np.asarray(entry["atomic_numbers"]).astype(np.int64)
            ref_pos = _ads_ref_pos_for_sample(
                nums_np, tags_np, int(entry.get("ads_id", -1)),
                ads_db, n_atoms=int(pos.shape[0]),
            )
            sample["ads_ref_pos"] = torch.from_numpy(ref_pos)
        return sample


# ---------------------------------------------------------------------------
# Placement-prior dataset (new training-time dataset for structured x_0)
# ---------------------------------------------------------------------------

_PRIOR_MODE_MAP = {
    "random":            "random",
    "heuristic":         "heuristic",
    "random_heuristic":  "random_site_heuristic_placement",
}


class PlacementPriorDataset(PreprocessedDisplacementDataset):
    """Training dataset whose `pos` is a fresh fairchem placement each call.

    Surface (tag==1) and bulk (tag==0) atoms use the LMDB's original `pos`
    (pristine slab). Adsorbate atoms (tag==2) are replaced with a freshly
    drawn placement from fairchem `AdsorbateSlabConfig`.

    Args:
        lmdb_path, max_samples, recenter, training_aug, translation_std,
        skip_anomaly: same as base class.
        prior_mode: one of ``"random"`` / ``"heuristic"`` / ``"random_heuristic"``.
        interstitial_gap: fairchem placement gap (Å), default 0.1.
        adsorbates_pkl: path to fairchem's adsorbates.pkl.
        on_failure: ``"fallback"`` (use LMDB pos, warn) or ``"raise"``.
    """

    def __init__(
        self,
        lmdb_path: str,
        prior_mode: str = "random_heuristic",
        interstitial_gap: float = 0.1,
        adsorbates_pkl: str = DEFAULT_ADSORBATES_PKL,
        on_failure: str = "fallback",
        **base_kwargs,
    ):
        super().__init__(lmdb_path, **base_kwargs)
        if prior_mode not in _PRIOR_MODE_MAP:
            raise ValueError(
                f"prior_mode must be one of {list(_PRIOR_MODE_MAP)}, got {prior_mode!r}"
            )
        self.prior_mode = prior_mode
        self._fairchem_mode = _PRIOR_MODE_MAP[prior_mode]
        self.interstitial_gap = float(interstitial_gap)
        self.adsorbates_pkl = str(adsorbates_pkl)
        if on_failure not in ("fallback", "raise"):
            raise ValueError(f"on_failure must be 'fallback' or 'raise'")
        self.on_failure = on_failure

        self._ads_db = None
        self._n_fallbacks = 0

    def _ensure_ads_db(self):
        if self._ads_db is None:
            with open(self.adsorbates_pkl, "rb") as f:
                self._ads_db = pickle.load(f)

    def _draw_placement(self, entry) -> Optional[np.ndarray]:
        """Return (n_ads, 3) adsorbate positions from a fresh fairchem call.

        Returns None on failure (caller decides fallback/raise).
        """
        from ase import Atoms
        from ase.constraints import FixAtoms
        from fairchem.data.oc.core.adsorbate import Adsorbate
        from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig
        from fairchem.data.oc.core.slab import Slab

        self._ensure_ads_db()

        pos = entry["pos"] if isinstance(entry["pos"], np.ndarray) \
              else np.asarray(entry["pos"])
        cell = entry["cell"] if isinstance(entry["cell"], np.ndarray) \
               else np.asarray(entry["cell"])
        if cell.ndim == 3:
            cell = cell[0]
        tags = np.asarray(entry["tags"]).astype(np.int64)
        fixed = np.asarray(entry["fixed"]).astype(np.int64)
        nums = np.asarray(entry["atomic_numbers"]).astype(np.int64)

        ads_id = int(entry["ads_id"])
        if ads_id < 0:
            return None

        slab_idx = np.where(tags != 2)[0]
        ads_idx = np.where(tags == 2)[0]
        if ads_idx.size == 0:
            return None

        slab_atoms = Atoms(
            numbers=nums[slab_idx], positions=pos[slab_idx], cell=cell,
            pbc=[True, True, False], tags=tags[slab_idx].tolist(),
        )
        fix_ind = np.where(fixed[slab_idx] == 1)[0].tolist()
        slab_atoms.set_constraint(FixAtoms(indices=fix_ind or [0]))

        slab = Slab(slab_atoms=slab_atoms, min_ab=0.0)
        ads = Adsorbate(adsorbate_id_from_db=ads_id, adsorbate_db=self._ads_db)
        cfg = AdsorbateSlabConfig(
            slab=slab, adsorbate=ads,
            num_sites=1, num_augmentations_per_site=1,
            interstitial_gap=self.interstitial_gap,
            mode=self._fairchem_mode,
        )
        if not cfg.atoms_list:
            return None
        new_atoms = cfg.atoms_list[0]
        new_pos = new_atoms.get_positions().astype(np.float32)
        n_slab = int(slab_idx.size)
        # fairchem's new_pos layout: [slab_atoms_first, ads_atoms_last]
        ads_placement = new_pos[n_slab:]
        return ads_placement

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._ensure_env()
        real_idx = int(self._idx_map[idx]) if self._idx_map is not None else int(idx)
        with self._env.begin() as txn:
            raw = txn.get(str(real_idx).encode("ascii"))
        if raw is None:
            raise IndexError(f"Key {real_idx} not found in {self.lmdb_path}")
        entry = pickle.loads(raw)

        # fresh placement for adsorbate atoms
        ads_placement = self._draw_placement(entry)

        pos = np.asarray(entry["pos"], dtype=np.float32).copy()
        tags_np = np.asarray(entry["tags"]).astype(np.int64)
        ads_idx_np = np.where(tags_np == 2)[0]

        if ads_placement is not None:
            assert ads_placement.shape == (len(ads_idx_np), 3), (
                f"placement shape {ads_placement.shape} vs expected "
                f"({len(ads_idx_np)}, 3) for sid={entry.get('sid')}"
            )
            pos[ads_idx_np] = ads_placement
        else:
            self._n_fallbacks += 1
            if self.on_failure == "raise":
                raise RuntimeError(
                    f"placement failed for sid={entry.get('sid')} ads_id={entry.get('ads_id')}"
                )
            # fallback: keep LMDB pos unchanged

        pos_t = torch.from_numpy(pos)
        pos_rel = _to_tensor(entry["pos_relaxed"], dtype=torch.float32)

        if self.recenter:
            shift = -pos_t.mean(dim=0, keepdim=True)
            pos_t = pos_t + shift
            pos_rel = pos_rel + shift
        if self.training_aug and self.translation_std > 0.0:
            trans = torch.randn(1, 3) * self.translation_std
            pos_t = pos_t + trans
            pos_rel = pos_rel + trans

        sample = {
            "pos": pos_t,
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
        if self.provide_ads_ref_pos:
            ads_db = load_ads_ref_db(self.adsorbates_pkl)
            ref_pos = _ads_ref_pos_for_sample(
                np.asarray(entry["atomic_numbers"]).astype(np.int64),
                tags_np, int(entry.get("ads_id", -1)),
                ads_db, n_atoms=int(pos_t.shape[0]),
            )
            sample["ads_ref_pos"] = torch.from_numpy(ref_pos)
        return sample


class MixedReplayDataset(Dataset):
    """Wraps a base PlacementPriorDataset and mixes in replay samples.

    Each __getitem__ flips a biased coin (alpha prob) to either:
      - sample from the replay buffer (use its pos_relaxed as the target),
        drawing a fresh placement for x_0 via fairchem, or
      - delegate to base dataset (regular training sample).

    Epoch size is defined as len(base). If buffer is empty, alpha is
    effectively treated as 0 (always falls through to base).
    """

    def __init__(
        self,
        base: Dataset,
        replay_buffer,
        alpha: float = 0.5,
        rng_seed: int = 0,
        placement_helper: Optional["PlacementPriorDataset"] = None,
    ):
        self.base = base
        self.buf = replay_buffer
        self.alpha = float(alpha)
        # placement_helper is any PlacementPriorDataset — used only to call
        # _draw_placement for replay samples. If None, we probe base; fail if
        # base is not a PlacementPriorDataset.
        self._helper = placement_helper
        if self._helper is None:
            if isinstance(base, PlacementPriorDataset):
                self._helper = base
            else:
                raise ValueError(
                    "MixedReplayDataset: base is not PlacementPriorDataset, "
                    "must provide placement_helper"
                )
        import random as _random
        self._rng = _random.Random(rng_seed)

    def __len__(self):
        return len(self.base)

    def _replay_to_sample(self, entry) -> Dict[str, torch.Tensor]:
        # Construct a PlacementPriorDataset-style dict from a replay entry.
        fake_entry = {
            "pos": entry.pos_relaxed.copy(),  # caller replaces ads portion below
            "pos_relaxed": entry.pos_relaxed.copy(),
            "cell": entry.cell, "tags": entry.tags,
            "atomic_numbers": entry.atomic_numbers, "fixed": entry.fixed,
            "sid": int(entry.sid), "ads_id": int(entry.ads_id),
            "y_relaxed": 0.0,
        }
        # Start pos = replay's predicted structure then re-place adsorbate
        # using fairchem (so model sees a fresh x_0 for this better target)
        ads_placement = self._helper._draw_placement(fake_entry)
        pos = np.asarray(fake_entry["pos"], dtype=np.float32).copy()
        if ads_placement is not None:
            ads_idx = np.where(np.asarray(entry.tags) == 2)[0]
            pos[ads_idx] = ads_placement
        pos_t = torch.from_numpy(pos)
        pos_rel = torch.from_numpy(np.asarray(entry.pos_relaxed, dtype=np.float32))
        if self._helper.recenter:
            shift = -pos_t.mean(dim=0, keepdim=True)
            pos_t = pos_t + shift
            pos_rel = pos_rel + shift
        return {
            "pos": pos_t,
            "pos_relaxed": pos_rel,
            "cell": torch.from_numpy(np.asarray(entry.cell, dtype=np.float32)),
            "tags": torch.from_numpy(np.asarray(entry.tags, dtype=np.int64)),
            "fixed": torch.from_numpy(np.asarray(entry.fixed, dtype=np.int64)),
            "atomic_numbers": torch.from_numpy(np.asarray(entry.atomic_numbers, dtype=np.int64)),
            "sid": torch.tensor(int(entry.sid), dtype=torch.long),
            "ads_id": torch.tensor(int(entry.ads_id), dtype=torch.long),
            "y_relaxed": torch.tensor(0.0, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if len(self.buf) > 0 and self._rng.random() < self.alpha:
            entry = self.buf.sample(self._rng)
            return self._replay_to_sample(entry)
        return self.base[idx]


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
    system_key:     list[str|None] optional, for Dense pristine lookup
    config_key:     list[str|None] optional, for Dense traceability
    """
    B = len(samples)
    N_max = max(int(s["pos"].shape[0]) for s in samples)
    has_ads_ref = any("ads_ref_pos" in s for s in samples)
    has_system_key = any("system_key" in s for s in samples)
    has_config_key = any("config_key" in s for s in samples)
    if has_ads_ref:
        assert all("ads_ref_pos" in s for s in samples), (
            "ads_ref_pos must be present on every sample if any sample has it"
        )

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
    ads_ref_pos = _empty((B, N_max, 3), torch.float32) if has_ads_ref else None

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
        if has_ads_ref:
            ads_ref_pos[i, :n] = s["ads_ref_pos"]

    movable_mask = ((tags == 1) | (tags == 2)) & (fixed == 0) & pad_mask

    out = {
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
    if has_ads_ref:
        out["ads_ref_pos"] = ads_ref_pos
    if has_system_key:
        out["system_key"] = [s.get("system_key") for s in samples]
    if has_config_key:
        out["config_key"] = [s.get("config_key") for s in samples]
    return out
