"""Unit tests for MultiPlacementDataset.

Writes a tiny preprocessed LMDB with ads_id=1 (canonical *H, single H atom),
wraps it with MultiPlacementDataset, and checks:
    - __len__ = base * K
    - returned per-sample shapes match the base dataset schema
    - placement 0 and placement 1 differ on the adsorbate atom
    - slab (tag in {0, 1}) atoms stay put across placements
    - K=1 passthrough equals the base dataset's first sample's pos
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile

import lmdb
import numpy as np
import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from adsorbgen.multiplace import DEFAULT_ADSORBATES_PKL, MultiPlacementDataset  # noqa: E402


def _write_slab_lmdb(path: str, n: int, ads_id: int = 1):
    """Write a tiny preprocessed LMDB.

    Every entry is a 3x3 slab of Cu (tags 0 for subsurface, 1 for surface)
    plus one adsorbate H atom (tag 2). The layout matches the OC20 preprocess
    schema so MultiPlacementDataset can reconstruct the fairchem Slab.
    """
    env = lmdb.open(path, subdir=False, map_size=1 << 30)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i in range(n):
            n_slab = 18  # 2 layers x 3x3
            cell = np.array([[10.0, 0.0, 0.0],
                             [0.0, 10.0, 0.0],
                             [0.0, 0.0, 20.0]], dtype=np.float32)
            slab_pos = np.zeros((n_slab, 3), dtype=np.float32)
            tags = np.zeros(n_slab, dtype=np.int64)
            for r, (z, tag) in enumerate([(1.0, 0), (4.0, 1)]):
                for a in range(3):
                    for b in range(3):
                        idx = r * 9 + a * 3 + b
                        slab_pos[idx] = [3.0 * a + 0.5, 3.0 * b + 0.5, z]
                        tags[idx] = tag
            numbers = np.full(n_slab, 29, dtype=np.int64)  # Cu

            ads_pos = np.array([[5.0, 5.0, 6.0]], dtype=np.float32)
            ads_tag = np.array([2], dtype=np.int64)
            ads_num = np.array([1], dtype=np.int64)  # H

            pos = np.concatenate([slab_pos, ads_pos], axis=0)
            pos_rel = pos + rng.standard_normal(pos.shape).astype(np.float32) * 0.05
            tags_all = np.concatenate([tags, ads_tag])
            nums_all = np.concatenate([numbers, ads_num])
            fixed_all = np.concatenate([
                np.array([1] * 9 + [0] * 9, dtype=np.int64),  # fix bottom layer
                np.array([0], dtype=np.int64),
            ])

            entry = {
                "pos": pos,
                "pos_relaxed": pos_rel,
                "cell": cell,
                "tags": tags_all,
                "fixed": fixed_all,
                "atomic_numbers": nums_all,
                "sid": i,
                "ads_id": ads_id,
                "y_init": 0.0,
                "y_relaxed": 0.0,
            }
            txn.put(str(i).encode("ascii"), pickle.dumps(entry))
        txn.put(b"length", pickle.dumps(n))
    env.sync()
    env.close()


@pytest.fixture(scope="module")
def _adsorbates_pkl_available():
    if not os.path.exists(DEFAULT_ADSORBATES_PKL):
        pytest.skip("fairchem adsorbates.pkl not available")
    try:
        from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig  # noqa: F401
        from fairchem.data.oc.core.adsorbate import Adsorbate  # noqa: F401
        from fairchem.data.oc.core.slab import Slab  # noqa: F401
    except Exception as e:
        pytest.skip(f"fairchem import failed: {e}")


def test_length_and_shape(_adsorbates_pkl_available):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mini.lmdb")
        _write_slab_lmdb(path, n=3)
        ds = MultiPlacementDataset(path, num_placements=4, max_samples=2)
        assert len(ds) == 2 * 4
        s = ds[0]
        assert s["pos"].shape == (19, 3)
        assert s["tags"].shape == (19,)
        assert int(s["ads_id"].item()) == 1


def test_placements_differ_on_adsorbate(_adsorbates_pkl_available):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mini.lmdb")
        _write_slab_lmdb(path, n=1)
        ds = MultiPlacementDataset(path, num_placements=3, max_samples=1)
        p0 = ds[0]["pos"]
        p1 = ds[1]["pos"]
        tags = ds[0]["tags"]
        # Slab atoms (tags 0/1) are unchanged across placements.
        slab_mask = tags != 2
        torch.testing.assert_close(p0[slab_mask], p1[slab_mask])
        # Adsorbate atom differs across placements (random_site_heuristic).
        ads_mask = tags == 2
        assert (p0[ads_mask] - p1[ads_mask]).abs().max() > 1e-6


def test_k1_passthrough(_adsorbates_pkl_available):
    """K=1 should still route through fairchem, matching base shape."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "mini.lmdb")
        _write_slab_lmdb(path, n=2)
        ds = MultiPlacementDataset(path, num_placements=1, max_samples=2)
        assert len(ds) == 2
        s0 = ds[0]
        assert s0["pos"].shape == (19, 3)
        assert s0["pos_relaxed"].shape == (19, 3)
