"""Unit tests for adsorbgen.eval primitives.

Covers covalent_radii lookup, pairwise-overlap detection on synthetic
structures, adsorbate-connectivity distance, and end-to-end metric
aggregation on a handful of hand-built records.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from adsorbgen.eval import (  # noqa: E402
    _adsorbate_connectivity_min,
    _pairwise_min_fraction,
    compute_metrics,
    covalent_radii,
)


def test_covalent_radii_known_values():
    z = torch.tensor([1, 6, 8, 78])
    r = covalent_radii(z)
    # H=31pm, C=73pm, O=66pm, Pt=136pm in the canonical table.
    torch.testing.assert_close(r, torch.tensor([0.31, 0.73, 0.66, 1.36]), rtol=0, atol=1e-4)


def test_covalent_radii_unknown_default():
    r = covalent_radii(torch.tensor([999]), default_pm=150.0)
    assert abs(r.item() - 1.5) < 1e-5


def test_pairwise_overlap_detects_close_atoms():
    # Two C atoms at 1 Å vs covalent sum 1.46 Å -> ratio ~0.685 (overlap).
    pos = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0]])
    z = torch.tensor([6, 6, 6])
    pad = torch.ones(3, dtype=torch.bool)
    ratio = _pairwise_min_fraction(pos, z, pad)
    assert ratio == pytest.approx(1.0 / 1.46, abs=1e-3)


def test_pairwise_overlap_single_atom_is_inf():
    pos = torch.zeros(1, 3)
    z = torch.tensor([6])
    pad = torch.ones(1, dtype=torch.bool)
    assert _pairwise_min_fraction(pos, z, pad) == float("inf")


def test_adsorbate_connectivity_no_adsorbate_is_inf():
    pos = torch.zeros(3, 3)
    tags = torch.tensor([0, 1, 1])
    pad = torch.ones(3, dtype=torch.bool)
    assert _adsorbate_connectivity_min(pos, tags, pad) == float("inf")


def test_adsorbate_connectivity_distance_known():
    # Adsorbate at x=5; nearest surface atom at x=2.5 -> distance 2.5 Å.
    pos = torch.tensor([[0.0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]])
    tags = torch.tensor([1, 1, 2])
    pad = torch.ones(3, dtype=torch.bool)
    assert _adsorbate_connectivity_min(pos, tags, pad) == pytest.approx(2.5, abs=1e-3)


def test_compute_metrics_valid_and_invalid_cases():
    # Sample 0: valid — C bound above the Pt-Pt bridge at ~1.8 Å.
    good = {
        "pos_pred": torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 1.8]]),
        "pos_gt": torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 1.8]]),
        "movable_mask": torch.tensor([False, True, True]),
        "atomic_numbers": torch.tensor([78, 78, 6]),
        "tags": torch.tensor([1, 1, 2]),
        "sid": 1,
        "delta_e": 0.0,
    }
    # Sample 1: atom overlap (two C atoms 0.3 Å apart)
    overlap = {
        "pos_pred": torch.tensor([[0.0, 0, 0], [0.3, 0, 0], [2.5, 0, 0]]),
        "pos_gt": torch.tensor([[0.0, 0, 0], [0.0, 0, 1.0], [2.5, 0, 0]]),
        "movable_mask": torch.tensor([False, True, True]),
        "atomic_numbers": torch.tensor([78, 6, 6]),
        "tags": torch.tensor([1, 2, 2]),
        "sid": 2,
        "delta_e": 0.0,
    }
    # Sample 2: adsorbate desorbed (far from slab)
    dissoc = {
        "pos_pred": torch.tensor([[0.0, 0, 0], [2.5, 0, 0], [2.5, 0, 20.0]]),
        "pos_gt": torch.tensor([[0.0, 0, 0], [2.5, 0, 0], [2.5, 0, 2.0]]),
        "movable_mask": torch.tensor([False, True, True]),
        "atomic_numbers": torch.tensor([78, 78, 6]),
        "tags": torch.tensor([1, 1, 2]),
        "sid": 3,
        "delta_e": 0.0,
    }
    out = compute_metrics([good, overlap, dissoc], overlap_factor=0.6, dissoc_max_dist=3.0)
    agg = out["aggregate"]
    per = out["per_sample"]
    assert agg["n_samples"] == 3
    assert agg["overlap_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert agg["dissociation_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert agg["valid_rate"] == pytest.approx(1 / 3, abs=1e-6)
    assert per[0]["valid"] and not per[1]["valid"] and not per[2]["valid"]
    assert per[1]["has_overlap"] and not per[1]["has_dissoc"]
    assert per[2]["has_dissoc"] and not per[2]["has_overlap"]
    assert per[0]["mae"] == pytest.approx(0.0, abs=1e-6)
