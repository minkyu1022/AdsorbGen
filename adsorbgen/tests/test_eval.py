"""Unit tests for adsorbgen.evaluation.metrics primitives.

Covers the displacement-error aggregator on a handful of hand-built records.
Strict validity (compute_anomaly_metrics) requires ase + fairchem and is
exercised via integration runs, not unit tests.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from adsorbgen.evaluation.metrics import compute_displacement_metrics  # noqa: E402


def test_compute_displacement_metrics_zero_error_is_perfect():
    rec = {
        "pos_pred": torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 1.8]]),
        "pos_gt":   torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 1.8]]),
        "movable_mask": torch.tensor([False, True, True]),
        "atomic_numbers": torch.tensor([78, 78, 6]),
        "tags": torch.tensor([1, 1, 2]),
        "sid": 1,
        "delta_e": 0.0,
    }
    out = compute_displacement_metrics([rec])
    agg = out["aggregate"]
    assert agg["n_samples"] == 1
    assert agg["displacement_mae_A"] == pytest.approx(0.0, abs=1e-6)
    assert agg["displacement_rmse_A"] == pytest.approx(0.0, abs=1e-6)
    assert out["per_sample"][0]["mae"] == pytest.approx(0.0, abs=1e-6)


def test_compute_displacement_metrics_movable_only():
    # Only movable atoms (mask True) contribute to MAE; the surface atom is
    # offset by 100 Å but mask=False so it must be ignored.
    rec = {
        "pos_pred": torch.tensor([[100.0, 0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 2.8]]),
        "pos_gt":   torch.tensor([[0.0,   0.0, 0.0], [2.5, 0.0, 0.0], [1.25, 0.0, 1.8]]),
        "movable_mask": torch.tensor([False, True, True]),
        "atomic_numbers": torch.tensor([78, 78, 6]),
        "tags": torch.tensor([1, 1, 2]),
    }
    out = compute_displacement_metrics([rec])
    # movable diffs: |[0,0,0]| = 0, |[0,0,1]| = 1; mean = 0.5
    assert out["aggregate"]["displacement_mae_A"] == pytest.approx(0.5, abs=1e-6)


def test_compute_displacement_metrics_handles_no_movable():
    rec = {
        "pos_pred": torch.zeros(2, 3),
        "pos_gt":   torch.zeros(2, 3),
        "movable_mask": torch.tensor([False, False]),
        "atomic_numbers": torch.tensor([78, 78]),
        "tags": torch.tensor([1, 1]),
    }
    out = compute_displacement_metrics([rec])
    # No movable atoms anywhere -> mae aggregate is nan.
    import math
    assert math.isnan(out["aggregate"]["displacement_mae_A"])
