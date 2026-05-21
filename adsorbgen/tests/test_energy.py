"""Unit tests for adsorbgen.evaluation.energy.UMAEnergy.

Loads fairchem's UMA-s-1p1 predictor with task='oc20' and runs a small
padded batch through the energy head. The test is skipped when fairchem
cannot fetch the checkpoint (offline or unauthorized), since we don't
want CI to fail when the network is unavailable.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _try_load_uma():
    try:
        from adsorbgen.evaluation.energy import UMAEnergy
        return UMAEnergy(model_name="uma-s-1p1", device="cpu", task_name="oc20")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"UMA-s-1p1 unavailable (offline or download failed): {e}")


def _tiny_slab_batch():
    """Two-system padded batch: (B=2, N=10), one with 10 real atoms, one
    with 9 real + 1 pad. Simple 3x3 Cu(100)-like surface + one H adsorbate.
    """
    B, N = 2, 10
    positions = torch.tensor(
        [[[(i % 3) * 3.0, (i // 3) * 3.0, 0.0] for i in range(9)] + [[4.5, 4.5, 2.0]]] * B,
        dtype=torch.float32,
    )
    numbers = torch.tensor([[29] * 9 + [1], [29] * 9 + [1]], dtype=torch.long)
    pad_mask = torch.ones(B, N, dtype=torch.bool)
    pad_mask[1, -1] = False  # second system has 9 real atoms
    numbers[1, -1] = 0
    cell = torch.tensor(
        [[[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 20.0]]] * B,
        dtype=torch.float32,
    )
    return positions, cell, numbers, pad_mask


def test_uma_energy_shape_and_finite():
    model = _try_load_uma()
    pos, cell, nums, pad = _tiny_slab_batch()
    e = model(pos, cell, nums, pad)
    assert e.shape == (2,)
    assert torch.isfinite(e).all()
    assert e.dtype == pos.dtype


def test_uma_energy_per_atom_normalization():
    model = _try_load_uma()
    pos, cell, nums, pad = _tiny_slab_batch()
    e_norm = model(pos, cell, nums, pad)
    model.normalize_per_atom = False
    e_raw = model(pos, cell, nums, pad)
    n_atoms = pad.sum(dim=1).float()
    torch.testing.assert_close(e_norm, e_raw / n_atoms, rtol=1e-5, atol=1e-5)


def test_make_fk_energy_fn_binds_context():
    from adsorbgen.evaluation.energy import make_fk_energy_fn

    class _Fake(torch.nn.Module):
        def forward(self, x, cell, nums, pad):
            # Return the sum of cell[0,0] so we can verify the closure
            # actually used the captured cell.
            return cell[:, 0, 0].to(x.dtype)

    fake = _Fake()
    nums = torch.zeros(2, 3, dtype=torch.long)
    cell = torch.tensor([[[7.0, 0, 0], [0, 7, 0], [0, 0, 7]],
                         [[9.0, 0, 0], [0, 9, 0], [0, 0, 9]]])
    fn = make_fk_energy_fn(fake, nums, cell)
    x = torch.zeros(2, 3, 3)
    pad = torch.ones(2, 3, dtype=torch.bool)
    mov = torch.ones(2, 3, dtype=torch.bool)
    out = fn(x, pad, mov)
    torch.testing.assert_close(out, torch.tensor([7.0, 9.0]))
