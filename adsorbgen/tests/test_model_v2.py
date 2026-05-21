"""Unit tests for DiTDenoiserV2.

Covers:
    - test_shape:         output shape matches (B, N, 3)
    - test_mask_zeroed:   padding/non-movable outputs are exactly zero
    - test_dtype_preserved: float32 input -> float32 output
    - test_gradient_flow:   backward populates grads on every trainable param
    - test_numerics:       NaN input raises; clean input -> finite output
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from adsorbgen.models.factory import build_model  # noqa: E402
from adsorbgen.models.dit_v2 import DiTDenoiserV2, DiTDenoiserV2Config  # noqa: E402


def _tiny_cfg() -> DiTDenoiserV2Config:
    return DiTDenoiserV2Config(
        dim=32, pair_dim=16, depth=2, num_heads=4, mlp_ratio=2.0,
    )


def _make_batch(B: int = 2, N: int = 8, device: str = "cpu", dtype=torch.float32):
    g = torch.Generator(device=device).manual_seed(0)
    pos = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
    delta_t = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype) * 0.1
    t = torch.rand(B, generator=g, device=device, dtype=dtype) * 0.9 + 0.05
    atomic_numbers = torch.randint(1, 50, (B, N), generator=g, device=device)
    tags = torch.randint(0, 3, (B, N), generator=g, device=device)
    movable_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    movable_mask[:, N // 2:] = True  # second half movable
    pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    pad_mask[0, -1] = False  # one padded atom
    cell = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3).contiguous() * 10.0
    return dict(
        pos=pos, delta_t=delta_t, t=t,
        atomic_numbers=atomic_numbers, tags=tags,
        movable_mask=movable_mask, pad_mask=pad_mask,
        cell=cell,
    )


def test_shape():
    model = DiTDenoiserV2(_tiny_cfg()).eval()
    batch = _make_batch(B=2, N=8)
    out = model(**batch)
    assert out.shape == (2, 8, 3)


def test_mask_zeroed():
    model = DiTDenoiserV2(_tiny_cfg()).eval()
    batch = _make_batch(B=2, N=8)
    # Force a non-zero output head so the masking is observable.
    with torch.no_grad():
        model.out_proj.weight.normal_(std=0.1)
        model.out_proj.bias.normal_(std=0.1)
    out = model(**batch)
    movable = batch["movable_mask"]
    # Non-movable atoms must be exactly zero.
    assert torch.equal(out[~movable], torch.zeros_like(out[~movable]))


def test_dtype_preserved():
    model = DiTDenoiserV2(_tiny_cfg()).eval()
    batch = _make_batch(B=2, N=8, dtype=torch.float32)
    out = model(**batch)
    assert out.dtype == torch.float32


def test_gradient_flow():
    model = DiTDenoiserV2(_tiny_cfg()).train()
    batch = _make_batch(B=2, N=8)
    out = model(**batch)
    out.sum().backward()
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
    # Allow at most the zero-init out_proj bias path to be all-zero, but the
    # grad object must still exist on every trainable parameter.
    assert not missing, f"params with no grad: {missing[:5]}"


def test_numerics():
    model = DiTDenoiserV2(_tiny_cfg()).eval()
    batch = _make_batch(B=2, N=8)

    out = model(**batch)
    assert torch.isfinite(out).all()

    bad = dict(batch)
    bad["pos"] = batch["pos"].clone()
    bad["pos"][0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        model(**bad)


def test_factory_returns_v2():
    m = build_model(_tiny_cfg())
    assert isinstance(m, DiTDenoiserV2)
