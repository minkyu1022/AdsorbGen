"""Tests for the current absolute-coordinate flow implementation."""

from __future__ import annotations

import torch

from adsorbgen.flow import (
    FKSteeringConfig,
    FlowConfig,
    cfg_model_forward,
    euler_sample,
    flow_loss_split,
    interpolate_xt,
    minimum_image,
    sample_t,
    x1_loss,
)
from adsorbgen.models.dit import DiTDenoiser, DiTDenoiserConfig


def _tiny_model() -> DiTDenoiser:
    return DiTDenoiser(
        DiTDenoiserConfig(
            atom_s=32,
            atom_z=16,
            token_s=32,
            token_z=16,
            enc_depth=1,
            trunk_depth=1,
            dec_depth=1,
            enc_heads=2,
            trunk_heads=2,
            dec_heads=2,
            mlp_ratio=2.0,
            num_elements=64,
        )
    )


def _batch(batch_size: int = 2, n_atoms: int = 6):
    g = torch.Generator().manual_seed(7)
    pos = torch.randn(batch_size, n_atoms, 3, generator=g)
    x_t = pos + 0.1 * torch.randn(batch_size, n_atoms, 3, generator=g)
    t = torch.full((batch_size,), 0.5)
    atomic_numbers = torch.randint(1, 50, (batch_size, n_atoms), generator=g)
    tags = torch.randint(0, 3, (batch_size, n_atoms), generator=g)
    movable_mask = torch.zeros(batch_size, n_atoms, dtype=torch.bool)
    movable_mask[:, n_atoms // 2:] = True
    pad_mask = torch.ones(batch_size, n_atoms, dtype=torch.bool)
    cell = torch.eye(3).unsqueeze(0).expand(batch_size, 3, 3) * 10.0
    return {
        "pos": pos,
        "x_t": x_t,
        "t": t,
        "atomic_numbers": atomic_numbers,
        "tags": tags,
        "movable_mask": movable_mask,
        "pad_mask": pad_mask,
        "cell": cell,
    }


def test_minimum_image_wraps_into_half_cell():
    cell = torch.eye(3).unsqueeze(0) * 10.0
    delta = torch.tensor([[[9.0, 0.0, 0.0]]])
    wrapped = minimum_image(delta, cell)
    assert torch.allclose(wrapped, torch.tensor([[[-1.0, 0.0, 0.0]]]), atol=1e-5)


def test_sample_t_respects_eps_bounds():
    cfg = FlowConfig(eps=0.1)
    t = sample_t(128, cfg, device=torch.device("cpu"))
    assert float(t.min()) >= 0.1
    assert float(t.max()) <= 0.9


def test_interpolate_xt_freezes_non_movable_atoms():
    pos_0 = torch.zeros(1, 4, 3)
    pos_1 = torch.ones(1, 4, 3)
    movable = torch.tensor([[True, False, True, False]])
    x_t = interpolate_xt(pos_0, pos_1, torch.tensor([0.25]), movable)
    assert torch.allclose(x_t[0, [0, 2]], torch.full((2, 3), 0.25))
    assert torch.allclose(x_t[0, [1, 3]], torch.zeros(2, 3))


def test_x1_loss_ignores_non_movable_atoms():
    target = torch.zeros(1, 4, 3)
    pred = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [99.0, 0.0, 0.0], [99.0, 0.0, 0.0]]])
    movable = torch.tensor([[True, True, False, False]])
    assert float(x1_loss(pred, target, movable)) == 0.0


def test_flow_loss_split_returns_current_groups():
    pos_0 = torch.zeros(1, 4, 3)
    pos_1 = torch.ones(1, 4, 3)
    pred = pos_1.clone()
    tags = torch.tensor([[0, 1, 2, 2]])
    movable = torch.tensor([[False, True, True, False]])
    losses = flow_loss_split(pred, pos_0, pos_1, movable, tags, loss_type="l1")
    assert set(losses) == {"total", "surf", "ads"}
    assert float(losses["total"]) == 0.0


def test_current_dit_forward_shapes_and_freezes_non_movable():
    model = _tiny_model().eval()
    batch = _batch()
    out = model(**batch)
    assert out.shape == batch["pos"].shape
    assert torch.isfinite(out).all()
    non_movable = ~batch["movable_mask"]
    assert torch.allclose(out[non_movable], batch["pos"][non_movable], atol=1e-6)


def test_euler_sampler_oracle_recovers_absolute_target():
    torch.manual_seed(0)
    pos_0 = torch.randn(1, 5, 3)
    target = pos_0 + 0.2 * torch.randn(1, 5, 3)
    movable = torch.ones(1, 5, dtype=torch.bool)
    pad = torch.ones(1, 5, dtype=torch.bool)
    cfg = FlowConfig(eps=1e-5)

    def oracle(x_t, t):  # noqa: ARG001
        return target

    x_out = euler_sample(oracle, pos_0, movable, pad, cfg, num_steps=200)
    assert torch.allclose(x_out, target, atol=1e-2)


def test_refine_final_uses_extra_forward_prediction():
    pos_0 = torch.zeros(1, 3, 3)
    first = torch.ones_like(pos_0)
    final = torch.full_like(pos_0, 2.0)
    movable = torch.ones(1, 3, dtype=torch.bool)
    pad = torch.ones(1, 3, dtype=torch.bool)
    cfg = FlowConfig(eps=1e-5)
    calls = {"n": 0}

    def model_forward(x_t, t):  # noqa: ARG001
        calls["n"] += 1
        return final if calls["n"] > 2 else first

    x_out = euler_sample(model_forward, pos_0, movable, pad, cfg, num_steps=2, refine_final=True)
    assert torch.allclose(x_out, final)


def test_return_trajectory_uses_absolute_key():
    pos_0 = torch.zeros(2, 4, 3)
    movable = torch.ones(2, 4, dtype=torch.bool)
    pad = torch.ones(2, 4, dtype=torch.bool)
    cfg = FlowConfig(eps=1e-5)

    def oracle(x_t, t):  # noqa: ARG001
        return torch.zeros_like(x_t)

    out = euler_sample(oracle, pos_0, movable, pad, cfg, num_steps=8, return_trajectory=True)
    assert out["x_out"].shape == pos_0.shape
    assert out["x_trajectory"].shape == (9, 2, 4, 3)


def test_fk_steering_requires_divisible_batch():
    pos_0 = torch.zeros(3, 4, 3)
    movable = torch.ones(3, 4, dtype=torch.bool)
    pad = torch.ones(3, 4, dtype=torch.bool)
    cfg = FlowConfig(eps=1e-5)

    def oracle(x_t, t):  # noqa: ARG001
        return torch.zeros_like(x_t)

    def energy_fn(x, pm, mm):  # noqa: ARG001
        return torch.zeros(x.shape[0])

    fk = FKSteeringConfig(num_particles=2, energy_fn=energy_fn)
    try:
        euler_sample(oracle, pos_0, movable, pad, cfg, num_steps=2, fk_steering=fk)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected FK batch divisibility error")


def test_cfg_model_forward_combines_predictions():
    x_t = torch.zeros(1, 2, 3)
    t = torch.tensor([0.5])

    def cond(x, tt):  # noqa: ARG001
        return torch.ones_like(x)

    def uncond(x, tt):  # noqa: ARG001
        return torch.full_like(x, 0.25)

    guided = cfg_model_forward(cond, uncond, w=2.0)
    assert torch.allclose(guided(x_t, t), torch.full_like(x_t, 2.5))
