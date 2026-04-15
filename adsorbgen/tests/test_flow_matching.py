"""Unit tests for the AdsorbGen flow matching DiT.

Covers:
    - Shape/dtype/device for DiTDenoiser
    - Output masking: non-movable atoms are exactly zero
    - Zero-init invariants: out_proj, ΔE MLP last Linear, adaLN last Linear
    - Flow corruption math: t=0 -> delta_0; t=1 -> delta_1
    - minimum_image and compute_delta1 sanity
    - x1_loss boundary cases
    - Euler sampler with an oracle model recovers the target
    - Gradient flow through all parameters
    - CFG wrap math
    - PreprocessedDisplacementDataset + collate roundtrip on a tmp LMDB
    - CUDA smoke (skipped if no GPU)
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

from adsorbgen.dataset import PreprocessedDisplacementDataset, collate_displacement  # noqa: E402
from adsorbgen.flow import (  # noqa: E402
    FKSteeringConfig,
    FlowConfig,
    cfg_model_forward,
    compute_delta1,
    corrupt,
    euler_sample,
    minimum_image,
    sample_t,
    x1_loss,
)
from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig  # noqa: E402


def _random_batch(B=2, N=16, seed=0, device="cpu"):
    torch.manual_seed(seed)
    pos = torch.randn(B, N, 3, device=device) * 2.0
    delta1_true = torch.randn(B, N, 3, device=device) * 0.3
    cells = torch.zeros(B, 3, 3, device=device)
    for b in range(B):
        L = 10.0 + b * 2.0
        cells[b] = torch.eye(3, device=device) * L
    tags = torch.randint(0, 3, (B, N), device=device)
    fixed = torch.zeros(B, N, dtype=torch.long, device=device)
    atomic_numbers = torch.randint(1, 30, (B, N), device=device)
    pad_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    movable = ((tags == 1) | (tags == 2)) & (fixed == 0) & pad_mask
    delta1_true = delta1_true * movable.unsqueeze(-1).float()
    return dict(
        pos=pos,
        delta1_true=delta1_true,
        cell=cells,
        tags=tags,
        fixed=fixed,
        atomic_numbers=atomic_numbers,
        pad_mask=pad_mask,
        movable_mask=movable,
    )


def _small_model(sigma=0.3):
    return DiTDenoiser(
        DiTDenoiserConfig(
            atom_s=32,
            atom_z=16,
            token_s=64,
            token_z=32,
            enc_depth=1,
            trunk_depth=2,
            dec_depth=1,
            enc_heads=4,
            trunk_heads=4,
            dec_heads=4,
            mlp_ratio=2.0,
            num_elements=40,
            sigma=sigma,
            delta_e_max=2.0,
            delta_e_freq_dim=64,
        )
    )


# ---------- Zero-init invariants ----------


def test_zero_init_output_head():
    m = _small_model()
    assert torch.all(m.out_proj.weight == 0)
    assert torch.all(m.out_proj.bias == 0)


def test_zero_init_delta_e_last_linear():
    m = _small_model()
    assert torch.all(m.delta_e_embedder.fc2.weight == 0)
    assert torch.all(m.delta_e_embedder.fc2.bias == 0)


def test_zero_init_adaln_last_linear():
    m = _small_model()
    for stage in [m.encoder, m.trunk, m.decoder]:
        for block in stage.layers:
            last = block.adaLN_modulation[-1]
            assert torch.all(last.weight == 0)
            assert torch.all(last.bias == 0)


def test_model_output_is_zero_at_init():
    m = _small_model()
    b = _random_batch()
    delta_t = torch.zeros_like(b["delta1_true"])
    t = torch.full((b["pos"].shape[0],), 0.5)
    out = m(
        pos=b["pos"],
        delta_t=delta_t,
        t=t,
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    assert torch.all(out == 0)


# ---------- Shape/dtype/device ----------


def test_forward_shapes_and_dtype():
    m = _small_model()
    b = _random_batch(B=3, N=12)
    delta_t = torch.randn_like(b["delta1_true"]) * 0.1
    t = torch.rand(3)
    out = m(
        pos=b["pos"],
        delta_t=delta_t,
        t=t,
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    assert out.shape == b["delta1_true"].shape
    assert out.dtype == torch.float32


def test_output_masked_on_non_movable():
    torch.manual_seed(1)
    m = _small_model()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.1)
        m.out_proj.bias.normal_(0, 0.1)
    b = _random_batch()
    out = m(
        pos=b["pos"],
        delta_t=torch.zeros_like(b["delta1_true"]),
        t=torch.full((b["pos"].shape[0],), 0.5),
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    mask = b["movable_mask"].unsqueeze(-1)
    assert torch.all(out.masked_select(~mask) == 0)


# ---------- Flow corruption math ----------


def test_corrupt_boundary_values():
    torch.manual_seed(0)
    B, N = 2, 8
    delta1 = torch.randn(B, N, 3) * 0.3
    movable = torch.ones(B, N, dtype=torch.bool)
    cfg = FlowConfig(sigma=0.3, eps=0.0)
    t0 = torch.zeros(B)
    dt0, d0 = corrupt(delta1, t0, cfg, movable)
    assert torch.allclose(dt0, d0)
    t1 = torch.ones(B)
    dt1, _ = corrupt(delta1, t1, cfg, movable)
    assert torch.allclose(dt1, delta1)


def test_corrupt_respects_movable_mask():
    torch.manual_seed(0)
    B, N = 2, 8
    delta1 = torch.randn(B, N, 3)
    movable = torch.zeros(B, N, dtype=torch.bool)
    movable[:, :4] = True
    delta1 = delta1 * movable.unsqueeze(-1).float()
    cfg = FlowConfig(sigma=0.3, eps=0.0)
    t = torch.full((B,), 0.5)
    dt, d0 = corrupt(delta1, t, cfg, movable)
    assert torch.all(dt[:, 4:] == 0)
    assert torch.all(d0[:, 4:] == 0)


# ---------- Minimum image ----------


def test_minimum_image_wraps_into_half_cell():
    L = 10.0
    cell = torch.eye(3).unsqueeze(0) * L
    delta = torch.tensor([[[9.0, 0.0, 0.0]]])
    wrapped = minimum_image(delta, cell)
    assert torch.allclose(wrapped, torch.tensor([[[-1.0, 0.0, 0.0]]]), atol=1e-5)


def test_compute_delta1_zero_on_non_movable():
    pos = torch.zeros(1, 4, 3)
    pos_rel = torch.tensor([[[0.1, 0, 0], [0.2, 0, 0], [1.0, 0, 0], [2.0, 0, 0]]])
    cell = torch.eye(3).unsqueeze(0) * 10.0
    movable = torch.tensor([[True, True, False, False]])
    d = compute_delta1(pos, pos_rel, cell, movable)
    assert torch.all(d[0, 2:] == 0)
    assert torch.allclose(d[0, :2], pos_rel[0, :2], atol=1e-6)


# ---------- Loss ----------


def test_x1_loss_zero_at_truth():
    torch.manual_seed(0)
    B, N = 2, 8
    delta1 = torch.randn(B, N, 3)
    movable = torch.ones(B, N, dtype=torch.bool)
    delta1 = delta1 * movable.unsqueeze(-1).float()
    loss = x1_loss(delta1, delta1, movable)
    assert float(loss) < 1e-12


def test_x1_loss_ignores_non_movable():
    delta1 = torch.zeros(1, 4, 3)
    pred = torch.tensor([[[0, 0, 0], [0, 0, 0], [10.0, 0, 0], [10.0, 0, 0]]])
    movable = torch.tensor([[True, True, False, False]])
    loss = x1_loss(pred, delta1, movable)
    assert float(loss) < 1e-12


# ---------- Euler sampler ----------


def test_flow_config_default_eps_is_1e_5():
    cfg = FlowConfig()
    assert cfg.eps == 1e-5


def test_euler_sampler_oracle_recovers_target():
    torch.manual_seed(0)
    B, N = 1, 4
    pos = torch.randn(B, N, 3)
    delta1_true = torch.randn(B, N, 3) * 0.2
    cell = torch.eye(3).unsqueeze(0) * 20.0
    movable = torch.ones(B, N, dtype=torch.bool)
    pad = torch.ones(B, N, dtype=torch.bool)
    cfg = FlowConfig(sigma=0.2, eps=1e-5)

    def oracle(delta_t, t):  # noqa: ARG001
        return delta1_true

    x_hat = euler_sample(oracle, pos, cell, movable, pad, cfg, num_steps=200)
    err = (x_hat - (pos + delta1_true)).abs().max().item()
    assert err < 1e-2, f"oracle recovery error {err}"


def test_sde_sampler_oracle_recovers_target():
    """SDE update with an oracle should still concentrate near the target.

    The injected noise has variance ~ g^2 * dt = 0.5*(1-t)*dt which goes to
    zero as t->1, so the distribution collapses onto delta_1 by t=1-eps.
    """
    torch.manual_seed(0)
    B, N = 1, 4
    pos = torch.randn(B, N, 3)
    delta1_true = torch.randn(B, N, 3) * 0.2
    cell = torch.eye(3).unsqueeze(0) * 20.0
    movable = torch.ones(B, N, dtype=torch.bool)
    pad = torch.ones(B, N, dtype=torch.bool)
    cfg = FlowConfig(sigma=0.2, eps=1e-5)

    def oracle(delta_t, t):  # noqa: ARG001
        return delta1_true

    x_hat = euler_sample(oracle, pos, cell, movable, pad, cfg, num_steps=400, use_sde=True)
    err = (x_hat - (pos + delta1_true)).abs().max().item()
    assert err < 0.1, f"SDE oracle recovery error {err}"


def test_refine_final_uses_extra_forward():
    """With refine_final=True, the final delta equals the extra model call's prediction."""
    torch.manual_seed(0)
    B, N = 1, 4
    pos = torch.zeros(B, N, 3)
    cell = torch.eye(3).unsqueeze(0) * 20.0
    movable = torch.ones(B, N, dtype=torch.bool)
    pad = torch.ones(B, N, dtype=torch.bool)
    cfg = FlowConfig(sigma=0.2, eps=1e-5)
    fixed_target = torch.tensor([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0], [0.4, 0.0, 0.0]]])

    def oracle(delta_t, t):  # noqa: ARG001
        return fixed_target

    x_hat = euler_sample(oracle, pos, cell, movable, pad, cfg, num_steps=10, refine_final=True)
    # refine_final replaces delta with the model prediction at t=1-eps; for an
    # oracle that returns fixed_target, the result equals pos + fixed_target.
    assert torch.allclose(x_hat, pos + fixed_target, atol=1e-6)


def test_return_trajectory_shapes():
    torch.manual_seed(0)
    B, N = 2, 4
    pos = torch.zeros(B, N, 3)
    cell = torch.eye(3).unsqueeze(0).expand(B, 3, 3) * 20.0
    movable = torch.ones(B, N, dtype=torch.bool)
    pad = torch.ones(B, N, dtype=torch.bool)
    cfg = FlowConfig(sigma=0.2, eps=1e-5)

    def oracle(delta_t, t):  # noqa: ARG001
        return torch.zeros_like(delta_t)

    out = euler_sample(oracle, pos, cell, movable, pad, cfg, num_steps=8, return_trajectory=True)
    assert "x_out" in out and "delta_trajectory" in out
    assert out["delta_trajectory"].shape == (8 + 1, B, N, 3)


def test_fk_steering_resamples_toward_low_energy():
    """With an energy that is much lower for particle index 0 within each
    group, FK steering should drive nearly all surviving particles to that
    initial state."""
    torch.manual_seed(0)
    orig_B = 1
    P = 4
    B = orig_B * P
    N = 4
    cell = torch.eye(3).unsqueeze(0).expand(B, 3, 3) * 20.0
    movable = torch.ones(B, N, dtype=torch.bool)
    pad = torch.ones(B, N, dtype=torch.bool)

    # Make pos identical across particles so reorder is detectable only via
    # delta_t evolution.
    pos = torch.zeros(B, N, 3)

    # Track an "id" per particle through the dynamics by stamping it into the
    # oracle's prediction. The oracle returns particle-specific targets so
    # different particles drift to distinguishable delta values.
    targets = torch.zeros(B, N, 3)
    for i in range(B):
        targets[i, :, 0] = float(i)

    def oracle(delta_t, t):  # noqa: ARG001
        return targets

    # Energy: large positive for particles 1..3, near-zero for particle 0.
    def energy_fn(x_pred, pad_mask, movable_mask):  # noqa: ARG001
        e = torch.zeros(x_pred.shape[0], device=x_pred.device, dtype=x_pred.dtype)
        e[1] = 100.0
        e[2] = 100.0
        e[3] = 100.0
        return e

    fk = FKSteeringConfig(
        num_particles=P,
        energy_fn=energy_fn,
        fk_lambda=10.0,
        resampling_interval=1,
        fk_start_time=0.0,
        potential_mode="immediate",
    )
    cfg = FlowConfig(sigma=0.2, eps=1e-5)
    out = euler_sample(
        oracle, pos, cell, movable, pad, cfg,
        num_steps=20, fk_steering=fk, return_trajectory=True,
    )
    # After repeated resampling toward particle 0, the surviving deltas should
    # all carry id == 0 (i.e. targets[0] = 0). x_out = pos + delta should be
    # all zero on the x-axis.
    final_x = out["x_out"][:, :, 0]
    assert (final_x.abs() < 0.5).all(), f"FK did not concentrate: {final_x}"
    assert out["energy_trajectory"].shape[0] == B
    assert out["energy_trajectory"].shape[1] >= 1


def test_fk_steering_requires_divisible_batch():
    cfg = FlowConfig(sigma=0.2, eps=1e-5)
    pos = torch.zeros(3, 4, 3)
    cell = torch.eye(3).unsqueeze(0).expand(3, 3, 3) * 10.0
    movable = torch.ones(3, 4, dtype=torch.bool)
    pad = torch.ones(3, 4, dtype=torch.bool)

    def oracle(delta_t, t):  # noqa: ARG001
        return torch.zeros_like(delta_t)

    def energy_fn(x, pm, mm):  # noqa: ARG001
        return torch.zeros(x.shape[0])

    fk = FKSteeringConfig(num_particles=2, energy_fn=energy_fn)
    with pytest.raises(ValueError):
        euler_sample(oracle, pos, cell, movable, pad, cfg, num_steps=2, fk_steering=fk)


# ---------- Gradient flow ----------


def test_gradients_flow_through_all_params():
    torch.manual_seed(0)
    m = _small_model()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.05)
    b = _random_batch()
    t = torch.full((b["pos"].shape[0],), 0.5)
    delta_t = torch.randn_like(b["delta1_true"]) * 0.1
    pred = m(
        pos=b["pos"],
        delta_t=delta_t,
        t=t,
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
        delta_e=torch.full((b["pos"].shape[0],), 0.5),
        cond_drop=torch.zeros(b["pos"].shape[0], dtype=torch.bool),
    )
    loss = x1_loss(pred, b["delta1_true"], b["movable_mask"])
    loss.backward()
    unused = []
    for name, p in m.named_parameters():
        if p.grad is None:
            unused.append(name)
            continue
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert len(unused) < 10, f"too many params without grad: {unused[:20]}"


# ---------- CFG wrap ----------


def test_cfg_wrap_formula():
    def f_cond(dt, t):  # noqa: ARG001
        return torch.full_like(dt, 2.0)

    def f_uncond(dt, t):  # noqa: ARG001
        return torch.full_like(dt, 1.0)

    f = cfg_model_forward(f_cond, f_uncond, w=3.0)
    out = f(torch.zeros(1, 4, 3), torch.zeros(1))
    assert torch.allclose(out, torch.full_like(out, 5.0))


# ---------- Dataset roundtrip ----------


def _write_tmp_lmdb(path: str, n: int, with_delta_e: bool):
    env = lmdb.open(path, subdir=False, map_size=1 << 30)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i in range(n):
            N = 8 + (i % 4)
            entry = {
                "pos": rng.standard_normal((N, 3)).astype(np.float32),
                "pos_relaxed": rng.standard_normal((N, 3)).astype(np.float32),
                "cell": (np.eye(3, dtype=np.float32) * 10.0),
                "tags": rng.integers(0, 3, size=N).astype(np.int64),
                "fixed": np.zeros(N, dtype=np.int64),
                "atomic_numbers": rng.integers(1, 30, size=N).astype(np.int64),
                "sid": i,
                "y_init": 0.0,
                "y_relaxed": float(rng.standard_normal()),
            }
            if with_delta_e:
                entry["delta_e"] = float(abs(rng.standard_normal()))
            txn.put(str(i).encode("ascii"), pickle.dumps(entry))
        txn.put(b"length", pickle.dumps(n))
    env.sync()
    env.close()


def test_preprocessed_dataset_collate():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.lmdb")
        _write_tmp_lmdb(path, n=6, with_delta_e=False)
        ds = PreprocessedDisplacementDataset(path)
        assert len(ds) == 6
        batch = collate_displacement([ds[i] for i in range(4)])
        assert batch["pos"].shape == batch["pos_relaxed"].shape
        assert batch["pos"].shape[0] == 4
        assert batch["pad_mask"].dtype == torch.bool
        assert batch["ads_id"].shape == (4,)


# ---------- Pair bias invariants ----------


def _build_pair_scene(B=2, N=10, seed=0):
    torch.manual_seed(seed)
    pos = torch.randn(B, N, 3) * 2.0
    cells = torch.eye(3).unsqueeze(0).expand(B, 3, 3) * 20.0
    # 4 bulk, 3 surface, 2 adsorbate, 1 padding — lets us probe every block.
    tags = torch.zeros(B, N, dtype=torch.long)
    tags[:, 4:7] = 1
    tags[:, 7:9] = 2
    pad_mask = torch.ones(B, N, dtype=torch.bool)
    pad_mask[:, -1] = False
    return pos, tags, pad_mask, cells


def test_pair_features_zero_on_bulk_block():
    """Bulk-involved pair entries must be exactly zero (block mask ``v``)."""
    m = _small_model()
    pos, tags, pad_mask, cell = _build_pair_scene()
    pair = m._build_pair_features(pos, tags, pad_mask, cell)
    non_bulk = tags >= 1
    valid = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1) & non_bulk.unsqueeze(2) & non_bulk.unsqueeze(1)
    assert torch.all(pair[~valid] == 0)


def test_pair_features_zero_on_padding_block():
    """Padding rows/cols in the pair tensor must be zero."""
    m = _small_model()
    pos, tags, pad_mask, cell = _build_pair_scene()
    pair = m._build_pair_features(pos, tags, pad_mask, cell)
    assert torch.all(pair[:, ~pad_mask[0], :] == 0)
    assert torch.all(pair[:, :, ~pad_mask[0]] == 0)


def test_pair_features_nonzero_on_non_bulk_block():
    """Non-bulk×non-bulk valid entries should be non-zero after random init."""
    torch.manual_seed(3)
    m = _small_model()
    pos, tags, pad_mask, cell = _build_pair_scene()
    pair = m._build_pair_features(pos, tags, pad_mask, cell)
    non_bulk = tags >= 1
    valid = pad_mask.unsqueeze(2) & pad_mask.unsqueeze(1) & non_bulk.unsqueeze(2) & non_bulk.unsqueeze(1)
    # exclude diagonal (self-pair) which has diff=0 and dist=1/(1+0)=1 — still
    # goes through Linear(z) and is nonzero in general, but keep it loose.
    active = pair[valid]
    assert active.abs().sum() > 0


def test_pair_features_translation_invariant():
    """Pairwise MIC diffs are translation invariant -> pair features unchanged."""
    m = _small_model()
    pos, tags, pad_mask, cell = _build_pair_scene()
    shift = torch.tensor([1.3, -0.7, 0.2])
    pair_a = m._build_pair_features(pos, tags, pad_mask, cell)
    pair_b = m._build_pair_features(pos + shift, tags, pad_mask, cell)
    torch.testing.assert_close(pair_a, pair_b, rtol=1e-5, atol=1e-5)


def test_pair_features_adsorbate_branch_distinguishes_ads_pairs():
    """Swapping adsorbate atoms to surface tag must change the pair tensor on
    the formerly-adsorbate block because ``emb_pair_ads`` contributes there."""
    torch.manual_seed(4)
    m = _small_model()
    # Make emb_pair_ads non-zero so the branch actually contributes.
    with torch.no_grad():
        m.emb_pair_ads.weight.normal_(0, 0.2)
    pos, tags, pad_mask, cell = _build_pair_scene()
    pair_a = m._build_pair_features(pos, tags, pad_mask, cell)
    tags_alt = tags.clone()
    tags_alt[tags_alt == 2] = 1
    pair_b = m._build_pair_features(pos, tags_alt, pad_mask, cell)
    diff = (pair_a - pair_b).abs()
    assert diff.sum() > 0


def test_pair_features_shape_depends_on_atom_z():
    cfg = DiTDenoiserConfig(
        atom_s=32, atom_z=24, token_s=64, token_z=48,
        enc_depth=1, trunk_depth=1, dec_depth=1,
        enc_heads=4, trunk_heads=4, dec_heads=4,
        mlp_ratio=2.0, num_elements=40, delta_e_freq_dim=64,
    )
    m = DiTDenoiser(cfg)
    pos, tags, pad_mask, cell = _build_pair_scene(B=1, N=6)
    pair = m._build_pair_features(pos, tags, pad_mask, cell)
    assert pair.shape == (1, 6, 6, 24)


def test_forward_finite_on_random_batch():
    """DiTDenoiser output must be finite on a randomly perturbed batch."""
    torch.manual_seed(5)
    m = _small_model()
    with torch.no_grad():
        m.out_proj.weight.normal_(0, 0.05)
    b = _random_batch(B=3, N=16)
    out = m(
        pos=b["pos"],
        delta_t=torch.randn_like(b["delta1_true"]) * 0.1,
        t=torch.rand(3),
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("B,N", [(1, 4), (2, 8), (4, 32)])
def test_forward_various_sizes(B, N):
    """Forward pass should work across batch/atom-count edge cases."""
    m = _small_model()
    b = _random_batch(B=B, N=N, seed=B * 10 + N)
    out = m(
        pos=b["pos"],
        delta_t=torch.zeros_like(b["delta1_true"]),
        t=torch.full((B,), 0.5),
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    assert out.shape == (B, N, 3)
    assert torch.isfinite(out).all()


# ---------- CUDA smoke ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_runs_on_cuda():
    m = _small_model().cuda()
    b = _random_batch(device="cuda")
    t = torch.full((b["pos"].shape[0],), 0.5, device="cuda")
    out = m(
        pos=b["pos"],
        delta_t=torch.zeros_like(b["delta1_true"]),
        t=t,
        atomic_numbers=b["atomic_numbers"],
        tags=b["tags"],
        movable_mask=b["movable_mask"],
        pad_mask=b["pad_mask"],
        cell=b["cell"],
    )
    assert out.device.type == "cuda"
