"""Euclidean flow matching in displacement space (x1-prediction).

Formulation:
    delta_1  = minimum_image(x_relax - x_ref, cell)  on movable atoms (0 elsewhere)
    delta_0  ~ N(0, sigma^2 I)
    delta_t  = (1 - t) * delta_0 + t * delta_1             (linear path)
    Loss     = || f_theta(delta_t, t, x_ref) - delta_1 ||^2 on movable atoms only
    ODE step: delta_t += dt * (f_theta - delta_t) / (1 - t)
    SDE step: delta_t += dt * (v + 0.5*g^2*score) + sqrt(g^2*dt) * noise,
              with v = (f_theta - delta_t)/(1 - t),
                   score = (t*v - delta_t)/(1 - t),
                   g^2(t) = 0.5*(1 - t).
    Output   = x_ref + delta_{t=1-eps} on movable atoms, x_ref elsewhere

Optional sampler features (ported from AtomMOF):
    - SDE coordinate update (`use_sde=True`)
    - Final refinement step (`refine_final=True`): one extra forward at t=1-eps
    - Feynman-Kac steering (`fk_steering=...`): particle resampling with an
      energy callback during sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Non-finite values in {name}: shape={tuple(t.shape)}")


def minimum_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Apply minimum-image convention to a Cartesian displacement."""
    _assert_finite(delta, "delta")
    _assert_finite(cell, "cell")
    cell_inv = torch.linalg.inv(cell)
    frac = torch.einsum("bnj,bjk->bnk", delta, cell_inv)
    frac = frac - torch.round(frac)
    return torch.einsum("bnj,bjk->bnk", frac, cell)


def compute_delta1(
    pos: torch.Tensor,
    pos_relaxed: torch.Tensor,
    cell: torch.Tensor,
    movable_mask: torch.Tensor,
) -> torch.Tensor:
    d = minimum_image(pos_relaxed - pos, cell)
    return d * movable_mask.unsqueeze(-1).to(d.dtype)


@dataclass
class FlowConfig:
    sigma: float = 0.5
    eps: float = 1e-5  # avoid 1/(1-t) singularity at t=1 (and exact-zero noise at t=0)


def sample_t(batch_size: int, cfg: FlowConfig, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    u = torch.rand(batch_size, device=device, dtype=dtype)
    return cfg.eps + (1 - 2 * cfg.eps) * u


def sample_delta0(shape: Tuple[int, ...], cfg: FlowConfig, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*shape, device=device, dtype=dtype) * cfg.sigma


def corrupt(
    delta1: torch.Tensor,
    t: torch.Tensor,
    cfg: FlowConfig,
    movable_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _assert_finite(delta1, "delta1")
    B, N, _ = delta1.shape
    delta0 = sample_delta0((B, N, 3), cfg, device=delta1.device, dtype=delta1.dtype)
    delta0 = delta0 * movable_mask.unsqueeze(-1).to(delta0.dtype)
    t_b = t.view(B, 1, 1).to(delta1.dtype)
    delta_t = (1 - t_b) * delta0 + t_b * delta1
    return delta_t, delta0


def x1_loss(
    pred: torch.Tensor,
    delta1: torch.Tensor,
    movable_mask: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """Loss on movable atoms, averaged within sample then across batch.

    Args:
        loss_type: ``"l2"`` (MSE, default) or ``"l1"`` (MAE).
    """
    _assert_finite(pred, "pred")
    diff = pred - delta1
    if loss_type == "l1":
        per_atom = diff.abs().sum(dim=-1)
    else:
        per_atom = diff.pow(2).sum(dim=-1)
    mask = movable_mask.to(per_atom.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sample = (per_atom * mask).sum(dim=1) / denom
    return per_sample.mean()


def _score_from_velocity(
    v: torch.Tensor,
    delta_t: torch.Tensor,
    t_scalar: float,
    eps: float,
) -> torch.Tensor:
    """Score of the marginal under the linear-path flow.

    For delta_t = (1-t)*delta_0 + t*delta_1 with delta_0 ~ N(0, sigma^2 I),
    one can show s(delta_t, t) = (t*v - delta_t) / (1 - t), where
    v = E[(delta_1 - delta_t) / (1 - t)] is the conditional flow velocity.

    Args:
        v:        (B, N, 3) flow velocity (already masked to movable atoms).
        delta_t:  (B, N, 3) current state.
        t_scalar: scalar timestep in [0, 1).
        eps:      regularizer for 1/(1 - t).

    Returns:
        (B, N, 3) score.
    """
    return (t_scalar * v - delta_t) / max(1.0 - float(t_scalar), eps)


@dataclass
class FKSteeringConfig:
    """Feynman-Kac particle steering during sampling.

    The caller is responsible for replicating the static context (`pos`, `cell`,
    `tags`, ...) by `num_particles` along the batch dim BEFORE invoking the
    sampler. Resampling reorders within each particle group, so the static
    context is unaffected.

    Args:
        num_particles: number of particles per original sample. The batch fed
            to `euler_sample` must already have shape (orig_B * num_particles,
            N, 3) — the sampler does not expand it for you.
        energy_fn: callable mapping (x_pred, pad_mask, movable_mask) ->
            (B,) energy tensor on the *predicted final* coordinates
            x_pred = pos + pred_delta1. Lower is better.
        fk_lambda: temperature for softmax on log weights.
        resampling_interval: only resample every K Euler steps.
        fk_start_time: only resample once t >= this value.
        potential_mode: "immediate" | "difference" | "max" | "sum"
            (matches AtomMOF semantics).
    """

    num_particles: int
    energy_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    fk_lambda: float = 10.0
    resampling_interval: int = 1
    fk_start_time: float = 0.0
    potential_mode: str = "difference"


def _fk_log_weights(energy_traj: torch.Tensor, mode: str) -> torch.Tensor:
    """log G_t selector matching AtomMOF.sample()."""
    cur = energy_traj[:, -1]
    if mode == "immediate":
        return -cur
    if mode == "difference":
        if energy_traj.shape[1] == 1:
            return torch.zeros_like(cur)
        return energy_traj[:, -2] - cur
    if mode == "max":
        return -energy_traj.min(dim=1).values
    if mode == "sum":
        return -energy_traj.mean(dim=1)
    raise ValueError(f"Unsupported potential_mode: {mode}")


@torch.no_grad()
def euler_sample(
    model_forward: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pos: torch.Tensor,
    cell: torch.Tensor,
    movable_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    cfg: FlowConfig,
    num_steps: int = 50,
    use_sde: bool = False,
    refine_final: bool = False,
    return_trajectory: bool = False,
    fk_steering: Optional[FKSteeringConfig] = None,
):
    """Euler integrator for the x1-prediction flow in displacement space.

    Args:
        model_forward: callable (delta_t, t) -> pred_delta1, closing over the
            static context (pos, tags, ..., cell).
        pos: (B, N, 3) Cartesian x_ref. Already replicated by num_particles if
            FK steering is enabled.
        cell, movable_mask, pad_mask: same batch dim as pos.
        cfg: FlowConfig (sigma, eps).
        num_steps: number of Euler steps.
        use_sde: if True, run the SDE update with g^2(t) = 0.5*(1-t).
        refine_final: if True, perform one extra forward at t = 1 - eps and
            use that prediction as the final delta_1.
        return_trajectory: if True, also return the (num_steps+1+, B, N, 3)
            stacked delta_t trajectory.
        fk_steering: optional FKSteeringConfig (caller pre-replicated batch).

    Returns:
        x_out: (B, N, 3) final Cartesian coordinates = pos + delta_final on
               movable atoms, pos elsewhere; padding atoms zeroed.
        If return_trajectory: dict with keys
            "x_out", "delta_trajectory" (T, B, N, 3),
            "energy_trajectory" (B, history) if FK steering was active.
    """
    device = pos.device
    dtype = pos.dtype
    B, N, _ = pos.shape
    movable_f = movable_mask.unsqueeze(-1).to(dtype)
    pad_f = pad_mask.unsqueeze(-1).to(dtype)

    # Initialize delta_0 ~ N(0, sigma^2 I) on movable atoms.
    delta_t = sample_delta0((B, N, 3), cfg, device=device, dtype=dtype)
    delta_t = delta_t * movable_f

    t_vals = torch.linspace(cfg.eps, 1.0 - cfg.eps, num_steps + 1, device=device, dtype=dtype)

    traj: List[torch.Tensor] = [delta_t.clone()] if return_trajectory else []
    energy_traj: Optional[torch.Tensor] = None
    if fk_steering is not None:
        energy_traj = torch.empty((B, 0), device=device, dtype=dtype)
        if B % fk_steering.num_particles != 0:
            raise ValueError(
                f"Batch size {B} must be divisible by num_particles {fk_steering.num_particles}"
            )

    for i in range(num_steps):
        t_scalar = float(t_vals[i].item())
        dt = float((t_vals[i + 1] - t_vals[i]).item())
        t = t_vals[i].expand(B)

        pred_delta1 = model_forward(delta_t, t)
        pred_delta1 = pred_delta1 * movable_f

        # FK steering: evaluate energy of predicted final coords and resample.
        if fk_steering is not None and t_scalar >= fk_steering.fk_start_time and (i % fk_steering.resampling_interval == 0):
            x_pred = pos + pred_delta1
            current_energy = fk_steering.energy_fn(x_pred, pad_mask, movable_mask).to(dtype)
            energy_traj = torch.cat([energy_traj, current_energy.unsqueeze(1)], dim=1)

            log_G = _fk_log_weights(energy_traj, fk_steering.potential_mode)
            P = fk_steering.num_particles
            log_G = log_G.reshape(-1, P)
            weights = F.softmax(log_G * fk_steering.fk_lambda, dim=1)
            sampled = torch.multinomial(weights, P, replacement=True)  # (G, P)
            offset = torch.arange(weights.shape[0], device=device).unsqueeze(1) * P
            idx = (sampled + offset).flatten()

            delta_t = delta_t[idx]
            pred_delta1 = pred_delta1[idx]
            energy_traj = energy_traj[idx]
            # Static context (pos/cell/tags/...) is identical within each
            # particle group, so reordering inside a group is a no-op for it.

        v = (pred_delta1 - delta_t) / max(1.0 - t_scalar, cfg.eps)
        v = v * movable_f

        if use_sde:
            g2 = 0.5 * (1.0 - t_scalar)
            score = _score_from_velocity(v, delta_t, t_scalar, cfg.eps) * movable_f
            drift = v + 0.5 * g2 * score
            noise = torch.randn_like(delta_t) * movable_f
            delta_t = delta_t + drift * dt + (g2 * dt) ** 0.5 * noise
        else:
            delta_t = delta_t + v * dt

        delta_t = delta_t * movable_f

        if return_trajectory:
            traj.append(delta_t.clone())

    if refine_final:
        t_final = t_vals[-1].expand(B)
        pred_final = model_forward(delta_t, t_final) * movable_f
        delta_final = pred_final
    else:
        delta_final = delta_t

    x_out = (pos + delta_final * movable_f) * pad_f

    if not return_trajectory:
        return x_out

    out = {
        "x_out": x_out,
        "delta_trajectory": torch.stack(traj, dim=0) if traj else None,
    }
    if energy_traj is not None:
        out["energy_trajectory"] = energy_traj
    return out


def cfg_model_forward(
    f_cond: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_uncond: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    w: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """f_hat = (1 + w) * f_cond - w * f_uncond."""

    def _f(delta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fc = f_cond(delta_t, t)
        fu = f_uncond(delta_t, t)
        return (1.0 + w) * fc - w * fu

    return _f
