"""Flow matching on absolute coordinates (AtomMOF-style).

Formulation:
    x_0   = [surface: LMDB pos_init, ads: fairchem placement, bulk: pos_init]
    x_1   = pos_relaxed (LMDB)
    x_t   = (1 - t) * x_0 + t * x_1        (absolute coord interp)
    Model: x_1_hat = f_theta(x_0, x_t, t)    (direct x_1 prediction)
    Loss  = || x_1_hat - x_1 ||  on movable atoms only
    ODE step: x_t += dt * (x_1_hat - x_t) / (1 - t)
    SDE step: x_t += dt * (v + 0.5 g^2 s) + sqrt(g^2 dt) * noise,
              with v = (x_1_hat - x_t)/(1-t),
                   s = (t*v - (x_t - x_0)) / (1-t)  [approx. retained from prior formulation]
                   g^2(t) = 0.5*(1-t).

MIC is used only inside pair-feature construction in model.py to compute
nearest-image distances; it is NEVER used in the loss or interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from ase.data import covalent_radii


_COVALENT_RADII = torch.tensor(covalent_radii, dtype=torch.float32)


def _assert_finite(t: torch.Tensor, name: str) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"Non-finite values in {name}: shape={tuple(t.shape)}")


def minimum_image(delta: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Apply minimum-image convention to a Cartesian displacement.

    Used by pair feature construction in model.py. Never in loss.
    """
    _assert_finite(delta, "delta")
    _assert_finite(cell, "cell")
    cell_inv = torch.linalg.inv(cell)
    frac = torch.einsum("bnj,bjk->bnk", delta, cell_inv)
    frac = frac - torch.round(frac)
    return torch.einsum("bnj,bjk->bnk", frac, cell)


@dataclass
class FlowConfig:
    eps: float = 1e-5  # avoid 1/(1-t) singularity
    prediction_type: str = "x1"  # "x1" -> model predicts x_1; "v" -> model predicts v = x_1 - x_0


def si_gamma(
    t: torch.Tensor,
    schedule: str,
    sigma: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Latent SI noise scale gamma(t).

    Mirrors OMatG's ``LatentGammaSqrt`` for ``sqrt_t1mt`` and keeps the
    earlier AdsorbGen ``linear_1mt`` schedule for ablations.
    """
    if schedule == "sqrt_t1mt":
        tt = (t * (1.0 - t)).clamp_min(float(eps))
        return float(sigma) * torch.sqrt(tt)
    if schedule == "linear_1mt":
        return float(sigma) * (1.0 - t)
    if schedule == "none" or float(sigma) == 0.0:
        return torch.zeros_like(t)
    raise ValueError(f"Unknown gamma_schedule={schedule!r}")


def si_gamma_dot(
    t: torch.Tensor,
    schedule: str,
    sigma: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Time derivative of gamma(t), used in the OMatG SI velocity target."""
    if schedule == "sqrt_t1mt":
        tt = (t * (1.0 - t)).clamp_min(float(eps))
        return float(sigma) * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(tt))
    if schedule == "linear_1mt":
        return torch.full_like(t, -float(sigma))
    if schedule == "none" or float(sigma) == 0.0:
        return torch.zeros_like(t)
    raise ValueError(f"Unknown gamma_schedule={schedule!r}")


def si_epsilon(
    t: torch.Tensor,
    schedule: str,
    scale: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """SDE epsilon(t) used by OMatG's drift b - epsilon/gamma * eta."""
    if schedule == "constant":
        return torch.full_like(t, float(scale))
    if schedule == "vanishing_1mt":
        return float(scale) * (1.0 - t).clamp_min(float(eps))
    if schedule == "zero":
        return torch.zeros_like(t)
    raise ValueError(f"Unknown si_epsilon_schedule={schedule!r}")


def _target_for_loss(prediction_type: str, pos_0: torch.Tensor,
                     pos_1: torch.Tensor) -> torch.Tensor:
    """Loss target: x_1 (default) or velocity v = x_1 - x_0 on every atom.
    Non-movable atoms are filtered out by the loss masking, so the value
    we put there does not matter.
    """
    if prediction_type == "v":
        return pos_1 - pos_0
    if prediction_type == "x1":
        return pos_1
    raise ValueError(f"Unknown prediction_type={prediction_type!r}")


def flow_loss_split(
    pred: torch.Tensor,
    pos_0: torch.Tensor,
    pos_1: torch.Tensor,
    movable_mask: torch.Tensor,
    tags: torch.Tensor,
    loss_type: str = "l2",
    prediction_type: str = "x1",
) -> dict:
    """Per-group loss against the prediction target picked by prediction_type."""
    target = _target_for_loss(prediction_type, pos_0, pos_1)
    return x1_loss_split(pred, target, movable_mask, tags, loss_type=loss_type)


def si_denoiser_loss(
    pred_eta: torch.Tensor,
    z: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked OMatG-style denoiser loss, with the constant restored as MSE.

    OMatG ``SingleStochasticInterpolant._sde_loss`` uses
    ``E[eta^2 - 2 eta z]`` for ``loss_z``. Adding the dropped ``E[z^2]``
    constant makes the logged value non-negative without changing gradients.
    """
    _assert_finite(pred_eta, "pred_eta")
    _assert_finite(z, "si_noise_z")
    per_atom = (pred_eta - z).pow(2).sum(dim=-1)
    mask = atom_mask.to(per_atom.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return ((per_atom * mask).sum(dim=1) / denom).mean()


def sample_t(
    batch_size: int, cfg: FlowConfig, device: torch.device,
    dtype: torch.dtype = torch.float32,
    sampling: str = "uniform",
    beta_alpha: float = 2.0,
    beta_beta: float = 1.0,
) -> torch.Tensor:
    if sampling == "uniform":
        u = torch.rand(batch_size, device=device, dtype=dtype)
    elif sampling == "beta":
        alpha = torch.tensor(float(beta_alpha), device=device, dtype=dtype)
        beta = torch.tensor(float(beta_beta), device=device, dtype=dtype)
        u = torch.distributions.Beta(alpha, beta).sample((batch_size,))
    else:
        raise ValueError(f"Unknown t sampling={sampling!r}")
    return cfg.eps + (1 - 2 * cfg.eps) * u


def interpolate_xt(
    pos_0: torch.Tensor,
    pos_1: torch.Tensor,
    t: torch.Tensor,
    movable_mask: torch.Tensor,
) -> torch.Tensor:
    """x_t = (1-t) * x_0 + t * x_1, only on movable atoms.

    Non-movable atoms remain at pos_0 (their pos_1 is expected to equal pos_0).
    """
    _assert_finite(pos_0, "pos_0")
    _assert_finite(pos_1, "pos_1")
    B = pos_0.shape[0]
    t_b = t.view(B, 1, 1).to(pos_0.dtype)
    x_t = (1 - t_b) * pos_0 + t_b * pos_1
    m = movable_mask.unsqueeze(-1).to(x_t.dtype)
    x_t = m * x_t + (1 - m) * pos_0
    return x_t


def x1_loss(
    pred: torch.Tensor,
    pos_1: torch.Tensor,
    movable_mask: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """|| pred - x_1 ||, averaged on movable atoms within sample then across batch."""
    _assert_finite(pred, "pred")
    diff = pred - pos_1
    if loss_type == "l1":
        per_atom = diff.abs().sum(dim=-1)
    else:
        per_atom = diff.pow(2).sum(dim=-1)
    mask = movable_mask.to(per_atom.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    per_sample = (per_atom * mask).sum(dim=1) / denom
    return per_sample.mean()


def x1_loss_split(
    pred: torch.Tensor,
    pos_1: torch.Tensor,
    movable_mask: torch.Tensor,
    tags: torch.Tensor,
    loss_type: str = "l2",
) -> dict:
    """Per-group loss breakdown (surface=tag 1, adsorbate=tag 2). Absolute x_1 target."""
    _assert_finite(pred, "pred")
    diff = pred - pos_1
    if loss_type == "l1":
        per_atom = diff.abs().sum(dim=-1)
    else:
        per_atom = diff.pow(2).sum(dim=-1)

    def _group_loss(group_mask):
        m = group_mask.to(per_atom.dtype)
        denom = m.sum(dim=1)
        has_any = denom > 0
        denom = denom.clamp_min(1.0)
        per_sample = (per_atom * m).sum(dim=1) / denom
        if has_any.any():
            return per_sample[has_any].mean()
        return torch.tensor(0.0, device=per_atom.device, dtype=per_atom.dtype)

    total_loss = _group_loss(movable_mask)
    surf_mask = movable_mask & (tags == 1)
    ads_mask = movable_mask & (tags == 2)
    return {
        "total": total_loss,
        "surf": _group_loss(surf_mask),
        "ads": _group_loss(ads_mask),
    }


def smooth_lddt_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    t: Optional[torch.Tensor] = None,
    time_weight: float = 0.0,
) -> torch.Tensor:
    """Differentiable lDDT-style pair-distance loss over selected atoms.

    Returns a scalar batch mean. Samples with fewer than two selected atoms
    contribute zero, which keeps single-atom adsorbates well-defined.
    """
    _assert_finite(pred_coords, "pred_coords")
    _assert_finite(true_coords, "true_coords")
    true_dists = torch.cdist(true_coords, true_coords)
    pred_dists = torch.cdist(pred_coords, pred_coords)

    B, N, _ = true_coords.shape
    pair_mask = (true_dists < float(cutoff)).to(pred_coords.dtype)
    eye = torch.eye(N, device=pred_coords.device, dtype=pred_coords.dtype).unsqueeze(0)
    pair_mask = pair_mask * (1.0 - eye)
    m = atom_mask.to(pred_coords.dtype)
    pair_mask = pair_mask * m.unsqueeze(1) * m.unsqueeze(2)

    dist_diff = (true_dists - pred_dists).abs()
    score = (
        torch.sigmoid(0.5 - dist_diff)
        + torch.sigmoid(1.0 - dist_diff)
        + torch.sigmoid(2.0 - dist_diff)
        + torch.sigmoid(4.0 - dist_diff)
    ) * 0.25
    denom = pair_mask.sum(dim=(1, 2)).clamp_min(1.0)
    loss = 1.0 - (score * pair_mask).sum(dim=(1, 2)) / denom

    valid = (atom_mask.sum(dim=1) >= 2).to(loss.dtype)
    loss = loss * valid
    if t is not None and float(time_weight) != 0.0:
        t_flat = t.reshape(-1).to(loss.device, loss.dtype)
        loss = loss * (1.0 + float(time_weight) * torch.relu(t_flat - 0.5))
    return loss.mean()


def adsorbate_pair_distance_losses(
    pred_coords: torch.Tensor,
    ref_coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    atom_mask: torch.Tensor,
    bond_factor: float = 1.25,
    clash_factor: float = 0.75,
) -> dict:
    """Adsorbate-internal distance auxiliaries against a reference geometry.

    ``atom_mask`` selects adsorbate atoms. Bonded pairs are inferred from the
    reference geometry with covalent-radius cutoffs, matching the connectivity
    style used by anomaly detection more directly than a soft lDDT score.
    """
    _assert_finite(pred_coords, "pred_coords")
    _assert_finite(ref_coords, "ref_coords")
    pred_d = torch.cdist(pred_coords, pred_coords)
    ref_d = torch.cdist(ref_coords, ref_coords)

    B, N = atom_mask.shape
    dtype = pred_coords.dtype
    device = pred_coords.device

    pair = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    upper = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    pair = pair & upper.unsqueeze(0)

    radii = _COVALENT_RADII.to(device=device, dtype=dtype)
    z = atomic_numbers.clamp(min=0, max=radii.numel() - 1)
    r = radii[z]
    cov_cut = (r.unsqueeze(1) + r.unsqueeze(2)) * float(bond_factor)
    bonded = pair & (ref_d > 0.1) & (ref_d <= cov_cut)
    nonbonded = pair & ~bonded

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(values.dtype)
        denom = m.sum(dim=(1, 2))
        has_any = denom > 0
        per_sample = (values * m).sum(dim=(1, 2)) / denom.clamp_min(1.0)
        if has_any.any():
            return per_sample[has_any].mean()
        return torch.tensor(0.0, device=device, dtype=dtype)

    dist_l1 = (pred_d - ref_d).abs()
    min_nonbonded = (r.unsqueeze(1) + r.unsqueeze(2)) * float(clash_factor)
    clash = torch.relu(min_nonbonded - pred_d).pow(2)

    return {
        "ads_pair_l1": _masked_mean(dist_l1, pair),
        "ads_bond_l1": _masked_mean(dist_l1, bonded),
        "ads_nonbonded_clash": _masked_mean(clash, nonbonded),
    }


def selected_pair_distance_l1_loss(
    pred_coords: torch.Tensor,
    ref_coords: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """All-pair distance L1 loss over a selected atom subset."""
    _assert_finite(pred_coords, "pred_coords")
    _assert_finite(ref_coords, "ref_coords")
    pred_d = torch.cdist(pred_coords, pred_coords)
    ref_d = torch.cdist(ref_coords, ref_coords)

    B, N = atom_mask.shape
    device = pred_coords.device
    dtype = pred_coords.dtype
    pair = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
    upper = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
    pair = pair & upper.unsqueeze(0)

    m = pair.to(dtype)
    denom = m.sum(dim=(1, 2))
    has_any = denom > 0
    per_sample = ((pred_d - ref_d).abs() * m).sum(dim=(1, 2)) / denom.clamp_min(1.0)
    if has_any.any():
        return per_sample[has_any].mean()
    return torch.tensor(0.0, device=device, dtype=dtype)


def _score_from_velocity(
    v: torch.Tensor,
    x_t: torch.Tensor,
    pos_0: torch.Tensor,
    t_scalar: float,
    eps: float,
) -> torch.Tensor:
    """Score for SDE path (absolute coordinates).

    Uses the identity delta_t = x_t - x_0 with linear flow, so the same analytic
    form (t*v - delta_t) / (1 - t) carries over.
    """
    return (t_scalar * v - (x_t - pos_0)) / max(1.0 - float(t_scalar), eps)


@dataclass
class FKSteeringConfig:
    """Feynman-Kac particle steering during sampling."""

    num_particles: int
    energy_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    fk_lambda: float = 10.0
    resampling_interval: int = 1
    fk_start_time: float = 0.0
    potential_mode: str = "difference"


def _fk_log_weights(energy_traj: torch.Tensor, mode: str) -> torch.Tensor:
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
    pos_0: torch.Tensor,
    movable_mask: torch.Tensor,
    pad_mask: torch.Tensor,
    cfg: FlowConfig,
    num_steps: int = 50,
    use_sde: bool = False,
    refine_final: bool = False,
    return_trajectory: bool = False,
    fk_steering: Optional[FKSteeringConfig] = None,
    sde_schedule: str = "atommof",
    sde_alpha: float = 1.0,
    sde_no_score: bool = False,
    sde_mode: str = "atommof",
    si_gamma_schedule: str = "sqrt_t1mt",
    si_gamma_sigma: float = 0.0,
    si_epsilon_schedule: str = "vanishing_1mt",
    si_epsilon_scale: float = 1.0,
    time_schedule: str = "uniform",
    time_schedule_beta: float = 2.0,
    solver: str = "euler",
):
    """Euler integrator on absolute coordinates (AtomMOF-style).

    When use_sde=True, applies the AtomMOF SDE update at every step:
        g^2(t) per ``sde_schedule``:
            "atommof"   -> 0.5 * (1 - t)               [paper default, α ignored]
            "zero_ends" -> sde_alpha * t * (1 - t)     [zero at both endpoints]
        score  = (t * v_centered - x_t_centered) / (1 - t)
        drift  = v + 0.5 * g^2 * score
        x_t   += drift * dt + sqrt(g^2 * dt) * noise_centered
    where centering subtracts the COM over movable atoms (preserving the
    all-atom COM=0 invariant the model was trained on, given non-movable
    atoms are frozen at pos_0).

    Args:
        model_forward: callable (x_t, t) -> pred_x_1. The caller closes over the
            static context (pos_0, atomic_numbers, tags, cell, masks, ...).
        pos_0: (B, N, 3) structured prior sample (x_0 absolute coords).
        movable_mask: (B, N) bool; non-movable atoms stay frozen at pos_0.
        pad_mask: (B, N) bool.
        cfg: FlowConfig (eps).
        num_steps: number of Euler steps.
        use_sde: enable SDE step. ``sde_mode='atommof'`` keeps the legacy
            AtomMOF score heuristic; ``sde_mode='omatg_si'`` uses an explicit
            OMatG-style denoiser head eta.
        refine_final: one extra forward at t=1-eps used as final x_1 prediction.
        return_trajectory: return stacked x_t trajectory dict.
        fk_steering: optional particle steering config.

    Returns:
        x_out: (B, N, 3) final absolute coords; non-movable at pos_0, padding zeroed.
        If return_trajectory: dict.
    """
    device = pos_0.device
    dtype = pos_0.dtype
    B, N, _ = pos_0.shape
    movable_f = movable_mask.unsqueeze(-1).to(dtype)
    pad_f = pad_mask.unsqueeze(-1).to(dtype)

    x_t = pos_0.clone()
    x_t = x_t * pad_f

    u_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=dtype)
    if time_schedule == "uniform":
        tau = u_vals
    elif time_schedule == "high_t_power":
        # More, smaller steps near t=1. beta=1 recovers uniform; larger beta
        # spends progressively more resolution at high t.
        beta = max(float(time_schedule_beta), 1.0)
        tau = 1.0 - torch.pow(1.0 - u_vals, beta)
    elif time_schedule == "low_t_power":
        beta = max(float(time_schedule_beta), 1.0)
        tau = torch.pow(u_vals, beta)
    elif time_schedule == "beta_train":
        # Deterministic quantile grid matching the train-time Beta(alpha, 1)
        # sampling used by the beta-scheduled SI run. For alpha=2 this is
        # tau=sqrt(u), giving more solver points at high t.
        alpha = max(float(time_schedule_beta), 1.0)
        tau = torch.pow(u_vals, 1.0 / alpha)
    else:
        raise ValueError(f"Unknown time_schedule={time_schedule!r}")
    t_vals = cfg.eps + (1.0 - 2.0 * cfg.eps) * tau

    traj: List[torch.Tensor] = [x_t.clone()] if return_trajectory else []
    energy_traj: Optional[torch.Tensor] = None
    if solver not in ("euler", "heun"):
        raise ValueError(f"Unknown solver={solver!r}; expected 'euler' or 'heun'")
    if fk_steering is not None:
        energy_traj = torch.empty((B, 0), device=device, dtype=dtype)
        if B % fk_steering.num_particles != 0:
            raise ValueError(
                f"Batch size {B} must be divisible by num_particles {fk_steering.num_particles}"
            )

    # AtomMOF SDE knobs:
    #   schedule:    g²(t) = 0.5 · (1 - t)
    #   score:       s = (t · v_centered - x_t_centered) / (1 - t)
    #                where centering subtracts the COM over movable atoms
    #   noise:       zero-mean Gaussian on movable atoms (sum-to-zero)
    # This preserves the all-atom COM=0 invariant the model trained on
    # (non-movable atoms are frozen at pos_0; centering movable atoms keeps
    # ads COM stable, matching the model's ads_center_rel head assumptions).
    def _sde_drift_noise(
        x: torch.Tensor,
        v_in: torch.Tensor,
        t_s: float,
        dt_s: float,
        eta: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_movable = movable_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        if sde_mode == "omatg_si":
            if eta is None:
                raise ValueError(
                    "sde_mode='omatg_si' requires model_forward to return "
                    "{'pred_x1': ..., 'eta': ...}"
                )
            t_tensor = torch.full((B,), t_s, device=device, dtype=dtype)
            gamma_t = si_gamma(
                t_tensor, si_gamma_schedule, si_gamma_sigma, eps=cfg.eps,
            ).view(B, 1, 1).to(dtype)
            eps_t = si_epsilon(
                t_tensor, si_epsilon_schedule, si_epsilon_scale, eps=cfg.eps,
            ).view(B, 1, 1).to(dtype)
            if (gamma_t <= 0).any() and (eps_t > 0).any():
                raise ValueError(
                    "OMatG SI SDE requires gamma(t)>0 whenever epsilon(t)>0; "
                    f"got schedule={si_gamma_schedule!r}, sigma={si_gamma_sigma}"
                )

            # OMatG drift: b_theta - epsilon/gamma * eta_theta. We keep the
            # existing AdsorbGen COM-preserving movable noise so non-movable
            # slab atoms remain exactly frozen during sampling.
            drift = v_in - (eps_t / gamma_t.clamp_min(cfg.eps)) * eta.to(dtype) * movable_f
            if noise is None:
                noise = torch.randn(x.shape, device=device, dtype=dtype) * movable_f
                noise_com = noise.sum(dim=1, keepdim=True) / n_movable
                noise = (noise - noise_com * movable_f) * movable_f
            noise_term = torch.sqrt((2.0 * eps_t * dt_s).clamp_min(0.0)) * noise
            return drift, noise_term

        # Centering of v, x over movable atoms (AtomMOF's score formula).
        x_com = (x * movable_f).sum(dim=1, keepdim=True) / n_movable
        v_com = (v_in * movable_f).sum(dim=1, keepdim=True) / n_movable
        x_cen = (x - x_com) * movable_f
        v_cen = (v_in - v_com) * movable_f

        # g²(t)
        if sde_schedule == "zero_ends":
            g2 = sde_alpha * t_s * (1.0 - t_s)
        else:  # "atommof"
            g2 = 0.5 * (1.0 - t_s)
        if sde_no_score:
            # OMatG-RL surrogate SDE: drift = v only (no score correction).
            score = torch.zeros_like(x_cen)
        else:
            score = (t_s * v_cen - x_cen) / max(1.0 - t_s, cfg.eps)
            score = score * movable_f

        # Zero-mean noise on movable atoms (preserves COM).
        if noise is None:
            noise = torch.randn(x.shape, device=device, dtype=dtype) * movable_f
            noise_com = noise.sum(dim=1, keepdim=True) / n_movable
            noise = (noise - noise_com * movable_f) * movable_f

        drift = v_in + 0.5 * g2 * score
        noise_term = (g2 * dt_s) ** 0.5 * noise
        return drift, noise_term

    def _sde_step(
        x: torch.Tensor,
        v_in: torch.Tensor,
        t_s: float,
        dt_s: float,
        eta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        drift, noise_term = _sde_drift_noise(x, v_in, t_s, dt_s, eta=eta)
        return x + drift * dt_s + noise_term

    def _constrain(x: torch.Tensor) -> torch.Tensor:
        return (x * movable_f + pos_0 * (1 - movable_f)) * pad_f

    def _velocity_from_output(
        model_out,
        x: torch.Tensor,
        t_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(model_out, dict):
            pred_out = model_out["pred_x1"]
            eta_out = model_out.get("eta")
        else:
            pred_out = model_out
            eta_out = None
        if cfg.prediction_type == "v":
            v_out = pred_out
        else:
            v_out = (pred_out - x) / max(1.0 - t_s, cfg.eps)
        return pred_out, v_out * movable_f, eta_out

    for i in range(num_steps):
        t_scalar = float(t_vals[i].item())
        dt = float((t_vals[i + 1] - t_vals[i]).item())
        t = t_vals[i].expand(B)

        model_out = model_forward(x_t, t)
        pred, v, eta = _velocity_from_output(model_out, x_t, t_scalar)

        if fk_steering is not None and t_scalar >= fk_steering.fk_start_time \
                and (i % fk_steering.resampling_interval == 0):
            # FK steering scores positions, so for v-pred we map to x_1 first.
            # Resampling happens before the Euler/Heun step.  For Heun, the
            # predictor-corrector then uses the resampled x_t/pred/eta together,
            # so particle ancestry stays aligned across the two drift estimates.
            if cfg.prediction_type == "v":
                pred_x_1_for_fk = pos_0 + pred
            else:
                pred_x_1_for_fk = pred
            current_energy = fk_steering.energy_fn(
                pred_x_1_for_fk, pad_mask, movable_mask,
            ).to(dtype)
            energy_traj = torch.cat([energy_traj, current_energy.unsqueeze(1)], dim=1)

            log_G = _fk_log_weights(energy_traj, fk_steering.potential_mode)
            P = fk_steering.num_particles
            log_G = log_G.reshape(-1, P)
            weights = F.softmax(log_G * fk_steering.fk_lambda, dim=1)
            sampled = torch.multinomial(weights, P, replacement=True)
            offset = torch.arange(weights.shape[0], device=device).unsqueeze(1) * P
            idx = (sampled + offset).flatten()

            x_t = x_t[idx]
            pred = pred[idx]
            if eta is not None:
                eta = eta[idx]
            energy_traj = energy_traj[idx]

        if solver == "heun":
            if use_sde:
                drift0, noise_term = _sde_drift_noise(x_t, v, t_scalar, dt, eta=eta)
                x_pred = _constrain(x_t + drift0 * dt + noise_term)
            else:
                drift0 = v
                noise_term = torch.zeros_like(x_t)
                x_pred = _constrain(x_t + drift0 * dt)

            t_next_scalar = float(t_vals[i + 1].item())
            t_next = t_vals[i + 1].expand(B)
            model_out_next = model_forward(x_pred, t_next)
            _, v_next, eta_next = _velocity_from_output(model_out_next, x_pred, t_next_scalar)
            if use_sde:
                drift1, _ = _sde_drift_noise(
                    x_pred,
                    v_next,
                    t_next_scalar,
                    dt,
                    eta=eta_next,
                    noise=torch.zeros_like(x_t),
                )
                x_t = x_t + 0.5 * (drift0 + drift1) * dt + noise_term
            else:
                x_t = x_t + 0.5 * (drift0 + v_next) * dt
        elif use_sde:
            x_t = _sde_step(x_t, v, t_scalar, dt, eta=eta)
        else:
            x_t = x_t + v * dt

        # Freeze non-movable atoms at pos_0, zero out padding
        x_t = _constrain(x_t)

        if return_trajectory:
            traj.append(x_t.clone())

    if refine_final:
        t_final = t_vals[-1].expand(B)
        pred_final = model_forward(x_t, t_final)
        if isinstance(pred_final, dict):
            pred_final = pred_final["pred_x1"]
        if cfg.prediction_type == "v":
            x_1_final = pos_0 + pred_final
        else:
            x_1_final = pred_final
        x_out = x_1_final * movable_f + pos_0 * (1 - movable_f)
        x_out = x_out * pad_f
    else:
        x_out = x_t

    if not return_trajectory:
        return x_out

    out = {
        "x_out": x_out,
        "x_trajectory": torch.stack(traj, dim=0) if traj else None,
    }
    if energy_traj is not None:
        out["energy_trajectory"] = energy_traj
    return out


def cfg_model_forward(
    f_cond: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    f_uncond: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    w: float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Classifier-free guidance combiner for (x_t, t) -> pred_x_1 callables."""

    def _f(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        fc = f_cond(x_t, t)
        fu = f_uncond(x_t, t)
        return (1.0 + w) * fc - w * fu

    return _f
