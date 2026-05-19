"""Sampling CLI for the AdsorbGen flow matching DiT.

Loads a trained checkpoint and runs the Euler (ODE or SDE) sampler on a
preprocessed displacement LMDB. Outputs predicted Cartesian positions
alongside the reference (ground-truth relaxed) structure so downstream
evaluation can compute metrics without re-running the model.

Sampler features:
    --use-sde             run the SDE update with g^2(t) = 0.5*(1-t)
    --refine-final        one extra forward at t=1-eps for the final step
    --fk-particles K      enable Feynman-Kac steering with K particles
    --fk-potential MODE   immediate|difference|max|sum (needs FK energy fn)

Usage:
    PYTHONPATH=AdsorbGen python -m adsorbgen.inference \
        --ckpt runs/v2/last.ckpt \
        --lmdb data/processed/oc20dense.lmdb \
        --out  runs/v2/samples.pt \
        --num-steps 50 --batch-size 8 --max-samples 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.dataset import PlacementPriorDataset, PreprocessedDisplacementDataset, collate_displacement  # noqa: E402
from adsorbgen.flow import FKSteeringConfig, FlowConfig, euler_sample  # noqa: E402
from adsorbgen.model import DiTDenoiser, DiTDenoiserConfig  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2, DiTDenoiserV2Config  # noqa: E402
from adsorbgen.multiplace import DEFAULT_ADSORBATES_PKL, MultiPlacementDataset  # noqa: E402


def _filter_dataclass_fields(cls, payload: dict) -> dict:
    """Keep only keys that ``cls`` accepts. Tolerates legacy/extra fields."""
    valid = {f.name for f in cls.__dataclass_fields__.values()}
    return {k: v for k, v in payload.items() if k in valid}


def _resolve_model_cfg(args_json_path: Path):
    if not args_json_path.exists():
        raise FileNotFoundError(
            f"Could not find {args_json_path}; pass --train-args-json to override."
        )
    with open(args_json_path) as f:
        a = json.load(f)
    arch = a.get("arch", "v1")
    # train.py writes {"arch": ..., "model_config": {...}}; older v1 runs (pre-v2
    # protocol) saved the config at the top level, so fall back to ``a`` when
    # "model_config" is absent. Without this fallback, v1 runs whose overrides
    # change atom_s/atom_z (e.g. v1-wide) load the default-width model and hit
    # shape mismatches at state_dict load.
    payload = a.get("model_config", a)
    if arch == "v1":
        return DiTDenoiserConfig(**_filter_dataclass_fields(DiTDenoiserConfig, payload))
    if arch == "v2":
        return DiTDenoiserV2Config(
            **_filter_dataclass_fields(DiTDenoiserV2Config, payload)
        )
    raise ValueError(f"unknown arch in {args_json_path}: {arch!r}")


def _extract_state_dict(state) -> dict:
    """Dispatch the three checkpoint formats AdsorbGen has shipped:

    * Lightning ``{"state_dict": {"model.X": ...}, ...}`` — strip ``"model."``.
    * Old custom ``{"model": state_dict}``.
    * Raw ``state_dict`` mapping.
    """
    if isinstance(state, dict) and "state_dict" in state:
        sd_full = state["state_dict"]
        sd = {k.removeprefix("model."): v for k, v in sd_full.items() if k.startswith("model.")}
        if not sd:
            sd = dict(sd_full)
        return sd
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _make_forward(
    model: torch.nn.Module,
    batch: dict,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the model_forward closure for euler_sample.

    Captures static batch context (pos, tags, atomic_numbers, cell, masks).
    If the v2 model has ``cfg.use_self_cond``, the closure threads the
    previous step's detached prediction through ``prev_pred`` so each Euler
    step sees the last forward's output (AF3-style self-conditioning).
    """
    extra_static = {}

    use_self_cond = (
        isinstance(model, DiTDenoiserV2) and getattr(model.cfg, "use_self_cond", False)
    )
    use_ads_ref = bool(getattr(getattr(model, "cfg", None), "use_ads_ref_pos", False))
    state = {"prev_pred": None}

    def _f(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        extra = dict(extra_static)
        if use_self_cond:
            extra["prev_pred"] = state["prev_pred"]
        if use_ads_ref:
            if "ads_ref_pos" not in batch:
                raise KeyError(
                    "model requires ads_ref_pos, but the inference dataset did not "
                    "provide it; construct the dataset with provide_ads_ref_pos=True"
                )
            extra["ads_ref_pos"] = batch["ads_ref_pos"]
        out = model(
            pos=batch["pos"],
            x_t=x_t,
            t=t,
            atomic_numbers=batch["atomic_numbers"],
            tags=batch["tags"],
            movable_mask=batch["movable_mask"],
            pad_mask=batch["pad_mask"],
            cell=batch["cell"],
            **extra,
        )
        if use_self_cond:
            state["prev_pred"] = out.detach()
        return out
    return _f


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--lmdb", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-trajectories", type=int, default=0,
                   help="store Euler x_t trajectories for the first N base samples in the output dump")

    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--flow-eps", type=float, default=1e-5)
    p.add_argument("--prediction-type", type=str, choices=["x1", "v"], default=None,
                   help="Override sampler prediction type. Default: read from args.json.")
    p.add_argument("--prior-mode",
                   choices=["random", "heuristic", "random_heuristic",
                            "harmonic_uniform", "harmonic_centered", "catflow_center_rel"],
                   default="random_heuristic")
    p.add_argument("--use-placement-prior", action="store_true",
                   help="for K=1, replace adsorbate pos with a fresh fairchem placement, matching training sample_eval")
    p.add_argument("--interstitial-gap", type=float, default=0.1)
    p.add_argument("--use-sde", action="store_true")
    p.add_argument("--refine-final", action="store_true")

    p.add_argument("--fk-particles", type=int, default=0,
                   help="enable FK steering with this many particles per sample")
    p.add_argument("--fk-lambda", type=float, default=10.0)
    p.add_argument("--fk-start-time", type=float, default=0.0)
    p.add_argument("--fk-potential", type=str, default="difference",
                   choices=["immediate", "difference", "max", "sum"])
    p.add_argument("--fk-resample-interval", type=int, default=1)
    p.add_argument("--fk-energy", type=str, default="zero",
                   choices=["zero", "uma"],
                   help="FK steering energy: 'zero' (uniform dummy) or 'uma' (UMA-s MLIP)")
    p.add_argument("--fk-uma-model", type=str, default="uma-s-1p1",
                   help="fairchem pretrained model id when --fk-energy=uma")

    p.add_argument("--num-samples", type=int, default=1,
                   help="K: number of random_site_heuristic_placement starts per system")
    p.add_argument("--adsorbates-pkl", type=str, default=DEFAULT_ADSORBATES_PKL,
                   help="path to fairchem's adsorbates.pkl for K-placement and ads_ref_pos")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-args-json", type=str, default=None,
                   help="override path to args.json (default: ckpt_dir/args.json)")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    torch.serialization.add_safe_globals(
        [DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig]
    )

    ckpt_path = Path(args.ckpt)
    assert ckpt_path.exists(), f"missing ckpt {ckpt_path}"
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = _extract_state_dict(state)

    args_json_path = Path(args.train_args_json) if args.train_args_json else (ckpt_path.parent / "args.json")
    model_cfg = _resolve_model_cfg(args_json_path)
    with open(args_json_path) as f:
        train_args_blob = json.load(f)
    saved_prediction_type = (
        train_args_blob.get("flow_config", {}).get("prediction_type")
        or train_args_blob.get("train_args", {}).get("prediction_type")
        or "x1"
    )
    prediction_type = args.prediction_type or saved_prediction_type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()
    use_ads_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))

    K = max(int(args.num_samples), 1)
    if K > 1:
        dataset = MultiPlacementDataset(
            args.lmdb,
            num_placements=K,
            adsorbates_pkl_path=args.adsorbates_pkl,
            max_samples=args.max_samples,
            prior_mode=args.prior_mode,
            provide_ads_ref_pos=use_ads_ref,
        )
        n_base = len(dataset.base)
        print(f"[data] {n_base} base systems × K={K} placements (mode={args.prior_mode}) from {args.lmdb}", flush=True)
    elif args.use_placement_prior:
        dataset = PlacementPriorDataset(
            args.lmdb,
            max_samples=args.max_samples,
            prior_mode=args.prior_mode,
            interstitial_gap=args.interstitial_gap,
            provide_ads_ref_pos=use_ads_ref,
            adsorbates_pkl=args.adsorbates_pkl,
        )
        n_base = len(dataset)
        print(
            f"[data] {n_base} samples with fresh placement "
            f"(mode={args.prior_mode}) from {args.lmdb}",
            flush=True,
        )
    else:
        dataset = PreprocessedDisplacementDataset(
            args.lmdb,
            max_samples=args.max_samples,
            provide_ads_ref_pos=use_ads_ref,
            adsorbates_pkl=args.adsorbates_pkl,
        )
        n_base = len(dataset)
        print(f"[data] {n_base} samples from {args.lmdb}", flush=True)
    # DataLoader batch_size is in "per-placement" units. Round up by K so each
    # outer iteration holds args.batch_size whole base systems, and worker-side
    # placement caching keeps fairchem calls at one per base system.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * K,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_displacement,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    flow_cfg = FlowConfig(eps=args.flow_eps, prediction_type=prediction_type)

    # FK steering: energy model is loaded once and then rebound to each
    # replicated batch inside the sampling loop via make_fk_energy_fn.
    energy_model = None
    if args.fk_particles > 0:
        if args.fk_energy == "uma":
            from adsorbgen.energy import UMAEnergy  # noqa: WPS433
            energy_model = UMAEnergy(
                model_name=args.fk_uma_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                task_name="oc20",
            )
            print(
                f"[fk] UMA energy ({args.fk_uma_model}, task=oc20) loaded for FK steering",
                flush=True,
            )
        elif args.fk_energy == "zero":
            print(
                "[fk] WARNING: FK steering enabled with zero-energy stub — "
                "pass --fk-energy uma for meaningful particle resampling.",
                flush=True,
            )

    def _replicate(batch: dict, P: int) -> dict:
        """Replicate every per-sample tensor P times along dim 0 for FK steering."""
        out = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            rep_shape = [P if i == 0 else 1 for i in range(v.dim())]
            out[k] = v.unsqueeze(1).expand(v.shape[0], P, *v.shape[1:]).reshape(-1, *v.shape[1:]) \
                if v.dim() >= 2 else v.unsqueeze(1).expand(v.shape[0], P).reshape(-1)
            _ = rep_shape  # unused, kept for clarity
        return out

    all_records = []
    n_done = 0
    t0 = time.time()
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        BK = batch["pos"].shape[0]
        assert BK % K == 0, f"batch size {BK} must be multiple of K={K}"
        B_orig = BK // K

        # For FK steering we must replicate the static context before calling
        # euler_sample; the sampler resamples indices within each P-group.
        P = args.fk_particles if args.fk_particles > 0 else 1
        work = _replicate(batch, P) if P > 1 else batch
        pos_gt_work = work["pos_relaxed"]

        # Rebuild fk_cfg per batch so the energy closure captures the
        # replicated atomic_numbers/cell for THIS batch.
        fk_cfg: Optional[FKSteeringConfig] = None
        if args.fk_particles > 0:
            if args.fk_energy == "uma":
                from adsorbgen.energy import make_fk_energy_fn  # noqa: WPS433
                energy_fn = make_fk_energy_fn(
                    energy_model, work["atomic_numbers"], work["cell"]
                )
            else:
                def energy_fn(x_pred, pad, mov):  # zero stub
                    return torch.zeros(x_pred.shape[0], device=x_pred.device, dtype=x_pred.dtype)
            fk_cfg = FKSteeringConfig(
                num_particles=args.fk_particles,
                energy_fn=energy_fn,
                fk_lambda=args.fk_lambda,
                resampling_interval=args.fk_resample_interval,
                fk_start_time=args.fk_start_time,
                potential_mode=args.fk_potential,
            )

        model_forward = _make_forward(model=model, batch=work)

        want_traj = args.save_trajectories > 0 and len(all_records) < args.save_trajectories
        sample_out = euler_sample(
            model_forward=model_forward,
            pos_0=work["pos"],
            movable_mask=work["movable_mask"],
            pad_mask=work["pad_mask"],
            cfg=flow_cfg,
            num_steps=args.num_steps,
            use_sde=args.use_sde,
            refine_final=args.refine_final,
            return_trajectory=want_traj,
            fk_steering=fk_cfg,
        )
        if want_traj:
            x_out = sample_out["x_out"]
            x_traj = sample_out["x_trajectory"]
        else:
            x_out = sample_out
            x_traj = None

        # Collapse FK groups by keeping particle 0 (FK reorders within a group,
        # so any particle is fine — downstream eval can re-rank if desired).
        def _pick(t: torch.Tensor) -> torch.Tensor:
            if P == 1:
                return t
            return t.view(BK, P, *t.shape[1:])[:, 0]

        x_final = _pick(x_out)          # (BK, N, 3)
        traj_g = None
        if x_traj is not None:
            # (T, BK*P, N, 3) -> keep particle 0 if FK is active, then group by
            # base system and placement.
            if P == 1:
                traj_pick = x_traj
            else:
                traj_pick = x_traj.view(x_traj.shape[0], BK, P, *x_traj.shape[2:])[:, :, 0]
            traj_g = traj_pick.view(traj_pick.shape[0], B_orig, K, *traj_pick.shape[2:])
        pos_ref = _pick(work["pos"])    # (BK, N, 3)
        pos_gt_bk = _pick(pos_gt_work)  # (BK, N, 3)
        pad = _pick(work["pad_mask"])
        mov = _pick(work["movable_mask"])
        tags = _pick(work["tags"])
        zs = _pick(work["atomic_numbers"])
        cells = _pick(work["cell"])
        sids = _pick(work["sid"])
        yr = _pick(work["y_relaxed"])

        # Reshape (BK, ...) -> (B_orig, K, ...). Non-position fields are
        # identical across the K axis; we keep placement 0 as the canonical
        # copy.
        def _group(t: torch.Tensor) -> torch.Tensor:
            return t.view(B_orig, K, *t.shape[1:])

        x_final_g = _group(x_final)          # (B, K, N, 3)
        pos_ref_g = _group(pos_ref)          # (B, K, N, 3)
        pos_gt_g = _group(pos_gt_bk)[:, 0]   # (B, N, 3)
        pad_g = _group(pad)[:, 0]
        mov_g = _group(mov)[:, 0]
        tags_g = _group(tags)[:, 0]
        zs_g = _group(zs)[:, 0]
        cells_g = _group(cells)[:, 0]
        sids_g = _group(sids)[:, 0]
        yr_g = _group(yr)[:, 0]
        system_keys = batch.get("system_key", [None] * (B_orig * K))
        config_keys = batch.get("config_key", [None] * (B_orig * K))

        for i in range(B_orig):
            n = int(pad_g[i].sum().item())
            rec = {
                "pos_ref": pos_ref_g[i, :, :n].cpu() if K > 1 else pos_ref_g[i, 0, :n].cpu(),
                "pos_pred": x_final_g[i, :, :n].cpu() if K > 1 else x_final_g[i, 0, :n].cpu(),
                "pos_gt": pos_gt_g[i, :n].cpu(),
                "movable_mask": mov_g[i, :n].cpu(),
                "atomic_numbers": zs_g[i, :n].cpu(),
                "tags": tags_g[i, :n].cpu(),
                "cell": cells_g[i].cpu(),
                "sid": int(sids_g[i].item()),
                "system_key": system_keys[i * K],
                "config_key": config_keys[i * K],
                "y_relaxed": float(yr_g[i].item()),
                "num_placements": K,
            }
            if traj_g is not None and len(all_records) < args.save_trajectories:
                rec["x_trajectory"] = (
                    traj_g[:, i, :, :n].cpu()
                    if K > 1 else traj_g[:, i, 0, :n].cpu()
                )
            all_records.append(rec)

        n_done += B_orig
        print(f"[sample] {n_done}/{n_base} elapsed={time.time() - t0:.1f}s", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "records": all_records,
            "meta": {
                "ckpt": str(ckpt_path),
                "lmdb": args.lmdb,
                "arch": "v2" if isinstance(model_cfg, DiTDenoiserV2Config) else "v1",
                "num_steps": args.num_steps,
                "prediction_type": prediction_type,
                "prior_mode": args.prior_mode,
                "use_sde": args.use_sde,
                "refine_final": args.refine_final,
                "fk_particles": args.fk_particles,
                "fk_potential": args.fk_potential if args.fk_particles > 0 else None,
                "fk_energy": args.fk_energy if args.fk_particles > 0 else None,
                "fk_uma_model": args.fk_uma_model if args.fk_particles > 0 and args.fk_energy == "uma" else None,
                "num_placements": K,
                "n_samples": len(all_records),
                "n_trajectories_saved": sum(1 for r in all_records if "x_trajectory" in r),
            },
        },
        out_path,
    )
    print(f"[done] wrote {len(all_records)} records -> {out_path} ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
