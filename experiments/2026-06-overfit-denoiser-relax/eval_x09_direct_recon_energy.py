#!/usr/bin/env python
"""Evaluate direct x1-noise denoiser energy reconstruction.

This evaluates checkpoints trained by ``train_x09_recon_denoiser_overfit.py``:

    x_noisy = x1 + gamma(t) z
    x_recon = model(pos=x0_context, x_t=x_noisy, t=t)

No SI/eta head is used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


def install_imports(repo: Path, adsorbates_pkl: Path) -> None:
    os.environ["ADSGEN_ROOT"] = str(repo)
    os.environ["ADSORBATES_PKL"] = str(adsorbates_pkl)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def masked_rmsd(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = a - b
    per_atom = diff.pow(2).sum(dim=-1)
    denom = mask.to(per_atom.dtype).sum(dim=1).clamp_min(1.0)
    return torch.sqrt((per_atom * mask.to(per_atom.dtype)).sum(dim=1) / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--lmdb", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adsorbates-pkl", default="")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--t", type=float, default=0.9)
    ap.add_argument("--gamma-schedule", default="", choices=["", "sqrt_t1mt", "linear_1mt"])
    ap.add_argument("--gamma-sigma", type=float, default=None)
    ap.add_argument("--prior-mode", default="random_heuristic",
                    choices=["random", "heuristic", "random_heuristic",
                             "harmonic_uniform", "harmonic_centered",
                             "catflow_center_rel", "gaussian_ads_train_stats"])
    ap.add_argument("--interstitial-gap", type=float, default=0.1)
    ap.add_argument("--uma-model", default="uma-s-1p1")
    ap.add_argument("--uma-task", default="oc20")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    adsorbates = Path(args.adsorbates_pkl or (repo / "data" / "pkls" / "adsorbates.pkl"))
    install_imports(repo, adsorbates)

    from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement
    from adsorbgen.evaluation.energy import UMAEnergy
    from adsorbgen.flow import si_gamma
    from adsorbgen.replay.eval import _model_cfg, _runtime_movable_mask
    from geoopt.geoopt import load_model_from_ckpt

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    ckpt = Path(args.ckpt)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    hp = dict(payload.get("hyper_parameters", {}))
    gamma_schedule = args.gamma_schedule or str(hp.get("gamma_schedule", "sqrt_t1mt"))
    gamma_sigma = float(args.gamma_sigma if args.gamma_sigma is not None else hp.get("gamma_sigma", 0.1))

    model, flow_cfg = load_model_from_ckpt(ckpt, device)
    cfg = _model_cfg(model)
    use_ads_ref = bool(getattr(cfg, "use_ads_ref_pos", False))
    slab_source = str(getattr(model, "adsorbgen_slab_source", "initial"))
    pristine_slabs = str(getattr(model, "adsorbgen_pristine_slabs", ""))
    pristine_index = str(getattr(model, "adsorbgen_pristine_index", ""))
    ds = PlacementPriorDataset(
        args.lmdb,
        prior_mode=args.prior_mode,
        interstitial_gap=args.interstitial_gap,
        adsorbates_pkl=str(adsorbates),
        max_samples=None,
        provide_ads_ref_pos=use_ads_ref,
        skip_anomaly=False,
        slab_source=slab_source,
        pristine_slabs=pristine_slabs,
        pristine_index=pristine_index,
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_displacement)
    energy = UMAEnergy(model_name=args.uma_model, task_name=args.uma_task,
                       device=str(device), normalize_per_atom=False)

    rows: list[dict[str, Any]] = []
    for batch_i, batch in enumerate(dl):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        B = int(batch["pos"].shape[0])
        t_vec = torch.full((B,), float(args.t), device=device, dtype=batch["pos"].dtype)
        pos = batch["pos"]
        x1 = batch["pos_relaxed"]
        movable = _runtime_movable_mask(model, batch)
        pad = batch["pad_mask"]
        movable_f = movable.unsqueeze(-1).to(pos.dtype)
        pad_f = pad.unsqueeze(-1).to(pos.dtype)
        z = torch.randn_like(x1) * movable_f
        gamma = si_gamma(t_vec, gamma_schedule, gamma_sigma, eps=float(flow_cfg.eps)).view(B, 1, 1).to(pos.dtype)
        x_noisy = x1 + gamma * z
        x_noisy = x_noisy * movable_f + pos * (1.0 - movable_f)
        x_noisy = x_noisy * pad_f

        extra = {}
        if use_ads_ref:
            extra["ads_ref_pos"] = batch["ads_ref_pos"]
        with torch.no_grad():
            out = model(
                pos=pos,
                x_t=x_noisy,
                t=t_vec,
                atomic_numbers=batch["atomic_numbers"],
                tags=batch["tags"],
                movable_mask=movable,
                pad_mask=pad,
                cell=batch["cell"],
                **extra,
            )
            x_recon = out["pred_x1"] if isinstance(out, dict) else out
            x_recon = x_recon * movable_f + pos * (1.0 - movable_f)
            x_recon = x_recon * pad_f
            x1_eval = (x1 * movable_f + pos * (1.0 - movable_f)) * pad_f
            e_target = energy(x1_eval, batch["cell"], batch["atomic_numbers"], pad)
            e_noisy = energy(x_noisy, batch["cell"], batch["atomic_numbers"], pad)
            e_recon = energy(x_recon, batch["cell"], batch["atomic_numbers"], pad)
            rmsd = masked_rmsd(x_recon, x1_eval, movable)

        for i in range(B):
            global_i = batch_i * int(args.batch_size) + i
            n = int(pad[i].sum().item())
            rows.append({
                "idx": int(global_i),
                "sid": int(batch["sid"][i].item()) if "sid" in batch else -1,
                "ads_id": int(batch["ads_id"][i].item()) if "ads_id" in batch else -1,
                "n_atoms": n,
                "n_movable": int(movable[i].sum().item()),
                "E_target": float(e_target[i].item()),
                "E_noisy": float(e_noisy[i].item()),
                "E_recon": float(e_recon[i].item()),
                "delta_E_noisy_target": float((e_noisy[i] - e_target[i]).item()),
                "delta_E_recon_target": float((e_recon[i] - e_target[i]).item()),
                "rmsd_recon_to_x1": float(rmsd[i].item()),
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    deltas = np.asarray([r["delta_E_recon_target"] for r in rows], dtype=np.float64)
    noisy = np.asarray([r["delta_E_noisy_target"] for r in rows], dtype=np.float64)
    summary = {
        "ckpt": str(ckpt),
        "lmdb": str(args.lmdb),
        "t": float(args.t),
        "gamma_schedule": gamma_schedule,
        "gamma_sigma": gamma_sigma,
        "n": len(rows),
        "noisy_delta_E_mae_eV": float(np.mean(np.abs(noisy))) if len(rows) else float("nan"),
        "recon_delta_E_mean_eV": float(np.mean(deltas)) if len(rows) else float("nan"),
        "recon_delta_E_mae_eV": float(np.mean(np.abs(deltas))) if len(rows) else float("nan"),
        "recon_delta_E_max_abs_eV": float(np.max(np.abs(deltas))) if len(rows) else float("nan"),
        "rmsd_recon_to_x1_mean_A": float(np.mean([r["rmsd_recon_to_x1"] for r in rows])) if rows else float("nan"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
