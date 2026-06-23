#!/usr/bin/env python
"""Train a direct x1-noise denoiser on the overfit subset.

This is intentionally not the OMatG/SI eta-denoiser.  It trains the normal
AdsorbGen x1 head to solve:

    x_noisy = x1 + gamma(t) z
    model(pos=x0_context, x_t=x_noisy, t=t) -> x1

The checkpoint is saved in the same lightweight format as Lightning runs:
``state_dict`` has ``model.*`` keys and ``hyper_parameters`` stores
``model_cfg``/``flow_cfg`` so existing loaders can read it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader


def install_imports(repo: Path, adsorbates_pkl: Path) -> None:
    os.environ["ADSGEN_ROOT"] = str(repo)
    os.environ["ADSORBATES_PKL"] = str(adsorbates_pkl)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def mean_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, loss_type: str) -> torch.Tensor:
    diff = pred - target
    per_atom = diff.abs().sum(dim=-1) if loss_type == "l1" else diff.pow(2).sum(dim=-1)
    m = mask.to(per_atom.dtype)
    denom = m.sum(dim=1).clamp_min(1.0)
    return ((per_atom * m).sum(dim=1) / denom).mean()


def save_ckpt(path: Path, model: torch.nn.Module, model_cfg, flow_cfg, args: argparse.Namespace,
              epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {f"model.{k}": v.detach().cpu() for k, v in model.state_dict().items()}
    hp = {
        "model_cfg": model_cfg,
        "flow_cfg": flow_cfg,
        "movable_mode": args.movable_mode,
        "slab_source": "initial",
        "pristine_slabs": "",
        "pristine_sid_index": "",
        "gamma_schedule": args.gamma_schedule,
        "gamma_sigma": args.gamma_sigma,
        "direct_x09_recon_denoiser": True,
        "recon_t": args.t,
        "loss_type": args.loss_type,
    }
    payload = {
        "state_dict": state,
        "hyper_parameters": hp,
        "epoch": int(epoch),
        "global_step": int(step),
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--train-lmdb", required=True)
    ap.add_argument("--val-lmdb", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adsorbates-pkl", default="")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--train-replicate", type=int, default=1000)
    ap.add_argument("--val-replicate", type=int, default=100)
    ap.add_argument("--max-train-samples", type=int, default=10)
    ap.add_argument("--max-val-samples", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="v0-ads-ref-adshead")
    ap.add_argument("--arch", default="v1", choices=["v1", "v2"])
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--pair-dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--t", type=float, default=0.9)
    ap.add_argument("--gamma-schedule", default="sqrt_t1mt", choices=["sqrt_t1mt", "linear_1mt"])
    ap.add_argument("--gamma-sigma", type=float, default=0.1)
    ap.add_argument("--flow-eps", type=float, default=1e-5)
    ap.add_argument("--loss-type", default="l1", choices=["l1", "l2"])
    ap.add_argument("--loss-surf-weight", type=float, default=10.0)
    ap.add_argument("--loss-ads-weight", type=float, default=1.0)
    ap.add_argument("--ads-pair-l1-weight", type=float, default=1.0)
    ap.add_argument("--ads-bond-factor", type=float, default=1.25)
    ap.add_argument("--ads-clash-factor", type=float, default=0.75)
    ap.add_argument("--movable-mode", default="surface_ads", choices=["surface_ads", "adsorbate_only"])
    ap.add_argument("--prior-mode", default="random_heuristic")
    ap.add_argument("--interstitial-gap", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every-n-epochs", type=int, default=50)
    ap.add_argument("--init-from-ckpt", default="")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    adsorbates = Path(args.adsorbates_pkl or (repo / "data" / "pkls" / "adsorbates.pkl"))
    install_imports(repo, adsorbates)

    from adsorbgen.data.dataset import PlacementPriorDataset, collate_displacement
    from adsorbgen.flow import FlowConfig, adsorbate_pair_distance_losses, si_gamma
    from adsorbgen.models.factory import build_model
    from adsorbgen.training.train_cli import build_config

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_args = SimpleNamespace(
        arch=args.arch, variant=args.variant, dropout=args.dropout,
        activation_checkpointing=False, dim=args.dim, pair_dim=args.pair_dim,
        depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        use_langevin_param=False, use_si_denoiser=False,
        si_denoiser_loss_weight=0.0,
    )
    model_cfg = build_config(cfg_args)
    flow_cfg = FlowConfig(eps=float(args.flow_eps), prediction_type="x1")
    model = build_model(model_cfg).to(device)
    if args.init_from_ckpt:
        payload = torch.load(args.init_from_ckpt, map_location="cpu", weights_only=False)
        state = {k[len("model."):]: v for k, v in payload["state_dict"].items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] loaded {args.init_from_ckpt} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    provide_ref = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    train_ds = PlacementPriorDataset(
        args.train_lmdb,
        prior_mode=args.prior_mode,
        interstitial_gap=args.interstitial_gap,
        adsorbates_pkl=str(adsorbates),
        max_samples=args.max_train_samples,
        training_aug=True,
        translation_std=0.5,
        provide_ads_ref_pos=provide_ref,
        skip_anomaly=False,
    )
    if args.train_replicate > 1:
        train_ds = ConcatDataset([train_ds] * int(args.train_replicate))
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_displacement,
        pin_memory=True,
        drop_last=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.precision.startswith("bf16") and device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    (out / "model_config.json").write_text(json.dumps(asdict(model_cfg), indent=2, sort_keys=True))

    global_step = 0
    for epoch in range(int(args.epochs)):
        model.train()
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pos = batch["pos"]
            x1 = batch["pos_relaxed"]
            tags = batch["tags"]
            pad = batch["pad_mask"]
            movable = batch["movable_mask"]
            if args.movable_mode == "adsorbate_only":
                movable = movable & (tags == 2)
            mov_f = movable.unsqueeze(-1).to(pos.dtype)
            pad_f = pad.unsqueeze(-1).to(pos.dtype)
            B = pos.shape[0]
            t = torch.full((B,), float(args.t), device=device, dtype=pos.dtype)
            gamma = si_gamma(t, args.gamma_schedule, args.gamma_sigma, eps=flow_cfg.eps).view(B, 1, 1).to(pos.dtype)
            z = torch.randn_like(x1) * mov_f
            x_noisy = x1 + gamma * z
            x_noisy = x_noisy * mov_f + pos * (1.0 - mov_f)
            x_noisy = x_noisy * pad_f

            kwargs = {}
            if provide_ref:
                kwargs["ads_ref_pos"] = batch["ads_ref_pos"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(
                    pos=pos,
                    x_t=x_noisy,
                    t=t,
                    atomic_numbers=batch["atomic_numbers"],
                    tags=tags,
                    movable_mask=movable,
                    pad_mask=pad,
                    cell=batch["cell"],
                    **kwargs,
                )
                pred_main = pred["pred_x1"] if isinstance(pred, dict) else pred
                surf_mask = movable & (tags == 1)
                ads_mask = movable & (tags == 2)
                surf_loss = mean_loss(pred_main, x1, surf_mask, args.loss_type)
                ads_loss = mean_loss(pred_main, x1, ads_mask, args.loss_type)
                loss = args.loss_surf_weight * surf_loss + args.loss_ads_weight * ads_loss
                if args.ads_pair_l1_weight != 0.0:
                    pair = adsorbate_pair_distance_losses(
                        pred_main,
                        batch.get("ads_ref_pos", x1),
                        batch["atomic_numbers"],
                        ads_mask,
                        bond_factor=args.ads_bond_factor,
                        clash_factor=args.ads_clash_factor,
                    )
                    loss = loss + args.ads_pair_l1_weight * pair["ads_pair_l1"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            global_step += 1
            if global_step % int(args.log_every) == 0:
                print(
                    json.dumps({
                        "epoch": epoch,
                        "step": global_step,
                        "loss": float(loss.detach().cpu()),
                        "surf": float(surf_loss.detach().cpu()),
                        "ads": float(ads_loss.detach().cpu()),
                    }),
                    flush=True,
                )
        if (epoch + 1) % int(args.save_every_n_epochs) == 0:
            save_ckpt(out / f"ckpt_epochepoch={epoch:03d}.ckpt", model, model_cfg, flow_cfg, args, epoch, global_step)
        save_ckpt(out / "last.ckpt", model, model_cfg, flow_cfg, args, epoch, global_step)

    print(f"[done] wrote {out / 'last.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
