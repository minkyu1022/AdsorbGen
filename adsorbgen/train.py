"""Lightning training script for the AdsorbGen flow matching DiT.

Usage:

    # Single-GPU smoke
    PYTHONPATH=AdsorbGen python -m adsorbgen.train \
        --train-lmdb data/processed/is2res_train.lmdb \
                     data/processed/is2res_val_ood_ads.lmdb \
                     data/processed/is2res_val_ood_cat.lmdb \
                     data/processed/is2res_val_ood_both.lmdb \
        --val-lmdb   data/processed/oc20dense_val.lmdb \
        --out runs/v2 --epochs 1 --batch-size 4

    # Multi-GPU DDP (Lightning handles torchrun automatically)
    PYTHONPATH=AdsorbGen python -m adsorbgen.train --devices 4 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.dataset import PreprocessedDisplacementDataset, collate_displacement  # noqa: E402
from adsorbgen.eval import compute_metrics  # noqa: E402
from adsorbgen.flow import FlowConfig, compute_delta1, corrupt, euler_sample, sample_t, x1_loss  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2Config  # noqa: E402


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class AdsorbGenModule(L.LightningModule):
    def __init__(
        self,
        model_cfg,
        flow_cfg: FlowConfig,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lr_warmup_steps: int = 500,
        loss_type: str = "l1",
        sample_eval_every_epochs: int = 0,
        sample_eval_max_samples: int = 64,
        sample_eval_steps: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_cfg)
        self.flow_cfg = flow_cfg
        self._sample_eval_records: list[dict] = []

    def forward(self, **kwargs):
        return self.model(**kwargs)

    # -- training --

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    # -- validation --

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        # accumulate sample-eval records (rank 0 only, every N epochs)
        hp = self.hparams
        if (
            hp.sample_eval_every_epochs > 0
            and (self.current_epoch + 1) % hp.sample_eval_every_epochs == 0
            and self.global_rank == 0
            and (hp.sample_eval_max_samples <= 0 or len(self._sample_eval_records) < hp.sample_eval_max_samples)
        ):
            self._accumulate_sample_eval(batch)

    def on_validation_epoch_end(self):
        if not self._sample_eval_records:
            return
        metrics = compute_metrics(self._sample_eval_records)
        agg = metrics["aggregate"]
        self.log("sample_eval/valid_rate", agg["valid_rate"], rank_zero_only=True)
        self.log("sample_eval/mae", agg["displacement_mae_A"], rank_zero_only=True)
        self.log("sample_eval/overlap", agg["overlap_rate"], rank_zero_only=True)
        self.log("sample_eval/dissoc", agg["dissociation_rate"], rank_zero_only=True)
        self._sample_eval_records.clear()

    # -- optimizer --

    def configure_optimizers(self):
        hp = self.hparams
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay,
        )
        warmup = max(int(hp.lr_warmup_steps), 0)

        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # -- helpers --

    def _compute_loss(self, batch):
        pos, pos_rel = batch["pos"], batch["pos_relaxed"]
        cell, movable = batch["cell"], batch["movable_mask"]

        delta1 = compute_delta1(pos, pos_rel, cell, movable)
        t = sample_t(pos.shape[0], self.flow_cfg, device=pos.device, dtype=pos.dtype)
        delta_t, _ = corrupt(delta1, t, self.flow_cfg, movable)

        pred = self.model(
            pos=pos, delta_t=delta_t, t=t,
            atomic_numbers=batch["atomic_numbers"], tags=batch["tags"],
            movable_mask=movable, pad_mask=batch["pad_mask"],
            cell=cell,
        )
        return x1_loss(pred, delta1, movable, loss_type=self.hparams.loss_type)

    @torch.no_grad()
    def _accumulate_sample_eval(self, batch):
        hp = self.hparams
        max_s = hp.sample_eval_max_samples
        if max_s > 0 and len(self._sample_eval_records) >= max_s:
            return

        B = batch["pos"].shape[0]
        take = min(B, max_s - len(self._sample_eval_records)) if max_s > 0 else B

        def model_forward(delta_t, t):
            return self.model(
                pos=batch["pos"], delta_t=delta_t, t=t,
                atomic_numbers=batch["atomic_numbers"], tags=batch["tags"],
                movable_mask=batch["movable_mask"], pad_mask=batch["pad_mask"],
                cell=batch["cell"],
            )

        x_out = euler_sample(
            model_forward, batch["pos"], batch["cell"],
            batch["movable_mask"], batch["pad_mask"], self.flow_cfg,
            num_steps=hp.sample_eval_steps,
        )
        for i in range(take):
            n = int(batch["pad_mask"][i].sum().item())
            self._sample_eval_records.append({
                "pos_pred": x_out[i, :n].cpu(),
                "pos_gt": batch["pos_relaxed"][i, :n].cpu(),
                "movable_mask": batch["movable_mask"][i, :n].cpu(),
                "atomic_numbers": batch["atomic_numbers"][i, :n].cpu(),
                "tags": batch["tags"][i, :n].cpu(),
            })


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class AdsorbGenDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        a = self.args
        train_paths = a.train_lmdb if isinstance(a.train_lmdb, list) else [a.train_lmdb]
        train_parts = [
            PreprocessedDisplacementDataset(
                p,
                max_samples=a.max_train_samples,
                training_aug=True,
                translation_std=a.translation_std,
            )
            for p in train_paths
        ]
        self.train_ds = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        self.val_ds = (
            PreprocessedDisplacementDataset(
                a.val_lmdb,
                max_samples=a.max_val_samples,
                training_aug=False,
            )
            if a.val_lmdb else None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.num_workers,
            collate_fn=collate_displacement, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return None
        return DataLoader(
            self.val_ds, batch_size=self.args.batch_size,
            shuffle=False, num_workers=self.args.num_workers,
            collate_fn=collate_displacement, pin_memory=True, drop_last=False,
        )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args) -> DiTDenoiserV2Config:
    return DiTDenoiserV2Config(
        dim=args.dim,
        pair_dim=args.pair_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        sigma=args.sigma,
        activation_checkpointing=args.activation_checkpointing,
    )


def _save_args_json(out_dir: Path, model_cfg: DiTDenoiserV2Config) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arch": "v2", "model_config": asdict(model_cfg)}
    with open(out_dir / "args.json", "w") as f:
        json.dump(payload, f, indent=2)


def _check_resume_arch(out_dir: Path, requested_arch: str) -> None:
    """Fail-fast if an existing run in ``out_dir`` was trained with a different arch."""
    args_path = out_dir / "args.json"
    if not args_path.exists():
        return
    with open(args_path) as f:
        a = json.load(f)
    existing = a.get("arch", "v1")
    if existing != requested_arch:
        raise RuntimeError(
            f"arch mismatch: out_dir has arch={existing!r}, but --arch={requested_arch!r} given. "
            f"Use a different --out, or remove the directory."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-lmdb", type=str, nargs="+", required=True,
                   help="one or more preprocessed train LMDBs (concatenated)")
    p.add_argument("--val-lmdb", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--lr-warmup-steps", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--devices", type=int, default=None,
                   help="number of GPUs (default: auto-detect)")

    # Model (v2)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--pair-dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=13)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=0.423)
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--translation-std", type=float, default=0.5)

    # Training
    p.add_argument("--loss-type", choices=["l1", "l2"], default="l1")
    p.add_argument("--flow-eps", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)

    # Sample eval
    p.add_argument("--sample-eval-every-epochs", type=int, default=0)
    p.add_argument("--sample-eval-max-samples", type=int, default=64)
    p.add_argument("--sample-eval-steps", type=int, default=10)

    # Logging
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)

    args = p.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    torch.serialization.add_safe_globals([DiTDenoiserV2Config, FlowConfig])

    model_cfg = build_config(args)
    flow_cfg = FlowConfig(sigma=args.sigma, eps=args.flow_eps)

    out_dir = Path(args.out)
    _check_resume_arch(out_dir, "v2")
    resume_ckpt = None
    last_ckpt = out_dir / "last.ckpt"

    module = AdsorbGenModule(
        model_cfg=model_cfg,
        flow_cfg=flow_cfg,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_warmup_steps=args.lr_warmup_steps,
        loss_type=args.loss_type,
        sample_eval_every_epochs=args.sample_eval_every_epochs,
        sample_eval_max_samples=args.sample_eval_max_samples,
        sample_eval_steps=args.sample_eval_steps,
    )

    if last_ckpt.exists():
        resume_ckpt = str(last_ckpt)
        print(f"[resume] {resume_ckpt}", flush=True)

    dm = AdsorbGenDataModule(args)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out_dir),
            filename="ckpt_epoch{epoch:03d}",
            save_last=True,
            every_n_epochs=1,
            save_top_k=-1,
        ),
        RichModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = None
    if args.wandb_project:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            save_dir=str(out_dir),
        )

    # Trainer
    n_gpus = args.devices or torch.cuda.device_count() or 1
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=n_gpus,
        strategy="ddp" if n_gpus > 1 else "auto",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        default_root_dir=str(out_dir),
        log_every_n_steps=args.log_every,
        enable_progress_bar=True,
    )

    if not (out_dir / "args.json").exists():
        _save_args_json(out_dir, model_cfg)

    trainer.fit(module, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
