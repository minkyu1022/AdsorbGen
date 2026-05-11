"""Lightning training script for the AdsorbGen flow matching DiT.

Usage:

    # Single-GPU smoke
    PYTHONPATH=AdsorbGen python -m adsorbgen.train \
        --train-lmdb data/processed/is2res_train.lmdb \
                     data/processed/is2res_val_ood_ads.lmdb \
                     data/processed/is2res_val_ood_cat.lmdb \
                     data/processed/is2res_val_ood_both.lmdb \
        --val-lmdb   data/processed/oc20dense.lmdb \
        --out runs/v2 --epochs 1 --batch-size 4

    # Multi-GPU DDP (Lightning handles torchrun automatically)
    PYTHONPATH=AdsorbGen python -m adsorbgen.train --devices 4 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from datetime import timedelta

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelCheckpoint, RichModelSummary, RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader

_REPO = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(os.environ.get("CAT_BENCH_ROOT", str(_REPO.parent)))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from adsorbgen.dataset import (  # noqa: E402
    MixedReplayDataset, PlacementPriorDataset, PreprocessedDisplacementDataset,
    collate_displacement,
)
from adsorbgen.eval import compute_anomaly_metrics, compute_displacement_metrics  # noqa: E402
from adsorbgen.eval_replay import (  # noqa: E402
    ReplayEvalConfig, ReplayScheduler, run_replay_eval,
)
from adsorbgen.flow import (  # noqa: E402
    adsorbate_pair_distance_losses,
    FlowConfig, euler_sample, flow_loss_split, interpolate_xt, sample_t,
    smooth_lddt_loss, x1_loss, x1_loss_split,
)
from adsorbgen.model import DiTDenoiserConfig  # noqa: E402
from adsorbgen.model_factory import build_model  # noqa: E402
from adsorbgen.model_v2 import DiTDenoiserV2Config  # noqa: E402
from adsorbgen.replay import ReplayBuffer  # noqa: E402
from adsorbgen.variants import get_variant, list_variants  # noqa: E402


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
        pristine_slabs: str = "",
        pristine_sid_index: str = "",
        loss_surf_weight: float = 5.0,
        loss_ads_weight: float = 1.0,
        lddt_ads_ads_weight: float = 0.0,
        lddt_cutoff: float = 15.0,
        lddt_time_weight: float = 0.0,
        ads_pair_l1_weight: float = 0.0,
        ads_bond_l1_weight: float = 0.0,
        ads_nonbonded_clash_weight: float = 0.0,
        ads_bond_factor: float = 1.25,
        ads_clash_factor: float = 0.75,
        # --- replay params ---
        use_replay: bool = False,
        replay_buffer_path: str = "",      # runs/<out>/replay_buffer.pkl
        replay_gt_index_path: str = "",
        replay_train_lmdb: str = "",       # for eval: lmdb to sample systems from
        replay_mode: str = "append",
        replay_ratio: float = 0.5,
        replay_eval_every: int = 30,
        replay_warmup_epochs: int = 30,
        replay_success_margin: float = 0.05,
        replay_cap: int = 1_070_000,
        replay_per_system_cap: int = 10,
        replay_weight_mode: str = "improvement",
        replay_initial_systems: int = 500,
        replay_initial_placements: int = 3,
        replay_scaled_systems: int = 2000,
        replay_scaled_placements: int = 5,
        replay_prior_mode: str = "random_heuristic",
        replay_uma_model: str = "uma-s-1p1",
        replay_uma_fmax: float = 0.05,
        replay_uma_max_steps: int = 100,
        replay_flow_steps: int = 50,
        replay_overlap_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_cfg)
        self.flow_cfg = flow_cfg
        # Per-source sample_eval records keyed by val-dataloader name
        # (e.g. "dense", "is2re"). Populated in validation_step and drained
        # in on_validation_epoch_end. Source names come from the
        # DataModule's ``val_names`` list.
        self._sample_eval_records: dict[str, list[dict]] = {}

        # Replay state (rank 0 owns)
        self._replay_buffer = None
        self._replay_scheduler = None
        self._replay_gt_index = None
        # Tracks the epoch (`current_epoch`) at which the most recent replay
        # eval *successfully completed*. Persisted via on_save_checkpoint, so
        # if a run is killed mid-replay (or mid-epoch after a replay), the
        # restart will retrigger the missed cycle as soon as
        # ``current_epoch - _last_replay_epoch >= replay_eval_every``.
        self._last_replay_epoch: int = -1
        if use_replay:
            buf_path = Path(replay_buffer_path)
            if buf_path.exists():
                self._replay_buffer = ReplayBuffer.load(buf_path)
                print(f"[replay] loaded {len(self._replay_buffer)} entries from {buf_path}")
            else:
                self._replay_buffer = ReplayBuffer(
                    mode=replay_mode, per_system_cap=replay_per_system_cap,
                    global_cap=replay_cap, weight_mode=replay_weight_mode,
                )
            # GT index
            if replay_gt_index_path:
                import pickle as _pkl
                with open(replay_gt_index_path, "rb") as f:
                    self._replay_gt_index = _pkl.load(f)
                print(f"[replay] loaded GT index: {len(self._replay_gt_index)} sids")
            viz_root = str(Path(replay_buffer_path).parent / "replay_viz") if replay_buffer_path else ""
            self._replay_scheduler = ReplayScheduler(
                initial=ReplayEvalConfig(
                    prior_mode=replay_prior_mode,
                    num_systems=replay_initial_systems,
                    num_placements=replay_initial_placements,
                    flow_steps=replay_flow_steps,
                    uma_model=replay_uma_model,
                    uma_fmax=replay_uma_fmax,
                    uma_max_steps=replay_uma_max_steps,
                    overlap_threshold=replay_overlap_threshold,
                    success_margin=replay_success_margin,
                    viz_root=viz_root,
                ),
                scaled=ReplayEvalConfig(
                    prior_mode=replay_prior_mode,
                    num_systems=replay_scaled_systems,
                    num_placements=replay_scaled_placements,
                    flow_steps=replay_flow_steps,
                    uma_model=replay_uma_model,
                    uma_fmax=replay_uma_fmax,
                    uma_max_steps=replay_uma_max_steps,
                    overlap_threshold=replay_overlap_threshold,
                    success_margin=replay_success_margin,
                    viz_root=viz_root,
                ),
            )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    # -- training --

    def training_step(self, batch, batch_idx):
        losses = self._compute_loss_split(batch)
        weighted = self._weighted_loss(losses)
        self.log("train/loss", weighted, prog_bar=True)         # backward target
        self.log("train/coord_loss", self._weighted_coord_loss(losses))
        self.log("train/loss_unweighted", losses["total"])      # all-movable mean, reference
        self.log("train/surf_loss", losses["surf"])
        self.log("train/ads_loss", losses["ads"])
        if "lddt_ads_ads" in losses:
            self.log("train/lddt_ads_ads_loss", losses["lddt_ads_ads"])
        if "ads_pair_l1" in losses:
            self.log("train/ads_pair_l1_loss", losses["ads_pair_l1"])
        if "ads_bond_l1" in losses:
            self.log("train/ads_bond_l1_loss", losses["ads_bond_l1"])
        if "ads_nonbonded_clash" in losses:
            self.log("train/ads_nonbonded_clash_loss", losses["ads_nonbonded_clash"])
        return weighted

    # -- validation --

    def _val_source_name(self, dataloader_idx: int) -> str:
        names = getattr(self.trainer.datamodule, "val_names", None) if self.trainer is not None else None
        if names and 0 <= dataloader_idx < len(names):
            return str(names[dataloader_idx])
        return f"src{dataloader_idx}"

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        src = self._val_source_name(dataloader_idx)
        losses = self._compute_loss_split(batch)
        weighted = self._weighted_loss(losses)
        # Always tag metrics by source so per-loader curves are independent.
        self.log(f"val/{src}/loss", weighted, prog_bar=(dataloader_idx == 0), sync_dist=True, add_dataloader_idx=False)
        self.log(f"val/{src}/coord_loss", self._weighted_coord_loss(losses), sync_dist=True, add_dataloader_idx=False)
        self.log(f"val/{src}/loss_unweighted", losses["total"], sync_dist=True, add_dataloader_idx=False)
        self.log(f"val/{src}/surf_loss", losses["surf"], sync_dist=True, add_dataloader_idx=False)
        self.log(f"val/{src}/ads_loss", losses["ads"], sync_dist=True, add_dataloader_idx=False)
        if "lddt_ads_ads" in losses:
            self.log(f"val/{src}/lddt_ads_ads_loss", losses["lddt_ads_ads"], sync_dist=True, add_dataloader_idx=False)
        if "ads_pair_l1" in losses:
            self.log(f"val/{src}/ads_pair_l1_loss", losses["ads_pair_l1"], sync_dist=True, add_dataloader_idx=False)
        if "ads_bond_l1" in losses:
            self.log(f"val/{src}/ads_bond_l1_loss", losses["ads_bond_l1"], sync_dist=True, add_dataloader_idx=False)
        if "ads_nonbonded_clash" in losses:
            self.log(f"val/{src}/ads_nonbonded_clash_loss", losses["ads_nonbonded_clash"], sync_dist=True, add_dataloader_idx=False)

        # accumulate sample-eval records on EVERY rank (each rank owns its
        # DDP shard of the val data); on_validation_epoch_end then all_reduces
        # counts across ranks so the logged metric is over all val samples.
        hp = self.hparams
        recs = self._sample_eval_records.setdefault(src, [])
        if (
            hp.sample_eval_every_epochs > 0
            and (self.current_epoch + 1) % hp.sample_eval_every_epochs == 0
            and (hp.sample_eval_max_samples <= 0 or len(recs) < hp.sample_eval_max_samples)
        ):
            self._accumulate_sample_eval(batch, src)

    # -- replay hook --
    def on_train_epoch_end(self):
        hp = self.hparams
        if not hp.use_replay:
            return
        ep = self.current_epoch
        warmup = int(hp.replay_warmup_epochs)
        every = int(hp.replay_eval_every)
        # Past warmup?
        if ep + 1 < warmup:
            return
        # Strict aligned schedule: fire at end of epochs where ep+1 is a
        # multiple of `every` (so first replay at ep=9 with warmup=10/every=10,
        # then ep=19, 29, 39, ...).
        strict_fire = (ep + 1) % every == 0
        # One-time recovery: if a previous successful replay exists but we
        # passed a strict trigger since then without firing (e.g. killed
        # mid-replay), fire once to catch up. After this fires the next
        # replay re-aligns to the strict schedule.
        recovery_fire = False
        if self._last_replay_epoch >= 0 and not strict_fire:
            epochs_since = ep - self._last_replay_epoch
            if epochs_since >= every:
                for missed in range(self._last_replay_epoch + 1, ep + 1):
                    if (missed + 1) % every == 0:
                        recovery_fire = True
                        break
        if not (strict_fire or recovery_fire):
            return

        # Save checkpoint BEFORE running the long UMA relax eval. Lightning's
        # ModelCheckpoint callback only saves after on_train_epoch_end returns,
        # so if we get killed during replay, the epoch would otherwise be lost
        # and we'd redo its training. This explicit save guarantees the
        # just-finished epoch's weights are on disk before the eval starts.
        pre_replay_ckpt = Path(hp.replay_buffer_path).parent / "last.ckpt"
        pre_replay_ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(str(pre_replay_ckpt))
        if self.global_rank == 0:
            print(f"[replay] pre-eval ckpt saved: {pre_replay_ckpt}", flush=True)

        # DDP-aware: rank 0 does eval + save; all others wait at barrier.
        if self.global_rank == 0:
            import torch.distributed as dist
            cfg = self._replay_scheduler.current_cfg()
            # fairchem's MLIPPredictUnit accepts only 'cpu' or 'cuda' (no index).
            # Rank 0's local CUDA device is already the default via torch.cuda.current_device().
            cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"\n[replay] epoch {ep+1}: eval start (systems={cfg.num_systems} × K={cfg.num_placements})", flush=True)
            # Source dataset: use the training LMDB (first if multiple)
            eval_ds = PreprocessedDisplacementDataset(
                hp.replay_train_lmdb, max_samples=None,
            )
            metrics = run_replay_eval(
                model=self.model,
                dataset=eval_ds,
                gt_index_by_sid=self._replay_gt_index,
                buffer=self._replay_buffer,
                cfg=cfg,
                flow_cfg=self.flow_cfg,
                epoch=ep,
            )
            print(f"[replay] metrics: {metrics}", flush=True)
            # persist buffer + log
            buf_path = Path(hp.replay_buffer_path)
            buf_path.parent.mkdir(parents=True, exist_ok=True)
            self._replay_buffer.save(buf_path)
            log_path = buf_path.parent / f"replay_new_samples_ep{ep+1}.pkl"
            import pickle as _pkl
            with open(log_path, "wb") as f:
                _pkl.dump(metrics, f)
            # log to W&B / logger
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.log(f"replay/{k}", float(v), rank_zero_only=True)
            for sk, sv in self._replay_buffer.stats().items():
                if isinstance(sv, (int, float)):
                    self.log(f"replay_buf/{sk}", float(sv), rank_zero_only=True)
            # scheduler updates
            self._replay_scheduler.record(metrics["n_added_to_buffer"])

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
        else:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        # Mark this epoch as having completed a replay eval so the next
        # trigger fires `replay_eval_every` epochs later. Done after the
        # barrier so a kill mid-replay doesn't advance the counter.
        self._last_replay_epoch = ep

    def on_save_checkpoint(self, checkpoint):
        # Persist replay schedule state across resume.
        checkpoint["last_replay_epoch"] = int(self._last_replay_epoch)

    def on_load_checkpoint(self, checkpoint):
        v = checkpoint.get("last_replay_epoch", -1)
        self._last_replay_epoch = int(v)

    def on_validation_epoch_end(self):
        # All ranks reach this hook. Each rank computes per-source counts on
        # its OWN DDP shard, then we all_reduce(sum) across ranks so the
        # logged metric reflects the full val set, not rank 0's slice.
        # Sources are gathered into a stable union (some ranks may be empty
        # for a given src if its shard happened to have 0 samples after
        # truncation) so all_reduce sees identical key sets across ranks.
        import torch.distributed as dist

        ddp_active = dist.is_available() and dist.is_initialized() and self.trainer.world_size > 1

        # Union of src names across ranks. Sources are deterministic from the
        # CLI (val_lmdb / val_lmdb_is2re) so all ranks see the same ordering.
        srcs = sorted(self._sample_eval_records.keys()) or sorted(getattr(self.trainer.datamodule, "val_names", []))
        if not srcs:
            return

        for src in srcs:
            recs = self._sample_eval_records.get(src, [])
            if recs:
                disp = compute_displacement_metrics(recs)
                _ps = getattr(self.hparams, "pristine_slabs", "") or None
                _psi = getattr(self.hparams, "pristine_sid_index", "") or None
                strict = compute_anomaly_metrics(
                    recs,
                    pristine_slabs=_ps,
                    pristine_sid_index=_psi,
                )
                sagg = strict["aggregate"]
                per = strict["per_sample"]
                n_local = int(sagg["n_samples"])
                # Recover integer counts from per-sample flags (avoids round
                # trip via rate*n_local which would lose to float drift).
                cnt = {
                    "overlap": sum(1 for p in per if p.get("has_overlap") is True),
                    "dissoc": sum(1 for p in per if p.get("has_dissoc") is True),
                    "desorbed": sum(1 for p in per if p.get("has_desorbed") is True),
                    "intercalated": sum(1 for p in per if p.get("has_intercalated") is True),
                    "surf_changed": sum(1 for p in per if p.get("has_surf_changed") is True),
                    "any_anomaly": sum(1 for p in per if p.get("is_any_anomaly") is True),
                    "n_errors": int(sagg.get("n_errors", 0)),
                }
                disp_sum = float(disp["aggregate"]["displacement_err_sum"])
                disp_sq_sum = float(disp["aggregate"]["displacement_err_sq_sum"])
                disp_count = int(disp["aggregate"]["displacement_err_count"])
            else:
                n_local = 0
                cnt = {k: 0 for k in ("overlap", "dissoc", "desorbed", "intercalated", "surf_changed", "any_anomaly", "n_errors")}
                disp_sum = 0.0
                disp_sq_sum = 0.0
                disp_count = 0

            # Pack into a single tensor for one all_reduce call.
            # Order: [n_samples, overlap, dissoc, desorbed, intercalated,
            #         surf_changed, any_anomaly, n_errors,
            #         disp_sum, disp_sq_sum, disp_count]
            packed = torch.tensor(
                [n_local, cnt["overlap"], cnt["dissoc"], cnt["desorbed"],
                 cnt["intercalated"], cnt["surf_changed"], cnt["any_anomaly"],
                 cnt["n_errors"], disp_sum, disp_sq_sum, disp_count],
                dtype=torch.float64, device=self.device,
            )
            if ddp_active:
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)

            n_global = int(packed[0].item())
            n_err_global = int(packed[7].item())
            assert n_err_global == 0, (
                f"[{src}] compute_anomaly_metrics encountered "
                f"{n_err_global}/{n_global} per-sample errors. fail-fast."
            )
            if n_global == 0:
                continue
            inv = 1.0 / n_global
            disp_count_global = int(packed[10].item())
            mae_global = float(packed[8].item()) / max(disp_count_global, 1)

            base = f"sample_eval/{src}"
            self.log(f"{base}/mae",                mae_global,                          rank_zero_only=True)
            self.log(f"{base}/valid_rate_strict",  1.0 - float(packed[6].item()) * inv, rank_zero_only=True)
            self.log(f"{base}/overlap_rate",       float(packed[1].item()) * inv,       rank_zero_only=True)
            self.log(f"{base}/dissoc_rate",        float(packed[2].item()) * inv,       rank_zero_only=True)
            self.log(f"{base}/desorbed_rate",      float(packed[3].item()) * inv,       rank_zero_only=True)
            self.log(f"{base}/intercalated_rate",  float(packed[4].item()) * inv,       rank_zero_only=True)
            self.log(f"{base}/surf_changed_rate",  float(packed[5].item()) * inv,       rank_zero_only=True)
            self.log(f"{base}/any_anomaly_rate",   float(packed[6].item()) * inv,       rank_zero_only=True)
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

    def _compute_loss_split(self, batch):
        """Returns dict with 'total', 'surf', 'ads' loss tensors.

        AtomMOF-style formulation:
          - Sample t ~ U(eps, 1-eps)
          - x_t = (1-t)*x_0 + t*x_1 on movable atoms (non-movable stay at x_0)
          - Model predicts x_1 directly
          - Loss = || pred - x_1 ||  on movable atoms
        """
        pos, pos_rel = batch["pos"], batch["pos_relaxed"]
        cell, movable = batch["cell"], batch["movable_mask"]
        tags = batch["tags"]

        t = sample_t(pos.shape[0], self.flow_cfg, device=pos.device, dtype=pos.dtype)
        x_t = interpolate_xt(pos, pos_rel, t, movable)

        model_kwargs = dict(
            pos=pos, x_t=x_t, t=t,
            atomic_numbers=batch["atomic_numbers"], tags=tags,
            movable_mask=movable, pad_mask=batch["pad_mask"],
            cell=cell,
        )
        if getattr(self._model_cfg(), "use_ads_ref_pos", False):
            model_kwargs["ads_ref_pos"] = batch["ads_ref_pos"]

        if getattr(self._model_cfg(), "use_self_cond", False) and torch.rand(()).item() < 0.5:
            with torch.no_grad():
                prev = self.model(**model_kwargs).detach()
            model_kwargs["prev_pred"] = prev

        pred = self.model(**model_kwargs)
        losses = flow_loss_split(
            pred, pos, pos_rel, movable, tags,
            loss_type=self.hparams.loss_type,
            prediction_type=self.flow_cfg.prediction_type,
        )
        lddt_w = float(getattr(self.hparams, "lddt_ads_ads_weight", 0.0))
        if lddt_w != 0.0:
            if self.flow_cfg.prediction_type == "x1":
                pred_x1 = pred
            elif self.flow_cfg.prediction_type == "v":
                pred_x1 = pos + pred
            else:
                raise ValueError(f"Unknown prediction_type={self.flow_cfg.prediction_type!r}")
            ads_mask = movable & (tags == 2)
            losses["lddt_ads_ads"] = smooth_lddt_loss(
                pred_x1,
                pos_rel,
                ads_mask,
                cutoff=float(getattr(self.hparams, "lddt_cutoff", 15.0)),
                t=t,
                time_weight=float(getattr(self.hparams, "lddt_time_weight", 0.0)),
            )
        pair_ws = (
            float(getattr(self.hparams, "ads_pair_l1_weight", 0.0)),
            float(getattr(self.hparams, "ads_bond_l1_weight", 0.0)),
            float(getattr(self.hparams, "ads_nonbonded_clash_weight", 0.0)),
        )
        if any(w != 0.0 for w in pair_ws):
            if self.flow_cfg.prediction_type == "x1":
                pred_x1 = pred
            elif self.flow_cfg.prediction_type == "v":
                pred_x1 = pos + pred
            else:
                raise ValueError(f"Unknown prediction_type={self.flow_cfg.prediction_type!r}")
            ads_mask = movable & (tags == 2)
            ref_coords = batch.get("ads_ref_pos", pos_rel)
            losses.update(adsorbate_pair_distance_losses(
                pred_x1,
                ref_coords,
                batch["atomic_numbers"],
                ads_mask,
                bond_factor=float(getattr(self.hparams, "ads_bond_factor", 1.25)),
                clash_factor=float(getattr(self.hparams, "ads_clash_factor", 0.75)),
            ))
        return losses

    def _compute_loss(self, batch):
        return self._compute_loss_split(batch)["total"]

    def _weighted_coord_loss(self, losses):
        sw = float(self.hparams.loss_surf_weight)
        aw = float(self.hparams.loss_ads_weight)
        return sw * losses["surf"] + aw * losses["ads"]

    def _weighted_loss(self, losses):
        weighted = self._weighted_coord_loss(losses)
        if "lddt_ads_ads" in losses:
            weighted = weighted + float(self.hparams.lddt_ads_ads_weight) * losses["lddt_ads_ads"]
        if "ads_pair_l1" in losses:
            weighted = weighted + float(self.hparams.ads_pair_l1_weight) * losses["ads_pair_l1"]
        if "ads_bond_l1" in losses:
            weighted = weighted + float(self.hparams.ads_bond_l1_weight) * losses["ads_bond_l1"]
        if "ads_nonbonded_clash" in losses:
            weighted = weighted + float(self.hparams.ads_nonbonded_clash_weight) * losses["ads_nonbonded_clash"]
        return weighted

    def _model_cfg(self):
        # self.model is wrapped (DDP / Lightning); unwrap to read cfg fields.
        m = self.model
        while hasattr(m, "module"):
            m = m.module
        return getattr(m, "cfg", None)

    @torch.no_grad()
    def _accumulate_sample_eval(self, batch, src: str):
        hp = self.hparams
        recs = self._sample_eval_records.setdefault(src, [])
        max_s = hp.sample_eval_max_samples
        if max_s > 0 and len(recs) >= max_s:
            return

        B = batch["pos"].shape[0]
        take = min(B, max_s - len(recs)) if max_s > 0 else B

        use_self_cond = getattr(self._model_cfg(), "use_self_cond", False)
        use_ads_ref = getattr(self._model_cfg(), "use_ads_ref_pos", False)
        state = {"prev_pred": None}

        def model_forward(x_t, t):
            extra = {}
            if use_self_cond:
                extra["prev_pred"] = state["prev_pred"]
            if use_ads_ref:
                extra["ads_ref_pos"] = batch["ads_ref_pos"]
            out = self.model(
                pos=batch["pos"], x_t=x_t, t=t,
                atomic_numbers=batch["atomic_numbers"], tags=batch["tags"],
                movable_mask=batch["movable_mask"], pad_mask=batch["pad_mask"],
                cell=batch["cell"],
                **extra,
            )
            if use_self_cond:
                state["prev_pred"] = out.detach()
            return out

        x_out = euler_sample(
            model_forward, batch["pos"],
            batch["movable_mask"], batch["pad_mask"], self.flow_cfg,
            num_steps=hp.sample_eval_steps,
        )
        for i in range(take):
            n = int(batch["pad_mask"][i].sum().item())
            cell_i = batch["cell"][i].cpu()
            if cell_i.dim() == 3:
                cell_i = cell_i[0]
            sid_i = int(batch["sid"][i].item()) if "sid" in batch else -1
            system_key_i = batch.get("system_key", [None] * B)[i]
            config_key_i = batch.get("config_key", [None] * B)[i]
            recs.append({
                "sid": sid_i,
                "system_key": system_key_i,
                "config_key": config_key_i,
                "pos_pred": x_out[i, :n].cpu(),
                "pos_gt": batch["pos_relaxed"][i, :n].cpu(),
                "pos_ref": batch["pos"][i, :n].cpu(),
                "movable_mask": batch["movable_mask"][i, :n].cpu(),
                "atomic_numbers": batch["atomic_numbers"][i, :n].cpu(),
                "tags": batch["tags"][i, :n].cpu(),
                "cell": cell_i,
            })


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class AdsorbGenDataModule(L.LightningDataModule):
    def __init__(self, args, replay_buffer=None):
        super().__init__()
        self.args = args
        self.replay_buffer = replay_buffer
        self.train_ds = None
        self.val_ds = None
        self._train_base = None

    def setup(self, stage=None):
        a = self.args
        provide_ref = bool(getattr(a, "use_ads_ref_pos", False))
        train_paths = a.train_lmdb if isinstance(a.train_lmdb, list) else [a.train_lmdb]
        include_anomaly = bool(getattr(a, "include_anomaly", False))
        train_parts = [
            PlacementPriorDataset(
                p,
                max_samples=a.max_train_samples,
                training_aug=True,
                translation_std=a.translation_std,
                prior_mode=getattr(a, "prior_mode", "random_heuristic"),
                interstitial_gap=getattr(a, "interstitial_gap", 0.1),
                provide_ads_ref_pos=provide_ref,
                skip_anomaly=not include_anomaly,
            )
            for p in train_paths
        ]
        self._train_base = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
        replicate = int(getattr(a, "train_replicate", 1) or 1)
        if replicate > 1:
            self._train_base = ConcatDataset([self._train_base] * replicate)
        if getattr(a, "use_replay", False) and self.replay_buffer is not None:
            # Pick any PlacementPriorDataset as placement helper for replay
            # sample construction (works with both single and ConcatDataset).
            helper = train_parts[0]
            self.train_ds = MixedReplayDataset(
                self._train_base, self.replay_buffer,
                alpha=getattr(a, "replay_ratio", 0.5),
                rng_seed=getattr(a, "seed", 0),
                placement_helper=helper,
            )
        else:
            self.train_ds = self._train_base

        # ---- Validation datasets ----
        # Two val loaders: (0) Dense + random_heuristic placement,
        #                  (1) IS2RE + random_heuristic placement.
        # Both use a fresh placement to mirror replay-style inference.
        # Single-loader fallback if --val-lmdb-is2re isn't given.
        self.val_datasets: list = []
        self.val_names: list = []
        if a.val_lmdb:
            self.val_datasets.append(PlacementPriorDataset(
                a.val_lmdb,
                max_samples=a.max_val_samples,
                training_aug=False,
                prior_mode=getattr(a, "prior_mode", "random_heuristic"),
                interstitial_gap=getattr(a, "interstitial_gap", 0.1),
                provide_ads_ref_pos=provide_ref,
            ))
            self.val_names.append("dense")
        is2re_path = getattr(a, "val_lmdb_is2re", None)
        if is2re_path:
            self.val_datasets.append(PlacementPriorDataset(
                is2re_path,
                max_samples=a.max_val_samples,
                training_aug=False,
                prior_mode=getattr(a, "prior_mode", "random_heuristic"),
                interstitial_gap=getattr(a, "interstitial_gap", 0.1),
                provide_ads_ref_pos=provide_ref,
            ))
            self.val_names.append("is2re")
        val_replicate = int(getattr(a, "val_replicate", 1) or 1)
        if val_replicate > 1:
            self.val_datasets = [
                ConcatDataset([ds] * val_replicate) for ds in self.val_datasets
            ]
        # Back-compat alias.
        self.val_ds = self.val_datasets[0] if self.val_datasets else None

    def train_dataloader(self):
        # Use spawn context + persistent_workers to avoid wandb/DDP state
        # leaking into forked workers. fork-inherited wandb state deadlocks
        # when combined with fairchem AdsorbateSlabConfig calls in workers.
        extra = {}
        if self.args.num_workers > 0:
            extra = dict(multiprocessing_context="spawn", persistent_workers=True)
        return DataLoader(
            self.train_ds, batch_size=self.args.batch_size,
            shuffle=True, num_workers=self.args.num_workers,
            collate_fn=collate_displacement, pin_memory=True, drop_last=True,
            **extra,
        )

    def val_dataloader(self):
        if not self.val_datasets:
            return None
        extra = {}
        if self.args.num_workers > 0:
            extra = dict(multiprocessing_context="spawn", persistent_workers=True)
        # Default sampler -> Lightning auto-wraps with DistributedSampler under
        # DDP, so each rank gets its own shard. on_validation_epoch_end
        # all_reduces counts across ranks to recover the global metric.
        loaders = [
            DataLoader(
                ds, batch_size=self.args.batch_size,
                shuffle=False, num_workers=self.args.num_workers,
                collate_fn=collate_displacement, pin_memory=True, drop_last=False,
                **extra,
            )
            for ds in self.val_datasets
        ]
        return loaders[0] if len(loaders) == 1 else loaders


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_config(args):
    if args.arch == "v1":
        # v1 legacy hparams (runs/v1/args.json) — retrain baseline for
        # apples-to-apples comparison under v2's data protocol.
        cfg = DiTDenoiserConfig(
            atom_s=256, atom_z=128, token_s=512, token_z=256,
            enc_depth=2, trunk_depth=16, dec_depth=2,
            enc_heads=4, trunk_heads=8, dec_heads=4,
            mlp_ratio=4.0, dropout=args.dropout,
            activation_checkpointing=args.activation_checkpointing,
        )
        if args.variant and args.variant not in ("v2", "v1-retrained"):
            overrides = get_variant(args.variant)
            for k, v in overrides.items():
                if not hasattr(cfg, k):
                    raise ValueError(
                        f"variant {args.variant!r} sets unknown v1 field {k!r}"
                    )
                setattr(cfg, k, v)
        return cfg
    cfg = DiTDenoiserV2Config(
        dim=args.dim,
        pair_dim=args.pair_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        activation_checkpointing=args.activation_checkpointing,
    )
    if args.variant and args.variant != "v2":
        overrides = get_variant(args.variant)
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"variant {args.variant!r} sets unknown field {k!r}")
            setattr(cfg, k, v)
    return cfg


def _jsonable_args(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        else:
            out[k] = v
    return out


def _save_args_json(out_dir: Path, arch: str, model_cfg,
                    args: argparse.Namespace | None = None,
                    flow_cfg: FlowConfig | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arch": arch, "model_config": asdict(model_cfg)}
    if flow_cfg is not None:
        payload["flow_config"] = asdict(flow_cfg)
    if args is not None:
        payload["train_args"] = _jsonable_args(args)
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
    p.add_argument("--val-lmdb", type=str, default=None,
                   help="primary val LMDB (logged as val/dense/* and sample_eval/dense/*)")
    p.add_argument("--val-lmdb-is2re", type=str, default=None,
                   help="optional second val LMDB (logged as val/is2re/* and sample_eval/is2re/*). "
                        "Both subsets use random_heuristic placement to match replay-style inference.")
    p.add_argument("--check-val-every-n-epoch", type=int, default=1,
                   help="run validation (and sample_eval, if its own gate fires) every N epochs.")
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--train-replicate", type=int, default=1,
                   help="replicate the (post-truncation) train dataset K times "
                        "via ConcatDataset. Each replica re-fires random_heuristic "
                        "placement on access, so same system seen K times per "
                        "epoch with different (placement, t) samples. Used for "
                        "small-data overfitting sanity checks.")
    p.add_argument("--val-replicate", type=int, default=1,
                   help="replicate the (post-truncation) val dataset K times "
                        "via ConcatDataset. Each replica re-fires random_heuristic "
                        "placement on access, so the same N val systems are "
                        "scored under K different placements (N*K total samples "
                        "per validation pass). Combined with a non-distributed "
                        "val sampler so rank 0 sees all N*K records.")
    p.add_argument("--include-anomaly", action="store_true",
                   help="train on the full LMDB including samples flagged as "
                        "anomaly (skip_anomaly=False). Validation always "
                        "filters anomalies for clean comparison.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
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
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--translation-std", type=float, default=0.5)
    p.add_argument("--prior-mode",
                   choices=["random", "heuristic", "random_heuristic"],
                   default="random_heuristic",
                   help="Placement prior for x_0: fairchem placement mode")
    p.add_argument("--interstitial-gap", type=float, default=0.1,
                   help="fairchem placement interstitial gap (Å)")
    p.add_argument("--variant", type=str, default="v2",
                   help=f"named architecture variant; one of {list_variants()}")
    p.add_argument("--arch", choices=["v1", "v2"], default="v2",
                   help="v1=legacy encoder-trunk-decoder, v2=single uniform DiT")

    # Training
    p.add_argument("--loss-type", choices=["l1", "l2"], default="l1")
    p.add_argument("--loss-surf-weight", type=float, default=10.0,
                   help="weight on surface-atom loss in the total training loss")
    p.add_argument("--loss-ads-weight", type=float, default=1.0,
                   help="weight on adsorbate-atom loss in the total training loss")
    p.add_argument("--lddt-ads-ads-weight", type=float, default=0.0,
                   help="auxiliary smooth lDDT loss weight on adsorbate-internal atom pairs")
    p.add_argument("--lddt-cutoff", type=float, default=15.0,
                   help="true-distance cutoff in Angstrom for smooth lDDT pair selection")
    p.add_argument("--lddt-time-weight", type=float, default=0.0,
                   help="extra multiplier slope for t>0.5: 1 + value * relu(t - 0.5)")
    p.add_argument("--ads-pair-l1-weight", type=float, default=0.0,
                   help="auxiliary L1 weight for all adsorbate-internal pair distances")
    p.add_argument("--ads-bond-l1-weight", type=float, default=0.0,
                   help="auxiliary L1 weight for bonded adsorbate-internal pair distances")
    p.add_argument("--ads-nonbonded-clash-weight", type=float, default=0.0,
                   help="auxiliary hinge weight for nonbonded adsorbate-internal clashes")
    p.add_argument("--ads-bond-factor", type=float, default=1.25,
                   help="covalent-radius multiplier for inferring adsorbate bonds")
    p.add_argument("--ads-clash-factor", type=float, default=0.75,
                   help="covalent-radius multiplier for nonbonded clash lower bounds")
    p.add_argument("--flow-eps", type=float, default=1e-5)
    p.add_argument("--prediction-type", type=str, default="x1",
                   choices=["x1", "v"],
                   help="x1: model predicts x_1 (default). v: model predicts "
                        "v = x_1 - x_0 (constant velocity field under linear flow).")
    p.add_argument("--seed", type=int, default=0)

    # Sample eval
    p.add_argument("--sample-eval-every-epochs", type=int, default=0)
    p.add_argument("--sample-eval-max-samples", type=int, default=64)
    p.add_argument("--sample-eval-steps", type=int, default=10)
    p.add_argument("--pristine-slabs", type=str, default="",
                   help="path to pristine relaxed-slab pkl (sid->slab_key->pos). "
                        "When set, sample_eval uses pristine pos as final_slab "
                        "reference for has_surface_changed.")
    p.add_argument("--pristine-index", "--pristine-sid-index",
                   dest="pristine_sid_index", type=str, default="",
                   help="sid/system_key->slab_key index pkl. Defaults to "
                        "<pristine-slabs>.sid_index.pkl or "
                        "<pristine-slabs>.system_index.pkl")

    # Replay buffer (expert iteration)
    p.add_argument("--use-replay", action="store_true")
    p.add_argument("--replay-gt-index",
                   default=str(_PROJECT_ROOT / "data" / "replay" / "gt_index_by_sid.pkl"))
    p.add_argument("--replay-train-lmdb", type=str, default=None,
                   help="LMDB to sample eval systems from (default: first --train-lmdb)")
    p.add_argument("--replay-mode", choices=["append", "replace"], default="append")
    p.add_argument("--replay-ratio", type=float, default=0.5)
    p.add_argument("--replay-eval-every", type=int, default=30)
    p.add_argument("--replay-warmup-epochs", type=int, default=30)
    p.add_argument("--replay-success-margin", type=float, default=0.05)
    p.add_argument("--replay-cap", type=int, default=1_070_000)
    p.add_argument("--replay-per-system-cap", type=int, default=10)
    p.add_argument("--replay-weight-mode",
                   choices=["improvement", "uniform", "recency"], default="improvement")
    p.add_argument("--replay-initial-systems", type=int, default=500)
    p.add_argument("--replay-initial-placements", type=int, default=3)
    p.add_argument("--replay-scaled-systems", type=int, default=2000)
    p.add_argument("--replay-scaled-placements", type=int, default=5)
    p.add_argument("--replay-uma-model", type=str, default="uma-s-1p1")
    p.add_argument("--replay-uma-fmax", type=float, default=0.05)
    p.add_argument("--replay-uma-max-steps", type=int, default=100)
    p.add_argument("--replay-flow-steps", type=int, default=50)
    p.add_argument("--replay-overlap-threshold", type=float, default=0.5)

    # Logging
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--validate-only", action="store_true",
                   help="Load last.ckpt and run a single validation pass; skip training. Errors out if last.ckpt is missing.")

    args = p.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    torch.serialization.add_safe_globals([
        DiTDenoiserConfig, DiTDenoiserV2Config, FlowConfig,
    ])

    model_cfg = build_config(args)
    # Mirror the resolved model config flag onto args so the DataModule (which
    # only sees args, not model_cfg) wires the ads-ref-pos channel correctly.
    args.use_ads_ref_pos = bool(getattr(model_cfg, "use_ads_ref_pos", False))
    flow_cfg = FlowConfig(eps=args.flow_eps, prediction_type=args.prediction_type)

    # -------- AtomMOF-style Rich config tree (rank 0 only) --------
    import rich
    import rich.syntax
    import rich.tree
    import yaml

    def _yaml_block(d):
        return yaml.safe_dump(d, default_flow_style=False, sort_keys=False)

    # group args into sections (data / trainer / model / flow / replay / sample_eval / wandb)
    _args_d = vars(args).copy()
    sections = {
        "data": {
            "train_lmdb": _args_d.get("train_lmdb"),
            "val_lmdb": _args_d.get("val_lmdb"),
            "max_train_samples": _args_d.get("max_train_samples"),
            "max_val_samples": _args_d.get("max_val_samples"),
            "prior_mode": _args_d.get("prior_mode"),
            "interstitial_gap": _args_d.get("interstitial_gap"),
            "translation_std": _args_d.get("translation_std"),
        },
        "trainer": {
            "epochs": _args_d.get("epochs"),
            "batch_size": _args_d.get("batch_size"),
            "num_workers": _args_d.get("num_workers"),
            "devices": _args_d.get("devices"),
            "lr": _args_d.get("lr"),
            "lr_warmup_steps": _args_d.get("lr_warmup_steps"),
            "weight_decay": _args_d.get("weight_decay"),
            "grad_clip": _args_d.get("grad_clip"),
            "accumulate_grad_batches": _args_d.get("accumulate_grad_batches"),
            "loss_type": _args_d.get("loss_type"),
            "lddt_ads_ads_weight": _args_d.get("lddt_ads_ads_weight"),
            "lddt_cutoff": _args_d.get("lddt_cutoff"),
            "lddt_time_weight": _args_d.get("lddt_time_weight"),
            "ads_pair_l1_weight": _args_d.get("ads_pair_l1_weight"),
            "ads_bond_l1_weight": _args_d.get("ads_bond_l1_weight"),
            "ads_nonbonded_clash_weight": _args_d.get("ads_nonbonded_clash_weight"),
            "ads_bond_factor": _args_d.get("ads_bond_factor"),
            "ads_clash_factor": _args_d.get("ads_clash_factor"),
            "flow_eps": _args_d.get("flow_eps"),
            "activation_checkpointing": _args_d.get("activation_checkpointing"),
            "seed": _args_d.get("seed"),
        },
        "model": {
            "arch": _args_d.get("arch"),
            "variant": _args_d.get("variant"),
            **asdict(model_cfg),
        },
        "flow": asdict(flow_cfg),
        "sample_eval": {
            "every_epochs": _args_d.get("sample_eval_every_epochs"),
            "max_samples": _args_d.get("sample_eval_max_samples"),
            "steps": _args_d.get("sample_eval_steps"),
        },
        "replay": ({
            "enabled": True,
            "mode": _args_d.get("replay_mode"),
            "ratio": _args_d.get("replay_ratio"),
            "warmup_epochs": _args_d.get("replay_warmup_epochs"),
            "eval_every": _args_d.get("replay_eval_every"),
            "success_margin_eV": _args_d.get("replay_success_margin"),
            "cap": _args_d.get("replay_cap"),
            "per_system_cap": _args_d.get("replay_per_system_cap"),
            "weight_mode": _args_d.get("replay_weight_mode"),
            "initial_systems": _args_d.get("replay_initial_systems"),
            "initial_placements": _args_d.get("replay_initial_placements"),
            "scaled_systems": _args_d.get("replay_scaled_systems"),
            "scaled_placements": _args_d.get("replay_scaled_placements"),
            "uma_model": _args_d.get("replay_uma_model"),
            "uma_fmax": _args_d.get("replay_uma_fmax"),
            "uma_max_steps": _args_d.get("replay_uma_max_steps"),
            "flow_steps": _args_d.get("replay_flow_steps"),
            "overlap_threshold": _args_d.get("replay_overlap_threshold"),
            "gt_index_path": _args_d.get("replay_gt_index"),
        } if _args_d.get("use_replay") else {"enabled": False}),
        "logger": {
            "wandb_project": _args_d.get("wandb_project") or "off",
            "wandb_run_name": _args_d.get("wandb_run_name") or "-",
            "wandb_entity": _args_d.get("wandb_entity") or "-",
            "log_every_n_steps": _args_d.get("log_every"),
        },
        "output": {
            "out": _args_d.get("out"),
        },
    }

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for field, content in sections.items():
        branch = tree.add(field, style=style, guide_style=style)
        branch.add(rich.syntax.Syntax(_yaml_block(content), "yaml", theme="ansi_dark"))

    # param count (separate branch — matches AtomMOF's log_hyperparameters)
    from adsorbgen.model_factory import build_model as _bm
    _m = _bm(model_cfg)
    _n = sum(p.numel() for p in _m.parameters())
    _t = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    del _m
    params_section = _yaml_block({
        "total": _n,
        "trainable": _t,
        "non_trainable": _n - _t,
        "total_M": round(_n / 1e6, 2),
    })
    params_branch = tree.add("model/params", style=style, guide_style=style)
    params_branch.add(rich.syntax.Syntax(params_section, "yaml", theme="ansi_dark"))

    rich.print(tree)

    out_dir = Path(args.out)
    _check_resume_arch(out_dir, args.arch)
    resume_ckpt = None
    last_ckpt = out_dir / "last.ckpt"

    # Default replay-train-lmdb = first train-lmdb
    if args.use_replay and not args.replay_train_lmdb:
        args.replay_train_lmdb = (
            args.train_lmdb[0] if isinstance(args.train_lmdb, list) else args.train_lmdb
        )

    replay_buffer_path = str(out_dir / "replay_buffer.pkl") if args.use_replay else ""

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
        pristine_slabs=args.pristine_slabs,
        pristine_sid_index=args.pristine_sid_index,
        loss_surf_weight=args.loss_surf_weight,
        loss_ads_weight=args.loss_ads_weight,
        lddt_ads_ads_weight=args.lddt_ads_ads_weight,
        lddt_cutoff=args.lddt_cutoff,
        lddt_time_weight=args.lddt_time_weight,
        ads_pair_l1_weight=args.ads_pair_l1_weight,
        ads_bond_l1_weight=args.ads_bond_l1_weight,
        ads_nonbonded_clash_weight=args.ads_nonbonded_clash_weight,
        ads_bond_factor=args.ads_bond_factor,
        ads_clash_factor=args.ads_clash_factor,
        use_replay=args.use_replay,
        replay_buffer_path=replay_buffer_path,
        replay_gt_index_path=args.replay_gt_index if args.use_replay else "",
        replay_train_lmdb=args.replay_train_lmdb or "",
        replay_mode=args.replay_mode,
        replay_ratio=args.replay_ratio,
        replay_eval_every=args.replay_eval_every,
        replay_warmup_epochs=args.replay_warmup_epochs,
        replay_success_margin=args.replay_success_margin,
        replay_cap=args.replay_cap,
        replay_per_system_cap=args.replay_per_system_cap,
        replay_weight_mode=args.replay_weight_mode,
        replay_initial_systems=args.replay_initial_systems,
        replay_initial_placements=args.replay_initial_placements,
        replay_scaled_systems=args.replay_scaled_systems,
        replay_scaled_placements=args.replay_scaled_placements,
        replay_prior_mode=args.prior_mode,
        replay_uma_model=args.replay_uma_model,
        replay_uma_fmax=args.replay_uma_fmax,
        replay_uma_max_steps=args.replay_uma_max_steps,
        replay_flow_steps=args.replay_flow_steps,
        replay_overlap_threshold=args.replay_overlap_threshold,
    )

    if last_ckpt.exists():
        resume_ckpt = str(last_ckpt)
        print(f"[resume] {resume_ckpt}", flush=True)

    dm = AdsorbGenDataModule(args, replay_buffer=module._replay_buffer)

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
        RichProgressBar(
            refresh_rate=1,
            console_kwargs={"force_terminal": True, "force_interactive": True},
        ),
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
        # AtomMOF-style log_hyperparameters: model param counts into wandb
        _n = sum(p.numel() for p in module.parameters())
        _t = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.log_hyperparams({
            "model/params/total": _n,
            "model/params/trainable": _t,
            "model/params/non_trainable": _n - _t,
        })

    # Trainer
    if args.validate_only:
        n_gpus = 1  # single-GPU is enough for a one-pass validation
    else:
        n_gpus = args.devices or torch.cuda.device_count() or 1
    # When replay is on, dataloaders must refresh after each eval so workers
    # see the updated buffer state.
    reload_every = 1 if args.use_replay else 0

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=n_gpus,
        strategy=(
            DDPStrategy(timeout=timedelta(hours=6))
            if n_gpus > 1 else "auto"
        ),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.grad_clip if args.grad_clip > 0 else None,
        default_root_dir=str(out_dir),
        log_every_n_steps=args.log_every,
        enable_progress_bar=True,
        reload_dataloaders_every_n_epochs=reload_every,
        check_val_every_n_epoch=int(args.check_val_every_n_epoch),
    )

    if not (out_dir / "args.json").exists():
        _save_args_json(out_dir, args.arch, model_cfg, args=args, flow_cfg=flow_cfg)

    if args.validate_only:
        if not last_ckpt.exists():
            raise FileNotFoundError(
                f"--validate-only requires {last_ckpt} to exist; none found."
            )
        # Force sample_eval to fire on this validation pass even if the run was
        # configured with sample_eval_every_epochs=0 etc.
        if module.hparams.sample_eval_every_epochs <= 0:
            module.hparams.sample_eval_every_epochs = 1
        print(f"[validate-only] loading {last_ckpt}", flush=True)
        trainer.validate(module, datamodule=dm, ckpt_path=str(last_ckpt))
        return

    trainer.fit(module, datamodule=dm, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
