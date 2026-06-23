#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA="${DATA:-$ROOT/data/overfit10/train.lmdb}"
OUT="${OUT:-$ROOT/runs/overfit10_x1_si_denoiser}"
ADS="${ADSORBATES_PKL:-$ROOT/data/pkls/adsorbates.pkl}"
DEVICES="${DEVICES:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-2000}"
TRAIN_REPLICATE="${TRAIN_REPLICATE:-1000}"
VAL_REPLICATE="${VAL_REPLICATE:-100}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ADSORBATES_PKL="$ADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m adsorbgen.training.train_cli \
  --train-lmdb "$DATA" \
  --val-lmdb "$DATA" \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --save-every-n-epochs 50 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-4}" \
  --devices "$DEVICES" \
  --precision "${PRECISION:-bf16-mixed}" \
  --variant "${VARIANT:-v0-ads-ref-adshead}" \
  --arch v1 \
  --prior-mode random_heuristic \
  --interstitial-gap 0.1 \
  --movable-mode surface_ads \
  --prediction-type x1 \
  --loss-target v \
  --flow-eps 1e-3 \
  --loss-type l1 \
  --loss-surf-weight 10.0 \
  --loss-ads-weight 1.0 \
  --ads-pair-l1-weight "${ADS_PAIR_L1_WEIGHT:-1.0}" \
  --gamma-schedule sqrt_t1mt \
  --gamma-sigma "${GAMMA_SIGMA:-0.1}" \
  --use-si-denoiser \
  --si-denoiser-loss-weight "${SI_DENOISER_LOSS_WEIGHT:-0.5}" \
  --si-denoiser-mask movable \
  --train-replicate "$TRAIN_REPLICATE" \
  --val-replicate "$VAL_REPLICATE" \
  --max-train-samples 10 \
  --max-val-samples 10 \
  --check-val-every-n-epoch "${CHECK_VAL_EVERY_N_EPOCH:-10}" \
  --sample-eval-every-epochs 0 \
  --log-every "${LOG_EVERY:-10}" \
  ${WANDB_PROJECT:+--wandb-project "$WANDB_PROJECT"} \
  ${WANDB_RUN_NAME:+--wandb-run-name "$WANDB_RUN_NAME"} \
  ${EXTRA_ARGS:-}
