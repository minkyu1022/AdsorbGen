#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA="${DATA:-$ROOT/data/overfit10/train.lmdb}"
OUT="${OUT:-$ROOT/runs/overfit10_x09_recon_denoiser}"
ADS="${ADSORBATES_PKL:-$ROOT/data/pkls/adsorbates.pkl}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-2000}"
TRAIN_REPLICATE="${TRAIN_REPLICATE:-1000}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ADSORBATES_PKL="$ADS"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "$ROOT/experiments/2026-06-overfit-denoiser-relax/train_x09_recon_denoiser_overfit.py" \
  --repo "$ROOT" \
  --train-lmdb "$DATA" \
  --val-lmdb "$DATA" \
  --out "$OUT" \
  --adsorbates-pkl "$ADS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "${NUM_WORKERS:-4}" \
  --variant "${VARIANT:-v0-ads-ref-adshead}" \
  --arch v1 \
  --t "${T_VALUE:-0.9}" \
  --gamma-schedule "${GAMMA_SCHEDULE:-sqrt_t1mt}" \
  --gamma-sigma "${GAMMA_SIGMA:-0.1}" \
  --movable-mode "${MOVABLE_MODE:-surface_ads}" \
  --loss-surf-weight "${LOSS_SURF_WEIGHT:-10.0}" \
  --loss-ads-weight "${LOSS_ADS_WEIGHT:-1.0}" \
  --ads-pair-l1-weight "${ADS_PAIR_L1_WEIGHT:-1.0}" \
  --train-replicate "$TRAIN_REPLICATE" \
  --max-train-samples 10 \
  --max-val-samples 10 \
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS:-50}" \
  --log-every "${LOG_EVERY:-10}" \
  ${EXTRA_ARGS:-}
