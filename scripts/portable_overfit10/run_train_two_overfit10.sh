#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-$ROOT/runs/overfit10_logs}"
mkdir -p "$LOG_DIR"

BASE_GPU="${BASE_GPU:-0}"
RECON_GPU="${RECON_GPU:-0}"

echo "[overfit10] train base x1 -> $ROOT/runs/overfit10_x1_base"
CUDA_VISIBLE_DEVICES="$BASE_GPU" \
  OUT="${BASE_OUT:-$ROOT/runs/overfit10_x1_base}" \
  WANDB_RUN_NAME="${BASE_WANDB_RUN_NAME:-overfit10_x1_base}" \
  bash "$ROOT/scripts/portable_overfit10/run_train_x1_base_overfit10.sh" \
  2>&1 | tee "$LOG_DIR/train_x1_base.log"

echo "[overfit10] train x0.9 direct recon denoiser -> $ROOT/runs/overfit10_x09_recon_denoiser"
CUDA_VISIBLE_DEVICES="$RECON_GPU" \
  OUT="${RECON_OUT:-$ROOT/runs/overfit10_x09_recon_denoiser}" \
  bash "$ROOT/scripts/portable_overfit10/run_train_x09_recon_denoiser_overfit10.sh" \
  2>&1 | tee "$LOG_DIR/train_x09_recon_denoiser.log"

echo "[overfit10] done"
