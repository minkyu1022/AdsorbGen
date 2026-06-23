#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA="${DATA:-$ROOT/data/overfit10/train.lmdb}"
CKPT="${CKPT:-$ROOT/runs/overfit10_x09_recon_denoiser/last.ckpt}"
OUT="${OUT:-$ROOT/runs/overfit10_x09_direct_recon_energy}"
ADS="${ADSORBATES_PKL:-$ROOT/data/pkls/adsorbates.pkl}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ADSORBATES_PKL="$ADS"

python "$ROOT/experiments/2026-06-overfit-denoiser-relax/eval_x09_direct_recon_energy.py" \
  --repo "$ROOT" \
  --lmdb "$DATA" \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --adsorbates-pkl "$ADS" \
  --batch-size "${BATCH_SIZE:-10}" \
  --device "${DEVICE:-cuda}" \
  --t "${T_VALUE:-0.9}" \
  --uma-model "${UMA_MODEL:-uma-s-1p1}" \
  --uma-task "${UMA_TASK:-oc20}"
