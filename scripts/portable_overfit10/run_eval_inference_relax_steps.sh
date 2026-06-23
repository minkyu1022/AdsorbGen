#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA="${DATA:-$ROOT/data/overfit10/train.lmdb}"
CKPT="${CKPT:-$ROOT/runs/overfit10_x1_si_denoiser/last.ckpt}"
OUT="${OUT:-$ROOT/runs/overfit10_inference_relax_steps}"
ADS="${ADSORBATES_PKL:-$ROOT/data/pkls/adsorbates.pkl}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ADSORBATES_PKL="$ADS"

python "$ROOT/experiments/2026-06-overfit-denoiser-relax/eval_overfit_inference_relax_steps.py" \
  --repo "$ROOT" \
  --lmdb "$DATA" \
  --ckpt "$CKPT" \
  --out "$OUT" \
  --adsorbates-pkl "$ADS" \
  --batch-size "${BATCH_SIZE:-10}" \
  --device "${DEVICE:-cuda}" \
  --flow-steps "${FLOW_STEPS:-50}" \
  --fmax "${FMAX:-0.05}" \
  --max-steps "${MAX_STEPS:-300}" \
  --lbfgs-memory "${LBFGS_MEMORY:-100}" \
  --lbfgs-maxstep "${LBFGS_MAXSTEP:-0.2}" \
  --lbfgs-damping "${LBFGS_DAMPING:-1.0}" \
  --lbfgs-alpha "${LBFGS_ALPHA:-70.0}" \
  --uma-model "${UMA_MODEL:-uma-s-1p1}" \
  --uma-task "${UMA_TASK:-oc20}"
