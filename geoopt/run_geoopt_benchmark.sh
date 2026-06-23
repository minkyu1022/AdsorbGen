#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${HERE}/configs/all.env}"

set -a
source "${HERE}/configs/common.env"
source "${CONFIG}"
set +a

mkdir -p "${OUT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_JSON="${OUT_JSON:-${OUT_DIR}/geoopt_${ALGORITHMS// /_}_${STAMP}.json}"

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1
export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" "${HERE}/geoopt.py" \
  --repo "${REPO}" \
  --adsorbates-pkl "${ADSORBATES_PKL}" \
  --ckpt "${CKPT}" \
  --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
  --selected-systems "${SELECTED_SYSTEMS}" \
  --out-json "${OUT_JSON}" \
  --algorithms ${ALGORITHMS} \
  --seed "${SEED}" \
  --num-systems "${NUM_SYSTEMS}" \
  --num-placements "${NUM_PLACEMENTS}" \
  --accuracy-systems "${ACCURACY_SYSTEMS}" \
  --flow-steps "${FLOW_STEPS}" \
  --flow-batch-size "${FLOW_BATCH_SIZE}" \
  --prior-mode "${PRIOR_MODE}" \
  --uma-model "${UMA_MODEL}" \
  --uma-task "${UMA_TASK}" \
  --fmax "${FMAX}" \
  --max-steps "${MAX_STEPS}" \
  --max-atoms "${MAX_ATOMS}" \
  --maxstep "${MAXSTEP}" \
  --fire-dt "${FIRE_DT:-0.1}" \
  --fire-dt-max "${FIRE_DT_MAX:-1.0}" \
  --lbfgs-memory "${LBFGS_MEMORY:-50}" \
  --lbfgs-damping "${LBFGS_DAMPING:-1.0}" \
  --lbfgs-alpha "${LBFGS_ALPHA:-70.0}" \
  --cg-step-size "${CG_STEP_SIZE:-0.04}"

echo "[geoopt] wrote ${OUT_JSON}"

