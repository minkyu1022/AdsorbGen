#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/home/irteam/AdsorbGen}"
TRANSFER="${TRANSFER:-/home1/irteam/full_replay_1p2_lbfgs_transfer_20260529}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"

CKPT="${CKPT:-/home1/irteam/data-vol1/minkyu/runs/training/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544/ckpt_epochepoch=149.ckpt}"
ADSORBATES_PKL="${ADSORBATES_PKL:-${TRANSFER}/data/pkls/adsorbates.pkl}"
TRAIN_LMDB="${TRAIN_LMDB:-${TRANSFER}/data/processed_ID/is2res_train.lmdb}"
VAL_LMDB="${VAL_LMDB:-${TRANSFER}/data/processed_ID/is2res_val.lmdb}"
SELECTED_SYSTEMS="${SELECTED_SYSTEMS:-${TRANSFER}/selected_systems/full_train_val_id_341893_x10_seed20260526.json}"

OUT_DIR="${OUT_DIR:-${TRANSFER}/runs/full_two_stage_custom_lbfgs_gpu4_7_$(date +%Y%m%d_%H%M%S)}"
GPUS=(${GPUS:-4 5 6 7})
TOTAL_SYSTEMS="${TOTAL_SYSTEMS:-341893}"
SYSTEM_OFFSET="${SYSTEM_OFFSET:-0}"
SHARD_SYSTEMS="${SHARD_SYSTEMS:-256}"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-10}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_ATOMS="${MAX_ATOMS:-32768}"
LBFGS_CHECK_INTERVAL="${LBFGS_CHECK_INTERVAL:-10}"
UMA_INFERENCE_SETTINGS="${UMA_INFERENCE_SETTINGS:-default}"
UMA_INTERNAL_GRAPH_VERSION="${UMA_INTERNAL_GRAPH_VERSION:-0}"
UMA_EXECUTION_MODE="${UMA_EXECUTION_MODE:-}"
UMA_COMPILE="${UMA_COMPILE:-0}"
LBFGS_GPU_HISTORY_GUARD="${LBFGS_GPU_HISTORY_GUARD:-0}"
LBFGS_KEEP_SURVIVORS_ON_GPU="${LBFGS_KEEP_SURVIVORS_ON_GPU:-0}"

mkdir -p "${OUT_DIR}/logs"

for f in "${ADSORBATES_PKL}" "${CKPT}" "${TRAIN_LMDB}" "${VAL_LMDB}" "${SELECTED_SYSTEMS}"; do
  if [[ ! -s "${f}" ]]; then
    echo "[two-stage] missing required file: ${f}" >&2
    exit 2
  fi
done

cat > "${OUT_DIR}/launch_settings.env" <<EOF
REPO=${REPO}
TRANSFER=${TRANSFER}
PYTHON_BIN=${PYTHON_BIN}
CKPT=${CKPT}
ADSORBATES_PKL=${ADSORBATES_PKL}
TRAIN_LMDB=${TRAIN_LMDB}
VAL_LMDB=${VAL_LMDB}
SELECTED_SYSTEMS=${SELECTED_SYSTEMS}
OUT_DIR=${OUT_DIR}
GPUS=${GPUS[*]}
TOTAL_SYSTEMS=${TOTAL_SYSTEMS}
SYSTEM_OFFSET=${SYSTEM_OFFSET}
SHARD_SYSTEMS=${SHARD_SYSTEMS}
NUM_PLACEMENTS=${NUM_PLACEMENTS}
FLOW_BATCH_SIZE=${FLOW_BATCH_SIZE}
MAX_STEPS=${MAX_STEPS}
MAX_ATOMS=${MAX_ATOMS}
LBFGS_CHECK_INTERVAL=${LBFGS_CHECK_INTERVAL}
UMA_INFERENCE_SETTINGS=${UMA_INFERENCE_SETTINGS}
UMA_INTERNAL_GRAPH_VERSION=${UMA_INTERNAL_GRAPH_VERSION}
UMA_EXECUTION_MODE=${UMA_EXECUTION_MODE}
UMA_COMPILE=${UMA_COMPILE}
LBFGS_GPU_HISTORY_GUARD=${LBFGS_GPU_HISTORY_GUARD}
LBFGS_KEEP_SURVIVORS_ON_GPU=${LBFGS_KEEP_SURVIVORS_ON_GPU}
EOF

monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 1 > "${OUT_DIR}/gpu_monitor.csv" &
  monitor_pid="$!"
  echo "${monitor_pid}" > "${OUT_DIR}/logs/gpu_monitor.pid"
fi

cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[two-stage] out=${OUT_DIR}"
echo "[two-stage] gpus=${GPUS[*]} total_systems=${TOTAL_SYSTEMS} placements=${NUM_PLACEMENTS}"
echo "[two-stage] shard_systems=${SHARD_SYSTEMS} max_atoms=${MAX_ATOMS} max_steps=${MAX_STEPS}"

"${PYTHON_BIN}" "${HERE}/two_stage_full_replay.py" generate \
  --repo "${REPO}" \
  --adsorbates-pkl "${ADSORBATES_PKL}" \
  --ckpt "${CKPT}" \
  --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
  --selected-systems "${SELECTED_SYSTEMS}" \
  --out-dir "${OUT_DIR}" \
  --gpus "${GPUS[@]}" \
  --total-systems "${TOTAL_SYSTEMS}" \
  --system-offset "${SYSTEM_OFFSET}" \
  --shard-systems "${SHARD_SYSTEMS}" \
  --num-placements "${NUM_PLACEMENTS}" \
  --flow-batch-size "${FLOW_BATCH_SIZE}" \
  --flow-steps 50 \
  --prior-mode random_heuristic

"${PYTHON_BIN}" "${HERE}/two_stage_full_replay.py" relax \
  --repo "${REPO}" \
  --adsorbates-pkl "${ADSORBATES_PKL}" \
  --ckpt "${CKPT}" \
  --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
  --selected-systems "${SELECTED_SYSTEMS}" \
  --out-dir "${OUT_DIR}" \
  --gpus "${GPUS[@]}" \
  --uma-model uma-s-1p2 \
  --uma-task oc20 \
  --uma-inference-settings "${UMA_INFERENCE_SETTINGS}" \
  --uma-internal-graph-version "${UMA_INTERNAL_GRAPH_VERSION}" \
  ${UMA_EXECUTION_MODE:+--uma-execution-mode "${UMA_EXECUTION_MODE}"} \
  $([ "${UMA_COMPILE}" = "1" ] && printf '%s' '--uma-compile') \
  --fmax 0.05 \
  --max-steps "${MAX_STEPS}" \
  --max-atoms "${MAX_ATOMS}" \
  --maxstep 0.04 \
  --lbfgs-memory 50 \
  --lbfgs-damping 1.0 \
  --lbfgs-alpha 70.0 \
  $([ "${LBFGS_GPU_HISTORY_GUARD}" = "1" ] && printf '%s' '--lbfgs-gpu-history-guard') \
  $([ "${LBFGS_KEEP_SURVIVORS_ON_GPU}" = "1" ] && printf '%s' '--lbfgs-keep-survivors-on-gpu') \
  --lbfgs-streaming \
  --lbfgs-check-interval "${LBFGS_CHECK_INTERVAL}" \
  --save-result-pkl

echo "[two-stage] wrote ${OUT_DIR}"
