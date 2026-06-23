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
PRISTINE_SLABS="${PRISTINE_SLABS:-${TRANSFER}/data/replay_uma_s_1p2/pristine_slabs_lbfgs.pkl}"

OUT_DIR="${OUT_DIR:-${HERE}/runs/ase_custom_compare_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_GPUS="$(wc -w <<<"${GPUS}")"
TOTAL_SYSTEMS="${TOTAL_SYSTEMS:-100}"
SYSTEMS_PER_GPU="$(( (TOTAL_SYSTEMS + NUM_GPUS - 1) / NUM_GPUS ))"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-1}"
STEP_LIST="${STEP_LIST:-300 500}"

FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
MAX_ATOMS="${MAX_ATOMS:-16384}"
ASE_CONCURRENCY="${ASE_CONCURRENCY:-8}"
BATCHER_MAX_ATOMS="${BATCHER_MAX_ATOMS:-8192}"
BATCHER_WAIT_TIMEOUT_S="${BATCHER_WAIT_TIMEOUT_S:-0.02}"
LBFGS_HISTORY_DTYPE="${LBFGS_HISTORY_DTYPE:-float32}"
LBFGS_POSITION_DTYPE="${LBFGS_POSITION_DTYPE:-float32}"
LBFGS_CURVATURE_GUARD="${LBFGS_CURVATURE_GUARD:-abs}"
ASE_REFERENCE_JSON_GLOB="${ASE_REFERENCE_JSON_GLOB:-}"

mkdir -p "${OUT_DIR}/logs"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

monitor_pid=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv -l 1 > "${OUT_DIR}/gpu_monitor.csv" &
  monitor_pid="$!"
fi

cleanup() {
  if [[ -n "${monitor_pid}" ]]; then
    kill "${monitor_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for steps in ${STEP_LIST}; do
  echo "[compare] max_steps=${steps} total_systems=${TOTAL_SYSTEMS}"
  pids=()
  shard=0
  for gpu in ${GPUS}; do
    offset=$((shard * SYSTEMS_PER_GPU))
    remain=$((TOTAL_SYSTEMS - offset))
    if [[ "${remain}" -le 0 ]]; then
      shard=$((shard + 1))
      continue
    fi
    n="${SYSTEMS_PER_GPU}"
    if [[ "${remain}" -lt "${n}" ]]; then
      n="${remain}"
    fi
    out_json="${OUT_DIR}/compare_steps${steps}_gpu${gpu}.json"
    log="${OUT_DIR}/logs/compare_steps${steps}_gpu${gpu}.log"
    extra_args=()
    if [[ -n "${ASE_REFERENCE_JSON_GLOB}" ]]; then
      # Intentional word splitting after glob expansion.
      extra_args+=(--ase-reference-json ${ASE_REFERENCE_JSON_GLOB})
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${HERE}/ase_custom_lbfgs_comparator.py" \
      --repo "${REPO}" \
      --adsorbates-pkl "${ADSORBATES_PKL}" \
      --ckpt "${CKPT}" \
      --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
      --selected-systems "${SELECTED_SYSTEMS}" \
      --out-json "${out_json}" \
      --seed 20260526 \
      --num-systems "${n}" \
      --system-offset "${offset}" \
      --num-placements "${NUM_PLACEMENTS}" \
      --flow-steps 50 \
      --flow-batch-size "${FLOW_BATCH_SIZE}" \
      --prior-mode random_heuristic \
      --uma-model uma-s-1p2 \
      --uma-task oc20 \
      --fmax 0.05 \
      --max-steps "${steps}" \
      --max-atoms "${MAX_ATOMS}" \
      --maxstep 0.04 \
      --lbfgs-memory 50 \
      --lbfgs-damping 1.0 \
      --lbfgs-alpha 70.0 \
      --lbfgs-history-dtype "${LBFGS_HISTORY_DTYPE}" \
      --lbfgs-position-dtype "${LBFGS_POSITION_DTYPE}" \
      --lbfgs-curvature-guard "${LBFGS_CURVATURE_GUARD}" \
      --window-ev 0.1 \
      --pristine-slabs "${PRISTINE_SLABS}" \
      --use-fairchem-batcher \
      --lbfgs-concurrency "${ASE_CONCURRENCY}" \
      --batcher-max-atoms "${BATCHER_MAX_ATOMS}" \
      --batcher-wait-timeout-s "${BATCHER_WAIT_TIMEOUT_S}" \
      "${extra_args[@]}" > "${log}" 2>&1 &
    pids+=("$!")
    shard=$((shard + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  "${PYTHON_BIN}" "${HERE}/summarize_comparator_runs.py" \
    --mode compare \
    --files "${OUT_DIR}"/compare_steps"${steps}"_gpu*.json \
    --out-json "${OUT_DIR}/summary_steps${steps}.json"
done

if [[ -f "${OUT_DIR}/summary_steps300.json" && -f "${OUT_DIR}/summary_steps500.json" ]]; then
  "${PYTHON_BIN}" "${HERE}/summarize_comparator_runs.py" \
    --mode ase-step-delta \
    --files "${OUT_DIR}"/compare_steps300_gpu*.json \
    --files-b "${OUT_DIR}"/compare_steps500_gpu*.json \
    --out-json "${OUT_DIR}/ase_300_vs_500.json"
fi

echo "[compare] wrote ${OUT_DIR}"
