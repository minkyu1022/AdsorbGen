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

OUT_DIR="${OUT_DIR:-${HERE}/runs/h200_sweep_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
NUM_GPUS="$(wc -w <<<"${GPUS}")"
SYSTEMS_PER_GPU="${SYSTEMS_PER_GPU:-8}"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-1}"
MAX_STEPS="${MAX_STEPS:-60}"
FLOW_BATCH_SIZES="${FLOW_BATCH_SIZES:-32 64}"
MAX_ATOMS_LIST="${MAX_ATOMS_LIST:-4096 8192 16384 32768}"
FMAX="${FMAX:-0.05}"
MAXSTEP="${MAXSTEP:-0.04}"
LBFGS_STREAMING="${LBFGS_STREAMING:-0}"
LBFGS_CHECK_INTERVAL="${LBFGS_CHECK_INTERVAL:-10}"
LBFGS_STREAM_SORT="${LBFGS_STREAM_SORT:-0}"

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

for flow_bs in ${FLOW_BATCH_SIZES}; do
  for max_atoms in ${MAX_ATOMS_LIST}; do
    echo "[sweep] flow_batch_size=${flow_bs} max_atoms=${max_atoms} max_steps=${MAX_STEPS}"
    pids=()
    shard=0
    for gpu in ${GPUS}; do
      offset=$((shard * SYSTEMS_PER_GPU))
      out_json="${OUT_DIR}/sweep_bs${flow_bs}_atoms${max_atoms}_gpu${gpu}.json"
      log="${OUT_DIR}/logs/sweep_bs${flow_bs}_atoms${max_atoms}_gpu${gpu}.log"
      extra_args=()
      if [[ "${LBFGS_STREAMING}" == "1" ]]; then
        extra_args+=(--lbfgs-streaming --lbfgs-check-interval "${LBFGS_CHECK_INTERVAL}")
      fi
      if [[ "${LBFGS_STREAM_SORT}" == "1" ]]; then
        extra_args+=(--lbfgs-stream-sort)
      fi
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${HERE}/geoopt.py" \
        --repo "${REPO}" \
        --adsorbates-pkl "${ADSORBATES_PKL}" \
        --ckpt "${CKPT}" \
        --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
        --selected-systems "${SELECTED_SYSTEMS}" \
        --out-json "${out_json}" \
        --algorithms lbfgs \
        --seed 20260526 \
        --num-systems "${SYSTEMS_PER_GPU}" \
        --system-offset "${offset}" \
        --num-placements "${NUM_PLACEMENTS}" \
        --accuracy-systems 0 \
        --flow-steps 50 \
        --flow-batch-size "${flow_bs}" \
        --prior-mode random_heuristic \
        --uma-model uma-s-1p2 \
        --uma-task oc20 \
        --fmax "${FMAX}" \
        --max-steps "${MAX_STEPS}" \
        --max-atoms "${max_atoms}" \
        --maxstep "${MAXSTEP}" \
        --lbfgs-memory 50 \
        --lbfgs-damping 1.0 \
        --lbfgs-alpha 70.0 \
        "${extra_args[@]}" > "${log}" 2>&1 &
      pids+=("$!")
      shard=$((shard + 1))
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done
done

"${PYTHON_BIN}" "${HERE}/summarize_comparator_runs.py" \
  --mode sweep \
  --files "${OUT_DIR}"/sweep_*.json \
  --out-json "${OUT_DIR}/sweep_summary.json"

echo "[sweep] wrote ${OUT_DIR}"
