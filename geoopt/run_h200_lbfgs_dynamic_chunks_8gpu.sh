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

OUT_DIR="${OUT_DIR:-${HERE}/runs/h200_dynamic_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
TOTAL_SYSTEMS="${TOTAL_SYSTEMS:-2048}"
CHUNK_SYSTEMS="${CHUNK_SYSTEMS:-128}"
MAX_STEPS="${MAX_STEPS:-30}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
MAX_ATOMS="${MAX_ATOMS:-32768}"

mkdir -p "${OUT_DIR}/logs"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo 0 > "${OUT_DIR}/next_offset.txt"
: > "${OUT_DIR}/claims.tsv"

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

worker() {
  local gpu="$1"
  while true; do
    local offset n
    {
      flock 9
      offset="$(cat "${OUT_DIR}/next_offset.txt")"
      if [[ "${offset}" -ge "${TOTAL_SYSTEMS}" ]]; then
        echo done
      else
        n="${CHUNK_SYSTEMS}"
        if [[ $((offset + n)) -gt "${TOTAL_SYSTEMS}" ]]; then
          n=$((TOTAL_SYSTEMS - offset))
        fi
        echo $((offset + n)) > "${OUT_DIR}/next_offset.txt"
        printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${gpu}" "${offset}" "${n}" >> "${OUT_DIR}/claims.tsv"
        echo "${offset} ${n}"
      fi
    } 9>"${OUT_DIR}/queue.lock" > "${OUT_DIR}/logs/claim_gpu${gpu}.tmp"
    read -r offset n < "${OUT_DIR}/logs/claim_gpu${gpu}.tmp"
    if [[ "${offset}" == "done" ]]; then
      break
    fi
    local out_json="${OUT_DIR}/chunk_offset${offset}_n${n}_gpu${gpu}.json"
    local log="${OUT_DIR}/logs/chunk_offset${offset}_n${n}_gpu${gpu}.log"
    echo "[dynamic] gpu=${gpu} offset=${offset} n=${n}" | tee -a "${OUT_DIR}/logs/worker_gpu${gpu}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${HERE}/geoopt.py" \
      --repo "${REPO}" \
      --adsorbates-pkl "${ADSORBATES_PKL}" \
      --ckpt "${CKPT}" \
      --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
      --selected-systems "${SELECTED_SYSTEMS}" \
      --out-json "${out_json}" \
      --algorithms lbfgs \
      --seed 20260526 \
      --num-systems "${n}" \
      --system-offset "${offset}" \
      --num-placements 1 \
      --accuracy-systems 0 \
      --flow-steps 50 \
      --flow-batch-size "${FLOW_BATCH_SIZE}" \
      --prior-mode random_heuristic \
      --uma-model uma-s-1p2 \
      --uma-task oc20 \
      --fmax 0.05 \
      --max-steps "${MAX_STEPS}" \
      --max-atoms "${MAX_ATOMS}" \
      --maxstep 0.04 \
      --lbfgs-memory 50 \
      --lbfgs-damping 1.0 \
      --lbfgs-alpha 70.0 > "${log}" 2>&1
  done
}

pids=()
for gpu in ${GPUS}; do
  worker "${gpu}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

"${PYTHON_BIN}" "${HERE}/summarize_comparator_runs.py" \
  --mode sweep \
  --files "${OUT_DIR}"/chunk_*.json \
  --out-json "${OUT_DIR}/dynamic_summary.json"

echo "[dynamic] wrote ${OUT_DIR}"
