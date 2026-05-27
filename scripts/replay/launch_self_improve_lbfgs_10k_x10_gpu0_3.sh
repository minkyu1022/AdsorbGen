#!/usr/bin/env bash
set -euo pipefail

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"

CKPT="${CKPT:-/home/irteam/runs/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544/ckpt_epochepoch=099.ckpt}"
OUT_DIR="${OUT_DIR:-/home/irteam/runs/self_improve_lbfgs_ID_mlip_pairdist_1x_ep99_10k_x10_20260523}"
GT_INDEX="${GT_INDEX:-/home/irteam/data/replay/gt_index_by_sid_oc20_lbfgs.pkl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,0,1,2,3}"
NUM_SYSTEMS="${NUM_SYSTEMS:-10000}"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-10}"
FLOW_STEPS="${FLOW_STEPS:-50}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-32}"
UMA_FMAX="${UMA_FMAX:-0.05}"
UMA_MAX_STEPS="${UMA_MAX_STEPS:-300}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
UMA_TASK="${UMA_TASK:-oc20}"
PRISTINE_SLABS="${PRISTINE_SLABS:-/home/irteam/results/pristine_slabs/is2res.pkl}"
PRISTINE_INDEX="${PRISTINE_INDEX:-/home/irteam/results/pristine_slabs/is2res.sid_index.pkl}"
SEED="${SEED:-20260523}"
SAVE_WINDOW_CANDIDATES="${SAVE_WINDOW_CANDIDATES:-1}"
CANDIDATE_WINDOW_EV="${CANDIDATE_WINDOW_EV:-0.1}"

LMDBS=(
  /home/irteam/data/processed_ID/is2res_train.lmdb
  /home/irteam/data/processed_ID/is2res_val.lmdb
)

mkdir -p "${OUT_DIR}/logs"
SELECTED_SYSTEMS="${OUT_DIR}/selected_systems.json"
echo "[launch] out=${OUT_DIR}"
echo "[launch] ckpt=${CKPT}"
echo "[launch] systems=${NUM_SYSTEMS} placements=${NUM_PLACEMENTS} shards=${NUM_SHARDS}"
echo "[launch] GPU_LIST=${GPU_LIST}"
echo "[launch] optimizer=ASE_LBFGS uma=${UMA_MODEL}/${UMA_TASK} fmax=${UMA_FMAX} max_steps=${UMA_MAX_STEPS}"
echo "[launch] pristine_slabs=${PRISTINE_SLABS}"
echo "[launch] moving-window candidates enabled=${SAVE_WINDOW_CANDIDATES} window_ev=${CANDIDATE_WINDOW_EV}"

WINDOW_CANDIDATE_ARGS=()
if [[ "${SAVE_WINDOW_CANDIDATES}" == "1" ]]; then
  WINDOW_CANDIDATE_ARGS=(--save-window-candidates --candidate-window-ev "${CANDIDATE_WINDOW_EV}")
fi

if [[ ! -s "${SELECTED_SYSTEMS}" ]]; then
  echo "[launch] preparing selected systems -> ${SELECTED_SYSTEMS}"
  (
    cd "${REPO}"
    exec env PYTHONUNBUFFERED=1 PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" scripts/replay/prepare_self_improve_selection.py \
        --train-lmdb "${LMDBS[0]}" "${LMDBS[1]}" \
        --gt-index "${GT_INDEX}" \
        --out "${SELECTED_SYSTEMS}" \
        --seed "${SEED}" \
        --num-systems "${NUM_SYSTEMS}" \
        --num-placements "${NUM_PLACEMENTS}"
  ) | tee "${OUT_DIR}/logs/prepare_selection.log"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  cuda="${GPUS[$shard]}"
  log="${OUT_DIR}/logs/shard_${shard}.log"
  echo "[launch] shard=${shard} cuda=${cuda} log=${log}"
  setsid -f bash -c "
    cd '${REPO}' &&
    exec env CUDA_VISIBLE_DEVICES='${cuda}' PYTHONUNBUFFERED=1 \
      PYTHONPATH='${REPO}:'\"\${PYTHONPATH:-}\" \
      ADSORBATES_PKL=/home/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      '${PYTHON_BIN}' scripts/replay/self_improve_lbfgs_worker.py \
        --ckpt '${CKPT}' \
        --train-lmdb '${LMDBS[0]}' '${LMDBS[1]}' \
        --gt-index '${GT_INDEX}' \
        --out-dir '${OUT_DIR}' \
        --shard-idx '${shard}' \
        --num-shards '${NUM_SHARDS}' \
        --seed '${SEED}' \
        --num-systems '${NUM_SYSTEMS}' \
        --num-placements '${NUM_PLACEMENTS}' \
        --selected-systems '${SELECTED_SYSTEMS}' \
        --flow-steps '${FLOW_STEPS}' \
        --flow-batch-size '${FLOW_BATCH_SIZE}' \
        --prior-mode random_heuristic \
        --uma-model '${UMA_MODEL}' \
        --uma-task '${UMA_TASK}' \
        --uma-fmax '${UMA_FMAX}' \
        --uma-max-steps '${UMA_MAX_STEPS}' \
        --pristine-slabs '${PRISTINE_SLABS}' \
        --pristine-index '${PRISTINE_INDEX}' \
        --success-margin 0.0 \
        --progress-every 32 \
        ${WINDOW_CANDIDATE_ARGS[*]}
  " >"${log}" 2>&1 &
  echo $! >"${OUT_DIR}/logs/shard_${shard}.pid"
  sleep "${LAUNCH_STAGGER_SEC:-5}"
done

echo "[launch] dispatched ${NUM_SHARDS} shards"
echo "[launch] progress: ${OUT_DIR}/logs/progress_shard*.json"
