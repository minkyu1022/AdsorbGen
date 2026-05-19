#!/usr/bin/env bash
# Spawn one replay_daemon.py process per GPU in $GPUS, each on its own shard.
#
# Typical setup: 4 GPUs continuously refining the training run's replay
# buffer while the trainer runs on a different set of GPUs.
#
# Usage:
#   GPUS="4 5 6 7" \
#   TRAIN_RUN_DIR=/home/irteam/runs/H200_catflow_center_rel \
#   STREAM_DIR=/home/irteam/runs/H200_catflow_center_rel/replay_stream \
#   GT_INDEX=/home/irteam/data/replay/gt_index_by_sid.pkl \
#   TRAIN_LMDB=/home/irteam/data/processed/is2res_train.lmdb \
#   PRIOR_MODE=catflow_center_rel \
#   bash scripts/run_replay_daemon_4gpu.sh
#
# To stop: touch $STREAM_DIR/KILL_FLAG (daemons finish current cycle, exit).
set -euo pipefail
source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
CODE="${PYTHONPATH%%:*}"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"
PYTHON_BIN="${MICROMAMBA} run -n adsorbgen python"

# --- required ---
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:?TRAIN_RUN_DIR is required (training run with last.ckpt)}"
STREAM_DIR="${STREAM_DIR:?STREAM_DIR is required (replay buffer + logs)}"
GT_INDEX="${GT_INDEX:?GT_INDEX is required}"
TRAIN_LMDB="${TRAIN_LMDB:?TRAIN_LMDB is required}"

# --- optional ---
GPUS="${GPUS:-4 5 6 7}"
PRIOR_MODE="${PRIOR_MODE:-random_heuristic}"
NUM_SYSTEMS_PER_SHARD="${NUM_SYSTEMS_PER_SHARD:-200}"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-3}"
FLOW_STEPS="${FLOW_STEPS:-50}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-32}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
UMA_FMAX="${UMA_FMAX:-0.05}"
UMA_MAX_STEPS="${UMA_MAX_STEPS:-100}"
UMA_ATOM_BUDGET="${UMA_ATOM_BUDGET:-4000}"
SUCCESS_MARGIN="${SUCCESS_MARGIN:-0.05}"
SUCCESS_MARGIN_FINAL="${SUCCESS_MARGIN_FINAL:-}"
SUCCESS_MARGIN_SCHEDULE_CYCLES="${SUCCESS_MARGIN_SCHEDULE_CYCLES:-0}"
OVERLAP_THRESHOLD="${OVERLAP_THRESHOLD:-0.5}"
VIZ_CAPTURE_N="${VIZ_CAPTURE_N:-0}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
CKPT_SETTLE_SEC="${CKPT_SETTLE_SEC:-5.0}"
CKPT_STALE_WARN_MIN="${CKPT_STALE_WARN_MIN:-60}"
CKPT_STALE_EXIT_MIN="${CKPT_STALE_EXIT_MIN:-180}"
MAX_CYCLES="${MAX_CYCLES:-0}"
UTIL_POLL_SEC="${UTIL_POLL_SEC:-1.0}"
PRISTINE_SLABS="${PRISTINE_SLABS:-}"
PRISTINE_INDEX="${PRISTINE_INDEX:-}"

mkdir -p "${STREAM_DIR}/logs" "${STREAM_DIR}/pids"
rm -f "${STREAM_DIR}/KILL_FLAG"

gpu_arr=(${GPUS})
num_shards="${#gpu_arr[@]}"
echo "[wrapper] GPUS=(${gpu_arr[*]})  num_shards=${num_shards}"
echo "[wrapper] TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
echo "[wrapper] STREAM_DIR=${STREAM_DIR}"
echo "[wrapper] PRIOR_MODE=${PRIOR_MODE}  num_systems_per_shard=${NUM_SYSTEMS_PER_SHARD}"

for shard_idx in "${!gpu_arr[@]}"; do
  gpu="${gpu_arr[${shard_idx}]}"
  log_file="${STREAM_DIR}/logs/daemon_shard${shard_idx}.log"
  pid_file="${STREAM_DIR}/pids/pid_shard${shard_idx}.txt"

  pristine_args=()
  [[ -n "${PRISTINE_SLABS}" ]] && pristine_args+=(--pristine-slabs "${PRISTINE_SLABS}")
  [[ -n "${PRISTINE_INDEX}" ]] && pristine_args+=(--pristine-index "${PRISTINE_INDEX}")

  echo "[wrapper] launching shard ${shard_idx} on GPU ${gpu} → ${log_file}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${CODE}:${PYTHONPATH:-}" \
    CAT_BENCH_ROOT="${ROOT}" \
    ADSORBATES_PKL="${ADSORBATES_PKL:-${ROOT}/data/pkls/adsorbates.pkl}" \
    MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  setsid ${PYTHON_BIN} "${CODE}/scripts/replay_daemon.py" \
    --train-run-dir "${TRAIN_RUN_DIR}" \
    --stream-dir "${STREAM_DIR}" \
    --shard-idx "${shard_idx}" \
    --num-shards "${num_shards}" \
    --gt-index "${GT_INDEX}" \
    --train-lmdb ${TRAIN_LMDB} \
    --num-systems "${NUM_SYSTEMS_PER_SHARD}" \
    --num-placements "${NUM_PLACEMENTS}" \
    --flow-steps "${FLOW_STEPS}" \
    --flow-batch-size "${FLOW_BATCH_SIZE}" \
    --prior-mode "${PRIOR_MODE}" \
    --uma-model "${UMA_MODEL}" --uma-fmax "${UMA_FMAX}" \
    --uma-max-steps "${UMA_MAX_STEPS}" --uma-atom-budget "${UMA_ATOM_BUDGET}" \
    --success-margin "${SUCCESS_MARGIN}" \
    $( [[ -n "${SUCCESS_MARGIN_FINAL}" ]] && echo "--success-margin-final ${SUCCESS_MARGIN_FINAL}" ) \
    --success-margin-schedule-cycles "${SUCCESS_MARGIN_SCHEDULE_CYCLES}" \
    --overlap-threshold "${OVERLAP_THRESHOLD}" \
    --viz-capture-n "${VIZ_CAPTURE_N}" \
    --chunk-size "${CHUNK_SIZE}" \
    --ckpt-settle-sec "${CKPT_SETTLE_SEC}" \
    --ckpt-stale-warn-min "${CKPT_STALE_WARN_MIN}" \
    --ckpt-stale-exit-min "${CKPT_STALE_EXIT_MIN}" \
    --max-cycles "${MAX_CYCLES}" \
    --util-poll-sec "${UTIL_POLL_SEC}" \
    "${pristine_args[@]}" \
    > "${log_file}" 2>&1 < /dev/null &

  echo "$!" > "${pid_file}"
  echo "[wrapper] shard ${shard_idx} pid=$(cat ${pid_file})"
done

echo "[wrapper] all ${num_shards} daemons launched. Touch ${STREAM_DIR}/KILL_FLAG to stop them."
