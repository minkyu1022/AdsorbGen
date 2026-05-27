#!/usr/bin/env bash
# One-off 5,000 unique-systems × 10 placements replay against a fixed ckpt.
# Spreads the work across 8 GPUs (625 systems / shard), runs exactly ONE
# cycle, dumps every per-candidate prediction so the report script can
# compute strict (E_pred < E_gt) + relaxed (+0.1/+0.2/+0.3 eV) tallies and
# both success rates (#systems-with-≥1-success / 5000 vs. #success-candidates / 50000).
#
# E_gt source: gt_index_by_sid_oc20.pkl  (E_sys_min per system — the lowest
#              UMA oc20 relaxed energy among the system's training configs).
set -euo pipefail
source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
REPO="${REPO:-/home/irteam/AdsorbGen}"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"

# --- required-ish (override via env) ---
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${ROOT}/runs/H200_ads_pair_dist_loss}"
STREAM_DIR="${STREAM_DIR:-${ROOT}/runs/replay_5000x10_oneshot}"
GT_INDEX="${GT_INDEX:-${ROOT}/data/replay/gt_index_by_sid_oc20.pkl}"
LMDB_DIR="${LMDB_DIR:-${ROOT}/data/processed}"
PRIOR_MODE="${PRIOR_MODE:-random_heuristic}"
PRISTINE_PKL="${PRISTINE_PKL:-${ROOT}/results/pristine_slabs/is2res.pkl}"
PRISTINE_IDX="${PRISTINE_IDX:-${ROOT}/results/pristine_slabs/is2res.sid_index.pkl}"

# --- replay budget knobs ---
NUM_SHARDS=${NUM_SHARDS:-8}
GPU_LIST="${GPU_LIST:-}"
NUM_SYSTEMS_PER_SHARD=${NUM_SYSTEMS_PER_SHARD:-625}   # 625 * 8 = 5000
NUM_PLACEMENTS=${NUM_PLACEMENTS:-10}                   # → 50,000 candidates
FLOW_STEPS=${FLOW_STEPS:-50}
FLOW_BATCH_SIZE=${FLOW_BATCH_SIZE:-32}
UMA_MAX_STEPS=${UMA_MAX_STEPS:-300}
UMA_ATOM_BUDGET=${UMA_ATOM_BUDGET:-4000}
UMA_FMAX=${UMA_FMAX:-0.05}
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
USE_SDE="${USE_SDE:-0}"
REFINE_FINAL="${REFINE_FINAL:-0}"
SYSTEM_SEED="${SYSTEM_SEED:--1}"

LMDBS=(
  "${LMDB_DIR}/is2res_train.lmdb"
  "${LMDB_DIR}/is2res_val.lmdb"
  "${LMDB_DIR}/is2res_val_ood_ads.lmdb"
  "${LMDB_DIR}/is2res_val_ood_cat.lmdb"
  "${LMDB_DIR}/is2res_val_ood_both.lmdb"
)

mkdir -p "${STREAM_DIR}/logs"

echo "[launch] TRAIN_RUN_DIR=${TRAIN_RUN_DIR}"
echo "[launch] STREAM_DIR=${STREAM_DIR}"
echo "[launch] ${NUM_SYSTEMS_PER_SHARD} systems / shard × ${NUM_SHARDS} shards × ${NUM_PLACEMENTS} placements"
echo "[launch] = $((NUM_SYSTEMS_PER_SHARD * NUM_SHARDS * NUM_PLACEMENTS)) total candidates"
echo "[launch] E_gt source = E_sys_min (gt_index_by_sid_oc20.pkl), success_margin = 0.0"
echo "[launch] USE_SDE=${USE_SDE}  REFINE_FINAL=${REFINE_FINAL}"
echo "[launch] SYSTEM_SEED=${SYSTEM_SEED}  GPU_LIST=${GPU_LIST:-0..$((NUM_SHARDS - 1))}"

EXTRA_REPLAY_ARGS=()
if [[ "${USE_SDE}" == "1" || "${USE_SDE}" == "true" || "${USE_SDE}" == "True" ]]; then
  EXTRA_REPLAY_ARGS+=(--use-sde)
fi
if [[ "${REFINE_FINAL}" == "1" || "${REFINE_FINAL}" == "true" || "${REFINE_FINAL}" == "True" ]]; then
  EXTRA_REPLAY_ARGS+=(--refine-final)
fi

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  if [[ -n "${GPU_LIST}" ]]; then
    IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
    cuda="${GPUS[$shard]}"
  else
    cuda="${shard}"
  fi
  log="${STREAM_DIR}/logs/launch_shard_${shard}.log"
  echo "[launch] shard $shard -> $log"
  (
    cd "${REPO}"
    echo "[launch] $(date -Is) shard=$shard cuda=$cuda"
    exec setsid -f env CUDA_VISIBLE_DEVICES="$cuda" PYTHONUNBUFFERED=1 \
      PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
      CAT_BENCH_ROOT="${ROOT}" \
      MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
      PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
      "${MICROMAMBA}" run -n adsorbgen python scripts/replay/replay_daemon.py \
        --train-run-dir "${TRAIN_RUN_DIR}" \
        --stream-dir "${STREAM_DIR}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --gt-index "${GT_INDEX}" \
        --train-lmdb "${LMDBS[@]}" \
        --num-systems "${NUM_SYSTEMS_PER_SHARD}" \
        --num-placements "${NUM_PLACEMENTS}" \
        --flow-steps "${FLOW_STEPS}" \
        --flow-batch-size "${FLOW_BATCH_SIZE}" \
        --prior-mode "${PRIOR_MODE}" \
        --uma-model "${UMA_MODEL}" \
        --uma-fmax "${UMA_FMAX}" \
        --uma-max-steps "${UMA_MAX_STEPS}" \
        --uma-atom-budget "${UMA_ATOM_BUDGET}" \
        --success-margin 0.0 \
        --success-margin-schedule-cycles 0 \
        --overlap-threshold 0.5 \
        --viz-capture-n 0 \
        --chunk-size 64 \
        --ckpt-settle-sec 5.0 \
        --ckpt-stale-warn-min 1000000 \
        --ckpt-stale-exit-min 1000000 \
        --max-cycles 1 \
        --util-poll-sec 1.0 \
        --system-seed "${SYSTEM_SEED}" \
        --pristine-slabs "${PRISTINE_PKL}" \
        --pristine-index "${PRISTINE_IDX}" \
        --e-gt-key E_sys_min \
        --collect-predictions \
        "${EXTRA_REPLAY_ARGS[@]}"
  ) >"$log" 2>&1 &
  echo $! >"${STREAM_DIR}/logs/launch_shard_${shard}.pid"
  sleep "${LAUNCH_STAGGER_SEC:-2}"
done

echo "[launch] dispatched ${NUM_SHARDS} shard processes"
echo "[launch] logs: ${STREAM_DIR}/logs"
echo "[launch] cycle JSON: ${STREAM_DIR}/logs/cycle_000000_shard*.json (after completion)"
echo "[launch] predictions pkl: ${STREAM_DIR}/logs/cycle_000000_shard*_predictions.pkl"
