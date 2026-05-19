#!/usr/bin/env bash
# 4-GPU parallel replay eval for one checkpoint.
#
# Each shard owns one GPU (CUDA_VISIBLE_DEVICES=N), runs ~num_systems/4 systems
# × num_placements via fast_dynamics batched FIRE. After all 4 finish,
# scripts/merge_replay_shards.py consolidates the per-shard viz dirs into one
# replay_viz/ep{TAG}/ for the web UI.
#
# Usage:
#   bash scripts/run_replay_4gpu.sh
set -euo pipefail

CODE_REPO="${CODE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CAT_BENCH_ROOT="${CAT_BENCH_ROOT:-$(dirname "${CODE_REPO}")}"
PYTHON="${PYTHON:-$(command -v python)}"

RUN_DIR="${RUN_DIR:-${CAT_BENCH_ROOT}/runs/full_v0_ads_ref_x1_l1_allpairL1ref1_noreplay}"
CKPT="${CKPT:-${RUN_DIR}/ckpt_epochepoch=029.ckpt}"
GT_INDEX="${GT_INDEX:-${CAT_BENCH_ROOT}/data/replay/gt_index_by_sid.pkl}"
TRAIN_LMDB="${TRAIN_LMDB:-${CAT_BENCH_ROOT}/data/processed/is2res_train.lmdb}"

EPOCH_TAG="${EPOCH_TAG:-30}"
NUM_SYSTEMS="${NUM_SYSTEMS:-500}"
NUM_PLACEMENTS="${NUM_PLACEMENTS:-3}"
UMA_MAX_STEPS="${UMA_MAX_STEPS:-100}"
PRISTINE_SLABS="${PRISTINE_SLABS:-${CAT_BENCH_ROOT}/results/pristine_slabs/is2res.pkl}"
PRISTINE_SID_INDEX="${PRISTINE_SID_INDEX:-${CAT_BENCH_ROOT}/results/pristine_slabs/is2res.sid_index.pkl}"
GPUS=(${GPUS:-0 1 2 3})
NUM_SHARDS="${#GPUS[@]}"

SHARD_ROOT="${RUN_DIR}/replay_shards"
FINAL_VIZ_ROOT="${RUN_DIR}/replay_viz"
FINAL_BUFFER="${RUN_DIR}/replay_buffer_ep${EPOCH_TAG}.pkl"
FINAL_METRICS="${RUN_DIR}/replay_metrics_ep${EPOCH_TAG}.json"

echo "============================================================"
echo "  Replay eval (4-GPU shard)"
echo "  ckpt:        ${CKPT}"
echo "  gt_index:    ${GT_INDEX}"
echo "  train_lmdb:  ${TRAIN_LMDB}"
echo "  num_systems: ${NUM_SYSTEMS}  num_placements: ${NUM_PLACEMENTS}"
echo "  GPUs:        ${GPUS[*]}  (shards: ${NUM_SHARDS})"
echo "  shard_root:  ${SHARD_ROOT}"
echo "  final viz:   ${FINAL_VIZ_ROOT}/ep${EPOCH_TAG}"
echo "============================================================"

mkdir -p "${SHARD_ROOT}"
rm -rf "${SHARD_ROOT}"/shard_*  # fresh

PIDS=()
for i in "${!GPUS[@]}"; do
  GPU="${GPUS[$i]}"
  SHARD_DIR="${SHARD_ROOT}/shard_${i}"
  mkdir -p "${SHARD_DIR}"
  LOG="${SHARD_DIR}/worker.log"
  BUF="${SHARD_DIR}/buffer.pkl"
  MET="${SHARD_DIR}/metrics.json"

  echo "[launch] shard ${i} on GPU ${GPU} → ${LOG}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  PYTHONPATH="${CODE_REPO}:${PYTHONPATH:-}" \
    "${PYTHON}" "${CODE_REPO}/scripts/replay_one_ckpt.py" \
      --ckpt "${CKPT}" \
      --gt-index "${GT_INDEX}" \
      --train-lmdb "${TRAIN_LMDB}" \
      --viz-root "${SHARD_DIR}/viz" \
      --buffer-path "${BUF}" \
      --metrics-path "${MET}" \
      --epoch-tag "${EPOCH_TAG}" \
      --num-systems "${NUM_SYSTEMS}" \
      --num-placements "${NUM_PLACEMENTS}" \
      --uma-max-steps "${UMA_MAX_STEPS}" \
      --pristine-slabs "${PRISTINE_SLABS}" \
      --pristine-index "${PRISTINE_SID_INDEX}" \
      --shard-idx "${i}" \
      --num-shards "${NUM_SHARDS}" \
      --viz-capture-n 8 \
      > "${LOG}" 2>&1 &
  PIDS+=("$!")
done

echo "[launch] all ${NUM_SHARDS} shards launched. Waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[done]  shard ${i} OK"
  else
    echo "[FAIL]  shard ${i} (pid ${PIDS[$i]}) — see ${SHARD_ROOT}/shard_${i}/worker.log"
    FAIL=1
  fi
done

if [[ "${FAIL}" -ne 0 ]]; then
  echo "[abort] one or more shards failed; skipping merge."
  exit 1
fi

echo ""
echo "[merge] consolidating shards → ${FINAL_VIZ_ROOT}/ep${EPOCH_TAG}"
PYTHONPATH="${CODE_REPO}:${PYTHONPATH:-}" \
"${PYTHON}" "${CODE_REPO}/scripts/merge_replay_shards.py" \
  --shard-root "${SHARD_ROOT}" \
  --final-viz-root "${FINAL_VIZ_ROOT}" \
  --final-buffer "${FINAL_BUFFER}" \
  --final-metrics "${FINAL_METRICS}" \
  --epoch-tag "${EPOCH_TAG}"

echo ""
echo "[done] replay eval complete."
echo "  viz:     ${FINAL_VIZ_ROOT}/ep${EPOCH_TAG}"
echo "  buffer:  ${FINAL_BUFFER}"
echo "  metrics: ${FINAL_METRICS}"
echo ""
echo "Launch UI:"
echo "  REPLAY_VIZ_ROOT=${FINAL_VIZ_ROOT} bash ${CODE_REPO}/viz/run_viz.sh backend"
echo "  bash ${CODE_REPO}/viz/run_viz.sh frontend"
