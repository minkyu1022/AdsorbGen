#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SUBSET_DIR="${SUBSET_DIR:-/home1/irteam/data/replay/goal1_overfit_subsets_20260615}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
LOG_ROOT="${LOG_ROOT:-/home1/irteam/runs/goal1_overfit_20260615_logs}"
METRIC_ROOT="${METRIC_ROOT:-/home1/irteam/data/replay/goal1_overfit_metrics_20260615}"
EVAL_SCRIPT="$REPO/experiments/2026-06-goal1-overfit/eval_goal1_overfit_metrics.py"
MERGE_SCRIPT="$REPO/experiments/2026-06-goal1-overfit/merge_goal1_eval_shards.py"
SHARD_ROOT="$METRIC_ROOT/base_n1000_shards"
OUT_DIR="$METRIC_ROOT/base_n1000"
NUM_SHARDS="${NUM_SHARDS:-8}"

mkdir -p "$LOG_ROOT" "$SHARD_ROOT" "$OUT_DIR"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES="$shard"
    "$PY" -u "$EVAL_SCRIPT" \
      --ckpt "$RUN_ROOT/goal1_overfit_base_n1000/last.ckpt" \
      --lmdb "$SUBSET_DIR/n1000/train.lmdb" \
      --out-dir "$SHARD_ROOT/shard${shard}" \
      --num-placements 3 \
      --num-shards "$NUM_SHARDS" \
      --shard-idx "$shard" \
      --batch-size 16 \
      --flow-steps 50 \
      --fmax 0.05 --max-steps 300 --max-atoms 4096 \
      --lbfgs-check-interval 20 --lbfgs-streaming --lbfgs-stream-sort \
      --geoopt-uma-model uma-s-1p1 --geoopt-uma-task oc20
  ) > "$LOG_ROOT/eval_n1000_shard${shard}.log" 2>&1 &
  echo $! > "$LOG_ROOT/eval_n1000_shard${shard}.pid"
done

wait
"$PY" -u "$MERGE_SCRIPT" \
  --eval-script "$EVAL_SCRIPT" \
  --shard-root "$SHARD_ROOT" \
  --out-dir "$OUT_DIR" \
  --num-shards "$NUM_SHARDS" \
  > "$LOG_ROOT/eval_n1000_merge.log" 2>&1
echo "[sharded-eval] done $(date '+%F %T')"
