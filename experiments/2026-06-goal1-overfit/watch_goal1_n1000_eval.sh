#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SUBSET_DIR="${SUBSET_DIR:-/home1/irteam/data/replay/goal1_overfit_subsets_20260615}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
LOG_ROOT="${LOG_ROOT:-/home1/irteam/runs/goal1_overfit_20260615_logs}"
METRIC_ROOT="${METRIC_ROOT:-/home1/irteam/data/replay/goal1_overfit_metrics_20260615}"
mkdir -p "$LOG_ROOT" "$METRIC_ROOT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-online}"

name="goal1_overfit_base_n1000"
ckpt="$RUN_ROOT/${name}/last.ckpt"

echo "[n1000-watch] waiting $name $(date '+%F %T')"
while true; do
  alive="$(pgrep -af "adsorbgen.training.train_cli.*${name}" | wc -l || true)"
  has_ckpt=no
  [[ -f "$ckpt" ]] && has_ckpt=yes
  echo "[n1000-watch] alive=$alive ckpt=$has_ckpt $(date '+%F %T')"
  if [[ "$alive" -eq 0 && -f "$ckpt" ]]; then
    break
  fi
  sleep 120
done

echo "[n1000-watch] eval n=1000 $(date '+%F %T')"
(
  cd "$REPO"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  "$PY" -u "$REPO/experiments/2026-06-goal1-overfit/eval_goal1_overfit_metrics.py" \
    --ckpt "$ckpt" \
    --lmdb "$SUBSET_DIR/n1000/train.lmdb" \
    --out-dir "$METRIC_ROOT/base_n1000" \
    --num-placements 3 \
    --batch-size 16 \
    --flow-steps 50 \
    --fmax 0.05 --max-steps 300 --max-atoms 4096 \
    --lbfgs-check-interval 20 --lbfgs-streaming --lbfgs-stream-sort \
    --geoopt-uma-model uma-s-1p1 --geoopt-uma-task oc20
) > "$LOG_ROOT/eval_n1000_manual.log" 2>&1
echo "[n1000-watch] done $(date '+%F %T')"
