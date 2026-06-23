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
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_for_train_done() {
  local name="$1"
  local ckpt="$RUN_ROOT/${name}/last.ckpt"
  echo "[remaining] waiting train $name"
  while true; do
    local alive
    alive="$(ps -eo cmd | rg "${name}" | rg 'adsorbgen.training.train_cli' | rg -v rg | wc -l)"
    echo "[remaining] $name alive=$alive ckpt=$([[ -f "$ckpt" ]] && echo yes || echo no) $(date '+%F %T')"
    if [[ "$alive" -eq 0 && -f "$ckpt" ]]; then
      break
    fi
    sleep 120
  done
}

eval_goal1() {
  local n="$1"
  local log="$LOG_ROOT/eval_n${n}.log"
  echo "[remaining] eval n=$n"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES=0
    "$PY" -u "$REPO/experiments/2026-06-goal1-overfit/eval_goal1_overfit_metrics.py" \
      --ckpt "$RUN_ROOT/goal1_overfit_base_n${n}/last.ckpt" \
      --lmdb "$SUBSET_DIR/n${n}/train.lmdb" \
      --out-dir "$METRIC_ROOT/base_n${n}" \
      --num-placements 3 \
      --batch-size 16 \
      --flow-steps 50 \
      --fmax 0.05 --max-steps 300 --max-atoms 4096 \
      --lbfgs-check-interval 20 --lbfgs-streaming --lbfgs-stream-sort \
      --geoopt-uma-model uma-s-1p1 --geoopt-uma-task oc20
  ) > "$log" 2>&1
  echo "[remaining] eval done n=$n"
}

train_n1000() {
  local out="$RUN_ROOT/goal1_overfit_base_n1000"
  if [[ -e "$out/last.ckpt" || -e "$out/args.json" ]]; then
    echo "[remaining] refusing to overwrite existing run dir: $out" >&2
    exit 3
  fi
  mkdir -p "$out"
  echo "[remaining] train n=1000 on GPUs 1-7"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    "$PY" -u -m adsorbgen.training.train_cli \
      --train-lmdb "$SUBSET_DIR/n1000/train.lmdb" \
      --val-lmdb "$SUBSET_DIR/n1000/train.lmdb" \
      --out "$out" \
      --batch-size 64 --num-workers 4 --epochs 300 --devices 7 \
      --precision bf16-mixed --lr 0.0001 --weight-decay 0.0 --grad-clip 10.0 \
      --accumulate-grad-batches 1 --lr-warmup-steps 500 --log-every 20 \
      --dim 512 --pair-dim 128 --depth 13 --num-heads 8 --mlp-ratio 4.0 --dropout 0.0 \
      --translation-std 0.5 --prior-mode random_heuristic --interstitial-gap 0.1 \
      --variant v0-ads-ref-adshead --arch v1 --loss-type l1 \
      --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
      --ads-pair-l1-weight 1.0 --ads-bond-factor 1.25 --ads-clash-factor 0.75 \
      --ads-center-loss-weight 0.0 --ads-rel-pos-loss-weight 0.0 --movable-mode surface_ads \
      --flow-eps 1e-5 --prediction-type x1 --seed 0 \
      --train-replicate 16 --val-replicate 3 \
      --sample-eval-every-epochs 0 --max-val-samples 1000 \
      --check-val-every-n-epoch 25 --save-every-n-epochs 25 \
      --wandb-project adsorbgen --wandb-run-name goal1_overfit_base_n1000
  ) > "$LOG_ROOT/train_n1000.log" 2>&1 &
  echo $! > "$LOG_ROOT/train_n1000.pid"
}

main() {
  wait_for_train_done goal1_overfit_base_n100
  eval_goal1 100 &
  echo $! > "$LOG_ROOT/eval_n100.pid"
  train_n1000
  wait
  wait_for_train_done goal1_overfit_base_n1000
  eval_goal1 1000
  echo "[remaining] all remaining done $(date '+%F %T')"
}

main "$@"
