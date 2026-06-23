#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
ACTIVE="${ACTIVE:-/home1/irteam/data/replay/adsorbdiff_passk_score32_active.txt}"
SUBSET_DIR="${SUBSET_DIR:-/home1/irteam/data/replay/goal1_overfit_subsets_20260615}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
LOG_ROOT="${LOG_ROOT:-/home1/irteam/runs/goal1_overfit_20260615_logs}"
mkdir -p "$LOG_ROOT"

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

wait_adsorbdiff() {
  if [[ ! -f "$ACTIVE" ]]; then
    echo "[goal1] no active AdsorbDiff pointer; skip wait"
    return 0
  fi
  local out
  out="$(cat "$ACTIVE")"
  echo "[goal1] waiting AdsorbDiff score32: $out"
  while true; do
    local alive json_count
    alive="$(pgrep -fc 'score_adsorbdiff_mlip_pass_ood50.py' || true)"
    json_count="$(find "$out" -maxdepth 1 -name 'shard_*.json' 2>/dev/null | wc -l)"
    echo "[goal1] adsorbdiff alive=$alive shard_json=$json_count/32 $(date '+%F %T')"
    if [[ "$json_count" -eq 32 ]]; then
      "$PY" "$REPO/experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py" \
        --out-dir "$out" --num-shards 32 --num-samples 100 \
        > "$LOG_ROOT/adsorbdiff_merge.log" 2>&1
      echo "[goal1] AdsorbDiff merged: $out/summary.json"
      break
    fi
    if [[ "$alive" -eq 0 ]]; then
      echo "[goal1] AdsorbDiff ended before all shard json files were written" >&2
      exit 2
    fi
    sleep 60
  done
}

make_subsets() {
  "$PY" "$REPO/experiments/2026-06-goal1-overfit/create_goal1_overfit_subsets.py" \
    --out-dir "$SUBSET_DIR" \
    --sizes 10,100,1000 \
    --seed 20260615 \
    > "$LOG_ROOT/create_subsets.log" 2>&1
  echo "[goal1] subsets ready: $SUBSET_DIR"
}

train_goal1() {
  local n="$1"
  local replicate="$2"
  local epochs="$3"
  local subset="$SUBSET_DIR/n${n}/train.lmdb"
  local out="$RUN_ROOT/goal1_overfit_base_n${n}"
  local log="$LOG_ROOT/train_n${n}.log"

  echo "[goal1] train n=$n replicate=$replicate epochs=$epochs out=$out"
  if [[ -e "$out/last.ckpt" || -e "$out/args.json" ]]; then
    echo "[goal1] refusing to overwrite existing run dir: $out" >&2
    exit 3
  fi
  mkdir -p "$out"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    "$PY" -u -m adsorbgen.training.train_cli \
      --train-lmdb "$subset" \
      --val-lmdb "$subset" \
      --out "$out" \
      --batch-size 64 --num-workers 4 --epochs "$epochs" --devices 8 \
      --precision bf16-mixed --lr 0.0001 --weight-decay 0.0 --grad-clip 10.0 \
      --accumulate-grad-batches 1 --lr-warmup-steps 500 --log-every 20 \
      --dim 512 --pair-dim 128 --depth 13 --num-heads 8 --mlp-ratio 4.0 --dropout 0.0 \
      --translation-std 0.5 --prior-mode random_heuristic --interstitial-gap 0.1 \
      --variant v0-ads-ref-adshead --arch v1 --loss-type l1 \
      --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
      --ads-pair-l1-weight 1.0 --ads-bond-factor 1.25 --ads-clash-factor 0.75 \
      --ads-center-loss-weight 0.0 --ads-rel-pos-loss-weight 0.0 --movable-mode surface_ads \
      --flow-eps 1e-5 --prediction-type x1 --seed 0 \
      --train-replicate "$replicate" --val-replicate 3 \
      --sample-eval-every-epochs 0 --max-val-samples 1000 \
      --check-val-every-n-epoch 25 --save-every-n-epochs 25 \
      --wandb-project adsorbgen --wandb-run-name "goal1_overfit_base_n${n}"
  ) > "$log" 2>&1
  echo "[goal1] train done n=$n"
}

eval_goal1() {
  local n="$1"
  local subset="$SUBSET_DIR/n${n}/train.lmdb"
  local ckpt="$RUN_ROOT/goal1_overfit_base_n${n}/last.ckpt"
  local out="/home1/irteam/data/replay/goal1_overfit_metrics_20260615/base_n${n}"
  local log="$LOG_ROOT/eval_n${n}.log"
  echo "[goal1] eval n=$n ckpt=$ckpt"
  mkdir -p "$out"
  (
    cd "$REPO"
    export CUDA_VISIBLE_DEVICES=0
    "$PY" -u "$REPO/experiments/2026-06-goal1-overfit/eval_goal1_overfit_metrics.py" \
      --ckpt "$ckpt" \
      --lmdb "$subset" \
      --out-dir "$out" \
      --num-placements 3 \
      --batch-size 16 \
      --flow-steps 50 \
      --fmax 0.05 --max-steps 300 --max-atoms 4096 \
      --lbfgs-check-interval 20 --lbfgs-streaming --lbfgs-stream-sort \
      --geoopt-uma-model uma-s-1p1 --geoopt-uma-task oc20
  ) > "$log" 2>&1
  echo "[goal1] eval done n=$n summary=$out/summary.json"
}

main() {
  wait_adsorbdiff
  make_subsets
  train_goal1 10 512 300
  eval_goal1 10
  train_goal1 100 128 300
  eval_goal1 100
  train_goal1 1000 16 300
  eval_goal1 1000
  echo "[goal1] all done $(date '+%F %T')"
}

main "$@"
