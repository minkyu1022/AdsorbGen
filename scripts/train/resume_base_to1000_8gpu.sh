#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
OUT="${OUT:-/home1/irteam/runs/training/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544}"
LOG="${LOG:-${OUT}/train_resume_wandb_wqk23mfz_to1000_8gpu_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$OUT"
echo "$LOG" > "${OUT}/latest_resume_to1000_log.txt"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-wqk23mfz}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$PY" -u -m adsorbgen.training.train_cli \
  --train-lmdb /home1/irteam/data/processed_ID/is2res_train.lmdb /home1/irteam/data/processed_ID/is2res_val.lmdb \
  --val-lmdb /home1/irteam/data/processed_old/oc20dense.lmdb \
  --out "$OUT" \
  --batch-size 64 --num-workers 4 --epochs 1000 --devices 8 \
  --precision bf16-mixed --lr 0.0001 --weight-decay 0.0 --grad-clip 10.0 \
  --accumulate-grad-batches 1 --lr-warmup-steps 5000 --log-every 50 \
  --dim 512 --pair-dim 256 --depth 16 --num-heads 8 --mlp-ratio 4.0 --dropout 0.0 \
  --translation-std 0.5 --prior-mode random_heuristic --interstitial-gap 0.1 \
  --variant v0-ads-ref-adshead --arch v1 --loss-type l1 \
  --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
  --ads-pair-l1-weight 1.0 --ads-bond-factor 1.25 --ads-clash-factor 0.75 \
  --ads-center-loss-weight 0.0 --ads-rel-pos-loss-weight 0.0 --movable-mode surface_ads \
  --flow-eps 1e-5 --prediction-type x1 --seed 0 \
  --sample-eval-every-epochs 1 --sample-eval-max-samples 1000 --sample-eval-steps 20 \
  --sample-eval-energy-uma-model uma-s-1p1 --sample-eval-energy-uma-task oc20 \
  --sample-eval-energy-batch-size 32 --sample-eval-energy-success-margin 0.1 \
  --max-val-samples 1000 --check-val-every-n-epoch 1 --save-every-n-epochs 10 \
  --wandb-project adsorbgen --wandb-run-name ID_mlip_pairdist_only_1x_bs64
