#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
DATA_ROOT="${DATA_ROOT:-/home1/irteam/data/uma_s_1p2_references}"

RUN_NAME="${RUN_NAME:-valID_x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform_clean_ep200}"
OUT="${OUT:-${RUN_ROOT}/${RUN_NAME}}"

mkdir -p "${OUT}/logs"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export MASTER_PORT="${MASTER_PORT:-29691}"

TRAIN_LMDB="${DATA_ROOT}/processed/is2res_val_unwrap_centered.lmdb"
DENSE_VAL_LMDB="${DATA_ROOT}/processed/oc20dense_unwrap_centered.lmdb"
ENERGY_COVER="${DATA_ROOT}/materialized/oc20dense_raw"
PRISTINE_SLABS="${PRISTINE_SLABS:-${DATA_ROOT}/materialized/bare_slab/bare_slabs_lbfgs.pkl}"
PRISTINE_INDEX="${PRISTINE_INDEX:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.sid_index.pkl}"
VAL_PRISTINE_SLABS="${VAL_PRISTINE_SLABS:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl}"
VAL_PRISTINE_INDEX="${VAL_PRISTINE_INDEX:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl}"

WANDB_ARGS=()
if [[ -n "${WANDB_ID:-}" ]]; then
  WANDB_ARGS+=(--wandb-id "${WANDB_ID}" --wandb-resume "${WANDB_RESUME:-allow}")
fi

exec "${PY}" -u -m adsorbgen.training.train_cli \
  --train-lmdb "${TRAIN_LMDB}" \
  --train-replicate 1 \
  --val-lmdb "${DENSE_VAL_LMDB}" \
  --out "${OUT}" \
  --batch-size 64 --num-workers 4 --epochs 200 --devices 8 \
  --precision bf16-mixed --lr 0.0001 --weight-decay 0.0 --grad-clip 10.0 \
  --accumulate-grad-batches 1 --lr-warmup-steps 5000 --log-every 50 \
  --dim 512 --pair-dim 256 --depth 16 --num-heads 8 --mlp-ratio 4.0 --dropout 0.0 \
  --translation-std 0.5 --prior-mode random_heuristic --interstitial-gap 0.1 \
  --variant v0-ads-ref-adshead --arch v1 --loss-type l1 \
  --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
  --ads-pair-l1-weight 1.0 --ads-bond-factor 1.25 --ads-clash-factor 0.75 \
  --ads-center-loss-weight 0.0 --ads-rel-pos-loss-weight 0.0 --movable-mode surface_ads \
  --slab-source initial \
  --pristine-slabs "${PRISTINE_SLABS}" \
  --pristine-index "${PRISTINE_INDEX}" \
  --val-pristine-slabs "${VAL_PRISTINE_SLABS}" \
  --val-pristine-index "${VAL_PRISTINE_INDEX}" \
  --flow-eps 1e-3 --prediction-type x1 --loss-target v --seed 0 \
  --gamma-schedule sqrt_t1mt --gamma-sigma 0.1 \
  --train-time-sampling uniform \
  --train-time-beta-alpha 2.0 --train-time-beta-beta 1.0 \
  --use-si-denoiser --si-denoiser-loss-weight 0.5 --si-denoiser-mask movable --si-denoiser-use-ads-specific-head \
  --sample-eval-every-epochs 1 --sample-eval-max-samples 1000 --sample-eval-steps 50 \
  --sample-eval-energy-cover-dir "${ENERGY_COVER}" \
  --sample-eval-energy-uma-model uma-s-1p2 --sample-eval-energy-uma-task oc20 \
  --sample-eval-energy-batch-size 32 --sample-eval-energy-success-margin 0.1 \
  --max-val-samples 1000 --check-val-every-n-epoch 1 --save-every-n-epochs 10 \
  --wandb-project adsorbgen --wandb-run-name "${RUN_NAME}" \
  "${WANDB_ARGS[@]}"
