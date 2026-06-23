#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
OUT="${OUT:-${RUN_ROOT}/v_LP_vector_102M}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
LOG="${LOG:-${OUT}/train.log}"

mkdir -p "${OUT}"

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

exec "${PY}" -u -m adsorbgen.training.train_cli \
  --arch v1 \
  --train-lmdb /home1/irteam/data/processed_ID/is2res_train.lmdb /home1/irteam/data/processed_ID/is2res_val.lmdb \
  --val-lmdb /home1/irteam/data/processed_old/oc20dense.lmdb \
  --out "${OUT}" \
  --batch-size 64 \
  --num-workers 4 \
  --epochs 500 \
  --devices 8 \
  --precision bf16-mixed \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --grad-clip 10.0 \
  --accumulate-grad-batches 1 \
  --lr-warmup-steps 5000 \
  --log-every 50 \
  --variant v0-ads-ref-adshead \
  --loss-type l1 \
  --loss-surf-weight 1.0 \
  --loss-ads-weight 1.0 \
  --ads-pair-l1-weight 1.0 \
  --ads-bond-l1-weight 0.0 \
  --ads-nonbonded-clash-weight 0.0 \
  --ads-center-loss-weight 0.0 \
  --ads-rel-pos-loss-weight 0.0 \
  --movable-mode surface_ads \
  --prior-mode random_heuristic \
  --slab-source initial \
  --flow-eps 1e-5 \
  --prediction-type v \
  --seed 0 \
  --use-langevin-param \
  --langevin-scale-mode vector \
  --langevin-uma-model uma-s-1p2 \
  --langevin-uma-task oc20 \
  --langevin-force-clip 100.0 \
  --langevin-eval-on x_t \
  --sample-eval-every-epochs 1 \
  --sample-eval-max-samples 1000 \
  --sample-eval-steps 20 \
  --sample-eval-energy-cover-dir "${COVER}" \
  --sample-eval-energy-uma-model uma-s-1p1 \
  --sample-eval-energy-uma-task oc20 \
  --sample-eval-energy-batch-size 32 \
  --sample-eval-energy-success-margin 0.1 \
  --max-val-samples 1000 \
  --check-val-every-n-epoch 1 \
  --save-every-n-epochs 10 \
  --wandb-project adsorbgen \
  --wandb-run-name v_LP_vector_102M
