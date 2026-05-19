#!/usr/bin/env bash
# Launch the "xt-only normalization" ablation: pos / ads_ref_pos / pair features
# stay in raw Å; only x_t (input to xt_proj) and the model's coord output go
# through the coord_scale wrapper. Two parallel 4-GPU runs, no auxiliary loss.
#
#   GPU 0-3:  v0-ads-ref-adshead-2x-fixedscale  (coord_scale = 4.0 uniform)
#   GPU 4-7:  v0-ads-ref-adshead-2x-statnorm    (coord_scale = train-stat std)
set -euo pipefail

source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
CODE="${PYTHONPATH%%:*}"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"

COMMON_ARGS=(
  --arch v1
  --train-lmdb
    "${ROOT}/data/processed/is2res_train.lmdb"
    "${ROOT}/data/processed/is2res_val.lmdb"
    "${ROOT}/data/processed/is2res_val_ood_ads.lmdb"
    "${ROOT}/data/processed/is2res_val_ood_cat.lmdb"
    "${ROOT}/data/processed/is2res_val_ood_both.lmdb"
  --val-lmdb "${ROOT}/data/processed/oc20dense.lmdb"
  --batch-size 48
  --num-workers 4
  --epochs 30
  --devices 4
  --precision bf16-mixed
  --lr 1e-4
  --lr-warmup-steps 5000
  --grad-clip 10.0
  --loss-type l1
  --loss-surf-weight 1.0
  --loss-ads-weight 1.0
  --ads-pair-l1-weight 0.0
  --ads-bond-l1-weight 0.0
  --ads-nonbonded-clash-weight 0.0
  --prediction-type x1
  --prior-mode random_heuristic
  --sample-eval-every-epochs 1
  --sample-eval-max-samples 1000
  --sample-eval-steps 20
  --max-val-samples 1000
  --check-val-every-n-epoch 1
)

launch_one() {
  local gpus="$1"
  local out="$2"
  local variant="$3"
  local exp_name="$4"

  mkdir -p "${out}"
  echo "[launch] GPUs=${gpus} out=${out} variant=${variant} exp=${exp_name}"
  env \
    CUDA_VISIBLE_DEVICES="${gpus}" \
    PYTHONPATH="${CODE}:${PYTHONPATH:-}" \
    CAT_BENCH_ROOT="${ROOT}" \
    ADSORBATES_PKL="${ROOT}/data/pkls/adsorbates.pkl" \
    MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    WANDB_MODE="${WANDB_MODE:-online}" \
  setsid "${MICROMAMBA}" run -n adsorbgen python -m adsorbgen.train \
    "${COMMON_ARGS[@]}" \
    --variant "${variant}" \
    --out "${out}" \
    --wandb-project adsorbgen \
    --wandb-run-name "${exp_name}" \
    > "${out}/train.log" 2>&1 < /dev/null &
  echo "$!" > "${out}/pid.txt"
  echo "[pid] $(cat "${out}/pid.txt")"
}

launch_one \
  "0,1,2,3" \
  "${ROOT}/runs/H200_xt_only_fixedscale" \
  "v0-ads-ref-adshead-2x-fixedscale" \
  "H200_xt_only_fixedscale"

launch_one \
  "4,5,6,7" \
  "${ROOT}/runs/H200_xt_only_statnorm" \
  "v0-ads-ref-adshead-2x-statnorm" \
  "H200_xt_only_statnorm"
