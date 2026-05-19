#!/usr/bin/env bash
# Two 4-GPU H200 ablations based on H200_xt_only_statnorm:
#
#   GPU 0-3: AdsorbSample-style harmonic adsorbate x_0 prior
#   GPU 4-7: CatFlow-style ads center + rel-pos prior and output heads
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
  --ads-center-loss-weight 1.0
  --ads-rel-pos-loss-weight 1.0
  --prediction-type x1
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
  local prior="$4"
  local exp_name="$5"

  if [[ -e "${out}/last.ckpt" ]]; then
    echo "[error] ${out}/last.ckpt exists; refusing to resume/overwrite" >&2
    exit 1
  fi
  mkdir -p "${out}"
  echo "[launch] GPUs=${gpus} out=${out} variant=${variant} prior=${prior} exp=${exp_name}"
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
    --prior-mode "${prior}" \
    --out "${out}" \
    --wandb-project adsorbgen \
    --wandb-run-name "${exp_name}" \
    > "${out}/train.log" 2>&1 < /dev/null &
  echo "$!" > "${out}/pid.txt"
  echo "[pid] $(cat "${out}/pid.txt")"
}

launch_one \
  "0,1,2,3" \
  "${ROOT}/runs/H200_harmonic_prior" \
  "v0-ads-ref-adshead-2x-statnorm" \
  "harmonic_uniform" \
  "H200_harmonic_prior"

launch_one \
  "4,5,6,7" \
  "${ROOT}/runs/H200_catflow_center_rel" \
  "v0-ads-ref-2x-statnorm-catflow-center-rel" \
  "catflow_center_rel" \
  "H200_catflow_center_rel"
