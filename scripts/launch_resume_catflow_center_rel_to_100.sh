#!/usr/bin/env bash
# Resume H200_catflow_center_rel (CatFlow center+rel head, no aux loss) on
# GPU 4-7 from epoch 30 (last.ckpt) to epoch 100.
set -euo pipefail
source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
CODE="${PYTHONPATH%%:*}"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"

OUT="${ROOT}/runs/H200_catflow_center_rel"
VARIANT="v0-ads-ref-2x-statnorm-catflow-center-rel"

env \
  CUDA_VISIBLE_DEVICES="4,5,6,7" \
  PYTHONPATH="${CODE}:${PYTHONPATH:-}" \
  CAT_BENCH_ROOT="${ROOT}" \
  ADSORBATES_PKL="${ROOT}/data/pkls/adsorbates.pkl" \
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  WANDB_MODE="${WANDB_MODE:-online}" \
setsid "${MICROMAMBA}" run -n adsorbgen python -m adsorbgen.train \
  --arch v1 \
  --train-lmdb \
    "${ROOT}/data/processed/is2res_train.lmdb" \
    "${ROOT}/data/processed/is2res_val.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_ads.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_cat.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_both.lmdb" \
  --val-lmdb "${ROOT}/data/processed/oc20dense.lmdb" \
  --batch-size 48 --num-workers 4 --epochs 100 --devices 4 \
  --precision bf16-mixed --lr 1e-4 --lr-warmup-steps 5000 --grad-clip 10.0 \
  --loss-type l1 --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
  --ads-pair-l1-weight 0.0 --ads-bond-l1-weight 0.0 --ads-nonbonded-clash-weight 0.0 \
  --ads-center-loss-weight 1.0 --ads-rel-pos-loss-weight 1.0 \
  --prediction-type x1 --prior-mode catflow_center_rel \
  --sample-eval-every-epochs 1 --sample-eval-max-samples 1000 --sample-eval-steps 20 \
  --max-val-samples 1000 --check-val-every-n-epoch 1 \
  --variant "${VARIANT}" --out "${OUT}" \
  --wandb-project adsorbgen \
  --wandb-run-name "H200_catflow_center_rel_resume_100" \
  > "${OUT}/train_resume_100.log" 2>&1 < /dev/null &

echo "$!" > "${OUT}/pid_resume_100.txt"
echo "[pid] $(cat ${OUT}/pid_resume_100.txt)"
