#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
OUT="${OUT:-/home/irteam/runs/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544}"
TRAIN_LMDB="${TRAIN_LMDB:?TRAIN_LMDB is required}"
EPOCHS="${EPOCHS:-150}"

mkdir -p "${OUT}"

exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  MASTER_PORT="${MASTER_PORT:-46163}" \
  PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
  ADSORBATES_PKL=/home/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON_BIN}" -u -m adsorbgen.training.train_cli \
    --arch v1 \
    --train-lmdb "${TRAIN_LMDB}" \
    --val-lmdb /home/irteam/data/processed_old/oc20dense.lmdb \
    --val-lmdb-is2re /home/irteam/data/processed_ID/is2res_train_val_id_eval1000_seed20260522.lmdb \
    --batch-size 64 --num-workers 4 --epochs "${EPOCHS}" --devices 4 --precision bf16-mixed \
    --lr 1e-4 --lr-warmup-steps 5000 --grad-clip 10.0 --loss-type l1 \
    --loss-surf-weight 1.0 --loss-ads-weight 1.0 \
    --ads-pair-l1-weight 1.0 --ads-bond-l1-weight 0.0 --ads-nonbonded-clash-weight 0.0 \
    --ads-center-loss-weight 0.0 --ads-rel-pos-loss-weight 0.0 \
    --movable-mode surface_ads --prediction-type x1 --prior-mode random_heuristic --translation-std 0.5 \
    --sample-eval-every-epochs 1 --sample-eval-max-samples 1000 --sample-eval-steps 20 \
    --max-val-samples 1000 --check-val-every-n-epoch 1 --save-every-n-epochs 10 \
    --variant v0-ads-ref-adshead \
    --out "${OUT}" \
    --wandb-project adsorbgen --wandb-run-name ID_mlip_pairdist_only_1x_bs64
