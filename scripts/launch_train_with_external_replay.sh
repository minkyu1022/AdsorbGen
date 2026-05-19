#!/usr/bin/env bash
# Launch (or resume) a training run on GPU 0-3 wired to consume the replay
# stream produced by scripts/run_replay_daemon_4gpu.sh.
#
# The replay daemon writes successful entries into ${STREAM_DIR}. This
# training process reads them at every epoch end via --external-replay-dir.
#
# Usage:
#   OUT=/home/irteam/runs/train_with_replay \
#   STREAM_DIR=/home/irteam/runs/train_with_replay/replay_stream \
#   VARIANT=v0-ads-ref-2x-statnorm-catflow-center-rel \
#   PRIOR_MODE=catflow_center_rel \
#   EPOCHS=100 \
#   bash scripts/launch_train_with_external_replay.sh
#
# Resumes from ${OUT}/last.ckpt automatically (train.py logic).
set -euo pipefail
source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
CODE="${PYTHONPATH%%:*}"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"

# --- required ---
OUT="${OUT:?OUT is required (training run directory)}"
STREAM_DIR="${STREAM_DIR:?STREAM_DIR is required (where replay daemon writes)}"
VARIANT="${VARIANT:?VARIANT is required}"

# --- optional ---
PRIOR_MODE="${PRIOR_MODE:-random_heuristic}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-48}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-5000}"
LOSS_TYPE="${LOSS_TYPE:-l1}"
SURF_W="${SURF_W:-1.0}"
ADS_W="${ADS_W:-1.0}"
ADS_PAIR_L1_W="${ADS_PAIR_L1_W:-0.0}"
ADS_CENTER_W="${ADS_CENTER_W:-1.0}"
ADS_REL_POS_W="${ADS_REL_POS_W:-1.0}"
PREDICTION_TYPE="${PREDICTION_TYPE:-x1}"
SAMPLE_EVAL_MAX="${SAMPLE_EVAL_MAX:-1000}"
SAMPLE_EVAL_STEPS="${SAMPLE_EVAL_STEPS:-20}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-1000}"

# Replay-specific knobs (external mode)
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-10}"          # archival cadence
RELOAD_EVERY_N_EPOCHS="${RELOAD_EVERY_N_EPOCHS:-1}"        # stream reload cadence
REPLAY_RATIO="${REPLAY_RATIO:-0.1}"                        # start conservative
REPLAY_MODE="${REPLAY_MODE:-append}"
REPLAY_PER_SYSTEM_CAP="${REPLAY_PER_SYSTEM_CAP:-10}"
REPLAY_CAP="${REPLAY_CAP:-1070000}"
REPLAY_WEIGHT_MODE="${REPLAY_WEIGHT_MODE:-improvement}"

GPUS="${GPUS:-0,1,2,3}"
N_DEVICES="${N_DEVICES:-4}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$(basename ${OUT})}"

mkdir -p "${OUT}" "${STREAM_DIR}"

echo "[train] OUT=${OUT}"
echo "[train] STREAM_DIR=${STREAM_DIR}"
echo "[train] VARIANT=${VARIANT}  PRIOR_MODE=${PRIOR_MODE}  EPOCHS=${EPOCHS}"
echo "[train] REPLAY_RATIO=${REPLAY_RATIO}  RELOAD_EVERY_N_EPOCHS=${RELOAD_EVERY_N_EPOCHS}"
echo "[train] SAVE_EVERY_N_EPOCHS=${SAVE_EVERY_N_EPOCHS}  (last.ckpt always updated per-epoch)"

env \
  CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH="${CODE}:${PYTHONPATH:-}" \
  CAT_BENCH_ROOT="${ROOT}" \
  ADSORBATES_PKL="${ADSORBATES_PKL:-${ROOT}/data/pkls/adsorbates.pkl}" \
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
  WANDB_MODE="${WANDB_MODE:-online}" \
setsid ${MICROMAMBA} run -n adsorbgen python -m adsorbgen.train \
  --arch v1 \
  --train-lmdb \
    "${ROOT}/data/processed/is2res_train.lmdb" \
    "${ROOT}/data/processed/is2res_val.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_ads.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_cat.lmdb" \
    "${ROOT}/data/processed/is2res_val_ood_both.lmdb" \
  --val-lmdb "${ROOT}/data/processed/oc20dense.lmdb" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" --devices "${N_DEVICES}" \
  --precision bf16-mixed --lr "${LR}" --lr-warmup-steps "${WARMUP_STEPS}" --grad-clip 10.0 \
  --loss-type "${LOSS_TYPE}" \
  --loss-surf-weight "${SURF_W}" --loss-ads-weight "${ADS_W}" \
  --ads-pair-l1-weight "${ADS_PAIR_L1_W}" --ads-bond-l1-weight 0.0 --ads-nonbonded-clash-weight 0.0 \
  --ads-center-loss-weight "${ADS_CENTER_W}" --ads-rel-pos-loss-weight "${ADS_REL_POS_W}" \
  --prediction-type "${PREDICTION_TYPE}" --prior-mode "${PRIOR_MODE}" \
  --sample-eval-every-epochs 1 --sample-eval-max-samples "${SAMPLE_EVAL_MAX}" \
  --sample-eval-steps "${SAMPLE_EVAL_STEPS}" \
  --max-val-samples "${MAX_VAL_SAMPLES}" --check-val-every-n-epoch 1 \
  --save-every-n-epochs "${SAVE_EVERY_N_EPOCHS}" \
  --external-replay-dir "${STREAM_DIR}" \
  --external-replay-reload-every-n-epochs "${RELOAD_EVERY_N_EPOCHS}" \
  --replay-ratio "${REPLAY_RATIO}" \
  --replay-mode "${REPLAY_MODE}" \
  --replay-per-system-cap "${REPLAY_PER_SYSTEM_CAP}" \
  --replay-cap "${REPLAY_CAP}" \
  --replay-weight-mode "${REPLAY_WEIGHT_MODE}" \
  --variant "${VARIANT}" \
  --out "${OUT}" \
  --wandb-project adsorbgen \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  > "${OUT}/train.log" 2>&1 < /dev/null &

echo "$!" > "${OUT}/pid.txt"
echo "[train] pid=$(cat ${OUT}/pid.txt)  log=${OUT}/train.log"
