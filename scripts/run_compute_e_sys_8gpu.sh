#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/irteam/AdsorbGen}
OUT_DIR=${OUT_DIR:-/home/irteam/data/replay/e_sys_shards}
LOG_DIR=${LOG_DIR:-/home/irteam/data/replay/e_sys_logs}
NUM_SHARDS=${NUM_SHARDS:-8}
ATOM_BUDGET=${ATOM_BUDGET:-4000}
UMA_MAX_STEPS=${UMA_MAX_STEPS:-200}
UMA_FMAX=${UMA_FMAX:-0.05}
PYTHON_BIN=${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}

LMDBS=(
  /home/irteam/data/processed/is2res_train.lmdb
  /home/irteam/data/processed/is2res_val.lmdb
  /home/irteam/data/processed/is2res_val_ood_ads.lmdb
  /home/irteam/data/processed/is2res_val_ood_cat.lmdb
  /home/irteam/data/processed/is2res_val_ood_both.lmdb
)

mkdir -p "$OUT_DIR" "$LOG_DIR"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log="$LOG_DIR/shard_${shard}.log"
  echo "[launch] shard $shard -> $log"
  (
    cd "$REPO"
    echo "[launch] $(date -Is) shard=$shard cuda=$shard python=$PYTHON_BIN"
    exec setsid -f env CUDA_VISIBLE_DEVICES="$shard" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" scripts/compute_e_sys.py \
        --lmdbs "${LMDBS[@]}" \
        --shard-idx "$shard" \
        --num-shards "$NUM_SHARDS" \
        --out-dir "$OUT_DIR" \
        --uma-model uma-s-1p1 \
        --uma-task oc20 \
        --uma-fmax "$UMA_FMAX" \
        --uma-max-steps "$UMA_MAX_STEPS" \
        --atom-budget "$ATOM_BUDGET" \
        --resume
  ) >"$log" 2>&1 &
  echo $! >"$LOG_DIR/shard_${shard}.pid"
  sleep "${LAUNCH_STAGGER_SEC:-2}"
done

echo "[launch] started $NUM_SHARDS shard processes"
echo "[launch] logs: $LOG_DIR"
