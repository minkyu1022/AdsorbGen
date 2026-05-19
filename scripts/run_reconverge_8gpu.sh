#!/usr/bin/env bash
# Re-relax the non-converged E_sys entries across 8 GPUs (one shard / GPU).
# Continues each non-converged system from its stored partially-relaxed
# geometry with a larger FIRE step cap. Updates e_sys_shard{N}.pkl in place.
set -euo pipefail

REPO=${REPO:-/home/irteam/AdsorbGen}
SHARDS_DIR=${SHARDS_DIR:-/home/irteam/data/replay/e_sys_shards}
LOG_DIR=${LOG_DIR:-/home/irteam/data/replay/reconverge_logs}
NUM_SHARDS=${NUM_SHARDS:-8}
ATOM_BUDGET=${ATOM_BUDGET:-4000}
UMA_MAX_STEPS=${UMA_MAX_STEPS:-300}
UMA_FMAX=${UMA_FMAX:-0.05}
PYTHON_BIN=${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}

LMDBS=(
  /home/irteam/data/processed/is2res_train.lmdb
  /home/irteam/data/processed/is2res_val.lmdb
  /home/irteam/data/processed/is2res_val_ood_ads.lmdb
  /home/irteam/data/processed/is2res_val_ood_cat.lmdb
  /home/irteam/data/processed/is2res_val_ood_both.lmdb
)

mkdir -p "$LOG_DIR"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  log="$LOG_DIR/shard_${shard}.log"
  echo "[launch] shard $shard -> $log"
  (
    cd "$REPO"
    echo "[launch] $(date -Is) shard=$shard cuda=$shard python=$PYTHON_BIN"
    exec setsid -f env CUDA_VISIBLE_DEVICES="$shard" PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" scripts/reconverge_e_sys.py \
        --lmdbs "${LMDBS[@]}" \
        --shard-idx "$shard" \
        --num-shards "$NUM_SHARDS" \
        --shards-dir "$SHARDS_DIR" \
        --uma-model uma-s-1p1 \
        --uma-task oc20 \
        --uma-fmax "$UMA_FMAX" \
        --uma-max-steps "$UMA_MAX_STEPS" \
        --atom-budget "$ATOM_BUDGET"
  ) >"$log" 2>&1 &
  echo $! >"$LOG_DIR/shard_${shard}.pid"
  sleep "${LAUNCH_STAGGER_SEC:-2}"
done

echo "[launch] started $NUM_SHARDS reconverge shard processes"
echo "[launch] logs: $LOG_DIR"
