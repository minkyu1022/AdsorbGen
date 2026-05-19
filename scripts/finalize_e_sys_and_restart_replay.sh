#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/irteam/AdsorbGen}
SHARD_DIR=${SHARD_DIR:-/home/irteam/data/replay/e_sys_shards}
OUT_DIR=${OUT_DIR:-/home/irteam/data/replay}
LOG_DIR=${LOG_DIR:-/home/irteam/data/replay/e_sys_logs}
FINALIZE_LOG=${FINALIZE_LOG:-/home/irteam/data/replay/e_sys_finalize.log}
NUM_SHARDS=${NUM_SHARDS:-8}

TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-/home/irteam/runs/H200_ads_pair_dist_loss}
STREAM_DIR=${STREAM_DIR:-/home/irteam/runs/H200_ads_pair_dist_loss/replay_stream}
TRAIN_LMDB=${TRAIN_LMDB:-"/home/irteam/data/processed/is2res_train.lmdb /home/irteam/data/processed/is2res_val.lmdb /home/irteam/data/processed/is2res_val_ood_ads.lmdb /home/irteam/data/processed/is2res_val_ood_cat.lmdb /home/irteam/data/processed/is2res_val_ood_both.lmdb"}

cd "$REPO"

{
  echo "[finalize] started $(date -Is)"
  echo "[finalize] waiting for compute_e_sys.py processes to finish..."
  while pgrep -f "scripts/compute_e_sys.py" >/dev/null; do
    ps -eo pid,etime,cmd | awk '/scripts\/compute_e_sys.py/ && !/awk/ {print "[finalize] still running:", $1, $2}'
    sleep 300
  done

  echo "[finalize] compute finished $(date -Is); merging shards"
  /home1/irteam/micromamba/envs/adsorbgen/bin/python scripts/merge_e_sys_and_rebuild_gt.py \
    --shard-dir "$SHARD_DIR" \
    --num-shards "$NUM_SHARDS" \
    --old-gt-index /home/irteam/data/replay/gt_index_by_sid.pkl \
    --out-dir "$OUT_DIR" \
    --require-all-shards

  echo "[finalize] building derived MLIP-relaxed training LMDBs"
  /home1/irteam/micromamba/envs/adsorbgen/bin/python scripts/build_mlip_relaxed_lmdbs.py \
    --e-sys "$OUT_DIR/E_sys.pkl" \
    --out-dir /home/irteam/data/processed_mlip_oc20 \
    --only-converged

  echo "[finalize] merge complete; restarting replay with oc20 gt index"
  rm -f "$STREAM_DIR/KILL_FLAG"
  GPUS="0 1 2 3 4 5 6 7" \
  TRAIN_RUN_DIR="$TRAIN_RUN_DIR" \
  STREAM_DIR="$STREAM_DIR" \
  GT_INDEX="$OUT_DIR/gt_index_by_sid_oc20.pkl" \
  TRAIN_LMDB="$TRAIN_LMDB" \
  PRIOR_MODE=random_heuristic \
  NUM_SYSTEMS_PER_SHARD=1250 \
  NUM_PLACEMENTS=10 \
  FLOW_STEPS=50 \
  FLOW_BATCH_SIZE=32 \
  UMA_MODEL=uma-s-1p1 \
  UMA_FMAX=0.05 \
  UMA_MAX_STEPS=300 \
  UMA_ATOM_BUDGET=4000 \
  SUCCESS_MARGIN=0.0 \
  SUCCESS_MARGIN_SCHEDULE_CYCLES=0 \
  OVERLAP_THRESHOLD=0.5 \
  VIZ_CAPTURE_N=0 \
  CHUNK_SIZE=64 \
  CKPT_SETTLE_SEC=5.0 \
  CKPT_STALE_WARN_MIN=100000 \
  CKPT_STALE_EXIT_MIN=100000 \
  MAX_CYCLES=0 \
  UTIL_POLL_SEC=1.0 \
  PRISTINE_SLABS=/home/irteam/results/pristine_slabs/is2res.pkl \
  PRISTINE_INDEX=/home/irteam/results/pristine_slabs/is2res.sid_index.pkl \
    bash scripts/run_replay_daemon_4gpu.sh

  echo "[finalize] replay restart launched $(date -Is)"
} >>"$FINALIZE_LOG" 2>&1
