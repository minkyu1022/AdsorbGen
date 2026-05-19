#!/usr/bin/env bash
# Wait for reconverge_e_sys.py, rebuild refs, run one 5000x10 replay, report.
set -euo pipefail

REPO=${REPO:-/home/irteam/AdsorbGen}
ROOT=${CAT_BENCH_ROOT:-/home/irteam}
PYTHON_BIN=${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}

OUT_DIR=${OUT_DIR:-/home/irteam/data/replay}
SHARDS_DIR=${SHARDS_DIR:-${OUT_DIR}/e_sys_shards}
RECONV_LOG_DIR=${RECONV_LOG_DIR:-${OUT_DIR}/reconverge_logs}
TRAIN_RUN_DIR=${TRAIN_RUN_DIR:-/home/irteam/runs/H200_ads_pair_dist_loss}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
STREAM_DIR=${STREAM_DIR:-${TRAIN_RUN_DIR}/replay_stream_oc20_mean_5000x10_${STAMP}}

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS=${#GPUS[@]}
TOTAL_SYSTEMS=${TOTAL_SYSTEMS:-5000}
NUM_SYSTEMS_PER_SHARD=$(( (TOTAL_SYSTEMS + NUM_SHARDS - 1) / NUM_SHARDS ))
NUM_PLACEMENTS=${NUM_PLACEMENTS:-10}
FLOW_STEPS=${FLOW_STEPS:-50}
FLOW_BATCH_SIZE=${FLOW_BATCH_SIZE:-32}
PRIOR_MODE=${PRIOR_MODE:-random_heuristic}
UMA_MODEL=${UMA_MODEL:-uma-s-1p1}
UMA_TASK=${UMA_TASK:-oc20}
UMA_FMAX=${UMA_FMAX:-0.05}
UMA_MAX_STEPS=${UMA_MAX_STEPS:-300}
UMA_ATOM_BUDGET=${UMA_ATOM_BUDGET:-4000}
SUCCESS_MARGIN=${SUCCESS_MARGIN:-0.0}
OVERLAP_THRESHOLD=${OVERLAP_THRESHOLD:-0.5}
CHUNK_SIZE=${CHUNK_SIZE:-64}

LMDBS=(
  /home/irteam/data/processed/is2res_train.lmdb
  /home/irteam/data/processed/is2res_val.lmdb
  /home/irteam/data/processed/is2res_val_ood_ads.lmdb
  /home/irteam/data/processed/is2res_val_ood_cat.lmdb
  /home/irteam/data/processed/is2res_val_ood_both.lmdb
)

cd "$REPO"
mkdir -p "$OUT_DIR" "$STREAM_DIR/logs" "$STREAM_DIR/pids"
exec > >(tee -a "${OUT_DIR}/finalize_reconverge_replay5000_${STAMP}.log") 2>&1

echo "[finalize5000] start $(date -Is)"
echo "[finalize5000] waiting for reconverge_e_sys.py processes"
while pgrep -f "scripts/reconverge_e_sys.py" >/dev/null; do
  ps -eo pid,etime,cmd | grep '[r]econverge_e_sys.py' || true
  sleep 300
done

echo "[finalize5000] reconverge finished $(date -Is); rebuilding merged E_sys/GT indexes"
"$PYTHON_BIN" scripts/merge_e_sys_and_rebuild_gt.py \
  --shards-dir "$SHARDS_DIR" \
  --out-dir "$OUT_DIR" \
  --old-gt-index "$OUT_DIR/gt_index_by_sid.pkl"

echo "[finalize5000] writing E_sys convergence-step statistics"
"$PYTHON_BIN" scripts/report_e_sys_steps.py \
  --e-sys "$OUT_DIR/E_sys.pkl" \
  --out "$OUT_DIR/E_sys_step_stats.json"

echo "[finalize5000] building MLIP-relaxed training LMDBs"
"$PYTHON_BIN" scripts/build_mlip_relaxed_lmdbs.py \
  --e-sys "$OUT_DIR/E_sys.pkl" \
  --out-dir /home/irteam/data/processed_mlip_oc20 \
  --only-converged

echo "[finalize5000] launching one replay cycle: ${TOTAL_SYSTEMS} systems x ${NUM_PLACEMENTS} placements"
rm -f "${STREAM_DIR}/KILL_FLAG"
for shard_idx in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_idx]}"
  log_file="${STREAM_DIR}/logs/daemon_shard${shard_idx}.log"
  pid_file="${STREAM_DIR}/pids/pid_shard${shard_idx}.txt"
  echo "[finalize5000] shard ${shard_idx}/${NUM_SHARDS} gpu=${gpu} -> ${log_file}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${REPO}:${PYTHONPATH:-}" \
    CAT_BENCH_ROOT="${ROOT}" \
    ADSORBATES_PKL="${ADSORBATES_PKL:-${ROOT}/data/pkls/adsorbates.pkl}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    "$PYTHON_BIN" scripts/replay_daemon.py \
      --train-run-dir "$TRAIN_RUN_DIR" \
      --stream-dir "$STREAM_DIR" \
      --shard-idx "$shard_idx" \
      --num-shards "$NUM_SHARDS" \
      --gt-index "$OUT_DIR/gt_index_by_sid_oc20.pkl" \
      --train-lmdb "${LMDBS[@]}" \
      --num-systems "$NUM_SYSTEMS_PER_SHARD" \
      --num-placements "$NUM_PLACEMENTS" \
      --flow-steps "$FLOW_STEPS" \
      --flow-batch-size "$FLOW_BATCH_SIZE" \
      --prior-mode "$PRIOR_MODE" \
      --uma-model "$UMA_MODEL" \
      --uma-fmax "$UMA_FMAX" \
      --uma-max-steps "$UMA_MAX_STEPS" \
      --uma-atom-budget "$UMA_ATOM_BUDGET" \
      --success-margin "$SUCCESS_MARGIN" \
      --success-margin-schedule-cycles 0 \
      --overlap-threshold "$OVERLAP_THRESHOLD" \
      --viz-capture-n 0 \
      --chunk-size "$CHUNK_SIZE" \
      --ckpt-settle-sec 5 \
      --ckpt-stale-warn-min 60 \
      --ckpt-stale-exit-min 180 \
      --max-cycles 1 \
      --util-poll-sec 1 \
      --pristine-slabs /home/irteam/results/pristine_slabs/is2res.pkl \
      --pristine-index /home/irteam/results/pristine_slabs/is2res.sid_index.pkl \
      > "$log_file" 2>&1 &
  echo "$!" > "$pid_file"
  sleep 2
done

echo "[finalize5000] waiting for replay daemons"
status=0
for pid_file in "${STREAM_DIR}"/pids/pid_shard*.txt; do
  pid=$(cat "$pid_file")
  if ! wait "$pid"; then
    echo "[finalize5000] WARN: pid $pid failed"
    status=1
  fi
done

echo "[finalize5000] replay daemons finished $(date -Is); writing report"
"$PYTHON_BIN" scripts/report_replay_cycle.py \
  --stream-dir "$STREAM_DIR" \
  --cycle 0 \
  --out "$STREAM_DIR/cycle_000000_report.json"

UPLOAD_STAGE="${OUT_DIR}/dropbox_upload_adsorbgen_replay_${STAMP}"
mkdir -p "$UPLOAD_STAGE"
cp "$OUT_DIR/E_sys.pkl" \
   "$OUT_DIR/gt_index_by_sid_oc20.pkl" \
   "$OUT_DIR/gt_index_by_system_oc20.pkl" \
   "$OUT_DIR/E_sys_step_stats.json" \
   "$STREAM_DIR/cycle_000000_report.json" \
   "$UPLOAD_STAGE"/
cp "${OUT_DIR}/finalize_reconverge_replay5000_${STAMP}.log" "$UPLOAD_STAGE"/
cat > "$UPLOAD_STAGE/README_UPLOAD.txt" <<EOF
Prepared outputs for Dropbox upload.

Target Dropbox folder:
김민규/SPML/research/26 AdsorbGen (Minkyu Kim)

This machine currently needs a configured Dropbox CLI/rclone remote or a mounted
Dropbox folder before Codex can upload directly.
EOF

if command -v rclone >/dev/null 2>&1 && rclone listremotes | grep -qx 'dropbox:'; then
  echo "[finalize5000] uploading with rclone remote dropbox:"
  rclone copy "$UPLOAD_STAGE" "dropbox:김민규/SPML/research/26 AdsorbGen (Minkyu Kim)/$(basename "$UPLOAD_STAGE")" --progress
else
  echo "[finalize5000] Dropbox upload staged at $UPLOAD_STAGE"
  echo "[finalize5000] No configured rclone remote named dropbox: was found."
fi

echo "[finalize5000] done $(date -Is) status=$status"
exit "$status"
