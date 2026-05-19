#!/usr/bin/env bash
# Wait for reconverge/finalize outputs, then upload only required artifacts.
set -euo pipefail

OUT_DIR=${OUT_DIR:-/home/irteam/data/replay}
RUN_DIR=${RUN_DIR:-/home/irteam/runs/H200_ads_pair_dist_loss}
DEST=${DEST:-dropbox:AdsorbGen/H200_ads_pair_dist_loss_required_after_reconverge}
LOG=${LOG:-${OUT_DIR}/dropbox_required_upload.log}
MIN_MTIME_EPOCH=${MIN_MTIME_EPOCH:-$(date +%s)}
SLEEP_SEC=${SLEEP_SEC:-300}

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[upload] start $(date -Is)"
echo "[upload] destination: $DEST"
echo "[upload] waiting for files newer than epoch: $MIN_MTIME_EPOCH ($(date -d "@${MIN_MTIME_EPOCH}" -Is))"

if ! command -v rclone >/dev/null 2>&1; then
  echo "[upload] ERROR: rclone not found"
  exit 1
fi

if ! rclone listremotes | grep -qx 'dropbox:'; then
  echo "[upload] ERROR: rclone remote dropbox: not configured"
  exit 1
fi

required_files=(
  "${OUT_DIR}/E_gas_only.pkl"
  "${OUT_DIR}/E_slab_only.pkl"
  "${OUT_DIR}/E_sys.pkl"
  "${OUT_DIR}/gt_index_by_sid_oc20.pkl"
  "${OUT_DIR}/gt_index_by_system_oc20.pkl"
)

while pgrep -f "scripts/reconverge_e_sys.py" >/dev/null; do
  echo "[upload] reconverge still running $(date -Is)"
  ps -eo pid,etime,cmd | grep '[r]econverge_e_sys.py' || true
  sleep "$SLEEP_SEC"
done

echo "[upload] reconverge processes finished $(date -Is); waiting for merged refs"
while true; do
  missing=0
  stale=0
  for f in "${required_files[@]}"; do
    if [[ ! -s "$f" ]]; then
      echo "[upload] waiting: missing $f"
      missing=1
      continue
    fi
  done

  for f in "${OUT_DIR}/E_sys.pkl" "${OUT_DIR}/gt_index_by_sid_oc20.pkl" "${OUT_DIR}/gt_index_by_system_oc20.pkl"; do
    mtime=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if (( mtime < MIN_MTIME_EPOCH )); then
      echo "[upload] waiting: $f is not rebuilt yet (mtime $(date -d "@${mtime}" -Is))"
      stale=1
    fi
  done

  if (( missing == 0 && stale == 0 )); then
    break
  fi
  sleep "$SLEEP_SEC"
done

if pgrep -f "scripts/finalize_reconverge_replay5000.sh" >/dev/null; then
  echo "[upload] merged refs are ready; finalize/replay may still be running. Will upload refs/logs now and run results currently present."
fi

echo "[upload] uploading required replay reference files"
rclone copy "$OUT_DIR" "${DEST}/replay_refs" \
  --progress \
  --include "E_gas_only.pkl" \
  --include "E_slab_only.pkl" \
  --include "E_sys.pkl" \
  --include "gt_index_by_sid_oc20.pkl" \
  --include "gt_index_by_system_oc20.pkl" \
  --include "E_sys_step_stats.json" \
  --include "e_sys_finalize.log" \
  --include "finalize_reconverge_replay5000_*.log" \
  --include "dropbox_required_upload.log" \
  --include "e_sys_logs/**" \
  --include "reconverge_logs/**" \
  --exclude "*"

echo "[upload] uploading required run outputs"
rclone copy "$RUN_DIR" "${DEST}/run_H200_ads_pair_dist_loss" \
  --progress \
  --include "args.json" \
  --include "train.log" \
  --include "overlap_diag_300.json" \
  --include "replay_stream/logs/**" \
  --include "replay_stream/shard_*/**" \
  --include "replay_stream_oc20_mean_5000x10_*/logs/**" \
  --include "replay_stream_oc20_mean_5000x10_*/shard_*/**" \
  --include "replay_stream_oc20_mean_5000x10_*/cycle_000000_report.json" \
  --include "success_trajectories/**" \
  --exclude "*"

echo "[upload] done $(date -Is)"
