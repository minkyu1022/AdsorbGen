#!/usr/bin/env bash
# Wait for reconverge_e_sys.py, then rebuild OC20 refs. No replay is launched.
set -euo pipefail

REPO=${REPO:-/home/irteam/AdsorbGen}
PYTHON_BIN=${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}

OUT_DIR=${OUT_DIR:-/home/irteam/data/replay}
SHARDS_DIR=${SHARDS_DIR:-${OUT_DIR}/e_sys_shards}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
BUILD_MLIP_LMDBS=${BUILD_MLIP_LMDBS:-0}
MLIP_LMDB_OUT_DIR=${MLIP_LMDB_OUT_DIR:-/home/irteam/data/processed_mlip_oc20}

cd "$REPO"
mkdir -p "$OUT_DIR"
exec > >(tee -a "${OUT_DIR}/finalize_reconverge_refs_${STAMP}.log") 2>&1

echo "[finalize_refs] start $(date -Is)"
echo "[finalize_refs] waiting for reconverge_e_sys.py processes"
while pgrep -f "scripts/reconverge_e_sys.py" >/dev/null; do
  ps -eo pid,etime,cmd | grep '[r]econverge_e_sys.py' || true
  sleep 300
done

echo "[finalize_refs] reconverge finished $(date -Is); rebuilding merged E_sys/GT indexes"
"$PYTHON_BIN" scripts/merge_e_sys_and_rebuild_gt.py \
  --shard-dir "$SHARDS_DIR" \
  --out-dir "$OUT_DIR" \
  --old-gt-index "$OUT_DIR/gt_index_by_sid.pkl"

echo "[finalize_refs] writing E_sys convergence-step statistics"
"$PYTHON_BIN" scripts/report_e_sys_steps.py \
  --e-sys "$OUT_DIR/E_sys.pkl" \
  --out "$OUT_DIR/E_sys_step_stats.json"

if [[ "$BUILD_MLIP_LMDBS" == "1" ]]; then
  echo "[finalize_refs] building optional MLIP-relaxed training LMDBs -> $MLIP_LMDB_OUT_DIR"
  "$PYTHON_BIN" scripts/build_mlip_relaxed_lmdbs.py \
    --e-sys "$OUT_DIR/E_sys.pkl" \
    --out-dir "$MLIP_LMDB_OUT_DIR" \
    --only-converged
else
  echo "[finalize_refs] skipping optional MLIP-relaxed LMDB build (BUILD_MLIP_LMDBS=$BUILD_MLIP_LMDBS)"
fi

echo "[finalize_refs] done $(date -Is)"
