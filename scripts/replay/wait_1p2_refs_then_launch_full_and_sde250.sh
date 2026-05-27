#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/irteam}"
REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
REPLAY_1P2="${REPLAY_1P2:-${ROOT}/data/replay_uma_s_1p2}"
POLL_SEC="${POLL_SEC:-300}"
LOG="${LOG:-${REPLAY_1P2}/post_reference_launcher.log}"

FULL_OUT="${FULL_OUT:-${ROOT}/runs/self_improvement/self_improve_lbfgs_ID_mlip_pairdist_1x_ep149_full_x10_1p2ref_6gpu_24shards_20260526}"
CKPT="${CKPT:-${ROOT}/runs/training/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544/ckpt_epochepoch=149.ckpt}"
GT_INDEX="${GT_INDEX:-${REPLAY_1P2}/gt_index_by_sid_oc20_lbfgs.pkl}"
PRISTINE_SLABS="${PRISTINE_SLABS:-${REPLAY_1P2}/pristine_slabs_lbfgs.pkl}"
PRISTINE_INDEX="${PRISTINE_INDEX:-${ROOT}/results/pristine_slabs/is2res.sid_index.pkl}"

SDE_ROOT="${SDE_ROOT:-${ROOT}/runs/self_improvement/sde250_H200_catflow_center_rel_oc20dense_20260526}"
SDE_TRAIN_RUN_DIR="${SDE_TRAIN_RUN_DIR:-${ROOT}/runs/training/H200_catflow_center_rel}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

echo "[post-1p2] started $(date -Is)"
echo "[post-1p2] waiting for ${REPLAY_1P2}"

required=(
  "${REPLAY_1P2}/E_sys_lbfgs_summary.json"
  "${REPLAY_1P2}/E_slab_only_lbfgs_summary.json"
  "${REPLAY_1P2}/gt_index_by_sid_oc20_lbfgs.pkl"
  "${REPLAY_1P2}/gt_index_by_system_oc20_lbfgs.pkl"
  "${REPLAY_1P2}/E_sys_lbfgs.pkl"
  "${REPLAY_1P2}/E_slab_only_lbfgs.pkl"
  "${REPLAY_1P2}/E_slab_only_lbfgs_by_slab.pkl"
  "${REPLAY_1P2}/pristine_slabs_lbfgs.pkl"
)

while true; do
  missing=0
  for f in "${required[@]}"; do
    [[ -s "${f}" ]] || missing=$((missing + 1))
  done
  if [[ "${missing}" -eq 0 ]]; then
    echo "[post-1p2] references ready $(date -Is)"
    break
  fi
  if ! pgrep -f "run_lbfgs_reference_recompute_8gpu.py" >/dev/null; then
    echo "[post-1p2] ERROR: reference supervisor is not running and ${missing} required files are missing"
    exit 2
  fi
  echo "[post-1p2] waiting: missing=${missing}/${#required[@]} $(date -Is)"
  sleep "${POLL_SEC}"
done

NUM_SYSTEMS="$("${PYTHON_BIN}" - <<'PY'
import pickle
import os
p = os.path.join(os.environ.get("REPLAY_1P2", "/home/irteam/data/replay_uma_s_1p2"), "gt_index_by_system_oc20_lbfgs.pkl")
with open(p, "rb") as f:
    db = pickle.load(f)
print(len(db))
PY
)"
echo "[post-1p2] eligible unique systems=${NUM_SYSTEMS}"
mkdir -p "${FULL_OUT}/logs"

if pgrep -f "self_improve_lbfgs_worker.py.*${FULL_OUT}" >/dev/null; then
  echo "[post-1p2] full self-improvement already running for ${FULL_OUT}"
else
  echo "[post-1p2] launching full 1p2 self-improvement on GPU0-5"
  (
    cd "${REPO}"
    env \
      REPO="${REPO}" \
      PYTHON_BIN="${PYTHON_BIN}" \
      CKPT="${CKPT}" \
      OUT_DIR="${FULL_OUT}" \
      GT_INDEX="${GT_INDEX}" \
      NUM_SYSTEMS="${NUM_SYSTEMS}" \
      NUM_PLACEMENTS=10 \
      NUM_SHARDS=24 \
      GPU_LIST=0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3,4,5 \
      FLOW_STEPS=50 \
      FLOW_BATCH_SIZE=32 \
      UMA_MODEL=uma-s-1p2 \
      UMA_TASK=oc20 \
      UMA_FMAX=0.05 \
      UMA_MAX_STEPS=300 \
      PRISTINE_SLABS="${PRISTINE_SLABS}" \
      PRISTINE_INDEX="${PRISTINE_INDEX}" \
      SEED=20260526 \
      SAVE_WINDOW_CANDIDATES=1 \
      CANDIDATE_WINDOW_EV=0.1 \
      LAUNCH_STAGGER_SEC=4 \
      bash scripts/replay/launch_self_improve_lbfgs_10k_x10_gpu0_3.sh \
      > "${FULL_OUT}/logs/launcher_1p2.nohup.log" 2>&1 &
    echo "$!" > "${FULL_OUT}/logs/launcher_1p2.pid"
  )
fi

mkdir -p "${SDE_ROOT}/logs"
echo "[post-1p2] launching SDE 250-step replay+UMA relaxation on GPU6-7"
echo "[post-1p2] SDE report command after completion:"
echo "  python ${REPO}/scripts/report_replay_5000x10.py --stream-dir ${SDE_ROOT}"

if [[ ! -s "${SDE_ROOT}/logs/cycle_000000_shard0.json" ]]; then
  (
    cd "${REPO}"
    env \
      TRAIN_RUN_DIR="${SDE_TRAIN_RUN_DIR}" \
      STREAM_DIR="${SDE_ROOT}" \
      GT_INDEX="${GT_INDEX}" \
      PRISTINE_PKL="${PRISTINE_SLABS}" \
      PRISTINE_IDX="${PRISTINE_INDEX}" \
      NUM_SHARDS=2 \
      GPU_LIST=6,7 \
      NUM_SYSTEMS_PER_SHARD="${SDE_NUM_SYSTEMS_PER_SHARD:-50}" \
      NUM_PLACEMENTS="${SDE_NUM_PLACEMENTS:-10}" \
      FLOW_STEPS=250 \
      FLOW_BATCH_SIZE=16 \
      UMA_MODEL=uma-s-1p2 \
      UMA_FMAX=0.05 \
      UMA_MAX_STEPS=300 \
      UMA_ATOM_BUDGET=4000 \
      USE_SDE=1 \
      REFINE_FINAL=0 \
      SYSTEM_SEED=20260526 \
      LAUNCH_STAGGER_SEC=2 \
      bash scripts/replay/run_replay_5000x10_8gpu.sh \
      > "${SDE_ROOT}/logs/launcher.log" 2>&1 &
    echo "$!" > "${SDE_ROOT}/logs/launcher.pid"
  )
fi

echo "[post-1p2] launched follow-up jobs $(date -Is)"
