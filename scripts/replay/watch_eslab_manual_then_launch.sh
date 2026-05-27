#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/irteam}"
REPO="${REPO:-${ROOT}/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
REPLAY_1P2="${REPLAY_1P2:-${ROOT}/data/replay_uma_s_1p2}"
NUM_SHARDS="${NUM_SHARDS:-24}"
POLL_SEC="${POLL_SEC:-60}"
LOG="${LOG:-${REPLAY_1P2}/watch_eslab_manual_then_launch.log}"
GPU_LIST_CSV="${GPU_LIST_CSV:-0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

IFS=',' read -r -a GPU_LIST <<< "${GPU_LIST_CSV}"

echo "[watch-eslab] started $(date -Is)"
echo "[watch-eslab] replay=${REPLAY_1P2} poll=${POLL_SEC}s"

launch_shard() {
  local shard="$1"
  local gpu="${GPU_LIST[$((shard % ${#GPU_LIST[@]}))]}"
  local log="${REPLAY_1P2}/lbfgs_recompute_logs/e_slab_lbfgs_shard${shard}.manual.log"
  local pidfile="${REPLAY_1P2}/lbfgs_recompute_logs/e_slab_lbfgs_shard${shard}.manual.pid"
  echo "[watch-eslab] launching shard=${shard} gpu=${gpu} $(date -Is)"
  (
    cd "${REPO}"
    CUDA_VISIBLE_DEVICES="${gpu}" setsid "${PY}" scripts/replay/e_sys/compute_e_slab_lbfgs.py \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --out-dir "${REPLAY_1P2}/e_slab_only_lbfgs_shards" \
      --uma-model uma-s-1p2 \
      --uma-task oc20 \
      --uma-fmax 0.05 \
      --uma-max-steps 300 \
      --resume \
      --pristine-slabs "${ROOT}/results/pristine_slabs/is2res.pkl" \
      > "${log}" 2>&1 &
    echo "$!" > "${pidfile}"
  )
}

done_count() {
  local n=0
  local shard
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local log="${REPLAY_1P2}/lbfgs_recompute_logs/e_slab_lbfgs_shard${shard}.manual.log"
    if grep -q "DONE .* slabs in" "${log}" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "${n}"
}

while true; do
  echo
  echo "[watch-eslab] tick $(date -Is)"
  alive="$(pgrep -fc 'compute_e_slab_lbfgs.py' || true)"
  done="$(done_count)"
  echo "[watch-eslab] alive=${alive} done=${done}/${NUM_SHARDS}"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true

  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    log="${REPLAY_1P2}/lbfgs_recompute_logs/e_slab_lbfgs_shard${shard}.manual.log"
    pidfile="${REPLAY_1P2}/lbfgs_recompute_logs/e_slab_lbfgs_shard${shard}.manual.pid"
    if grep -q "DONE .* slabs in" "${log}" 2>/dev/null; then
      continue
    fi
    pid=""
    [[ -s "${pidfile}" ]] && pid="$(cat "${pidfile}")"
    if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
      echo "[watch-eslab] shard=${shard} not done and not alive; relaunching"
      launch_shard "${shard}"
      sleep 2
    fi
  done

  done="$(done_count)"
  if [[ "${done}" -eq "${NUM_SHARDS}" ]]; then
    echo "[watch-eslab] all E_slab shards DONE; merging $(date -Is)"
    (
      cd "${REPO}"
      "${PY}" scripts/replay/e_sys/merge_e_slab_lbfgs.py \
        --shard-dir "${REPLAY_1P2}/e_slab_only_lbfgs_shards" \
        --num-shards "${NUM_SHARDS}" \
        --sid-index "${ROOT}/results/pristine_slabs/is2res.sid_index.pkl" \
        --out-dir "${REPLAY_1P2}" \
        --uma-model uma-s-1p2 \
        --uma-task oc20 \
        --relaxed-pristine-out "${REPLAY_1P2}/pristine_slabs_lbfgs.pkl" \
        --require-all-shards
    ) > "${REPLAY_1P2}/lbfgs_recompute_logs/merge_e_slab_lbfgs.manual.log" 2>&1
    echo "[watch-eslab] merge done $(date -Is)"
    echo "[watch-eslab] launching post-reference jobs"
    (
      cd "${REPO}"
      env REPLAY_1P2="${REPLAY_1P2}" POLL_SEC=30 bash scripts/replay/wait_1p2_refs_then_launch_full_and_sde250.sh
    ) > "${REPLAY_1P2}/post_reference_watcher.manual_after_eslab.log" 2>&1 &
    echo "$!" > "${REPLAY_1P2}/post_reference_watcher.manual_after_eslab.pid"
    exit 0
  fi

  sleep "${POLL_SEC}"
done
