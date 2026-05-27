#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/irteam}"
REPO="${REPO:-${ROOT}/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
REPLAY_DIR="${REPLAY_DIR:-${ROOT}/data/replay}"
NUM_SHARDS="${NUM_SHARDS:-24}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7}"

E_SYS_SHARDS="${E_SYS_SHARDS:-${REPLAY_DIR}/e_sys_lbfgs_shards}"
E_SLAB_SHARDS="${E_SLAB_SHARDS:-${REPLAY_DIR}/e_slab_only_lbfgs_shards}"
LOG_DIR="${LOG_DIR:-${REPLAY_DIR}/lbfgs_recompute_logs}"

mkdir -p "${E_SYS_SHARDS}" "${E_SLAB_SHARDS}" "${LOG_DIR}"
cd "${REPO}"

echo "[lbfgs] stage 1/4: E_sys L-BFGS shards"
read -r -a GPUS <<< "${GPU_LIST}"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" scripts/replay/e_sys/compute_e_sys_lbfgs.py \
      --lmdbs "${ROOT}/data/processed_ID/is2res_train.lmdb" "${ROOT}/data/processed_ID/is2res_val.lmdb" \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --out-dir "${E_SYS_SHARDS}" \
      --uma-model uma-s-1p1 \
      --uma-task oc20 \
      --uma-fmax 0.05 \
      --uma-max-steps 300 \
      --resume \
      > "${LOG_DIR}/e_sys_lbfgs_shard${shard}.log" 2>&1
  ) &
  echo $! > "${LOG_DIR}/e_sys_lbfgs_shard${shard}.pid"
  sleep 5
done
wait

echo "[lbfgs] stage 2/4: merge E_sys and rebuild gt_index"
"${PY}" scripts/replay/e_sys/merge_e_sys_lbfgs_and_rebuild_gt.py \
  --shard-dir "${E_SYS_SHARDS}" \
  --num-shards "${NUM_SHARDS}" \
  --old-gt-index "${REPLAY_DIR}/gt_index_by_sid.pkl" \
  --out-dir "${REPLAY_DIR}" \
  --require-all-shards \
  > "${LOG_DIR}/merge_e_sys_lbfgs.log" 2>&1

echo "[lbfgs] stage 3/4: E_slab_only L-BFGS shards"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PY}" scripts/replay/e_sys/compute_e_slab_lbfgs.py \
      --pristine-slabs "${ROOT}/results/pristine_slabs/is2res.pkl" \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --out-dir "${E_SLAB_SHARDS}" \
      --uma-model uma-s-1p1 \
      --uma-task oc20 \
      --uma-fmax 0.05 \
      --uma-max-steps 300 \
      --resume \
      > "${LOG_DIR}/e_slab_lbfgs_shard${shard}.log" 2>&1
  ) &
  echo $! > "${LOG_DIR}/e_slab_lbfgs_shard${shard}.pid"
  sleep 5
done
wait

echo "[lbfgs] stage 4/4: merge E_slab_only"
"${PY}" scripts/replay/e_sys/merge_e_slab_lbfgs.py \
  --shard-dir "${E_SLAB_SHARDS}" \
  --num-shards "${NUM_SHARDS}" \
  --sid-index "${ROOT}/results/pristine_slabs/is2res.sid_index.pkl" \
  --out-dir "${REPLAY_DIR}" \
  --require-all-shards \
  > "${LOG_DIR}/merge_e_slab_lbfgs.log" 2>&1

echo "[lbfgs] DONE"
