#!/usr/bin/env bash
# Relax all OC20-Dense configs with UMA-s-1p1/task=oc20 on GPUs 4-7.
set -euo pipefail

REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
LMDB="${LMDB:-/home/irteam/data/processed/oc20dense.lmdb}"
OUT_DIR="${OUT_DIR:-/home/irteam/data/replay/oc20dense_mlip_relax_shards}"
LOG_DIR="${LOG_DIR:-/home/irteam/data/replay/oc20dense_mlip_relax_logs}"
NUM_SHARDS="${NUM_SHARDS:-4}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
UMA_MODEL="${UMA_MODEL:-uma-s-1p1}"
UMA_TASK="${UMA_TASK:-oc20}"
UMA_FMAX="${UMA_FMAX:-0.05}"
UMA_MAX_STEPS="${UMA_MAX_STEPS:-300}"
ATOM_BUDGET="${ATOM_BUDGET:-4000}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"

echo "[launch] lmdb=${LMDB}"
echo "[launch] out=${OUT_DIR}"
echo "[launch] logs=${LOG_DIR}"
echo "[launch] UMA=${UMA_MODEL} task=${UMA_TASK} fmax=${UMA_FMAX} max_steps=${UMA_MAX_STEPS}"

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  cuda="${GPUS[$shard]}"
  log="${LOG_DIR}/shard_${shard}.log"
  echo "[launch] shard ${shard}/${NUM_SHARDS} -> GPU ${cuda} log=${log}"
  (
    cd "${REPO}"
    exec setsid -f env CUDA_VISIBLE_DEVICES="${cuda}" PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" experiments/2026-05-oc20dense/compute_oc20dense_mlip_relax.py \
        --lmdb "${LMDB}" \
        --out-dir "${OUT_DIR}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --uma-model "${UMA_MODEL}" \
        --uma-task "${UMA_TASK}" \
        --uma-fmax "${UMA_FMAX}" \
        --uma-max-steps "${UMA_MAX_STEPS}" \
        --atom-budget "${ATOM_BUDGET}" \
        --resume
  ) > "${log}" 2>&1 &
  echo "$!" > "${LOG_DIR}/shard_${shard}.pid"
  sleep "${LAUNCH_STAGGER_SEC:-2}"
done

cat > "${LOG_DIR}/merge_when_done.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
while pgrep -f "compute_oc20dense_mlip_relax.py --lmdb ${LMDB} --out-dir ${OUT_DIR}" >/dev/null; do
  sleep 300
done
cd "${REPO}"
"${PYTHON_BIN}" experiments/2026-05-oc20dense/merge_oc20dense_mlip_relax.py \
  --shards-dir "${OUT_DIR}" \
  --out-dir /home/irteam/data/replay \
  --num-shards "${NUM_SHARDS}" \
  > "${LOG_DIR}/merge.log" 2>&1
"${PYTHON_BIN}" experiments/2026-05-oc20dense/materialize_oc20dense_slab_refs.py \
  --out-dir /home/irteam/data/replay \
  > "${LOG_DIR}/materialize_slab_refs.log" 2>&1
EOF
chmod +x "${LOG_DIR}/merge_when_done.sh"
setsid -f bash "${LOG_DIR}/merge_when_done.sh" >/dev/null 2>&1 < /dev/null

echo "[launch] started ${NUM_SHARDS} shards and merge watcher"
