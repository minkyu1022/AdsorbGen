#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
OUT_DIR="${OUT_DIR:-/home/irteam/data/replay/mlip_pass_lbfgs_ood50}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
WAIT_FOR_DENSE="${WAIT_FOR_DENSE:-1}"
DENSE_SUMMARY="${DENSE_SUMMARY:-/home/irteam/data/replay/oc20dense_mlip_relax_summary.json}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

if [[ "${WAIT_FOR_DENSE}" == "1" ]]; then
  echo "[launcher] waiting for OC20-Dense relaxation to finish: ${DENSE_SUMMARY}" | tee -a "${LOG_DIR}/launcher.log"
  while true; do
    if [[ -f "${DENSE_SUMMARY}" ]] && ! pgrep -f "compute_oc20dense_mlip_relax.py --lmdb /home/irteam/data/processed/oc20dense.lmdb" >/dev/null; then
      break
    fi
    sleep 300
  done
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi

echo "[launcher] starting MLIP Pass OOD-50 L-BFGS eval at $(date)" | tee -a "${LOG_DIR}/launcher.log"
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON_BIN}" experiments/2026-05-mlip-pass/eval_mlip_pass_lbfgs_ood50.py \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --out-dir "${OUT_DIR}"
  ) > "${LOG_DIR}/shard_${shard}.log" 2>&1 &
  echo $! > "${LOG_DIR}/pid_shard${shard}.txt"
  echo "[launcher] shard ${shard} -> GPU ${gpu}, pid $(cat "${LOG_DIR}/pid_shard${shard}.txt")" | tee -a "${LOG_DIR}/launcher.log"
  sleep 2
done

set +e
failed=0
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  pid="$(cat "${LOG_DIR}/pid_shard${shard}.txt")"
  wait "${pid}"
  rc=$?
  if [[ "${rc}" != "0" ]]; then
    echo "[launcher] shard ${shard} failed with rc=${rc}" | tee -a "${LOG_DIR}/launcher.log"
    failed=1
  fi
done
set -e

if [[ "${failed}" != "0" ]]; then
  echo "[launcher] not merging because at least one shard failed" | tee -a "${LOG_DIR}/launcher.log"
  exit 1
fi

"${PYTHON_BIN}" experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py \
  --out-dir "${OUT_DIR}" \
  --num-shards "${NUM_SHARDS}" \
  > "${LOG_DIR}/merge.log" 2>&1
echo "[launcher] done at $(date); summary=${OUT_DIR}/summary.json" | tee -a "${LOG_DIR}/launcher.log"
