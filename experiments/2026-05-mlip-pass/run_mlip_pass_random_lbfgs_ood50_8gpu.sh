#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
OUT_DIR="${OUT_DIR:-/home/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline2_random}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-8}"
ADSORBATES_PKL="${ADSORBATES_PKL:-/home/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
LMDB="${LMDB:-/home/irteam/data-vol1/minkyu/data/processed_old/oc20dense.lmdb}"
COVER_DIR="${COVER_DIR:-/home/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
SPLIT_MEMBERSHIP="${SPLIT_MEMBERSHIP:-/home/irteam/data/replay/oc20dense_oc20_split_membership.json}"
PRISTINE_SLABS="${PRISTINE_SLABS:-/home/irteam/results/pristine_slabs/oc20dense_uma.pkl}"
PRISTINE_INDEX="${PRISTINE_INDEX:-/home/irteam/results/pristine_slabs/oc20dense.system_index.pkl}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
cd "${ROOT}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi

echo "[launcher] starting Baseline2 random MLIP Pass OOD-50 L-BFGS eval at $(date)" | tee -a "${LOG_DIR}/launcher.log"
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export ADSORBATES_PKL="${ADSORBATES_PKL}"
    "${PYTHON_BIN}" experiments/2026-05-mlip-pass/eval_mlip_pass_random_lbfgs_ood50.py \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --out-dir "${OUT_DIR}" \
      --lmdb "${LMDB}" \
      --cover-dir "${COVER_DIR}" \
      --split-membership "${SPLIT_MEMBERSHIP}" \
      --pristine-slabs "${PRISTINE_SLABS}" \
      --pristine-index "${PRISTINE_INDEX}"
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
