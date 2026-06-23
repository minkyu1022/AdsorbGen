#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
OUT="${OUT:-/home/irteam/data/replay/mlip_pass_lbfgs_ood50_pristine_slab_epoch149_last_20260605_1642_retry1}"
CKPT="${CKPT:-/home/irteam/runs/training/ID_mlip_pairdist_only_1x_bs64_pristine_slab_8gpu/last.ckpt}"
LMDB="${LMDB:-/home/irteam/data/processed_old/oc20dense.lmdb}"
COVER="${COVER:-/home/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
SPLIT="${SPLIT:-/home/irteam/data/replay/oc20dense_oc20_split_membership.json}"
PRISTINE="${PRISTINE:-/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl}"
PINDEX="${PINDEX:-/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl}"
ADS="${ADS:-/home/irteam/data/pkls/adsorbates.pkl}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

LOG="${OUT}/logs"
mkdir -p "${LOG}"
cd "${ROOT}"

echo "[supervisor] start $(date) out=${OUT} ckpt=${CKPT}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPUS[$shard]}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONUNBUFFERED=1
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export ADSORBATES_PKL="${ADS}"
    "${PY}" experiments/2026-05-mlip-pass/eval_mlip_pass_lbfgs_ood50.py \
      --ckpt "${CKPT}" \
      --lmdb "${LMDB}" \
      --cover-dir "${COVER}" \
      --split-membership "${SPLIT}" \
      --out-dir "${OUT}" \
      --shard-idx "${shard}" \
      --num-shards "${NUM_SHARDS}" \
      --num-systems 50 \
      --num-samples 100 \
      --flow-steps 50 \
      --flow-batch-size 32 \
      --prior-mode random_heuristic \
      --slab-source pristine_relaxed \
      --placement-pristine-slabs "${PRISTINE}" \
      --placement-pristine-index "${PINDEX}" \
      --pristine-slabs "${PRISTINE}" \
      --pristine-index "${PINDEX}"
  ) > "${LOG}/shard_${shard}.log" 2>&1 &
  echo $! > "${LOG}/pid_shard${shard}.txt"
  echo "[supervisor] shard ${shard} -> GPU ${gpu} pid $(cat "${LOG}/pid_shard${shard}.txt")"
  sleep 2
done

failed=0
for ((shard=0; shard<NUM_SHARDS; shard++)); do
  pid="$(cat "${LOG}/pid_shard${shard}.txt")"
  if wait "${pid}"; then
    echo "[supervisor] shard ${shard} done"
  else
    echo "[supervisor] shard ${shard} failed"
    failed=1
  fi
done

if [[ "${failed}" != "0" ]]; then
  echo "[supervisor] not merging due to failed shard"
  exit 1
fi

"${PY}" experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py \
  --out-dir "${OUT}" \
  --num-shards "${NUM_SHARDS}" \
  --num-samples 100 > "${LOG}/merge.log" 2>&1

echo "[supervisor] merged $(date) summary=${OUT}/summary.json"
