#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/irteam
PY=${PY:-${ROOT}/micromamba/envs/adsorbgen/bin/python}
REPO=${REPO:-${ROOT}/AdsorbGen}
BASE=${BASE:-${ROOT}/data/uma_s_1p2_references}
ID_TRAIN_RAW=${ID_TRAIN_RAW:-${BASE}/raw_relax/processed_ID_train}
ID_VAL_RAW=${ID_VAL_RAW:-${BASE}/raw_relax/processed_ID_val}
BARE_RAW=${BARE_RAW:-${BASE}/raw_relax/bare_slab}
ID_MAT=${ID_MAT:-${BASE}/materialized/processed_ID_raw}
ID_PROCESSED=${ID_PROCESSED:-${BASE}/processed/processed_ID_unwrap_centered}
LOG=${LOG:-${BASE}/pipeline_continue_id.log}

mkdir -p "${ID_VAL_RAW}/logs" "${BARE_RAW}/logs" "${ID_MAT}" "${ID_PROCESSED}"

echo "[$(date)] watcher start: wait ID train" >> "${LOG}"
while pgrep -f "batched_uma_relabel.py --source-kind id_lmdb .*processed_ID_train" >/dev/null; do
  echo "[$(date)] waiting processed_ID_train relabel..." >> "${LOG}"
  sleep 300
done
echo "[$(date)] processed_ID_train relabel finished" >> "${LOG}"

echo "[$(date)] launch processed_ID val relabel" >> "${LOG}"
for g in 0 1 2 3 4 5 6 7; do
  : > "${ID_VAL_RAW}/logs/relabel_id_val_shard${g}.log"
  setsid -f env CUDA_VISIBLE_DEVICES="${g}" PYTHONPATH="${REPO}" "${PY}" -u "${REPO}/scripts/replay/batched_uma_relabel.py" \
    --source-kind id_lmdb \
    --input "${ROOT}/data/processed_ID/is2res_val.lmdb" \
    --out-dir "${ID_VAL_RAW}" \
    --shard-idx "${g}" \
    --num-shards 8 \
    --records-per-chunk 512 \
    --skip-anomaly-mask \
    --uma-model uma-s-1p2 \
    --uma-task oc20 \
    --fmax 0.05 \
    --max-steps 300 \
    --max-atoms 65536 \
    --maxstep 0.2 \
    --lbfgs-memory 100 \
    --lbfgs-streaming \
    --lbfgs-check-interval 20 \
    --lbfgs-keep-survivors-on-gpu \
    --resume > "${ID_VAL_RAW}/logs/relabel_id_val_shard${g}.log" 2>&1
  sleep 2
done

while pgrep -f "batched_uma_relabel.py --source-kind id_lmdb .*processed_ID_val" >/dev/null; do
  echo "[$(date)] waiting processed_ID_val relabel..." >> "${LOG}"
  sleep 300
done
echo "[$(date)] processed_ID val relabel finished" >> "${LOG}"

(
  set -euo pipefail
  echo "[$(date)] materialize processed_ID start" >> "${LOG}"
  "${PY}" "${REPO}/scripts/replay/materialize_processed_id_uma1p2.py" \
    --shard-dirs "${ID_TRAIN_RAW}" "${ID_VAL_RAW}" \
    --num-shards 8 \
    --out-dir "${ID_MAT}" \
    --require-all-shards \
    > "${ID_MAT}/materialize.log" 2>&1
  echo "[$(date)] unwrap/center processed_ID train/val start" >> "${LOG}"
  PYTHONPATH="${REPO}" "${PY}" -m adsorbgen.scripts.unwrap_preprocess \
    --src "${ID_MAT}/is2res_train.lmdb" \
    --dst "${ID_PROCESSED}/is2res_train.lmdb" \
    --adsorbates-pkl "${REPO}/data/pkls/adsorbates.pkl" \
    --center-mode relaxed_all \
    --pbc-axes xy \
    > "${ID_MAT}/unwrap_train.log" 2>&1
  PYTHONPATH="${REPO}" "${PY}" -m adsorbgen.scripts.unwrap_preprocess \
    --src "${ID_MAT}/is2res_val.lmdb" \
    --dst "${ID_PROCESSED}/is2res_val.lmdb" \
    --adsorbates-pkl "${REPO}/data/pkls/adsorbates.pkl" \
    --center-mode relaxed_all \
    --pbc-axes xy \
    > "${ID_MAT}/unwrap_val.log" 2>&1
  echo "[$(date)] processed_ID materialize+unwrap done" >> "${LOG}"
) &

echo "[$(date)] launch bare slab relabel" >> "${LOG}"
for g in 0 1 2 3 4 5 6 7; do
  : > "${BARE_RAW}/logs/relabel_bare_slab_shard${g}.log"
  setsid -f env CUDA_VISIBLE_DEVICES="${g}" PYTHONPATH="${REPO}" "${PY}" -u "${REPO}/scripts/replay/batched_uma_relabel.py" \
    --source-kind bare_slab_pkl \
    --input "${ROOT}/data-vol1/minkyu/data/replay/E_slab_only_lbfgs_by_slab.pkl" \
    --out-dir "${BARE_RAW}" \
    --shard-idx "${g}" \
    --num-shards 8 \
    --records-per-chunk 512 \
    --uma-model uma-s-1p2 \
    --uma-task oc20 \
    --fmax 0.05 \
    --max-steps 300 \
    --max-atoms 65536 \
    --maxstep 0.2 \
    --lbfgs-memory 100 \
    --lbfgs-streaming \
    --lbfgs-check-interval 20 \
    --lbfgs-keep-survivors-on-gpu \
    --resume > "${BARE_RAW}/logs/relabel_bare_slab_shard${g}.log" 2>&1
  sleep 2
done
echo "[$(date)] bare slab relabel launched" >> "${LOG}"
