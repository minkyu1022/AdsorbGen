#!/usr/bin/env bash
set -euo pipefail

ROOT=/home1/irteam
PY=${PY:-${ROOT}/micromamba/envs/adsorbgen/bin/python}
REPO=${REPO:-${ROOT}/AdsorbGen}
BASE=${BASE:-${ROOT}/data/uma_s_1p2_references}
OC20DENSE_RAW=${OC20DENSE_RAW:-${BASE}/raw_relax/oc20dense_lmdb}
OC20DENSE_MAT=${OC20DENSE_MAT:-${BASE}/materialized/oc20dense_raw}
OC20DENSE_PRE=${OC20DENSE_PRE:-${BASE}/processed/oc20dense_unwrap_centered.lmdb}
ID_TRAIN_RAW=${ID_TRAIN_RAW:-${BASE}/raw_relax/processed_ID_train}
LOG=${LOG:-${BASE}/pipeline_continue.log}

mkdir -p "${BASE}" "${OC20DENSE_MAT}" "${ID_TRAIN_RAW}/logs"

echo "[$(date)] watcher start" >> "${LOG}"
while pgrep -f "batched_uma_relabel.py --source-kind oc20dense_lmdb" >/dev/null; do
  echo "[$(date)] waiting oc20dense relabel..." >> "${LOG}"
  sleep 120
done
echo "[$(date)] oc20dense relabel finished" >> "${LOG}"

(
  set -euo pipefail
  echo "[$(date)] materialize oc20dense start" >> "${LOG}"
  "${PY}" "${REPO}/scripts/replay/materialize_oc20dense_uma1p2.py" \
    --source-lmdb "${ROOT}/data/processed_old/oc20dense.lmdb" \
    --shard-dir "${OC20DENSE_RAW}" \
    --num-shards 8 \
    --out-dir "${OC20DENSE_MAT}" \
    --require-all-shards \
    > "${OC20DENSE_MAT}/materialize.log" 2>&1
  echo "[$(date)] unwrap/center oc20dense start" >> "${LOG}"
  PYTHONPATH="${REPO}" "${PY}" -m adsorbgen.scripts.unwrap_preprocess \
    --src "${OC20DENSE_MAT}/oc20dense.lmdb" \
    --dst "${OC20DENSE_PRE}" \
    --adsorbates-pkl "${REPO}/data/pkls/adsorbates.pkl" \
    --center-mode relaxed_all \
    --pbc-axes xy \
    > "${OC20DENSE_MAT}/unwrap_center.log" 2>&1
  echo "[$(date)] oc20dense materialize+unwrap done" >> "${LOG}"
) &

echo "[$(date)] launch processed_ID train relabel" >> "${LOG}"
for g in 0 1 2 3 4 5 6 7; do
  if pgrep -f "batched_uma_relabel.py --source-kind id_lmdb .*--out-dir ${ID_TRAIN_RAW} .*--shard-idx ${g} " >/dev/null; then
    echo "[$(date)] processed_ID train shard ${g} already running; skip" >> "${LOG}"
    continue
  fi
  : > "${ID_TRAIN_RAW}/logs/relabel_id_train_shard${g}.log"
  setsid -f env CUDA_VISIBLE_DEVICES="${g}" PYTHONPATH="${REPO}" "${PY}" -u "${REPO}/scripts/replay/batched_uma_relabel.py" \
    --source-kind id_lmdb \
    --input "${ROOT}/data/processed_ID/is2res_train.lmdb" \
    --out-dir "${ID_TRAIN_RAW}" \
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
    --resume > "${ID_TRAIN_RAW}/logs/relabel_id_train_shard${g}.log" 2>&1
  sleep 2
done
echo "[$(date)] processed_ID train relabel launched" >> "${LOG}"
