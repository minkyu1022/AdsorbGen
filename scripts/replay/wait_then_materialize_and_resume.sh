#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
REPLAY_DIR="${REPLAY_DIR:-/home/irteam/runs/self_improve_lbfgs_ID_mlip_pairdist_1x_ep99_10k_x10_20260523}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DATA_DIR="${OUT_DATA_DIR:-/home/irteam/data/processed_ID_self_improve_onebest_20260523}"
TRAIN_OUT="${TRAIN_OUT:-/home/irteam/runs/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544}"
EPOCHS="${EPOCHS:-150}"

mkdir -p "${REPLAY_DIR}/logs" "${OUT_DATA_DIR}"
LOG="${REPLAY_DIR}/logs/wait_then_materialize_and_resume.log"

{
  echo "[auto] start $(date -Is)"
  echo "[auto] waiting for ${NUM_SHARDS} shard json files under ${REPLAY_DIR}"
} >> "${LOG}"

while true; do
  n_done=$(find "${REPLAY_DIR}" -maxdepth 1 -name 'shard_*.json' | wc -l)
  n_active=$(pgrep -af 'self_improve_lbfgs_worker.py' | grep -F -- "${REPLAY_DIR}" | wc -l)
  echo "[auto] $(date -Is) done_json=${n_done}/${NUM_SHARDS} active_workers=${n_active}" >> "${LOG}"
  if [[ "${n_done}" -ge "${NUM_SHARDS}" ]]; then
    break
  fi
  if [[ "${n_active}" -eq 0 ]]; then
    echo "[auto] ERROR: no active self-improve workers before completion" >> "${LOG}"
    exit 1
  fi
  sleep 600
done

echo "[auto] replay complete; merging successes" >> "${LOG}"
"${PYTHON_BIN}" "${REPO}/scripts/replay/merge_self_improve_successes.py" \
  --out-dir "${REPLAY_DIR}" >> "${LOG}" 2>&1

TRAIN_LMDB="${OUT_DATA_DIR}/is2res_train_val_onebest_replaced.lmdb"
echo "[auto] materializing ${TRAIN_LMDB}" >> "${LOG}"
env PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${REPO}/scripts/replay/materialize_onebest_lmdb.py" \
  --train-lmdb /home/irteam/data/processed_ID/is2res_train.lmdb /home/irteam/data/processed_ID/is2res_val.lmdb \
  --gt-index /home/irteam/data/replay/gt_index_by_sid_oc20.pkl \
  --replay-dir "${REPLAY_DIR}" \
  --out-lmdb "${TRAIN_LMDB}" >> "${LOG}" 2>&1

echo "[auto] launching replacement resume training to epoch ${EPOCHS}" >> "${LOG}"
setsid -f bash -c "
  cd '${REPO}' &&
  exec env TRAIN_LMDB='${TRAIN_LMDB}' OUT='${TRAIN_OUT}' EPOCHS='${EPOCHS}' \
    '${REPO}/scripts/train/launch_replacement_onebest_resume.sh' \
      > '${TRAIN_OUT}/train_replacement_onebest_resume.log' 2>&1
" &
echo "$!" > "${TRAIN_OUT}/pid_replacement_onebest_resume.txt"
echo "[auto] launched pid=$(cat "${TRAIN_OUT}/pid_replacement_onebest_resume.txt")" >> "${LOG}"
