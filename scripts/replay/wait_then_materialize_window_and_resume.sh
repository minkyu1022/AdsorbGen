#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
REPLAY_DIR="${REPLAY_DIR:-/home/irteam/runs/self_improve_lbfgs_ID_mlip_pairdist_1x_ep99_10k_x10_20260523}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DATA_DIR="${OUT_DATA_DIR:-/home/irteam/data/processed_ID_self_improve_window_20260526}"
TRAIN_OUT="${TRAIN_OUT:-/home/irteam/runs/ID_mlip_pairdist_only_1x_bs64_window_20260526}"
EPOCHS="${EPOCHS:-150}"
WINDOW_EV="${WINDOW_EV:-0.1}"
GT_INDEX="${GT_INDEX:-/home/irteam/data/replay/gt_index_by_sid_oc20_lbfgs.pkl}"
ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/AdsorbGen/data/pkls/adsorbates.pkl}"
TRAIN_LMDBS=(${TRAIN_LMDBS:-/home/irteam/data/processed_ID/is2res_train.lmdb /home/irteam/data/processed_ID/is2res_val.lmdb})

mkdir -p "${REPLAY_DIR}/logs" "${OUT_DATA_DIR}" "${TRAIN_OUT}"
LOG="${REPLAY_DIR}/logs/wait_then_materialize_window_and_resume.log"

{
  echo "[auto-window] start $(date -Is)"
  echo "[auto-window] waiting for ${NUM_SHARDS} shard json files under ${REPLAY_DIR}"
} >> "${LOG}"

while true; do
  n_done=$(find "${REPLAY_DIR}" -maxdepth 1 -name 'shard_*.json' | wc -l)
  n_active=$(pgrep -af 'self_improve_lbfgs_worker.py' | grep -F -- "${REPLAY_DIR}" | wc -l)
  echo "[auto-window] $(date -Is) done_json=${n_done}/${NUM_SHARDS} active_workers=${n_active}" >> "${LOG}"
  if [[ "${n_done}" -ge "${NUM_SHARDS}" ]]; then
    break
  fi
  if [[ "${n_active}" -eq 0 ]]; then
    echo "[auto-window] ERROR: no active self-improve workers before completion" >> "${LOG}"
    exit 1
  fi
  sleep 600
done

echo "[auto-window] replay complete; merging best-success report" >> "${LOG}"
"${PYTHON_BIN}" "${REPO}/scripts/replay/merge_self_improve_successes.py" \
  --out-dir "${REPLAY_DIR}" >> "${LOG}" 2>&1

RAW_TRAIN_LMDB="${OUT_DATA_DIR}/is2res_train_val_window_${WINDOW_EV}ev.raw.lmdb"
TRAIN_LMDB="${OUT_DATA_DIR}/is2res_train_val_window_${WINDOW_EV}ev.unwrap_centered.lmdb"
echo "[auto-window] materializing raw window LMDB ${RAW_TRAIN_LMDB}" >> "${LOG}"
env PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" "${REPO}/scripts/replay/materialize_window_lmdb.py" \
  --train-lmdb "${TRAIN_LMDBS[@]}" \
  --gt-index "${GT_INDEX}" \
  --replay-dir "${REPLAY_DIR}" \
  --window-ev "${WINDOW_EV}" \
  --out-lmdb "${RAW_TRAIN_LMDB}" >> "${LOG}" 2>&1

echo "[auto-window] unwrap/center window LMDB ${TRAIN_LMDB}" >> "${LOG}"
rm -f "${TRAIN_LMDB}" "${TRAIN_LMDB}-lock"
env PYTHONPATH="${REPO}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m adsorbgen.scripts.unwrap_preprocess \
  --src "${RAW_TRAIN_LMDB}" \
  --dst "${TRAIN_LMDB}" \
  --adsorbates-pkl "${ADSORBATES_PKL}" \
  --center-mode relaxed_all \
  --pbc-axes xy >> "${LOG}" 2>&1

echo "[auto-window] launching window-buffer resume training to epoch ${EPOCHS}" >> "${LOG}"
setsid -f bash -c "
  cd '${REPO}' &&
  exec env TRAIN_LMDB='${TRAIN_LMDB}' OUT='${TRAIN_OUT}' EPOCHS='${EPOCHS}' \
    '${REPO}/scripts/train/launch_replacement_onebest_resume.sh' \
      > '${TRAIN_OUT}/train_replacement_window_resume.log' 2>&1
" &
echo "$!" > "${TRAIN_OUT}/pid_replacement_window_resume.txt"
echo "[auto-window] launched pid=$(cat "${TRAIN_OUT}/pid_replacement_window_resume.txt")" >> "${LOG}"
