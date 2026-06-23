#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/eval_energy_gap_curve.py"
DIAG_ROOT="${DIAG_ROOT:-/home1/irteam/data/replay/diag_replay_setting_ood50_20260614_151458}"
OUT_DIR="${OUT_DIR:-/home1/irteam/data/replay/energy_gap_curve_train_unique1000_3placements_100epoch_20260614}"
GPUS=(${GPUS:-0 1 2 3})

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}/logs"
LOCK_DIR="${OUT_DIR}/.run_lock"
DONE_FILE="${OUT_DIR}/energy_gap_curve.done"
echo "[energy-gap] waiting for GPU0-3 diagnostic chain: ${DIAG_ROOT}" | tee -a "${OUT_DIR}/logs/launcher.log"
while [[ ! -f "${DIAG_ROOT}/combined_summary.json" ]]; do
  running=$(pgrep -fc "diag_replay_setting_ood50.py.*${DIAG_ROOT}" || true)
  echo "[energy-gap] $(date -Is) combined_summary=no diag_running=${running}" | tee -a "${OUT_DIR}/logs/launcher.log"
  if [[ "${running}" -eq 0 ]]; then
    echo "[energy-gap] ERROR: diagnostic chain is not running and combined_summary.json is absent" | tee -a "${OUT_DIR}/logs/launcher.log" >&2
    exit 1
  fi
  sleep 60
done

if [[ -f "${DONE_FILE}" ]]; then
  echo "[energy-gap] already complete; skip $(date -Is)" | tee -a "${OUT_DIR}/logs/launcher.log"
  exit 0
fi
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[energy-gap] another launcher owns ${LOCK_DIR}; skip $(date -Is)" | tee -a "${OUT_DIR}/logs/launcher.log"
  exit 0
fi

echo "[energy-gap] diagnostic chain complete; launching curve eval $(date -Is)" | tee -a "${OUT_DIR}/logs/launcher.log"
"${PY}" "${SCRIPT}" \
  --mode list \
  --dataset train_unique \
  --out-dir "${OUT_DIR}" \
  --max-epoch 99 \
  --seeds 0 \
  --max-samples 1000 \
  --num-placements 3 | tee -a "${OUT_DIR}/logs/launcher.log"

pids=()
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  log="${OUT_DIR}/logs/worker_${i}_gpu${gpu}.log"
  (
    cd "${REPO}"
    exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" \
      --mode worker \
      --dataset train_unique \
      --out-dir "${OUT_DIR}" \
      --worker-idx "${i}" \
      --num-workers "${#GPUS[@]}" \
      --max-epoch 99 \
      --seeds 0 \
      --max-samples 1000 \
      --num-placements 3
  ) >"${log}" 2>&1 &
  pids+=("$!")
  echo "[energy-gap] worker=${i} gpu=${gpu} pid=${pids[-1]} log=${log}" | tee -a "${OUT_DIR}/logs/launcher.log"
  sleep 2
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[energy-gap] ERROR: at least one worker failed" | tee -a "${OUT_DIR}/logs/launcher.log" >&2
  exit 1
fi

"${PY}" "${SCRIPT}" \
  --mode merge \
  --dataset train_unique \
  --out-dir "${OUT_DIR}" \
  --max-epoch 99 \
  --seeds 0 \
  --max-samples 1000 \
  --num-placements 3 | tee -a "${OUT_DIR}/logs/launcher.log"
touch "${DONE_FILE}"
echo "[energy-gap] done OUT_DIR=${OUT_DIR} $(date -Is)" | tee -a "${OUT_DIR}/logs/launcher.log"
