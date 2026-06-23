#!/usr/bin/env bash
set -euo pipefail

FULL_DIR="${FULL_DIR:?FULL_DIR is required}"
GEN_PID="${GEN_PID:-}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${FULL_DIR}/remote_half_relax_package_${STAMP}}"
LOG_DIR="${FULL_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/split_then_local_relax_${STAMP}.log"

echo "[watch] start $(date -Is)" | tee -a "${LOG}"
echo "[watch] full_dir=${FULL_DIR}" | tee -a "${LOG}"
echo "[watch] gen_pid=${GEN_PID}" | tee -a "${LOG}"
echo "[watch] package_root=${PACKAGE_ROOT}" | tee -a "${LOG}"

while true; do
  if [[ -n "${GEN_PID}" ]]; then
    if ! kill -0 "${GEN_PID}" 2>/dev/null; then
      break
    fi
  else
    if ! pgrep -f "two_stage_full_replay.py generate .*${FULL_DIR}" >/dev/null 2>&1; then
      break
    fi
  fi
  pkl_count="$(find "${FULL_DIR}/flow_jobs" -maxdepth 1 -name 'jobs_*.pkl' 2>/dev/null | wc -l || true)"
  echo "[watch] generate still running $(date -Is) pkl_count=${pkl_count}" | tee -a "${LOG}"
  sleep 300
done

echo "[watch] generate appears finished $(date -Is)" | tee -a "${LOG}"

if [[ -f "${FULL_DIR}/relax_skipped.json" ]]; then
  echo "[watch] automatic relax skipped marker observed" | tee -a "${LOG}"
fi

PY="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
"${PY}" /home1/irteam/AdsorbGen/geoopt/split_full_replay_for_remote_relax.py \
  --full-dir "${FULL_DIR}" \
  --package-root "${PACKAGE_ROOT}" \
  --force 2>&1 | tee -a "${LOG}"

echo "[watch] starting local half relaxation $(date -Is)" | tee -a "${LOG}"
nohup bash "${PACKAGE_ROOT}/run_local_relax.sh" > "${LOG_DIR}/local_half_relax_${STAMP}.nohup.log" 2>&1 &
echo "$!" > "${LOG_DIR}/local_half_relax_${STAMP}.pid"
echo "[watch] local half pid=$(cat "${LOG_DIR}/local_half_relax_${STAMP}.pid")" | tee -a "${LOG}"
echo "[watch] done $(date -Is)" | tee -a "${LOG}"
