#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
LAUNCH="${REPO}/experiments/2026-05-mlip-pass/launch_diag_replay_setting_ood50_defaultmaxstep_base_random_adsorbdiff.sh"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/diag_replay_setting_ood50_defaultmaxstep_$(date +%Y%m%d_%H%M%S)}"
GPUS="${GPUS:-0 1 2 3}"
MAX_USED_MB="${MAX_USED_MB:-30000}"
POLL_SEC="${POLL_SEC:-300}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"
echo "[wait-defaultmaxstep] OUT_ROOT=${OUT_ROOT}"
echo "[wait-defaultmaxstep] waiting GPUS='${GPUS}' max_used_mb=${MAX_USED_MB}"

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  ok=1
  for gpu in ${GPUS}; do
    val=$(printf '%s\n' "${used[@]}" | awk -F, -v g="${gpu}" '$1+0==g {gsub(/ /,"",$2); print $2}')
    val=${val:-999999}
    if (( val > MAX_USED_MB )); then
      ok=0
    fi
  done
  printf '[wait-defaultmaxstep] %s ' "$(date -Is)"
  for gpu in ${GPUS}; do
    val=$(printf '%s\n' "${used[@]}" | awk -F, -v g="${gpu}" '$1+0==g {gsub(/ /,"",$2); print $2}')
    printf 'gpu%s=%sMB ' "${gpu}" "${val:-NA}"
  done
  printf '\n'
  if (( ok == 1 )); then
    break
  fi
  sleep "${POLL_SEC}"
done

echo "[wait-defaultmaxstep] GPUs available; launching $(date -Is)"
exec env OUT_ROOT="${OUT_ROOT}" GPUS="${GPUS}" "${LAUNCH}"
