#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"

LABEL="${LABEL:?LABEL is required}"
CKPT="${CKPT:?CKPT is required}"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/${LABEL}_uma1p2_two_lbfgs_$(date +%Y%m%d_%H%M%S)}"

LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
FLOW_STEPS="${FLOW_STEPS:-50}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
MAX_ATOMS="${MAX_ATOMS:-32768}"
SP_CHUNK_JOBS="${SP_CHUNK_JOBS:-128}"
EXTRA_ARGS=(${EXTRA_ARGS:-})
RUN_STRICT="${RUN_STRICT:-1}"
RUN_DEFAULT="${RUN_DEFAULT:-1}"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

run_one() {
  local setting="$1"
  local fmax="$2"
  local maxstep="$3"
  local memory="$4"
  local out="${OUT_ROOT}/${setting}/${LABEL}"
  mkdir -p "${out}/logs"
  echo "[model-passk] start setting=${setting} label=${LABEL} fmax=${fmax} maxstep=${maxstep} memory=${memory} $(date -Is)" \
    | tee -a "${OUT_ROOT}/logs/launcher.log"

  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local log="${out}/logs/shard_${shard}.log"
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
        --mode model \
        --label "${LABEL}" \
        --ckpt "${CKPT}" \
        --out-dir "${out}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --lmdb "${LMDB}" \
        --selected-systems "${SELECTED}" \
        --cover-dir "${COVER}" \
        --num-samples "${NUM_SAMPLES}" \
        --flow-steps "${FLOW_STEPS}" \
        --flow-batch-size "${FLOW_BATCH_SIZE}" \
        --prior-mode random_heuristic \
        --uma-model uma-s-1p2 \
        --uma-task oc20 \
        --fmax "${fmax}" \
        --max-steps 300 \
        --max-atoms "${MAX_ATOMS}" \
        --sp-chunk-jobs "${SP_CHUNK_JOBS}" \
        --maxstep "${maxstep}" \
        --lbfgs-memory "${memory}" \
        --lbfgs-damping 1.0 \
        --lbfgs-alpha 70.0 \
        --epsilon-succ 0.1 \
        --lbfgs-streaming \
        --lbfgs-check-interval 10 \
        "${EXTRA_ARGS[@]}"
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[model-passk] ${setting}/${LABEL} shard=${shard} gpu=${gpu} pid=${pids[-1]}" \
      | tee -a "${OUT_ROOT}/logs/launcher.log"
    sleep 2
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[model-passk] ERROR setting=${setting} label=${LABEL}; logs=${out}/logs" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -120 "${f}" || true
    done
    exit 1
  fi
  "${PYTHON_BIN}" "${MERGE}" "${out}" --num-samples "${NUM_SAMPLES}" | tee "${out}/logs/merge.log"
  echo "[model-passk] done setting=${setting} label=${LABEL} $(date -Is)" \
    | tee -a "${OUT_ROOT}/logs/launcher.log"
}

if [[ "${RUN_STRICT}" == "1" ]]; then
  run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "0.01" "0.04" "50"
fi
if [[ "${RUN_DEFAULT}" == "1" ]]; then
  run_one "ase_default_fmax0p05_maxstep0p2_mem100" "0.05" "0.2" "100"
fi

echo "[model-passk] all done OUT_ROOT=${OUT_ROOT} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
