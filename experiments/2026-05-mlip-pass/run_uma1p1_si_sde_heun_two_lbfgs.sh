#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/uma1p1_si_sde_heun_two_lbfgs_$(date +%Y%m%d_%H%M%S)}"

LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
SI_CKPT="${SI_CKPT:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5/ckpt_epochepoch=139.ckpt}"

SI_RUN_DIR="${SI_RUN_DIR:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5}"
GAUSS_RUN_DIR="${GAUSS_RUN_DIR:-/home1/irteam/runs/training/base_gaussian_adsprior_102M}"
SI_LAUNCH="${SI_LAUNCH:-${SI_RUN_DIR}/launch_command.sh}"
GAUSS_LAUNCH="${GAUSS_LAUNCH:-${GAUSS_RUN_DIR}/launch_command.sh}"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-8}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_ATOMS="${MAX_ATOMS:-32768}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
SP_CHUNK_JOBS="${SP_CHUNK_JOBS:-128}"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

stop_training() {
  echo "[uma1p1-heun] stop current training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
  mapfile -t train_pids < <(
    ps -eo pid=,args= \
      | awk -v si="${SI_RUN_DIR}" -v gauss="${GAUSS_RUN_DIR}" \
        'index($0, "adsorbgen.training.train_cli") && (index($0, "--out " si) || index($0, "--out " gauss)) {print $1}'
  )
  if [[ "${#train_pids[@]}" -gt 0 ]]; then
    printf '%s\n' "${train_pids[@]}" > "${OUT_ROOT}/logs/stopped_training_pids.txt"
    kill -TERM "${train_pids[@]}" || true
  fi
  sleep 20
  mapfile -t train_pids < <(
    ps -eo pid=,args= \
      | awk -v si="${SI_RUN_DIR}" -v gauss="${GAUSS_RUN_DIR}" \
        'index($0, "adsorbgen.training.train_cli") && (index($0, "--out " si) || index($0, "--out " gauss)) {print $1}'
  )
  if [[ "${#train_pids[@]}" -gt 0 ]]; then
    kill -KILL "${train_pids[@]}" || true
  fi
}

resume_training() {
  echo "[uma1p1-heun] resume training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
  if [[ -x "${SI_LAUNCH}" ]] && ! pgrep -af "${SI_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${SI_LAUNCH}" > "${SI_RUN_DIR}/train.log" 2>&1 &
    echo "[uma1p1-heun] resumed SI pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
  fi
  if [[ -x "${GAUSS_LAUNCH}" ]] && ! pgrep -af "${GAUSS_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${GAUSS_LAUNCH}" > "${GAUSS_RUN_DIR}/train.log" 2>&1 &
    echo "[uma1p1-heun] resumed gaussian pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
  fi
}

trap 'resume_training' EXIT

run_heun() {
  local setting="$1"
  local fmax="$2"
  local maxstep="$3"
  local memory="$4"
  local out="${OUT_ROOT}/${setting}/si_sde200_eps0p01_heun"
  mkdir -p "${out}/logs"
  echo "[uma1p1-heun] start setting=${setting} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"

  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local log="${out}/logs/shard_${shard}.log"
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
        --mode model \
        --label si_sde200_eps0p01_heun \
        --ckpt "${SI_CKPT}" \
        --out-dir "${out}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --lmdb "${LMDB}" \
        --selected-systems "${SELECTED}" \
        --cover-dir "${COVER}" \
        --num-samples "${NUM_SAMPLES}" \
        --sample-mode sde \
        --solver heun \
        --flow-steps 200 \
        --flow-batch-size "${FLOW_BATCH_SIZE}" \
        --prior-mode random_heuristic \
        --sde-mode omatg_si \
        --sde-schedule atommof \
        --sde-alpha 1.0 \
        --si-gamma-schedule sqrt_t1mt \
        --si-gamma-sigma 0.1 \
        --si-epsilon-schedule vanishing_1mt \
        --si-epsilon-scale 0.01 \
        --time-schedule uniform \
        --uma-model uma-s-1p1 \
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
        --lbfgs-check-interval 10
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[uma1p1-heun] ${setting} shard=${shard} gpu=${gpu} pid=${pids[-1]}" \
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
    echo "[uma1p1-heun] ERROR setting=${setting}; logs=${out}/logs" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -120 "${f}" || true
    done
    exit 1
  fi
  "${PYTHON_BIN}" "${MERGE}" "${out}" --num-samples "${NUM_SAMPLES}" | tee "${out}/logs/merge.log"
  echo "[uma1p1-heun] done setting=${setting} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
}

stop_training
run_heun "pass_strict_fmax0p01_maxstep0p04_mem50" "0.01" "0.04" "50"
run_heun "ase_default_fmax0p05_maxstep0p2_mem100" "0.05" "0.2" "100"

echo "[uma1p1-heun] all done OUT_ROOT=${OUT_ROOT} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
