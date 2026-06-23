#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/valid_passk_uma1p2_two_lbfgs_$(date +%Y%m%d_%H%M%S)}"

LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
BASE_CKPT="${BASE_CKPT:-/home1/irteam/runs/training/base/base.ckpt}"
ADSORBDIFF_METADATA="${ADSORBDIFF_METADATA:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/metadata.json}"
ADSORBDIFF_LMDB="${ADSORBDIFF_LMDB:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/adsorbdiff_ood50_100.lmdb}"
ADSORBDIFF_RESULTS_DIR="${ADSORBDIFF_RESULTS_DIR:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/results_setsid_20260615_001248}"

SI_RUN_DIR="${SI_RUN_DIR:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5}"
GAUSS_RUN_DIR="${GAUSS_RUN_DIR:-/home1/irteam/runs/training/base_gaussian_adsprior_102M}"
SI_LAUNCH="${SI_LAUNCH:-${SI_RUN_DIR}/launch_command.sh}"
GAUSS_LAUNCH="${GAUSS_LAUNCH:-${GAUSS_RUN_DIR}/launch_command.sh}"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_ATOMS="${MAX_ATOMS:-32768}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
SP_CHUNK_JOBS="${SP_CHUNK_JOBS:-128}"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
SKIP_TRAINING_CONTROL="${SKIP_TRAINING_CONTROL:-0}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

stop_training() {
  echo "[valid-passk] stop current training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
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
  echo "[valid-passk] resume training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
  if ! pgrep -af "${SI_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${SI_LAUNCH}" > "${SI_RUN_DIR}/train.log" 2>&1 &
    echo "[valid-passk] resumed SI pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
  fi
  if ! pgrep -af "${GAUSS_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${GAUSS_LAUNCH}" > "${GAUSS_RUN_DIR}/train.log" 2>&1 &
    echo "[valid-passk] resumed gaussian pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
  fi
}

if [[ "${SKIP_TRAINING_CONTROL}" != "1" ]]; then
  trap 'resume_training' EXIT
fi

run_one() {
  local setting="$1"
  local label="$2"
  local mode="$3"
  local fmax="$4"
  local maxstep="$5"
  local memory="$6"
  local ckpt="${7:-}"
  local out="${OUT_ROOT}/${setting}/${label}"
  mkdir -p "${out}/logs"
  echo "[valid-passk] start setting=${setting} label=${label} mode=${mode} fmax=${fmax} maxstep=${maxstep} memory=${memory} $(date -Is)" \
    | tee -a "${OUT_ROOT}/logs/launcher.log"

  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local log="${out}/logs/shard_${shard}.log"
    local args=(
      --mode "${mode}"
      --label "${label}"
      --out-dir "${out}"
      --shard-idx "${shard}"
      --num-shards "${NUM_SHARDS}"
      --lmdb "${LMDB}"
      --selected-systems "${SELECTED}"
      --cover-dir "${COVER}"
      --num-samples 100
      --flow-steps 50
      --flow-batch-size "${FLOW_BATCH_SIZE}"
      --prior-mode random_heuristic
      --uma-model uma-s-1p2
      --uma-task oc20
      --fmax "${fmax}"
      --max-steps 300
      --max-atoms "${MAX_ATOMS}"
      --sp-chunk-jobs "${SP_CHUNK_JOBS}"
      --maxstep "${maxstep}"
      --lbfgs-memory "${memory}"
      --lbfgs-damping 1.0
      --lbfgs-alpha 70.0
      --epsilon-succ 0.1
      --lbfgs-streaming
      --lbfgs-check-interval 10
    )
    if [[ "${mode}" == "model" ]]; then
      args+=(--ckpt "${ckpt}")
    elif [[ "${mode}" == "adsorbdiff" ]]; then
      args+=(
        --adsorbdiff-metadata "${ADSORBDIFF_METADATA}"
        --adsorbdiff-lmdb "${ADSORBDIFF_LMDB}"
        --adsorbdiff-results-dir "${ADSORBDIFF_RESULTS_DIR}"
      )
    fi
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" "${args[@]}"
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[valid-passk] ${setting}/${label} shard=${shard} gpu=${gpu} pid=${pids[-1]}" \
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
    echo "[valid-passk] ERROR setting=${setting} label=${label}; logs=${out}/logs" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -120 "${f}" || true
    done
    exit 1
  fi
  "${PYTHON_BIN}" "${MERGE}" "${out}" --num-samples 100 | tee "${out}/logs/merge.log"
  echo "[valid-passk] done setting=${setting} label=${label} $(date -Is)" \
    | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_combined_summary() {
  "${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(root.glob("*/*/summary.json")):
    d = json.loads(p.read_text())
    rows.append({
        "setting": p.parents[1].name,
        "method": p.parent.name,
        "valid_rate": d.get("valid_rate"),
        "converged_rate": d.get("converged_rate"),
        "valid_pre_gap_mean": (d.get("valid_sp_delta_E_sys") or {}).get("mean"),
        "valid_post_gap_mean": (d.get("valid_final_delta_E_sys") or {}).get("mean"),
        "valid_steps_mean": (d.get("valid_n_steps") or {}).get("mean"),
        "valid_success_rate": d.get("valid_success_rate"),
        "valid_success_systems": d.get("valid_success_systems"),
        "valid_mlip_pass@1": d.get("valid_mlip_pass@1"),
        "valid_mlip_pass@2": d.get("valid_mlip_pass@2"),
        "valid_mlip_pass@5": d.get("valid_mlip_pass@5"),
        "valid_mlip_pass@10": d.get("valid_mlip_pass@10"),
        "settings": d.get("settings"),
    })

(root / "combined_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
lines = ["# Valid-only UMA-s-1p2 Pass@k / LBFGS comparison", ""]
lines.append("| setting | method | valid rate | conv rate | pre-gap mean | post-gap mean | steps mean | succ systems | pass@1 | pass@2 | pass@5 | pass@10 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    def fmt(x):
        return "NA" if x is None else f"{float(x):.6g}"
    lines.append(
        f"| {r['setting']} | {r['method']} | {fmt(r['valid_rate'])} | {fmt(r['converged_rate'])} | "
        f"{fmt(r['valid_pre_gap_mean'])} | {fmt(r['valid_post_gap_mean'])} | {fmt(r['valid_steps_mean'])} | "
        f"{r['valid_success_systems']} | {fmt(r['valid_mlip_pass@1'])} | {fmt(r['valid_mlip_pass@2'])} | "
        f"{fmt(r['valid_mlip_pass@5'])} | {fmt(r['valid_mlip_pass@10'])} |"
    )
(root / "combined_summary.md").write_text("\n".join(lines) + "\n")
print(json.dumps(rows, indent=2, sort_keys=True))
PY
}

if [[ "${SKIP_TRAINING_CONTROL}" != "1" ]]; then
  stop_training
else
  echo "[valid-passk] skip training stop/resume control $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
fi

run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "base" "model" "0.01" "0.04" "50" "${BASE_CKPT}"
run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "random" "random" "0.01" "0.04" "50"
run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "adsorbdiff" "adsorbdiff" "0.01" "0.04" "50"

run_one "ase_default_fmax0p05_maxstep0p2_mem100" "base" "model" "0.05" "0.2" "100" "${BASE_CKPT}"
run_one "ase_default_fmax0p05_maxstep0p2_mem100" "random" "random" "0.05" "0.2" "100"
run_one "ase_default_fmax0p05_maxstep0p2_mem100" "adsorbdiff" "adsorbdiff" "0.05" "0.2" "100"

write_combined_summary
echo "[valid-passk] all done OUT_ROOT=${OUT_ROOT} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
