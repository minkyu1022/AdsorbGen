#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/diag_replay_setting_ood50_defaultmaxstep_$(date +%Y%m%d_%H%M%S)}"
ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
BASE_CKPT="${BASE_CKPT:-/home1/irteam/runs/training/base/base.ckpt}"
ADSORBDIFF_METADATA="${ADSORBDIFF_METADATA:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff_score32_20260615_010951/metadata.json}"
ADSORBDIFF_LMDB="${ADSORBDIFF_LMDB:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/adsorbdiff_ood50_100.lmdb}"
ADSORBDIFF_RESULTS_DIR="${ADSORBDIFF_RESULTS_DIR:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff_score32_20260615_010951}"
GPUS=(${GPUS:-0 1 2 3})
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

common_args=(
  --lmdb "${LMDB}"
  --selected-systems "${SELECTED}"
  --cover-dir "${COVER}"
  --num-samples 100
  --flow-steps 50
  --flow-batch-size 64
  --prior-mode random_heuristic
  --uma-model uma-s-1p2
  --uma-task oc20
  --fmax 0.05
  --max-steps 300
  --max-atoms 32768
  --maxstep 0.2
  --lbfgs-memory 100
  --lbfgs-damping 1.0
  --lbfgs-alpha 70.0
  --lbfgs-streaming
  --lbfgs-check-interval 10
)

run_one() {
  local label="$1"
  local mode="$2"
  local ckpt="${3:-}"
  local out="${OUT_ROOT}/${label}"
  mkdir -p "${out}/logs"
  echo "[diag-defaultmaxstep] start label=${label} mode=${mode} $(date -Is)"
  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local log="${out}/logs/shard_${shard}.log"
    local args=(--mode "${mode}" --label "${label}" --out-dir "${out}" --shard-idx "${shard}" --num-shards "${NUM_SHARDS}" "${common_args[@]}")
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
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[diag-defaultmaxstep] ${label} shard=${shard} gpu=${gpu} pid=${pids[-1]}"
    sleep 2
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[diag-defaultmaxstep] ERROR label=${label}; logs=${out}/logs" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -120 "${f}" || true
    done
    exit 1
  fi
  "${PYTHON_BIN}" "${MERGE}" "${out}" | tee "${out}/logs/merge.log"
  echo "[diag-defaultmaxstep] done label=${label} $(date -Is)"
}

run_one "base_ID_pairdist_epoch149_defaultmaxstep" "model" "${BASE_CKPT}"
run_one "B2_random_heuristic_defaultmaxstep" "random"
run_one "B3_adsorbdiff_defaultmaxstep" "adsorbdiff"

"${PYTHON_BIN}" - <<PY
import json, pathlib
root = pathlib.Path("${OUT_ROOT}")
rows = []
for p in sorted(root.glob("*/summary.json")):
    d = json.loads(p.read_text())
    rows.append({
        "label": d.get("label"),
        "candidates": d.get("candidates"),
        "converged_rate": d.get("converged_rate"),
        "sp_delta_mean": (d.get("sp_delta_E_sys") or {}).get("mean"),
        "sp_abs_delta_mean": (d.get("sp_abs_delta_E_sys") or {}).get("mean"),
        "conv_steps_mean": (d.get("converged_n_steps") or {}).get("mean"),
        "all_steps_mean": (d.get("all_n_steps") or {}).get("mean"),
        "all_steps_median": (d.get("all_n_steps") or {}).get("median"),
        "final_delta_mean": (d.get("final_delta_E_sys") or {}).get("mean"),
        "settings": d.get("settings"),
    })
(root / "combined_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
print(json.dumps(rows, indent=2, sort_keys=True))
PY

echo "[diag-defaultmaxstep] all done OUT_ROOT=${OUT_ROOT}"
