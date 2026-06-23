#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
ROOT="${ROOT:?ROOT required}"
ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
GPUS=(${GPUS:-0 1 2 3})

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
  --maxstep 0.04
  --lbfgs-memory 50
  --lbfgs-damping 1.0
  --lbfgs-alpha 70.0
  --lbfgs-streaming
  --lbfgs-check-interval 10
)

wait_existing_and_merge() {
  local label="$1"
  local out="${ROOT}/${label}"
  echo "[chain] waiting existing ${label} $(date -Is)"
  while true; do
    local done_count
    done_count=$(find "${out}" -maxdepth 1 -name 'shard_*.json' 2>/dev/null | wc -l)
    if [[ "${done_count}" -ge 4 ]]; then
      break
    fi
    local running
    running=$(pgrep -fc "diag_replay_setting_ood50.py.*${label}" || true)
    if [[ "${running}" -eq 0 ]]; then
      echo "[chain] ERROR ${label} stopped before all shards completed (${done_count}/4)" >&2
      for f in "${out}"/logs/shard_*.log; do
        echo "==== ${f} ===="
        tail -80 "${f}" || true
      done
      exit 1
    fi
    sleep 30
  done
  "${PY}" "${MERGE}" "${out}" | tee "${out}/logs/merge.log"
}

launch_and_wait() {
  local label="$1"
  local mode="$2"
  local ckpt="${3:-}"
  local out="${ROOT}/${label}"
  mkdir -p "${out}/logs"
  rm -f "${out}"/shard_*.json "${out}"/shard_*.pkl
  echo "[chain] launch ${label} mode=${mode} $(date -Is)"
  local pids=()
  for shard in 0 1 2 3; do
    local gpu="${GPUS[$shard]}"
    local log="${out}/logs/shard_${shard}.log"
    local args=(--mode "${mode}" --label "${label}" --out-dir "${out}" --shard-idx "${shard}" --num-shards 4 "${common_args[@]}")
    if [[ "${mode}" == "model" ]]; then
      args+=(--ckpt "${ckpt}")
    fi
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${SCRIPT}" "${args[@]}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[chain] ${label} shard=${shard} gpu=${gpu} pid=${pids[-1]}"
    sleep 2
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[chain] ERROR ${label} failed" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -100 "${f}" || true
    done
    exit 1
  fi
  "${PY}" "${MERGE}" "${out}" | tee "${out}/logs/merge.log"
  echo "[chain] done ${label} $(date -Is)"
}

wait_existing_and_merge "ID_pairdist_epoch149"
launch_and_wait "x1_LP_102M_epoch099" "model" "/home1/irteam/runs/training/x1_LP_102M/ckpt_epochepoch=099.ckpt"
launch_and_wait "B1_adsorbate_only_pair_dist" "model" "/home1/irteam/runs/B1_adsorbate_only_pair_dist_loss/last.ckpt"
launch_and_wait "baseline2_random_heuristic" "random"

"${PY}" - <<PY
import json, pathlib
root = pathlib.Path("${ROOT}")
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
        "final_delta_mean": (d.get("final_delta_E_sys") or {}).get("mean"),
    })
(root / "combined_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
print(json.dumps(rows, indent=2, sort_keys=True))
PY
echo "[chain] all done ROOT=${ROOT}"
