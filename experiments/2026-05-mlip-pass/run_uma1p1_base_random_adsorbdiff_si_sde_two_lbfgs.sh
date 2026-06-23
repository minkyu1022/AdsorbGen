#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/uma1p1_base_random_adsorbdiff_si_sde_two_lbfgs_$(date +%Y%m%d_%H%M%S)}"

LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
BASE_CKPT="${BASE_CKPT:-/home1/irteam/runs/training/base/base.ckpt}"
SI_CKPT="${SI_CKPT:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5/ckpt_epochepoch=139.ckpt}"
ADSORBDIFF_METADATA="${ADSORBDIFF_METADATA:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/metadata.json}"
ADSORBDIFF_LMDB="${ADSORBDIFF_LMDB:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/adsorbdiff_ood50_100.lmdb}"
ADSORBDIFF_RESULTS_DIR="${ADSORBDIFF_RESULTS_DIR:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/results_setsid_20260615_001248}"

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
SKIP_TRAINING_CONTROL="${SKIP_TRAINING_CONTROL:-0}"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

stop_training() {
  echo "[uma1p1-passk] stop current training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
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
  echo "[uma1p1-passk] resume training $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
  if [[ -x "${SI_LAUNCH}" ]] && ! pgrep -af "${SI_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${SI_LAUNCH}" > "${SI_RUN_DIR}/train.log" 2>&1 &
    echo "[uma1p1-passk] resumed SI pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
  fi
  if [[ -x "${GAUSS_LAUNCH}" ]] && ! pgrep -af "${GAUSS_RUN_DIR}.*train_cli" >/dev/null; then
    nohup "${GAUSS_LAUNCH}" > "${GAUSS_RUN_DIR}/train.log" 2>&1 &
    echo "[uma1p1-passk] resumed gaussian pid=$!" | tee -a "${OUT_ROOT}/logs/launcher.log"
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
  local flow_steps="$7"
  local ckpt="${8:-}"
  local out="${OUT_ROOT}/${setting}/${label}"
  mkdir -p "${out}/logs"
  echo "[uma1p1-passk] start setting=${setting} label=${label} mode=${mode} fmax=${fmax} maxstep=${maxstep} memory=${memory} flow_steps=${flow_steps} $(date -Is)" \
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
      --num-samples "${NUM_SAMPLES}"
      --flow-steps "${flow_steps}"
      --flow-batch-size "${FLOW_BATCH_SIZE}"
      --prior-mode random_heuristic
      --uma-model uma-s-1p1
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
    if [[ "${label}" == "si_sde200_eps0p01" ]]; then
      args+=(
        --sample-mode sde
        --sde-mode omatg_si
        --sde-schedule atommof
        --sde-alpha 1.0
        --si-gamma-schedule sqrt_t1mt
        --si-gamma-sigma 0.1
        --si-epsilon-schedule vanishing_1mt
        --si-epsilon-scale 0.01
      )
    fi
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" "${args[@]}"
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[uma1p1-passk] ${setting}/${label} shard=${shard} gpu=${gpu} pid=${pids[-1]}" \
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
    echo "[uma1p1-passk] ERROR setting=${setting} label=${label}; logs=${out}/logs" >&2
    for f in "${out}"/logs/shard_*.log; do
      echo "==== ${f} ===="
      tail -120 "${f}" || true
    done
    exit 1
  fi
  "${PYTHON_BIN}" "${MERGE}" "${out}" --num-samples "${NUM_SAMPLES}" | tee "${out}/logs/merge.log"
  echo "[uma1p1-passk] done setting=${setting} label=${label} $(date -Is)" \
    | tee -a "${OUT_ROOT}/logs/launcher.log"
}

write_combined_paper_style() {
  "${PYTHON_BIN}" - "${OUT_ROOT}" "${NUM_SAMPLES}" <<'PY'
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
expected_n = int(sys.argv[2])
ks = (1, 2, 5, 10)

def pass_at_k(n, c, k):
    if c <= 0:
        return 0.0
    if k > n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

def finite(x):
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def stats(vals):
    vals = [float(v) for v in vals if finite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)

rows_out = []
for summary_path in sorted(root.glob("*/*/summary.json")):
    out_dir = summary_path.parent
    summary = json.loads(summary_path.read_text())
    rows = []
    all_rows = out_dir / "all_rows.pkl"
    if all_rows.exists():
        payload = pickle.load(all_rows.open("rb"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    else:
        for shard in sorted(out_dir.glob("shard_*.pkl")):
            payload = pickle.load(shard.open("rb"))
            rows.extend(payload.get("rows", payload) if isinstance(payload, dict) else payload)

    by_system = defaultdict(list)
    for r in rows:
        by_system[str(r["system_key"])].append(r)

    pvals = {k: [] for k in ks}
    success_systems = 0
    for rs in by_system.values():
        n = len(rs) or expected_n
        c = sum(1 for r in rs if bool(r.get("success")))
        if c > 0:
            success_systems += 1
        for k in ks:
            pvals[k].append(pass_at_k(n, c, k))

    item = {
        "setting": summary_path.parents[1].name,
        "method": summary_path.parent.name,
        "systems": len(by_system),
        "candidates": len(rows),
        "valid_rate": summary.get("valid_rate"),
        "converged_rate": summary.get("converged_rate"),
        "success_systems": success_systems,
        "paper_mlip_pass@1": stats(pvals[1]),
        "paper_mlip_pass@2": stats(pvals[2]),
        "paper_mlip_pass@5": stats(pvals[5]),
        "paper_mlip_pass@10": stats(pvals[10]),
        "valid_pool_mlip_pass@10": summary.get("valid_mlip_pass@10"),
        "valid_pre_gap_mean_eV": (summary.get("valid_sp_delta_E_sys") or {}).get("mean"),
        "valid_post_gap_mean_eV": (summary.get("valid_final_delta_E_sys") or {}).get("mean"),
        "valid_steps_mean": (summary.get("valid_n_steps") or {}).get("mean"),
        "throughput_cand_per_sec": (summary.get("throughput_8gpu") or {}).get("post_relax_candidates_per_sec"),
        "status_counts": summary.get("status_counts"),
        "settings": summary.get("settings"),
    }
    rows_out.append(item)

(root / "combined_paper_style.json").write_text(json.dumps(rows_out, indent=2, sort_keys=True))

order = {
    "base": 0,
    "random": 1,
    "adsorbdiff": 2,
    "si_sde200_eps0p01": 3,
}
rows_out.sort(key=lambda r: (r["setting"], order.get(r["method"], 99)))
lines = ["# UMA-s-1p1 OOD50 comparison", ""]
lines.append("Pass@k uses paper-style denominator: all generated candidates per system.")
lines.append("")
for setting in sorted({r["setting"] for r in rows_out}):
    lines.append(f"## {setting}")
    lines.append("| method | valid % | conv % | p@1 | p@2 | p@5 | p@10 | valid-pool p@10 | success systems | pre-gap | post-gap | steps | cand/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in [x for x in rows_out if x["setting"] == setting]:
        def pct(x):
            return "NA" if x is None else f"{100*float(x):.1f}"
        def num(x):
            return "NA" if x is None else f"{float(x):.2f}"
        lines.append(
            f"| {r['method']} | {pct(r['valid_rate'])} | {pct(r['converged_rate'])} | "
            f"{pct(r['paper_mlip_pass@1'])} | {pct(r['paper_mlip_pass@2'])} | "
            f"{pct(r['paper_mlip_pass@5'])} | {pct(r['paper_mlip_pass@10'])} | "
            f"{pct(r['valid_pool_mlip_pass@10'])} | {r['success_systems']}/{r['systems']} | "
            f"{num(r['valid_pre_gap_mean_eV'])} | {num(r['valid_post_gap_mean_eV'])} | "
            f"{num(r['valid_steps_mean'])} | {num(r['throughput_cand_per_sec'])} |"
        )
    lines.append("")
(root / "combined_paper_style.md").write_text("\n".join(lines))
print("\n".join(lines))
PY
}

if [[ "${SKIP_TRAINING_CONTROL}" != "1" ]]; then
  stop_training
else
  echo "[uma1p1-passk] skip training stop/resume control $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
fi

run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "base" "model" "0.01" "0.04" "50" "50" "${BASE_CKPT}"
run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "random" "random" "0.01" "0.04" "50" "50"
run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "adsorbdiff" "adsorbdiff" "0.01" "0.04" "50" "50"
run_one "pass_strict_fmax0p01_maxstep0p04_mem50" "si_sde200_eps0p01" "model" "0.01" "0.04" "50" "200" "${SI_CKPT}"

run_one "ase_default_fmax0p05_maxstep0p2_mem100" "base" "model" "0.05" "0.2" "100" "50" "${BASE_CKPT}"
run_one "ase_default_fmax0p05_maxstep0p2_mem100" "random" "random" "0.05" "0.2" "100" "50"
run_one "ase_default_fmax0p05_maxstep0p2_mem100" "adsorbdiff" "adsorbdiff" "0.05" "0.2" "100" "50"
run_one "ase_default_fmax0p05_maxstep0p2_mem100" "si_sde200_eps0p01" "model" "0.05" "0.2" "100" "200" "${SI_CKPT}"

write_combined_paper_style
echo "[uma1p1-passk] all done OUT_ROOT=${OUT_ROOT} $(date -Is)" | tee -a "${OUT_ROOT}/logs/launcher.log"
