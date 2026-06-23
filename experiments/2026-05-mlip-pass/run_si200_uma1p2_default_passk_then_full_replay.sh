#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
PASS_SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
FULL_REPLAY="${REPO}/geoopt/two_stage_full_replay.py"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/si200_uma1p2_defaultlbfgs_passk_${STAMP}}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

LMDB="${LMDB:-/home1/irteam/data/uma_s_1p2_references/processed/oc20dense_unwrap_centered.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data/uma_s_1p2_references/materialized/oc20dense_global_min_cover}"
BASELINE_COMBINED="${BASELINE_COMBINED:-/home1/irteam/data-vol1/minkyu/data/replay/valid_passk_uma1p2_two_lbfgs_20260617_224056/combined_summary.json}"

UNIFORM_CKPT="${UNIFORM_CKPT:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_uniform/ckpt_epochepoch=199.ckpt}"
BETA_CKPT="${BETA_CKPT:-/home1/irteam/runs/training/x1_SI_vloss_eta_102M_sigma0p1_w0p5_uma1p2_beta2_1/ckpt_epochepoch=199.ckpt}"

ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
TRAIN_LMDB="${TRAIN_LMDB:-/home1/irteam/data/uma_s_1p2_references/processed/is2res_train_unwrap_centered.lmdb}"
VAL_LMDB="${VAL_LMDB:-/home1/irteam/data/uma_s_1p2_references/processed/is2res_val_unwrap_centered.lmdb}"
FULL_SELECTED="${FULL_SELECTED:-/home1/irteam/data/uma_s_1p2_references/selected_systems/full_train_val_id_uma1p2_all_x5_seed20260526.json}"
FULL_OUT_DIR="${FULL_OUT_DIR:-/home1/irteam/data/replay/full_replay_si200_uma1p2_defaultlbfgs_x5_${STAMP}}"

GPUS=(${GPUS:-0 1 2 3 4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-8}"
MAX_ATOMS="${MAX_ATOMS:-32768}"
FLOW_BATCH_SIZE="${FLOW_BATCH_SIZE:-64}"
SP_CHUNK_JOBS="${SP_CHUNK_JOBS:-128}"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for f in "${PASS_SCRIPT}" "${MERGE}" "${FULL_REPLAY}" "${LMDB}" "${SELECTED}" "${COVER}/gt_results/global_minima.json" "${UNIFORM_CKPT}" "${BETA_CKPT}" "${ADSORBATES_PKL}" "${TRAIN_LMDB}" "${VAL_LMDB}"; do
  if [[ ! -s "${f}" ]]; then
    echo "[chain] missing required file: ${f}" >&2
    exit 2
  fi
done

echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"
echo "[chain] out=${OUT_ROOT}" | tee -a "${LOG_DIR}/launcher.log"
echo "[chain] lmdb=${LMDB}" | tee -a "${LOG_DIR}/launcher.log"
echo "[chain] cover=${COVER}" | tee -a "${LOG_DIR}/launcher.log"

run_passk_one() {
  local label="$1"
  local ckpt="$2"
  local time_schedule="$3"
  local out="${OUT_ROOT}/ase_default_fmax0p05_maxstep0p2_mem100/${label}"
  mkdir -p "${out}/logs"
  echo "[chain] passk start label=${label} ckpt=${ckpt} sampler=sde_heun200_eps0p01 time_schedule=${time_schedule} $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"

  local pids=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    local gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
    local log="${out}/logs/shard_${shard}.log"
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${PASS_SCRIPT}" \
        --mode model \
        --label "${label}" \
        --ckpt "${ckpt}" \
        --out-dir "${out}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --lmdb "${LMDB}" \
        --selected-systems "${SELECTED}" \
        --cover-dir "${COVER}" \
        --num-samples 100 \
        --sample-mode sde \
        --solver heun \
        --sde-mode omatg_si \
        --si-gamma-schedule sqrt_t1mt \
        --si-gamma-sigma 0.1 \
        --si-epsilon-schedule vanishing_1mt \
        --si-epsilon-scale 0.01 \
        --time-schedule "${time_schedule}" \
        --time-schedule-beta 2.0 \
        --flow-steps 200 \
        --flow-batch-size "${FLOW_BATCH_SIZE}" \
        --prior-mode random_heuristic \
        --uma-model uma-s-1p2 \
        --uma-task oc20 \
        --fmax 0.05 \
        --max-steps 300 \
        --max-atoms "${MAX_ATOMS}" \
        --sp-chunk-jobs "${SP_CHUNK_JOBS}" \
        --maxstep 0.2 \
        --lbfgs-memory 100 \
        --lbfgs-damping 1.0 \
        --lbfgs-alpha 70.0 \
        --epsilon-succ 0.1 \
        --lbfgs-streaming \
        --lbfgs-check-interval 10
    ) > "${log}" 2>&1 &
    pids+=("$!")
    echo "$!" > "${out}/logs/pid_shard${shard}.txt"
    echo "[chain] passk ${label} shard=${shard} gpu=${gpu} pid=${pids[-1]}" | tee -a "${LOG_DIR}/launcher.log"
    sleep 2
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[chain] passk failed label=${label}; see ${out}/logs" | tee -a "${LOG_DIR}/launcher.log"
    exit 1
  fi
  "${PY}" "${MERGE}" "${out}" --num-samples 100 | tee "${out}/logs/merge.log"
  echo "[chain] passk done label=${label} $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"
}

write_combined_and_best() {
  "${PY}" - "${OUT_ROOT}" "${BASELINE_COMBINED}" "${UNIFORM_CKPT}" "${BETA_CKPT}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
baseline_path = pathlib.Path(sys.argv[2])
uniform_ckpt = sys.argv[3]
beta_ckpt = sys.argv[4]

rows = []
if baseline_path.exists():
    for r in json.loads(baseline_path.read_text()):
        if r.get("setting") == "ase_default_fmax0p05_maxstep0p2_mem100":
            rows.append(r)

for p in sorted(root.glob("ase_default_fmax0p05_maxstep0p2_mem100/*/summary.json")):
    d = json.loads(p.read_text())
    rows.append({
        "setting": "ase_default_fmax0p05_maxstep0p2_mem100",
        "method": p.parent.name,
        "valid_rate": d.get("valid_rate"),
        "converged_rate": d.get("converged_rate"),
        "valid_pre_gap_mean": (d.get("valid_sp_delta_E_sys") or {}).get("mean"),
        "valid_pre_gap_median": (d.get("valid_sp_delta_E_sys") or {}).get("median"),
        "valid_post_gap_mean": (d.get("valid_final_delta_E_sys") or {}).get("mean"),
        "valid_post_gap_median": (d.get("valid_final_delta_E_sys") or {}).get("median"),
        "valid_steps_mean": (d.get("valid_n_steps") or {}).get("mean"),
        "valid_steps_median": (d.get("valid_n_steps") or {}).get("median"),
        "valid_success_rate": d.get("valid_success_rate"),
        "valid_success_systems": d.get("valid_success_systems"),
        "valid_mlip_pass@1": d.get("valid_mlip_pass@1"),
        "valid_mlip_pass@2": d.get("valid_mlip_pass@2"),
        "valid_mlip_pass@5": d.get("valid_mlip_pass@5"),
        "valid_mlip_pass@10": d.get("valid_mlip_pass@10"),
        "elapsed_sec": d.get("elapsed_sec"),
        "throughput": d.get("throughput_per_shard"),
        "settings": d.get("settings"),
    })

(root / "combined_with_baselines_defaultlbfgs.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

def fmt(x, pct=False):
    if x is None:
        return "NA"
    x = float(x)
    return f"{x*100:.2f}%" if pct else f"{x:.4g}"

lines = [
    "# UMA-s-1p2 + ASE default LBFGS pass@k",
    "",
    "| method | valid | conv | pre-gap mean | post-gap mean | steps mean | succ systems | pass@1 | pass@2 | pass@5 | pass@10 |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['method']} | {fmt(r.get('valid_rate'), True)} | {fmt(r.get('converged_rate'), True)} | "
        f"{fmt(r.get('valid_pre_gap_mean'))} | {fmt(r.get('valid_post_gap_mean'))} | {fmt(r.get('valid_steps_mean'))} | "
        f"{r.get('valid_success_systems')} | {fmt(r.get('valid_mlip_pass@1'), True)} | {fmt(r.get('valid_mlip_pass@2'), True)} | "
        f"{fmt(r.get('valid_mlip_pass@5'), True)} | {fmt(r.get('valid_mlip_pass@10'), True)} |"
    )
(root / "combined_with_baselines_defaultlbfgs.md").write_text("\n".join(lines) + "\n")

new = [r for r in rows if r.get("method") in {"si_uniform_epoch199", "si_beta2_1_epoch199"}]
if not new:
    raise SystemExit("no SI rows found")

def neg(x, default=1.0e30):
    return -float(x if x is not None else default)

def pos(x, default=-1.0):
    return float(x if x is not None else default)

def rank_tuple(r):
    # Full replay model selection: compare pass@k, pre/post relaxation quality,
    # relaxation cost, and validity/convergence. Higher tuple is better.
    return (
        pos(r.get("valid_mlip_pass@10")),
        pos(r.get("valid_mlip_pass@5")),
        pos(r.get("valid_mlip_pass@2")),
        pos(r.get("valid_mlip_pass@1")),
        pos(r.get("valid_success_systems")),
        neg(r.get("valid_post_gap_mean")),
        neg(r.get("valid_pre_gap_mean")),
        neg(r.get("valid_steps_mean")),
        pos(r.get("valid_rate")),
        pos(r.get("converged_rate")),
    )

best = max(new, key=rank_tuple)
best_ckpt = uniform_ckpt if best["method"] == "si_uniform_epoch199" else beta_ckpt
best_schedule = "uniform" if best["method"] == "si_uniform_epoch199" else "beta_train"
(root / "best_si_for_full_replay.json").write_text(json.dumps({
    "method": best["method"],
    "ckpt": best_ckpt,
    "time_schedule": best_schedule,
    "sampler": "sde_heun200_eps0p01",
    "selection_rule": [
        "valid_mlip_pass@10 desc",
        "valid_mlip_pass@5 desc",
        "valid_mlip_pass@2 desc",
        "valid_mlip_pass@1 desc",
        "valid_success_systems desc",
        "valid_post_gap_mean asc",
        "valid_pre_gap_mean asc",
        "valid_steps_mean asc",
        "valid_rate desc",
        "converged_rate desc",
    ],
    "selection_rank": list(rank_tuple(best)),
    "candidate_ranks": {r["method"]: list(rank_tuple(r)) for r in new},
    "row": best,
}, indent=2, sort_keys=True))
print("\n".join(lines), flush=True)
print(json.dumps({"best_method": best["method"], "best_ckpt": best_ckpt, "time_schedule": best_schedule}, indent=2), flush=True)
PY
}

build_full_selected_if_needed() {
  if [[ -s "${FULL_SELECTED}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${FULL_SELECTED}")"
  "${PY}" - "${FULL_SELECTED}" "${TRAIN_LMDB}" "${VAL_LMDB}" <<'PY'
import json
import pickle
import sys
from pathlib import Path

import lmdb

out = Path(sys.argv[1])
lmdbs = sys.argv[2:]

def freeze(x):
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    return x

def jsonable(x):
    if isinstance(x, tuple):
        return [jsonable(v) for v in x]
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return x

systems = []
seen = set()
for lmdb_id, path in enumerate(lmdbs):
    env = lmdb.open(path, subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        n = int(pickle.loads(txn.get(b"length")))
        for i in range(n):
            e = pickle.loads(txn.get(str(i).encode()))
            key = freeze(e.get("system_key", e.get("sid", i)))
            if key in seen:
                continue
            seen.add(key)
            systems.append({
                "lmdb_id": int(lmdb_id),
                "raw_idx": int(i),
                "sid": int(e.get("sid", i)),
                "system_key": jsonable(key),
                "E_sys_ref": float(e.get("mlip_e_total", e.get("y_relaxed", e.get("y", 0.0)))),
            })
    env.close()

payload = {
    "seed": 20260526,
    "num_placements": 5,
    "num_systems": len(systems),
    "num_eligible_unique_systems": len(systems),
    "systems": systems,
    "source_lmdbs": lmdbs,
    "source": "uma-s-1p2 relabeled train+val unique system_key",
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(out, len(systems), flush=True)
PY
}

launch_full_replay() {
  build_full_selected_if_needed
  local best_ckpt
  best_ckpt="$("${PY}" - "${OUT_ROOT}/best_si_for_full_replay.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["ckpt"])
PY
)"
  local time_schedule
  time_schedule="$("${PY}" - "${OUT_ROOT}/best_si_for_full_replay.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["time_schedule"])
PY
)"
  local total_systems
  total_systems="$("${PY}" - "${FULL_SELECTED}" <<'PY'
import json, sys
print(len(json.load(open(sys.argv[1]))["systems"]))
PY
)"
  mkdir -p "${FULL_OUT_DIR}/logs"
  echo "[chain] full replay start ckpt=${best_ckpt} sampler=sde_heun200_eps0p01 time_schedule=${time_schedule} systems=${total_systems} out=${FULL_OUT_DIR} $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 5 > "${FULL_OUT_DIR}/gpu_monitor.csv" &
    echo "$!" > "${FULL_OUT_DIR}/logs/gpu_monitor.pid"
  fi

  "${PY}" "${FULL_REPLAY}" generate \
    --repo "${REPO}" \
    --adsorbates-pkl "${ADSORBATES_PKL}" \
    --ckpt "${best_ckpt}" \
    --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
    --selected-systems "${FULL_SELECTED}" \
    --out-dir "${FULL_OUT_DIR}" \
    --gpus "${GPUS[@]}" \
    --total-systems "${total_systems}" \
    --system-offset 0 \
    --shard-systems 256 \
    --num-placements 5 \
    --flow-batch-size 64 \
    --flow-steps 200 \
    --sample-mode sde \
    --solver heun \
    --sde-mode omatg_si \
    --si-gamma-schedule sqrt_t1mt \
    --si-gamma-sigma 0.1 \
    --si-epsilon-schedule vanishing_1mt \
    --si-epsilon-scale 0.01 \
    --time-schedule "${time_schedule}" \
    --time-schedule-beta 2.0 \
    --prior-mode random_heuristic 2>&1 | tee "${FULL_OUT_DIR}/logs/generate.log"

  "${PY}" "${FULL_REPLAY}" relax \
    --repo "${REPO}" \
    --adsorbates-pkl "${ADSORBATES_PKL}" \
    --ckpt "${best_ckpt}" \
    --train-lmdb "${TRAIN_LMDB}" "${VAL_LMDB}" \
    --selected-systems "${FULL_SELECTED}" \
    --out-dir "${FULL_OUT_DIR}" \
    --gpus "${GPUS[@]}" \
    --uma-model uma-s-1p2 \
    --uma-task oc20 \
    --fmax 0.05 \
    --max-steps 300 \
    --max-atoms 32768 \
    --maxstep 0.2 \
    --lbfgs-memory 100 \
    --lbfgs-damping 1.0 \
    --lbfgs-alpha 70.0 \
    --lbfgs-streaming \
    --lbfgs-check-interval 10 \
    --save-result-pkl 2>&1 | tee "${FULL_OUT_DIR}/logs/relax.log"

  echo "[chain] full replay done out=${FULL_OUT_DIR} $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"
}

run_passk_one "si_uniform_epoch199" "${UNIFORM_CKPT}" "uniform"
run_passk_one "si_beta2_1_epoch199" "${BETA_CKPT}" "beta_train"
write_combined_and_best
launch_full_replay

echo "[chain] all done $(date -Is)" | tee -a "${LOG_DIR}/launcher.log"
