#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
RUN_DIR="${RUN_DIR:-/home1/irteam/runs/training/base_gaussian_adsprior_102M}"
LAUNCH="${REPO}/experiments/2026-05-mlip-pass/launch_diag_replay_setting_ood50_defaultmaxstep_base_random_adsorbdiff.sh"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/diag_replay_setting_ood50_defaultLBFGS_valid_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_JSON="${OUT_ROOT}/valid_only_summary.json"
SUMMARY_TXT="${OUT_ROOT}/valid_only_summary.md"

resume_training() {
  if pgrep -af "adsorbgen.training.train_cli .*--out ${RUN_DIR}" >/dev/null 2>&1; then
    echo "[valid-diag] training already running; skip resume $(date -Is)"
    return 0
  fi
  echo "[valid-diag] resume training ${RUN_DIR} $(date -Is)"
  (
    cd "${RUN_DIR}"
    nohup bash "${RUN_DIR}/launch_command.sh" > "${RUN_DIR}/train.log" 2>&1 &
    echo "$!" > "${RUN_DIR}/pid.txt"
  )
}

trap resume_training EXIT

echo "[valid-diag] stop training ${RUN_DIR} $(date -Is)"
mapfile -t pids < <(pgrep -f "adsorbgen.training.train_cli .*--out ${RUN_DIR}" || true)
if ((${#pids[@]})); then
  kill "${pids[@]}" || true
  sleep 10
  mapfile -t live < <(pgrep -f "adsorbgen.training.train_cli .*--out ${RUN_DIR}" || true)
  if ((${#live[@]})); then
    kill -9 "${live[@]}" || true
  fi
fi

export OUT_ROOT
export GPUS="4 5 6 7"
export NUM_SHARDS=4
export ADSORBDIFF_RESULTS_DIR="/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/results_setsid_20260615_001248"
export BASE_CKPT="/home1/irteam/runs/training/base/base.ckpt"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[valid-diag] start replay-setting valid diagnostic OUT_ROOT=${OUT_ROOT} $(date -Is)"
bash "${LAUNCH}"

"${PYTHON_BIN}" - <<PY
from __future__ import annotations
import json, math, pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

root = Path("${OUT_ROOT}")
methods = {
    "base": root / "base_ID_pairdist_epoch149_defaultmaxstep",
    "random": root / "B2_random_heuristic_defaultmaxstep",
    "adsorbdiff": root / "B3_adsorbdiff_defaultmaxstep",
}

def load_rows(path: Path):
    rows = []
    for p in sorted(path.glob("shard_*.pkl")):
        with p.open("rb") as f:
            payload = pickle.load(f)
        rows.extend(payload["rows"] if isinstance(payload, dict) else payload)
    return rows

def stats(rows, key):
    vals = []
    for r in rows:
        v = r.get(key)
        try:
            v = float(v)
        except Exception:
            continue
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
    }

def best_by(rows, selector):
    by = defaultdict(list)
    for r in rows:
        by[str(r["system_key"])].append(r)
    out = []
    for rs in by.values():
        cand = [r for r in rs if r.get("valid")]
        if not cand:
            continue
        out.append(min(cand, key=selector))
    return out

def pack(name, rows):
    valid = [r for r in rows if r.get("valid")]
    conv = [r for r in rows if r.get("converged")]
    systems = sorted({str(r["system_key"]) for r in rows})
    valid_systems = sorted({str(r["system_key"]) for r in valid})
    return {
        "method": name,
        "n": len(rows),
        "systems": len(systems),
        "valid_n": len(valid),
        "valid_systems": len(valid_systems),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "converged_rate": len(conv) / len(rows) if rows else 0.0,
        "invalid_top": Counter(str(r.get("anomaly") or r.get("status")) for r in rows if not r.get("valid")).most_common(10),
        "all": {
            "sp_delta_E_sys": stats(rows, "sp_delta_E_sys"),
            "final_delta_E_sys": stats(rows, "final_delta_E_sys"),
            "n_steps": stats(rows, "n_steps"),
        },
        "valid": {
            "sp_delta_E_sys": stats(valid, "sp_delta_E_sys"),
            "final_delta_E_sys": stats(valid, "final_delta_E_sys"),
            "n_steps": stats(valid, "n_steps"),
        },
        "valid_best_by_pre_gap": {
            "rows": len(best_by(rows, lambda r: float(r["sp_delta_E_sys"]))),
            "sp_delta_E_sys": stats(best_by(rows, lambda r: float(r["sp_delta_E_sys"])), "sp_delta_E_sys"),
            "final_delta_E_sys": stats(best_by(rows, lambda r: float(r["sp_delta_E_sys"])), "final_delta_E_sys"),
            "n_steps": stats(best_by(rows, lambda r: float(r["sp_delta_E_sys"])), "n_steps"),
        },
        "valid_best_by_post_gap": {
            "rows": len(best_by(rows, lambda r: float(r["final_delta_E_sys"]))),
            "sp_delta_E_sys": stats(best_by(rows, lambda r: float(r["final_delta_E_sys"])), "sp_delta_E_sys"),
            "final_delta_E_sys": stats(best_by(rows, lambda r: float(r["final_delta_E_sys"])), "final_delta_E_sys"),
            "n_steps": stats(best_by(rows, lambda r: float(r["final_delta_E_sys"])), "n_steps"),
        },
        "valid_best_by_steps": {
            "rows": len(best_by(rows, lambda r: (int(r["n_steps"]), float(r["final_delta_E_sys"])))),
            "sp_delta_E_sys": stats(best_by(rows, lambda r: (int(r["n_steps"]), float(r["final_delta_E_sys"]))), "sp_delta_E_sys"),
            "final_delta_E_sys": stats(best_by(rows, lambda r: (int(r["n_steps"]), float(r["final_delta_E_sys"]))), "final_delta_E_sys"),
            "n_steps": stats(best_by(rows, lambda r: (int(r["n_steps"]), float(r["final_delta_E_sys"]))), "n_steps"),
        },
    }

summary = [pack(name, load_rows(path)) for name, path in methods.items()]
Path("${SUMMARY_JSON}").write_text(json.dumps(summary, indent=2, sort_keys=True))

lines = []
lines.append(f"# Valid-only default-LBFGS diagnostic\\n")
lines.append(f"OUT_ROOT: {root}\\n")
lines.append("| method | valid rate | valid n | valid systems | valid pre-gap mean | valid post-gap mean | valid steps mean | conv rate |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for s in summary:
    lines.append(
        f"| {s['method']} | {s['valid_rate']:.4f} | {s['valid_n']} | {s['valid_systems']} | "
        f"{s['valid']['sp_delta_E_sys']['mean']:.6g} | {s['valid']['final_delta_E_sys']['mean']:.6g} | "
        f"{s['valid']['n_steps']['mean']:.6g} | {s['converged_rate']:.4f} |"
    )
lines.append("\\n## Best-of-100 among valid candidates\\n")
for key, title in [
    ("valid_best_by_pre_gap", "Best by pre-relax gap"),
    ("valid_best_by_post_gap", "Best by post-relax gap"),
    ("valid_best_by_steps", "Best by step count"),
]:
    lines.append(f"### {title}\\n")
    lines.append("| method | systems covered | pre-gap mean | post-gap mean | steps mean |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in summary:
        b = s[key]
        lines.append(
            f"| {s['method']} | {b['rows']} | {b['sp_delta_E_sys']['mean']:.6g} | "
            f"{b['final_delta_E_sys']['mean']:.6g} | {b['n_steps']['mean']:.6g} |"
        )
    lines.append("")
Path("${SUMMARY_TXT}").write_text("\\n".join(lines))
print(Path("${SUMMARY_TXT}").read_text())
PY

echo "[valid-diag] done OUT_ROOT=${OUT_ROOT} $(date -Is)"
