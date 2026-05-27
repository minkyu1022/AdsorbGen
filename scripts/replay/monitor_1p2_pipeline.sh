#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/irteam}"
REPO="${REPO:-${ROOT}/AdsorbGen}"
REPLAY_1P2="${REPLAY_1P2:-${ROOT}/data/replay_uma_s_1p2}"
POLL_SEC="${POLL_SEC:-600}"
LOG="${LOG:-${REPLAY_1P2}/monitor_1p2_pipeline.log}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

echo "[monitor-1p2] started $(date -Is)"
echo "[monitor-1p2] replay_dir=${REPLAY_1P2} poll=${POLL_SEC}s"

while true; do
  echo
  echo "========== $(date -Is) =========="

  echo "[gpu]"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits || true

  echo "[processes]"
  pgrep -af 'run_lbfgs_reference_recompute|compute_e_sys_lbfgs|compute_e_slab_lbfgs|wait_1p2_refs|self_improve_lbfgs|run_replay_5000x10' | head -120 || true

  echo "[reference-progress]"
  python - <<'PY' || true
import datetime
import pathlib
import re

base = pathlib.Path("/home/irteam/data/replay_uma_s_1p2/lbfgs_recompute_logs")
now = datetime.datetime.now()

for stage, pat, unit in [
    ("E_sys", "e_sys_lbfgs_shard*.log", "records"),
    ("E_slab", "e_slab_lbfgs_shard*.log", "slabs"),
]:
    rows = []
    for f in sorted(base.glob(pat)):
        txt = f.read_text(errors="ignore")
        assigned = re.search(r"assigned\s+(\d+)\s+/", txt)
        done = re.search(r"DONE\s+(\d+)\s+(?:records|slabs)\s+in\s+(\d+)s", txt)
        vals = [int(m.group(1)) for m in re.finditer(r"(?:records|slabs)=(\d+)", txt)]
        rates = [float(m.group(1)) for m in re.finditer(r"rate=([0-9.]+)/s", txt)]
        count = int(done.group(1)) if done else (vals[-1] if vals else 0)
        rows.append((
            int(assigned.group(1)) if assigned else None,
            count,
            rates[-1] if rates else 0.0,
            bool(done),
        ))
    if not rows:
        print(f"{stage}: not started")
        continue
    total = sum(r[0] or 0 for r in rows)
    count = sum(r[1] for r in rows)
    rate = sum(r[2] for r in rows)
    done = sum(r[3] for r in rows)
    progress = (count / total * 100.0) if total else 0.0
    print(f"{stage}: shards_done={done}/{len(rows)} {unit}={count}/{total} progress={progress:.2f}% rate={rate:.2f}/s")
    if total and rate and done < len(rows):
        eta = (total - count) / rate
        finish = now + datetime.timedelta(seconds=eta)
        print(f"{stage}: ETA={eta/3600:.2f}h finish={finish.isoformat(timespec='minutes')}")
PY

  echo "[required-reference-files]"
  for f in \
    "${REPLAY_1P2}/E_sys_lbfgs_summary.json" \
    "${REPLAY_1P2}/E_slab_only_lbfgs_summary.json" \
    "${REPLAY_1P2}/gt_index_by_sid_oc20_lbfgs.pkl" \
    "${REPLAY_1P2}/gt_index_by_system_oc20_lbfgs.pkl" \
    "${REPLAY_1P2}/E_sys_lbfgs.pkl" \
    "${REPLAY_1P2}/E_slab_only_lbfgs.pkl" \
    "${REPLAY_1P2}/E_slab_only_lbfgs_by_slab.pkl" \
    "${REPLAY_1P2}/pristine_slabs_lbfgs.pkl"; do
    [[ -s "${f}" ]] && echo "OK ${f}" || echo "WAIT ${f}"
  done

  echo "[post-reference-watcher-tail]"
  tail -20 "${REPLAY_1P2}/post_reference_watcher.nohup.log" 2>/dev/null || true
  tail -20 "${REPLAY_1P2}/post_reference_launcher.log" 2>/dev/null || true

  echo "[full-self-improve-progress]"
  full_dir="${ROOT}/runs/self_improvement/self_improve_lbfgs_ID_mlip_pairdist_1x_ep149_full_x10_1p2ref_6gpu_24shards_20260526"
  if [[ -d "${full_dir}" ]]; then
    python "${REPO}/scripts/replay/report_self_improve_progress.py" --run-dir "${full_dir}" 2>/dev/null || true
    tail -20 "${full_dir}/logs/launcher_1p2.nohup.log" 2>/dev/null || true
  else
    echo "not started: ${full_dir}"
  fi

  echo "[sde250-progress]"
  sde_dir="${ROOT}/runs/self_improvement/sde250_H200_catflow_center_rel_oc20dense_20260526"
  if [[ -d "${sde_dir}" ]]; then
    python "${REPO}/scripts/replay/report_replay_5000x10.py" --stream-dir "${sde_dir}" 2>/dev/null || true
    tail -20 "${sde_dir}/logs/launcher.log" 2>/dev/null || true
  else
    echo "not started: ${sde_dir}"
  fi

  sleep "${POLL_SEC}"
done
