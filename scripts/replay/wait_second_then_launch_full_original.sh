#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"

SECOND_DIR="${SECOND_DIR:-/home/irteam/runs/self_improve_lbfgs_ID_mlip_pairdist_1x_ep149_second10k_x10_24shards_20260525}"
SECOND_SHARDS="${SECOND_SHARDS:-24}"

FULL_OUT="${FULL_OUT:-/home/irteam/runs/self_improve_lbfgs_ID_mlip_pairdist_1x_ep149_full338500_x5_originalref_20260525}"
FULL_CKPT="${FULL_CKPT:-/home/irteam/runs/ID_mlip_pairdist_only_1x_bs64_expand_20260521_213544/ckpt_epochepoch=149.ckpt}"
FULL_SHARDS="${FULL_SHARDS:-24}"
FULL_GPU_LIST="${FULL_GPU_LIST:-0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7}"
FULL_SYSTEMS="${FULL_SYSTEMS:-338500}"
FULL_PLACEMENTS="${FULL_PLACEMENTS:-5}"
FULL_SEED="${FULL_SEED:-20260525}"
FULL_FLOW_STEPS="${FULL_FLOW_STEPS:-50}"
FULL_FLOW_BATCH_SIZE="${FULL_FLOW_BATCH_SIZE:-32}"
FULL_UMA_FMAX="${FULL_UMA_FMAX:-0.05}"
FULL_UMA_MAX_STEPS="${FULL_UMA_MAX_STEPS:-300}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"

mkdir -p "${FULL_OUT}/logs"
LOG="${FULL_OUT}/logs/wait_second_then_launch_full_original.log"
LAUNCHED_MARK="${FULL_OUT}/logs/full_launch.dispatched"

echo "[$(date '+%F %T')] watcher start second=${SECOND_DIR} full=${FULL_OUT}" >> "${LOG}"

while true; do
  done_count="$(find "${SECOND_DIR}" -maxdepth 1 -name 'shard_*.json' -type f 2>/dev/null | wc -l)"
  echo "[$(date '+%F %T')] second_done=${done_count}/${SECOND_SHARDS}" >> "${LOG}"
  if [[ "${done_count}" -ge "${SECOND_SHARDS}" ]]; then
    break
  fi
  sleep "${INTERVAL_SEC}"
done

if [[ -e "${LAUNCHED_MARK}" ]]; then
  echo "[$(date '+%F %T')] full replay already dispatched; exiting" >> "${LOG}"
  exit 0
fi

if [[ ! -s "${FULL_OUT}/selected_systems.json" ]]; then
  echo "[$(date '+%F %T')] missing ${FULL_OUT}/selected_systems.json" >> "${LOG}"
  exit 2
fi

echo "[$(date '+%F %T')] launching full replay" >> "${LOG}"
(
  cd "${REPO}"
  CKPT="${FULL_CKPT}" \
  OUT_DIR="${FULL_OUT}" \
  NUM_SHARDS="${FULL_SHARDS}" \
  GPU_LIST="${FULL_GPU_LIST}" \
  NUM_SYSTEMS="${FULL_SYSTEMS}" \
  NUM_PLACEMENTS="${FULL_PLACEMENTS}" \
  SEED="${FULL_SEED}" \
  FLOW_STEPS="${FULL_FLOW_STEPS}" \
  FLOW_BATCH_SIZE="${FULL_FLOW_BATCH_SIZE}" \
  UMA_FMAX="${FULL_UMA_FMAX}" \
  UMA_MAX_STEPS="${FULL_UMA_MAX_STEPS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash scripts/replay/launch_self_improve_lbfgs_10k_x10_gpu0_3.sh
) >> "${LOG}" 2>&1

date '+%F %T' > "${LAUNCHED_MARK}"
echo "[$(date '+%F %T')] full replay dispatched" >> "${LOG}"

setsid -f bash -c "
OUT='${FULL_OUT}'
PY='${PYTHON_BIN}'
REPO='${REPO}'
while true; do
  \"\${PY}\" - <<PY
import json, glob, time
out=\"\${OUT}\"
rows=[]
for p in sorted(glob.glob(out + '/logs/progress_shard*.json')):
    try:
        rows.append(json.load(open(p)))
    except Exception:
        pass
cand=sum(int(r.get('candidates', 0)) for r in rows)
target=max([int(r.get('target_candidates', 0)) for r in rows] or [${FULL_SYSTEMS} * ${FULL_PLACEMENTS}])
conv=sum(int(r.get('converged', 0)) for r in rows)
valid=sum(int(r.get('valid', 0)) for r in rows)
succ=sum(int(r.get('success', 0)) for r in rows)
elapsed=max([float(r.get('elapsed_sec', 0.0)) for r in rows] or [0.0])
rate=cand / max(elapsed, 1.0)
eta=(target - cand) / rate if rate > 0 and cand > 0 else None
print(json.dumps({
    'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'progress_files': len(rows),
    'done_shards': len(glob.glob(out + '/shard_*.json')),
    'candidates': cand,
    'target': target,
    'candidate_rate': cand / max(target, 1),
    'converged': conv,
    'valid': valid,
    'success': succ,
    'valid_rate': valid / max(cand, 1),
    'success_rate': succ / max(cand, 1),
    'eta_sec': eta,
}, sort_keys=True))
PY
  done_count=\$(find \"\${OUT}\" -maxdepth 1 -name 'shard_*.json' -type f 2>/dev/null | wc -l)
  if [[ \"\${done_count}\" -ge '${FULL_SHARDS}' ]]; then
    cd \"\${REPO}\" && \"\${PY}\" scripts/replay/merge_self_improve_successes.py --out-dir \"\${OUT}\" >> \"\${OUT}/logs/full_replay_monitor.log\" 2>&1 || true
    break
  fi
  sleep 600
done
" >> "${FULL_OUT}/logs/full_replay_monitor.log" 2>&1
