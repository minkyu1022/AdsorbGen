#!/usr/bin/env bash
set -euo pipefail

PASS_OUT="${PASS_OUT:?PASS_OUT required}"
REPO="${REPO:-/home1/irteam/AdsorbGen}"
PY="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
EG_SCRIPT="${REPO}/experiments/2026-05-mlip-pass/eval_energy_gap_curve.py"
EG_OUT="${EG_OUT:-/home1/irteam/data/replay/energy_gap_curve_train_unique1000_3placements_100epoch_20260614}"
GPUS=(${GPUS:-4 5 6 7})
PASS_NUM_SHARDS="${PASS_NUM_SHARDS:-4}"
LOG="${PASS_OUT}/logs/after_gpu4_7_passk.log"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${PASS_OUT}/logs" "${EG_OUT}/logs"
echo "[gpu4-7-watch] start $(date -Is) PASS_OUT=${PASS_OUT}" >> "${LOG}"

while true; do
  n=$(find "${PASS_OUT}" -maxdepth 1 -name 'shard_*.json' | wc -l)
  running=$(pgrep -fc "eval_mlip_pass_lbfgs_ood50.py.*${PASS_OUT}" || true)
  echo "[gpu4-7-watch] $(date -Is) pass_shards=${n} pass_running=${running}" >> "${LOG}"
  if [[ "${n}" -ge "${PASS_NUM_SHARDS}" ]]; then
    break
  fi
  if [[ "${running}" -eq 0 ]]; then
    echo "[gpu4-7-watch] ERROR pass@k stopped before all shards" >> "${LOG}"
    for f in "${PASS_OUT}"/logs/shard_*.log; do
      echo "==== ${f} ====" >> "${LOG}"
      tail -100 "${f}" >> "${LOG}" || true
    done
    exit 1
  fi
  sleep 60
done

if [[ ! -f "${PASS_OUT}/summary.json" ]]; then
  PYTHONPATH="${REPO}" "${PY}" \
    "${REPO}/experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py" \
    --out-dir "${PASS_OUT}" --num-shards "${PASS_NUM_SHARDS}" --num-samples 100 >> "${LOG}" 2>&1
fi
echo "[gpu4-7-watch] pass@k merged $(date -Is)" >> "${LOG}"

LOCK_DIR="${EG_OUT}/.run_lock"
DONE_FILE="${EG_OUT}/energy_gap_curve.done"
if [[ -f "${DONE_FILE}" ]]; then
  echo "[gpu4-7-watch] energy-gap already complete; skip" >> "${LOG}"
elif mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "[gpu4-7-watch] acquired energy-gap lock; launch on GPUs ${GPUS[*]} $(date -Is)" >> "${LOG}"
  "${PY}" "${EG_SCRIPT}" \
    --mode list \
    --dataset train_unique \
    --out-dir "${EG_OUT}" \
    --max-epoch 99 \
    --seeds 0 \
    --max-samples 1000 \
    --num-placements 3 >> "${LOG}" 2>&1

  pids=()
  for i in "${!GPUS[@]}"; do
    gpu="${GPUS[$i]}"
    wlog="${EG_OUT}/logs/worker_${i}_gpu${gpu}.log"
    (
      cd "${REPO}"
      exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" "${EG_SCRIPT}" \
        --mode worker \
        --dataset train_unique \
        --out-dir "${EG_OUT}" \
        --worker-idx "${i}" \
        --num-workers "${#GPUS[@]}" \
        --max-epoch 99 \
        --seeds 0 \
        --max-samples 1000 \
        --num-placements 3
    ) >"${wlog}" 2>&1 &
    pids+=("$!")
    echo "[gpu4-7-watch] energy worker=${i} gpu=${gpu} pid=${pids[-1]}" >> "${LOG}"
    sleep 2
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[gpu4-7-watch] ERROR energy-gap worker failed" >> "${LOG}"
    exit 1
  fi
  "${PY}" "${EG_SCRIPT}" \
    --mode merge \
    --dataset train_unique \
    --out-dir "${EG_OUT}" \
    --max-epoch 99 \
    --seeds 0 \
    --max-samples 1000 \
    --num-placements 3 >> "${LOG}" 2>&1
  touch "${DONE_FILE}"
  echo "[gpu4-7-watch] energy-gap done ${EG_OUT} $(date -Is)" >> "${LOG}"
else
  echo "[gpu4-7-watch] energy-gap lock exists; skip energy-gap and proceed to replaydiag" >> "${LOG}"
fi

DIAG_OUT=/home1/irteam/data/replay/diag_replay_setting_ood50_relaxed_bare_slab_x0_last_$(date +%Y%m%d_%H%M%S)
mkdir -p "${DIAG_OUT}/logs"
echo "${DIAG_OUT}" > /home1/irteam/data/replay/relaxed_bare_slab_last_replaydiag_active.txt
echo "[gpu4-7-watch] launch replaydiag ${DIAG_OUT} $(date -Is)" >> "${LOG}"
for shard in 0 1 2 3; do
  gpu=$((4 + shard))
  rlog="${DIAG_OUT}/logs/shard_${shard}.log"
  (
    cd "${REPO}"
    exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
      "${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py" \
      --mode model --label relaxed_bare_slab_x0_last \
      --ckpt /home1/irteam/runs/training/x1_LP_102M_relaxed_bare_slab_x0/last.ckpt \
      --out-dir "${DIAG_OUT}" --shard-idx "${shard}" --num-shards 4 \
      --lmdb /home1/irteam/data/processed_old/oc20dense.lmdb \
      --selected-systems /home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json \
      --cover-dir /home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover \
      --num-samples 100 --flow-steps 50 --flow-batch-size 64 --prior-mode random_heuristic \
      --slab-source pristine_relaxed \
      --placement-pristine-slabs /home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl \
      --placement-pristine-index /home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl \
      --uma-model uma-s-1p2 --uma-task oc20 \
      --fmax 0.05 --max-steps 300 --max-atoms 32768 --maxstep 0.04 \
      --lbfgs-memory 50 --lbfgs-damping 1.0 --lbfgs-alpha 70.0 \
      --lbfgs-streaming --lbfgs-check-interval 10
  ) >"${rlog}" 2>&1 &
  echo "$!" > "${DIAG_OUT}/logs/pid_shard${shard}.txt"
  echo "[gpu4-7-watch] replaydiag shard=${shard} gpu=${gpu} pid=$!" >> "${LOG}"
  sleep 2
done

while true; do
  n=$(find "${DIAG_OUT}" -maxdepth 1 -name 'shard_*.json' | wc -l)
  running=$(pgrep -fc "diag_replay_setting_ood50.py.*relaxed_bare_slab_x0_last.*${DIAG_OUT}" || true)
  echo "[gpu4-7-watch] $(date -Is) replaydiag_shards=${n} running=${running}" >> "${LOG}"
  if [[ "${n}" -ge 4 ]]; then
    break
  fi
  if [[ "${running}" -eq 0 ]]; then
    echo "[gpu4-7-watch] ERROR replaydiag stopped early" >> "${LOG}"
    exit 1
  fi
  sleep 60
done
PYTHONPATH="${REPO}:${REPO}/geoopt" "${PY}" \
  "${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py" \
  "${DIAG_OUT}" >> "${LOG}" 2>&1
echo "[gpu4-7-watch] replaydiag merged $(date -Is) DIAG_OUT=${DIAG_OUT}" >> "${LOG}"
