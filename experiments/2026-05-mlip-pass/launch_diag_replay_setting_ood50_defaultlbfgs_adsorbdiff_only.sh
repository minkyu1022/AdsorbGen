#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home1/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
SCRIPT="${REPO}/experiments/2026-05-mlip-pass/diag_replay_setting_ood50.py"
MERGE="${REPO}/experiments/2026-05-mlip-pass/merge_diag_replay_setting_ood50.py"
OUT_ROOT="${OUT_ROOT:-/home1/irteam/data/replay/diag_replay_setting_ood50_defaultLBFGS_adsorbdiff_$(date +%Y%m%d_%H%M%S)}"
ADSORBATES_PKL="${ADSORBATES_PKL:-/home1/irteam/data-vol1/minkyu/data/pkls/adsorbates.pkl}"
LMDB="${LMDB:-/home1/irteam/data/processed_old/oc20dense.lmdb}"
SELECTED="${SELECTED:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50/selected_ood50_systems.json}"
COVER="${COVER:-/home1/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
ADSORBDIFF_METADATA="${ADSORBDIFF_METADATA:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff_score32_20260615_010951/metadata.json}"
ADSORBDIFF_LMDB="${ADSORBDIFF_LMDB:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/adsorbdiff_ood50_100.lmdb}"
ADSORBDIFF_RESULTS_DIR="${ADSORBDIFF_RESULTS_DIR:-/home1/irteam/data/replay/mlip_pass_lbfgs_ood50_baseline3_adsorbdiff/results_setsid_20260615_001248}"
GPUS=(${GPUS:-4 5 6 7})
NUM_SHARDS="${NUM_SHARDS:-4}"
LABEL="${LABEL:-B3_adsorbdiff_defaultlbfgs}"

mkdir -p "${OUT_ROOT}/${LABEL}/logs" "${OUT_ROOT}/logs"
echo "${OUT_ROOT}" > "${OUT_ROOT}/OUT_ROOT.txt"

export ADSGEN_ROOT="${REPO}"
export ADSORBATES_PKL
export PYTHONPATH="${REPO}:${REPO}/geoopt:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

common_args=(
  --mode adsorbdiff
  --label "${LABEL}"
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
  --adsorbdiff-metadata "${ADSORBDIFF_METADATA}"
  --adsorbdiff-lmdb "${ADSORBDIFF_LMDB}"
  --adsorbdiff-results-dir "${ADSORBDIFF_RESULTS_DIR}"
)

out="${OUT_ROOT}/${LABEL}"
echo "[diag-defaultlbfgs-b3] start ${LABEL} OUT_ROOT=${OUT_ROOT} $(date -Is)"
echo "[diag-defaultlbfgs-b3] results=${ADSORBDIFF_RESULTS_DIR}"
pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPUS[$((shard % ${#GPUS[@]}))]}"
  log="${out}/logs/shard_${shard}.log"
  (
    cd "${REPO}"
    exec env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT}" \
      --out-dir "${out}" --shard-idx "${shard}" --num-shards "${NUM_SHARDS}" "${common_args[@]}"
  ) >"${log}" 2>&1 &
  pids+=("$!")
  echo "$!" > "${out}/logs/pid_shard${shard}.txt"
  echo "[diag-defaultlbfgs-b3] shard=${shard} gpu=${gpu} pid=${pids[-1]}"
  sleep 2
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[diag-defaultlbfgs-b3] ERROR logs=${out}/logs" >&2
  for f in "${out}"/logs/shard_*.log; do
    echo "==== ${f} ===="
    tail -120 "${f}" || true
  done
  exit 1
fi

"${PYTHON_BIN}" "${MERGE}" "${out}" | tee "${out}/logs/merge.log"
echo "[diag-defaultlbfgs-b3] done ${LABEL} $(date -Is)"
