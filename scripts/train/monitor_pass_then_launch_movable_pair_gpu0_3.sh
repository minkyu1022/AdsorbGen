#!/usr/bin/env bash
set -u

PASS_OUT="${PASS_OUT:-/home/irteam/data/replay/mlip_pass_lbfgs_ood50_B1_adsorbate_only_last_20260603_231036}"
REPO="${REPO:-/home/irteam/AdsorbGen}"
PYTHON_BIN="${PYTHON_BIN:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
LOG_DIR="${PASS_OUT}/logs"
SUP_LOG="${LOG_DIR}/pass_5min_then_train_movable_pair.log"
REPORT="${PASS_OUT}/pass_summary_report.txt"
TRAIN_BASE="${TRAIN_BASE:-/home/irteam/runs/training}"
RUN_STEM="${RUN_STEM:-ID_mlip_movable_pairdist_1x}"
WANDB_PROJECT="${WANDB_PROJECT:-adsorbgen}"

mkdir -p "${LOG_DIR}" "${TRAIN_BASE}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $*" | tee -a "${SUP_LOG}"
}

write_pass_report() {
  "${PYTHON_BIN}" - "${PASS_OUT}/summary.json" "${REPORT}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
report_path = Path(sys.argv[2])
d = json.loads(summary_path.read_text())
keys = [
    "systems",
    "candidates",
    "complete_systems",
    "converged_rate",
    "valid_rate",
    "success_sample_rate",
    "mlip_pass@1",
    "mlip_pass@2",
    "mlip_pass@5",
    "mlip_pass@10",
]
lines = [f"summary={summary_path}"]
for k in keys:
    if k in d:
        v = d[k]
        if isinstance(v, float):
            lines.append(f"{k}: {v:.8f}")
        else:
            lines.append(f"{k}: {v}")
report_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
PY
}

run_training() {
  local bs="$1"
  local suffix="$2"
  local out_dir="${TRAIN_BASE}/${RUN_STEM}_bs${bs}_${suffix}_$(date +%Y%m%d_%H%M%S)"
  local train_log="${out_dir}/train.log"
  mkdir -p "${out_dir}"
  log "launch training batch_size=${bs} out=${out_dir}"
  (
    cd "${REPO}" || exit 2
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
    export CAT_BENCH_ROOT=/home/irteam
    export ADSORBATES_PKL=/home/irteam/data/pkls/adsorbates.pkl
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    "${PYTHON_BIN}" -m adsorbgen.training.train_cli \
      --arch v1 \
      --train-lmdb /home/irteam/data/processed_ID/is2res_train.lmdb /home/irteam/data/processed_ID/is2res_val.lmdb \
      --val-lmdb /home/irteam/data/processed_old/oc20dense.lmdb \
      --batch-size "${bs}" \
      --num-workers 4 \
      --epochs 150 \
      --devices 4 \
      --precision bf16-mixed \
      --lr 1e-4 \
      --lr-warmup-steps 5000 \
      --grad-clip 10.0 \
      --loss-type l1 \
      --loss-surf-weight 1.0 \
      --loss-ads-weight 1.0 \
      --ads-pair-l1-weight 0.0 \
      --movable-pair-l1-weight 1.0 \
      --ads-bond-l1-weight 0.0 \
      --ads-nonbonded-clash-weight 0.0 \
      --ads-center-loss-weight 0.0 \
      --ads-rel-pos-loss-weight 0.0 \
      --movable-mode surface_ads \
      --prediction-type x1 \
      --prior-mode random_heuristic \
      --sample-eval-every-epochs 1 \
      --sample-eval-max-samples 1000 \
      --sample-eval-steps 20 \
      --max-val-samples 1000 \
      --check-val-every-n-epoch 1 \
      --save-every-n-epochs 10 \
      --variant v0-ads-ref-adshead \
      --out "${out_dir}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "${RUN_STEM}_bs${bs}"
  ) > "${train_log}" 2>&1
  local rc=$?
  log "training batch_size=${bs} rc=${rc} log=${train_log}"
  echo "${out_dir}" > "${LOG_DIR}/last_movable_pair_train_out.txt"
  echo "${rc}" > "${out_dir}/exit_code.txt"
  return "${rc}"
}

log "supervisor start pass_out=${PASS_OUT}"
while [[ ! -s "${PASS_OUT}/summary.json" ]]; do
  log "pass still running"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >> "${SUP_LOG}" 2>&1 || true
  scan_files=("${LOG_DIR}"/shard_*.log "${LOG_DIR}/launcher.log")
  [[ -s "${LOG_DIR}/merge.log" ]] && scan_files+=("${LOG_DIR}/merge.log")
  rg -n "Traceback|RuntimeError|CUDA out of memory|Killed|Error|Exception|flow\\+lbfgs|merge rc|rc=" \
    "${scan_files[@]}" >> "${SUP_LOG}" 2>&1 || true
  sleep 300
done

log "pass summary detected"
write_pass_report >> "${SUP_LOG}" 2>&1

if [[ -s "${LOG_DIR}/training_launch_done.marker" ]]; then
  log "training already launched; marker exists"
  exit 0
fi
date > "${LOG_DIR}/training_launch_done.marker"

run_training 48 "movablepair"
rc=$?
last_out="$(cat "${LOG_DIR}/last_movable_pair_train_out.txt" 2>/dev/null || true)"
if [[ "${rc}" != "0" ]] && [[ -n "${last_out}" ]] && rg -qi "out of memory|CUDA out of memory|OOM" "${last_out}/train.log"; then
  log "OOM detected for batch_size=48; retry batch_size=32"
  run_training 32 "movablepair_retry_oom"
  exit $?
fi
exit "${rc}"
