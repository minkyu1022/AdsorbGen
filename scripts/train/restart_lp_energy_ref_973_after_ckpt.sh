#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
RUN_ROOT="${RUN_ROOT:-/home1/irteam/runs/training}"
X1_OUT="${X1_OUT:-${RUN_ROOT}/x1_LP_102M}"
RELAXED_OUT="${RELAXED_OUT:-${RUN_ROOT}/x1_LP_102M_relaxed_bare_slab_x0}"
COVER="${COVER:-/home/irteam/data/replay/oc20dense_mlip_global_min_by_system.pkl}"
ADS="${ADS:-/home/irteam/data/pkls/adsorbates.pkl}"
LOG="${LOG:-${RUN_ROOT}/lp_102M_energy_ref_973_restart_watch.log}"
X1_WANDB_RUN_ID="${X1_WANDB_RUN_ID:-baayxl1v}"
RELAXED_WANDB_RUN_ID="${RELAXED_WANDB_RUN_ID:-acoqaaqv}"
PRISTINE_SLABS="${PRISTINE_SLABS:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.pkl}"
PRISTINE_INDEX="${PRISTINE_INDEX:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/is2res.sid_index.pkl}"
VAL_PRISTINE_SLABS="${VAL_PRISTINE_SLABS:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl}"
VAL_PRISTINE_INDEX="${VAL_PRISTINE_INDEX:-/home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl}"

COMMON_ARGS=(
  --arch v1
  --train-lmdb /home/irteam/data/processed_ID/is2res_train.lmdb /home/irteam/data/processed_ID/is2res_val.lmdb
  --val-lmdb /home/irteam/data/processed_old/oc20dense.lmdb
  --batch-size 64
  --num-workers 4
  --epochs 100
  --devices 4
  --precision bf16-mixed
  --lr 1e-4
  --lr-warmup-steps 5000
  --grad-clip 10.0
  --loss-type l1
  --loss-surf-weight 1.0
  --loss-ads-weight 1.0
  --ads-pair-l1-weight 1.0
  --ads-bond-l1-weight 0.0
  --ads-nonbonded-clash-weight 0.0
  --ads-center-loss-weight 0.0
  --ads-rel-pos-loss-weight 0.0
  --movable-mode surface_ads
  --prior-mode random_heuristic
  --sample-eval-every-epochs 1
  --sample-eval-max-samples 1000
  --sample-eval-steps 20
  --sample-eval-energy-cover-dir "${COVER}"
  --sample-eval-energy-uma-model uma-s-1p1
  --sample-eval-energy-uma-task oc20
  --sample-eval-energy-batch-size 32
  --sample-eval-energy-success-margin 0.1
  --max-val-samples 1000
  --check-val-every-n-epoch 1
  --save-every-n-epochs 10
  --variant v0-ads-ref-adshead
  --use-langevin-param
  --langevin-uma-model uma-s-1p2
  --langevin-uma-task oc20
  --wandb-project adsorbgen
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG}"
}

ckpt_mtime() {
  local path="$1/last.ckpt"
  if [[ -f "${path}" ]]; then
    stat -c '%Y' "${path}"
  else
    echo 0
  fi
}

train_pgids_for_out() {
  local out="$1"
  local self_pgid
  self_pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
  ps -eo pid=,pgid=,args= \
    | awk -v out="${out}" -v self_pgid="${self_pgid}" '
      index($0, " -m adsorbgen.training.train_cli") &&
      index($0, "--out") &&
      index($0, out) &&
      $2 != self_pgid {print $2}
    ' \
    | sort -u
}

wait_ckpt_advanced() {
  local out="$1"
  local start="$2"
  local name="$3"
  while true; do
    local now
    now="$(ckpt_mtime "${out}")"
    log "poll ${name}: last.ckpt mtime=${now}, start=${start}"
    if (( now > start )); then
      log "${name}: checkpoint advanced"
      sleep 60
      return
    fi
    sleep 60
  done
}

stop_run_for_out() {
  local out="$1"
  local pgids
  pgids="$(train_pgids_for_out "${out}" || true)"
  if [[ -z "${pgids}" ]]; then
    log "no active train process for ${out}"
    return
  fi
  while read -r pgid; do
    [[ -z "${pgid}" ]] && continue
    log "TERM process group ${pgid} for ${out}"
    kill -TERM "-${pgid}" 2>/dev/null || true
  done <<< "${pgids}"

  for _ in $(seq 1 120); do
    if [[ -z "$(train_pgids_for_out "${out}" || true)" ]]; then
      return
    fi
    sleep 1
  done

  pgids="$(train_pgids_for_out "${out}" || true)"
  while read -r pgid; do
    [[ -z "${pgid}" ]] && continue
    log "KILL process group ${pgid} for ${out}"
    kill -KILL "-${pgid}" 2>/dev/null || true
  done <<< "${pgids}"
}

archive_log() {
  local out="$1"
  local tag="$2"
  if [[ -f "${out}/train.log" ]]; then
    mv "${out}/train.log" "${out}/train_before_energy_ref_973_restart_${tag}.log"
  fi
  rm -f "${out}/exit_code.txt"
}

launch_run() {
  local cuda="$1"
  local out="$2"
  local name="$3"
  local wandb_run_id="$4"
  shift 4
  local extra_args=("$@")
  mkdir -p "${out}"
  (
    cd "${ROOT}" || exit 2
    export CUDA_VISIBLE_DEVICES="${cuda}"
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    export ADSORBATES_PKL="${ADS}"
    export PYTHONUNBUFFERED=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export WANDB_MODE=online
    export WANDB_RUN_ID="${wandb_run_id}"
    export WANDB_RESUME=allow
    export WANDB__SERVICE_WAIT=300
    exec "${PY}" -m adsorbgen.training.train_cli \
      "${COMMON_ARGS[@]}" \
      "${extra_args[@]}" \
      --prediction-type x1 \
      --out "${out}" \
      --wandb-run-name "${name}"
  ) > "${out}/train.log" 2>&1 &
  echo $! > "${out}/launcher_after_energy_ref_973_restart.pid"
  log "launched ${name} on CUDA_VISIBLE_DEVICES=${cuda}, launcher pid $(cat "${out}/launcher_after_energy_ref_973_restart.pid")"
}

main() {
  mkdir -p "$(dirname "${LOG}")"
  log "watcher start; energy cover=${COVER}"
  local x1_start relaxed_start
  x1_start="$(ckpt_mtime "${X1_OUT}")"
  relaxed_start="$(ckpt_mtime "${RELAXED_OUT}")"
  log "initial mtimes: x1=${x1_start}, relaxed=${relaxed_start}"

  wait_ckpt_advanced "${RELAXED_OUT}" "${relaxed_start}" "relaxed_bare_slab_x0"
  wait_ckpt_advanced "${X1_OUT}" "${x1_start}" "x1_LP_102M"

  local tag
  tag="$(date '+%Y%m%d_%H%M%S')"
  stop_run_for_out "${X1_OUT}"
  stop_run_for_out "${RELAXED_OUT}"
  archive_log "${X1_OUT}" "${tag}"
  archive_log "${RELAXED_OUT}" "${tag}"

  launch_run 0,1,2,3 "${X1_OUT}" x1_LP_102M "${X1_WANDB_RUN_ID}" \
    --slab-source initial

  launch_run 4,5,6,7 "${RELAXED_OUT}" x1_LP_102M_relaxed_bare_slab_x0 "${RELAXED_WANDB_RUN_ID}" \
    --slab-source pristine_relaxed \
    --pristine-slabs "${PRISTINE_SLABS}" \
    --pristine-index "${PRISTINE_INDEX}" \
    --val-pristine-slabs "${VAL_PRISTINE_SLABS}" \
    --val-pristine-index "${VAL_PRISTINE_INDEX}"

  log "restart complete; both runs now use 973-system energy reference"
}

main "$@"
