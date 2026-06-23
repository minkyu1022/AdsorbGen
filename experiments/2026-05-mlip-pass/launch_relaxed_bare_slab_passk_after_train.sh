#!/usr/bin/env bash
set -u

ROOT="${ROOT:-/home1/irteam/AdsorbGen}"
PY="${PY:-/home1/irteam/micromamba/envs/adsorbgen/bin/python}"
RUN_DIR="${RUN_DIR:-/home1/irteam/runs/training/x1_LP_102M_relaxed_bare_slab_x0}"
WANDB_RUN="${WANDB_RUN:-minkyu1022_/adsorbgen/acoqaaqv}"
OUT_ROOT="${OUT_ROOT:-/home/irteam/data/replay}"
GPU_LIST="${GPU_LIST:-4,5,6,7}"
NUM_SHARDS="${NUM_SHARDS:-4}"
LMDB="${LMDB:-/home/irteam/data/processed_old/oc20dense.lmdb}"
COVER="${COVER:-/home/irteam/data-vol1/minkyu/data/OC20-dense_FT_global_min_cover}"
SPLIT="${SPLIT:-/home/irteam/data/replay/oc20dense_oc20_split_membership.json}"
PRISTINE="${PRISTINE:-/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl}"
PINDEX="${PINDEX:-/home/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl}"
ADS="${ADS:-/home/irteam/data/pkls/adsorbates.pkl}"
POLL_SEC="${POLL_SEC:-600}"

mkdir -p "${OUT_ROOT}"
cd "${ROOT}" || exit 1

echo "[watcher] start $(date) run_dir=${RUN_DIR} wandb=${WANDB_RUN}"
while true; do
  if grep -q '`Trainer.fit` stopped: `max_epochs=100` reached.' "${RUN_DIR}/train.log" 2>/dev/null; then
    if ! pgrep -f "adsorbgen.training.train_cli .*--out ${RUN_DIR}" >/dev/null; then
      break
    fi
  fi
  echo "[watcher] training still running $(date)"
  sleep "${POLL_SEC}"
done
echo "[watcher] training finished $(date)"

mapfile -t CKPT_ROWS < <("${PY}" - <<'PY'
import re
from pathlib import Path
import wandb

run_dir = Path(__import__("os").environ["RUN_DIR"])
run_name = __import__("os").environ["WANDB_RUN"]
api = wandb.Api(timeout=60)
run = api.run(run_name)
keys = [
    "epoch",
    "sample_eval/dense/uma_sp_valid_delta_E_sys_mean_eV",
    "sample_eval/dense/uma_sp_valid_delta_E_sys_mae_eV",
    "sample_eval/dense/uma_sp_valid_energy_count",
    "sample_eval/dense/uma_sp_valid_rate",
]
hist = run.history(keys=keys, pandas=True, samples=50000)
hist = hist[[c for c in keys if c in hist.columns]].dropna(subset=["epoch"]).copy()
hist["epoch"] = hist["epoch"].astype(int)
available = {}
for p in run_dir.glob("ckpt_epochepoch=*.ckpt"):
    m = re.search(r"epoch=(\d+)", p.name)
    if m:
        available[int(m.group(1))] = p
metric = "sample_eval/dense/uma_sp_valid_delta_E_sys_mean_eV"
count_metric = "sample_eval/dense/uma_sp_valid_energy_count"
rate_metric = "sample_eval/dense/uma_sp_valid_rate"
chosen = []
if metric in hist.columns and available:
    ck = hist[hist["epoch"].isin(available)].dropna(subset=[metric]).copy()
    if count_metric in ck.columns:
        max_count = ck[count_metric].max()
        # Keep checkpoints with near-best valid coverage, then rank by energy gap.
        ck = ck[ck[count_metric] >= max_count * 0.95].copy()
    sort_cols = [metric]
    ascending = [True]
    if rate_metric in ck.columns:
        sort_cols.append(rate_metric)
        ascending.append(False)
    if count_metric in ck.columns:
        sort_cols.append(count_metric)
        ascending.append(False)
    ck = ck.sort_values(sort_cols, ascending=ascending)
    for _, row in ck.head(2).iterrows():
        ep = int(row["epoch"])
        score = float(row[metric])
        count = float(row[count_metric]) if count_metric in row and not __import__("math").isnan(float(row[count_metric])) else float("nan")
        rate = float(row[rate_metric]) if rate_metric in row and not __import__("math").isnan(float(row[rate_metric])) else float("nan")
        chosen.append((f"epoch{ep:03d}_valid{count:.0f}_rate{rate:.3f}", str(available[ep]), score))
for label, path, score in chosen:
    print(f"{label}\t{path}\t{score:.9g}")
last = run_dir / "last.ckpt"
if last.exists():
    print(f"last\t{last}\tnan")
PY
)

printf "[watcher] selected ckpts:\n"
printf "  %s\n" "${CKPT_ROWS[@]}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -lt "${NUM_SHARDS}" ]]; then
  echo "GPU_LIST has ${#GPUS[@]} entries but NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 1
fi

for row in "${CKPT_ROWS[@]}"; do
  IFS=$'\t' read -r label ckpt score <<< "${row}"
  out="${OUT_ROOT}/mlip_pass_lbfgs_ood50_relaxed_bare_slab_x0_${label}_$(date +%Y%m%d_%H%M%S)"
  log="${out}/logs"
  mkdir -p "${log}"
  echo "[supervisor] start label=${label} score=${score} out=${out} ckpt=${ckpt} $(date)" | tee -a "${log}/supervisor.log"
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    gpu="${GPUS[$shard]}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONUNBUFFERED=1
      export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
      export ADSORBATES_PKL="${ADS}"
      "${PY}" experiments/2026-05-mlip-pass/eval_mlip_pass_lbfgs_ood50.py \
        --ckpt "${ckpt}" \
        --lmdb "${LMDB}" \
        --cover-dir "${COVER}" \
        --split-membership "${SPLIT}" \
        --out-dir "${out}" \
        --shard-idx "${shard}" \
        --num-shards "${NUM_SHARDS}" \
        --num-systems 50 \
        --num-samples 100 \
        --flow-steps 50 \
        --flow-batch-size 32 \
        --prior-mode random_heuristic \
        --slab-source pristine_relaxed \
        --placement-pristine-slabs "${PRISTINE}" \
        --placement-pristine-index "${PINDEX}" \
        --pristine-slabs "${PRISTINE}" \
        --pristine-index "${PINDEX}"
    ) > "${log}/shard_${shard}.log" 2>&1 &
    echo $! > "${log}/pid_shard${shard}.txt"
    echo "[supervisor] shard ${shard} gpu=${gpu} pid=$(cat "${log}/pid_shard${shard}.txt")" | tee -a "${log}/supervisor.log"
    sleep 2
  done

  failed=0
  for ((shard=0; shard<NUM_SHARDS; shard++)); do
    pid="$(cat "${log}/pid_shard${shard}.txt")"
    if wait "${pid}"; then
      echo "[supervisor] shard ${shard} done $(date)" | tee -a "${log}/supervisor.log"
    else
      rc=$?
      echo "[supervisor] shard ${shard} failed rc=${rc} $(date)" | tee -a "${log}/supervisor.log"
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "[supervisor] not merging label=${label} due to failed shard $(date)" | tee -a "${log}/supervisor.log"
    continue
  fi
  "${PY}" experiments/2026-05-mlip-pass/merge_mlip_pass_lbfgs_ood50.py \
    --out-dir "${out}" \
    --num-shards "${NUM_SHARDS}" \
    --num-samples 100 > "${log}/merge.log" 2>&1
  echo "[supervisor] merged label=${label} $(date) summary=${out}/summary.json" | tee -a "${log}/supervisor.log"
done

echo "[watcher] all requested pass@k jobs finished $(date)"
