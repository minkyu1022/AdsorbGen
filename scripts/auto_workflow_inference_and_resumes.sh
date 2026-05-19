#!/usr/bin/env bash
# Orchestrator: kicks off the GPU 0-3 resume immediately, runs inference on
# GPU 5 & GPU 6 in parallel, exports ASE .traj files filtered for
# dissoc/overlap, then launches the GPU 4-7 resume.
set -euo pipefail
source /home/irteam/adsorbgen_env.sh

ROOT="${CAT_BENCH_ROOT:-/home/irteam}"
SCRIPTS="${ROOT}/AdsorbGen/scripts"
MICROMAMBA="${MICROMAMBA:-/home/irteam/.local/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/irteam/micromamba}"

DUMP_ROOT="${ROOT}/runs/dissoc_traj"
mkdir -p "${DUMP_ROOT}/catflow_center_rel" "${DUMP_ROOT}/ads_pair_dist_loss"

run_inference() {
  local gpu="$1" ckpt="$2" prior="$3" out="$4"
  env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="${ROOT}/AdsorbGen:${PYTHONPATH:-}" \
      CAT_BENCH_ROOT="${ROOT}" \
      ADSORBATES_PKL="${ROOT}/data/pkls/adsorbates.pkl" \
      MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
    "${MICROMAMBA}" run -n adsorbgen python -m adsorbgen.inference \
      --ckpt "${ckpt}" \
      --lmdb "${ROOT}/data/processed/oc20dense.lmdb" \
      --out "${out}/records.pt" \
      --batch-size 30 --max-samples 60 \
      --num-steps 50 --save-trajectories 30 \
      --prior-mode "${prior}" --use-placement-prior
}

echo "[orch] $(date) STEP 1: launch ads_pair_dist_loss resume on GPU 0-3"
bash "${SCRIPTS}/launch_resume_ads_pair_dist_loss_to_100.sh"

echo "[orch] $(date) STEP 2: run two inferences in parallel on GPU 5 / GPU 6"
run_inference 5 "${ROOT}/runs/H200_catflow_center_rel/last.ckpt" \
                "catflow_center_rel" "${DUMP_ROOT}/catflow_center_rel" \
                > "${DUMP_ROOT}/catflow_center_rel/inference.log" 2>&1 &
PID_INF_CF=$!
run_inference 6 "${ROOT}/runs/H200_ads_pair_dist_loss/last.ckpt" \
                "random_heuristic" "${DUMP_ROOT}/ads_pair_dist_loss" \
                > "${DUMP_ROOT}/ads_pair_dist_loss/inference.log" 2>&1 &
PID_INF_AP=$!

echo "[orch] inference PIDs: catflow=${PID_INF_CF} ads_pair=${PID_INF_AP}"
wait "${PID_INF_CF}" "${PID_INF_AP}"
echo "[orch] $(date) STEP 2 inferences finished"

echo "[orch] $(date) STEP 3: export ASE trajectories"
env PYTHONPATH="${ROOT}/AdsorbGen:${PYTHONPATH:-}" \
    "${MICROMAMBA}" run -n adsorbgen python "${SCRIPTS}/export_dissoc_trajectories.py" \
    "${DUMP_ROOT}/catflow_center_rel" \
    >> "${DUMP_ROOT}/catflow_center_rel/inference.log" 2>&1
env PYTHONPATH="${ROOT}/AdsorbGen:${PYTHONPATH:-}" \
    "${MICROMAMBA}" run -n adsorbgen python "${SCRIPTS}/export_dissoc_trajectories.py" \
    "${DUMP_ROOT}/ads_pair_dist_loss" \
    >> "${DUMP_ROOT}/ads_pair_dist_loss/inference.log" 2>&1

echo "[orch] $(date) STEP 4: launch catflow_center_rel resume on GPU 4-7"
bash "${SCRIPTS}/launch_resume_catflow_center_rel_to_100.sh"

echo "[orch] $(date) ALL DONE. Trajectories at ${DUMP_ROOT}/<run>/trajectories/_index.json"
