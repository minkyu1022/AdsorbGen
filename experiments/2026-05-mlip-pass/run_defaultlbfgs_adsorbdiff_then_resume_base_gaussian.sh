#!/usr/bin/env bash
set +e

OUT_ROOT="${1:?OUT_ROOT required}"
export OUT_ROOT
export GPUS="${GPUS:-4 5 6 7}"

bash /home1/irteam/AdsorbGen/experiments/2026-05-mlip-pass/launch_diag_replay_setting_ood50_defaultlbfgs_adsorbdiff_only.sh
status=$?

echo "[b3-then-resume] diag status=${status} $(date -Is)"
echo "[b3-then-resume] restarting base_gaussian_adsprior_102M $(date -Is)"
nohup /home1/irteam/runs/training/base_gaussian_adsprior_102M/launch_command.sh \
  >> /home1/irteam/runs/training/base_gaussian_adsprior_102M/train.log 2>&1 &
echo $! > /home1/irteam/runs/training/base_gaussian_adsprior_102M/pid.txt

exit "${status}"
