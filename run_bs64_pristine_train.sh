#!/usr/bin/env bash
set -o pipefail

OUT="$1"
RUN_NAME="${2:-$(basename "$OUT")}"
NUM_WORKERS="${NUM_WORKERS:-${3:-4}}"
if [ -z "$OUT" ]; then
  echo "usage: $0 OUT_DIR [RUN_NAME] [NUM_WORKERS]" >&2
  exit 2
fi

mkdir -p "$OUT"
cd /home1/irteam/AdsorbGen || exit 3
PYTHON=/home1/irteam/micromamba/envs/adsorbgen/bin/python
if [ ! -x "$PYTHON" ]; then
  echo "missing python: $PYTHON" >&2
  exit 4
fi

export PYTHONFAULTHANDLER=1
export PYTHONPATH=/home/irteam/AdsorbGen:${PYTHONPATH:-}
export ADSORBATES_PKL=/home/irteam/data/pkls/adsorbates.pkl
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB__SERVICE_WAIT=300

(
  while true; do
    printf "%s," "$(date +%s)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | tr "\n" ";"
    printf "\n"
    sleep 10
  done
) > "$OUT/gpu_watch.csv" 2>&1 &
MON=$!
echo "$MON" > "$OUT/monitor.pid"

"$PYTHON" -m adsorbgen.training.train_cli \
  --arch v1 \
  --train-lmdb /home/irteam/data/processed_ID/is2res_train.lmdb /home/irteam/data/processed_ID/is2res_val.lmdb \
  --val-lmdb /home/irteam/data/processed_old/oc20dense.lmdb \
  --batch-size 64 \
  --num-workers "$NUM_WORKERS" \
  --epochs 150 \
  --devices 8 \
  --precision bf16-mixed \
  --lr 1e-4 \
  --lr-warmup-steps 5000 \
  --grad-clip 10.0 \
  --loss-type l1 \
  --loss-surf-weight 1.0 \
  --loss-ads-weight 1.0 \
  --ads-pair-l1-weight 1.0 \
  --ads-bond-l1-weight 0.0 \
  --ads-nonbonded-clash-weight 0.0 \
  --ads-center-loss-weight 0.0 \
  --ads-rel-pos-loss-weight 0.0 \
  --movable-mode surface_ads \
  --prediction-type x1 \
  --prior-mode random_heuristic \
  --slab-source pristine_relaxed \
  --pristine-slabs /home1/irteam/full_replay_1p2_lbfgs_transfer_20260529/data/replay_uma_s_1p2/pristine_slabs_lbfgs.pkl \
  --pristine-index /home1/irteam/full_replay_1p2_lbfgs_transfer_20260529/results/pristine_slabs/is2res.sid_index.pkl \
  --val-pristine-slabs /home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense_uma.pkl \
  --val-pristine-index /home1/irteam/data-vol1/minkyu/results/pristine_slabs/oc20dense.system_index.pkl \
  --sample-eval-every-epochs 1 \
  --sample-eval-max-samples 1000 \
  --sample-eval-steps 20 \
  --max-val-samples 1000 \
  --check-val-every-n-epoch 1 \
  --save-every-n-epochs 10 \
  --variant v0-ads-ref-adshead \
  --out "$OUT" \
  --wandb-project adsorbgen \
  --wandb-run-name "$RUN_NAME"
RC=$?
echo "$RC" > "$OUT/exit_code.txt"
kill "$MON" 2>/dev/null || true
wait "$MON" 2>/dev/null || true
exit "$RC"
