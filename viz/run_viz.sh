#!/usr/bin/env bash
# Launch the AdsorbGen Replay Viz stack in parallel (backend + frontend dev server).
# Two panes required; use tmux or two terminals.
#
# Usage:
#   bash viz/run_viz.sh backend                     # start FastAPI on :8000
#   bash viz/run_viz.sh frontend                    # start Next.js dev on :3000
#   bash viz/run_viz.sh install                     # one-time npm install
#
# Then open http://localhost:3000 in a browser.

set -euo pipefail

REPO=/home/irteam/AdsorbGen
PYTHON=/home/irteam/micromamba/envs/adsorbgen/bin/python3
VIZ_ROOT_DEFAULT=/home/irteam/runs/H200_ads_pair_dist_loss/replay_viz

cmd="${1:-help}"

case "${cmd}" in
  install)
    cd "${REPO}/viz/frontend"
    npm install
    ;;

  backend)
    export REPLAY_VIZ_ROOT="${REPLAY_VIZ_ROOT:-${VIZ_ROOT_DEFAULT}}"
    cd "${REPO}"
    exec "${PYTHON}" -m uvicorn viz.backend.main:app \
        --host 0.0.0.0 --port "${VIZ_PORT:-8000}" --reload
    ;;

  frontend)
    cd "${REPO}/viz/frontend"
    export NEXT_PUBLIC_BACKEND_URL="${NEXT_PUBLIC_BACKEND_URL:-http://localhost:${VIZ_PORT:-8000}}"
    exec npm run dev
    ;;

  help|*)
    cat <<EOF
AdsorbGen Replay Viz launcher

Commands:
  install    One-time npm install (in viz/frontend)
  backend    Start FastAPI on :8000
  frontend   Start Next.js dev server on :3000

Env:
  REPLAY_VIZ_ROOT        default: ${VIZ_ROOT_DEFAULT}
  NEXT_PUBLIC_BACKEND_URL default: http://localhost:8000

Typical workflow:
  # one-time:
  bash viz/run_viz.sh install
  # two tmux panes:
  bash viz/run_viz.sh backend
  bash viz/run_viz.sh frontend
  # browser:
  open http://localhost:3000
EOF
    ;;
esac
