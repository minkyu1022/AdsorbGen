#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/run_geoopt_benchmark.sh" "${HERE}/configs/cg.env"
