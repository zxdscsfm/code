#!/bin/bash
# Official FedLPPA + WANN + RGFTD-v2.2 on ODOC.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_PREFIX="${RUN_PREFIX:-official_rgftd_v22_odoc_paper_r500_l10}"
V22_ARGS="--rgftd_max_bg_fg_ratio 1.0 --rgftd_allow_bg_without_fg 0 --rgftd_lambda_eff_cap 0.02"
if [ -n "${EXTRA_ARGS:-}" ]; then
    export EXTRA_ARGS="${V22_ARGS} ${EXTRA_ARGS}"
else
    export EXTRA_ARGS="${V22_ARGS}"
fi

exec "${SCRIPT_DIR}/run_odoc_rgftd_v2_paper_r500_l10.sh"
