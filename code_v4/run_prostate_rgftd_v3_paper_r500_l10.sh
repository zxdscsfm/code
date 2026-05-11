#!/bin/bash
# Official FedLPPA + WANN + RGFTD-v3 on PROSTATE.
#SBATCH --job-name=FedLPPA_PROSTATE_RGFTD_V3
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

SCRIPT_DIR="/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4"
export RUN_PREFIX="${RUN_PREFIX:-official_rgftd_v3_prostate_paper_r500_l10}"

V22_ARGS="--rgftd_max_bg_fg_ratio 1.0 --rgftd_allow_bg_without_fg 0 --rgftd_lambda_eff_cap 0.02"
V3_ARGS="--rgftd_v3_enabled 1 --rgftd_v3_teacher_pool_topk 1 --rgftd_v3_audit_start_iters 800 --rgftd_v3_audit_interval_iters 1000 --rgftd_v3_audit_batches 4 --rgftd_v3_audit_score_thresh 0.05 --rgftd_v3_audit_teacher_reliability_min 0.30 --rgftd_v3_fallback_lambda_eff_cap 0.01"

if [ -n "${EXTRA_ARGS:-}" ]; then
    export EXTRA_ARGS="${V22_ARGS} ${V3_ARGS} ${EXTRA_ARGS}"
else
    export EXTRA_ARGS="${V22_ARGS} ${V3_ARGS}"
fi

exec "${SCRIPT_DIR}/run_prostate_rgftd_v2_paper_r500_l10.sh"
