#!/bin/bash
# Official FedLPPA + WANN + RGFTD-v3.3 on PROSTATE.
# V3.3 keeps v3.2 teacher leases and softly weights foreground release by target-side spatial support.
#SBATCH --job-name=FedLPPA_PROSTATE_RGFTD_V33
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

SCRIPT_DIR="/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4"
export RUN_PREFIX="${RUN_PREFIX:-official_rgftd_v33_prostate_paper_r500_l10}"

V22_ARGS="--rgftd_max_bg_fg_ratio 1.0 --rgftd_allow_bg_without_fg 0 --rgftd_lambda_eff_cap 0.02"
V3_ARGS="--rgftd_v3_enabled 1 --rgftd_v3_teacher_pool_topk 1 --rgftd_v3_audit_start_iters 800 --rgftd_v3_audit_interval_iters 1000 --rgftd_v3_audit_batches 4 --rgftd_v3_audit_score_thresh 0.05 --rgftd_v3_audit_teacher_reliability_min 0.30 --rgftd_v3_fallback_lambda_eff_cap 0.01"
V32_ARGS="--rgftd_v3_benefit_enabled 1 --rgftd_v3_lease_iters 1000 --rgftd_v3_benefit_momentum 0.80 --rgftd_v3_benefit_good_thresh 0.70 --rgftd_v3_benefit_decay_thresh 0.50 --rgftd_v3_benefit_revoke_thresh 0.35 --rgftd_v3_routing_benefit_floor 0.25 --rgftd_v3_recover_audit_score_thresh 0.60 --rgftd_v3_decay_cap_scale 0.50 --rgftd_v3_min_cap_scale 0.25 --rgftd_v3_bgfg_warn_thresh 0.75 --rgftd_v3_cap_hit_penalty 0.20 --rgftd_v3_lowmaxp_delta_thresh 0.02 --rgftd_v3_wann_mass_drop_thresh 0.05"
V33_ARGS="--rgftd_spatial_support_enabled 1 --rgftd_spatial_support_radius 2 --rgftd_spatial_candidate_weight 1.0 --rgftd_spatial_near_seed_weight 0.75 --rgftd_spatial_far_weight 0.15"

if [ -n "${EXTRA_ARGS:-}" ]; then
    export EXTRA_ARGS="${V22_ARGS} ${V3_ARGS} ${V32_ARGS} ${V33_ARGS} ${EXTRA_ARGS}"
else
    export EXTRA_ARGS="${V22_ARGS} ${V3_ARGS} ${V32_ARGS} ${V33_ARGS}"
fi

exec "${SCRIPT_DIR}/run_prostate_rgftd_v2_paper_r500_l10.sh"
