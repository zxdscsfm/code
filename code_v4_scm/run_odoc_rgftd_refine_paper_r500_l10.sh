#!/bin/bash
# Official FedLPPA + WANN + full RGFTD with target-side soft refinement on ODOC.

#SBATCH --job-name=FedLPPA_ODOC_RGFTD_REF
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:6
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

SCRIPT_DIR="/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4"
export RUN_PREFIX="${RUN_PREFIX:-official_rgftd_refine_odoc_paper_r500_l10}"

REFINE_ARGS="--rgftd_refine_enabled 1 --rgftd_refine_iters 3 --rgftd_refine_affinity_sigma 0.75 --rgftd_refine_affinity_mix 0.35 --rgftd_refine_seed_strength 0.95 --rgftd_refine_core_anchor_radius 1 --rgftd_refine_unsupported_fg_scale 0.25 --rgftd_refine_fg_floor 0.02 --rgftd_refine_bg_ceiling 0.98 --rgftd_refine_min_fg_mass 1.0 --rgftd_refine_min_roi_pixels 1.0"

if [ -n "${EXTRA_ARGS:-}" ]; then
    export EXTRA_ARGS="${REFINE_ARGS} ${EXTRA_ARGS}"
else
    export EXTRA_ARGS="${REFINE_ARGS}"
fi

exec "${SCRIPT_DIR}/run_odoc_rgftd_v33_paper_r500_l10.sh"
