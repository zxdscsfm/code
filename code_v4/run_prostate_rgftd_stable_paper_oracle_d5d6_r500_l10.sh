#!/bin/bash
# PROSTATE final/stable run on paper-aligned high-quality weak labels:
# Domain5 uses mask-derived Scribble2 through the existing "scribble" key.
# Domain6 uses mask-eroded Block through the existing "block" key.

set -euo pipefail

export RUN_PREFIX="${RUN_PREFIX:-official_rgftd_stable_paper_oracle_d5scribble2elastic_d6maskblock_e8_prostate_paper_r500_l10}"
export ROOT_PATH="${ROOT_PATH:-/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_paper_oracle_d5scribble2elastic_d6maskblock_e8}"
export CLIENT6_SUP_TYPE="${CLIENT6_SUP_TYPE:-block}"

export WANN_PRED_START_ITER="${WANN_PRED_START_ITER:-800}"
export WANN_SOFT_RAMPUP_ITERS="${WANN_SOFT_RAMPUP_ITERS:-800}"
export WANN_CONS_RAMPUP_ITERS="${WANN_CONS_RAMPUP_ITERS:-800}"
export RGFTD_WARMUP_ITERS="${RGFTD_WARMUP_ITERS:-800}"
export RGFTD_RAMPUP_ITERS="${RGFTD_RAMPUP_ITERS:-800}"
export RGFTD_V3_AUDIT_START_ITERS="${RGFTD_V3_AUDIT_START_ITERS:-800}"

exec bash /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/run_prostate_rgftd_stable_paper_r500_l10.sh
