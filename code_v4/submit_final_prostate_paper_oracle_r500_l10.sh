#!/bin/bash
# Submit wrapper for the finalized RGFTD stable Prostate run on the paper-aligned data.

#SBATCH --job-name=Final_PROSTATE_ORACLE
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4

export RUN_PREFIX="${RUN_PREFIX:-final_rgftd_stable_paper_oracle_d5scribble2elastic_d6maskblock_e8_prostate_r500_l10}"
export SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8642}"
export ROOT_PATH="${ROOT_PATH:-/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_paper_oracle_d5scribble2elastic_d6maskblock_e8}"
export CLIENT6_SUP_TYPE="${CLIENT6_SUP_TYPE:-block}"
export ITERS="${ITERS:-10}"
export EVAL_ITERS="${EVAL_ITERS:-10}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
export TSNE_ITERS="${TSNE_ITERS:-0}"
export WANN_PRED_START_ITER="${WANN_PRED_START_ITER:-800}"
export WANN_SOFT_RAMPUP_ITERS="${WANN_SOFT_RAMPUP_ITERS:-800}"
export WANN_CONS_RAMPUP_ITERS="${WANN_CONS_RAMPUP_ITERS:-800}"
export RGFTD_WARMUP_ITERS="${RGFTD_WARMUP_ITERS:-800}"
export RGFTD_RAMPUP_ITERS="${RGFTD_RAMPUP_ITERS:-800}"
export RGFTD_V3_AUDIT_START_ITERS="${RGFTD_V3_AUDIT_START_ITERS:-800}"

exec bash /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/run_prostate_rgftd_stable_paper_oracle_d5d6_r500_l10.sh
