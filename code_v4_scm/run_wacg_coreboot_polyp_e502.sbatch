#!/bin/bash
# E502: weak mid-stage foreground-core bootstrap for ACG Polyp.

#SBATCH --job-name=WACG_COREBT_POLY
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:5
#SBATCH --mem=64G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/slurm_wacg_coreboot_polyp_%j.log

set -euo pipefail

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4

export DATASET=polyp
export RUN_PREFIX=wannacgnwrhp_coreboot_polyp_r500_l10
export SERVER_ADDRESS=127.0.0.1:8865
export SEED=2022
export MAX_ITERATIONS=5000
export ITERS=10
export EVAL_ITERS=10
export EXTRA_ARGS="\
--acg_nwr_prior_warmup_enabled 0 \
--acg_soft_exterior_enabled 0 \
--acg_core_bootstrap_enabled 1 \
--acg_core_bootstrap_lambda 0.08 \
--acg_core_bootstrap_start_iter 1000 \
--acg_core_bootstrap_end_iter 2600 \
--acg_core_bootstrap_rampup_iters 600 \
--acg_core_bootstrap_target_start 0.76 \
--acg_core_bootstrap_target_end 0.84 \
--acg_core_bootstrap_gate_floor 0.55 \
--acg_core_bootstrap_bg_ceiling 0.06 \
--acg_core_bootstrap_bg_max 0.12 \
--save_code_snapshot 1 \
--save_checkpoint_copies 0"

bash run_wannacgnwrhp_r500_l10.sh
