#!/bin/bash
#SBATCH --job-name=b2_learnnwr_prostate
#SBATCH --partition=a100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=160G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/yc47942/fl_run/logs/b2_learnnwr_prostate_%j.out
#SBATCH --error=/home/yc47942/fl_run/logs/b2_learnnwr_prostate_%j.err

set -euo pipefail

mkdir -p /home/yc47942/fl_run/logs
cd /home/yc47942/fl_run/code_v6_learnable_nwr

DATASET=prostate \
AGC_MODE=conservative \
GPU_COUNT=3 \
./run_annocal_agc_mech_r500_l10.sh
