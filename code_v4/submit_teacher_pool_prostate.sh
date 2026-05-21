#!/bin/bash
#SBATCH --job-name=RGFTDTP_PRO
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/teacher_pool_analysis/prostate_%j.log
set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
python -u analyze_rgftd_teacher_pool_feasibility.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_official_rgftd_v22_prostate_paper_r500_l10_20260509_110337_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5 \
  --img_class prostate \
  --gpu 0
