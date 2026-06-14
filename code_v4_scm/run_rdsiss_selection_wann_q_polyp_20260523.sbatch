#!/bin/bash
#SBATCH --job-name=SEL_RSS_POLY
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/selection_wann_q_rdsiss_polyp_%j.log

set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
python analyze_rdsiss_selection_wann_q.py \
  --device cuda \
  --batch_size 2 \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsiss_polyp_r500_l10_20260523_080746_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --img_class polyp \
  --client_sup_types scribble,keypoint,scribble,block \
  --analysis_iter 4960 \
  --rgftd_use_soft_band 0 \
  --rdsi_teacher_sup_types scribble,keypoint,scribble,block \
  --output_dir results/rdsiss_selection_wann_q_polyp_best4960_20260523
