#!/bin/bash
#SBATCH --job-name=ORA_SOTA_POLY
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/oracle_current_sota_polyp_%j.log

set -euo pipefail

source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4

python analyze_rdsi_oracle_diagnostics.py \
  --device cuda \
  --batch_size 2 \
  --snapshot_path /data/jianbingshen/yanghongji/fedlppaSOTA/model/polyp/FedLPPA_sota_fedlppa_rdsi3_sd_polyp_r500_l10_20260522_230837_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --img_class polyp \
  --client_sup_types scribble,keypoint,scribble,block \
  --analysis_iter 4210 \
  --output_dir results/rdsi_oracle_current_sota_polyp_best_20260523
