#!/bin/bash
# Validate seed-support quality on all ODOC sites under a fixed W1' checkpoint stage.

#SBATCH -J seed_all_v100
#SBATCH -p v100_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH -o /data/jianbingshen/yanghongji/FedLPPA/code_v4/all_sites_seed_support_validation_v100.%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

python tmp_validate_seed_support_all_sites.py \
  --model_dir /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v34_w1p_lgcalib_paper_r500_l10_odoc_20260419_083158_seed2022 \
  --iter_tag 3800 \
  --output /data/jianbingshen/yanghongji/FedLPPA/code_v4/all_sites_seed_support_validation_v100.json
