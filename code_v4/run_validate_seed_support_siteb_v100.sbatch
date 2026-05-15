#!/bin/bash
#SBATCH -J siteb_seed_v100
#SBATCH -p v100_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o /data/jianbingshen/yanghongji/FedLPPA/code_v4/siteb_seed_support_validation_v100.%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA/code_v4
python tmp_validate_seed_support_siteb.py   --checkpoint /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v34_w1p_lgcalib_paper_r500_l10_odoc_20260419_083158_seed2022/client_1_iter_3800_dice_0.7523.pth   --output /data/jianbingshen/yanghongji/FedLPPA/code_v4/siteb_seed_support_validation_v100.json
