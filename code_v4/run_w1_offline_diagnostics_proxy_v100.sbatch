#!/bin/bash
# Offline W1' diagnostics without retraining.
# Default mode uses a local-vs-aux proxy because a matching saved global checkpoint
# is not currently wired into this launcher.

#SBATCH -J w1diag_v100
#SBATCH -p v100_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH -o /data/jianbingshen/yanghongji/FedLPPA/code_v4/w1_offline_diagnostics_v100.%j.out

source ~/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

python tmp_w1_offline_diagnostics.py \
  --local_model_dir /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v34_w1p_lgcalib_paper_r500_l10_odoc_20260419_083158_seed2022 \
  --iter_tag 3800 \
  --score_mode local_aux_proxy \
  --tau_source fixed \
  --tau_init 0.55 \
  --output /data/jianbingshen/yanghongji/FedLPPA/code_v4/w1_offline_diagnostics_proxy_v100.json
