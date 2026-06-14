#!/bin/bash
#SBATCH --job-name=DiagOracle_POLY
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/diag_oracle_polyp_%j.log

set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
OUT=/data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/oracle_pool_compare_ted_trp_tp_20260529
mkdir -p "${OUT}"

python analyze_rdsi_oracle_pool_compare.py \
  --img_class polyp \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --client_sup_types scribble,keypoint,scribble,block \
  --device cuda \
  --batch_size 1 \
  --output_dir "${OUT}" \
  --run TED_success=/data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsited_polyp_r500_l10_20260527_134611_seed2022 \
  --run TRP_success=/data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsitrp_polyp_r500_l10_20260526_145404_seed2022 \
  --run TP_failed=/data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsitp_polyp_r500_l10_20260528_092821_seed2022
