#!/bin/bash
#SBATCH --job-name=DiagTCR_POLY
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/diag_tcr_polyp_%j.log

set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
OUT=/data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsitp_polyp_r500_l10_20260528_092821_seed2022/rdsi_tcr_diag_full_gpu_slurm
mkdir -p "${OUT}"

python analyze_rdsi_tcr_diagnostics.py \
  --img_class polyp \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsitp_polyp_r500_l10_20260528_092821_seed2022 \
  --client_sup_types scribble,keypoint,scribble,block \
  --device cuda \
  --batch_size 1 \
  --output_dir "${OUT}"
