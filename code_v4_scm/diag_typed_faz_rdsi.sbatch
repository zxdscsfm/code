#!/bin/bash
#SBATCH --partition=v100_batch
#SBATCH --gres=gpu:1
#SBATCH --job-name=DiagTyped_FAZ
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/diag_typed_faz_%j.log

set -euo pipefail

source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
TS="$(date +%Y%m%d_%H%M%S)"
echo "typed action oracle FAZ diagnostics TS=${TS}"

python -u analyze_rdsi_oracle_diagnostics.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/faz/FedLPPA_rdsitrp_faz_r500_l10_20260527_032024_seed2024 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd \
  --img_class faz \
  --client_sup_types scribble,keypoint,scribble,block,scribble \
  --output_dir "results/typed_action_oracle_rdsitrp_faz_seed2024_${TS}" \
  --device cuda \
  --analysis_iter 5000 \
  --img_size 256 \
  --max_cases_per_client 0
