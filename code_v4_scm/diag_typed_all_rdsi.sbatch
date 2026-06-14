#!/bin/bash
#SBATCH --partition=v100_batch
#SBATCH --gres=gpu:1
#SBATCH --job-name=DiagTyped_ALL
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/diag_typed_all_%j.log

set -euo pipefail

source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
TS="$(date +%Y%m%d_%H%M%S)"
echo "typed action oracle diagnostics TS=${TS}"

python -u analyze_rdsi_oracle_diagnostics.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_rdsitrp_prostate_r500_l10_20260526_205646_seed2024 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd \
  --img_class prostate \
  --client_sup_types scribble,keypoint,scribble,block,scribble,scribble \
  --output_dir "results/typed_action_oracle_rdsitrp_prostate_seed2024_${TS}" \
  --device cuda \
  --analysis_iter 5000 \
  --max_cases_per_client 0

python -u analyze_rdsi_oracle_diagnostics.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsitrp_polyp_r500_l10_20260527_014854_seed2024 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --img_class polyp \
  --client_sup_types scribble,keypoint,scribble,block \
  --output_dir "results/typed_action_oracle_rdsitrp_polyp_seed2024_${TS}" \
  --device cuda \
  --analysis_iter 5000 \
  --max_cases_per_client 0

python -u analyze_rdsi_oracle_diagnostics.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/faz/FedLPPA_rdsitrp_faz_r500_l10_20260527_032024_seed2024 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd \
  --img_class faz \
  --client_sup_types scribble,keypoint,scribble,block,scribble \
  --output_dir "results/typed_action_oracle_rdsitrp_faz_seed2024_${TS}" \
  --device cuda \
  --analysis_iter 5000 \
  --max_cases_per_client 0
