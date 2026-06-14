#!/bin/bash
#SBATCH --partition=v100_batch
#SBATCH --gres=gpu:1
#SBATCH --job-name=SimTEDCons
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/sim_ted_cons_%j.log

set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
TS="$(date +%Y%m%d_%H%M%S)"
echo "SimTEDConsensus ${TS}"

python -u simulate_ted_consensus_reweight.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_rdsited_prostate_r500_l10_20260527_134611_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd \
  --img_class prostate \
  --client_sup_types scribble,keypoint,scribble,block,scribble,scribble \
  --output_dir "results/ted_consensus_reweight_prostate_seed2022_${TS}" \
  --device cuda --analysis_iter 5000 --img_size 384 --batch_size 2 --gain 0.25 --min_weight 1.0 --max_weight 1.25

python -u simulate_ted_consensus_reweight.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsited_polyp_r500_l10_20260527_134611_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  --img_class polyp \
  --client_sup_types scribble,keypoint,scribble,block \
  --output_dir "results/ted_consensus_reweight_polyp_seed2022_${TS}" \
  --device cuda --analysis_iter 5000 --img_size 384 --batch_size 2 --gain 0.25 --min_weight 1.0 --max_weight 1.25

python -u simulate_ted_consensus_reweight.py \
  --snapshot_path /data/jianbingshen/yanghongji/FedLPPA_Original/model/faz/FedLPPA_rdsited_faz_r500_l10_20260527_190324_seed2022 \
  --root_path /data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd \
  --img_class faz \
  --client_sup_types scribble,keypoint,scribble,block,scribble \
  --output_dir "results/ted_consensus_reweight_faz_seed2022_${TS}" \
  --device cuda --analysis_iter 5000 --img_size 256 --batch_size 2 --gain 0.25 --min_weight 1.0 --max_weight 1.25
