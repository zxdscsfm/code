#!/bin/bash
#SBATCH --job-name=FINAL_PC_EVAL
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/final_per_client_eval_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

OUT_DIR="${OUT_DIR:-/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/final_per_client_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

cp /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/eval_final_per_client.py "${OUT_DIR}/eval_final_per_client.py"
cp /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/combine_final_per_client.py "${OUT_DIR}/combine_final_per_client.py"

echo "OUT_DIR=${OUT_DIR}"

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
python -u eval_final_per_client.py --method ted --out-dir "${OUT_DIR}" --use-cuda 1
python -u eval_final_per_client.py --method sota --out-dir "${OUT_DIR}" --use-cuda 1
python -u combine_final_per_client.py --out-dir "${OUT_DIR}"
