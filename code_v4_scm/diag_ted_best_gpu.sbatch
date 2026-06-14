#!/bin/bash
#SBATCH --job-name=TEDBestDiag
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs/diag_ted_best_%j.log

set -euo pipefail
source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

CODE_DIR=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4
cd "$CODE_DIR"
mkdir -p results logs
STAMP=$(date +%Y%m%d_%H%M%S)

run_diag() {
  local IMG_CLASS="$1"
  local ROOT_PATH="$2"
  local SNAPSHOT="$3"
  local SUP_TYPES="$4"
  local IMG_SIZE="$5"
  local OUT_DIR="results/ted_best_diag_${IMG_CLASS}_seed2022_${STAMP}"
  echo "[TEDBestDiag] img_class=${IMG_CLASS} out=${OUT_DIR}"
  python -u analyze_ted_best_diagnostics.py \
    --snapshot_path "$SNAPSHOT" \
    --root_path "$ROOT_PATH" \
    --img_class "$IMG_CLASS" \
    --output_dir "$OUT_DIR" \
    --client_sup_types "$SUP_TYPES" \
    --img_size "$IMG_SIZE" \
    --analysis_iter 5000 \
    --batch_size 2 \
    --num_workers 0 \
    --device cuda \
    --max_cases_per_client 0
}

run_diag prostate \
  /data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd \
  /data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_rdsited_prostate_r500_l10_20260527_134611_seed2022 \
  scribble,keypoint,scribble,block,scribble,scribble \
  384

run_diag polyp \
  /data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd \
  /data/jianbingshen/yanghongji/FedLPPA_Original/model/polyp/FedLPPA_rdsited_polyp_r500_l10_20260527_134611_seed2022 \
  scribble,keypoint,scribble,block \
  384

run_diag faz \
  /data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd \
  /data/jianbingshen/yanghongji/FedLPPA_Original/model/faz/FedLPPA_rdsited_faz_r500_l10_20260527_190324_seed2022 \
  scribble,keypoint,scribble,block,scribble \
  256
