#!/bin/bash
# Generic formal personalized inference entrypoint for micro ODOC runs.

#SBATCH --job-name=MicroODOCInfer
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/infer_micro_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

WORKSPACE="${WORKSPACE:?WORKSPACE is required}"
EXP_NAME="${EXP_NAME:?EXP_NAME is required}"
METHOD_NAME="${METHOD_NAME:-FedLPPA_micro}"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-async_best}"
OUTPUT_TAG="${OUTPUT_TAG:-formal}"
MODEL="${MODEL:-unet_univ5}"
ROOT_PATH="${ROOT_PATH:-../data/ODOC_h5}"
GPU="${GPU:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "${WORKSPACE}/code_v4"
mkdir -p logs

OUTPUT_DIR="../model/${EXP_NAME}/test_${CHECKPOINT_KIND}_${OUTPUT_TAG}"

python infer_fedlppa_personalized.py \
    --root_path "${ROOT_PATH}" \
    --exp "${EXP_NAME}" \
    --checkpoint_kind "${CHECKPOINT_KIND}" \
    --output_dir "${OUTPUT_DIR}" \
    --method_name "${METHOD_NAME}_${CHECKPOINT_KIND}" \
    --model "${MODEL}" \
    --img_class odoc \
    --num_classes 3 \
    --in_chns 3 \
    --img_size 384 \
    --min_num_clients 5 \
    --prompt universal \
    --attention dual \
    --dual_init aggregated \
    --label_prompt 1 \
    --gpu "${GPU}" \
    --site_labels SiteA SiteB SiteC SiteD SiteE \
    --clients client1 client2 client3 client4 client5 \
    --sup_type_list scribble scribble_noisy scribble_noisy keypoint block \
    ${EXTRA_ARGS}
