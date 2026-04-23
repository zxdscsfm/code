#!/bin/bash
# Formal personalized inference for ODOC step1 run 118912.

#SBATCH --job-name=ODOCInfer118912
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/infer_118912_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

EXP_NAME="odoc/FedLPPA_v32_step1_geompl_odoc_20260405_180143_seed2022"
CHECKPOINT_KIND="${CHECKPOINT_KIND:-async_best}"
OUTPUT_TAG="${OUTPUT_TAG:-formal}"
METHOD_NAME="${METHOD_NAME:-ODOC_step1}"
OUTPUT_DIR="../model/odoc/FedLPPA_v32_step1_geompl_odoc_20260405_180143_seed2022/test_${CHECKPOINT_KIND}_${OUTPUT_TAG}"

python infer_fedlppa_personalized.py \
    --root_path ../data/ODOC_h5 \
    --exp "${EXP_NAME}" \
    --checkpoint_kind "${CHECKPOINT_KIND}" \
    --output_dir "${OUTPUT_DIR}" \
    --method_name "${METHOD_NAME}_${CHECKPOINT_KIND}" \
    --model unet_univ5 \
    --img_class odoc \
    --num_classes 3 \
    --in_chns 3 \
    --img_size 384 \
    --min_num_clients 5 \
    --prompt universal \
    --attention dual \
    --dual_init aggregated \
    --label_prompt 1 \
    --gpu 0 \
    --site_labels SiteA SiteB SiteC SiteD SiteE \
    --clients client1 client2 client3 client4 client5 \
    --sup_type_list scribble scribble_noisy scribble_noisy keypoint block
