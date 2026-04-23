#!/bin/bash
# Formal async_best inference for completed Polyp baseline run 118882.

#SBATCH --job-name=PolypInfer118882
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/infer_118882_%j.log

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

EXP_NAME="polyp/FedLPPA_v3_0_polyp_baseline_univ5_20260404_081120"
OUTPUT_DIR="../model/polyp/FedLPPA_v3_0_polyp_baseline_univ5_20260404_081120/test_asyncbest_$(date +%Y%m%d_%H%M%S)"

python infer_fedlppa_personalized.py \
    --root_path ../data/POLYP_h5 \
    --exp ${EXP_NAME} \
    --checkpoint_kind async_best \
    --output_dir ${OUTPUT_DIR} \
    --method_name "FedLPPA_v3_0_polyp_baseline_async_best" \
    --model unet_univ5 \
    --img_class polyp \
    --num_classes 2 \
    --in_chns 3 \
    --img_size 384 \
    --min_num_clients 4 \
    --prompt universal \
    --attention dual \
    --label_prompt 1 \
    --gpu 0 \
    --site_labels SiteA SiteB SiteC SiteD \
    --clients client1 client2 client3 client4 \
    --sup_type_list keypoint scribble box block
