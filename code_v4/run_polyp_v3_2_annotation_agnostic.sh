#!/bin/bash
# v3.2: annotation-agnostic / metadata-free FedLPPA
# This is the primary benchmark script for the new setting.
# The current polyp loader in this repo uses 4 clients: Domain1-4.

#SBATCH --job-name=FedLPPA_Polyp_V32Agnostic
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4
RUN_TAG="v3_2_polyp_annotation_agnostic_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="polyp/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
mkdir -p "$LOG_DIR"

# Polyp weak-label assignment in this repo:
# Domain1 -> keypoint, Domain2 -> scribble, Domain3 -> box, Domain4 -> block
SUP_TYPES="keypoint scribble box block"

BASE_ARGS="\
--root_path ../data/POLYP_h5 \
--num_classes 2 \
--in_chns 3 \
--img_class polyp \
--exp ${EXP_NAME} \
--model unet_univ6 \
--max_iterations 30000 \
--iters 50 \
--eval_iters 50 \
--tsne_iters 500 \
--batch_size 8 \
--base_lr 0.001 \
--amp 0 \
--seed 2022 \
--server_address 127.0.0.1:8095 \
--strategy FedUniV2.1 \
--min_num_clients 4 \
--img_size 384 \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 0 \
--prompt_agg similarity \
--prompt_agg_temp 1.0 \
--pack_train_images 0 \
--pack_client_prompts 0 \
--prompt_agg_keys basic \
--ala_threshold 0.1 \
--ala_num_pre_loss 10 \
--ala_max_init_epochs 10 \
--prompt_agg_lambda 0.8 \
--prompt_affinity_mode raw \
--prompt_agg_min_spread 1e-3 \
--annotation_agnostic 1 \
--geometry_guided 1 \
--geometry_num_bins 4 \
--geometry_distance_clip 64 \
--geometry_density_kernel 15 \
--geometry_near_radius 8 \
--geometry_mid_radius 24 \
--geometry_profile_momentum 0.9 \
--geometry_warmup_iters 2000 \
--geometry_route_lambda 0.5 \
--geometry_pseudo_weights 1.0,0.75,0.5,0.25 \
--quality_target_mode auto \
--asp_mode continuous \
--quality_static_dim 7 \
--quality_dynamic_dim 5 \
--quality_dim 16 \
--quality_hidden_dim 32 \
--quality_prompt_channels 2 \
--quality_agg_enabled 1 \
--quality_agg_alpha_p 0.55 \
--quality_agg_alpha_q 0.25 \
--quality_agg_alpha_c 0.20 \
--quality_agg_beta 1.0 \
--quality_self_floor 0.35 \
--quality_momentum 0.9 \
--quality_prior_weight 0.05 \
--quality_prior_agg_enabled 0 \
--branch_consistency_weight 0.10 \
--reliability_gate 1 \
--reliability_gate_hidden 32 \
--pseudo_conf_base 0.65 \
--pseudo_conf_scale 0.20 \
--sup_type_list ${SUP_TYPES} \
--verbose_logging 1"

echo "Starting v3.2 annotation-agnostic FedLPPA run on Polyp. EXP=${EXP_NAME} Logs=${LOG_DIR}"
echo "If your dataset root is not ../data/POLYP_h5, update --root_path before submitting."
echo "If OOM occurs, reduce --batch_size from 8 to 6 or 4."

python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role server --client client_all --sup_type mask --gpu 0 > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server to listen on 127.0.0.1:8095 ..."
SERVER_READY=0
for _ in $(seq 1 24); do
    if (echo > /dev/tcp/127.0.0.1/8095) >/dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 $SERVER_PID >/dev/null 2>&1; then
        echo "Server process exited before opening port 8095. Check $LOG_DIR/server.log"
        wait $SERVER_PID
        exit 1
    fi
    sleep 5
done

if [ "$SERVER_READY" -ne 1 ]; then
    echo "Server did not open port 8095 within 120s. Check $LOG_DIR/server.log"
    exit 1
fi

python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 0 --client client1 --sup_type keypoint --gpu 1 > "$LOG_DIR/client0.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 1 --client client2 --sup_type scribble --gpu 2 > "$LOG_DIR/client1.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 2 --client client3 --sup_type box --gpu 3 > "$LOG_DIR/client2.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 3 --client client4 --sup_type block --gpu 4 > "$LOG_DIR/client3.log" 2>&1 &

wait
echo "v3.2 annotation-agnostic Polyp run finished. EXP=${EXP_NAME}"
