#!/bin/bash
#SBATCH --job-name=FedLPPA_FAZ_V2Prior
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4
RUN_TAG="v2_prioragg_promptonly_repro2_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="faz/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
mkdir -p "$LOG_DIR"

SUP_TYPES="scribble_noisy keypoint block box scribble"

BASE_ARGS="\
--root_path ../data/FAZ_h5 \
--num_classes 2 \
--in_chns 1 \
--img_class faz \
--exp ${EXP_NAME} \
--model unet_univ5 \
--max_iterations 30000 \
--iters 50 \
--eval_iters 50 \
--tsne_iters 500 \
--batch_size 12 \
--base_lr 0.01 \
--amp 0 \
--seed 2022 \
--server_address 127.0.0.1:8092 \
--strategy FedUniV2.1 \
--min_num_clients 5 \
--img_size 256 \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1 \
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
--quality_prior_agg_enabled 1 \
--quality_prior_agg_beta 0.5 \
--quality_prior_self_floor 0.60 \
--quality_prior_decay_start_iter -1 \
--quality_prior_decay_end_iter -1 \
--sup_type_list ${SUP_TYPES} \
--verbose_logging 1"

echo "Starting prior-guided prompt-only FedLPPA repro run. EXP=${EXP_NAME} Logs=${LOG_DIR}"

# 启动 Server
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role server --client client_all --sup_type mask --gpu 0 > "$LOG_DIR/server.log" 2>&1 &
sleep 30

# 启动 5 个 Clients
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 0 --client client1 --sup_type scribble_noisy --gpu 1 > "$LOG_DIR/client0.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 1 --client client2 --sup_type keypoint --gpu 2 > "$LOG_DIR/client1.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 2 --client client3 --sup_type block --gpu 3 > "$LOG_DIR/client2.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 3 --client client4 --sup_type box --gpu 4 > "$LOG_DIR/client3.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 4 --client client5 --sup_type scribble --gpu 5 > "$LOG_DIR/client4.log" 2>&1 &

wait
echo "Prior-guided prompt-only FedLPPA repro run finished. EXP=${EXP_NAME}"