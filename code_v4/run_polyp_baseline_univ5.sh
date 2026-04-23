#!/bin/bash
# Polyp baseline: original FedLPPA-style configuration on unet_univ5.
# This script is intended as the clean reference before evaluating v3.2.

#SBATCH --job-name=FedLPPA_Polyp_Baseline
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
RUN_TAG="v3_0_polyp_baseline_univ5_$(date +%Y%m%d_%H%M%S)"
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
--model unet_univ5 \
--max_iterations 30000 \
--iters 5 \
--eval_iters 5 \
--tsne_iters 200 \
--batch_size 8 \
--base_lr 0.001 \
--amp 0 \
--seed 2022 \
--server_address 127.0.0.1:8096 \
--strategy FedUniV2.1 \
--min_num_clients 4 \
--img_size 384 \
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
--gatedcrf_weight 0.0 \
--verbose_logging 1"

echo "Starting Polyp baseline on unet_univ5. EXP=${EXP_NAME} Logs=${LOG_DIR}"
echo "This run is the clean baseline reference before annotation-agnostic / geometry-guided variants."

python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role server --client client_all --sup_type mask --gpu 0 > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server to listen on 127.0.0.1:8096 ..."
SERVER_READY=0
for _ in $(seq 1 24); do
    if (echo > /dev/tcp/127.0.0.1/8096) >/dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 $SERVER_PID >/dev/null 2>&1; then
        echo "Server process exited before opening port 8096. Check $LOG_DIR/server.log"
        wait $SERVER_PID
        exit 1
    fi
    sleep 5
done

if [ "$SERVER_READY" -ne 1 ]; then
    echo "Server did not open port 8096 within 120s. Check $LOG_DIR/server.log"
    exit 1
fi

python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 0 --client client1 --sup_type keypoint --gpu 1 > "$LOG_DIR/client0.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 1 --client client2 --sup_type scribble --gpu 2 > "$LOG_DIR/client1.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 2 --client client3 --sup_type box --gpu 3 > "$LOG_DIR/client2.log" 2>&1 &
python flower_pCE_2D_v4_FedLPPA.py $BASE_ARGS --role client --cid 3 --client client4 --sup_type block --gpu 4 > "$LOG_DIR/client3.log" 2>&1 &

wait
echo "Polyp baseline run finished. EXP=${EXP_NAME}"
