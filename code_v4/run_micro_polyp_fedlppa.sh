#!/bin/bash
# Micro Polyp benchmark for difficult-set validation.

#SBATCH --job-name=FedLPPA_micro_Polyp
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:5
#SBATCH --mem=96G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

RUN_TAG="micro_polyp_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="polyp/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8103}"
ITERS="${ITERS:-5}"
EVAL_ITERS="${EVAL_ITERS:-5}"
TSNE_ITERS="${TSNE_ITERS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-250}"
MAX_TRAIN_SAMPLES_PER_CLIENT="${MAX_TRAIN_SAMPLES_PER_CLIENT:-80}"
MAX_VAL_SAMPLES_PER_CLIENT="${MAX_VAL_SAMPLES_PER_CLIENT:-20}"

mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ../data/POLYP_h5 \
--num_classes 2 \
--in_chns 3 \
--img_class polyp \
--exp ${EXP_NAME} \
--model unet_univ5 \
--max_iterations ${MAX_ITERATIONS} \
--iters ${ITERS} \
--eval_iters ${EVAL_ITERS} \
--tsne_iters ${TSNE_ITERS} \
--batch_size 12 \
--base_lr 0.01 \
--amp 0 \
--seed 2022 \
--server_address ${SERVER_ADDRESS} \
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
--prompt_affinity_mode raw \
--prompt_agg_keys basic \
--prompt_agg_temp 1.0 \
--prompt_agg_lambda 0.8 \
--pack_train_images 0 \
--pack_client_prompts 0 \
--max_train_samples_per_client ${MAX_TRAIN_SAMPLES_PER_CLIENT} \
--max_val_samples_per_client ${MAX_VAL_SAMPLES_PER_CLIENT}"

echo "Starting micro Polyp benchmark"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS} ITERS=${ITERS} EVAL_ITERS=${EVAL_ITERS}"
echo "Per-client subset: train=${MAX_TRAIN_SAMPLES_PER_CLIENT} val=${MAX_VAL_SAMPLES_PER_CLIENT}"

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role server --client client_all --sup_type mask --gpu 0 > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for server to listen on ${SERVER_ADDRESS}"
SERVER_READY=0
SERVER_PORT="${SERVER_ADDRESS##*:}"
for _ in $(seq 1 24); do
    if (echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "Server exited before opening port. Check ${LOG_DIR}/server.log"
        wait "${SERVER_PID}" || true
        exit 1
    fi
    sleep 5
done

if [ "${SERVER_READY}" -ne 1 ]; then
    echo "Server did not open port ${SERVER_PORT} within 120 seconds"
    exit 1
fi

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 0 --client client1 --sup_type keypoint --gpu 1 > "${LOG_DIR}/client0.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 1 --client client2 --sup_type scribble --gpu 2 > "${LOG_DIR}/client1.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 2 --client client3 --sup_type box --gpu 3 > "${LOG_DIR}/client2.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 3 --client client4 --sup_type block --gpu 4 > "${LOG_DIR}/client3.log" 2>&1 &

wait
echo "Micro Polyp benchmark finished. EXP_NAME=${EXP_NAME}"
