#!/bin/bash
# Method-development proportional micro ODOC line.

#SBATCH --job-name=FedLPPA_method_prop_micro_ODOC
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:6
#SBATCH --mem=96G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA_github_official_micro_method/code_v4

RUN_TAG="method_prop_micro_odoc_$(date +%Y%m%d_%H%M%S)"
METHOD_TAG="${METHOD_TAG:-v32_step1_geompl}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8213}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-5}"
EVAL_ITERS="${EVAL_ITERS:-5}"
TSNE_ITERS="${TSNE_ITERS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-250}"
TRAIN_SAMPLE_RATIO="${TRAIN_SAMPLE_RATIO:-0.25}"
VAL_SAMPLE_RATIO="${VAL_SAMPLE_RATIO:-0.25}"
TRAIN_SAMPLE_FLOOR="${TRAIN_SAMPLE_FLOOR:-20}"
VAL_SAMPLE_FLOOR="${VAL_SAMPLE_FLOOR:-20}"

RUN_TAG="${METHOD_TAG}_${RUN_TAG}_seed${SEED}"
EXP_NAME="odoc/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"

mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ../data/ODOC_h5 \
--num_classes 3 \
--in_chns 3 \
--img_class odoc \
--exp ${EXP_NAME} \
--model unet_univ5 \
--max_iterations ${MAX_ITERATIONS} \
--iters ${ITERS} \
--eval_iters ${EVAL_ITERS} \
--tsne_iters ${TSNE_ITERS} \
--batch_size 8 \
--base_lr 0.01 \
--amp 0 \
--seed ${SEED} \
--server_address ${SERVER_ADDRESS} \
--strategy FedUniV2.1 \
--min_num_clients 5 \
--img_size 384 \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1 \
--geometry_guided 1 \
--geometry_num_bins 4 \
--geometry_near_radius 8.0 \
--geometry_mid_radius 24.0 \
--geometry_pseudo_weights 1.0,0.8,0.5,0.2 \
--train_sample_ratio ${TRAIN_SAMPLE_RATIO} \
--val_sample_ratio ${VAL_SAMPLE_RATIO} \
--train_sample_floor ${TRAIN_SAMPLE_FLOOR} \
--val_sample_floor ${VAL_SAMPLE_FLOOR}"

echo "Starting method-development proportional micro ODOC run"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "METHOD_TAG=${METHOD_TAG}"
echo "SEED=${SEED}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS} ITERS=${ITERS} EVAL_ITERS=${EVAL_ITERS}"
echo "Proportional subset: train_ratio=${TRAIN_SAMPLE_RATIO} val_ratio=${VAL_SAMPLE_RATIO}"
echo "Floors: train=${TRAIN_SAMPLE_FLOOR} val=${VAL_SAMPLE_FLOOR}"

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

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 0 --client client1 --sup_type scribble --gpu 1 > "${LOG_DIR}/client0.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 1 --client client2 --sup_type scribble_noisy --gpu 2 > "${LOG_DIR}/client1.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 2 --client client3 --sup_type scribble_noisy --gpu 3 > "${LOG_DIR}/client2.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 3 --client client4 --sup_type keypoint --gpu 4 > "${LOG_DIR}/client3.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 4 --client client5 --sup_type block --gpu 5 > "${LOG_DIR}/client4.log" 2>&1 &

wait
echo "Method-development proportional micro ODOC run finished. EXP_NAME=${EXP_NAME}"
