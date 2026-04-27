#!/bin/bash
# External launcher for a clean official FedLPPA baseline on PROSTATE.

#SBATCH --job-name=FedLPPA_Clean_PROSTATE
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_clean_launchers/logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

REPO_ROOT="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean"
CODE_DIR="${REPO_ROOT}/code_v4"
LAUNCH_ROOT="/data/jianbingshen/yanghongji/FedLPPA_clean_launchers"
mkdir -p "${LAUNCH_ROOT}/logs"

cd "${CODE_DIR}"

RUN_PREFIX="${RUN_PREFIX:-official_clean_prostate_paper_r500_l10}"
RUN_TAG="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8165}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-10}"
TSNE_ITERS="${TSNE_ITERS:-200}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT_PATH="${ROOT_PATH:-/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5}"

RUN_TAG="${RUN_TAG}_seed${SEED}"
EXP_NAME="prostate/FedLPPA_${RUN_TAG}"
LOG_DIR="${LAUNCH_ROOT}/logs/run_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ${ROOT_PATH} \
--num_classes 2 \
--in_chns 1 \
--img_class prostate \
--exp ${EXP_NAME} \
--model unet_univ5 \
--max_iterations ${MAX_ITERATIONS} \
--iters ${ITERS} \
--eval_iters ${EVAL_ITERS} \
--tsne_iters ${TSNE_ITERS} \
--batch_size ${BATCH_SIZE} \
--base_lr 0.01 \
--amp 0 \
--seed ${SEED} \
--server_address ${SERVER_ADDRESS} \
--strategy FedUniV2.1 \
--min_num_clients 6 \
--img_size 384 \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1"

echo "Starting clean official FedLPPA PROSTATE baseline"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"

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

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 0 --client client1 --sup_type block --gpu 1 > "${LOG_DIR}/client0.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 1 --client client2 --sup_type keypoint --gpu 2 > "${LOG_DIR}/client1.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 2 --client client3 --sup_type scribble --gpu 3 > "${LOG_DIR}/client2.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 3 --client client4 --sup_type keypoint --gpu 4 > "${LOG_DIR}/client3.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 4 --client client5 --sup_type scribble --gpu 5 > "${LOG_DIR}/client4.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 5 --client client6 --sup_type box --gpu 6 > "${LOG_DIR}/client5.log" 2>&1 &

wait
echo "Clean official FedLPPA PROSTATE baseline finished. EXP_NAME=${EXP_NAME}"
