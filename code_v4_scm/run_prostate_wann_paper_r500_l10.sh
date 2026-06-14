#!/bin/bash
# Official FedLPPA + WANN on PROSTATE.

#SBATCH --job-name=FedLPPA_PROSTATE_WANN
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:7
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

REPO_ROOT="/data/jianbingshen/yanghongji/FedLPPA_Original"
CODE_DIR="${REPO_ROOT}/code_v4"

cd "${CODE_DIR}"
mkdir -p logs

RUN_PREFIX="${RUN_PREFIX:-official_wann_prostate_paper_r500_l10}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8512}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-10}"
TSNE_ITERS="${TSNE_ITERS:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT_PATH="${ROOT_PATH:-/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_TAG="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
EXP_NAME="prostate/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
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
--label_prompt 1 \
--wann_enabled 1 \
--wann_core_thresh 0.65 \
--wann_soft_thresh 0.25 \
--wann_core_min_weight 0.8 \
--wann_r_max 1.2 \
--wann_dilated_support_score 0.55 \
--wann_appearance_temp 1.5 \
--wann_texture_kernel_size 5 \
--wann_texture_temp 1.0 \
--wann_texture_weight 0.25 \
--wann_pred_start_iter 800 \
--wann_entropy_weight 0.5 \
--wann_agreement_weight 0.5 \
--wann_global_agreement_weight 0.5 \
--wann_keypoint_soft_radius 2 \
--wann_scribble_soft_radius 4 \
--wann_box_soft_radius 2 \
--wann_mask_soft_radius 1 \
--wann_soft_lambda 0.2 \
--wann_cons_lambda 0.05 \
--wann_soft_rampup_iters 800 \
--wann_cons_rampup_iters 800 \
${EXTRA_ARGS}"

echo "Starting official FedLPPA + WANN PROSTATE run"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"

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
echo "Official FedLPPA + WANN PROSTATE run finished. EXP_NAME=${EXP_NAME}"
