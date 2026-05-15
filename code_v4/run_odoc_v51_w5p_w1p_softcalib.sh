#!/bin/bash
# ODOC W5':
# restored-W1' backbone + soft-only risk-aware calibration
# paper-aligned schedule: global 5000 / local 10

#SBATCH --job-name=FedLPPA_ODOC_W5P
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --gres=gpu:6
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

METHOD_TAG="${METHOD_TAG:-v51_w5p_w1p_softcalib_paper_r500_l10}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8178}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-10}"
TSNE_ITERS="${TSNE_ITERS:-200}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
ROOT_PATH="${ROOT_PATH:-../data/ODOC_h5}"
ADAPTIVE_TAU_UPDATE="${ADAPTIVE_TAU_UPDATE:-1}"
ADAPTIVE_RESUME_STATE="${ADAPTIVE_RESUME_STATE:-0}"

RUN_TAG="${METHOD_TAG}_odoc_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
EXP_NAME="odoc/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"

mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ${ROOT_PATH} \
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
--ala_threshold 0.1 \
--ala_num_pre_loss 10 \
--ala_max_init_epochs 10 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1 \
--geometry_guided 1 \
--geometry_num_bins 4 \
--geometry_near_radius 8.0 \
--geometry_mid_radius 24.0 \
--geometry_pseudo_weights 1.0,0.8,0.5,0.2 \
--adaptive_pl_enabled 1 \
--adaptive_pl_tau_init 0.55 \
--adaptive_pl_tau_update ${ADAPTIVE_TAU_UPDATE} \
--adaptive_pl_resume_state ${ADAPTIVE_RESUME_STATE} \
--adaptive_pl_target_accept 0.35 \
--adaptive_pl_warmup_iters 800 \
--adaptive_pl_soft_lambda 0.2 \
--adaptive_pl_min_pixels_per_bin 64 \
--adaptive_pl_log_interval 50 \
--adaptive_pl_gamma_prob 4.0 \
--adaptive_pl_gamma_conf 3.0 \
--adaptive_pl_blend_kappa 2.0 \
--adaptive_pl_global_min_conf 0.6 \
--adaptive_pl_boundary_lambda 0.0 \
--risk_calibration_enabled 1 \
--risk_calibration_tau_lambda 0.0 \
--risk_calibration_conf_lambda 0.0 \
--risk_calibration_prior_power 1.0 \
--risk_calibration_prior_clip_min 0.5 \
--risk_calibration_prior_clip_max 1.5 \
--risk_calibration_hard_dampen 0.0 \
--risk_calibration_soft_boost 0.5 \
--risk_calibration_global_soft_lambda 0.15 \
--risk_calibration_stateful_release_enabled 0 \
--risk_calibration_regime_ema_momentum 0.9 \
--risk_calibration_preserve_agreement_threshold 0.92 \
--risk_calibration_correction_agreement_threshold 0.88 \
--risk_calibration_risk_agreement_credit 0.20 \
--risk_calibration_w5_soft_only_enabled 1 \
--risk_calibration_w5_start_iter 800 \
--risk_calibration_w5_peak_iter 1400 \
--risk_calibration_w5_end_iter 2400 \
--risk_calibration_w5_min_correction 0.15 \
--risk_calibration_w5_risk_scale 0.60"

echo "Starting ODOC W5' restored-W1'-backbone soft-only risk calibration run"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "METHOD_TAG=${METHOD_TAG}"
echo "SEED=${SEED}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS} ITERS=${ITERS} EVAL_ITERS=${EVAL_ITERS} TSNE_ITERS=${TSNE_ITERS}"
echo "ADAPTIVE_TAU_UPDATE=${ADAPTIVE_TAU_UPDATE} ADAPTIVE_RESUME_STATE=${ADAPTIVE_RESUME_STATE}"

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
echo "ODOC W5' restored-W1'-backbone soft-only risk calibration run finished. EXP_NAME=${EXP_NAME}"
