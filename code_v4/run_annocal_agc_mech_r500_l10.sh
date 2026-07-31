#!/bin/bash
# AnnoCal-AGC under the same r500/l10 weak-label protocols.

set -euo pipefail

CONDA_PATH="${CONDA_PATH:-$HOME/anaconda3}"
CONDA_ENV="${CONDA_ENV:-fed39v2}"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONUNBUFFERED=1

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CODE_DIR="${REPO_ROOT}/code_v4"

cd "${CODE_DIR}"
mkdir -p logs
source "${CODE_DIR}/run_monitor_common.sh"

DATASET="${DATASET:-}"
if [ -z "${DATASET}" ]; then
    echo "DATASET must be one of: prostate, polyp, faz, isic, busi, tn3k, duts, glas, ebhiseg"
    exit 1
fi

case "${DATASET}" in
    prostate)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/PROSTATE_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-prostate}"
        IN_CHNS=1
        IMG_SIZE=384
        MIN_NUM_CLIENTS=6
        CLIENTS=(client1 client2 client3 client4 client5 client6)
        SUP_TYPES=(scribble keypoint scribble block scribble scribble)
        ;;
    polyp)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/POLYP_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-polyp}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    faz)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/FAZ_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-faz}"
        IN_CHNS=1
        IMG_SIZE=256
        MIN_NUM_CLIENTS=5
        CLIENTS=(client1 client2 client3 client4 client5)
        SUP_TYPES=(scribble keypoint scribble block scribble)
        ;;
    isic)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/ISIC_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-isic}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    busi)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/BUSI_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-busi}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    tn3k)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/TN3K_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-tn3k}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    duts)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/DUTS_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-duts}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    glas)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/GlaS_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-glas}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    ebhiseg)
        ROOT_PATH="${ROOT_PATH:-${REPO_ROOT}/data/EBHISeg_h5_rdsi3_sd}"
        IMG_CLASS="${IMG_CLASS:-ebhiseg}"
        IN_CHNS=3
        IMG_SIZE=384
        MIN_NUM_CLIENTS=4
        CLIENTS=(client1 client2 client3 client4)
        SUP_TYPES=(scribble keypoint scribble block)
        ;;
    *)
        echo "Unsupported DATASET=${DATASET}"
        exit 1
        ;;
esac

AGC_MODE="${AGC_MODE:-conservative}"
if [ "${AGC_MODE}" != "conservative" ] && [ "${AGC_MODE}" != "moderate" ]; then
    echo "AGC_MODE must be conservative or moderate"
    exit 1
fi

if [ "${AGC_MODE}" = "moderate" ]; then
    AGC_TAG="agcm"
else
    AGC_TAG="agcc"
fi

RUN_PREFIX="${RUN_PREFIX:-annocal_${AGC_TAG}_${DATASET}_r500_l10}"
case "${DATASET}_${AGC_MODE}" in
    prostate_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8951"
        ;;
    prostate_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8952"
        ;;
    polyp_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8953"
        ;;
    polyp_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8954"
        ;;
    faz_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8955"
        ;;
    faz_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8956"
        ;;
    isic_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8957"
        ;;
    isic_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8958"
        ;;
    busi_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8959"
        ;;
    busi_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8960"
        ;;
    tn3k_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8961"
        ;;
    tn3k_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8962"
        ;;
    duts_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8963"
        ;;
    duts_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8964"
        ;;
    glas_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8965"
        ;;
    glas_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8966"
        ;;
    ebhiseg_conservative)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8967"
        ;;
    ebhiseg_moderate)
        DEFAULT_SERVER_ADDRESS="127.0.0.1:8968"
        ;;
    *)
        echo "Unsupported DATASET/AGC_MODE combination: ${DATASET}_${AGC_MODE}"
        exit 1
        ;;
esac
SERVER_ADDRESS="${SERVER_ADDRESS:-${DEFAULT_SERVER_ADDRESS}}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-10}"
TSNE_ITERS="${TSNE_ITERS:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_TAG="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
EXP_NAME="${DATASET}/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ${ROOT_PATH} \
--num_classes 2 \
--in_chns ${IN_CHNS} \
--img_class ${IMG_CLASS} \
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
--min_num_clients ${MIN_NUM_CLIENTS} \
--img_size ${IMG_SIZE} \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1 \
--disable_tensorboard 1 \
--save_code_snapshot 0 \
--save_checkpoint_copies 0 \
--ala_max_epochs ${ALA_MAX_EPOCHS:-500} \
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
--wann_seed_support_erode_radius 1 \
--wann_seed_support_box_erode_radius 1 \
--wann_seed_support_block_erode_radius 1 \
--wann_soft_lambda 0.2 \
--wann_cons_lambda 0.05 \
--wann_soft_rampup_iters 800 \
--wann_cons_rampup_iters 800 \
--wann_sparse_adaptive_core 1 \
--wann_sparse_min_core_ratio 0.06 \
--wann_sparse_max_core_ratio 0.12 \
--wann_sparse_core_min_reliability 0.25 \
--wann_low_confident_thresh 0.95 \
--wann_target_core_ratio 0.08 \
--wann_core_deficit_soft_boost 2.0 \
--acg_enabled 1 \
--acg_lambda 0.08 \
--acg_warmup_iters 800 \
--acg_context_soft_foreground 1 \
--acg_context_margin_floor 0.35 \
--acg_context_margin_ceiling 0.85 \
--acg_context_band_width 0.15 \
--acg_context_band_over_weight 0.25 \
--agc_enabled 1 \
--agc_mode ${AGC_MODE} \
--agc_start_iter 1000 \
--agc_ramp_iters 800 \
--agc_connect_steps 64 \
--agc_conservative_tau 0.75 \
--agc_conservative_max_target_mult 0.5 \
--agc_conservative_max_image_ratio 0.03 \
--agc_conservative_weight 0.5 \
--agc_conservative_reliability 0.65 \
--agc_moderate_tau 0.65 \
--agc_moderate_max_target_mult 1.5 \
--agc_moderate_max_image_ratio 0.08 \
--agc_moderate_weight 0.35 \
--agc_moderate_reliability 0.55 \
--rgftd_enabled 0 \
--rgftd_v3_enabled 0 \
${EXTRA_ARGS}"

echo "Starting AnnoCal-AGC ${DATASET} run"
echo "AGC_MODE=${AGC_MODE}"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "ROOT_PATH=${ROOT_PATH}"
echo "SUP_TYPES=${SUP_TYPES[*]}"

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role server --client client_all --sup_type mask --gpu 0 > "${LOG_DIR}/server.log" 2>&1 &
SERVER_PID=$!
init_run_monitor "${SERVER_PID}" "client_all gpu=0" "${LOG_DIR}/server.log"

echo "Waiting for server to listen on ${SERVER_ADDRESS}"
SERVER_READY=0
SERVER_PORT="${SERVER_ADDRESS##*:}"
for _ in $(seq 1 180); do
    if (echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
    if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "Server exited before opening port. Check ${LOG_DIR}/server.log"
        wait "${SERVER_PID}" || true
        print_log_tail "${LOG_DIR}/server.log"
        cleanup_monitored_processes
        exit 1
    fi
    sleep 5
done

if [ "${SERVER_READY}" -ne 1 ]; then
    echo "Server did not open port ${SERVER_PORT} within 900 seconds"
    print_log_tail "${LOG_DIR}/server.log"
    cleanup_monitored_processes
    exit 1
fi

for idx in "${!CLIENTS[@]}"; do
    launch_flower_client "${idx}" "${CLIENTS[$idx]}" "${SUP_TYPES[$idx]}" "$((idx + 1))" "${LOG_DIR}/client${idx}.log"
done

monitor_flower_processes
echo "AnnoCal-AGC ${DATASET} run finished. EXP_NAME=${EXP_NAME}"
