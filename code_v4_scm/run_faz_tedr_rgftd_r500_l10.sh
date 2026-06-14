#!/bin/bash
# FedRAP-RDSI on FAZ with scribble-dominant heterogeneous weak labels.

#SBATCH --job-name=TEDR_FAZ
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:6
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

RUN_PREFIX="${RUN_PREFIX:-tedr_faz_r500_l10}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8635}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-10}"
EVAL_ITERS="${EVAL_ITERS:-10}"
TSNE_ITERS="${TSNE_ITERS:-0}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ROOT_PATH="${ROOT_PATH:-/data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd}"
CLIENT1_SUP_TYPE="${CLIENT1_SUP_TYPE:-scribble}"
CLIENT2_SUP_TYPE="${CLIENT2_SUP_TYPE:-keypoint}"
CLIENT3_SUP_TYPE="${CLIENT3_SUP_TYPE:-scribble}"
CLIENT4_SUP_TYPE="${CLIENT4_SUP_TYPE:-block}"
CLIENT5_SUP_TYPE="${CLIENT5_SUP_TYPE:-scribble}"
WANN_PRED_START_ITER="${WANN_PRED_START_ITER:-800}"
WANN_SOFT_RAMPUP_ITERS="${WANN_SOFT_RAMPUP_ITERS:-800}"
WANN_CONS_RAMPUP_ITERS="${WANN_CONS_RAMPUP_ITERS:-800}"
RGFTD_WARMUP_ITERS="${RGFTD_WARMUP_ITERS:-800}"
RGFTD_RAMPUP_ITERS="${RGFTD_RAMPUP_ITERS:-800}"
RGFTD_V3_AUDIT_START_ITERS="${RGFTD_V3_AUDIT_START_ITERS:-800}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_TAG="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
EXP_NAME="faz/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

BASE_ARGS="\
--root_path ${ROOT_PATH} \
--num_classes 2 \
--in_chns 1 \
--img_class faz \
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
--min_num_clients 5 \
--img_size 256 \
--alpha 0.1 \
--beta 0.5 \
--prompt universal \
--attention dual \
--dual_init aggregated \
--label_prompt 1 \
--disable_tensorboard 1 \
--save_code_snapshot 0 \
--save_checkpoint_copies 0 \
--ala_max_epochs 500 \
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
--wann_pred_start_iter ${WANN_PRED_START_ITER} \
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
--wann_soft_rampup_iters ${WANN_SOFT_RAMPUP_ITERS} \
--wann_cons_rampup_iters ${WANN_CONS_RAMPUP_ITERS} \
--wann_sparse_adaptive_core 1 \
--wann_sparse_min_core_ratio 0.06 \
--wann_sparse_max_core_ratio 0.12 \
--wann_sparse_core_min_reliability 0.25 \
--wann_low_confident_thresh 0.95 \
--wann_target_core_ratio 0.08 \
--wann_core_deficit_soft_boost 2.0 \
--rgftd_enabled 1 \
--rgftd_lambda 0.1 \
--rgftd_warmup_iters ${RGFTD_WARMUP_ITERS} \
--rgftd_rampup_iters ${RGFTD_RAMPUP_ITERS} \
--rgftd_light_audit_enabled 1 \
--rgftd_light_audit_start_iters 200 \
--rgftd_light_audit_lambda 0.01 \
--rgftd_light_audit_core_ratio_thresh 0.05 \
--rgftd_light_audit_low_maxp_thresh 0.95 \
--rgftd_teacher_ema_decay 0.99 \
--rgftd_teacher_conf_thresh 0.90 \
--rgftd_teacher_bg_conf_thresh 0.98 \
--rgftd_bg_max_fg_prob 0.15 \
--rgftd_student_conf_thresh 0.80 \
--rgftd_student_entropy_thresh 0.35 \
--rgftd_low_r_thresh 0.25 \
--rgftd_temperature 1.0 \
--rgftd_use_soft_band 0 \
--rgftd_background_weight 0.25 \
--rgftd_teacher_validation_enabled 1 \
--rgftd_teacher_reliability_min 0.55 \
--rgftd_teacher_core_agree_floor 0.80 \
--rgftd_teacher_support_agree_floor 0.70 \
--rgftd_teacher_support_prob_floor 0.35 \
--rgftd_teacher_conf_floor 0.85 \
--rgftd_teacher_class_reliability_min 0.50 \
--rgftd_teacher_max_core_conflict 0.20 \
--rgftd_teacher_score_core_weight 0.35 \
--rgftd_teacher_score_support_weight 0.30 \
--rgftd_teacher_score_class_weight 0.25 \
--rgftd_teacher_score_conf_weight 0.10 \
--rgftd_teacher_release_prob_floor 0.35 \
--rgftd_teacher_release_margin_floor 0.05 \
--rgftd_teacher_release_class_floor 0.50 \
--rgftd_teacher_release_min 0.03 \
--rgftd_active_fg_topk_ratio 0.002 \
--rgftd_active_fg_topk_min_pixels 8 \
--rgftd_active_fg_topk_max_pixels 4096 \
--rgftd_max_bg_fg_ratio 1.0 \
--rgftd_allow_bg_without_fg 0 \
--rgftd_lambda_eff_cap 0.02 \
--rgftd_spatial_support_enabled 1 \
--rgftd_spatial_support_radius 2 \
--rgftd_spatial_candidate_weight 1.0 \
--rgftd_spatial_near_seed_weight 0.75 \
--rgftd_spatial_far_weight 0.15 \
--rgftd_refine_enabled 1 \
--rgftd_refine_iters 3 \
--rgftd_refine_affinity_sigma 0.75 \
--rgftd_refine_affinity_mix 0.35 \
--rgftd_refine_seed_strength 0.95 \
--rgftd_refine_core_anchor_radius 1 \
--rgftd_refine_unsupported_fg_scale 0.25 \
--rgftd_refine_fg_floor 0.02 \
--rgftd_refine_bg_ceiling 0.98 \
--rgftd_refine_min_fg_mass 1.0 \
--rgftd_refine_min_roi_pixels 1.0 \
--rgftd_v3_enabled 1 \
--rdsi_teacher_sup_types "${CLIENT1_SUP_TYPE},${CLIENT2_SUP_TYPE},${CLIENT3_SUP_TYPE},${CLIENT4_SUP_TYPE},${CLIENT5_SUP_TYPE}" \
--rgftd_v3_stable_teacher_enabled 0 \
--rgftd_v3_server_ema_fallback -1 \
--rgftd_v3_teacher_pool_topk 1 \
--rgftd_v3_audit_start_iters ${RGFTD_V3_AUDIT_START_ITERS} \
--rgftd_v3_audit_interval_iters 1000 \
--rgftd_v3_audit_batches 4 \
--rgftd_v3_audit_score_thresh 0.05 \
--rgftd_v3_audit_teacher_reliability_min 0.30 \
--rgftd_v3_fallback_lambda_eff_cap 0.01 \
--rgftd_v3_benefit_enabled 1 \
--rgftd_v3_lease_iters 0 \
--rgftd_v3_benefit_momentum 0.80 \
--rgftd_v3_benefit_good_thresh 0.70 \
--rgftd_v3_benefit_decay_thresh 0.50 \
--rgftd_v3_benefit_revoke_thresh 0.35 \
--rgftd_v3_routing_benefit_floor 0.25 \
--rgftd_v3_recover_audit_score_thresh 0.60 \
--rgftd_v3_decay_cap_scale 0.50 \
--rgftd_v3_min_cap_scale 0.25 \
--rgftd_v3_bgfg_warn_thresh 0.75 \
--rgftd_v3_cap_hit_penalty 0.20 \
--rgftd_v3_lowmaxp_delta_thresh 0.02 \
--rgftd_v3_wann_mass_drop_thresh 0.05 \
--rgftd_stable_min_score 0.05 \
--rgftd_stable_update_margin 0.01 \
--rgftd_stable_seed_prob_floor 0.35 \
--rgftd_stable_seed_margin_floor 0.05 \
--rgftd_stable_max_core_conflict 0.20 \
--rgftd_stable_lowmaxp_delta_thresh 0.02 \
--rgftd_stable_wann_mass_drop_thresh 0.05 \
${EXTRA_ARGS}"

echo "Starting official FedLPPA + WANN + RGFTD-v3.5 stable FAZ sparse5 run"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "ROOT_PATH=${ROOT_PATH}"
echo "SUP_TYPES=${CLIENT1_SUP_TYPE},${CLIENT2_SUP_TYPE},${CLIENT3_SUP_TYPE},${CLIENT4_SUP_TYPE},${CLIENT5_SUP_TYPE}"
echo "RELEASE_START=${RGFTD_WARMUP_ITERS}"
echo "RELEASE_FULL=$((RGFTD_WARMUP_ITERS + RGFTD_RAMPUP_ITERS))"

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

python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 0 --client client1 --sup_type "${CLIENT1_SUP_TYPE}" --gpu 1 > "${LOG_DIR}/client0.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 1 --client client2 --sup_type "${CLIENT2_SUP_TYPE}" --gpu 2 > "${LOG_DIR}/client1.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 2 --client client3 --sup_type "${CLIENT3_SUP_TYPE}" --gpu 3 > "${LOG_DIR}/client2.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 3 --client client4 --sup_type "${CLIENT4_SUP_TYPE}" --gpu 4 > "${LOG_DIR}/client3.log" 2>&1 &
python -u flower_pCE_2D_v4_FedLPPA.py ${BASE_ARGS} --role client --cid 4 --client client5 --sup_type "${CLIENT5_SUP_TYPE}" --gpu 5 > "${LOG_DIR}/client4.log" 2>&1 &

wait
echo "FedRAP-RDSI FAZ rdsi3_sd run finished. EXP_NAME=${EXP_NAME}"
