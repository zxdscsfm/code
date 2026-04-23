#!/bin/bash
# Micro ODOC run for V1 shared-teacher + local-student experiments.

#SBATCH --job-name=FedLPPA_STv1_micro_ODOC
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

cd /data/jianbingshen/yanghongji/FedLPPA_github_official_full/code_v4

RUN_PREFIX="${RUN_PREFIX:-student_teacher_v1_micro_odoc}"
RUN_TAG="${RUN_PREFIX}_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="odoc/FedLPPA_${RUN_TAG}"
LOG_DIR="logs/run_${RUN_TAG}"
SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8142}"
SEED="${SEED:-2022}"
ITERS="${ITERS:-5}"
EVAL_ITERS="${EVAL_ITERS:-5}"
TSNE_ITERS="${TSNE_ITERS:-50}"
MAX_ITERATIONS="${MAX_ITERATIONS:-250}"
TRAIN_SAMPLE_RATIO="${TRAIN_SAMPLE_RATIO:-0.25}"
VAL_SAMPLE_RATIO="${VAL_SAMPLE_RATIO:-0.25}"
TRAIN_SAMPLE_FLOOR="${TRAIN_SAMPLE_FLOOR:-20}"
VAL_SAMPLE_FLOOR="${VAL_SAMPLE_FLOOR:-20}"
KD_LAMBDA="${KD_LAMBDA:-0.2}"
KD_TEMPERATURE="${KD_TEMPERATURE:-2.0}"
KD_SCHEDULE="${KD_SCHEDULE:-constant}"
KD_CUTOFF_ITER="${KD_CUTOFF_ITER:--1}"
KD_DECAY_START_ITER="${KD_DECAY_START_ITER:-0}"
KD_DECAY_END_ITER="${KD_DECAY_END_ITER:--1}"
REFRESH_MODE="${REFRESH_MODE:-soft_ema}"
REFRESH_INTERVAL="${REFRESH_INTERVAL:-25}"
REFRESH_ALPHA="${REFRESH_ALPHA:-0.2}"
STUDENT_PERSISTENT="${STUDENT_PERSISTENT:-1}"
STUDENT_LR_MODE="${STUDENT_LR_MODE:-global_decay}"
STUDENT_LR_VALUE="${STUDENT_LR_VALUE:--1}"
KD_VETO="${KD_VETO:-0}"
KD_VETO_CONFIDENCE="${KD_VETO_CONFIDENCE:-0.85}"
KD_WEIGHT_MODE="${KD_WEIGHT_MODE:-hard_confidence_veto}"
REFRESH_VETO="${REFRESH_VETO:-0}"
REFRESH_SKIP_ON_TEACHER_DROP="${REFRESH_SKIP_ON_TEACHER_DROP:-1}"
REFRESH_SKIP_ON_STUDENT_BEST="${REFRESH_SKIP_ON_STUDENT_BEST:-1}"
REFRESH_TEACHER_DROP_WINDOW="${REFRESH_TEACHER_DROP_WINDOW:-3}"
REFRESH_TEACHER_DROP_EPSILON="${REFRESH_TEACHER_DROP_EPSILON:-0.0}"
REFRESH_SKIP_HISTORY_WINDOW="${REFRESH_SKIP_HISTORY_WINDOW:-32}"
ASYM_PHOTOMETRIC="${ASYM_PHOTOMETRIC:-0}"
ASYM_BRIGHTNESS="${ASYM_BRIGHTNESS:-0.2}"
ASYM_CONTRAST="${ASYM_CONTRAST:-0.2}"
ASYM_GAMMA="${ASYM_GAMMA:-0.2}"
ASYM_NOISE_STD="${ASYM_NOISE_STD:-0.05}"
ASYM_BLUR_PROB="${ASYM_BLUR_PROB:-0.3}"
ASYM_BLUR_KERNEL="${ASYM_BLUR_KERNEL:-3}"
EMA_ENABLED="${EMA_ENABLED:-0}"
EMA_DECAY="${EMA_DECAY:-0.99}"
EMA_LAMBDA="${EMA_LAMBDA:-0.05}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

RUN_TAG="${RUN_TAG}_seed${SEED}"
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
--train_sample_ratio ${TRAIN_SAMPLE_RATIO} \
--val_sample_ratio ${VAL_SAMPLE_RATIO} \
--train_sample_floor ${TRAIN_SAMPLE_FLOOR} \
--val_sample_floor ${VAL_SAMPLE_FLOOR} \
--prototype_filter_enabled 0 \
--student_teacher_v1 1 \
--student_teacher_teacher_track 1 \
--student_teacher_persistent ${STUDENT_PERSISTENT} \
--student_teacher_kd_lambda ${KD_LAMBDA} \
--student_teacher_kd_temperature ${KD_TEMPERATURE} \
--student_teacher_kd_schedule ${KD_SCHEDULE} \
--student_teacher_kd_cutoff_iter ${KD_CUTOFF_ITER} \
--student_teacher_kd_decay_start_iter ${KD_DECAY_START_ITER} \
--student_teacher_kd_decay_end_iter ${KD_DECAY_END_ITER} \
--student_teacher_refresh_mode ${REFRESH_MODE} \
--student_teacher_refresh_interval ${REFRESH_INTERVAL} \
--student_teacher_refresh_alpha ${REFRESH_ALPHA} \
--student_teacher_kd_veto ${KD_VETO} \
--student_teacher_kd_veto_confidence ${KD_VETO_CONFIDENCE} \
--student_teacher_kd_weight_mode ${KD_WEIGHT_MODE} \
--student_teacher_refresh_veto ${REFRESH_VETO} \
--student_teacher_refresh_skip_on_teacher_drop ${REFRESH_SKIP_ON_TEACHER_DROP} \
--student_teacher_refresh_skip_on_student_best ${REFRESH_SKIP_ON_STUDENT_BEST} \
--student_teacher_refresh_teacher_drop_window ${REFRESH_TEACHER_DROP_WINDOW} \
--student_teacher_refresh_teacher_drop_epsilon ${REFRESH_TEACHER_DROP_EPSILON} \
--student_teacher_refresh_skip_history_window ${REFRESH_SKIP_HISTORY_WINDOW} \
--student_teacher_asym_photometric ${ASYM_PHOTOMETRIC} \
--student_teacher_asym_brightness ${ASYM_BRIGHTNESS} \
--student_teacher_asym_contrast ${ASYM_CONTRAST} \
--student_teacher_asym_gamma ${ASYM_GAMMA} \
--student_teacher_asym_noise_std ${ASYM_NOISE_STD} \
--student_teacher_asym_blur_prob ${ASYM_BLUR_PROB} \
--student_teacher_asym_blur_kernel ${ASYM_BLUR_KERNEL} \
--student_teacher_ema_enabled ${EMA_ENABLED} \
--student_teacher_ema_decay ${EMA_DECAY} \
--student_teacher_ema_lambda ${EMA_LAMBDA} \
--student_teacher_student_lr_mode ${STUDENT_LR_MODE} \
--student_teacher_student_lr_value ${STUDENT_LR_VALUE} \
${EXTRA_ARGS}"

echo "Starting V1 shared-teacher/local-student micro ODOC run"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "SEED=${SEED}"
echo "SERVER_ADDRESS=${SERVER_ADDRESS}"
echo "MAX_ITERATIONS=${MAX_ITERATIONS} ITERS=${ITERS} EVAL_ITERS=${EVAL_ITERS}"
echo "Subset ratio: train=${TRAIN_SAMPLE_RATIO} val=${VAL_SAMPLE_RATIO}"
echo "Subset floors: train=${TRAIN_SAMPLE_FLOOR} val=${VAL_SAMPLE_FLOOR}"
echo "KD_LAMBDA=${KD_LAMBDA} KD_TEMPERATURE=${KD_TEMPERATURE} STUDENT_PERSISTENT=${STUDENT_PERSISTENT}"
echo "KD_SCHEDULE=${KD_SCHEDULE} KD_CUTOFF_ITER=${KD_CUTOFF_ITER} KD_DECAY_START_ITER=${KD_DECAY_START_ITER} KD_DECAY_END_ITER=${KD_DECAY_END_ITER}"
echo "REFRESH_MODE=${REFRESH_MODE} REFRESH_INTERVAL=${REFRESH_INTERVAL} REFRESH_ALPHA=${REFRESH_ALPHA}"
echo "KD_VETO=${KD_VETO} KD_VETO_CONFIDENCE=${KD_VETO_CONFIDENCE} KD_WEIGHT_MODE=${KD_WEIGHT_MODE}"
echo "REFRESH_VETO=${REFRESH_VETO} SKIP_TEACHER_DROP=${REFRESH_SKIP_ON_TEACHER_DROP} SKIP_STUDENT_BEST=${REFRESH_SKIP_ON_STUDENT_BEST} DROP_WINDOW=${REFRESH_TEACHER_DROP_WINDOW} DROP_EPS=${REFRESH_TEACHER_DROP_EPSILON} SKIP_HISTORY=${REFRESH_SKIP_HISTORY_WINDOW}"
echo "ASYM_PHOTOMETRIC=${ASYM_PHOTOMETRIC} BRIGHTNESS=${ASYM_BRIGHTNESS} CONTRAST=${ASYM_CONTRAST} GAMMA=${ASYM_GAMMA} NOISE_STD=${ASYM_NOISE_STD} BLUR_PROB=${ASYM_BLUR_PROB} BLUR_KERNEL=${ASYM_BLUR_KERNEL}"
echo "EMA_ENABLED=${EMA_ENABLED} EMA_DECAY=${EMA_DECAY} EMA_LAMBDA=${EMA_LAMBDA}"
echo "STUDENT_LR_MODE=${STUDENT_LR_MODE} STUDENT_LR_VALUE=${STUDENT_LR_VALUE}"

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
echo "V1 shared-teacher/local-student micro ODOC run finished. EXP_NAME=${EXP_NAME}"
