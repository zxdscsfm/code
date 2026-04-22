#!/bin/bash
# Submit A/B/C/D micro ODOC student-teacher experiments on Slurm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_micro_odoc_fedlppa_student_teacher_v1.sh"

submit_run() {
    local tag="$1"
    local port="$2"
    shift 2
    echo "Submitting ${tag} on port ${port}"
    sbatch --export=ALL,RUN_PREFIX="${tag}",SERVER_ADDRESS="127.0.0.1:${port}",$* "${BASE_SCRIPT}"
}

submit_run "stA_micro_baseline" 8142 \
    KD_VETO=0,REFRESH_VETO=0,ASYM_PHOTOMETRIC=0,EMA_ENABLED=0

submit_run "stB_micro_veto" 8143 \
    KD_VETO=1,REFRESH_VETO=1,ASYM_PHOTOMETRIC=0,EMA_ENABLED=0

submit_run "stC_micro_veto_asym" 8144 \
    KD_VETO=1,REFRESH_VETO=1,ASYM_PHOTOMETRIC=1,EMA_ENABLED=0

submit_run "stD_micro_veto_asym_ema" 8145 \
    KD_VETO=1,REFRESH_VETO=1,ASYM_PHOTOMETRIC=1,EMA_ENABLED=1
