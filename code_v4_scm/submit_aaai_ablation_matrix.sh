#!/usr/bin/env bash
# Submit AAAI ablation waves for the current RDSI-TED line.
#
# Important: the training code still uses legacy option names such as
# --rgftd_enabled for the TED/distillation path. Paper tables and run labels
# must use TED/RDSI-TED, not RGFTD.
#
# Default is DRY_RUN=1: print commands only.
# Example:
#   WAVE=prostate_core MODE=smoke DRY_RUN=0 bash submit_aaai_ablation_matrix.sh
#   WAVE=prostate_core MODE=full  DRY_RUN=0 bash submit_aaai_ablation_matrix.sh

set -euo pipefail

CODE_DIR="${CODE_DIR:-/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4}"
cd "${CODE_DIR}"

WAVE="${WAVE:-prostate_core}"
MODE="${MODE:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
SEED="${SEED:-2022}"

case "${MODE}" in
  smoke)
    MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
    ITERS="${ITERS:-5}"
    EVAL_ITERS="${EVAL_ITERS:-5}"
    TAG_MODE="smoke"
    ;;
  full)
    MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
    ITERS="${ITERS:-10}"
    EVAL_ITERS="${EVAL_ITERS:-10}"
    TAG_MODE="full"
    ;;
  *)
    echo "MODE must be smoke or full, got: ${MODE}" >&2
    exit 2
    ;;
esac

submit_job() {
  local prefix="$1"
  local script="$2"
  local port="$3"
  local extra_args="$4"

  local server_address="127.0.0.1:${port}"
  local run_prefix="ab_${prefix}_${TAG_MODE}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'RUN_PREFIX=%q SERVER_ADDRESS=%q SEED=%q MAX_ITERATIONS=%q ITERS=%q EVAL_ITERS=%q EXTRA_ARGS=%q sbatch %q\n' \
      "${run_prefix}" "${server_address}" "${SEED}" "${MAX_ITERATIONS}" "${ITERS}" "${EVAL_ITERS}" "${extra_args}" "${script}"
  else
    echo "Submitting ${run_prefix}"
    RUN_PREFIX="${run_prefix}" \
    SERVER_ADDRESS="${server_address}" \
    SEED="${SEED}" \
    MAX_ITERATIONS="${MAX_ITERATIONS}" \
    ITERS="${ITERS}" \
    EVAL_ITERS="${EVAL_ITERS}" \
    EXTRA_ARGS="${extra_args}" \
      sbatch "${script}"
  fi
}

prostate_core() {
  local script="run_prostate_rdsi3_sd_rgftd_r500_l10.sh"
  submit_job "pros_A0_base" "${script}" 8910 "--wann_enabled 0 --rgftd_enabled 0 --rgftd_v3_enabled 0 --rdsi_enabled 0"
  submit_job "pros_A1_wann" "${script}" 8911 "--rgftd_enabled 0 --rgftd_v3_enabled 0 --rdsi_enabled 0"
  submit_job "pros_A2_wann_ted_no_rdsi" "${script}" 8912 "--rdsi_enabled 0"
  submit_job "pros_A3_full" "${script}" 8913 ""
}

prostate_design() {
  local script="run_prostate_rdsi3_sd_rgftd_r500_l10.sh"
  # Reuse prostate_core A3 as Full and A2 as w/o RDSI in the design table.
  submit_job "pros_B2_no_teacher_val" "${script}" 8921 "--rgftd_teacher_validation_enabled 0"
  submit_job "pros_B3_no_rdsi_safety" "${script}" 8922 "--rdsi_boundary_support_weight 0 --rdsi_core_preserving_fg_weight 0 --rdsi_core_damage_weight 0 --rdsi_core_damage_veto 1.0 --rdsi_unsafe_gap_weight 0 --rdsi_foreground_excess_weight 0"
  submit_job "pros_B4_no_wann_soft_cons" "${script}" 8923 "--wann_soft_lambda 0 --wann_cons_lambda 0"
}

polyp_core() {
  local script="run_polyp_rdsi3_sd_rgftd_r500_l10.sh"
  submit_job "poly_P0_base" "${script}" 8930 "--wann_enabled 0 --rgftd_enabled 0 --rgftd_v3_enabled 0 --rdsi_enabled 0"
  submit_job "poly_P1_wann" "${script}" 8931 "--rgftd_enabled 0 --rgftd_v3_enabled 0 --rdsi_enabled 0"
  submit_job "poly_P2_wann_ted_no_rdsi" "${script}" 8932 "--rdsi_enabled 0"
  submit_job "poly_P3_full" "${script}" 8933 ""
}

case "${WAVE}" in
  prostate_core) prostate_core ;;
  prostate_design) prostate_design ;;
  polyp_core) polyp_core ;;
  all)
    prostate_core
    prostate_design
    polyp_core
    ;;
  *)
    echo "Unknown WAVE: ${WAVE}" >&2
    echo "Valid: prostate_core, prostate_design, polyp_core, all" >&2
    exit 2
    ;;
esac
