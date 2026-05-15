#!/bin/bash
# POLYP V1B + reliability gate:
# keep the uniform-geometry V1B backbone path,
# and add a lightweight local external-support gate.

#SBATCH --job-name=FedLPPA_POLYP_V1B-gate
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gres=gpu:5
#SBATCH --mem=128G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export METHOD_TAG="${METHOD_TAG:-v1b-gate_polyp_paper_r500_l10}"
export SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8210}"
export EXTRA_ARGS="${EXTRA_ARGS:---geometry_density_kernel 15 --reliability_gate_enabled 1 --reliability_gate_hidden_channels 16 --reliability_gate_warmup_iters 1200 --reliability_gate_support_kernel_size 11 --reliability_gate_agreement_lambda 0.05 --reliability_gate_conflict_lambda 0.10 --reliability_gate_balance_lambda 0.01}"

bash /data/jianbingshen/yanghongji/FedLPPA/code_v4/run_polyp_v1_uniform_geom_ablation_paper_r500_l10.sh
