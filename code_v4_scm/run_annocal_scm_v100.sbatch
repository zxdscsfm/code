#!/bin/bash
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:5
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4_scm/logs/main_scm_%j.log

set -euo pipefail

cd /data/jianbingshen/yanghongji/FedLPPA_Original/code_v4_scm
bash run_annocal_scm_r500_l10.sh
