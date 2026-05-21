#!/bin/bash
# Evaluate sparse_scribble_5 per-client async-best checkpoints.

#SBATCH --job-name=EVAL_S5_PROS
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/main_%j.log

set -euo pipefail

export CONDA_PATH="/data/jianbingshen/yanghongji/anaconda3"
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate fed39v2
export PYTHONUNBUFFERED=1

CODE_DIR="${CODE_DIR:-/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4}"
cd "${CODE_DIR}"
mkdir -p logs

DATASET="${DATASET:-prostate}"
SRC_EXP="${SRC_EXP:-prostate/FedLPPA_sota_fedlppa_prostate_sparse_scribble5_r500_l10_20260520_064938_seed2022}"
EVAL_TAG="${EVAL_TAG:-prostate/eval_sota_sparse5_async_client_best_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="${MODEL_ROOT:-../model}"
SRC_DIR="${MODEL_ROOT}/${SRC_EXP}"
LOG_DIR="logs/run_eval_sparse5_${DATASET}_async_client_best_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

case "${DATASET}" in
  prostate)
    NUM_CLIENTS=6; NUM_CLASSES=2; IN_CHNS=1; IMG_CLASS=prostate; IMG_SIZE=384; ROOT_PATH=/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5 ;;
  polyp)
    NUM_CLIENTS=4; NUM_CLASSES=2; IN_CHNS=3; IMG_CLASS=polyp; IMG_SIZE=384; ROOT_PATH=/data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5 ;;
  faz)
    NUM_CLIENTS=5; NUM_CLASSES=2; IN_CHNS=1; IMG_CLASS=faz; IMG_SIZE=256; ROOT_PATH=/data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5 ;;
  *) echo "Unknown DATASET=${DATASET}"; exit 1 ;;
esac

echo "CODE_DIR=${CODE_DIR}"
echo "DATASET=${DATASET}"
echo "SRC_EXP=${SRC_EXP}"
echo "SRC_DIR=${SRC_DIR}"
echo "EVAL_TAG=${EVAL_TAG}"
echo "LOG_DIR=${LOG_DIR}"
echo "ROOT_PATH=${ROOT_PATH}"

python - <<'PY' "${SRC_DIR}" "${MODEL_ROOT}" "${EVAL_TAG}" "${LOG_DIR}" "${NUM_CLIENTS}"
import os
import shutil
import sys
src_dir, model_root, eval_tag, log_dir, num_clients = sys.argv[1:6]
num_clients = int(num_clients)
if not os.path.isdir(src_dir):
    raise RuntimeError(f"Missing source checkpoint dir: {src_dir}")
selected_path = os.path.join(log_dir, "selected_checkpoints.txt")
with open(selected_path, "w", encoding="utf-8") as out:
    for cid in range(num_clients):
        candidates = [
            os.path.join(src_dir, f"client_{cid}_async_unet_univ5_best_model.pth"),
            os.path.join(src_dir, f"client_{cid}_unet_univ5_best_model.pth"),
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if src is None:
            raise RuntimeError(f"No best checkpoint found for client {cid} in {src_dir}")
        eval_exp = f"{eval_tag}/client{cid}"
        dst_dir = os.path.join(model_root, eval_exp)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, "unet_best_model.pth")
        shutil.copy2(src, dst_path)
        out.write(f"client{cid},{src},{dst_path},{eval_exp}\n")
        print(f"client{cid}: {src} -> {dst_path}")
PY

for cid in $(seq 0 $((NUM_CLIENTS - 1))); do
    eval_exp="${EVAL_TAG}/client${cid}"
    client_name="client${cid}"
    echo "Evaluating ${client_name} with exp=${eval_exp}"
    python -u test_client4onemod_FL_Personalize.py \
        --client "${client_name}" \
        --num_classes "${NUM_CLASSES}" \
        --in_chns "${IN_CHNS}" \
        --root_path "${ROOT_PATH}" \
        --img_class "${IMG_CLASS}" \
        --exp "${eval_exp}" \
        --min_num_clients "${NUM_CLIENTS}" \
        --cid "${cid}" \
        --model unet_univ5 \
        --img_size "${IMG_SIZE}" \
        --sup_type sparse_scribble_5 \
        --label_prompt 1 \
        > "${LOG_DIR}/client${cid}_test.log" 2>&1
done

python - <<'PY' "${MODEL_ROOT}" "${EVAL_TAG}" "${LOG_DIR}" "${NUM_CLIENTS}"
import os
import sys
import pandas as pd
model_root, eval_tag, log_dir, num_clients = sys.argv[1:5]
num_clients = int(num_clients)
rows = []
for cid in range(num_clients):
    csv_path = os.path.join(model_root, f"{eval_tag}/client{cid}_test/client{cid}/mean_std_result.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"Missing result CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    mean = df[df["name"] == "mean"].iloc[0]
    rows.append({
        "client": cid,
        "dice": float(mean["dice"]),
        "hd95": float(mean["HD95"]),
        "jaccard": float(mean["jaccard"]),
        "assd": float(mean["ASSD"]),
        "se": float(mean["SE"]),
        "sp": float(mean["SP"]),
        "rec": float(mean["Rec"]),
        "pre": float(mean["Pre"]),
        "csv": csv_path,
    })
summary = pd.DataFrame(rows)
avg = {"client": "avg", "csv": ""}
for col in ["dice", "hd95", "jaccard", "assd", "se", "sp", "rec", "pre"]:
    avg[col] = summary[col].mean()
summary = pd.concat([summary, pd.DataFrame([avg])], ignore_index=True)
out_path = os.path.join(log_dir, "summary.csv")
summary.to_csv(out_path, index=False)
print(summary.to_string(index=False))
print(f"summary={out_path}")
PY
