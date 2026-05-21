#!/bin/bash
# Evaluate a PROSTATE run with per-client own-best checkpoints.

#SBATCH --job-name=EVAL_PROSTATE_ORACLE
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

SRC_EXP="${SRC_EXP:?SRC_EXP is required, e.g. prostate/FedLPPA_xxx}"
ROOT_PATH="${ROOT_PATH:?ROOT_PATH is required}"
EVAL_TAG="${EVAL_TAG:-prostate/eval_oracle_client_best_$(date +%Y%m%d_%H%M%S)}"
MODEL_ROOT="${MODEL_ROOT:-../model}"
SRC_DIR="${MODEL_ROOT}/${SRC_EXP}"
LOG_DIR="${LOG_DIR:-logs/run_eval_prostate_oracle_client_best_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

echo "CODE_DIR=${CODE_DIR}"
echo "SRC_EXP=${SRC_EXP}"
echo "SRC_DIR=${SRC_DIR}"
echo "ROOT_PATH=${ROOT_PATH}"
echo "EVAL_TAG=${EVAL_TAG}"
echo "MODEL_ROOT=${MODEL_ROOT}"
echo "LOG_DIR=${LOG_DIR}"

python - <<'PY' "${SRC_DIR}" "${MODEL_ROOT}" "${EVAL_TAG}" "${LOG_DIR}"
import glob
import os
import re
import shutil
import sys

src_dir, model_root, eval_tag, log_dir = sys.argv[1:5]
selected_path = os.path.join(log_dir, "selected_checkpoints.txt")
with open(selected_path, "w", encoding="utf-8") as out:
    for cid in range(6):
        best = None
        for path in glob.glob(os.path.join(src_dir, f"client_{cid}*_iter_*_dice_*.pth")):
            match = re.search(rf"client_{cid}(?:_async)?_iter_(\d+)_dice_([0-9.]+)\.pth$", path)
            if not match:
                continue
            iteration = int(match.group(1))
            dice = float(match.group(2))
            is_async = "_async_" in os.path.basename(path)
            key = (dice, is_async, iteration)
            if best is None or key > best[0]:
                best = (key, path)
        if best is None:
            raise RuntimeError(f"No checkpoint found for client {cid} in {src_dir}")
        eval_exp = f"{eval_tag}/client{cid}"
        dst_dir = os.path.join(model_root, eval_exp)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, "unet_best_model.pth")
        shutil.copy2(best[1], dst_path)
        out.write(f"client{cid},{best[0][0]},{best[0][2]},{'async' if best[0][1] else 'sync'},{best[1]},{eval_exp}\n")
        print(f"client{cid}: {best[1]} -> {dst_path}")
PY

SUP_TYPES=(block keypoint scribble keypoint scribble block)
for cid in 0 1 2 3 4 5; do
    eval_exp="${EVAL_TAG}/client${cid}"
    client_name="client${cid}"
    echo "Evaluating ${client_name} with exp=${eval_exp}"
    python -u test_client4onemod_FL_Personalize.py \
        --client "${client_name}" \
        --num_classes 2 \
        --in_chns 1 \
        --root_path "${ROOT_PATH}" \
        --img_class prostate \
        --exp "${eval_exp}" \
        --min_num_clients 6 \
        --cid "${cid}" \
        --model unet_univ5 \
        --img_size 384 \
        --sup_type "${SUP_TYPES[$cid]}" \
        --label_prompt 1 \
        > "${LOG_DIR}/client${cid}_test.log" 2>&1
done

python - <<'PY' "${MODEL_ROOT}" "${EVAL_TAG}" "${LOG_DIR}"
import os
import sys
import pandas as pd

model_root, eval_tag, log_dir = sys.argv[1:4]
rows = []
for cid in range(6):
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
summary.loc["avg"] = {
    "client": "avg",
    "dice": summary["dice"].mean(),
    "hd95": summary["hd95"].mean(),
    "jaccard": summary["jaccard"].mean(),
    "assd": summary["assd"].mean(),
    "se": summary["se"].mean(),
    "sp": summary["sp"].mean(),
    "rec": summary["rec"].mean(),
    "pre": summary["pre"].mean(),
    "csv": "",
}
out_path = os.path.join(log_dir, "summary.csv")
summary.to_csv(out_path, index=False)
print(summary.to_string(index=False))
print(f"summary={out_path}")
PY
