#!/bin/bash
set -e

source /data/jianbingshen/yanghongji/anaconda3/etc/profile.d/conda.sh
conda activate fed39v2

cd /data/jianbingshen/yanghongji/FedLPPA/code_v4

for cid in 0 1 2 3 4 5; do
  echo "PROSTATE client${cid}"
  python -u test_client4onemod_FL_Personalize.py \
    --client client${cid} --cid ${cid} --num_classes 2 --in_chns 1 \
    --root_path /data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5 \
    --img_class prostate \
    --exp prostate/FedLPPA_cleanV1_prostate_paper_r500_l10_prostate_20260506_151150_seed2022 \
    --snapshot_path /data/jianbingshen/yanghongji/FedLPPA/model/prostate/FedLPPA_cleanV1_prostate_paper_r500_l10_prostate_20260506_151150_seed2022 \
    --min_num_clients 6 --model unet_univ5 --img_size 384 \
    --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --gpu 0 \
    > /data/jianbingshen/yanghongji/FedLPPA/code_v4/logs/eval_cleanV1_prostate_client${cid}.log 2>&1
done

for cid in 0 1 2 3 4; do
  echo "ODOC client${cid}"
  python -u test_client4onemod_FL_Personalize.py \
    --client client${cid} --cid ${cid} --num_classes 3 --in_chns 3 \
    --root_path /data/jianbingshen/yanghongji/FedLPPA/code_v4/../data/ODOC_h5 \
    --img_class odoc \
    --exp odoc/FedLPPA_cleanV1_odoc_paper_r500_l10_odoc_20260506_221615_seed2022 \
    --snapshot_path /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_cleanV1_odoc_paper_r500_l10_odoc_20260506_221615_seed2022 \
    --min_num_clients 5 --model unet_univ5 --img_size 384 \
    --prompt universal --attention dual --dual_init aggregated --label_prompt 1 --gpu 0 \
    > /data/jianbingshen/yanghongji/FedLPPA/code_v4/logs/eval_cleanV1_odoc_client${cid}.log 2>&1
done
