#!/bin/bash
#SBATCH --job-name=ODOCeval1760
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA/code_v4/logs/infer_1760_%j.log
set -euo pipefail
cd /data/jianbingshen/yanghongji/FedLPPA/code_v4
/data/jianbingshen/yanghongji/anaconda3/bin/conda run -n fed39v2 python infer_fedlppa_personalized.py \
  --root_path ../data/ODOC_h5 \
  --exp odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022 \
  --checkpoint_paths \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/client_0_iter_1760_dice_0.812.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/client_1_iter_1760_dice_0.3674.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/client_2_iter_1760_dice_0.8529.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/client_3_iter_1760_dice_0.8623.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/client_4_iter_1760_dice_0.8211.pth \
  --output_dir ../model/odoc/FedLPPA_v35_w1p_d2boundary_paper_r500_l10_odoc_20260419_165129_seed2022/test_i1760_manual \
  --method_name ODOC_v35_D2_i1760 \
  --model unet_univ5 \
  --img_class odoc \
  --num_classes 3 \
  --in_chns 3 \
  --img_size 384 \
  --min_num_clients 5 \
  --prompt universal \
  --attention dual \
  --dual_init aggregated \
  --label_prompt 1 \
  --gpu 0 \
  --site_labels SiteA SiteB SiteC SiteD SiteE \
  --clients client1 client2 client3 client4 client5 \
  --sup_type_list scribble scribble_noisy scribble_noisy keypoint block
