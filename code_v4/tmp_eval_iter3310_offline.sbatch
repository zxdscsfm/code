#!/bin/bash
#SBATCH --job-name=ODOCeval3310
#SBATCH --partition=v100_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/data/jianbingshen/yanghongji/FedLPPA/code_v4/logs/infer_3310_%j.log
set -euo pipefail
cd /data/jianbingshen/yanghongji/FedLPPA/code_v4
/data/jianbingshen/yanghongji/anaconda3/bin/conda run -n fed39v2 python infer_fedlppa_personalized.py \
  --root_path ../data/ODOC_h5 \
  --exp odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022 \
  --checkpoint_paths \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/client_0_iter_3310_dice_0.8711.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/client_1_iter_3310_dice_0.6963.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/client_2_iter_3310_dice_0.8857.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/client_3_iter_3310_dice_0.8704.pth \
  /data/jianbingshen/yanghongji/FedLPPA/model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/client_4_iter_3310_dice_0.8725.pth \
  --output_dir ../model/odoc/FedLPPA_v32_step1_geompl_paper_r500_l10_odoc_20260418_124252_seed2022/test_iter3310_offline \
  --method_name ODOC_step1_iter3310 \
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
