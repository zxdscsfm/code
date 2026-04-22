import argparse
import glob
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory


IGNORE = 3
RATIOS = [0.01, 0.05, 0.10]
SITE_CONFIGS = [
    {"site": "SiteA", "client": "client1", "sup_type": "scribble", "cid": 0},
    {"site": "SiteB", "client": "client2", "sup_type": "scribble_noisy", "cid": 1},
    {"site": "SiteC", "client": "client3", "sup_type": "scribble_noisy", "cid": 2},
    {"site": "SiteD", "client": "client4", "sup_type": "keypoint", "cid": 3},
    {"site": "SiteE", "client": "client5", "sup_type": "block", "cid": 4},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        default="/data/jianbingshen/yanghongji/FedLPPA/data/ODOC_h5",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--iter_tag",
        type=int,
        default=3800,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    return parser.parse_args()


def resize_long(mask_tensor, size_hw):
    return F.interpolate(mask_tensor.unsqueeze(1).float(), size=size_hw, mode="nearest").squeeze(1).long()


def load_state(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def build_args(cid, sup_type):
    return SimpleNamespace(
        min_num_clients=5,
        cid=cid,
        prompt="universal",
        attention="dual",
        sup_type=sup_type,
        label_prompt=1,
        img_size=384,
    )


def summarize_class(margins, labels):
    labels = labels.bool()
    result = {
        "count": int(labels.numel()),
        "pos_count": int(labels.sum().item()),
        "prevalence": float(labels.float().mean().item()) if labels.numel() else None,
        "pos_mean": float(margins[labels].mean().item()) if labels.any() else None,
        "neg_mean": float(margins[~labels].mean().item()) if (~labels).any() else None,
    }
    for ratio in RATIOS:
        key = f"top{int(ratio * 100)}"
        if labels.numel() == 0:
            result[f"{key}_precision"] = None
            result[f"{key}_recall"] = None
            continue
        k = max(1, int(np.ceil(labels.numel() * ratio)))
        topk_idx = torch.topk(margins, k=k, largest=True).indices
        hit = labels[topk_idx].float()
        result[f"{key}_precision"] = float(hit.mean().item())
        result[f"{key}_recall"] = float(hit.sum().item() / max(1, labels.sum().item()))
    return result


def build_prototypes(model, dataset):
    accum = [None, None, None]
    counts = [0, 0, 0]
    model.eval()
    with torch.no_grad():
        for sample in dataset.data_list:
            image = torch.from_numpy(sample["image"]).unsqueeze(0).cuda().float()
            weak = torch.from_numpy(sample["label"]).unsqueeze(0).cuda().long()
            feature = model.encoder(image)[-1]
            weak_low = resize_long(weak, feature.shape[-2:])
            feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
            weak_flat = weak_low.reshape(-1)
            for cls in [0, 1, 2]:
                class_mask = weak_flat == cls
                class_count = int(class_mask.sum().item())
                if class_count <= 0:
                    continue
                class_sum = feature_flat[class_mask].sum(dim=0)
                accum[cls] = class_sum if accum[cls] is None else accum[cls] + class_sum
                counts[cls] += class_count

    prototypes = []
    for cls in [0, 1, 2]:
        if counts[cls] <= 0 or accum[cls] is None:
            raise RuntimeError(f"missing prototype for class {cls}")
        prototypes.append(accum[cls] / float(counts[cls]))
    prototypes = torch.stack(prototypes, dim=0)
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes, counts


def find_checkpoint(model_dir, cid, iter_tag):
    pattern = os.path.join(model_dir, f"client_{cid}_iter_{iter_tag}_dice_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matches {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple checkpoints match {pattern}: {matches}")
    return matches[0]


def validate_site(root_path, model_dir, iter_tag, config):
    checkpoint = find_checkpoint(model_dir, config["cid"], iter_tag)
    model_args = build_args(config["cid"], config["sup_type"])
    model = net_factory(model_args, net_type="unet_univ5", in_chns=3, class_num=3)
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.eval()

    dataset = BaseDataSets(
        base_dir=root_path,
        split="train",
        client=config["client"],
        sup_type=config["sup_type"],
        img_class="odoc",
    )
    prototypes, counts = build_prototypes(model, dataset)

    oc_margins, oc_labels = [], []
    od_margins, od_labels = [], []

    with torch.no_grad():
        for case, sample in zip(dataset.sample_list, dataset.data_list):
            image = torch.from_numpy(sample["image"]).unsqueeze(0).cuda().float()
            weak = torch.from_numpy(sample["label"]).unsqueeze(0).cuda().long()
            with h5py.File(os.path.join(root_path, case), "r") as f:
                gt = torch.from_numpy(f["mask"][:]).unsqueeze(0).cuda().long()

            feature = model.encoder(image)[-1]
            feature_flat = F.normalize(
                feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1]),
                dim=1,
            )
            sims = torch.matmul(feature_flat, prototypes.t()).reshape(
                1, feature.shape[-2], feature.shape[-1], 3
            )
            weak_low = resize_long(weak, feature.shape[-2:])
            gt_low = resize_long(gt, feature.shape[-2:])
            unlabeled = weak_low == IGNORE
            if unlabeled.sum().item() == 0:
                continue

            sim_bg = sims[..., 0]
            sim_oc = sims[..., 1]
            sim_od = sims[..., 2]
            oc_margin = (sim_oc - torch.maximum(sim_bg, sim_od))[unlabeled]
            od_margin = (sim_od - torch.maximum(sim_bg, sim_oc))[unlabeled]
            oc_true = (gt_low == 1)[unlabeled]
            od_true = (gt_low == 2)[unlabeled]

            oc_margins.append(oc_margin.detach().cpu())
            oc_labels.append(oc_true.detach().cpu())
            od_margins.append(od_margin.detach().cpu())
            od_labels.append(od_true.detach().cpu())

    oc_margins = torch.cat(oc_margins) if oc_margins else torch.empty(0)
    oc_labels = torch.cat(oc_labels) if oc_labels else torch.empty(0, dtype=torch.bool)
    od_margins = torch.cat(od_margins) if od_margins else torch.empty(0)
    od_labels = torch.cat(od_labels) if od_labels else torch.empty(0, dtype=torch.bool)

    return {
        "site": config["site"],
        "client": config["client"],
        "cid": config["cid"],
        "sup_type": config["sup_type"],
        "checkpoint": checkpoint,
        "prototype_counts": {
            "bg": int(counts[0]),
            "oc": int(counts[1]),
            "od_ring": int(counts[2]),
        },
        "oc": summarize_class(oc_margins, oc_labels),
        "od_ring": summarize_class(od_margins, od_labels),
    }


def build_summary(results):
    summary_rows = []
    for row in results:
        summary_rows.append(
            {
                "site": row["site"],
                "client": row["client"],
                "sup_type": row["sup_type"],
                "oc_prevalence": row["oc"]["prevalence"],
                "oc_top1_precision": row["oc"]["top1_precision"],
                "oc_top5_precision": row["oc"]["top5_precision"],
                "oc_top10_precision": row["oc"]["top10_precision"],
                "oc_top10_recall": row["oc"]["top10_recall"],
                "od_top1_precision": row["od_ring"]["top1_precision"],
                "od_top5_precision": row["od_ring"]["top5_precision"],
                "od_top10_precision": row["od_ring"]["top10_precision"],
                "od_top10_recall": row["od_ring"]["top10_recall"],
            }
        )
    return summary_rows


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = [validate_site(args.root_path, args.model_dir, args.iter_tag, config) for config in SITE_CONFIGS]
    payload = {
        "iter_tag": args.iter_tag,
        "model_dir": args.model_dir,
        "sites": results,
        "summary": build_summary(results),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
