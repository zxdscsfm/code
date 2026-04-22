import argparse
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path",
        type=str,
        default="/data/jianbingshen/yanghongji/FedLPPA/data/ODOC_h5",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument("--client", type=str, default="client2")
    parser.add_argument("--sup_type", type=str, default="scribble_noisy")
    parser.add_argument("--cid", type=int, default=1)
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


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    model_args = build_args(args.cid, args.sup_type)
    model = net_factory(model_args, net_type="unet_univ5", in_chns=3, class_num=3)
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    model.eval()

    dataset = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        client=args.client,
        sup_type=args.sup_type,
        img_class="odoc",
    )
    prototypes, counts = build_prototypes(model, dataset)

    oc_margins, oc_labels = [], []
    od_margins, od_labels = [], []

    with torch.no_grad():
        for case, sample in zip(dataset.sample_list, dataset.data_list):
            image = torch.from_numpy(sample["image"]).unsqueeze(0).cuda().float()
            weak = torch.from_numpy(sample["label"]).unsqueeze(0).cuda().long()
            with h5py.File(os.path.join(args.root_path, case), "r") as f:
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

    result = {
        "site": "SiteB",
        "client": args.client,
        "sup_type": args.sup_type,
        "checkpoint": args.checkpoint,
        "prototype_counts": {
            "bg": int(counts[0]),
            "oc": int(counts[1]),
            "od_ring": int(counts[2]),
        },
        "oc": summarize_class(oc_margins, oc_labels),
        "od_ring": summarize_class(od_margins, od_labels),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
