import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dataloaders.dataset import BaseDataSets  # noqa: E402
from networks.unet import UNet_UniV5  # noqa: E402


DEFAULT_SITE_LABELS = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]
DEFAULT_CLIENTS = ["client1", "client2", "client3", "client4", "client5"]
DEFAULT_SUP_TYPES = ["scribble", "scribble_noisy", "scribble_noisy", "keypoint", "block"]
CLASS_NAMES = {0: "bg", 1: "oc", 2: "od"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype bank vs pseudo-label quality analysis.")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet_univ5")
    parser.add_argument("--img_class", type=str, default="odoc")
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--in_chns", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--min_num_clients", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="universal")
    parser.add_argument("--attention", type=str, default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_train_cases_per_site", type=int, default=0)
    parser.add_argument("--top_fracs", nargs="+", type=float, default=[0.1, 0.25, 0.5, 0.75])
    parser.add_argument("--bank_mode", type=str, default="pixel_pool", choices=["pixel_pool", "site_mean"])
    parser.add_argument("--site_labels", nargs="+", default=DEFAULT_SITE_LABELS)
    parser.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS)
    parser.add_argument("--sup_type_list", nargs="+", default=DEFAULT_SUP_TYPES)
    args = parser.parse_args()

    expected_len = args.min_num_clients
    for field_name in ["site_labels", "clients", "sup_type_list"]:
        values = getattr(args, field_name)
        if len(values) != expected_len:
            raise ValueError(f"{field_name} must have {expected_len} entries, got {len(values)}")
    return args


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_site_args(args, cid, client_name, sup_type):
    return SimpleNamespace(
        cid=cid,
        client=client_name,
        sup_type=sup_type,
        min_num_clients=args.min_num_clients,
        prompt=args.prompt,
        attention=args.attention,
        label_prompt=args.label_prompt,
        img_size=args.img_size,
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def prepare_input(image_np, device):
    image_np = np.asarray(image_np)
    tensor = torch.from_numpy(image_np).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor


def resize_label(mask_np, hw):
    mask_t = torch.from_numpy(np.asarray(mask_np)).unsqueeze(0).unsqueeze(0).float()
    mask_t = F.interpolate(mask_t, size=hw, mode="nearest")
    return mask_t.squeeze(0).squeeze(0).long().cpu().numpy()


def iter_site_cases(args, client_name, sup_type, split="train"):
    dataset_kwargs = dict(
        base_dir=args.root_path,
        split=split,
        transform=None,
        client=client_name,
        sup_type=sup_type,
        img_class=args.img_class,
    )
    dataset_signature = inspect.signature(BaseDataSets.__init__)
    filtered_kwargs = {k: v for k, v in dataset_kwargs.items() if k in dataset_signature.parameters}
    dataset = BaseDataSets(**filtered_kwargs)
    total = len(dataset)
    limit = total if args.max_train_cases_per_site <= 0 else min(args.max_train_cases_per_site, total)
    for idx in range(limit):
        sample = dataset[idx]
        case_rel = dataset.sample_list[idx]
        yield case_rel, sample


def load_site_models(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    models = {}
    if args.model != "unet_univ5":
        raise ValueError(f"Unsupported model for this analysis: {args.model}")

    for cid, (site_label, client_name, sup_type) in enumerate(zip(args.site_labels, args.clients, args.sup_type_list)):
        site_args = build_site_args(args, cid, client_name, sup_type)
        model = UNet_UniV5(
            in_chns=args.in_chns,
            class_num=args.num_classes,
            prompt_type=site_args.prompt,
            attention_type=site_args.attention,
            sup_type=site_args.sup_type,
            use_label_prompt=site_args.label_prompt,
            client_num=site_args.min_num_clients,
            client_id=site_args.cid,
            img_size=site_args.img_size,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)
        models[site_label] = model
    return models, device


def normalize_rows(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return arr / norms


def build_global_prototype_bank(args, models, device):
    accum = {cls_idx: None for cls_idx in CLASS_NAMES}
    counts = {cls_idx: 0 for cls_idx in CLASS_NAMES}
    per_site_counts = []
    per_site_prototypes = {}

    for site_label, client_name, sup_type in zip(args.site_labels, args.clients, args.sup_type_list):
        model = models[site_label]
        site_counts = {"site": site_label}
        site_accum = {cls_idx: None for cls_idx in CLASS_NAMES}
        site_class_counts = {cls_idx: 0 for cls_idx in CLASS_NAMES}
        for cls_idx in CLASS_NAMES:
            site_counts[f"{CLASS_NAMES[cls_idx]}_count"] = 0

        for _, sample in iter_site_cases(args, client_name, sup_type, split="train"):
            image_np = sample["image"]
            weak_np = sample["label"]
            with torch.no_grad():
                feat = model.encoder(prepare_input(image_np, device))[-1].squeeze(0).detach().cpu().numpy()
            c, h, w = feat.shape
            weak_rs = resize_label(weak_np, (h, w))
            feat_flat = feat.reshape(c, -1).T  # [N,C]
            weak_flat = weak_rs.reshape(-1)
            for cls_idx in CLASS_NAMES:
                cls_mask = weak_flat == cls_idx
                cls_count = int(cls_mask.sum())
                if cls_count == 0:
                    continue
                cls_sum = feat_flat[cls_mask].sum(axis=0)
                if accum[cls_idx] is None:
                    accum[cls_idx] = cls_sum.astype(np.float64)
                else:
                    accum[cls_idx] += cls_sum
                if site_accum[cls_idx] is None:
                    site_accum[cls_idx] = cls_sum.astype(np.float64)
                else:
                    site_accum[cls_idx] += cls_sum
                counts[cls_idx] += cls_count
                site_class_counts[cls_idx] += cls_count
                site_counts[f"{CLASS_NAMES[cls_idx]}_count"] += cls_count
        per_site_counts.append(site_counts)
        per_site_prototypes[site_label] = {}
        for cls_idx, cls_name in CLASS_NAMES.items():
            if site_class_counts[cls_idx] <= 0 or site_accum[cls_idx] is None:
                per_site_prototypes[site_label][cls_name] = None
            else:
                per_site_prototypes[site_label][cls_name] = (
                    site_accum[cls_idx] / float(site_class_counts[cls_idx])
                ).astype(np.float32)

    prototypes = {}
    if args.bank_mode == "pixel_pool":
        for cls_idx in CLASS_NAMES:
            if counts[cls_idx] <= 0 or accum[cls_idx] is None:
                prototypes[cls_idx] = None
            else:
                prototypes[cls_idx] = (accum[cls_idx] / float(counts[cls_idx])).astype(np.float32)
    elif args.bank_mode == "site_mean":
        for cls_idx, cls_name in CLASS_NAMES.items():
            site_vecs = [per_site_prototypes[site][cls_name] for site in args.site_labels if per_site_prototypes[site][cls_name] is not None]
            if not site_vecs:
                prototypes[cls_idx] = None
            else:
                prototypes[cls_idx] = np.mean(np.stack(site_vecs, axis=0), axis=0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported bank_mode: {args.bank_mode}")
    return prototypes, counts, pd.DataFrame(per_site_counts), per_site_prototypes


def read_gt_mask(root_path, case_rel):
    with h5py.File(os.path.join(root_path, case_rel), "r") as h5f:
        return h5f["mask"][:]


def gather_unlabeled_predictions(args, models, device, prototypes):
    proto_matrix = np.stack([prototypes[0], prototypes[1], prototypes[2]], axis=0)
    proto_matrix = normalize_rows(proto_matrix)

    rows = []
    for site_label, client_name, sup_type in zip(args.site_labels, args.clients, args.sup_type_list):
        model = models[site_label]
        for case_rel, sample in iter_site_cases(args, client_name, sup_type, split="train"):
            image_np = sample["image"]
            weak_np = sample["label"]
            gt_np = read_gt_mask(args.root_path, case_rel)
            x = prepare_input(image_np, device)
            with torch.no_grad():
                feat = model.encoder(x)[-1]
                logits = model(x)[0]

            feat_np = feat.squeeze(0).detach().cpu().numpy()
            c, h, w = feat_np.shape
            logits_low = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            probs = torch.softmax(logits_low, dim=1).squeeze(0).detach().cpu().numpy()
            pred = probs.argmax(axis=0)
            conf = probs.max(axis=0)

            weak_rs = resize_label(weak_np, (h, w))
            gt_rs = resize_label(gt_np, (h, w))
            unlabeled = weak_rs == 3

            feat_flat = feat_np.reshape(c, -1).T
            feat_flat = normalize_rows(feat_flat)
            sims = feat_flat @ proto_matrix.T

            pred_flat = pred.reshape(-1)
            conf_flat = conf.reshape(-1)
            gt_flat = gt_rs.reshape(-1)
            weak_flat = weak_rs.reshape(-1)
            unlabeled_flat = unlabeled.reshape(-1)

            pred_sim = sims[np.arange(sims.shape[0]), pred_flat]
            sims_for_margin = sims.copy()
            sims_for_margin[np.arange(sims.shape[0]), pred_flat] = -np.inf
            other_max = sims_for_margin.max(axis=1)
            sim_margin = pred_sim - other_max
            sim_shifted = np.clip((pred_sim + 1.0) / 2.0, 0.0, 1.0)
            combined = conf_flat * sim_shifted

            is_correct = pred_flat == gt_flat

            for class_group, class_selector in [
                ("all", np.ones_like(pred_flat, dtype=bool)),
                ("fg", np.isin(pred_flat, [1, 2])),
                ("oc", pred_flat == 1),
                ("od", pred_flat == 2),
            ]:
                group_mask = unlabeled_flat & class_selector
                group_idx = np.where(group_mask)[0]
                for idx in group_idx:
                    rows.append(
                        {
                            "site": site_label,
                            "case": case_rel,
                            "group": class_group,
                            "pred_class": int(pred_flat[idx]),
                            "gt_class": int(gt_flat[idx]),
                            "is_correct": int(is_correct[idx]),
                            "confidence": float(conf_flat[idx]),
                            "prototype_similarity": float(pred_sim[idx]),
                            "prototype_margin": float(sim_margin[idx]),
                            "combined_score": float(combined[idx]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_scores(df, top_fracs):
    rows = []
    for group in sorted(df["group"].unique().tolist()):
        sub = df[df["group"] == group].copy()
        if sub.empty:
            continue
        base_acc = float(sub["is_correct"].mean())
        n = int(len(sub))
        for score_name in ["confidence", "prototype_similarity", "prototype_margin", "combined_score"]:
            xs = sub[score_name].to_numpy()
            ys = sub["is_correct"].to_numpy()
            rho, p = spearmanr(xs, ys)
            row = {
                "group": group,
                "score_name": score_name,
                "num_pixels": n,
                "base_accuracy_all": base_acc,
                "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
            }
            ranked = sub.sort_values(score_name, ascending=False).reset_index(drop=True)
            for frac in top_fracs:
                k = max(1, int(round(n * frac)))
                top_acc = float(ranked.iloc[:k]["is_correct"].mean())
                row[f"top_{int(frac * 100)}_coverage"] = k
                row[f"top_{int(frac * 100)}_accuracy"] = top_acc
                row[f"top_{int(frac * 100)}_gain"] = top_acc - base_acc
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_sitewise(df):
    rows = []
    for (site, group), sub in df.groupby(["site", "group"]):
        base_acc = float(sub["is_correct"].mean())
        for score_name in ["confidence", "prototype_similarity", "prototype_margin", "combined_score"]:
            rho, p = spearmanr(sub[score_name].to_numpy(), sub["is_correct"].to_numpy())
            rows.append(
                {
                    "site": site,
                    "group": group,
                    "score_name": score_name,
                    "num_pixels": int(len(sub)),
                    "base_accuracy_all": base_acc,
                    "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_summary(path, args, prototype_counts, score_summary_df):
    lines = []
    lines.append("# Prototype Pseudo-Label Filter Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- bank_mode: `{args.bank_mode}`")
    lines.append(f"- train_cases_per_site: `{args.max_train_cases_per_site if args.max_train_cases_per_site > 0 else 'all'}`")
    lines.append(f"- top_fracs: `{args.top_fracs}`")
    lines.append("")
    lines.append("## Prototype Bank")
    lines.append("")
    for cls_idx, cls_name in CLASS_NAMES.items():
        lines.append(f"- {cls_name}: weak-labeled pixels = {prototype_counts[cls_idx]}")
    lines.append("")
    lines.append("## Score Summary")
    lines.append("")
    for _, row in score_summary_df.iterrows():
        lines.append(
            f"- group={row['group']}, score={row['score_name']}: "
            f"all_acc={row['base_accuracy_all']:.4f}, rho={row['spearman_rho']:.4f}, p={row['p_value']:.4g}"
        )
        for frac in args.top_fracs:
            frac_name = int(frac * 100)
            lines.append(
                f"  top{frac_name}% acc={row[f'top_{frac_name}_accuracy']:.4f} "
                f"(gain={row[f'top_{frac_name}_gain']:+.4f}, n={int(row[f'top_{frac_name}_coverage'])})"
            )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    models, device = load_site_models(args)

    prototypes, prototype_counts, prototype_site_counts_df, per_site_prototypes = build_global_prototype_bank(args, models, device)
    proto_json = {
        CLASS_NAMES[k]: None if v is None else [float(x) for x in v.tolist()]
        for k, v in prototypes.items()
    }
    save_json(os.path.join(args.output_dir, "global_weak_prototype_bank.json"), proto_json)
    per_site_proto_json = {}
    for site, regions in per_site_prototypes.items():
        per_site_proto_json[site] = {}
        for cls_name, vec in regions.items():
            per_site_proto_json[site][cls_name] = None if vec is None else [float(x) for x in vec.tolist()]
    save_json(os.path.join(args.output_dir, "site_weak_prototypes.json"), per_site_proto_json)
    prototype_site_counts_df.to_csv(os.path.join(args.output_dir, "prototype_bank_site_counts.csv"), index=False)

    pixel_df = gather_unlabeled_predictions(args, models, device, prototypes)
    pixel_df.to_csv(os.path.join(args.output_dir, "unlabeled_pixel_scores.csv"), index=False)

    score_summary_df = summarize_scores(pixel_df, args.top_fracs)
    score_summary_df.to_csv(os.path.join(args.output_dir, "score_summary.csv"), index=False)

    sitewise_df = summarize_sitewise(pixel_df)
    sitewise_df.to_csv(os.path.join(args.output_dir, "sitewise_spearman.csv"), index=False)

    write_summary(
        os.path.join(args.output_dir, "summary.md"),
        args,
        prototype_counts,
        score_summary_df,
    )
    print(f"Saved pseudo-label prototype analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
