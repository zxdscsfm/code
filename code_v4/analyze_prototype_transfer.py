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


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype difference and helper-correlation analysis.")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--global_checkpoint", type=str, required=True)
    parser.add_argument("--functional_dir", type=str, required=True)
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
    parser.add_argument("--max_val_cases_per_site", type=int, default=0)
    parser.add_argument("--max_train_cases_per_site", type=int, default=0)
    parser.add_argument("--od_region_mode", type=str, default="inclusive", choices=["inclusive", "rim_only"])
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


def load_global_model(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    site_args = build_site_args(args, 0, args.clients[0], args.sup_type_list[0])
    if args.model != "unet_univ5":
        raise ValueError(f"Unsupported model for this analysis: {args.model}")
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
    checkpoint = torch.load(args.global_checkpoint, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    model.eval()
    model.to(device)
    return model, device


def prepare_input(image_np, device):
    image_np = np.asarray(image_np)
    tensor = torch.from_numpy(image_np).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor


def resize_mask(mask_np, hw):
    mask_t = torch.from_numpy(np.asarray(mask_np)).unsqueeze(0).unsqueeze(0).float()
    mask_t = F.interpolate(mask_t, size=hw, mode="nearest")
    return mask_t.squeeze(0).squeeze(0).long().cpu().numpy()


def iter_site_cases(args, client_name, sup_type, split, max_cases):
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
    limit = total if max_cases <= 0 else min(max_cases, total)
    for idx in range(limit):
        sample = dataset[idx]
        case_rel = dataset.sample_list[idx]
        yield case_rel, sample


def get_od_region(mask_np, od_region_mode):
    if od_region_mode == "inclusive":
        return mask_np >= 1
    if od_region_mode == "rim_only":
        return mask_np == 2
    raise ValueError(f"Unsupported od_region_mode: {od_region_mode}")


def get_gt_regions(mask_np, od_region_mode):
    mask_np = np.asarray(mask_np)
    return {
        "bg": mask_np == 0,
        "oc": mask_np == 1,
        "od": get_od_region(mask_np, od_region_mode),
    }


def get_weak_regions(weak_np, od_region_mode):
    weak_np = np.asarray(weak_np)
    labeled_mask = weak_np != 3
    return {
        "bg": weak_np == 0,
        "oc": weak_np == 1,
        "od": np.logical_and(labeled_mask, get_od_region(weak_np, od_region_mode)),
    }


def accumulate_site_prototype(model, device, image_np, region_masks, accum):
    with torch.no_grad():
        features = model.encoder(prepare_input(image_np, device))[-1]  # [1,C,H,W]
    features = features.squeeze(0).detach().cpu()  # [C,H,W]
    h, w = features.shape[-2:]
    channel_dim = features.shape[0]

    for region_name, region_mask in region_masks.items():
        region_mask_rs = resize_mask(region_mask.astype(np.uint8), (h, w)).astype(bool)
        count = int(region_mask_rs.sum())
        if count == 0:
            continue
        feat_flat = features[:, region_mask_rs]  # [C,N]
        if region_name not in accum:
            accum[region_name] = {"sum": torch.zeros(channel_dim), "count": 0}
        accum[region_name]["sum"] += feat_flat.sum(dim=1)
        accum[region_name]["count"] += count


def finalize_prototypes(accum):
    result = {}
    for region_name, stats in accum.items():
        if stats["count"] <= 0:
            result[region_name] = None
        else:
            vec = stats["sum"] / float(stats["count"])
            result[region_name] = vec.numpy()
    return result


def cosine_sim(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return np.nan
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom < 1e-12:
        return np.nan
    return float(np.dot(vec_a, vec_b) / denom)


def l2_dist(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return np.nan
    return float(np.linalg.norm(vec_a - vec_b))


def build_similarity_matrix(site_labels, prototypes_by_site, region_name):
    matrix = np.zeros((len(site_labels), len(site_labels)), dtype=np.float64)
    for i, site_i in enumerate(site_labels):
        for j, site_j in enumerate(site_labels):
            matrix[i, j] = cosine_sim(
                prototypes_by_site[site_i].get(region_name),
                prototypes_by_site[site_j].get(region_name),
            )
    return matrix


def save_matrix_csv(path, site_labels, matrix):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["site"] + site_labels)
        for label, row in zip(site_labels, matrix):
            writer.writerow([label] + [f"{x:.6f}" if np.isfinite(x) else "" for x in row])


def compute_gt_prototypes(args, model, device):
    prototypes = {}
    counts = {}
    for site_label, client_name, sup_type in zip(args.site_labels, args.clients, args.sup_type_list):
        accum = {}
        num_cases = 0
        for _, sample in iter_site_cases(args, client_name, sup_type, split="val", max_cases=args.max_val_cases_per_site):
            image_np = sample["image"]
            mask_np = sample["label"]
            region_masks = get_gt_regions(mask_np, args.od_region_mode)
            accumulate_site_prototype(model, device, image_np, region_masks, accum)
            num_cases += 1
        prototypes[site_label] = finalize_prototypes(accum)
        counts[site_label] = num_cases
    return prototypes, counts


def compute_train_gt_and_weak_prototypes(args, model, device):
    gt_prototypes = {}
    weak_prototypes = {}
    counts = {}
    for site_label, client_name, sup_type in zip(args.site_labels, args.clients, args.sup_type_list):
        gt_accum = {}
        weak_accum = {}
        num_cases = 0
        for case_rel, sample in iter_site_cases(args, client_name, sup_type, split="train", max_cases=args.max_train_cases_per_site):
            image_np = sample["image"]
            weak_np = sample["label"]
            case_path = os.path.join(args.root_path, case_rel)
            with h5py.File(case_path, "r") as h5f:
                gt_np = h5f["mask"][:]
            accumulate_site_prototype(model, device, image_np, get_gt_regions(gt_np, args.od_region_mode), gt_accum)
            accumulate_site_prototype(model, device, image_np, get_weak_regions(weak_np, args.od_region_mode), weak_accum)
            num_cases += 1
        gt_prototypes[site_label] = finalize_prototypes(gt_accum)
        weak_prototypes[site_label] = finalize_prototypes(weak_accum)
        counts[site_label] = num_cases
    return gt_prototypes, weak_prototypes, counts


def load_helper_matrix(functional_dir, site_labels):
    path = os.path.join(functional_dir, "aggregated_probe_metrics.csv")
    df = pd.read_csv(path)
    helper = {}
    for ref in site_labels:
        helper[ref] = {}
        for model_site in site_labels:
            sub = df[(df["reference_site"] == ref) & (df["model_site"] == model_site)]
            if sub.empty:
                helper[ref][model_site] = {"overall": np.nan, "oc": np.nan, "od": np.nan}
                continue
            oc = sub[sub["eval_class_idx"] == 1]["mean_dice"].mean()
            od = sub[sub["eval_class_idx"] == 2]["mean_dice"].mean()
            overall = np.nanmean([oc, od])
            helper[ref][model_site] = {"overall": float(overall), "oc": float(oc), "od": float(od)}
    return helper


def compute_spearman(site_labels, similarity_matrix, helper_matrix, metric_name):
    xs = []
    ys = []
    for i, ref in enumerate(site_labels):
        for j, model_site in enumerate(site_labels):
            if ref == model_site:
                continue
            sim = similarity_matrix[i, j]
            score = helper_matrix[ref][model_site][metric_name]
            if np.isfinite(sim) and np.isfinite(score):
                xs.append(sim)
                ys.append(score)
    if len(xs) < 3:
        return np.nan, np.nan, len(xs)
    rho, p = spearmanr(xs, ys)
    return float(rho), float(p), len(xs)


def compute_ranking_agreement(site_labels, similarity_matrix, helper_matrix, metric_name, prototype_metric):
    detail_rows = []
    summary_rows = []
    for i, ref in enumerate(site_labels):
        pairs = []
        for j, model_site in enumerate(site_labels):
            if ref == model_site:
                continue
            sim = similarity_matrix[i, j]
            score = helper_matrix[ref][model_site][metric_name]
            if np.isfinite(sim) and np.isfinite(score):
                pairs.append((model_site, float(sim), float(score)))
        if len(pairs) < 2:
            continue

        helper_desc = sorted(pairs, key=lambda x: (-x[2], x[0]))
        proto_desc = sorted(pairs, key=lambda x: (-x[1], x[0]))
        helper_asc = sorted(pairs, key=lambda x: (x[2], x[0]))
        proto_asc = sorted(pairs, key=lambda x: (x[1], x[0]))

        helper_top1 = helper_desc[0][0]
        proto_top1 = proto_desc[0][0]
        helper_worst1 = helper_asc[0][0]
        proto_worst1 = proto_asc[0][0]

        xs = [x[1] for x in pairs]
        ys = [x[2] for x in pairs]
        rho, p = spearmanr(xs, ys)
        rho = float(rho) if np.isfinite(rho) else np.nan
        p = float(p) if np.isfinite(p) else np.nan

        detail_rows.append(
            {
                "prototype_metric": prototype_metric,
                "helper_metric": metric_name,
                "reference_site": ref,
                "num_candidates": len(pairs),
                "prototype_top1": proto_top1,
                "helper_top1": helper_top1,
                "top1_match": int(proto_top1 == helper_top1),
                "prototype_worst1": proto_worst1,
                "helper_worst1": helper_worst1,
                "worst1_match": int(proto_worst1 == helper_worst1),
                "within_site_spearman_rho": rho,
                "within_site_p_value": p,
                "prototype_rank_desc": " > ".join([x[0] for x in proto_desc]),
                "helper_rank_desc": " > ".join([x[0] for x in helper_desc]),
            }
        )

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        sub = detail_df[
            (detail_df["prototype_metric"] == prototype_metric) &
            (detail_df["helper_metric"] == metric_name)
        ]
        summary_rows.append(
            {
                "prototype_metric": prototype_metric,
                "helper_metric": metric_name,
                "num_reference_sites": int(len(sub)),
                "top1_match_rate": float(sub["top1_match"].mean()),
                "worst1_match_rate": float(sub["worst1_match"].mean()),
                "mean_within_site_spearman_rho": float(sub["within_site_spearman_rho"].mean()),
            }
        )
        return detail_df, pd.DataFrame(summary_rows)

    return pd.DataFrame(), pd.DataFrame()


def compare_weak_vs_gt(site_labels, gt_prototypes, weak_prototypes):
    rows = []
    for site in site_labels:
        for region in ["bg", "oc", "od"]:
            gt_vec = gt_prototypes[site].get(region)
            weak_vec = weak_prototypes[site].get(region)
            rows.append(
                {
                    "site": site,
                    "region": region,
                    "cosine_weak_vs_gt": cosine_sim(weak_vec, gt_vec),
                    "l2_weak_vs_gt": l2_dist(weak_vec, gt_vec),
                }
            )
    return pd.DataFrame(rows)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def to_serializable_prototypes(prototypes):
    out = {}
    for site, regions in prototypes.items():
        out[site] = {}
        for region, vec in regions.items():
            out[site][region] = None if vec is None else [float(x) for x in vec.tolist()]
    return out


def write_summary(
    path,
    args,
    gt_counts,
    train_counts,
    same_site_region_df,
    spearman_rows,
    ranking_summary_df,
    weak_gt_df,
):
    lines = []
    lines.append("# Prototype Transfer Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- global_checkpoint: `{args.global_checkpoint}`")
    lines.append(f"- functional_dir: `{args.functional_dir}`")
    lines.append(f"- od_region_mode: `{args.od_region_mode}`")
    lines.append(f"- val_cases_per_site: `{args.max_val_cases_per_site if args.max_val_cases_per_site > 0 else 'all'}`")
    lines.append(f"- train_cases_per_site: `{args.max_train_cases_per_site if args.max_train_cases_per_site > 0 else 'all'}`")
    lines.append("")
    lines.append("## Validation 1: GT Prototype Differences")
    lines.append("")
    for site in args.site_labels:
        lines.append(f"- {site}: val cases = {gt_counts.get(site, 0)}")
    lines.append("")
    lines.append("### Same-site OD vs OC cosine")
    lines.append("")
    for _, row in same_site_region_df.iterrows():
        lines.append(f"- {row['site']}: OD-vs-OC cosine = {row['od_vs_oc_cosine']:.4f}")
    lines.append("")
    lines.append("## Validation 2: Prototype Similarity vs Helper Matrix")
    lines.append("")
    for row in spearman_rows:
        lines.append(
            f"- prototype={row['prototype_metric']}, helper={row['helper_metric']}: "
            f"Spearman rho = {row['spearman_rho']:.4f}, p = {row['p_value']:.4g}, n = {row['num_pairs']}"
        )
    lines.append("")
    lines.append("### Per-reference ranking agreement")
    lines.append("")
    for _, row in ranking_summary_df.iterrows():
        lines.append(
            f"- prototype={row['prototype_metric']}, helper={row['helper_metric']}: "
            f"top1 match rate = {row['top1_match_rate']:.4f}, "
            f"worst1 match rate = {row['worst1_match_rate']:.4f}, "
            f"mean within-site rho = {row['mean_within_site_spearman_rho']:.4f}, "
            f"sites = {int(row['num_reference_sites'])}"
        )
    lines.append("")
    lines.append("## Validation 3: Weak Prototype vs GT Prototype (train split)")
    lines.append("")
    for site in args.site_labels:
        lines.append(f"- {site}: train cases = {train_counts.get(site, 0)}")
    lines.append("")
    for _, row in weak_gt_df.iterrows():
        cos = row["cosine_weak_vs_gt"]
        l2 = row["l2_weak_vs_gt"]
        cos_str = "nan" if not np.isfinite(cos) else f"{cos:.4f}"
        l2_str = "nan" if not np.isfinite(l2) else f"{l2:.4f}"
        lines.append(f"- {row['site']} {row['region']}: cosine = {cos_str}, l2 = {l2_str}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    model, device = load_global_model(args)

    # Validation 1
    gt_prototypes, gt_counts = compute_gt_prototypes(args, model, device)
    save_json(os.path.join(args.output_dir, "gt_prototypes.json"), to_serializable_prototypes(gt_prototypes))

    sim_matrices = {}
    for region in ["bg", "oc", "od"]:
        matrix = build_similarity_matrix(args.site_labels, gt_prototypes, region)
        sim_matrices[region] = matrix
        save_matrix_csv(os.path.join(args.output_dir, f"gt_{region}_prototype_similarity.csv"), args.site_labels, matrix)

    same_site_rows = []
    for site in args.site_labels:
        same_site_rows.append(
            {
                "site": site,
                "od_vs_oc_cosine": cosine_sim(gt_prototypes[site].get("od"), gt_prototypes[site].get("oc")),
                "od_vs_bg_cosine": cosine_sim(gt_prototypes[site].get("od"), gt_prototypes[site].get("bg")),
                "oc_vs_bg_cosine": cosine_sim(gt_prototypes[site].get("oc"), gt_prototypes[site].get("bg")),
            }
        )
    same_site_region_df = pd.DataFrame(same_site_rows)
    same_site_region_df.to_csv(os.path.join(args.output_dir, "same_site_region_cosines.csv"), index=False)

    # Validation 2
    helper_matrix = load_helper_matrix(args.functional_dir, args.site_labels)
    combined_matrix = np.nanmean(np.stack([sim_matrices["od"], sim_matrices["oc"]], axis=0), axis=0)
    save_matrix_csv(os.path.join(args.output_dir, "gt_combined_od_oc_similarity.csv"), args.site_labels, combined_matrix)

    spearman_rows = []
    for proto_name, proto_matrix in [
        ("od", sim_matrices["od"]),
        ("oc", sim_matrices["oc"]),
        ("combined_od_oc", combined_matrix),
    ]:
        for helper_metric in ["overall", "oc", "od"]:
            rho, p, n = compute_spearman(args.site_labels, proto_matrix, helper_matrix, helper_metric)
            spearman_rows.append(
                {
                    "prototype_metric": proto_name,
                    "helper_metric": helper_metric,
                    "spearman_rho": rho,
                    "p_value": p,
                    "num_pairs": n,
                }
            )
    spearman_df = pd.DataFrame(spearman_rows)
    spearman_df.to_csv(os.path.join(args.output_dir, "prototype_helper_spearman.csv"), index=False)

    ranking_detail_parts = []
    ranking_summary_parts = []
    for proto_name, proto_matrix in [
        ("od", sim_matrices["od"]),
        ("oc", sim_matrices["oc"]),
        ("combined_od_oc", combined_matrix),
    ]:
        for helper_metric in ["overall", "oc", "od"]:
            detail_df, summary_df = compute_ranking_agreement(
                args.site_labels, proto_matrix, helper_matrix, helper_metric, proto_name
            )
            if not detail_df.empty:
                ranking_detail_parts.append(detail_df)
            if not summary_df.empty:
                ranking_summary_parts.append(summary_df)
    ranking_detail_df = pd.concat(ranking_detail_parts, ignore_index=True) if ranking_detail_parts else pd.DataFrame()
    ranking_summary_df = pd.concat(ranking_summary_parts, ignore_index=True) if ranking_summary_parts else pd.DataFrame()
    ranking_detail_df.to_csv(os.path.join(args.output_dir, "prototype_helper_ranking_detail.csv"), index=False)
    ranking_summary_df.to_csv(os.path.join(args.output_dir, "prototype_helper_ranking_summary.csv"), index=False)

    # Validation 3
    train_gt_prototypes, weak_prototypes, train_counts = compute_train_gt_and_weak_prototypes(args, model, device)
    save_json(os.path.join(args.output_dir, "train_gt_prototypes.json"), to_serializable_prototypes(train_gt_prototypes))
    save_json(os.path.join(args.output_dir, "train_weak_prototypes.json"), to_serializable_prototypes(weak_prototypes))
    weak_gt_df = compare_weak_vs_gt(args.site_labels, train_gt_prototypes, weak_prototypes)
    weak_gt_df.to_csv(os.path.join(args.output_dir, "weak_vs_gt_prototype_alignment.csv"), index=False)

    write_summary(
        os.path.join(args.output_dir, "summary.md"),
        args,
        gt_counts,
        train_counts,
        same_site_region_df,
        spearman_rows,
        ranking_summary_df,
        weak_gt_df,
    )

    print(f"Saved prototype analysis to: {args.output_dir}")


if __name__ == "__main__":
    main()
