import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from dataloaders.dataset import BaseDataSets, infer_unlabeled_value_from_label


DATASETS = {
    "odoc": {
        "root": "../data/ODOC_h5",
        "clients": [
            ("client1", "scribble"),
            ("client2", "scribble_noisy"),
            ("client3", "scribble_noisy"),
            ("client4", "keypoint"),
            ("client5", "block"),
        ],
    },
    "faz": {
        "root": "/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/FAZ_h5",
        "clients": [
            ("client1", "scribble_noisy"),
            ("client2", "keypoint"),
            ("client3", "block"),
            ("client4", "box"),
            ("client5", "scribble"),
        ],
    },
    "prostate": {
        "root": "/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5",
        "clients": [
            ("client1", "block"),
            ("client2", "keypoint"),
            ("client3", "scribble"),
            ("client4", "keypoint"),
            ("client5", "scribble"),
            ("client6", "box"),
        ],
    },
    "polyp": {
        "root": "/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/POLYP_h5",
        "clients": [
            ("client1", "keypoint"),
            ("client2", "scribble"),
            ("client3", "box"),
            ("client4", "block"),
        ],
    },
}


def extract_labeled_and_fg_masks(label, img_class):
    label = np.asarray(label)
    unlabeled_value = infer_unlabeled_value_from_label(label, img_class)
    if unlabeled_value is None:
        labeled_mask = np.ones_like(label, dtype=bool)
    else:
        labeled_mask = label != unlabeled_value
    fg_mask = np.logical_and(labeled_mask, label > 0)
    return labeled_mask, fg_mask


def prepare_feature_maps(image, local_std_kernel):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image_chw = image[None, ...]
    elif image.ndim == 3:
        if image.shape[0] <= 4:
            image_chw = image
        elif image.shape[-1] <= 4:
            image_chw = np.transpose(image, (2, 0, 1))
        else:
            image_chw = image[None, ...]
    else:
        raise ValueError(f"Unsupported audit image shape: {image.shape}")

    norm_channels = []
    for channel in image_chw.astype(np.float32, copy=False):
        norm_channels.append((channel - float(channel.mean())) / max(float(channel.std()), 1e-6))
    image_norm = np.stack(norm_channels, axis=0)
    gray = image_norm.mean(axis=0)

    grad_x = ndimage.sobel(gray, axis=0, mode="reflect")
    grad_y = ndimage.sobel(gray, axis=1, mode="reflect")
    grad_mag = np.sqrt(np.maximum((grad_x ** 2) + (grad_y ** 2), 0.0)).astype(np.float32)

    if local_std_kernel <= 0:
        local_std_kernel = 15
    if local_std_kernel % 2 == 0:
        local_std_kernel += 1
    mean_map = ndimage.uniform_filter(gray, size=local_std_kernel, mode="reflect")
    sq_mean_map = ndimage.uniform_filter(gray * gray, size=local_std_kernel, mode="reflect")
    local_std = np.sqrt(np.maximum(sq_mean_map - (mean_map * mean_map), 0.0)).astype(np.float32)

    feature_stack = np.stack([gray.astype(np.float32), grad_mag, local_std], axis=0)
    for idx in range(feature_stack.shape[0]):
        feat = feature_stack[idx]
        feature_stack[idx] = (feat - float(feat.mean())) / max(float(feat.std()), 1e-6)
    return local_std, feature_stack.astype(np.float32)


def compute_homogeneity(local_std_map, support_mask, args):
    if support_mask.any():
        dist_to_seed = ndimage.distance_transform_edt(~support_mask)
        non_seed_mask = dist_to_seed >= args.dg_far_radius
    else:
        non_seed_mask = np.ones_like(support_mask, dtype=bool)
    if int(non_seed_mask.sum()) < args.dg_min_non_seed_pixels and support_mask.any():
        dist_to_seed = ndimage.distance_transform_edt(~support_mask)
        non_seed_mask = dist_to_seed >= max(8.0, 0.5 * args.dg_far_radius)
    if int(non_seed_mask.sum()) < args.dg_min_non_seed_pixels:
        return 0.5
    mean_local_std = float(local_std_map[non_seed_mask].mean())
    return float(np.exp(-mean_local_std / max(args.dg_hom_tau, 1e-6)))


def compute_separability(feature_stack, support_mask, args):
    if int(support_mask.sum()) < args.dg_min_fg_pixels:
        return 0.5
    dist_to_seed = ndimage.distance_transform_edt(~support_mask)
    near_mask = np.logical_and(dist_to_seed > 0.0, dist_to_seed <= args.dg_sep_near_radius)
    mid_mask = np.logical_and(dist_to_seed > args.dg_sep_near_radius, dist_to_seed <= args.dg_sep_mid_radius)
    far_mask = np.logical_and(dist_to_seed > args.dg_sep_mid_radius, dist_to_seed <= args.dg_sep_far_radius)
    if min(int(near_mask.sum()), int(mid_mask.sum()), int(far_mask.sum())) < args.dg_min_ring_pixels:
        return 0.5

    seed_feat = feature_stack[:, support_mask].mean(axis=1)
    near_feat = feature_stack[:, near_mask].mean(axis=1)
    mid_feat = feature_stack[:, mid_mask].mean(axis=1)
    far_feat = feature_stack[:, far_mask].mean(axis=1)
    d_near = float(np.linalg.norm(near_feat - seed_feat))
    d_mid = float(np.linalg.norm(mid_feat - seed_feat))
    d_far = float(np.linalg.norm(far_feat - seed_feat))
    score_nm = 1.0 / (1.0 + np.exp(-(d_mid - d_near) / max(args.dg_sep_tau, 1e-6)))
    score_mf = 1.0 / (1.0 + np.exp(-(d_far - d_mid) / max(args.dg_sep_tau, 1e-6)))
    return float(0.5 * (score_nm + score_mf))


def perturb_support_mask(support_mask, keep_ratio, rng, args):
    coords = np.argwhere(np.asarray(support_mask, dtype=bool))
    if len(coords) <= args.dg_prop_min_seed_pixels:
        return support_mask.copy()
    keep_count = max(args.dg_prop_min_keep_pixels, int(round(len(coords) * keep_ratio)))
    keep_count = min(keep_count, len(coords))
    keep_coords = coords[rng.permutation(len(coords))[:keep_count]]
    perturbed = np.zeros_like(support_mask, dtype=bool)
    if len(keep_coords) > 0:
        perturbed[tuple(keep_coords.T)] = True
    return perturbed


def binary_iou(mask_a, mask_b):
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    return float(np.logical_and(mask_a, mask_b).sum()) / float(union)


def build_support_region(feature_stack, support_mask, args):
    if int(support_mask.sum()) < args.dg_prop_min_seed_pixels:
        return np.zeros_like(support_mask, dtype=bool)
    support_feat = feature_stack[:, support_mask]
    support_mean = support_feat.mean(axis=1)
    support_dist = np.linalg.norm(support_feat.T - support_mean[None, :], axis=1)
    dist_map = np.linalg.norm(feature_stack - support_mean[:, None, None], axis=0)
    dist_thr = float(np.median(support_dist) + (args.dg_prop_dist_std_scale * np.std(support_dist)))
    return np.asarray(dist_map <= max(dist_thr, 1e-6), dtype=bool)


def compute_propagation(feature_stack, support_mask, case_tag, args):
    if int(support_mask.sum()) < args.dg_min_fg_pixels:
        return 0.5
    base_fg = build_support_region(feature_stack, support_mask, args)
    if int(base_fg.sum()) < args.dg_prop_min_region_pixels:
        return 0.0
    rng = np.random.RandomState(int(sum(ord(ch) for ch in str(case_tag)) % (2 ** 32 - 1)))
    mask_p1 = perturb_support_mask(support_mask, args.dg_prop_seed_keep_ratio1, rng, args)
    mask_p2 = perturb_support_mask(support_mask, args.dg_prop_seed_keep_ratio2, rng, args)
    fg_p1 = build_support_region(feature_stack, mask_p1, args)
    fg_p2 = build_support_region(feature_stack, mask_p2, args)
    return float(np.mean([binary_iou(base_fg, fg_p1), binary_iou(base_fg, fg_p2), binary_iou(fg_p1, fg_p2)]))


def subsample_cases(cases, limit):
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    keep_indices = np.linspace(0, len(cases) - 1, num=limit, dtype=np.int64)
    keep_indices = sorted(set(int(x) for x in keep_indices.tolist()))
    return [cases[idx] for idx in keep_indices]


def iter_cases(dataset, cid):
    for idx, sample in enumerate(dataset.data_list):
        yield {
            "case_tag": f"client{cid}_sample{idx}",
            "image": np.asarray(sample["image"], dtype=np.float32),
            "label": np.asarray(sample["label"]),
        }


def compute_local_audit(dataset, cid, args):
    cases = subsample_cases(list(iter_cases(dataset, cid)), args.dg_max_audit_samples)
    prop_tags = {x["case_tag"] for x in subsample_cases(cases, args.dg_max_prop_samples)}
    hom, sep, prop = [], [], []
    for case in cases:
        support_mask, _ = extract_labeled_and_fg_masks(case["label"], args.img_class)
        if int(support_mask.sum()) <= 0:
            continue
        local_std, feature_stack = prepare_feature_maps(case["image"], args.dg_local_std_kernel)
        hom.append(compute_homogeneity(local_std, support_mask, args))
        sep.append(compute_separability(feature_stack, support_mask, args))
        if case["case_tag"] in prop_tags:
            prop.append(compute_propagation(feature_stack, support_mask, case["case_tag"], args))
    return {
        "homogeneity": float(np.median(np.asarray(hom, dtype=np.float32))) if hom else 0.5,
        "separability": float(np.median(np.asarray(sep, dtype=np.float32))) if sep else 0.5,
        "propagation": float(np.median(np.asarray(prop, dtype=np.float32))) if prop else 0.5,
        "num_cases": len(cases),
        "num_prop_cases": len(prop),
    }


def weak_label_stats(dataset, img_class, dilation_radius=5, density_kernel=11, mask_mode="foreground"):
    if dilation_radius > 0:
        dilation_structure = np.ones(((2 * dilation_radius) + 1, (2 * dilation_radius) + 1), dtype=np.uint8)
    else:
        dilation_structure = None
    if density_kernel % 2 == 0:
        density_kernel += 1
    coverages, dilated_coverages, densities = [], [], []
    for sample in dataset.data_list:
        label = np.asarray(sample["label"])
        unlabeled_value = infer_unlabeled_value_from_label(label, img_class)
        labeled_mask = np.ones_like(label, dtype=bool) if unlabeled_value is None else label != unlabeled_value
        fg_mask = np.logical_and(labeled_mask, label > 0)
        if mask_mode == "foreground":
            support_mask = fg_mask
        elif mask_mode == "labeled":
            support_mask = labeled_mask
        else:
            raise ValueError(mask_mode)
        coverages.append(float(support_mask.mean()))
        if dilation_structure is not None and support_mask.any():
            dilated = ndimage.binary_dilation(support_mask, structure=dilation_structure)
        else:
            dilated = support_mask
        dilated_coverages.append(float(dilated.mean()))
        if support_mask.any():
            density_map = ndimage.uniform_filter(support_mask.astype(np.float32), size=density_kernel, mode="constant")
            densities.append(float(np.median(density_map[support_mask])))
        else:
            densities.append(0.0)
    return (
        float(np.median(np.asarray(coverages, dtype=np.float32))),
        float(np.median(np.asarray(dilated_coverages, dtype=np.float32))),
        float(np.median(np.asarray(densities, dtype=np.float32))),
    )


def compute_e_client(dataset, args):
    fg_cov, dil_cov, fg_den = weak_label_stats(dataset, args.img_class, args.trustgeo_dilation_radius, args.trustgeo_density_kernel, "foreground")
    sup_cov, dil_sup_cov, sup_den = weak_label_stats(dataset, args.img_class, args.trustgeo_dilation_radius, args.trustgeo_density_kernel, "labeled")
    s_c = float(np.clip((fg_cov - args.trustgeo_c_low) / max(args.trustgeo_c_high - args.trustgeo_c_low, 1e-8), 0.0, 1.0))
    s_d = float(np.clip((dil_cov - args.trustgeo_d_low) / max(args.trustgeo_d_high - args.trustgeo_d_low, 1e-8), 0.0, 1.0))
    support_score = float(np.clip((sup_cov - args.trustgeo_support_low) / max(args.trustgeo_support_high - args.trustgeo_support_low, 1e-8), 0.0, 1.0))
    support_mod = float(np.clip(args.trustgeo_support_mod_min + ((1.0 - args.trustgeo_support_mod_min) * support_score), args.trustgeo_support_mod_min, 1.0))
    if fg_cov <= args.trustgeo_c_low:
        base = 0.0
        e_client = 0.0
    else:
        base = float(np.clip((args.trustgeo_w_c * s_c) + (args.trustgeo_w_d * s_d), 0.0, 1.0))
        e_client = base * support_mod
    return {
        "fg_coverage": fg_cov,
        "dilated_fg_coverage": dil_cov,
        "fg_density": fg_den,
        "support_coverage": sup_cov,
        "dilated_support_coverage": dil_sup_cov,
        "support_density": sup_den,
        "s_c": s_c,
        "s_d": s_d,
        "support_score": support_score,
        "support_mod": support_mod,
        "e_client_base": base,
        "e_client": e_client,
    }


def aggregate_dataset(local_audits, args):
    sep = float(np.median(np.asarray([x["separability"] for x in local_audits], dtype=np.float32)))
    hom = float(np.median(np.asarray([x["homogeneity"] for x in local_audits], dtype=np.float32)))
    prop = float(np.median(np.asarray([x["propagation"] for x in local_audits], dtype=np.float32)))
    q_no_prop = float(np.clip((0.7 * sep) + (0.3 * hom), 0.0, 1.0))
    g_old = float(np.clip((args.dg_dataset_w_sep * sep) + (args.dg_dataset_w_hom * hom) + (args.dg_dataset_w_prop * prop), 0.0, 1.0))
    return {
        "separability": sep,
        "homogeneity": hom,
        "propagation": prop,
        "q_no_prop": q_no_prop,
        "g_old": g_old,
        "num_cases": int(sum(x["num_cases"] for x in local_audits)),
        "num_prop_cases": int(sum(x["num_prop_cases"] for x in local_audits)),
        "num_client_audits": len(local_audits),
    }


def audit_dataset(name, cfg, args):
    args.img_class = name
    root = args.root_override or cfg["root"]
    local_audits = []
    client_rows = []
    for cid, (client, sup_type) in enumerate(cfg["clients"]):
        dataset = BaseDataSets(
            base_dir=root,
            split="train",
            transform=None,
            client=client,
            sup_type=sup_type,
            img_class=name,
            geometry_guided=False,
        )
        local = compute_local_audit(dataset, cid, args)
        local_audits.append(local)
        e_stats = compute_e_client(dataset, args)
        client_rows.append({
            "cid": cid,
            "client": client,
            "sup_type": sup_type,
            **local,
            **e_stats,
        })
    ds = aggregate_dataset(local_audits, args)
    for row in client_rows:
        row["lambda_old"] = float(ds["g_old"] * row["e_client"])
        row["lambda_no_prop"] = float(ds["q_no_prop"] * row["e_client"])
    return {"dataset": name, "root": root, "dataset_audit": ds, "clients": client_rows}


def print_summary(results):
    print("\nDATASET SUMMARY")
    print("dataset\tsep\thom\tprop\tq_no_prop\tg_old\tcases\tprop_cases")
    for r in results:
        a = r["dataset_audit"]
        print(
            f"{r['dataset']}\t{a['separability']:.6f}\t{a['homogeneity']:.6f}\t"
            f"{a['propagation']:.6f}\t{a['q_no_prop']:.6f}\t{a['g_old']:.6f}\t"
            f"{a['num_cases']}\t{a['num_prop_cases']}"
        )
    print("\nCLIENT SUMMARY")
    print("dataset\tcid\tclient\tsup_type\te_client\tlambda_old\tlambda_no_prop\tfg_cov\tdil_fg_cov")
    for r in results:
        for c in r["clients"]:
            print(
                f"{r['dataset']}\t{c['cid']}\t{c['client']}\t{c['sup_type']}\t"
                f"{c['e_client']:.6f}\t{c['lambda_old']:.6f}\t{c['lambda_no_prop']:.6f}\t"
                f"{c['fg_coverage']:.6f}\t{c['dilated_fg_coverage']:.6f}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="odoc,faz,prostate,polyp")
    parser.add_argument("--root_override", default="")
    parser.add_argument("--out", default="logs/dg_audit_table.json")
    parser.add_argument("--dg_max_audit_samples", type=int, default=64)
    parser.add_argument("--dg_max_prop_samples", type=int, default=16)
    parser.add_argument("--dg_local_std_kernel", type=int, default=15)
    parser.add_argument("--dg_far_radius", type=float, default=24.0)
    parser.add_argument("--dg_sep_near_radius", type=float, default=8.0)
    parser.add_argument("--dg_sep_mid_radius", type=float, default=24.0)
    parser.add_argument("--dg_sep_far_radius", type=float, default=48.0)
    parser.add_argument("--dg_min_fg_pixels", type=int, default=8)
    parser.add_argument("--dg_min_ring_pixels", type=int, default=64)
    parser.add_argument("--dg_min_non_seed_pixels", type=int, default=128)
    parser.add_argument("--dg_hom_tau", type=float, default=0.75)
    parser.add_argument("--dg_sep_tau", type=float, default=0.25)
    parser.add_argument("--dg_prop_seed_keep_ratio1", type=float, default=0.85)
    parser.add_argument("--dg_prop_seed_keep_ratio2", type=float, default=0.70)
    parser.add_argument("--dg_prop_min_seed_pixels", type=int, default=4)
    parser.add_argument("--dg_prop_min_keep_pixels", type=int, default=2)
    parser.add_argument("--dg_prop_min_region_pixels", type=int, default=16)
    parser.add_argument("--dg_prop_dist_std_scale", type=float, default=1.0)
    parser.add_argument("--dg_dataset_w_sep", type=float, default=0.60)
    parser.add_argument("--dg_dataset_w_hom", type=float, default=0.30)
    parser.add_argument("--dg_dataset_w_prop", type=float, default=0.10)
    parser.add_argument("--trustgeo_dilation_radius", type=int, default=5)
    parser.add_argument("--trustgeo_density_kernel", type=int, default=11)
    parser.add_argument("--trustgeo_c_low", type=float, default=0.0015)
    parser.add_argument("--trustgeo_c_high", type=float, default=0.0400)
    parser.add_argument("--trustgeo_d_low", type=float, default=0.0060)
    parser.add_argument("--trustgeo_d_high", type=float, default=0.0900)
    parser.add_argument("--trustgeo_support_low", type=float, default=0.0100)
    parser.add_argument("--trustgeo_support_high", type=float, default=0.1500)
    parser.add_argument("--trustgeo_support_mod_min", type=float, default=0.85)
    parser.add_argument("--trustgeo_w_c", type=float, default=0.70)
    parser.add_argument("--trustgeo_w_d", type=float, default=0.30)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    for name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        if name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {name}")
        print(f"\nAuditing {name}...")
        results.append(audit_dataset(name, DATASETS[name], args))
    print_summary(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
