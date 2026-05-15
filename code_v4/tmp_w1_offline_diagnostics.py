import argparse
import glob
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from dataloaders.dataset import BaseDataSets, compute_annotation_geometry_bin
from networks.net_factory import net_factory


IGNORE = 3
SITE_CONFIGS = [
    {"site": "SiteA", "client": "client1", "sup_type": "scribble", "cid": 0},
    {"site": "SiteB", "client": "client2", "sup_type": "scribble_noisy", "cid": 1},
    {"site": "SiteC", "client": "client3", "sup_type": "scribble_noisy", "cid": 2},
    {"site": "SiteD", "client": "client4", "sup_type": "keypoint", "cid": 3},
    {"site": "SiteE", "client": "client5", "sup_type": "block", "cid": 4},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline W1' diagnostics: tau sweep, reliability composition, prior mismatch."
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="/data/jianbingshen/yanghongji/FedLPPA/data/ODOC_h5",
    )
    parser.add_argument(
        "--local_model_dir",
        type=str,
        required=True,
        help="Directory containing client_{cid}_iter_{iter}_dice_*.pth checkpoints.",
    )
    parser.add_argument(
        "--global_checkpoint",
        type=str,
        default="",
        help="Optional shared global checkpoint path for strict-global diagnostics.",
    )
    parser.add_argument(
        "--global_checkpoint_paths",
        nargs="+",
        default=None,
        help="Optional per-site global checkpoint paths, one per client.",
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
    parser.add_argument(
        "--score_mode",
        type=str,
        default="local_aux_proxy",
        choices=["strict_global", "local_aux_proxy"],
        help=(
            "strict_global uses a provided global checkpoint; local_aux_proxy uses "
            "main-vs-aux disagreement as a proxy when no global checkpoint is available."
        ),
    )
    parser.add_argument(
        "--pseudo_alpha",
        type=float,
        default=0.5,
        help="Mixing weight used to build pseudo_label_mix offline.",
    )
    parser.add_argument(
        "--gamma_prob",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--gamma_conf",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--blend_kappa",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--global_min_conf",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--tau_values",
        type=str,
        default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90",
    )
    parser.add_argument(
        "--tau_source",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive_latest"],
        help="Tau source for reliable/unreliable composition analysis.",
    )
    parser.add_argument(
        "--tau_init",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--geometry_near_radius",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--geometry_mid_radius",
        type=float,
        default=24.0,
    )
    return parser.parse_args()


def parse_tau_values(tau_str):
    return [float(x) for x in str(tau_str).split(",") if x.strip()]


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


def find_local_checkpoint(model_dir, cid, iter_tag):
    pattern = os.path.join(model_dir, f"client_{cid}_iter_{iter_tag}_dice_*.pth")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no checkpoint matches {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"multiple checkpoints match {pattern}: {matches}")
    return matches[0]


def resolve_global_checkpoint(args, site_index):
    if args.global_checkpoint_paths is not None:
        if len(args.global_checkpoint_paths) != len(SITE_CONFIGS):
            raise ValueError("global_checkpoint_paths must match number of sites")
        return args.global_checkpoint_paths[site_index]
    if args.global_checkpoint:
        return args.global_checkpoint
    return ""


def load_tau_bins(args, cid):
    if args.tau_source == "fixed":
        return [float(args.tau_init)] * 4
    state_path = os.path.join(args.local_model_dir, f"client_{cid}_adaptive_pl_latest.pth")
    if not os.path.exists(state_path):
        return [float(args.tau_init)] * 4
    state = torch.load(state_path, map_location="cpu")
    tau = state.get("adaptive_pl_tau")
    if tau is None or len(tau) != 4:
        return [float(args.tau_init)] * 4
    return [float(x) for x in tau]


def extract_outputs(model, image):
    out = model(image)
    outputs = out[0]
    outputs_auxiliary = out[8]
    return outputs, outputs_auxiliary


def compute_adaptive_pseudo_reliability(local_prob, global_prob, gamma_prob, gamma_conf):
    conf_local, pred_local = torch.max(local_prob, dim=1)
    conf_global, pred_global = torch.max(global_prob, dim=1)
    prob_gap = 0.5 * torch.abs(local_prob - global_prob).sum(dim=1)
    conf_gap = torch.abs(conf_local - conf_global)
    agree_lg = pred_local == pred_global
    score = torch.exp(-float(gamma_prob) * prob_gap) * torch.exp(-float(gamma_conf) * conf_gap)
    score = score * (0.5 + 0.5 * agree_lg.float())
    return {
        "conf_local": conf_local.clamp(0.0, 1.0),
        "conf_global": conf_global.clamp(0.0, 1.0),
        "prob_gap": prob_gap.clamp(0.0, 1.0),
        "conf_gap": conf_gap.clamp(0.0, 1.0),
        "agree_lg": agree_lg,
        "score": score.clamp(0.0, 1.0),
    }


def prevalence_for_mask(mask, values, class_idx):
    denom = int(mask.sum().item())
    if denom <= 0:
        return None
    return float(((values == class_idx) & mask).sum().item() / denom)


def safe_div(num, den):
    return None if den <= 0 else float(num / den)


def summarize_partition(mask, gt, pred, score, conf_ref):
    count = int(mask.sum().item())
    if count <= 0:
        return {
            "count": 0,
            "overall_accuracy": None,
            "score_mean": None,
            "conf_ref_mean": None,
            "gt_bg_ratio": None,
            "gt_oc_ratio": None,
            "gt_od_ratio": None,
            "pred_bg_ratio": None,
            "pred_oc_ratio": None,
            "pred_od_ratio": None,
            "oc_precision": None,
            "oc_recall_within_partition": None,
            "od_precision": None,
            "od_recall_within_partition": None,
        }

    gt_sel = gt[mask]
    pred_sel = pred[mask]
    oc_tp = int(((pred_sel == 1) & (gt_sel == 1)).sum().item())
    od_tp = int(((pred_sel == 2) & (gt_sel == 2)).sum().item())
    oc_pred = int((pred_sel == 1).sum().item())
    od_pred = int((pred_sel == 2).sum().item())
    oc_gt = int((gt_sel == 1).sum().item())
    od_gt = int((gt_sel == 2).sum().item())
    return {
        "count": count,
        "overall_accuracy": float((pred_sel == gt_sel).float().mean().item()),
        "score_mean": float(score[mask].mean().item()),
        "conf_ref_mean": float(conf_ref[mask].mean().item()),
        "gt_bg_ratio": float((gt_sel == 0).float().mean().item()),
        "gt_oc_ratio": float((gt_sel == 1).float().mean().item()),
        "gt_od_ratio": float((gt_sel == 2).float().mean().item()),
        "pred_bg_ratio": float((pred_sel == 0).float().mean().item()),
        "pred_oc_ratio": float((pred_sel == 1).float().mean().item()),
        "pred_od_ratio": float((pred_sel == 2).float().mean().item()),
        "oc_precision": safe_div(oc_tp, oc_pred),
        "oc_recall_within_partition": safe_div(oc_tp, oc_gt),
        "od_precision": safe_div(od_tp, od_pred),
        "od_recall_within_partition": safe_div(od_tp, od_gt),
    }


def summarize_tau_sweep(score, conf_ref, pred, gt, unlabeled_mask, tau_values, global_min_conf):
    gt_oc_total = int(((gt == 1) & unlabeled_mask).sum().item())
    gt_od_total = int(((gt == 2) & unlabeled_mask).sum().item())
    rows = []
    for tau in tau_values:
        accepted = unlabeled_mask & (score >= tau)
        soft_only = unlabeled_mask & (score < tau) & (conf_ref >= float(global_min_conf))
        both_uncertain = unlabeled_mask & (score < tau) & (conf_ref < float(global_min_conf))
        accept_count = int(accepted.sum().item())
        if accept_count > 0:
            gt_sel = gt[accepted]
            pred_sel = pred[accepted]
            overall_precision = float((gt_sel == pred_sel).float().mean().item())
            oc_tp = int(((pred_sel == 1) & (gt_sel == 1)).sum().item())
            od_tp = int(((pred_sel == 2) & (gt_sel == 2)).sum().item())
            oc_pred = int((pred_sel == 1).sum().item())
            od_pred = int((pred_sel == 2).sum().item())
        else:
            overall_precision = None
            oc_tp = od_tp = oc_pred = od_pred = 0
        rows.append(
            {
                "tau": float(tau),
                "accepted_ratio": float(accept_count / max(1, int(unlabeled_mask.sum().item()))),
                "soft_ratio": float(soft_only.sum().item() / max(1, int(unlabeled_mask.sum().item()))),
                "both_uncertain_ratio": float(both_uncertain.sum().item() / max(1, int(unlabeled_mask.sum().item()))),
                "hard_overall_precision": overall_precision,
                "hard_oc_precision": safe_div(oc_tp, oc_pred),
                "hard_oc_recall": safe_div(oc_tp, gt_oc_total),
                "hard_od_precision": safe_div(od_tp, od_pred),
                "hard_od_recall": safe_div(od_tp, gt_od_total),
            }
        )
    return rows


def build_prior_summary(unlabeled_mask, gt, local_main_pred, local_aux_pred, mixed_pred, corrected_pred, global_pred=None):
    sources = {
        "gt": gt,
        "local_main": local_main_pred,
        "local_aux": local_aux_pred,
        "local_mix": mixed_pred,
        "corrected": corrected_pred,
    }
    if global_pred is not None:
        sources["global_mix"] = global_pred

    rows = []
    for source_name, values in sources.items():
        rows.append(
            {
                "source": source_name,
                "bg_ratio": prevalence_for_mask(unlabeled_mask, values, 0),
                "oc_ratio": prevalence_for_mask(unlabeled_mask, values, 1),
                "od_ratio": prevalence_for_mask(unlabeled_mask, values, 2),
            }
        )
    return rows


def analyze_site(args, config, site_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_ckpt = find_local_checkpoint(args.local_model_dir, config["cid"], args.iter_tag)
    local_args = build_args(config["cid"], config["sup_type"])
    local_model = net_factory(local_args, net_type="unet_univ5", in_chns=3, class_num=3)
    local_model.load_state_dict(load_state(local_ckpt), strict=True)
    local_model.to(device)
    local_model.eval()

    global_model = None
    global_ckpt = resolve_global_checkpoint(args, site_index)
    if args.score_mode == "strict_global":
        if not global_ckpt:
            raise ValueError("strict_global mode requires --global_checkpoint or --global_checkpoint_paths")
        global_model = net_factory(local_args, net_type="unet_univ5", in_chns=3, class_num=3)
        global_model.load_state_dict(load_state(global_ckpt), strict=True)
        global_model.to(device)
        global_model.eval()

    dataset = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        client=config["client"],
        sup_type=config["sup_type"],
        img_class="odoc",
    )

    tau_values = parse_tau_values(args.tau_values)
    tau_bins = load_tau_bins(args, config["cid"])

    score_all = []
    conf_ref_all = []
    pred_all = []
    gt_all = []
    unlabeled_all = []
    geometry_all = []
    local_main_all = []
    local_aux_all = []
    mixed_all = []
    corrected_all = []
    global_mix_all = []
    prob_gap_all = []
    conf_gap_all = []
    agree_all = []

    pseudo_alpha = float(args.pseudo_alpha)
    with torch.no_grad():
        for case, sample in zip(dataset.sample_list, dataset.data_list):
            image = torch.from_numpy(sample["image"]).unsqueeze(0).to(device).float()
            weak = torch.from_numpy(sample["label"]).unsqueeze(0).to(device).long()
            with h5py.File(os.path.join(args.root_path, case), "r") as f:
                gt = torch.from_numpy(f["mask"][:]).unsqueeze(0).to(device).long()

            geometry_np = compute_annotation_geometry_bin(
                sample["label"],
                "odoc",
                near_radius=args.geometry_near_radius,
                mid_radius=args.geometry_mid_radius,
            )
            geometry_bin = torch.from_numpy(geometry_np).unsqueeze(0).to(device).long()

            outputs, outputs_auxiliary = extract_outputs(local_model, image)
            local_main_prob = torch.softmax(outputs, dim=1)
            local_aux_prob = torch.softmax(outputs_auxiliary, dim=1)
            local_mix = pseudo_alpha * local_main_prob + (1.0 - pseudo_alpha) * local_aux_prob

            if args.score_mode == "strict_global":
                global_outputs, global_outputs_auxiliary = extract_outputs(global_model, image)
                global_main_prob = torch.softmax(global_outputs, dim=1)
                global_aux_prob = torch.softmax(global_outputs_auxiliary, dim=1)
                global_mix = pseudo_alpha * global_main_prob + (1.0 - pseudo_alpha) * global_aux_prob
            else:
                global_mix = local_aux_prob

            reliability = compute_adaptive_pseudo_reliability(
                local_mix,
                global_mix,
                gamma_prob=args.gamma_prob,
                gamma_conf=args.gamma_conf,
            )
            lambda_dynamic = torch.exp(-float(args.blend_kappa) * reliability["conf_gap"]).clamp(1e-6, 1.0)
            corrected_prob = (
                lambda_dynamic.unsqueeze(1) * local_mix
                + (1.0 - lambda_dynamic).unsqueeze(1) * global_mix
            )
            corrected_pred = torch.argmax(corrected_prob, dim=1)
            unlabeled = weak == IGNORE

            score_all.append(reliability["score"].cpu())
            conf_ref_all.append(reliability["conf_global"].cpu())
            pred_all.append(corrected_pred.cpu())
            gt_all.append(gt.cpu())
            unlabeled_all.append(unlabeled.cpu())
            geometry_all.append(geometry_bin.cpu())
            local_main_all.append(torch.argmax(local_main_prob, dim=1).cpu())
            local_aux_all.append(torch.argmax(local_aux_prob, dim=1).cpu())
            mixed_all.append(torch.argmax(local_mix, dim=1).cpu())
            corrected_all.append(corrected_pred.cpu())
            prob_gap_all.append(reliability["prob_gap"].cpu())
            conf_gap_all.append(reliability["conf_gap"].cpu())
            agree_all.append(reliability["agree_lg"].cpu())
            if args.score_mode == "strict_global":
                global_mix_all.append(torch.argmax(global_mix, dim=1).cpu())

    score = torch.cat(score_all, dim=0)
    conf_ref = torch.cat(conf_ref_all, dim=0)
    pred = torch.cat(pred_all, dim=0)
    gt = torch.cat(gt_all, dim=0)
    unlabeled_mask = torch.cat(unlabeled_all, dim=0)
    geometry_bin = torch.cat(geometry_all, dim=0)
    local_main_pred = torch.cat(local_main_all, dim=0)
    local_aux_pred = torch.cat(local_aux_all, dim=0)
    mixed_pred = torch.cat(mixed_all, dim=0)
    corrected_pred = torch.cat(corrected_all, dim=0)
    prob_gap = torch.cat(prob_gap_all, dim=0)
    conf_gap = torch.cat(conf_gap_all, dim=0)
    agree_lg = torch.cat(agree_all, dim=0)
    global_mix_pred = torch.cat(global_mix_all, dim=0) if global_mix_all else None

    tau_map = torch.tensor(tau_bins, dtype=score.dtype)[geometry_bin.long().clamp(0, 3)]
    reliable_mask = unlabeled_mask & (score >= tau_map)
    soft_mask = unlabeled_mask & (score < tau_map) & (conf_ref >= float(args.global_min_conf))
    both_uncertain_mask = unlabeled_mask & (score < tau_map) & (conf_ref < float(args.global_min_conf))

    site_result = {
        "site": config["site"],
        "client": config["client"],
        "cid": config["cid"],
        "sup_type": config["sup_type"],
        "score_mode": args.score_mode,
        "local_checkpoint": local_ckpt,
        "global_checkpoint": global_ckpt if global_ckpt else None,
        "tau_bins_for_composition": tau_bins,
        "unlabeled_pixel_count": int(unlabeled_mask.sum().item()),
        "score_summary": {
            "mean_score": float(score[unlabeled_mask].mean().item()) if unlabeled_mask.any() else None,
            "mean_conf_ref": float(conf_ref[unlabeled_mask].mean().item()) if unlabeled_mask.any() else None,
            "mean_prob_gap": float(prob_gap[unlabeled_mask].mean().item()) if unlabeled_mask.any() else None,
            "mean_conf_gap": float(conf_gap[unlabeled_mask].mean().item()) if unlabeled_mask.any() else None,
            "lg_agree_ratio": float(agree_lg[unlabeled_mask].float().mean().item()) if unlabeled_mask.any() else None,
        },
        "tau_sweep": summarize_tau_sweep(
            score,
            conf_ref,
            pred,
            gt,
            unlabeled_mask,
            tau_values,
            args.global_min_conf,
        ),
        "reliability_composition": {
            "reliable": summarize_partition(reliable_mask, gt, pred, score, conf_ref),
            "unreliable_soft": summarize_partition(soft_mask, gt, pred, score, conf_ref),
            "both_uncertain": summarize_partition(both_uncertain_mask, gt, pred, score, conf_ref),
        },
        "prior_mismatch": build_prior_summary(
            unlabeled_mask,
            gt,
            local_main_pred,
            local_aux_pred,
            mixed_pred,
            corrected_pred,
            global_pred=global_mix_pred,
        ),
        "geometry_bin_summary": [],
    }

    for bin_idx in range(4):
        bin_mask = unlabeled_mask & (geometry_bin == bin_idx)
        site_result["geometry_bin_summary"].append(
            {
                "bin": bin_idx,
                "count": int(bin_mask.sum().item()),
                "score_mean": float(score[bin_mask].mean().item()) if bin_mask.any() else None,
                "conf_ref_mean": float(conf_ref[bin_mask].mean().item()) if bin_mask.any() else None,
                "oc_gt_ratio": prevalence_for_mask(bin_mask, gt, 1),
                "od_gt_ratio": prevalence_for_mask(bin_mask, gt, 2),
                "corrected_oc_ratio": prevalence_for_mask(bin_mask, corrected_pred, 1),
                "corrected_od_ratio": prevalence_for_mask(bin_mask, corrected_pred, 2),
            }
        )

    return site_result


def build_summary(sites):
    rows = []
    for site in sites:
        tau_055 = next((row for row in site["tau_sweep"] if abs(row["tau"] - 0.55) < 1e-8), None)
        rows.append(
            {
                "site": site["site"],
                "score_mode": site["score_mode"],
                "mean_score": site["score_summary"]["mean_score"],
                "lg_agree_ratio": site["score_summary"]["lg_agree_ratio"],
                "reliable_count": site["reliability_composition"]["reliable"]["count"],
                "unreliable_soft_count": site["reliability_composition"]["unreliable_soft"]["count"],
                "both_uncertain_count": site["reliability_composition"]["both_uncertain"]["count"],
                "tau055_hard_precision": None if tau_055 is None else tau_055["hard_overall_precision"],
                "tau055_oc_precision": None if tau_055 is None else tau_055["hard_oc_precision"],
                "tau055_oc_recall": None if tau_055 is None else tau_055["hard_oc_recall"],
                "tau055_od_precision": None if tau_055 is None else tau_055["hard_od_precision"],
                "tau055_od_recall": None if tau_055 is None else tau_055["hard_od_recall"],
            }
        )
    return rows


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    results = [analyze_site(args, config, idx) for idx, config in enumerate(SITE_CONFIGS)]
    payload = {
        "iter_tag": args.iter_tag,
        "score_mode": args.score_mode,
        "local_model_dir": args.local_model_dir,
        "global_checkpoint": args.global_checkpoint if args.global_checkpoint else None,
        "tau_source": args.tau_source,
        "pseudo_alpha": args.pseudo_alpha,
        "gamma_prob": args.gamma_prob,
        "gamma_conf": args.gamma_conf,
        "blend_kappa": args.blend_kappa,
        "global_min_conf": args.global_min_conf,
        "sites": results,
        "summary": build_summary(results),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
