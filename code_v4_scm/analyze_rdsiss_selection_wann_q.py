# -*- coding: utf-8 -*-
"""Detailed offline prior diagnostics for the current RDSI-SS logic.

This script does not train. It reuses the production WANN/RDSI selection code
and measures, with full GT offline, whether the selected teacher regions and
residual target move in the correct direction.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_rdsi_oracle_diagnostics import (
    POLYP_CLIENTS,
    PROSTATE_CLIENTS,
    auxiliary_logits,
    build_parser as build_oracle_parser,
    default_setup,
    get_client_sup_type,
    load_full_gt,
    load_model,
    parse_client_sup_types,
    primary_logits,
    to_image_tensor,
)
from dataloaders.dataset import BaseDataSets
from rgftd_reliability_distillation import select_rdsi_teacher_logits
from weak_annotation_reliability import build_wann_maps


def rdsi_feature(model_out):
    if not isinstance(model_out, (tuple, list)) or len(model_out) < 3:
        raise ValueError("RDSI-TED diagnostics require decoder feature de1 at model output index 2")
    feature = model_out[2]
    if not torch.is_tensor(feature) or feature.dim() != 4:
        raise ValueError("RDSI-TED diagnostics require a 4D decoder feature map")
    return feature


def _add_arg(parser, *args, **kwargs):
    existing = {action.dest for action in parser._actions}
    dest = kwargs.get("dest")
    if dest is None:
        for item in args:
            if item.startswith("--"):
                dest = item[2:].replace("-", "_")
                break
    if dest in existing:
        return
    parser.add_argument(*args, **kwargs)


def build_parser():
    parser = build_oracle_parser()
    _add_arg(parser, "--rdsi_teacher_sup_types", default="")
    _add_arg(parser, "--rdsi_residual_alpha", type=float, default=0.35)
    _add_arg(parser, "--rdsi_hard_core_conf_thresh", type=float, default=0.90)
    _add_arg(parser, "--rdsi_hard_core_entropy_thresh", type=float, default=0.25)
    _add_arg(parser, "--rdsi_hard_core_reliability_thresh", type=float, default=0.65)
    _add_arg(parser, "--rdsi_benefit_topk_ratio", type=float, default=0.002)
    _add_arg(parser, "--rdsi_benefit_topk_min_pixels", type=int, default=8)
    _add_arg(parser, "--rdsi_benefit_topk_max_pixels", type=int, default=4096)
    _add_arg(parser, "--rdsi_benefit_score_floor", type=float, default=1e-6)

    _add_arg(parser, "--rgftd_teacher_foreground_radius", type=int, default=2)
    _add_arg(parser, "--rgftd_min_foreground_pixels", type=int, default=8)
    _add_arg(parser, "--rgftd_min_foreground_ratio", type=float, default=0.05)
    _add_arg(parser, "--rgftd_teacher_fg_topk_ratio", type=float, default=0.002)
    _add_arg(parser, "--rgftd_teacher_fg_topk_min_pixels", type=int, default=8)
    _add_arg(parser, "--rgftd_teacher_student_fg_margin", type=float, default=0.05)
    _add_arg(parser, "--rgftd_teacher_bg_conf_thresh", type=float, default=0.98)
    _add_arg(parser, "--rgftd_student_conf_thresh", type=float, default=0.80)
    _add_arg(parser, "--rgftd_student_entropy_thresh", type=float, default=0.35)
    _add_arg(parser, "--rgftd_low_r_thresh", type=float, default=0.25)
    _add_arg(parser, "--rgftd_temperature", type=float, default=1.0)
    _add_arg(parser, "--rgftd_use_soft_band", type=int, default=0)
    _add_arg(parser, "--rgftd_background_weight", type=float, default=0.25)
    _add_arg(parser, "--rgftd_skip_background_only", type=int, default=1)
    _add_arg(parser, "--rgftd_teacher_validation_enabled", type=int, default=1)
    _add_arg(parser, "--rgftd_teacher_reliability_min", type=float, default=0.55)
    _add_arg(parser, "--rgftd_teacher_core_agree_floor", type=float, default=0.80)
    _add_arg(parser, "--rgftd_teacher_support_agree_floor", type=float, default=0.70)
    _add_arg(parser, "--rgftd_teacher_support_prob_floor", type=float, default=0.35)
    _add_arg(parser, "--rgftd_teacher_conf_floor", type=float, default=0.85)
    _add_arg(parser, "--rgftd_teacher_class_reliability_min", type=float, default=0.50)
    _add_arg(parser, "--rgftd_teacher_max_core_conflict", type=float, default=0.20)
    _add_arg(parser, "--rgftd_teacher_score_core_weight", type=float, default=0.35)
    _add_arg(parser, "--rgftd_teacher_score_support_weight", type=float, default=0.30)
    _add_arg(parser, "--rgftd_teacher_score_class_weight", type=float, default=0.25)
    _add_arg(parser, "--rgftd_teacher_score_conf_weight", type=float, default=0.10)
    _add_arg(parser, "--rgftd_teacher_release_prob_floor", type=float, default=0.35)
    _add_arg(parser, "--rgftd_teacher_release_margin_floor", type=float, default=0.05)
    _add_arg(parser, "--rgftd_teacher_release_class_floor", type=float, default=0.50)
    _add_arg(parser, "--rgftd_teacher_release_min", type=float, default=0.03)
    _add_arg(parser, "--rgftd_active_fg_topk_ratio", type=float, default=0.002)
    _add_arg(parser, "--rgftd_active_fg_topk_min_pixels", type=int, default=8)
    _add_arg(parser, "--rgftd_active_fg_topk_max_pixels", type=int, default=4096)
    _add_arg(parser, "--rgftd_max_bg_fg_ratio", type=float, default=1.0)
    _add_arg(parser, "--rgftd_allow_bg_without_fg", type=int, default=0)
    _add_arg(parser, "--rgftd_spatial_support_enabled", type=int, default=1)
    _add_arg(parser, "--rgftd_spatial_support_radius", type=int, default=2)
    _add_arg(parser, "--rgftd_spatial_candidate_weight", type=float, default=1.0)
    _add_arg(parser, "--rgftd_spatial_near_seed_weight", type=float, default=0.75)
    _add_arg(parser, "--rgftd_spatial_far_weight", type=float, default=0.15)

    _add_arg(parser, "--top_fraction", type=float, default=0.10)
    return parser


def normalize_entropy(prob):
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    return (entropy / max(math.log(float(prob.shape[1])), 1e-6)).clamp(0.0, 1.0)


def masked_mean(value, mask):
    denom = mask.float().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return float((value * mask.float()).sum().detach().cpu().item() / denom.detach().cpu().item())


def masked_count(mask):
    return int(mask.float().sum().detach().cpu().item())


def top_fraction_mask(score, valid, frac):
    valid_count = masked_count(valid)
    if valid_count <= 0:
        return torch.zeros_like(valid, dtype=torch.bool)
    k = int(max(1, math.ceil(valid_count * max(0.0, min(float(frac), 1.0)))))
    flat_score = score.flatten(1)
    flat_valid = valid.flatten(1)
    out = torch.zeros_like(flat_valid, dtype=torch.bool)
    for b in range(flat_score.shape[0]):
        count = int(flat_valid[b].sum().detach().cpu().item())
        if count <= 0:
            continue
        cur_k = min(k, count)
        s = flat_score[b].masked_fill(~flat_valid[b], -1e9)
        idx = torch.topk(s, k=cur_k, largest=True).indices
        out[b, idx] = True
    return out.view_as(valid)


def selected_teacher_id_map(selected_logits, teacher_logits_list):
    stack = torch.stack([x.detach() for x in teacher_logits_list], dim=0)
    diff = (stack - selected_logits.detach().unsqueeze(0)).abs().mean(dim=2)
    return diff.argmin(dim=0)


def region_accuracy_gain(student_pred, teacher_pred, gt, mask):
    denom = mask.float().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    s_ok = (student_pred == gt) & mask
    t_ok = (teacher_pred == gt) & mask
    return float((t_ok.float().sum() - s_ok.float().sum()).detach().cpu().item() / denom.detach().cpu().item())


def region_dice_gain(student_pred, teacher_pred, gt, mask):
    def dice(pred):
        pred_fg = (pred > 0) & mask
        gt_fg = (gt > 0) & mask
        denom = pred_fg.float().sum() + gt_fg.float().sum()
        if float(denom.detach().cpu().item()) <= 0.0:
            return 1.0
        return float((2.0 * (pred_fg & gt_fg).float().sum()).detach().cpu().item() / denom.detach().cpu().item())
    return dice(teacher_pred) - dice(student_pred)


def summarize_records(records, keys):
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    rows = []
    for name, g in df.groupby(keys, dropna=False):
        if not isinstance(name, tuple):
            name = (name,)
        row = dict(zip(keys, name))
        for col in df.columns:
            if col in keys:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                row[col + "_mean"] = float(g[col].mean())
                row[col + "_sum"] = float(g[col].sum())
        row["rows"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = build_parser().parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    clients, num_classes, in_chns, min_num_clients = default_setup(args.img_class)
    args.num_classes = num_classes
    args.in_chns = in_chns
    args.min_num_clients = min_num_clients
    args.client_sup_type_list = parse_client_sup_types(args, len(clients))
    if not str(args.rdsi_teacher_sup_types or "").strip():
        args.rdsi_teacher_sup_types = ",".join(args.client_sup_type_list)

    out_dir = args.output_dir or os.path.join(args.snapshot_path, "rdsi_selection_wann_q_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_paths = []
    for cid in range(len(clients)):
        name = args.checkpoint_pattern.format(cid=cid, model=args.model)
        path = os.path.join(args.snapshot_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        checkpoint_paths.append(path)
    models = [load_model(args, cid, path) for cid, path in enumerate(checkpoint_paths)]

    actual_records = []
    component_records = []
    q_records = []
    profile_records = []

    with torch.no_grad():
        for target_cid, client in enumerate(clients):
            dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=client,
                sup_type=get_client_sup_type(args, target_cid),
                img_class=args.img_class,
            )
            if args.max_cases_per_client > 0:
                dataset = Subset(dataset, list(range(min(args.max_cases_per_client, len(dataset)))))
                sample_list = dataset.dataset.sample_list
            else:
                sample_list = dataset.sample_list
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            seen = 0
            for batch in loader:
                images = to_image_tensor(batch["image"], args.img_class).to(args.device)
                weak_label = batch["label"].long().to(args.device)
                idxs = batch["idx"].tolist()
                rel_paths = [sample_list[int(idx)] for idx in idxs]
                gt = torch.stack([load_full_gt(args.root_path, rel_path) for rel_path in rel_paths], dim=0).long().to(args.device)

                student_out = models[target_cid](images)
                student_logits = primary_logits(student_out)
                student_feature = rdsi_feature(student_out)
                student_aux = auxiliary_logits(student_out)
                student_prob = torch.softmax(student_logits, dim=1)
                student_pred = student_prob.argmax(dim=1)
                student_entropy = normalize_entropy(student_prob)
                maps = build_wann_maps(
                    image=images,
                    label=weak_label,
                    logits=student_logits,
                    aux_logits=student_aux,
                    sup_type=get_client_sup_type(args, target_cid),
                    img_class=args.img_class,
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )
                teacher_out_list = [model(images) for model in models]
                teacher_logits_list = [primary_logits(model_out) for model_out in teacher_out_list]
                teacher_feature_list = [rdsi_feature(model_out) for model_out in teacher_out_list]
                teacher_pred_list = [torch.softmax(x, dim=1).argmax(dim=1) for x in teacher_logits_list]
                selected_logits, profile = select_rdsi_teacher_logits(
                    student_logits,
                    teacher_logits_list,
                    list(range(len(teacher_logits_list))),
                    weak_label,
                    maps,
                    args,
                    args.analysis_iter,
                    student_feature=student_feature,
                    teacher_feature_list=teacher_feature_list,
                )
                selected_prob = torch.softmax(selected_logits, dim=1)
                alpha = getattr(args, "_rdsi_residual_alpha_map").detach()
                active = getattr(args, "_rdsi_active_mask").detach()
                fg_active = getattr(args, "_rdsi_foreground_active_mask").detach()
                bg_active = getattr(args, "_rdsi_background_active_mask").detach()
                rdsi_region = getattr(args, "_rdsi_risk_region").detach()
                selected_score = getattr(args, "_rdsi_selected_score").detach()
                selected_benefit = getattr(args, "_rdsi_benefit_score").detach()
                selected_lift = getattr(args, "_rdsi_foreground_lift_score").detach()
                selected_tid = selected_teacher_id_map(selected_logits, teacher_logits_list)
                selected_pred = selected_prob.argmax(dim=1)
                q_prob = (student_prob + alpha.unsqueeze(1) * (selected_prob - student_prob)).clamp_min(1e-6)
                q_prob = q_prob / q_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

                for key, value in profile.items():
                    if torch.is_tensor(value):
                        value = float(value.detach().cpu().item())
                    profile_records.append({
                        "dataset": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "profile_key": key,
                        "profile_value": float(value),
                    })

                if args.num_classes > 1:
                    student_fg_prob = student_prob[:, 1:].sum(dim=1)
                    selected_fg_prob = selected_prob[:, 1:].sum(dim=1)
                    q_fg_prob = q_prob[:, 1:].sum(dim=1)
                else:
                    student_fg_prob = torch.zeros_like(student_entropy)
                    selected_fg_prob = torch.zeros_like(student_entropy)
                    q_fg_prob = torch.zeros_like(student_entropy)

                gt_error = student_pred != gt
                global_error_rate = masked_mean(gt_error.float(), torch.ones_like(gt_error, dtype=torch.bool))
                entropy_top = top_fraction_mask(student_entropy, torch.ones_like(gt_error, dtype=torch.bool), args.top_fraction)
                low_rel_top = top_fraction_mask((1.0 - maps.reliability).clamp(0, 1), torch.ones_like(gt_error, dtype=torch.bool), args.top_fraction)
                fg_prob_low_top = top_fraction_mask((1.0 - student_fg_prob).clamp(0, 1), torch.ones_like(gt_error, dtype=torch.bool), args.top_fraction)
                component_map = {
                    "core_mask": maps.core_mask,
                    "soft_band": maps.soft_band,
                    "ignore_mask": maps.ignore_mask,
                    "candidate_mask": maps.candidate_mask,
                    "low_confident": maps.low_confident_mask,
                    "low_agree": maps.low_agree_mask,
                    "low_conflict": maps.low_conflict_mask,
                    "wann_risk_mask": maps.risk_mask,
                    "rdsi_region": rdsi_region,
                    "rdsi_active": active,
                    "rdsi_fg_active": fg_active,
                    "rdsi_bg_active": bg_active,
                    "entropy_top10pct": entropy_top,
                    "low_reliability_top10pct": low_rel_top,
                    "student_fg_deficit_top10pct": fg_prob_low_top,
                }

                for b, rel_path in enumerate(rel_paths):
                    one = torch.ones_like(gt[b:b + 1], dtype=torch.bool)
                    gt_b = gt[b:b + 1]
                    student_pred_b = student_pred[b:b + 1]
                    selected_pred_b = selected_pred[b:b + 1]
                    selected_tid_b = selected_tid[b:b + 1]
                    active_b = active[b:b + 1]
                    fg_active_b = fg_active[b:b + 1]
                    bg_active_b = bg_active[b:b + 1]
                    region_b = rdsi_region[b:b + 1]
                    gt_fg = gt_b > 0
                    student_fg = student_pred_b > 0
                    false_negative = active_b & gt_fg & (~student_fg)
                    false_positive = active_b & (~gt_fg) & student_fg
                    true_background = active_b & (~gt_fg) & (~student_fg)
                    true_foreground = active_b & gt_fg & student_fg

                    active_pixels = masked_count(active_b)
                    selected_gain_active = region_accuracy_gain(student_pred_b, selected_pred_b, gt_b, active_b)
                    selected_dice_gain_active = region_dice_gain(student_pred_b, selected_pred_b, gt_b, active_b)
                    best_gain_active = 0.0
                    best_teacher_active = -1
                    best_dice_gain_active = 0.0
                    if active_pixels > 0:
                        gains = []
                        dice_gains = []
                        for tid, tpred in enumerate(teacher_pred_list):
                            g = region_accuracy_gain(student_pred_b, tpred[b:b + 1], gt_b, active_b)
                            dg = region_dice_gain(student_pred_b, tpred[b:b + 1], gt_b, active_b)
                            gains.append(g)
                            dice_gains.append(dg)
                        best_teacher_active = int(np.argmax(gains))
                        best_gain_active = float(np.max(gains))
                        best_dice_gain_active = float(dice_gains[best_teacher_active])
                    if active_pixels > 0:
                        tids = selected_tid_b[active_b].detach().cpu().numpy().astype(int)
                        majority_teacher = int(np.bincount(tids, minlength=len(clients)).argmax())
                    else:
                        majority_teacher = -1
                    # Existing oracle over WANN risk, not actual active. Still useful for missed-positive accounting.
                    oracle_region_best = 0.0
                    oracle_region_teacher = -1
                    if masked_count(region_b) > 0:
                        gains = [
                            region_dice_gain(student_pred_b, tpred[b:b + 1], gt_b, region_b)
                            for tpred in teacher_pred_list
                        ]
                        oracle_region_teacher = int(np.argmax(gains))
                        oracle_region_best = float(np.max(gains))

                    actual_records.append({
                        "dataset": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "sample_idx": int(idxs[b]),
                        "case": rel_path,
                        "active_pixels": active_pixels,
                        "fg_active_pixels": masked_count(fg_active_b),
                        "bg_active_pixels": masked_count(bg_active_b),
                        "rdsi_region_pixels": masked_count(region_b),
                        "active_ratio_in_region": float(active_pixels / max(masked_count(region_b), 1)),
                        "selected_accuracy_gain_active": selected_gain_active,
                        "selected_dice_gain_active": selected_dice_gain_active,
                        "best_accuracy_gain_active": best_gain_active,
                        "best_dice_gain_active": best_dice_gain_active,
                        "accepted_oracle_positive": int(active_pixels > 0 and selected_gain_active > 1e-8),
                        "false_intervention": int(active_pixels > 0 and selected_gain_active < -1e-8),
                        "missed_active_positive": int(active_pixels > 0 and best_gain_active > 1e-8 and selected_gain_active <= 1e-8),
                        "missed_region_positive": int(active_pixels <= 0 and oracle_region_best > 1e-8),
                        "majority_selected_teacher": majority_teacher,
                        "best_teacher_active": best_teacher_active,
                        "selected_matches_active_oracle": int(active_pixels > 0 and majority_teacher == best_teacher_active),
                        "oracle_region_teacher": oracle_region_teacher,
                        "oracle_region_best_dice_gain": oracle_region_best,
                        "active_fg_missing_ratio": float(masked_count(false_negative) / max(active_pixels, 1)),
                        "active_fg_excess_ratio": float(masked_count(false_positive) / max(active_pixels, 1)),
                        "active_true_bg_ratio": float(masked_count(true_background) / max(active_pixels, 1)),
                        "active_true_fg_ratio": float(masked_count(true_foreground) / max(active_pixels, 1)),
                        "selected_score_active_mean": masked_mean(selected_score[b:b + 1], active_b),
                        "selected_benefit_active_mean": masked_mean(selected_benefit[b:b + 1], active_b),
                        "selected_lift_active_mean": masked_mean(selected_lift[b:b + 1], active_b),
                        "alpha_active_mean": masked_mean(alpha[b:b + 1], active_b),
                    })

                    gt_class_prob_student = student_prob[b:b + 1].gather(1, gt_b.unsqueeze(1)).squeeze(1)
                    gt_class_prob_q = q_prob[b:b + 1].gather(1, gt_b.unsqueeze(1)).squeeze(1)
                    delta_gt = gt_class_prob_q - gt_class_prob_student
                    delta_fg = q_fg_prob[b:b + 1] - student_fg_prob[b:b + 1]
                    delta_bg = q_prob[b:b + 1, 0] - student_prob[b:b + 1, 0]
                    q_records.append({
                        "dataset": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "sample_idx": int(idxs[b]),
                        "case": rel_path,
                        "active_pixels": active_pixels,
                        "toward_gt_ratio": masked_mean((delta_gt > 1e-8).float(), active_b),
                        "away_from_gt_ratio": masked_mean((delta_gt < -1e-8).float(), active_b),
                        "delta_gt_prob_active": masked_mean(delta_gt, active_b),
                        "q_fg_delta_on_false_negative": masked_mean(delta_fg, false_negative),
                        "q_fg_delta_on_true_background": masked_mean(delta_fg, true_background),
                        "q_bg_delta_on_false_positive": masked_mean(delta_bg, false_positive),
                        "q_fg_delta_on_true_foreground": masked_mean(delta_fg, true_foreground),
                        "alpha_active_mean": masked_mean(alpha[b:b + 1], active_b),
                    })

                    for cname, cmask in component_map.items():
                        mask_b = cmask[b:b + 1]
                        pixels = masked_count(mask_b)
                        if pixels <= 0:
                            continue
                        err_rate = masked_mean(gt_error[b:b + 1].float(), mask_b)
                        fn_rate = masked_mean(((gt_b > 0) & (student_pred_b == 0)).float(), mask_b)
                        fp_rate = masked_mean(((gt_b == 0) & (student_pred_b > 0)).float(), mask_b)
                        error_pixels = masked_count(mask_b & (student_pred_b != gt_b))
                        fn_errors = masked_count(mask_b & (gt_b > 0) & (student_pred_b == 0))
                        best_dice_gain = -1e9
                        best_acc_gain = -1e9
                        for tpred in teacher_pred_list:
                            best_dice_gain = max(best_dice_gain, region_dice_gain(student_pred_b, tpred[b:b + 1], gt_b, mask_b))
                            best_acc_gain = max(best_acc_gain, region_accuracy_gain(student_pred_b, tpred[b:b + 1], gt_b, mask_b))
                        component_records.append({
                            "dataset": args.img_class,
                            "target_cid": target_cid,
                            "target_client": client,
                            "sample_idx": int(idxs[b]),
                            "case": rel_path,
                            "component": cname,
                            "pixels": pixels,
                            "pixel_ratio": float(pixels / max(masked_count(one), 1)),
                            "error_rate": err_rate,
                            "error_enrichment": float(err_rate / max(global_error_rate, 1e-8)),
                            "fn_rate": fn_rate,
                            "fp_rate": fp_rate,
                            "fg_missing_error_ratio": float(fn_errors / max(error_pixels, 1)),
                            "best_teacher_dice_gain": best_dice_gain,
                            "best_teacher_accuracy_gain": best_acc_gain,
                            "oracle_positive_in_component": int(best_dice_gain > 1e-8),
                        })

                seen += len(idxs)
                print("processed {} target_client={} samples={}".format(args.img_class, client, seen), flush=True)

    actual_df = pd.DataFrame(actual_records)
    component_df = pd.DataFrame(component_records)
    q_df = pd.DataFrame(q_records)
    profile_df = pd.DataFrame(profile_records)
    actual_df.to_csv(os.path.join(out_dir, "actual_rdsi_selection_quality.csv"), index=False)
    component_df.to_csv(os.path.join(out_dir, "wann_component_sweep.csv"), index=False)
    q_df.to_csv(os.path.join(out_dir, "q_intervention_direction.csv"), index=False)
    profile_df.to_csv(os.path.join(out_dir, "rdsi_profile_runtime.csv"), index=False)

    actual_active = actual_df[actual_df["active_pixels"] > 0].copy()
    summary = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "analysis_iter": args.analysis_iter,
        "num_samples": int(len(actual_df)),
        "num_active_samples": int(len(actual_active)),
        "active_sample_ratio": float(len(actual_active) / max(len(actual_df), 1)),
        "accepted_region_oracle_positive_ratio": float(actual_active["accepted_oracle_positive"].mean()) if len(actual_active) else 0.0,
        "accepted_region_oracle_mean_gain": float(actual_active["selected_accuracy_gain_active"].mean()) if len(actual_active) else 0.0,
        "accepted_region_oracle_mean_dice_gain": float(actual_active["selected_dice_gain_active"].mean()) if len(actual_active) else 0.0,
        "best_possible_active_accuracy_gain": float(actual_active["best_accuracy_gain_active"].mean()) if len(actual_active) else 0.0,
        "missed_positive_ratio_active": float(actual_active["missed_active_positive"].mean()) if len(actual_active) else 0.0,
        "false_intervention_ratio": float(actual_active["false_intervention"].mean()) if len(actual_active) else 0.0,
        "selected_teacher_matches_active_oracle_ratio": float(actual_active["selected_matches_active_oracle"].mean()) if len(actual_active) else 0.0,
        "missed_region_positive_no_active_ratio": float(actual_df["missed_region_positive"].mean()) if len(actual_df) else 0.0,
        "mean_active_fg_missing_ratio": float(actual_active["active_fg_missing_ratio"].mean()) if len(actual_active) else 0.0,
        "mean_active_fg_excess_ratio": float(actual_active["active_fg_excess_ratio"].mean()) if len(actual_active) else 0.0,
        "mean_active_true_bg_ratio": float(actual_active["active_true_bg_ratio"].mean()) if len(actual_active) else 0.0,
        "mean_active_true_fg_ratio": float(actual_active["active_true_fg_ratio"].mean()) if len(actual_active) else 0.0,
        "q_toward_gt_ratio": float(q_df.loc[q_df["active_pixels"] > 0, "toward_gt_ratio"].mean()) if len(actual_active) else 0.0,
        "q_away_from_gt_ratio": float(q_df.loc[q_df["active_pixels"] > 0, "away_from_gt_ratio"].mean()) if len(actual_active) else 0.0,
        "q_delta_gt_prob_active": float(q_df.loc[q_df["active_pixels"] > 0, "delta_gt_prob_active"].mean()) if len(actual_active) else 0.0,
    }
    comp_summary = component_df.groupby(["dataset", "component"], as_index=False).agg(
        pixel_ratio=("pixel_ratio", "mean"),
        error_rate=("error_rate", "mean"),
        error_enrichment=("error_enrichment", "mean"),
        fn_rate=("fn_rate", "mean"),
        fp_rate=("fp_rate", "mean"),
        fg_missing_error_ratio=("fg_missing_error_ratio", "mean"),
        best_teacher_dice_gain=("best_teacher_dice_gain", "mean"),
        oracle_positive_in_component=("oracle_positive_in_component", "mean"),
        rows=("component", "count"),
    )
    q_summary = q_df[q_df["active_pixels"] > 0].groupby(["dataset", "target_client"], as_index=False).mean(numeric_only=True)
    actual_summary_client = actual_active.groupby(["dataset", "target_client"], as_index=False).mean(numeric_only=True) if len(actual_active) else pd.DataFrame()
    profile_summary = profile_df.groupby(["dataset", "profile_key"], as_index=False).agg(
        mean=("profile_value", "mean"),
        std=("profile_value", "std"),
        rows=("profile_value", "count"),
    )

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    try:
        with pd.ExcelWriter(os.path.join(out_dir, "rdsi_selection_wann_q_diagnostics.xlsx"), engine="openpyxl") as writer:
            pd.DataFrame([summary]).to_excel(writer, sheet_name="summary", index=False)
            actual_df.to_excel(writer, sheet_name="actual_selection", index=False)
            actual_summary_client.to_excel(writer, sheet_name="selection_by_client", index=False)
            component_df.to_excel(writer, sheet_name="wann_components_raw", index=False)
            comp_summary.to_excel(writer, sheet_name="wann_components_summary", index=False)
            q_df.to_excel(writer, sheet_name="q_direction_raw", index=False)
            q_summary.to_excel(writer, sheet_name="q_direction_by_client", index=False)
            profile_summary.to_excel(writer, sheet_name="runtime_profile", index=False)
    except ModuleNotFoundError:
        pass
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
