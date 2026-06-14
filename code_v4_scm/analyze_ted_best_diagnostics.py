# -*- coding:utf-8 -*-
"""Offline diagnostics for TED best checkpoints.

This script is read-only: it loads trained TED checkpoints, recomputes WANN and
TED selection on training cases, and uses full GT only for offline diagnosis.

It answers four questions:
1. Do TED-selected regions have oracle benefit?
2. Do unselected risk errors still have oracle-positive teachers?
3. Does the selected teacher intervention produce non-trivial q/loss signal?
4. Does WANN risk cover the main student errors?
"""

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_rdsi_tcr_diagnostics import (
    auxiliary_logits,
    default_setup,
    load_model,
    parse_client_sup_types,
    primary_logits,
    rdsi_feature,
    ratio,
    to_image_tensor,
)
from dataloaders.dataset import BaseDataSets
from rgftd_reliability_distillation import rgftd_loss, select_rdsi_teacher_logits
from weak_annotation_reliability import build_wann_maps


def load_full_gt(root_path, rel_path):
    h5_path = os.path.join(root_path, rel_path)
    with h5py.File(h5_path, "r") as h5f:
        if "mask" not in h5f:
            raise KeyError("Full GT mask not found in {}".format(h5_path))
        return torch.from_numpy(h5f["mask"][:].astype(np.int64))


def resolve_checkpoints(args, num_clients):
    out = []
    for cid in range(num_clients):
        path = os.path.join(args.snapshot_path, args.checkpoint_pattern.format(cid=cid, model=args.model))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out.append(path)
    return out


def scalar(value, default=0.0):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().mean().cpu().item())
    if value is None:
        return float(default)
    return float(value)


def masked_mean(value, mask):
    denom = float(mask.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 0.0
    return float((value.float() * mask.float()).sum().detach().cpu().item() / denom)


def binary_dice(pred, gt):
    pred_fg = pred > 0
    gt_fg = gt > 0
    inter = float((pred_fg & gt_fg).float().sum().detach().cpu().item())
    denom = float(pred_fg.float().sum().detach().cpu().item() + gt_fg.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 1.0
    return 2.0 * inter / denom


def get_bool_runtime(args, name, like):
    value = getattr(args, name, None)
    if torch.is_tensor(value):
        return value.to(device=like.device).bool()
    return torch.zeros_like(like, dtype=torch.bool)


def get_float_runtime(args, name, like):
    value = getattr(args, name, None)
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=torch.float32)
    return torch.zeros_like(like, dtype=torch.float32)


def clear_rdsi_runtime(args):
    for name in list(vars(args).keys()):
        if name.startswith("_rdsi_"):
            delattr(args, name)


def run(args):
    clients, num_classes, in_chns, min_num_clients = default_setup(args.img_class)
    args.num_classes = num_classes
    args.in_chns = in_chns
    args.min_num_clients = min_num_clients
    args.client_sup_type_list = parse_client_sup_types(args, len(clients))
    checkpoints = resolve_checkpoints(args, len(clients))
    models = [load_model(args, cid, path) for cid, path in enumerate(checkpoints)]
    rows = []

    with torch.no_grad():
        for target_cid, client in enumerate(clients):
            dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=client,
                sup_type=args.client_sup_type_list[target_cid],
                img_class=args.img_class,
            )
            base_sample_list = dataset.sample_list
            if args.max_cases_per_client > 0:
                keep = list(range(min(args.max_cases_per_client, len(dataset))))
                dataset = Subset(dataset, keep)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            processed = 0
            for batch in loader:
                clear_rdsi_runtime(args)
                images = to_image_tensor(batch["image"], args.img_class).to(args.device)
                weak_label = batch["label"].long().to(args.device)
                idxs = batch["idx"].tolist()
                rel_paths = [base_sample_list[int(idx)] for idx in idxs]
                gt = torch.stack([load_full_gt(args.root_path, path) for path in rel_paths], dim=0).long().to(args.device)

                student_out = models[target_cid](images)
                student_logits = primary_logits(student_out)
                student_feature = rdsi_feature(student_out)
                student_prob = F.softmax(student_logits.detach(), dim=1)
                student_pred = student_prob.argmax(dim=1)

                maps = build_wann_maps(
                    image=images,
                    label=weak_label,
                    logits=student_logits,
                    aux_logits=auxiliary_logits(student_out),
                    sup_type=args.client_sup_type_list[target_cid],
                    img_class=args.img_class,
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )

                teacher_logits_list = []
                teacher_feature_list = []
                teacher_ids = []
                teacher_preds = []
                for teacher_cid, model in enumerate(models):
                    if int(args.exclude_same_client) == 1 and teacher_cid == target_cid:
                        continue
                    teacher_out = model(images)
                    logits = primary_logits(teacher_out).detach()
                    teacher_logits_list.append(logits)
                    teacher_feature_list.append(rdsi_feature(teacher_out).detach())
                    teacher_ids.append(teacher_cid)
                    teacher_preds.append(F.softmax(logits, dim=1).argmax(dim=1))
                if not teacher_logits_list:
                    continue

                selected_logits, select_profile = select_rdsi_teacher_logits(
                    student_logits=student_logits,
                    teacher_logits_list=teacher_logits_list,
                    teacher_ids=teacher_ids,
                    label=weak_label,
                    wann_maps=maps,
                    args=args,
                    iter_num=args.analysis_iter,
                    student_feature=student_feature,
                    teacher_feature_list=teacher_feature_list,
                )
                loss, lambda_eff, loss_profile = rgftd_loss(
                    student_logits=student_logits,
                    teacher_logits=selected_logits,
                    label=weak_label,
                    wann_maps=maps,
                    args=args,
                    iter_num=args.analysis_iter,
                    image=images,
                )

                teacher_pred_stack = torch.stack(teacher_preds, dim=0)
                teacher_correct_stack = teacher_pred_stack == gt.unsqueeze(0)
                any_teacher_correct = teacher_correct_stack.any(dim=0)

                selected_stack_index = getattr(args, "_rdsi_selected_teacher_stack_index")
                selected_stack_index = selected_stack_index.to(device=args.device).long()
                selected_teacher_pred = teacher_pred_stack.gather(
                    0,
                    selected_stack_index.unsqueeze(0).clamp(0, len(teacher_preds) - 1),
                ).squeeze(0)

                active = get_bool_runtime(args, "_rdsi_active_mask", student_pred)
                fg_active = get_bool_runtime(args, "_rdsi_fg_repair_active_mask", student_pred)
                bg_active = get_bool_runtime(args, "_rdsi_bg_suppress_active_mask", student_pred)
                bd_active = get_bool_runtime(args, "_rdsi_boundary_active_mask", student_pred)
                ted_region = get_bool_runtime(args, "_rdsi_risk_region", student_pred)
                hard_core = get_bool_runtime(args, "_rdsi_hard_core_mask", student_pred)
                selected_score = get_float_runtime(args, "_rdsi_selected_score", student_pred)
                alpha_map = get_float_runtime(args, "_rdsi_residual_alpha_map", student_pred)

                valid = getattr(maps, "valid_mask", torch.ones_like(student_pred, dtype=torch.bool))
                wann_core = maps.core_mask & valid
                wann_raw_risk = (
                    getattr(maps, "risk_mask", torch.zeros_like(valid, dtype=torch.bool))
                    | getattr(maps, "ignore_mask", torch.zeros_like(valid, dtype=torch.bool))
                    | getattr(maps, "soft_band", torch.zeros_like(valid, dtype=torch.bool))
                ) & valid & (~wann_core)
                if not bool(wann_raw_risk.any().detach().cpu().item()):
                    wann_raw_risk = ((~wann_core) & valid)
                if not bool(ted_region.any().detach().cpu().item()):
                    ted_region = wann_raw_risk

                student_error = (student_pred != gt) & valid
                student_correct = (student_pred == gt) & valid
                selected_teacher_correct = (selected_teacher_pred == gt) & valid
                selected_teacher_wrong = (selected_teacher_pred != gt) & valid

                risk_error = ted_region & student_error
                unselected_risk_error = risk_error & (~active)
                oracle_positive_risk_error = risk_error & any_teacher_correct
                selected_oracle_positive = active & student_error & selected_teacher_correct
                selected_damage = active & student_correct & selected_teacher_wrong

                for b, rel_path in enumerate(rel_paths):
                    cur_active = active[b]
                    cur_risk = ted_region[b]
                    cur_wann_risk = wann_raw_risk[b]
                    cur_core = hard_core[b] if bool(hard_core.any().detach().cpu().item()) else wann_core[b]
                    cur_error = student_error[b]
                    cur_risk_error = risk_error[b]
                    cur_unselected_risk_error = unselected_risk_error[b]
                    cur_oracle_positive = oracle_positive_risk_error[b]
                    cur_selected_oracle_positive = selected_oracle_positive[b]
                    cur_selected_damage = selected_damage[b]

                    oracle_all_pred = student_pred[b].clone()
                    oracle_all_fix = cur_risk_error & any_teacher_correct[b]
                    oracle_all_pred[oracle_all_fix] = gt[b][oracle_all_fix]
                    oracle_selected_pred = student_pred[b].clone()
                    selected_fix = cur_selected_oracle_positive
                    oracle_selected_pred[selected_fix] = gt[b][selected_fix]

                    row = {
                        "img_class": args.img_class,
                        "target_cid": int(target_cid),
                        "target_client": client,
                        "case": rel_path,
                        "sample_idx": int(idxs[b]),
                        "student_dice": binary_dice(student_pred[b], gt[b]),
                        "oracle_all_risk_dice": binary_dice(oracle_all_pred, gt[b]),
                        "oracle_selected_dice": binary_dice(oracle_selected_pred, gt[b]),
                        "ted_region_ratio": float(cur_risk.float().mean().detach().cpu().item()),
                        "wann_raw_risk_ratio": float(cur_wann_risk.float().mean().detach().cpu().item()),
                        "core_ratio": float(cur_core.float().mean().detach().cpu().item()),
                        "active_ratio": float(cur_active.float().mean().detach().cpu().item()),
                        "fg_active_ratio": float(fg_active[b].float().mean().detach().cpu().item()),
                        "bg_active_ratio": float(bg_active[b].float().mean().detach().cpu().item()),
                        "boundary_active_ratio": float(bd_active[b].float().mean().detach().cpu().item()),
                        "selected_score_on_active": masked_mean(selected_score[b], cur_active),
                        "alpha_on_active": masked_mean(alpha_map[b], cur_active),
                        "wann_risk_error_rate": ratio(cur_error & cur_wann_risk, cur_wann_risk),
                        "ted_region_error_rate": ratio(cur_error & cur_risk, cur_risk),
                        "core_error_rate": ratio(cur_error & cur_core, cur_core),
                        "wann_risk_error_recall": ratio(cur_error & cur_wann_risk, cur_error),
                        "ted_region_error_recall": ratio(cur_risk_error, cur_error),
                        "core_error_share": ratio(cur_error & cur_core, cur_error),
                        "selected_student_error_precision": ratio(cur_active & cur_error, cur_active),
                        "selected_teacher_correct_on_active": ratio(cur_active & selected_teacher_correct[b], cur_active),
                        "selected_teacher_better_precision": ratio(cur_selected_oracle_positive, cur_active),
                        "selected_teacher_damage_precision": ratio(cur_selected_damage, cur_active),
                        "selected_net_benefit_precision": ratio(cur_selected_oracle_positive, cur_active) - ratio(cur_selected_damage, cur_active),
                        "selected_oracle_positive_recall": ratio(cur_selected_oracle_positive, cur_oracle_positive),
                        "unselected_risk_error_ratio": ratio(cur_unselected_risk_error, cur_risk),
                        "unselected_oracle_positive_ratio": ratio(cur_unselected_risk_error & any_teacher_correct[b], cur_unselected_risk_error),
                        "missed_oracle_positive_share": ratio(cur_unselected_risk_error & any_teacher_correct[b], cur_oracle_positive),
                        "oracle_all_risk_dice_gain": binary_dice(oracle_all_pred, gt[b]) - binary_dice(student_pred[b], gt[b]),
                        "oracle_selected_dice_gain": binary_dice(oracle_selected_pred, gt[b]) - binary_dice(student_pred[b], gt[b]),
                        "loss_raw": scalar(loss_profile.get("rdsi_loss_raw", loss_profile.get("loss", 0.0))),
                        "loss_weighted": scalar(loss_profile.get("rdsi_loss_weighted", loss_profile.get("teacher_active_loss", 0.0))),
                        "teacher_active_loss": scalar(loss_profile.get("teacher_active_loss", 0.0)),
                        "lambda_effective": float(lambda_eff),
                        "return_reason": scalar(loss_profile.get("return_reason", 0.0)),
                        "rdsi_q_fg_delta": scalar(loss_profile.get("rdsi_q_fg_delta", 0.0)),
                        "rdsi_fg_repair_q_delta": scalar(loss_profile.get("rdsi_fg_repair_q_delta", 0.0)),
                        "rdsi_bg_suppress_q_delta": scalar(loss_profile.get("rdsi_bg_suppress_q_delta", 0.0)),
                        "rdsi_boundary_q_delta": scalar(loss_profile.get("rdsi_boundary_q_delta", 0.0)),
                        "rdsi_fg_repair_loss": scalar(loss_profile.get("rdsi_fg_repair_loss", 0.0)),
                        "rdsi_bg_suppress_loss": scalar(loss_profile.get("rdsi_bg_suppress_loss", 0.0)),
                        "rdsi_boundary_loss": scalar(loss_profile.get("rdsi_boundary_loss", 0.0)),
                        "rdsi_selected_score_mean": scalar(select_profile.get("rdsi_selected_score_mean", 0.0)),
                        "rdsi_benefit_mean": scalar(select_profile.get("rdsi_benefit_mean", 0.0)),
                        "rdsi_core_damage_mean": scalar(select_profile.get("rdsi_core_damage_mean", 0.0)),
                        "rdsi_seed_conflict_mean": scalar(select_profile.get("rdsi_seed_conflict_mean", 0.0)),
                    }
                    rows.append(row)

                processed += len(idxs)
                print("processed {} client={} samples={}".format(args.img_class, client, processed), flush=True)

    return pd.DataFrame(rows)


def summarize(df):
    metrics = [c for c in df.columns if c not in ["img_class", "target_cid", "target_client", "case", "sample_idx"]]
    run_summary = df[metrics].mean().to_frame().T
    run_summary.insert(0, "img_class", df["img_class"].iloc[0] if len(df) else "")
    client_summary = df.groupby(["img_class", "target_client"])[metrics].mean().reset_index()
    return run_summary, client_summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--img_class", choices=["prostate", "polyp", "faz"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="client_{cid}_async_{model}_best_model.pth")
    parser.add_argument("--client_sup_types", default="")
    parser.add_argument("--max_cases_per_client", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--analysis_iter", type=int, default=5000)
    parser.add_argument("--exclude_same_client", type=int, default=1)
    parser.add_argument("--model", default="unet_univ5")
    parser.add_argument("--prompt", default="universal")
    parser.add_argument("--attention", default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=384)

    parser.add_argument("--wann_core_thresh", type=float, default=0.65)
    parser.add_argument("--wann_soft_thresh", type=float, default=0.25)
    parser.add_argument("--wann_core_min_weight", type=float, default=0.8)
    parser.add_argument("--wann_r_max", type=float, default=1.2)
    parser.add_argument("--wann_dilated_support_score", type=float, default=0.55)
    parser.add_argument("--wann_appearance_temp", type=float, default=1.5)
    parser.add_argument("--wann_texture_kernel_size", type=int, default=5)
    parser.add_argument("--wann_texture_temp", type=float, default=1.0)
    parser.add_argument("--wann_texture_weight", type=float, default=0.25)
    parser.add_argument("--wann_pred_start_iter", type=int, default=800)
    parser.add_argument("--wann_entropy_weight", type=float, default=0.5)
    parser.add_argument("--wann_agreement_weight", type=float, default=0.5)
    parser.add_argument("--wann_global_agreement_weight", type=float, default=0.5)
    parser.add_argument("--wann_keypoint_soft_radius", type=int, default=2)
    parser.add_argument("--wann_scribble_soft_radius", type=int, default=4)
    parser.add_argument("--wann_box_soft_radius", type=int, default=2)
    parser.add_argument("--wann_mask_soft_radius", type=int, default=1)
    parser.add_argument("--wann_seed_support_erode_radius", type=int, default=1)
    parser.add_argument("--wann_seed_support_box_erode_radius", type=int, default=1)
    parser.add_argument("--wann_seed_support_block_erode_radius", type=int, default=1)
    parser.add_argument("--wann_sparse_adaptive_core", type=int, default=1)
    parser.add_argument("--wann_sparse_min_core_ratio", type=float, default=0.06)
    parser.add_argument("--wann_sparse_max_core_ratio", type=float, default=0.12)
    parser.add_argument("--wann_sparse_core_min_reliability", type=float, default=0.25)
    parser.add_argument("--wann_low_confident_thresh", type=float, default=0.95)
    parser.add_argument("--wann_target_core_ratio", type=float, default=0.08)
    parser.add_argument("--wann_core_deficit_soft_boost", type=float, default=2.0)

    parser.add_argument("--rgftd_lambda", type=float, default=0.1)
    parser.add_argument("--rgftd_warmup_iters", type=int, default=800)
    parser.add_argument("--rgftd_rampup_iters", type=int, default=800)
    parser.add_argument("--rgftd_temperature", type=float, default=1.0)
    parser.add_argument("--rgftd_low_r_thresh", type=float, default=0.25)
    parser.add_argument("--rgftd_teacher_foreground_radius", type=int, default=2)
    parser.add_argument("--rgftd_teacher_fg_prob_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_fg_topk_ratio", type=float, default=0.002)
    parser.add_argument("--rgftd_min_foreground_pixels", type=int, default=8)
    parser.add_argument("--rgftd_min_foreground_ratio", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_fg_topk_min_pixels", type=int, default=8)
    parser.add_argument("--rgftd_teacher_student_fg_margin", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rgftd_teacher_bg_conf_thresh", type=float, default=0.98)
    parser.add_argument("--rgftd_bg_max_fg_prob", type=float, default=0.15)
    parser.add_argument("--rgftd_student_conf_thresh", type=float, default=0.80)
    parser.add_argument("--rgftd_student_entropy_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_use_soft_band", type=int, default=0)
    parser.add_argument("--rgftd_background_weight", type=float, default=0.25)
    parser.add_argument("--rgftd_teacher_validation_enabled", type=int, default=1)
    parser.add_argument("--rgftd_teacher_reliability_min", type=float, default=0.55)
    parser.add_argument("--rgftd_teacher_core_agree_floor", type=float, default=0.80)
    parser.add_argument("--rgftd_teacher_support_agree_floor", type=float, default=0.70)
    parser.add_argument("--rgftd_teacher_support_prob_floor", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_conf_floor", type=float, default=0.85)
    parser.add_argument("--rgftd_teacher_class_reliability_min", type=float, default=0.50)
    parser.add_argument("--rgftd_teacher_max_core_conflict", type=float, default=0.20)
    parser.add_argument("--rgftd_teacher_score_core_weight", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_score_support_weight", type=float, default=0.30)
    parser.add_argument("--rgftd_teacher_score_class_weight", type=float, default=0.25)
    parser.add_argument("--rgftd_teacher_score_conf_weight", type=float, default=0.10)
    parser.add_argument("--rgftd_teacher_release_prob_floor", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_release_margin_floor", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_release_class_floor", type=float, default=0.50)
    parser.add_argument("--rgftd_teacher_release_min", type=float, default=0.03)
    parser.add_argument("--rgftd_active_fg_topk_ratio", type=float, default=0.002)
    parser.add_argument("--rgftd_active_fg_topk_min_pixels", type=int, default=8)
    parser.add_argument("--rgftd_active_fg_topk_max_pixels", type=int, default=4096)
    parser.add_argument("--rgftd_max_bg_fg_ratio", type=float, default=1.0)
    parser.add_argument("--rgftd_allow_bg_without_fg", type=int, default=0)
    parser.add_argument("--rgftd_lambda_eff_cap", type=float, default=0.02)
    parser.add_argument("--rgftd_spatial_support_enabled", type=int, default=1)
    parser.add_argument("--rgftd_spatial_support_radius", type=int, default=2)
    parser.add_argument("--rgftd_spatial_candidate_weight", type=float, default=1.0)
    parser.add_argument("--rgftd_spatial_near_seed_weight", type=float, default=0.75)
    parser.add_argument("--rgftd_spatial_far_weight", type=float, default=0.15)
    parser.add_argument("--rgftd_refine_enabled", type=int, default=1)
    parser.add_argument("--rgftd_refine_iters", type=int, default=3)
    parser.add_argument("--rgftd_refine_affinity_sigma", type=float, default=0.75)
    parser.add_argument("--rgftd_refine_affinity_mix", type=float, default=0.35)
    parser.add_argument("--rgftd_refine_seed_strength", type=float, default=0.95)
    parser.add_argument("--rgftd_refine_core_anchor_radius", type=int, default=1)
    parser.add_argument("--rgftd_refine_unsupported_fg_scale", type=float, default=0.25)
    parser.add_argument("--rgftd_refine_fg_floor", type=float, default=0.02)
    parser.add_argument("--rgftd_refine_bg_ceiling", type=float, default=0.98)
    parser.add_argument("--rgftd_refine_min_fg_mass", type=float, default=1.0)
    parser.add_argument("--rgftd_refine_min_roi_pixels", type=float, default=1.0)

    parser.add_argument("--rdsi_teacher_sup_types", default="")
    parser.add_argument("--rdsi_entropy_increase_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_entropy_increase_scale", type=float, default=0.35)
    parser.add_argument("--rdsi_fg_excess_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_fg_excess_scale", type=float, default=0.20)
    parser.add_argument("--rdsi_residual_alpha", type=float, default=0.35)
    parser.add_argument("--rdsi_benefit_score_floor", type=float, default=1e-6)
    parser.add_argument("--rdsi_benefit_topk_ratio", type=float, default=0.002)
    parser.add_argument("--rdsi_benefit_topk_min_pixels", type=int, default=8)
    parser.add_argument("--rdsi_benefit_topk_max_pixels", type=int, default=0)
    parser.add_argument("--rdsi_boundary_radius", type=int, default=-1)
    parser.add_argument("--rdsi_boundary_uncertainty_width", type=float, default=0.25)
    parser.add_argument("--rdsi_core_damage_veto", type=float, default=0.30)
    parser.add_argument("--rdsi_unsafe_gap_scale", type=float, default=0.40)
    parser.add_argument("--rdsi_boundary_support_weight", type=float, default=0.35)
    parser.add_argument("--rdsi_core_preserving_fg_weight", type=float, default=0.25)
    parser.add_argument("--rdsi_teacher_reliability_weight", type=float, default=0.20)
    parser.add_argument("--rdsi_student_risk_weight", type=float, default=0.20)
    parser.add_argument("--rdsi_core_damage_weight", type=float, default=0.45)
    parser.add_argument("--rdsi_unsafe_gap_weight", type=float, default=0.30)
    parser.add_argument("--rdsi_foreground_excess_weight", type=float, default=0.25)
    parser.add_argument("--rdsi_knowledge_tiebreak_weight", type=float, default=0.0)
    parser.add_argument("--rdsi_safe_budget_gain", type=float, default=1.50)
    parser.add_argument("--rdsi_unsafe_budget_decay", type=float, default=1.00)
    parser.add_argument("--rdsi_max_budget_factor", type=float, default=4.0)
    parser.add_argument("--rdsi_hard_core_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rdsi_hard_core_entropy_thresh", type=float, default=0.25)
    parser.add_argument("--rdsi_hard_core_reliability_thresh", type=float, default=0.65)
    parser.add_argument("--rdsi_core_reopen_boundary_floor", type=float, default=0.20)
    return parser


def main():
    args = build_parser().parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    if not args.rdsi_teacher_sup_types:
        args.rdsi_teacher_sup_types = args.client_sup_types
    os.makedirs(args.output_dir, exist_ok=True)
    df = run(args)
    sample_csv = os.path.join(args.output_dir, "ted_best_diagnostic_samples.csv")
    run_csv = os.path.join(args.output_dir, "ted_best_diagnostic_run_summary.csv")
    client_csv = os.path.join(args.output_dir, "ted_best_diagnostic_client_summary.csv")
    summary_json = os.path.join(args.output_dir, "ted_best_diagnostic_summary.json")
    df.to_csv(sample_csv, index=False)
    run_summary, client_summary = summarize(df)
    run_summary.to_csv(run_csv, index=False)
    client_summary.to_csv(client_csv, index=False)
    payload = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "analysis_iter": args.analysis_iter,
        "num_samples": int(len(df)),
        "files": {
            "samples": sample_csv,
            "run_summary": run_csv,
            "client_summary": client_csv,
        },
        "run_summary": run_summary.to_dict(orient="records"),
        "client_summary": client_summary.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
