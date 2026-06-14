# -*- coding:utf-8 -*-
"""Offline simulation of consensus reweighting on original TED selections.

This diagnostic does not train or modify model checkpoints. It imports the
original TED selection module and measures what would happen if we only
reweighted already-selected TED boundary pixels by teacher-pool consensus.
"""

import argparse
import json
import os

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_rdsi_tcr_diagnostics import (
    auxiliary_logits,
    default_setup,
    dilate,
    load_model,
    parse_client_sup_types,
    primary_logits,
    ratio,
    rdsi_feature,
    to_image_tensor,
)
from dataloaders.dataset import BaseDataSets
from rgftd_reliability_distillation_ted_orig import select_rdsi_teacher_logits
from weak_annotation_reliability import build_wann_maps


def load_full_gt(root_path, rel_path):
    with h5py.File(os.path.join(root_path, rel_path), "r") as h5f:
        if "mask" not in h5f:
            raise KeyError("Full GT mask not found in {}".format(rel_path))
        return torch.from_numpy(h5f["mask"][:].astype("int64"))


def resolve_checkpoints(args, num_clients):
    out = []
    for cid in range(num_clients):
        path = os.path.join(args.snapshot_path, args.checkpoint_pattern.format(cid=cid, model=args.model))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        out.append(path)
    return out


def numeric_map(args, name, like):
    value = getattr(args, name, None)
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=torch.float32)
    return torch.zeros_like(like, dtype=torch.float32)


def bool_map(args, name, like):
    value = getattr(args, name, None)
    if torch.is_tensor(value):
        return value.to(device=like.device).bool()
    return torch.zeros_like(like, dtype=torch.bool)


def masked_mean(value, mask):
    denom = float(mask.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 0.0
    return float((value.float() * mask.float()).sum().detach().cpu().item() / denom)


def run(args):
    clients, num_classes, in_chns, min_num_clients = default_setup(args.img_class)
    args.num_classes = num_classes
    args.in_chns = in_chns
    args.min_num_clients = min_num_clients
    args.client_sup_type_list = parse_client_sup_types(args, len(clients))
    models = [load_model(args, cid, path) for cid, path in enumerate(resolve_checkpoints(args, len(clients)))]
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
                dataset = Subset(dataset, list(range(min(args.max_cases_per_client, len(dataset)))))
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            processed = 0
            for batch in loader:
                images = to_image_tensor(batch["image"], args.img_class).to(args.device)
                weak_label = batch["label"].long().to(args.device)
                idxs = batch["idx"].tolist()
                rel_paths = [base_sample_list[int(i)] for i in idxs]
                gt = torch.stack([load_full_gt(args.root_path, p) for p in rel_paths], dim=0).long().to(args.device)
                gt_fg = gt > 0

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
                    tout = model(images)
                    logits = primary_logits(tout).detach()
                    teacher_logits_list.append(logits)
                    teacher_feature_list.append(rdsi_feature(tout).detach())
                    teacher_ids.append(teacher_cid)
                    teacher_preds.append(F.softmax(logits, dim=1).argmax(dim=1))
                if not teacher_logits_list:
                    continue

                select_rdsi_teacher_logits(
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

                teacher_pred_stack = torch.stack(teacher_preds, dim=0)
                teacher_fg_vote = (teacher_pred_stack > 0).float().mean(dim=0)
                teacher_bg_vote = (teacher_pred_stack == 0).float().mean(dim=0)
                majority = torch.maximum(teacher_fg_vote, teacher_bg_vote)
                disagreement = 1.0 - majority
                consensus_signal = (majority - args.majority_thresh).clamp_min(0.0) / max(1.0 - args.majority_thresh, 1e-6)
                reweight = (1.0 + args.gain * consensus_signal).clamp(args.min_weight, args.max_weight)

                active = bool_map(args, "_rdsi_active_mask", student_pred)
                boundary_active = bool_map(args, "_rdsi_boundary_active_mask", student_pred)
                selected_score = numeric_map(args, "_rdsi_selected_score", student_pred)
                selected_index = getattr(args, "_rdsi_selected_teacher_stack_index", None)
                if not torch.is_tensor(selected_index):
                    selected_index = torch.zeros_like(student_pred, dtype=torch.long)
                selected_index = selected_index.to(device=args.device).long()
                selected_teacher_pred = teacher_pred_stack.gather(0, selected_index.unsqueeze(0).clamp(0, len(teacher_preds) - 1)).squeeze(0)

                valid = getattr(maps, "valid_mask", torch.ones_like(student_pred, dtype=torch.bool))
                core = maps.core_mask & valid
                risk = ((maps.ignore_mask | maps.soft_band) & (~core) & valid)
                gt_boundary = risk & dilate(gt_fg, args.boundary_radius) & dilate(~gt_fg, args.boundary_radius)
                boundary_error = gt_boundary & (student_pred != gt)
                boundary_correct = gt_boundary & (student_pred == gt)
                fp = risk & (student_pred > 0) & (~gt_fg)
                fn = risk & (student_pred == 0) & gt_fg

                weighted_score_before = selected_score * boundary_active.float()
                weighted_score_after = weighted_score_before * reweight
                boost_region = boundary_active & (reweight > (1.0 + 1e-6))
                suppress_region = boundary_active & (reweight < (1.0 - 1e-6))

                for b, rel_path in enumerate(rel_paths):
                    ba = boundary_active[b]
                    row = {
                        "img_class": args.img_class,
                        "target_cid": int(target_cid),
                        "target_client": client,
                        "case": rel_path,
                        "sample_idx": int(idxs[b]),
                        "active_ratio": float(active[b].float().mean().detach().cpu().item()),
                        "boundary_active_ratio": float(ba.float().mean().detach().cpu().item()),
                        "boundary_share_of_active": ratio(ba, active[b]),
                        "active_coverage_changed": 0.0,
                        "boundary_error_precision": ratio(ba & boundary_error[b], ba),
                        "boundary_error_recall": ratio(ba & boundary_error[b], boundary_error[b]),
                        "boundary_correct_overlap": ratio(ba & boundary_correct[b], ba),
                        "selected_teacher_correct_on_boundary": ratio(ba & (selected_teacher_pred[b] == gt[b]), ba),
                        "consensus_mean_on_boundary": masked_mean(consensus_signal[b], ba),
                        "disagreement_mean_on_boundary": masked_mean(disagreement[b], ba),
                        "reweight_mean_on_boundary": masked_mean(reweight[b], ba),
                        "weighted_score_mass_ratio": float(weighted_score_after[b].sum().detach().cpu().item() / max(float(weighted_score_before[b].sum().detach().cpu().item()), 1e-6)),
                        "boost_boundary_ratio": ratio(boost_region[b], ba),
                        "boost_error_precision": ratio(boost_region[b] & boundary_error[b], boost_region[b]),
                        "boost_error_recall": ratio(boost_region[b] & boundary_error[b], boundary_error[b]),
                        "boost_correct_overlap": ratio(boost_region[b] & boundary_correct[b], boost_region[b]),
                        "fp_overlap_with_boundary_active": ratio(ba & fp[b], fp[b]),
                        "fn_overlap_with_boundary_active": ratio(ba & fn[b], fn[b]),
                        "risk_error_rate": ratio((student_pred[b] != gt[b]) & risk[b], risk[b]),
                    }
                    rows.append(row)
                processed += len(idxs)
                print("processed {} client={} samples={}".format(args.img_class, client, processed), flush=True)

    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ["img_class", "target_cid", "target_client", "case", "sample_idx"]]
    run_summary = df[metrics].mean().to_frame().T
    run_summary.insert(0, "img_class", args.img_class)
    client_summary = df.groupby(["img_class", "target_client"])[metrics].mean().reset_index()
    os.makedirs(args.output_dir, exist_ok=True)
    sample_csv = os.path.join(args.output_dir, "consensus_reweight_samples.csv")
    run_csv = os.path.join(args.output_dir, "consensus_reweight_run_summary.csv")
    client_csv = os.path.join(args.output_dir, "consensus_reweight_client_summary.csv")
    summary_json = os.path.join(args.output_dir, "consensus_reweight_summary.json")
    df.to_csv(sample_csv, index=False)
    run_summary.to_csv(run_csv, index=False)
    client_summary.to_csv(client_csv, index=False)
    payload = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "num_samples": int(len(df)),
        "gain": args.gain,
        "files": {"samples": sample_csv, "run_summary": run_csv, "client_summary": client_csv},
        "run_summary": run_summary.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


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
    parser.add_argument("--majority_thresh", type=float, default=0.5)
    parser.add_argument("--gain", type=float, default=0.25)
    parser.add_argument("--min_weight", type=float, default=1.0)
    parser.add_argument("--max_weight", type=float, default=1.25)
    parser.add_argument("--boundary_radius", type=int, default=2)

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

    parser.add_argument("--rgftd_temperature", type=float, default=1.0)
    parser.add_argument("--rgftd_low_r_thresh", type=float, default=0.25)
    parser.add_argument("--rgftd_teacher_foreground_radius", type=int, default=2)
    parser.add_argument("--rgftd_teacher_fg_prob_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rgftd_teacher_bg_conf_thresh", type=float, default=0.98)
    parser.add_argument("--rgftd_bg_max_fg_prob", type=float, default=0.15)
    parser.add_argument("--rgftd_student_conf_thresh", type=float, default=0.80)
    parser.add_argument("--rgftd_student_entropy_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_spatial_support_enabled", type=int, default=1)
    parser.add_argument("--rgftd_spatial_support_radius", type=int, default=2)
    parser.add_argument("--rgftd_spatial_candidate_weight", type=float, default=1.0)
    parser.add_argument("--rgftd_spatial_near_seed_weight", type=float, default=0.75)
    parser.add_argument("--rgftd_spatial_far_weight", type=float, default=0.15)
    parser.add_argument("--rdsi_entropy_increase_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_entropy_increase_scale", type=float, default=0.35)
    parser.add_argument("--rdsi_fg_excess_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_fg_excess_scale", type=float, default=0.20)
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
    run(args)


if __name__ == "__main__":
    main()
