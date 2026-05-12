# -*- coding:utf-8 -*-
import argparse
import copy
import json
import os
from collections import defaultdict
from contextlib import contextmanager

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory
from rgftd_reliability_distillation import get_rgftd_lambda, rgftd_loss
from weak_annotation_reliability import build_wann_maps


PROSTATE_CLIENTS = ["client1", "client2", "client3", "client4", "client5", "client6"]
PROSTATE_SUP_TYPES = ["block", "keypoint", "scribble", "keypoint", "scribble", "box"]
ODOC_CLIENTS = ["client1", "client2", "client3", "client4", "client5"]
ODOC_SUP_TYPES = ["scribble", "scribble_noisy", "scribble_noisy", "keypoint", "block"]


def _default_client_setup(img_class):
    if img_class == "prostate":
        return PROSTATE_CLIENTS, PROSTATE_SUP_TYPES, 2, 1, 6, 12
    if img_class == "odoc":
        return ODOC_CLIENTS, ODOC_SUP_TYPES, 3, 3, 5, 8
    raise ValueError("Unsupported img_class: {}".format(img_class))


def _primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def _extract_student_logits(model_name, model_out):
    if torch.is_tensor(model_out):
        return model_out, None
    primary = model_out[0]
    auxiliary = None
    if model_name in ["unet_univ2", "unet_univ3", "unet_univ4", "unet_univ5"]:
        if len(model_out) > 8:
            auxiliary = model_out[8]
    return primary, auxiliary


def _average_profiles(profile_a, profile_b):
    keys = sorted(set(profile_a.keys()) | set(profile_b.keys()))
    profile = {}
    for key in keys:
        a = profile_a.get(key)
        b = profile_b.get(key)
        if a is None:
            profile[key] = b
        elif b is None:
            profile[key] = a
        else:
            profile[key] = 0.5 * (a + b)
    return profile


def _to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _prepare_batch(sampled_batch, img_class, device):
    image = sampled_batch["image"]
    label = sampled_batch["label"]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)
    if img_class in ["faz", "prostate"]:
        if image.dim() == 3:
            image = image.unsqueeze(1)
    elif img_class in ["odoc", "polyp"]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
    return image.float().to(device), label.long().to(device)


def _make_model_args(base_args, cid, sup_type):
    model_args = copy.deepcopy(base_args)
    model_args.cid = cid
    model_args.sup_type = sup_type
    return model_args


@contextmanager
def _cpu_safe_cuda_patch(enabled):
    if not enabled:
        yield
        return

    original_module_cuda = torch.nn.Module.cuda
    original_tensor_cuda = torch.Tensor.cuda

    def _module_cuda_noop(self, device=None):
        del device
        return self

    def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        del device, non_blocking, memory_format
        return self

    torch.nn.Module.cuda = _module_cuda_noop
    torch.Tensor.cuda = _tensor_cuda_noop
    try:
        yield
    finally:
        torch.nn.Module.cuda = original_module_cuda
        torch.Tensor.cuda = original_tensor_cuda


def _load_client_model(base_args, cid, sup_type, checkpoint_path, device):
    model_args = _make_model_args(base_args, cid, sup_type)
    with _cpu_safe_cuda_patch(device.type != "cuda"):
        model = net_factory(
            model_args,
            net_type=model_args.model,
            in_chns=model_args.in_chns,
            class_num=model_args.num_classes,
        )
    model = model.to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _cache_target_batch(image, label, student_logits, student_aux_logits):
    return {
        "image": image.detach().cpu(),
        "label": label.detach().cpu(),
        "student_logits": student_logits.detach().cpu(),
        "student_aux_logits": None if student_aux_logits is None else student_aux_logits.detach().cpu(),
    }


def _resolve_checkpoint_path(snapshot_path, checkpoint_pattern, cid, model_name):
    filename = checkpoint_pattern.format(cid=cid, model=model_name)
    checkpoint_path = os.path.join(snapshot_path, filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
    return checkpoint_path


def _build_target_loader(args, target_client, target_sup_type):
    dataset = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        transform=None,
        client=target_client,
        sup_type=target_sup_type,
        img_class=args.img_class,
    )
    if args.max_cases_per_client > 0:
        max_cases = min(len(dataset), int(args.max_cases_per_client))
        dataset = Subset(dataset, list(range(max_cases)))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def _aggregate_pair_metrics(batch_records):
    if not batch_records:
        return {}

    metric_keys = sorted({
        key
        for record in batch_records
        for key in record["metrics"].keys()
    })
    agg = {}
    total_cases = sum(record["cases"] for record in batch_records)
    total_batches = len(batch_records)
    reason_counts = defaultdict(int)
    for record in batch_records:
        reason = int(round(record["metrics"].get("return_reason", 0.0)))
        reason_counts[reason] += record["cases"]

    for key in metric_keys:
        weighted_sum = sum(record["metrics"].get(key, 0.0) * record["cases"] for record in batch_records)
        agg[key] = weighted_sum / max(total_cases, 1)

    agg["num_cases"] = int(total_cases)
    agg["num_batches"] = int(total_batches)
    for reason_id in [0, 1, 2, 3]:
        agg["return_reason_{}_ratio".format(reason_id)] = reason_counts[reason_id] / max(total_cases, 1)
    return agg


def _long_record(base_info, metrics):
    record = dict(base_info)
    record.update(metrics)
    return record


def _matrix_from_records(df, value_key):
    if value_key not in df.columns:
        return None
    matrix = df.pivot(index="teacher_name", columns="target_name", values=value_key)
    return matrix


def _summarize_top_teachers(df, score_key, exclude_self=False):
    if score_key not in df.columns:
        return {}
    top_summary = {}
    for target_name, sub_df in df.groupby("target_name"):
        if exclude_self and "same_client" in sub_df.columns:
            sub_df = sub_df[sub_df["same_client"] == 0]
        sub_df = sub_df.sort_values(score_key, ascending=False)
        top_summary[target_name] = [
            {
                "teacher_name": row["teacher_name"],
                "teacher_cid": int(row["teacher_cid"]),
                "teacher_sup_type": row["teacher_sup_type"],
                score_key: float(row[score_key]),
                "teacher_reliability": float(row.get("teacher_reliability", 0.0)),
                "release_factor": float(row.get("release_factor", 0.0)),
                "lambda_effective": float(row.get("lambda_effective", 0.0)),
                "return_reason_0_ratio": float(row.get("return_reason_0_ratio", 0.0)),
            }
            for _, row in sub_df.iterrows()
        ]
    return top_summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_path", type=str, required=True,
                        help="Directory containing per-client best checkpoints.")
    parser.add_argument("--root_path", type=str, required=True,
                        help="Dataset root path used for train split loading.")
    parser.add_argument("--img_class", type=str, required=True, choices=["prostate", "odoc"],
                        help="Dataset type.")
    parser.add_argument("--client_list", nargs="+", type=str, default=None,
                        help="Ordered physical client names, e.g. client1 client2 ...")
    parser.add_argument("--sup_type_list", nargs="+", type=str, default=None,
                        help="Ordered supervision types aligned with client_list.")
    parser.add_argument("--checkpoint_pattern", type=str, default="client_{cid}_async_{model}_best_model.pth",
                        help="Checkpoint filename pattern under snapshot_path.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for csv/json summaries. Defaults to snapshot_path/teacher_pool_analysis.")
    parser.add_argument("--analysis_iter", type=int, default=-1,
                        help="Iter number fed into RGFTD analysis. -1 means warmup+rampup.")
    parser.add_argument("--max_cases_per_client", type=int, default=0,
                        help="Limit train cases per target client. <=0 means all.")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Offline analysis batch size. 0 means dataset default.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Runtime device for offline analysis.")
    parser.add_argument("--model", type=str, default="unet_univ5")
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--in_chns", type=int, default=0)
    parser.add_argument("--min_num_clients", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--prompt", type=str, default="universal")
    parser.add_argument("--attention", type=str, default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="FedUniV2.1")
    parser.add_argument("--wann_enabled", type=int, default=1)
    parser.add_argument("--rgftd_enabled", type=int, default=1)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[256, 256])

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

    parser.add_argument("--rgftd_lambda", type=float, default=0.1)
    parser.add_argument("--rgftd_warmup_iters", type=int, default=800)
    parser.add_argument("--rgftd_rampup_iters", type=int, default=800)
    parser.add_argument("--rgftd_teacher_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rgftd_min_foreground_pixels", type=int, default=8)
    parser.add_argument("--rgftd_min_foreground_ratio", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_fg_prob_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_teacher_fg_topk_ratio", type=float, default=0.002)
    parser.add_argument("--rgftd_teacher_fg_topk_min_pixels", type=int, default=8)
    parser.add_argument("--rgftd_teacher_foreground_radius", type=int, default=2)
    parser.add_argument("--rgftd_teacher_student_fg_margin", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_bg_conf_thresh", type=float, default=0.98)
    parser.add_argument("--rgftd_bg_max_fg_prob", type=float, default=0.15)
    parser.add_argument("--rgftd_student_conf_thresh", type=float, default=0.80)
    parser.add_argument("--rgftd_student_entropy_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_low_r_thresh", type=float, default=0.25)
    parser.add_argument("--rgftd_temperature", type=float, default=1.0)
    parser.add_argument("--rgftd_use_soft_band", type=int, default=0)
    parser.add_argument("--rgftd_background_weight", type=float, default=0.25)
    parser.add_argument("--rgftd_skip_background_only", type=int, default=1)
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
    parser.add_argument("--rgftd_spatial_support_enabled", type=int, default=1)
    parser.add_argument("--rgftd_spatial_support_radius", type=int, default=2)
    parser.add_argument("--rgftd_spatial_candidate_weight", type=float, default=1.0)
    parser.add_argument("--rgftd_spatial_near_seed_weight", type=float, default=0.75)
    parser.add_argument("--rgftd_spatial_far_weight", type=float, default=0.15)
    parser.add_argument("--rgftd_max_bg_fg_ratio", type=float, default=1.0)
    parser.add_argument("--rgftd_allow_bg_without_fg", type=int, default=0)
    parser.add_argument("--rgftd_lambda_eff_cap", type=float, default=0.02)
    parser.add_argument("--rgftd_refine_enabled", type=int, default=0)
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
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    default_clients, default_sup_types, default_num_classes, default_in_chns, default_min_clients, default_batch_size = _default_client_setup(args.img_class)
    if args.client_list is None:
        args.client_list = default_clients
    if args.sup_type_list is None:
        args.sup_type_list = default_sup_types
    if len(args.client_list) != len(args.sup_type_list):
        raise ValueError("client_list and sup_type_list must have the same length")
    if args.num_classes <= 0:
        args.num_classes = default_num_classes
    if args.in_chns <= 0:
        args.in_chns = default_in_chns
    if args.min_num_clients <= 0:
        args.min_num_clients = default_min_clients
    if args.batch_size <= 0:
        args.batch_size = default_batch_size
    if args.analysis_iter < 0:
        args.analysis_iter = int(args.rgftd_warmup_iters + args.rgftd_rampup_iters)

    output_dir = args.output_dir or os.path.join(args.snapshot_path, "teacher_pool_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    checkpoint_paths = {}
    for cid in range(len(args.client_list)):
        checkpoint_paths[cid] = _resolve_checkpoint_path(
            args.snapshot_path,
            args.checkpoint_pattern,
            cid,
            args.model,
        )

    long_records = []
    for target_cid, (target_client, target_sup_type) in enumerate(zip(args.client_list, args.sup_type_list)):
        target_model = _load_client_model(
            args,
            target_cid,
            target_sup_type,
            checkpoint_paths[target_cid],
            device,
        )
        target_loader = _build_target_loader(args, target_client, target_sup_type)
        cached_target_batches = []

        with torch.no_grad():
            for sampled_batch in target_loader:
                volume_batch, label_batch = _prepare_batch(sampled_batch, args.img_class, device)
                student_out = target_model(volume_batch)
                student_logits, student_aux_logits = _extract_student_logits(args.model, student_out)
                cached_target_batches.append(
                    _cache_target_batch(volume_batch, label_batch, student_logits, student_aux_logits)
                )

        pair_batch_records = defaultdict(list)
        for teacher_cid, teacher_sup_type in enumerate(args.sup_type_list):
            if teacher_cid == target_cid:
                teacher_model = target_model
            else:
                teacher_model = _load_client_model(
                    args,
                    teacher_cid,
                    teacher_sup_type,
                    checkpoint_paths[teacher_cid],
                    device,
                )

            with torch.no_grad():
                for cached_batch in cached_target_batches:
                    volume_batch = cached_batch["image"].to(device)
                    label_batch = cached_batch["label"].to(device)
                    student_logits = cached_batch["student_logits"].to(device)
                    student_aux_logits = cached_batch["student_aux_logits"]
                    if student_aux_logits is not None:
                        student_aux_logits = student_aux_logits.to(device)

                    batch_cases = int(label_batch.shape[0])
                    wann_maps = build_wann_maps(
                        image=volume_batch,
                        label=label_batch,
                        logits=student_logits,
                        aux_logits=student_aux_logits,
                        sup_type=target_sup_type,
                        img_class=args.img_class,
                        num_classes=args.num_classes,
                        iter_num=args.analysis_iter,
                        args=args,
                        ref_logits=None,
                    )

                    teacher_out = teacher_model(volume_batch)
                    teacher_logits = _primary_logits(teacher_out)
                    loss_seg, lambda_seg, profile_seg = rgftd_loss(
                        student_logits,
                        teacher_logits,
                        label_batch,
                        wann_maps,
                        args,
                        args.analysis_iter,
                        image=volume_batch,
                    )
                    if student_aux_logits is not None:
                        loss_aux, lambda_aux, profile_aux = rgftd_loss(
                            student_aux_logits,
                            teacher_logits,
                            label_batch,
                            wann_maps,
                            args,
                            args.analysis_iter,
                            image=volume_batch,
                        )
                        loss_value = 0.5 * (_to_float(loss_seg) + _to_float(loss_aux))
                        lambda_value = 0.5 * (float(lambda_seg) + float(lambda_aux))
                        profile = _average_profiles(profile_seg, profile_aux)
                    else:
                        loss_value = _to_float(loss_seg)
                        lambda_value = float(lambda_seg)
                        profile = profile_seg

                    metric_map = {
                        "analysis_loss": loss_value,
                        "analysis_lambda_effective": lambda_value,
                        "schedule_lambda": float(get_rgftd_lambda(args.analysis_iter, args)),
                    }
                    for key, value in wann_maps.profile.items():
                        metric_map["wann_{}".format(key)] = _to_float(value)
                    for key, value in profile.items():
                        metric_map[key] = _to_float(value)
                    pair_batch_records[teacher_cid].append({
                        "cases": batch_cases,
                        "metrics": metric_map,
                    })

            if teacher_cid != target_cid:
                del teacher_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del target_model
        del cached_target_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for teacher_cid, batch_records in pair_batch_records.items():
            metrics = _aggregate_pair_metrics(batch_records)
            long_records.append(_long_record({
                "target_cid": target_cid,
                "target_name": target_client,
                "target_sup_type": target_sup_type,
                "teacher_cid": teacher_cid,
                "teacher_name": args.client_list[teacher_cid],
                "teacher_sup_type": args.sup_type_list[teacher_cid],
                "same_client": int(target_cid == teacher_cid),
            }, metrics))

    df = pd.DataFrame(long_records)
    long_csv = os.path.join(output_dir, "teacher_pool_pair_metrics.csv")
    df.to_csv(long_csv, index=False)

    matrix_keys = [
        "analysis_lambda_effective",
        "lambda_pre_safety",
        "lambda_after_safety",
        "teacher_reliability",
        "release_factor",
        "release_raw",
        "teacher_core_conflict",
        "teacher_seed_support_fg_recall",
        "teacher_seed_support_fg_prob_mean",
        "teacher_seed_support_fg_margin_mean",
        "active_foreground_ratio",
        "active_background_ratio",
        "spatial_support_ratio",
        "foreground_candidate_ratio",
        "spatial_weight_mean",
        "spatial_weight_candidate_mean",
        "spatial_weight_near_seed_mean",
        "spatial_weight_far_mean",
        "spatial_loss_scale",
        "active_foreground_pixels_pre_spatial",
        "active_foreground_spatial_keep_ratio",
        "active_foreground_pixels_pre_budget",
        "active_foreground_pixels",
        "active_background_pixels",
        "foreground_budget_ratio",
        "background_foreground_ratio",
        "active_fg_seed_precision",
        "active_fg_seed_recall",
        "active_fg_support_precision",
        "active_fg_support_recall",
        "active_fg_candidate_ratio",
        "active_fg_fg_candidate_ratio",
        "active_fg_near_seed_ratio",
        "active_fg_pre_budget_seed_precision",
        "active_fg_pre_budget_candidate_ratio",
        "teacher_foreground_ratio",
        "return_reason_0_ratio",
        "return_reason_1_ratio",
        "return_reason_2_ratio",
        "return_reason_3_ratio",
    ]
    saved_matrices = {}
    for key in matrix_keys:
        matrix = _matrix_from_records(df, key)
        if matrix is None:
            continue
        matrix_path = os.path.join(output_dir, "{}.csv".format(key))
        matrix.to_csv(matrix_path)
        saved_matrices[key] = matrix_path

    top_by_lambda = _summarize_top_teachers(df, "analysis_lambda_effective")
    top_by_release = _summarize_top_teachers(df, "release_factor")
    top_external_by_lambda = _summarize_top_teachers(df, "analysis_lambda_effective", exclude_self=True)
    summary = {
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "img_class": args.img_class,
        "analysis_iter": int(args.analysis_iter),
        "num_pairs": int(len(long_records)),
        "long_csv": long_csv,
        "saved_matrices": saved_matrices,
        "top_by_lambda_effective": top_by_lambda,
        "top_by_release_factor": top_by_release,
        "top_external_by_lambda_effective": top_external_by_lambda,
        "clients": [
            {
                "cid": cid,
                "client": client_name,
                "sup_type": sup_type,
                "checkpoint": checkpoint_paths[cid],
            }
            for cid, (client_name, sup_type) in enumerate(zip(args.client_list, args.sup_type_list))
        ],
    }
    summary_path = os.path.join(output_dir, "teacher_pool_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Teacher-pool feasibility analysis finished.")
    print("long_csv={}".format(long_csv))
    print("summary_json={}".format(summary_path))
    print("analysis_iter={}".format(args.analysis_iter))
    print("num_pairs={}".format(len(long_records)))
    print("")
    for target_name, teachers in top_by_lambda.items():
        print("Target {} top teachers by analysis_lambda_effective:".format(target_name))
        for item in teachers[:3]:
            print(
                "  teacher={} cid={} sup={} lambda_eff={:.6f} rel={:.6f} release={:.6f} ret0={:.4f}".format(
                    item["teacher_name"],
                    item["teacher_cid"],
                    item["teacher_sup_type"],
                    item["analysis_lambda_effective"],
                    item["teacher_reliability"],
                    item["release_factor"],
                    item["return_reason_0_ratio"],
                )
            )
    print("")
    for target_name, teachers in top_external_by_lambda.items():
        print("Target {} top external teachers by analysis_lambda_effective:".format(target_name))
        for item in teachers[:3]:
            print(
                "  teacher={} cid={} sup={} lambda_eff={:.6f} rel={:.6f} release={:.6f} ret0={:.4f}".format(
                    item["teacher_name"],
                    item["teacher_cid"],
                    item["teacher_sup_type"],
                    item["analysis_lambda_effective"],
                    item["teacher_reliability"],
                    item["release_factor"],
                    item["return_reason_0_ratio"],
                )
            )


if __name__ == "__main__":
    main()
