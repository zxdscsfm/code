import argparse
import copy
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analyze_rgftd_teacher_pool_feasibility import (
    _default_client_setup,
    _extract_student_logits,
    _load_client_model,
    _make_model_args,
    _prepare_batch,
)
from dataloaders.dataset import BaseDataSets
from rgftd_reliability_distillation import (
    _build_refined_teacher_target,
    _classwise_teacher_validation,
    _dilate_mask,
    _masked_mean,
    _masked_ratio,
    _normalize_gate,
    _normalized_entropy,
    _seed_or_fallback,
    _topk_foreground_anchor,
    get_rgftd_lambda,
)
from weak_annotation_reliability import build_wann_maps


PROSTATE_CLIENTS = ["client1", "client2", "client3", "client4", "client5", "client6"]
PROSTATE_SUP_TYPES = ["block", "keypoint", "scribble", "keypoint", "scribble", "box"]


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--analysis_iter", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_cases_per_pair", type=int, default=0,
                        help="0 means scan all target train cases.")
    parser.add_argument("--pair_specs", nargs="*", default=[
        "client2->client3:success",
        "client5->client6:rescue",
        "client1->client5:silence",
    ])

    parser.add_argument("--img_class", type=str, default="prostate")
    parser.add_argument("--model", type=str, default="unet_univ5")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_chns", type=int, default=1)
    parser.add_argument("--min_num_clients", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--prompt", type=str, default="universal")
    parser.add_argument("--attention", type=str, default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="FedUniV2.1")
    parser.add_argument("--wann_enabled", type=int, default=1)
    parser.add_argument("--rgftd_enabled", type=int, default=1)
    parser.add_argument("--amp", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--patch_size", nargs="+", type=int, default=[384, 384])

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
    return parser


def _primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def _parse_pair_specs(pair_specs):
    parsed = []
    for spec in pair_specs:
        pair, mode = spec.split(":")
        teacher_name, target_name = pair.split("->")
        parsed.append((teacher_name.strip(), target_name.strip(), mode.strip()))
    return parsed


def _to_cpu_numpy(t):
    return t.detach().cpu().numpy()


def _ensure_batched_inputs(image, label):
    """Normalize dataset samples to network-ready batch tensors."""
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.dim() == 3:
        image = image.unsqueeze(0)

    if label.dim() == 2:
        label = label.unsqueeze(0)

    return image, label


def _norm_image(image_2d):
    img = image_2d.astype(np.float32)
    mn, mx = np.percentile(img, 1), np.percentile(img, 99)
    if mx <= mn:
        mx = mn + 1.0
    img = np.clip((img - mn) / (mx - mn), 0.0, 1.0)
    return img


def _overlay_mask(image, masks_with_colors, alpha=0.55):
    base = np.stack([image, image, image], axis=-1)
    out = base.copy()
    for mask, color in masks_with_colors:
        if mask is None:
            continue
        mask = mask.astype(bool)
        if not mask.any():
            continue
        color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color.reshape(3)
    return np.clip(out, 0.0, 1.0)


def _build_debug_artifacts(student_logits, teacher_logits, label, wann_maps, args, iter_num, image):
    device = student_logits.device
    num_classes = student_logits.shape[1]
    lambda_rgftd = float(get_rgftd_lambda(iter_num, args))
    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
    student_prob = torch.softmax(student_logits.detach(), dim=1)
    teacher_pred = teacher_prob.argmax(dim=1)
    student_pred = student_prob.argmax(dim=1)
    teacher_conf = teacher_prob.max(dim=1)[0]
    student_conf = student_prob.max(dim=1)[0]
    student_entropy = _normalized_entropy(student_prob)

    label = label.long()
    core_mask = wann_maps.core_mask
    support_mask = wann_maps.support_mask
    seed_support_mask = wann_maps.seed_support_mask
    fg_support = support_mask & (label > 0)
    bg_support = support_mask & (label == 0)
    seed_fg_support = seed_support_mask & (label > 0)
    seed_bg_support = seed_support_mask & (label == 0)

    low_r_thresh = float(getattr(args, "rgftd_low_r_thresh", 0.25))
    teacher_fg_prob_thresh = float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))
    teacher_fg_topk_ratio = float(getattr(args, "rgftd_teacher_fg_topk_ratio", 0.002))
    teacher_fg_topk_min_pixels = int(getattr(args, "rgftd_teacher_fg_topk_min_pixels", 8))
    teacher_fg_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    min_fg_pixels = int(getattr(args, "rgftd_min_foreground_pixels", 8))
    min_fg_ratio = float(getattr(args, "rgftd_min_foreground_ratio", 0.05))
    use_soft_band = int(getattr(args, "rgftd_use_soft_band", 0)) == 1
    active_fg_topk_ratio = float(getattr(args, "rgftd_active_fg_topk_ratio", teacher_fg_topk_ratio))
    active_fg_topk_min_pixels = int(getattr(args, "rgftd_active_fg_topk_min_pixels", min_fg_pixels))
    active_fg_topk_max_pixels = int(getattr(args, "rgftd_active_fg_topk_max_pixels", 4096))
    spatial_support_enabled = int(getattr(args, "rgftd_spatial_support_enabled", 1)) == 1
    spatial_support_radius = int(getattr(args, "rgftd_spatial_support_radius", teacher_fg_radius))
    spatial_candidate_weight = float(getattr(args, "rgftd_spatial_candidate_weight", 1.0))
    spatial_near_seed_weight = float(getattr(args, "rgftd_spatial_near_seed_weight", 0.75))
    spatial_far_weight = float(getattr(args, "rgftd_spatial_far_weight", 0.15))
    max_bg_fg_ratio = float(getattr(args, "rgftd_max_bg_fg_ratio", 1.0))
    allow_bg_without_fg = int(getattr(args, "rgftd_allow_bg_without_fg", 0)) == 1
    teacher_student_fg_margin = float(getattr(args, "rgftd_teacher_student_fg_margin", 0.05))
    teacher_bg_conf_thresh = float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98))
    bg_max_fg_prob = float(getattr(args, "rgftd_bg_max_fg_prob", 0.15))
    teacher_support_prob_floor = float(getattr(args, "rgftd_teacher_support_prob_floor", teacher_fg_prob_thresh))
    skip_background_only = int(getattr(args, "rgftd_skip_background_only", 1)) == 1
    validation_enabled = int(getattr(args, "rgftd_teacher_validation_enabled", 0)) == 1

    candidate_mask = getattr(wann_maps, "candidate_mask", None)
    if candidate_mask is None:
        candidate_mask = (~wann_maps.core_mask) & (wann_maps.reliability >= low_r_thresh)
    candidate_mask = candidate_mask & (~wann_maps.core_mask)

    non_core_region = wann_maps.ignore_mask | wann_maps.soft_band
    if use_soft_band:
        non_core_region = non_core_region | candidate_mask
    region = non_core_region & (~wann_maps.core_mask)

    teacher_fg = teacher_pred > 0
    teacher_bg = ~teacher_fg
    teacher_fg_prob = teacher_prob[:, 1:].max(dim=1)[0]
    student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
    teacher_bg_prob = teacher_prob[:, 0]
    teacher_fg_margin = teacher_fg_prob - teacher_bg_prob
    teacher_fg_conf = teacher_fg & (teacher_conf >= float(getattr(args, "rgftd_teacher_conf_thresh", 0.90)))
    teacher_fg_prob_anchor = (teacher_fg_prob >= teacher_fg_prob_thresh) & region
    teacher_fg_seed = (teacher_fg_conf | teacher_fg_prob_anchor) & region
    teacher_fg_topk_valid = _seed_or_fallback(teacher_fg_seed, region)
    teacher_fg_anchor = _topk_foreground_anchor(
        teacher_fg_prob,
        teacher_fg_topk_valid,
        topk_ratio=teacher_fg_topk_ratio,
        min_pixels=teacher_fg_topk_min_pixels,
    )
    teacher_fg_ready = teacher_fg_anchor | teacher_fg_prob_anchor
    teacher_bg_conf = teacher_bg & (teacher_conf >= teacher_bg_conf_thresh)
    teacher_bg_safe = teacher_fg_prob <= bg_max_fg_prob
    teacher_fg_context = _dilate_mask(teacher_fg_ready, teacher_fg_radius)
    teacher_bg_context = teacher_fg_context & teacher_bg_conf
    region = region & (teacher_fg_context | teacher_bg_context)

    core_pixels = core_mask.float().sum().detach()
    core_valid = bool(float(core_pixels.cpu().item()) > 0.0)
    if core_valid:
        teacher_core_agreement = _masked_ratio((teacher_pred == label) & core_mask, core_mask).detach()
        teacher_core_conflict = (1.0 - teacher_core_agreement).detach()
    else:
        teacher_core_agreement = teacher_conf.mean().detach() * 0.0
        teacher_core_conflict = teacher_conf.mean().detach() * 0.0

    teacher_seed_support_fg_prob_mean = _masked_mean(teacher_fg_prob, seed_fg_support).detach()
    teacher_seed_support_fg_margin_mean = _masked_mean(teacher_fg_margin, seed_fg_support).detach()

    class_weight_map = torch.ones_like(teacher_conf)
    class_reliability_mean = teacher_conf.mean() * 0.0
    class_reliability_floor = 0.0
    if validation_enabled and teacher_prob.shape[1] > 1:
        class_weight_map, class_reliability_mean, class_reliability_floor, _ = _classwise_teacher_validation(
            teacher_pred=teacher_pred,
            teacher_conf=teacher_conf,
            label=label,
            support_mask=seed_fg_support,
            core_mask=core_mask & (label > 0),
            num_classes=teacher_prob.shape[1],
            args=args,
        )

    if validation_enabled:
        if core_valid:
            core_gate = _normalize_gate(
                teacher_core_agreement,
                float(getattr(args, "rgftd_teacher_core_agree_floor", 0.80)),
            )
        else:
            core_gate = torch.tensor(1.0, device=device)
        support_gate = _normalize_gate(
            teacher_seed_support_fg_prob_mean,
            teacher_support_prob_floor,
        )
        class_gate = _normalize_gate(class_reliability_mean, class_reliability_floor)
        margin_floor = float(getattr(args, "rgftd_teacher_release_margin_floor", 0.05))
        conf_gate = _normalize_gate(
            teacher_seed_support_fg_margin_mean,
            margin_floor,
        )
        score_core_weight = float(getattr(args, "rgftd_teacher_score_core_weight", 0.35))
        score_support_weight = float(getattr(args, "rgftd_teacher_score_support_weight", 0.30))
        score_class_weight = float(getattr(args, "rgftd_teacher_score_class_weight", 0.25))
        score_conf_weight = float(getattr(args, "rgftd_teacher_score_conf_weight", 0.10))
        teacher_reliability = (
            score_core_weight * core_gate
            + score_support_weight * support_gate
            + score_class_weight * class_gate
            + score_conf_weight * conf_gate
        ) / max(score_core_weight + score_support_weight + score_class_weight + score_conf_weight, 1e-6)
        release_prob_gate = _normalize_gate(
            teacher_seed_support_fg_prob_mean,
            float(getattr(args, "rgftd_teacher_release_prob_floor", teacher_fg_prob_thresh)),
        )
        release_conf_gate = _normalize_gate(
            teacher_seed_support_fg_margin_mean,
            margin_floor,
        )
        release_raw = release_prob_gate * release_conf_gate
        if float(seed_fg_support.float().sum().detach().cpu().item()) > 0.0:
            release_min = float(getattr(args, "rgftd_teacher_release_min", 0.03))
            max_lambda = float(max(getattr(args, "rgftd_lambda", 0.1), 1e-6))
            release_schedule = float(max(min(lambda_rgftd / max_lambda, 1.0), 0.0))
            release_quality = release_prob_gate * release_conf_gate
            release_min_effective = release_min * release_quality * release_schedule
            release_factor = release_raw + (1.0 - release_raw) * release_min_effective
        else:
            release_factor = teacher_conf.mean() * 0.0
        teacher_validation_veto = bool(
            float(teacher_core_conflict.cpu().item()) >
            float(getattr(args, "rgftd_teacher_max_core_conflict", 0.20))
        )
    else:
        teacher_reliability = torch.tensor(1.0, device=device)
        release_factor = torch.tensor(1.0, device=device)
        teacher_validation_veto = False

    student_uncertain = (
        (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
        | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
    )
    foreground_correction = teacher_fg_ready & ((teacher_fg_prob - student_fg_prob) >= teacher_student_fg_margin)
    seed_fg_spatial_context = _dilate_mask(seed_fg_support, spatial_support_radius)
    fg_support_context = _dilate_mask(fg_support, spatial_support_radius)
    foreground_candidate_mask = candidate_mask & fg_support_context
    spatial_support_gate = (foreground_candidate_mask | seed_fg_spatial_context) & region
    active_fg_pre_spatial = region & teacher_fg_ready & (student_uncertain | foreground_correction)
    if spatial_support_enabled:
        spatial_weight = region.float() * spatial_far_weight
        spatial_weight = torch.maximum(
            spatial_weight,
            seed_fg_spatial_context.float() * spatial_near_seed_weight,
        )
        spatial_weight = torch.maximum(
            spatial_weight,
            foreground_candidate_mask.float() * spatial_candidate_weight,
        )
        spatial_weight = spatial_weight * region.float()
    else:
        spatial_weight = region.float()

    active_fg = active_fg_pre_spatial
    fg_budget_score = (
        teacher_fg_prob
        * student_entropy
        * (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
        * spatial_weight
    )
    active_fg = _topk_foreground_anchor(
        fg_budget_score,
        active_fg,
        topk_ratio=active_fg_topk_ratio,
        min_pixels=active_fg_topk_min_pixels,
        max_pixels=active_fg_topk_max_pixels,
        preserve_mask=foreground_correction & active_fg,
    )

    active_bg = region & teacher_bg_conf & teacher_bg_safe & (~teacher_fg_ready) & student_uncertain
    bg_budget_score = teacher_bg_prob * student_entropy * (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
    flat_active_fg = active_fg.flatten(1)
    flat_active_bg = active_bg.flatten(1)
    max_bg_pixels = []
    for batch_idx in range(active_bg.shape[0]):
        fg_count = int(flat_active_fg[batch_idx].sum().detach().cpu().item())
        bg_count = int(flat_active_bg[batch_idx].sum().detach().cpu().item())
        if fg_count > 0:
            max_bg_pixels.append(int(np.ceil(float(fg_count) * max(max_bg_fg_ratio, 0.0))))
        elif allow_bg_without_fg:
            max_bg_pixels.append(int(max(min_fg_pixels, bg_count)))
        else:
            max_bg_pixels.append(0)
    if max(max_bg_pixels) <= 0:
        active_bg = torch.zeros_like(active_bg, dtype=torch.bool)
    else:
        active_bg = _topk_foreground_anchor(
            bg_budget_score,
            active_bg,
            topk_ratio=1.0,
            min_pixels=0,
            max_pixels=max_bg_pixels,
        )

    active_fg_pixels = float(active_fg.float().sum().detach().cpu().item())
    active_bg_pixels = float(active_bg.float().sum().detach().cpu().item())
    active_total = max(active_fg_pixels + active_bg_pixels, 1.0)
    active_fg_ratio = active_fg_pixels / active_total
    foreground_veto = bool(
        skip_background_only and (
            active_fg_pixels < float(min_fg_pixels)
            or active_fg_ratio < float(min_fg_ratio)
        )
    )

    q = teacher_prob.detach()
    refine_profile = {
        "refine_enabled": torch.tensor(0.0, device=device),
        "refine_silent": torch.tensor(0.0, device=device),
        "refine_q_candidate_ratio": torch.tensor(0.0, device=device),
        "refine_q_near_seed_ratio": torch.tensor(0.0, device=device),
        "refine_unsupported_fg_ratio": torch.tensor(0.0, device=device),
        "refine_teacher_q_kl": torch.tensor(0.0, device=device),
        "refine_q_fg_delta": torch.tensor(0.0, device=device),
        "refine_q_entropy_mean": torch.tensor(0.0, device=device),
        "refine_roi_ratio": torch.tensor(0.0, device=device),
    }
    q_silent = False
    if (not teacher_validation_veto) and (not foreground_veto) and active_fg_pixels > 0:
        q, q_silent, refine_profile = _build_refined_teacher_target(
            teacher_prob=teacher_prob,
            image=image,
            label=label,
            wann_maps=wann_maps,
            args=args,
            active_fg_mask=active_fg,
            active_bg_mask=active_bg,
            region=region,
            candidate_mask=candidate_mask,
            foreground_candidate_mask=foreground_candidate_mask,
            seed_fg_support=seed_fg_support,
            seed_bg_support=seed_bg_support,
            core_mask=core_mask,
            seed_fg_context=_dilate_mask(seed_fg_support, teacher_fg_radius),
            teacher_fg_prob=teacher_fg_prob,
        )

    q_fg = q[:, 1]
    active_fg_candidate_ratio = float(_masked_ratio(active_fg & candidate_mask, active_fg).detach().cpu().item()) if active_fg.any() else 0.0
    active_fg_near_seed_ratio = float(_masked_ratio(active_fg & _dilate_mask(seed_fg_support, teacher_fg_radius), active_fg).detach().cpu().item()) if active_fg.any() else 0.0

    lambda_after_safety = lambda_rgftd * float(teacher_reliability.detach().cpu().item()) * float(release_factor.detach().cpu().item())
    lambda_effective = min(lambda_after_safety, float(getattr(args, "rgftd_lambda_eff_cap", 0.02)))
    if teacher_validation_veto or foreground_veto or active_fg_pixels <= 0 or q_silent:
        lambda_effective = 0.0

    profile = {
        "analysis_lambda_effective": lambda_effective,
        "teacher_reliability": float(teacher_reliability.detach().cpu().item()),
        "release_factor": float(release_factor.detach().cpu().item()),
        "teacher_core_conflict": float(teacher_core_conflict.detach().cpu().item()),
        "teacher_seed_support_fg_prob_mean": float(teacher_seed_support_fg_prob_mean.detach().cpu().item()),
        "teacher_seed_support_fg_margin_mean": float(teacher_seed_support_fg_margin_mean.detach().cpu().item()),
        "active_fg_candidate_ratio": active_fg_candidate_ratio,
        "active_fg_near_seed_ratio": active_fg_near_seed_ratio,
        "refine_q_candidate_ratio": float(refine_profile["refine_q_candidate_ratio"].detach().cpu().item()),
        "refine_q_near_seed_ratio": float(refine_profile["refine_q_near_seed_ratio"].detach().cpu().item()),
        "refine_unsupported_fg_ratio": float(refine_profile["refine_unsupported_fg_ratio"].detach().cpu().item()),
        "refine_teacher_q_kl": float(refine_profile["refine_teacher_q_kl"].detach().cpu().item()),
        "refine_q_fg_delta": float(refine_profile["refine_q_fg_delta"].detach().cpu().item()),
        "refine_q_entropy_mean": float(refine_profile["refine_q_entropy_mean"].detach().cpu().item()),
        "refine_enabled": float(refine_profile["refine_enabled"].detach().cpu().item()),
        "refine_silent": float(refine_profile["refine_silent"].detach().cpu().item()),
        "refine_roi_ratio": float(refine_profile["refine_roi_ratio"].detach().cpu().item()),
        "foreground_veto": float(foreground_veto),
        "teacher_validation_veto": float(teacher_validation_veto),
        "active_fg_pixels": active_fg_pixels,
    }

    return {
        "teacher_fg_prob": teacher_fg_prob.detach(),
        "student_fg_prob": student_fg_prob.detach(),
        "student_pred": student_pred.detach(),
        "teacher_pred": teacher_pred.detach(),
        "reliability": wann_maps.reliability.detach(),
        "core_mask": wann_maps.core_mask.detach(),
        "soft_band": wann_maps.soft_band.detach(),
        "ignore_mask": wann_maps.ignore_mask.detach(),
        "candidate_mask": candidate_mask.detach(),
        "seed_fg_support": seed_fg_support.detach(),
        "seed_bg_support": seed_bg_support.detach(),
        "active_fg_mask": active_fg.detach(),
        "active_bg_mask": active_bg.detach(),
        "refined_q_fg": q_fg.detach(),
        "profile": profile,
    }


def _score_artifacts(profile, mode):
    if mode == "success":
        return (
            profile["analysis_lambda_effective"]
            * max(profile["refine_q_candidate_ratio"], 1e-6)
            * max(profile["refine_q_near_seed_ratio"], 1e-6)
        )
    if mode == "rescue":
        return profile["analysis_lambda_effective"] * (
            profile["refine_q_near_seed_ratio"] - profile["active_fg_near_seed_ratio"]
        )
    if mode == "silence":
        if profile["analysis_lambda_effective"] > 0:
            return -1e9
        return max(profile["teacher_seed_support_fg_prob_mean"], 0.0) + max(-profile["teacher_seed_support_fg_margin_mean"], 0.0)
    return profile["analysis_lambda_effective"]


def _render_case_panel(image, label, case_name, teacher_name, target_name, mode, artifacts, out_path):
    img = _norm_image(image)
    label = label.astype(np.int64)
    ignore_id = 2
    valid = label != ignore_id
    weak_panel = _overlay_mask(
        img,
        [
            ((artifacts["seed_bg_support"] > 0.5).astype(bool), (0.2, 0.45, 1.0)),
            ((artifacts["seed_fg_support"] > 0.5).astype(bool), (1.0, 0.15, 0.15)),
        ],
        alpha=0.75,
    )
    wann_panel = _overlay_mask(
        img,
        [
            ((artifacts["ignore_mask"] > 0.5).astype(bool), (0.45, 0.10, 0.60)),
            ((artifacts["soft_band"] > 0.5).astype(bool), (1.0, 0.85, 0.10)),
            ((artifacts["core_mask"] > 0.5).astype(bool), (0.15, 0.75, 0.25)),
        ],
        alpha=0.45,
    )
    active_panel = _overlay_mask(
        img,
        [
            ((artifacts["candidate_mask"] > 0.5).astype(bool), (0.15, 0.95, 0.95)),
            ((artifacts["active_fg_mask"] > 0.5).astype(bool), (1.0, 0.12, 0.12)),
        ],
        alpha=0.55,
    )
    pred_panel = _overlay_mask(
        img,
        [((artifacts["student_pred"] > 0.5).astype(bool), (1.0, 0.20, 0.20))],
        alpha=0.55,
    )
    teacher_pred_panel = _overlay_mask(
        img,
        [((artifacts["teacher_pred"] > 0.5).astype(bool), (1.0, 0.45, 0.10))],
        alpha=0.45,
    )

    profile = artifacts["profile"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=200)
    axes = axes.ravel()

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image")

    axes[1].imshow(weak_panel)
    axes[1].set_title("Weak Label / Seed")

    axes[2].imshow(wann_panel)
    axes[2].set_title("WANN Partition")

    im3 = axes[3].imshow(artifacts["teacher_fg_prob"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Teacher FG Prob")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].imshow(teacher_pred_panel)
    axes[4].set_title("Teacher Prediction")

    axes[5].imshow(active_panel)
    axes[5].set_title("Raw Active FG")

    im6 = axes[6].imshow(artifacts["refined_q_fg"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[6].set_title("Refined q FG")
    plt.colorbar(im6, ax=axes[6], fraction=0.046, pad=0.04)

    axes[7].imshow(pred_panel)
    axes[7].set_title("Student Prediction")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    metrics_text = (
        f"teacher={teacher_name} -> target={target_name} | mode={mode} | case={case_name}\n"
        f"lambda_eff={profile['analysis_lambda_effective']:.4f} | core_cf={profile['teacher_core_conflict']:.4f} | "
        f"seed_fgp={profile['teacher_seed_support_fg_prob_mean']:.4f} | seed_fgm={profile['teacher_seed_support_fg_margin_mean']:.4f}\n"
        f"raw_cand={profile['active_fg_candidate_ratio']:.4f} | raw_near={profile['active_fg_near_seed_ratio']:.4f} | "
        f"q_cand={profile['refine_q_candidate_ratio']:.4f} | q_near={profile['refine_q_near_seed_ratio']:.4f}\n"
        f"q_unsup={profile['refine_unsupported_fg_ratio']:.4f} | tq_kl={profile['refine_teacher_q_kl']:.4f} | "
        f"q_fg_delta={profile['refine_q_fg_delta']:.2f} | q_ent={profile['refine_q_entropy_mean']:.4f}"
    )
    fig.suptitle(metrics_text, fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _render_lean_case_panel(image, label, case_name, teacher_name, target_name, mode, artifacts, out_path):
    img = _norm_image(image)
    weak_panel = _overlay_mask(
        img,
        [
            ((artifacts["seed_bg_support"] > 0.5).astype(bool), (0.2, 0.45, 1.0)),
            ((artifacts["seed_fg_support"] > 0.5).astype(bool), (1.0, 0.12, 0.12)),
        ],
        alpha=0.72,
    )
    support_panel = _overlay_mask(
        img,
        [
            ((artifacts["candidate_mask"] > 0.5).astype(bool), (0.10, 0.78, 0.82)),
            ((artifacts["core_mask"] > 0.5).astype(bool), (0.18, 0.72, 0.25)),
            ((artifacts["ignore_mask"] > 0.5).astype(bool), (0.45, 0.16, 0.62)),
        ],
        alpha=0.42,
    )
    raw_panel = _overlay_mask(
        img,
        [
            ((artifacts["candidate_mask"] > 0.5).astype(bool), (0.10, 0.78, 0.82)),
            ((artifacts["active_fg_mask"] > 0.5).astype(bool), (1.0, 0.18, 0.10)),
        ],
        alpha=0.58,
    )
    refined_panel = _overlay_mask(
        img,
        [
            ((artifacts["candidate_mask"] > 0.5).astype(bool), (0.10, 0.78, 0.82)),
            ((artifacts["refined_q_fg"] > 0.35).astype(bool), (1.0, 0.80, 0.12)),
        ],
        alpha=0.58,
    )
    student_panel = _overlay_mask(
        img,
        [((artifacts["student_pred"] > 0.5).astype(bool), (1.0, 0.20, 0.20))],
        alpha=0.55,
    )

    panels = [
        ("Image", img, "gray"),
        ("Weak seed", weak_panel, None),
        ("Target support", support_panel, None),
        ("Raw release", raw_panel, None),
        ("Refined q", refined_panel, None),
        ("Student", student_panel, None),
    ]

    profile = artifacts["profile"]
    fig, axes = plt.subplots(1, len(panels), figsize=(14.2, 2.9), dpi=240)
    for ax, (title, panel, cmap) in zip(axes, panels):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title, fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#333333")

    metrics_text = (
        f"{teacher_name}->{target_name} | {mode} | "
        f"lambda={profile['analysis_lambda_effective']:.3f}, "
        f"margin={profile['teacher_seed_support_fg_margin_mean']:.2f}, "
        f"core={profile['teacher_core_conflict']:.2f}, "
        f"raw_near={profile['active_fg_near_seed_ratio']:.2f}, "
        f"q_near={profile['refine_q_near_seed_ratio']:.2f}"
    )
    fig.suptitle(metrics_text, fontsize=9, y=1.02)
    fig.tight_layout(pad=0.35)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(args.snapshot_path, "case_visualizations_v35")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    client_list, sup_type_list, _, _, _, _ = _default_client_setup("prostate")
    client_to_idx = {name: idx for idx, name in enumerate(client_list)}
    client_to_sup = {name: sup for name, sup in zip(client_list, sup_type_list)}

    parsed_pairs = _parse_pair_specs(args.pair_specs)
    selection_rows = []

    model_cache = {}
    def get_model(client_name):
        if client_name in model_cache:
            return model_cache[client_name]
        cid = client_to_idx[client_name]
        sup_type = client_to_sup[client_name]
        checkpoint_path = os.path.join(args.snapshot_path, f"client_{cid}_async_{args.model}_best_model.pth")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(args.snapshot_path, f"client_{cid}_{args.model}_best_model.pth")
        model = _load_client_model(args, cid, sup_type, checkpoint_path, device)
        model_cache[client_name] = model
        return model

    with torch.no_grad():
        for teacher_name, target_name, mode in parsed_pairs:
            target_dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=target_name,
                sup_type=client_to_sup[target_name],
                img_class="prostate",
            )
            teacher_model = get_model(teacher_name)
            target_model = get_model(target_name)
            best = None
            max_cases = len(target_dataset) if args.max_cases_per_pair <= 0 else min(len(target_dataset), args.max_cases_per_pair)
            for idx in range(max_cases):
                sample = copy.deepcopy(target_dataset[idx])
                case_name = target_dataset.sample_list[idx]
                image, label = _prepare_batch(sample, "prostate", device)
                image, label = _ensure_batched_inputs(image, label)
                student_out = target_model(image)
                student_logits, _ = _extract_student_logits(args.model, student_out)
                teacher_logits = _primary_logits(teacher_model(image))
                wann_maps = build_wann_maps(
                    image=image,
                    label=label,
                    logits=student_logits,
                    aux_logits=None,
                    sup_type=client_to_sup[target_name],
                    img_class="prostate",
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )
                artifacts = _build_debug_artifacts(student_logits, teacher_logits, label, wann_maps, args, args.analysis_iter, image)
                score = _score_artifacts(artifacts["profile"], mode)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "case_name": case_name,
                        "sample_idx": idx,
                        "image": _to_cpu_numpy(image[0, 0]),
                        "label": _to_cpu_numpy(label[0]),
                        "artifacts": {
                            "teacher_fg_prob": _to_cpu_numpy(artifacts["teacher_fg_prob"][0]),
                            "student_pred": _to_cpu_numpy(artifacts["student_pred"][0]),
                            "teacher_pred": _to_cpu_numpy(artifacts["teacher_pred"][0]),
                            "reliability": _to_cpu_numpy(artifacts["reliability"][0]),
                            "core_mask": _to_cpu_numpy(artifacts["core_mask"][0]),
                            "soft_band": _to_cpu_numpy(artifacts["soft_band"][0]),
                            "ignore_mask": _to_cpu_numpy(artifacts["ignore_mask"][0]),
                            "candidate_mask": _to_cpu_numpy(artifacts["candidate_mask"][0]),
                            "seed_fg_support": _to_cpu_numpy(artifacts["seed_fg_support"][0]),
                            "seed_bg_support": _to_cpu_numpy(artifacts["seed_bg_support"][0]),
                            "active_fg_mask": _to_cpu_numpy(artifacts["active_fg_mask"][0]),
                            "active_bg_mask": _to_cpu_numpy(artifacts["active_bg_mask"][0]),
                            "refined_q_fg": _to_cpu_numpy(artifacts["refined_q_fg"][0]),
                            "profile": artifacts["profile"],
                        },
                    }
            if best is None:
                continue

            out_name = f"{teacher_name}_to_{target_name}_{mode}.png"
            out_path = os.path.join(args.output_dir, out_name)
            lean_out_path = os.path.join(args.output_dir, f"lean_{out_name}")
            _render_case_panel(
                image=best["image"],
                label=best["label"],
                case_name=best["case_name"],
                teacher_name=teacher_name,
                target_name=target_name,
                mode=mode,
                artifacts=best["artifacts"],
                out_path=out_path,
            )
            _render_lean_case_panel(
                image=best["image"],
                label=best["label"],
                case_name=best["case_name"],
                teacher_name=teacher_name,
                target_name=target_name,
                mode=mode,
                artifacts=best["artifacts"],
                out_path=lean_out_path,
            )
            row = {
                "teacher_name": teacher_name,
                "target_name": target_name,
                "mode": mode,
                "case_name": best["case_name"],
                "sample_idx": best["sample_idx"],
                "score": best["score"],
                "figure": out_path,
                "lean_figure": lean_out_path,
            }
            row.update(best["artifacts"]["profile"])
            selection_rows.append(row)

    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(os.path.join(args.output_dir, "selected_cases.csv"), index=False)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
