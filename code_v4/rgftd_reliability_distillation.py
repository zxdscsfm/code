# -*- coding:utf-8 -*-
import math

import torch
import torch.nn.functional as F


RGFTD_PROFILE_KEYS = [
    "loss",
    "lambda",
    "lambda_pre_safety",
    "lambda_after_safety",
    "lambda_effective",
    "core_safety_factor",
    "release_factor",
    "release_raw",
    "release_quality",
    "release_min_effective",
    "release_prob_gate",
    "release_conf_gate",
    "release_class_gate",
    "release_score_mean",
    "release_score_top",
    "rdsi_enabled",
    "rdsi_teacher_compete_count",
    "rdsi_candidate_teacher_count",
    "rdsi_multi_teacher_active",
    "rdsi_best_vs_second_gap",
    "rdsi_selected_score_mean",
    "rdsi_selected_score_top",
    "rdsi_score_benefit_mean",
    "rdsi_selected_teacher_reliable",
    "rdsi_selected_teacher_benefit",
    "rdsi_selected_teacher_gap",
    "rdsi_selected_student_risk",
    "rdsi_selected_teacher_mean",
    "rdsi_selected_teacher_switch_ratio",
    "rdsi_teacher_reliable_score",
    "rdsi_knowledge_gap_score",
    "rdsi_risk_region_ratio",
    "rdsi_hard_core_ratio",
    "rdsi_soft_core_ratio",
    "rdsi_fg_deficient_ratio",
    "rdsi_fg_excessive_ratio",
    "rdsi_candidate_ratio",
    "rdsi_accept_ratio",
    "rdsi_reject_ratio",
    "rdsi_reject_by_core",
    "rdsi_reject_by_seed",
    "rdsi_reject_by_prior",
    "rdsi_reject_by_entropy",
    "rdsi_reject_by_fg_excess",
    "rdsi_benefit_mean",
    "rdsi_benefit_top",
    "rdsi_alpha_mean",
    "rdsi_alpha_top",
    "rdsi_raw_teacher_fg_delta",
    "rdsi_raw_teacher_conf_mean",
    "rdsi_raw_teacher_fg_ratio",
    "rdsi_q_fg_delta",
    "rdsi_target_conf_mean",
    "rdsi_target_entropy_mean",
    "rdsi_loss_raw",
    "rdsi_loss_weighted",
    "rdsi_teacher0_ratio",
    "rdsi_teacher1_ratio",
    "rdsi_teacher2_ratio",
    "rdsi_teacher3_ratio",
    "rdsi_teacher4_ratio",
    "rdsi_teacher5_ratio",
    "rdsi_teacher6_ratio",
    "rdsi_teacher7_ratio",
    "rdsi_teacher8_ratio",
    "rdsi_teacher9_ratio",
    "teacher_reliable_score",
    "student_risk_score",
    "knowledge_gap_score",
    "selected_gap_mean",
    "rejected_gap_mean",
    "candidate_ratio",
    "active_ratio",
    "region_ratio",
    "risk_region_ratio",
    "preserve_region_ratio",
    "teacher_accept_ratio",
    "teacher_reject_ratio",
    "reject_by_support",
    "reject_by_core_conflict",
    "reject_by_fg_ratio",
    "teacher_active_loss",
    "student_uncertain_ratio",
    "teacher_conf_mean",
    "student_conf_mean",
    "student_entropy_mean",
    "kl_mean",
    "weight_mean",
    "teacher_foreground_ratio",
    "teacher_reliability",
    "teacher_core_agreement",
    "teacher_core_conflict",
    "teacher_core_valid",
    "teacher_core_conf_mean",
    "teacher_support_agreement",
    "teacher_support_conflict",
    "teacher_support_conf_mean",
    "teacher_support_fg_recall",
    "teacher_support_bg_agreement",
    "teacher_seed_support_agreement",
    "teacher_seed_support_conflict",
    "teacher_seed_support_conf_mean",
    "teacher_seed_support_fg_recall",
    "teacher_seed_support_fg_prob_mean",
    "teacher_seed_support_fg_conf_mean",
    "teacher_seed_support_fg_margin_mean",
    "teacher_seed_support_bg_agreement",
    "seed_fg_support_pixels",
    "core_conflict_veto_ratio",
    "spatial_support_ratio",
    "foreground_candidate_ratio",
    "spatial_weight_mean",
    "spatial_weight_candidate_mean",
    "spatial_weight_near_seed_mean",
    "spatial_weight_far_mean",
    "spatial_loss_scale",
    "active_foreground_pixels_pre_spatial",
    "active_foreground_spatial_keep_ratio",
    "active_foreground_ratio",
    "active_background_ratio",
    "foreground_veto_ratio",
    "active_foreground_pixels_pre_budget",
    "active_background_pixels_pre_budget",
    "foreground_budget_ratio",
    "background_budget_ratio",
    "background_foreground_ratio",
    "background_balance_factor",
    "active_foreground_pixels_pre_return",
    "active_background_pixels_pre_return",
    "active_foreground_pixels",
    "active_background_pixels",
    "active_fg_seed_precision",
    "active_fg_seed_recall",
    "active_fg_support_precision",
    "active_fg_support_recall",
    "active_fg_candidate_ratio",
    "active_fg_fg_candidate_ratio",
    "active_fg_near_seed_ratio",
    "active_fg_pre_budget_seed_precision",
    "active_fg_pre_budget_candidate_ratio",
    "refine_enabled",
    "refine_silent",
    "refine_roi_ratio",
    "refine_affinity_mean",
    "refine_teacher_q_kl",
    "refine_q_entropy_mean",
    "refine_q_fg_mass",
    "refine_q_fg_ratio",
    "refine_q_fg_delta",
    "refine_q_seed_precision",
    "refine_q_seed_recall",
    "refine_q_candidate_ratio",
    "refine_q_near_seed_ratio",
    "refine_unsupported_fg_ratio",
    "refine_unsupported_fg_scale",
    "refine_q_core_conflict",
    "background_suppression_mean",
    "return_reason",
    "v3_pool_size",
    "v3_no_teacher",
    "v3_selected_teacher",
    "v3_selected_score",
    "v3_best_failed_teacher",
    "v3_best_failed_score",
    "v3_routing_score",
    "v3_audit_seed_fg_prob_mean",
    "v3_audit_seed_fg_margin_mean",
    "v3_audit_seed_fg_recall",
    "v3_audit_core_conflict",
    "v3_audit_teacher_reliability",
    "v3_audit_release_factor",
]


def zero_rgftd_profile(device):
    return {key: torch.tensor(0.0, device=device) for key in RGFTD_PROFILE_KEYS}


def _normalized_entropy(prob):
    num_classes = prob.shape[1]
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    normalizer = math.log(float(num_classes))
    return (entropy / max(normalizer, 1e-6)).clamp(0.0, 1.0)


def _dilate_mask(mask, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _topk_foreground_anchor(score, valid_mask, topk_ratio, min_pixels, max_pixels=None, preserve_mask=None):
    anchors = torch.zeros_like(valid_mask, dtype=torch.bool)
    flat_score = score.flatten(1)
    flat_valid = valid_mask.flatten(1)
    flat_preserve = preserve_mask.flatten(1) if preserve_mask is not None else None
    for batch_idx in range(score.shape[0]):
        valid_count = int(flat_valid[batch_idx].sum().detach().cpu().item())
        if valid_count <= 0:
            continue
        k = max(int(min_pixels), int(math.ceil(float(valid_count) * float(topk_ratio))))
        if max_pixels is not None:
            if torch.is_tensor(max_pixels):
                batch_max_pixels = int(max_pixels[batch_idx].detach().cpu().item())
            elif isinstance(max_pixels, (list, tuple)):
                batch_max_pixels = int(max_pixels[batch_idx])
            else:
                batch_max_pixels = int(max_pixels)
            if batch_max_pixels > 0:
                k = min(k, batch_max_pixels)
        k = min(k, valid_count)
        if flat_preserve is None:
            masked_score = flat_score[batch_idx].masked_fill(~flat_valid[batch_idx], -1.0)
            topk_index = torch.topk(masked_score, k=k, largest=True).indices
            anchors.flatten(1)[batch_idx, topk_index] = True
            continue

        preserve = flat_preserve[batch_idx] & flat_valid[batch_idx]
        preserve_count = int(preserve.sum().detach().cpu().item())
        if preserve_count >= k:
            preserve_score = flat_score[batch_idx].masked_fill(~preserve, -1.0)
            topk_index = torch.topk(preserve_score, k=k, largest=True).indices
            anchors.flatten(1)[batch_idx, topk_index] = True
            continue

        anchors.flatten(1)[batch_idx, preserve] = True
        remaining_k = k - preserve_count
        if remaining_k <= 0:
            continue
        residual_valid = flat_valid[batch_idx] & (~preserve)
        residual_score = flat_score[batch_idx].masked_fill(~residual_valid, -1.0)
        topk_index = torch.topk(residual_score, k=remaining_k, largest=True).indices
        anchors.flatten(1)[batch_idx, topk_index] = True
    return anchors & valid_mask


def _seed_or_fallback(seed_mask, fallback_mask):
    valid_mask = fallback_mask.clone()
    for batch_idx in range(seed_mask.shape[0]):
        if seed_mask[batch_idx].any():
            valid_mask[batch_idx] = seed_mask[batch_idx]
    return valid_mask & fallback_mask


def _masked_ratio(mask, denom_mask):
    denom = denom_mask.float().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return denom * 0.0
    return mask.float().sum() / denom


def _masked_mean(value, mask):
    denom = mask.float().sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return denom * 0.0
    return (value * mask.float()).sum() / denom


def _safe_target_masks(label, wann_maps, num_classes):
    raw_label = label.long()
    target_label = getattr(wann_maps, "target_label", raw_label).long()
    target_valid = (target_label >= 0) & (target_label < num_classes)
    valid_mask = getattr(wann_maps, "valid_mask", raw_label != num_classes) & target_valid
    raw_support_mask = getattr(wann_maps, "support_mask", valid_mask) & target_valid
    raw_seed_support_mask = getattr(wann_maps, "seed_support_mask", raw_support_mask) & target_valid
    target_support_mask = valid_mask & target_valid
    support_mask = raw_support_mask | target_support_mask
    seed_support_mask = raw_seed_support_mask | target_support_mask
    return target_label, target_valid, valid_mask, support_mask, seed_support_mask


def _normalize_gate(score, floor):
    floor = float(max(min(floor, 0.999999), 0.0))
    return ((score - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)


def _normalize_image_for_affinity(image):
    if image is None:
        return None
    if image.dim() == 3:
        image = image.unsqueeze(1)
    gray = image.detach().float().mean(dim=1, keepdim=True)
    flat = gray.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    return (gray - mean) / std


def _shift_with_valid(x, direction):
    b, c, h, w = x.shape
    shifted = torch.zeros_like(x)
    valid = torch.zeros(b, 1, h, w, dtype=torch.bool, device=x.device)
    if direction == "up":
        shifted[:, :, 1:, :] = x[:, :, :-1, :]
        valid[:, :, 1:, :] = True
    elif direction == "down":
        shifted[:, :, :-1, :] = x[:, :, 1:, :]
        valid[:, :, :-1, :] = True
    elif direction == "left":
        shifted[:, :, :, 1:] = x[:, :, :, :-1]
        valid[:, :, :, 1:] = True
    elif direction == "right":
        shifted[:, :, :, :-1] = x[:, :, :, 1:]
        valid[:, :, :, :-1] = True
    else:
        raise ValueError("Unsupported direction: {}".format(direction))
    return shifted, valid


def _local_affinity_refine(q, image, roi_mask, locked_mask, locked_q, args):
    """Diffuse soft labels inside target-side ROI while keeping seeds/core fixed."""
    gray = _normalize_image_for_affinity(image)
    if gray is None:
        return q, q[:, :1].sum() * 0.0

    refine_iters = int(getattr(args, "rgftd_refine_iters", 3))
    if refine_iters <= 0:
        return q, q[:, :1].sum() * 0.0

    sigma = float(getattr(args, "rgftd_refine_affinity_sigma", 0.75))
    sigma = max(sigma, 1e-6)
    mix = float(getattr(args, "rgftd_refine_affinity_mix", 0.35))
    mix = max(0.0, min(mix, 1.0))
    roi_f = roi_mask.float().unsqueeze(1)
    locked_f = locked_mask.float().unsqueeze(1)
    free_f = roi_f * (1.0 - locked_f)
    if float(free_f.sum().detach().cpu().item()) <= 0.0:
        return q, q[:, :1].sum() * 0.0

    current = q
    affinity_accum = q[:, :1].sum() * 0.0
    affinity_count = 0
    for _ in range(refine_iters):
        weighted_sum = current * 0.0
        weight_sum = current[:, :1] * 0.0
        for direction in ["up", "down", "left", "right"]:
            shifted_q, valid = _shift_with_valid(current, direction)
            shifted_gray, _ = _shift_with_valid(gray, direction)
            shifted_roi, _ = _shift_with_valid(roi_f, direction)
            affinity = torch.exp(-((gray - shifted_gray) / sigma) ** 2).clamp(0.0, 1.0)
            affinity = affinity * valid.float() * roi_f * shifted_roi
            weighted_sum = weighted_sum + shifted_q * affinity
            weight_sum = weight_sum + affinity
            affinity_accum = affinity_accum + affinity.sum()
            affinity_count += 1

        smoothed = weighted_sum / weight_sum.clamp_min(1e-6)
        updated = (1.0 - mix) * current + mix * smoothed
        updated = torch.where(free_f > 0.0, updated, current)
        current = torch.where(locked_f > 0.0, locked_q, updated)
        current = current.clamp_min(1e-6)
        current = current / current.sum(dim=1, keepdim=True).clamp_min(1e-6)

    denom = float(max(affinity_count, 1)) * roi_f.sum().clamp_min(1.0)
    affinity_mean = affinity_accum / denom
    return current.detach(), affinity_mean.detach()


def _one_hot_label(label, num_classes, like):
    safe_label = label.clamp(0, int(num_classes) - 1)
    one_hot = torch.zeros_like(like)
    one_hot.scatter_(1, safe_label.unsqueeze(1), 1.0)
    return one_hot


_RDSI_RUNTIME_ATTRS = [
    "_rdsi_active_mask",
    "_rdsi_residual_alpha_map",
    "_rdsi_benefit_score",
    "_rdsi_selected_score",
    "_rdsi_risk_region",
    "_rdsi_hard_core_mask",
    "_rdsi_soft_core_mask",
]


def _clear_rdsi_runtime_state(args):
    for attr in _RDSI_RUNTIME_ATTRS:
        if hasattr(args, attr):
            delattr(args, attr)


def _runtime_map(args, attr, ref, as_bool=False):
    value = getattr(args, attr, None)
    if value is None or not torch.is_tensor(value) or tuple(value.shape) != tuple(ref.shape):
        return None
    value = value.to(device=ref.device)
    if as_bool:
        return value > 0.5 if value.dtype != torch.bool else value
    return value.float()


def select_rdsi_teacher_logits(student_logits, teacher_logits_list, teacher_ids, label, wann_maps, args, iter_num):
    """Select a region-wise domain-specialist teacher from a cross-client pool."""
    device = student_logits.device
    _clear_rdsi_runtime_state(args)
    profile = {}
    profile["rdsi_enabled"] = torch.tensor(1.0, device=device)

    if teacher_logits_list is None or len(teacher_logits_list) == 0:
        return None, profile
    if teacher_ids is None or len(teacher_ids) != len(teacher_logits_list):
        teacher_ids = list(range(len(teacher_logits_list)))

    temperature = max(float(getattr(args, "rgftd_temperature", 1.0)), 1e-6)
    student_prob = F.softmax(student_logits.detach(), dim=1)
    student_conf = student_prob.max(dim=1)[0]
    student_entropy = _normalized_entropy(student_prob)
    log_student_prob = torch.log(student_prob.clamp_min(1e-6))

    num_classes = int(getattr(args, "num_classes", student_logits.shape[1]))
    target_label, _, valid_mask, support_mask, seed_support_mask = _safe_target_masks(
        label, wann_maps, num_classes
    )
    core_mask_raw = wann_maps.core_mask & valid_mask
    student_pred = student_prob.argmax(dim=1)
    hard_core_conf_thresh = float(getattr(args, "rdsi_hard_core_conf_thresh", 0.90))
    hard_core_entropy_thresh = float(getattr(args, "rdsi_hard_core_entropy_thresh", 0.25))
    hard_core_reliability_thresh = float(getattr(args, "rdsi_hard_core_reliability_thresh", 0.65))
    hard_core_mask = (
        core_mask_raw
        & (student_pred == target_label)
        & (student_conf >= hard_core_conf_thresh)
        & (student_entropy <= hard_core_entropy_thresh)
        & (wann_maps.reliability >= hard_core_reliability_thresh)
    )
    soft_core_mask = core_mask_raw & (~hard_core_mask)
    core_mask = hard_core_mask
    core_valid = bool(float(core_mask.float().sum().detach().cpu().item()) > 0.0)

    low_r_thresh = float(getattr(args, "rgftd_low_r_thresh", getattr(args, "wann_soft_thresh", 0.25)))
    use_soft_band = int(getattr(args, "rgftd_use_soft_band", 1)) == 1
    candidate_mask = getattr(wann_maps, "candidate_mask", None)
    if candidate_mask is None:
        candidate_mask = (~hard_core_mask) & (wann_maps.reliability >= low_r_thresh)
    candidate_mask = (candidate_mask | soft_core_mask) & (~hard_core_mask)
    non_conflict_mask = ~getattr(wann_maps, "low_conflict_mask", torch.zeros_like(valid_mask, dtype=torch.bool))
    reliable_candidate_mask = candidate_mask & non_conflict_mask & (wann_maps.reliability >= low_r_thresh)

    non_core_region = wann_maps.ignore_mask | wann_maps.soft_band
    if use_soft_band:
        non_core_region = non_core_region | candidate_mask
    region = (non_core_region | soft_core_mask) & (~hard_core_mask)
    if not bool(region.any().detach().cpu().item()):
        zero_map = torch.zeros_like(student_conf)
        zero_bool = torch.zeros_like(student_conf, dtype=torch.bool)
        setattr(args, "_rdsi_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_residual_alpha_map", zero_map.detach())
        setattr(args, "_rdsi_benefit_score", zero_map.detach())
        setattr(args, "_rdsi_selected_score", zero_map.detach())
        setattr(args, "_rdsi_risk_region", zero_bool.detach())
        setattr(args, "_rdsi_hard_core_mask", hard_core_mask.detach())
        setattr(args, "_rdsi_soft_core_mask", soft_core_mask.detach())
        profile["rdsi_risk_region_ratio"] = zero_map.mean().detach()
        profile["rdsi_hard_core_ratio"] = hard_core_mask.float().mean().detach()
        profile["rdsi_soft_core_ratio"] = soft_core_mask.float().mean().detach()
        profile["rdsi_accept_ratio"] = zero_map.mean().detach()
        profile["rdsi_reject_ratio"] = zero_map.mean().detach()
        return teacher_logits_list[0].detach(), profile
    region_f = region.float()
    region_sum = region_f.sum().clamp_min(1.0)

    teacher_fg_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    teacher_fg_prob_thresh = float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))
    teacher_fg_topk_ratio = float(getattr(args, "rgftd_teacher_fg_topk_ratio", 0.002))
    min_fg_pixels = int(getattr(args, "rgftd_min_foreground_pixels", 8))
    teacher_fg_topk_min_pixels = int(getattr(args, "rgftd_teacher_fg_topk_min_pixels", min_fg_pixels))
    teacher_bg_conf_thresh = float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98))
    bg_max_fg_prob = float(getattr(args, "rgftd_bg_max_fg_prob", 0.15))
    spatial_support_enabled = int(getattr(args, "rgftd_spatial_support_enabled", 1)) == 1
    spatial_support_radius = int(getattr(args, "rgftd_spatial_support_radius", teacher_fg_radius))
    spatial_candidate_weight = max(0.0, min(float(getattr(args, "rgftd_spatial_candidate_weight", 1.0)), 1.0))
    spatial_near_seed_weight = max(0.0, min(float(getattr(args, "rgftd_spatial_near_seed_weight", 0.75)), 1.0))
    spatial_far_weight = max(0.0, min(float(getattr(args, "rgftd_spatial_far_weight", 0.15)), 1.0))
    validation_enabled = int(getattr(args, "rgftd_teacher_validation_enabled", 0)) == 1
    entropy_margin = float(getattr(args, "rdsi_entropy_increase_margin", 0.05))
    entropy_scale = max(float(getattr(args, "rdsi_entropy_increase_scale", 0.35)), 1e-6)
    fg_excess_margin = float(getattr(args, "rdsi_fg_excess_margin", 0.05))
    fg_excess_scale = max(float(getattr(args, "rdsi_fg_excess_scale", 0.20)), 1e-6)
    residual_alpha_max = max(0.0, min(float(getattr(args, "rdsi_residual_alpha", 0.35)), 1.0))
    score_floor = max(float(getattr(args, "rdsi_benefit_score_floor", 1e-6)), 0.0)
    benefit_topk_ratio = max(0.0, min(float(getattr(args, "rdsi_benefit_topk_ratio", getattr(args, "rgftd_active_fg_topk_ratio", 0.002))), 1.0))
    benefit_topk_min_pixels = int(getattr(args, "rdsi_benefit_topk_min_pixels", getattr(args, "rgftd_active_fg_topk_min_pixels", min_fg_pixels)))
    benefit_topk_max_pixels = int(getattr(args, "rdsi_benefit_topk_max_pixels", getattr(args, "rgftd_active_fg_topk_max_pixels", 0)))

    fg_support = support_mask & (target_label > 0)
    seed_fg_support = seed_support_mask & (target_label > 0)
    seed_fg_spatial_context = _dilate_mask(seed_fg_support, spatial_support_radius)
    fg_support_context = _dilate_mask(fg_support, spatial_support_radius)
    foreground_candidate_mask = reliable_candidate_mask & fg_support_context
    if spatial_support_enabled:
        spatial_weight = region.float() * spatial_far_weight
        spatial_weight = torch.maximum(spatial_weight, seed_fg_spatial_context.float() * spatial_near_seed_weight)
        spatial_weight = torch.maximum(spatial_weight, foreground_candidate_mask.float() * spatial_candidate_weight)
        spatial_weight = spatial_weight * region.float()
    else:
        spatial_weight = region.float()

    weak_anchor_mask = seed_support_mask & region
    region_denom = region_f.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
    if student_prob.shape[1] > 1:
        student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
    else:
        student_fg_prob = torch.zeros_like(student_conf)
    student_fg_region_ratio = (student_fg_prob * region_f).flatten(1).sum(dim=1).view(-1, 1, 1) / region_denom
    support_fg_region_ratio = (
        ((seed_fg_spatial_context | foreground_candidate_mask) & region).float().flatten(1).sum(dim=1).view(-1, 1, 1)
        / region_denom
    )
    fg_prior_reference = torch.maximum(student_fg_region_ratio, support_fg_region_ratio)
    fg_deficit = (support_fg_region_ratio - student_fg_region_ratio).clamp_min(0.0)
    fg_excess_state = (student_fg_region_ratio - (support_fg_region_ratio + fg_excess_margin)).clamp_min(0.0)
    fg_excess_state_safe = (1.0 - fg_excess_state / fg_excess_scale).clamp(0.0, 1.0)
    fg_deficient_region = (fg_deficit > 1e-6).expand_as(region)
    fg_excessive_region = (fg_excess_state > 1e-6).expand_as(region)

    score_stack_items = []
    benefit_stack_items = []
    reliable_stack_items = []
    gap_stack_items = []
    student_risk_stack_items = []
    candidate_stack_items = []
    core_reject_items = []
    seed_reject_items = []
    prior_reject_items = []
    entropy_reject_items = []
    fg_excess_reject_items = []
    logits_stack_items = []
    teacher_id_values = []
    reliable_means = []
    gap_means = []

    for teacher_id, teacher_logits in zip(teacher_ids, teacher_logits_list):
        teacher_logits = teacher_logits.detach()
        teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
        teacher_conf = teacher_prob.max(dim=1)[0]
        teacher_pred = teacher_prob.argmax(dim=1)
        teacher_entropy = _normalized_entropy(teacher_prob)
        log_teacher_prob = torch.log(teacher_prob.clamp_min(1e-6))
        kl_teacher_student = (teacher_prob * (log_teacher_prob - log_student_prob)).sum(dim=1)
        kl_student_teacher = (student_prob * (log_student_prob - log_teacher_prob)).sum(dim=1)
        kl_normalizer = max(math.log(float(teacher_prob.shape[1])), 1e-6)
        knowledge_gap_score = (0.5 * (kl_teacher_student + kl_student_teacher) / kl_normalizer).clamp(0.0, 1.0)
        teacher_student_disagreement = (1.0 - (teacher_prob * student_prob).sum(dim=1)).clamp(0.0, 1.0)

        teacher_fg = teacher_pred > 0
        teacher_bg = ~teacher_fg
        if teacher_prob.shape[1] > 1:
            teacher_fg_prob = teacher_prob[:, 1:].max(dim=1)[0]
        else:
            teacher_fg_prob = torch.zeros_like(teacher_conf)
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
        teacher_fg_lift = (teacher_fg_prob > student_fg_prob) & region
        teacher_fg_ready = teacher_fg_anchor | teacher_fg_prob_anchor
        teacher_bg_conf = teacher_bg & (teacher_conf >= teacher_bg_conf_thresh)
        teacher_bg_safe = teacher_fg_prob <= bg_max_fg_prob
        teacher_accept = teacher_fg_ready | (teacher_bg_conf & teacher_bg_safe)
        teacher_fg_context = _dilate_mask(teacher_fg_ready, teacher_fg_radius)
        teacher_bg_context = teacher_bg_conf & teacher_bg_safe
        teacher_candidate_region = teacher_accept & (region & (teacher_fg_context | teacher_bg_context))

        if core_valid:
            teacher_core_agreement = _masked_ratio((teacher_pred == target_label) & core_mask, core_mask).detach()
            teacher_core_conflict = (1.0 - teacher_core_agreement).detach()
            core_consistency = (1.0 - teacher_core_conflict).clamp(0.0, 1.0)
        else:
            teacher_core_conflict = teacher_conf.mean().detach() * 0.0
            core_consistency = torch.tensor(1.0, device=device)

        if validation_enabled:
            teacher_seed_support_fg_prob_mean = _masked_mean(teacher_fg_prob, seed_fg_support).detach()
            teacher_seed_support_fg_margin_mean = _masked_mean(teacher_fg_margin, seed_fg_support).detach()
            support_gate = _normalize_gate(
                teacher_seed_support_fg_prob_mean,
                float(getattr(args, "rgftd_teacher_support_prob_floor", teacher_fg_prob_thresh)),
            )
            conf_gate = _normalize_gate(
                teacher_seed_support_fg_margin_mean,
                float(getattr(args, "rgftd_teacher_release_margin_floor", 0.05)),
            )
            if core_valid:
                core_gate = _normalize_gate(
                    1.0 - teacher_core_conflict,
                    float(getattr(args, "rgftd_teacher_core_agree_floor", 0.80)),
                )
            else:
                core_gate = torch.tensor(1.0, device=device)
            score_core_weight = float(getattr(args, "rgftd_teacher_score_core_weight", 0.35))
            score_support_weight = float(getattr(args, "rgftd_teacher_score_support_weight", 0.30))
            score_conf_weight = float(getattr(args, "rgftd_teacher_score_conf_weight", 0.10))
            weight_sum = score_core_weight + score_support_weight + score_conf_weight
            teacher_reliability = (
                score_core_weight * core_gate
                + score_support_weight * support_gate
                + score_conf_weight * conf_gate
            ) / max(weight_sum, 1e-6)
            core_conflict_limit = float(getattr(args, "rgftd_teacher_max_core_conflict", 0.20))
            if core_valid and float(teacher_core_conflict.cpu().item()) > core_conflict_limit:
                teacher_reliability = teacher_reliability * 0.0
        else:
            teacher_reliability = torch.tensor(1.0, device=device)

        weak_anchor_agreement = torch.where(
            weak_anchor_mask,
            ((teacher_pred == target_label) & weak_anchor_mask).float(),
            torch.ones_like(teacher_conf),
        )
        prior_safe = torch.where(teacher_fg_ready, torch.ones_like(teacher_conf), teacher_bg_safe.float())
        seed_safe = weak_anchor_agreement
        core_safe = torch.ones_like(teacher_conf) * core_consistency.detach().clamp(0.0, 1.0)
        entropy_increase = (teacher_entropy - student_entropy - entropy_margin).clamp_min(0.0)
        entropy_safe = torch.where(
            teacher_fg_ready,
            (1.0 - entropy_increase / entropy_scale).clamp(0.0, 1.0),
            torch.ones_like(teacher_conf),
        )
        teacher_fg_region_ratio = (
            (teacher_fg_prob * region_f).flatten(1).sum(dim=1).view(-1, 1, 1) / region_denom
        )
        fg_excess = (teacher_fg_region_ratio - (fg_prior_reference + fg_excess_margin)).clamp_min(0.0)
        fg_excess_safe_scalar = (1.0 - fg_excess / fg_excess_scale).clamp(0.0, 1.0)
        fg_excess_safe = fg_excess_safe_scalar.expand_as(teacher_conf)
        fg_state_safe = torch.where(
            teacher_fg_lift,
            fg_excess_state_safe.expand_as(teacher_conf),
            torch.ones_like(teacher_conf),
        )
        prior_benefit_safe = torch.where(teacher_fg_ready, fg_excess_safe * fg_state_safe, teacher_bg_safe.float())
        benefit_score = (
            core_safe
            * seed_safe
            * prior_benefit_safe
            * entropy_safe
        ).clamp(0.0, 1.0)
        teacher_local_reliability = (
            teacher_conf
            * (1.0 - teacher_entropy).clamp(0.0, 1.0)
            * weak_anchor_agreement
            * teacher_reliability.detach().clamp(0.0, 1.0)
            * core_consistency.detach()
            * prior_safe
        ).clamp(0.0, 1.0)
        student_uncertainty_score = torch.maximum(student_entropy, (1.0 - student_conf).clamp(0.0, 1.0))
        student_risk_evidence = torch.maximum(student_uncertainty_score, teacher_student_disagreement)
        student_local_risk = (
            (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
            * student_risk_evidence
        ).clamp(0.0, 1.0)
        candidate_eligibility = teacher_candidate_region.float() * region.float()
        release_score = (
            teacher_local_reliability
            * benefit_score
            * spatial_weight
            * candidate_eligibility
        ).clamp(0.0, 1.0)

        score_stack_items.append(release_score)
        benefit_stack_items.append(benefit_score)
        reliable_stack_items.append(teacher_local_reliability)
        gap_stack_items.append(knowledge_gap_score)
        student_risk_stack_items.append(student_local_risk)
        candidate_stack_items.append(teacher_candidate_region.float())
        core_reject_items.append((teacher_candidate_region.float() * (1.0 - core_safe)).clamp(0.0, 1.0))
        seed_reject_items.append((teacher_candidate_region & weak_anchor_mask & (teacher_pred != target_label)).float())
        prior_reject_items.append((teacher_candidate_region.float() * (1.0 - prior_benefit_safe)).clamp(0.0, 1.0))
        entropy_reject_items.append((teacher_candidate_region.float() * (1.0 - entropy_safe)).clamp(0.0, 1.0))
        fg_excess_reject_items.append((teacher_candidate_region.float() * (1.0 - fg_excess_safe)).clamp(0.0, 1.0))
        logits_stack_items.append(teacher_logits)
        teacher_id_values.append(float(teacher_id))
        reliable_means.append(_masked_mean(teacher_local_reliability, teacher_candidate_region).detach())
        gap_means.append(_masked_mean(knowledge_gap_score, teacher_candidate_region).detach())

    score_stack = torch.stack(score_stack_items, dim=0)
    benefit_stack = torch.stack(benefit_stack_items, dim=0)
    reliable_stack = torch.stack(reliable_stack_items, dim=0)
    gap_stack = torch.stack(gap_stack_items, dim=0)
    student_risk_stack = torch.stack(student_risk_stack_items, dim=0)
    candidate_stack = torch.stack(candidate_stack_items, dim=0)
    core_reject_stack = torch.stack(core_reject_items, dim=0)
    seed_reject_stack = torch.stack(seed_reject_items, dim=0)
    prior_reject_stack = torch.stack(prior_reject_items, dim=0)
    entropy_reject_stack = torch.stack(entropy_reject_items, dim=0)
    fg_excess_reject_stack = torch.stack(fg_excess_reject_items, dim=0)
    logits_stack = torch.stack(logits_stack_items, dim=0)
    best_score, best_index = score_stack.max(dim=0)
    gather_index = best_index.unsqueeze(0).unsqueeze(2).expand(
        1,
        logits_stack.shape[1],
        logits_stack.shape[2],
        logits_stack.shape[3],
        logits_stack.shape[4],
    )
    selected_logits = logits_stack.gather(0, gather_index).squeeze(0).detach()
    gather_map_index = best_index.unsqueeze(0)
    selected_benefit = benefit_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_reliable = reliable_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_gap = gap_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_student_risk = student_risk_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_candidate = candidate_stack.gather(0, gather_map_index).squeeze(0) > 0.5
    selected_core_reject = core_reject_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_seed_reject = seed_reject_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_prior_reject = prior_reject_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_entropy_reject = entropy_reject_stack.gather(0, gather_map_index).squeeze(0).detach()
    selected_fg_excess_reject = fg_excess_reject_stack.gather(0, gather_map_index).squeeze(0).detach()

    raw_active_mask = (best_score > score_floor) & selected_candidate & region
    active_mask = _topk_foreground_anchor(
        best_score,
        raw_active_mask,
        topk_ratio=benefit_topk_ratio,
        min_pixels=benefit_topk_min_pixels,
        max_pixels=benefit_topk_max_pixels if benefit_topk_max_pixels > 0 else None,
    ) & raw_active_mask
    residual_alpha = (
        residual_alpha_max
        * selected_reliable
        * selected_benefit
        * active_mask.float()
    ).clamp(0.0, residual_alpha_max)
    selected_teacher_prob = F.softmax(selected_logits / temperature, dim=1)

    active_sum = active_mask.float().sum().clamp_min(1.0)
    candidate_any = (candidate_stack.max(dim=0).values > 0.5) & region
    if score_stack.shape[0] > 1:
        top2 = torch.topk(score_stack, k=2, dim=0).values
        best_vs_second = (top2[0] - top2[1]).clamp_min(0.0)
    else:
        best_vs_second = best_score
    teacher_id_tensor = torch.tensor(teacher_id_values, device=device, dtype=torch.float32)
    selected_teacher_id = teacher_id_tensor[best_index]

    profile["rdsi_teacher_compete_count"] = torch.tensor(float(len(teacher_logits_list)), device=device)
    profile["rdsi_candidate_teacher_count"] = torch.tensor(float(len(teacher_logits_list)), device=device)
    profile["rdsi_risk_region_ratio"] = region.float().mean().detach()
    profile["rdsi_hard_core_ratio"] = hard_core_mask.float().mean().detach()
    profile["rdsi_soft_core_ratio"] = soft_core_mask.float().mean().detach()
    profile["rdsi_fg_deficient_ratio"] = (fg_deficient_region & region).float().mean().detach()
    profile["rdsi_fg_excessive_ratio"] = (fg_excessive_region & region).float().mean().detach()
    profile["rdsi_candidate_ratio"] = (candidate_any.float().sum() / region_sum).detach()
    profile["rdsi_accept_ratio"] = (active_mask.float().sum() / region_sum).detach()
    profile["rdsi_reject_ratio"] = ((candidate_any & (~active_mask)).float().sum() / region_sum).detach()
    profile["rdsi_reject_by_core"] = (selected_core_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_seed"] = (selected_seed_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_prior"] = (selected_prior_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_entropy"] = (selected_entropy_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_fg_excess"] = (selected_fg_excess_reject.sum() / region_sum).detach()
    profile["rdsi_multi_teacher_active"] = active_mask.float().mean().detach()
    profile["rdsi_best_vs_second_gap"] = _masked_mean(best_vs_second, active_mask).detach()
    profile["rdsi_selected_score_mean"] = _masked_mean(best_score, active_mask).detach()
    profile["rdsi_selected_score_top"] = _masked_mean(best_score, active_mask & (best_score >= best_score.mean())).detach()
    profile["rdsi_score_benefit_mean"] = _masked_mean(selected_reliable * selected_benefit, active_mask).detach()
    profile["rdsi_selected_teacher_reliable"] = _masked_mean(selected_reliable, active_mask).detach()
    profile["rdsi_selected_teacher_benefit"] = _masked_mean(selected_benefit, active_mask).detach()
    profile["rdsi_selected_teacher_gap"] = _masked_mean(selected_gap, active_mask).detach()
    profile["rdsi_selected_student_risk"] = _masked_mean(selected_student_risk, active_mask).detach()
    profile["rdsi_benefit_mean"] = _masked_mean(selected_benefit, active_mask).detach()
    profile["rdsi_benefit_top"] = _masked_mean(selected_benefit, active_mask & (selected_benefit >= selected_benefit.mean())).detach()
    profile["rdsi_alpha_mean"] = _masked_mean(residual_alpha, active_mask).detach()
    profile["rdsi_alpha_top"] = _masked_mean(residual_alpha, active_mask & (residual_alpha >= residual_alpha.mean())).detach()
    if student_prob.shape[1] > 1:
        selected_teacher_fg_prob = selected_teacher_prob[:, 1:].sum(dim=1)
        student_fg_mass = student_prob[:, 1:].sum(dim=1)
        profile["rdsi_raw_teacher_fg_delta"] = _masked_mean(
            selected_teacher_fg_prob - student_fg_mass,
            active_mask,
        ).detach()
    else:
        profile["rdsi_raw_teacher_fg_delta"] = residual_alpha.sum().detach() * 0.0
    profile["rdsi_selected_teacher_mean"] = ((selected_teacher_id * active_mask.float()).sum() / active_sum).detach()
    if bool(active_mask.any().detach().cpu().item()):
        hist_values = []
        for teacher_id in teacher_id_values:
            teacher_ratio = (((selected_teacher_id == float(teacher_id)) & active_mask).float().sum() / active_sum).detach()
            hist_values.append(teacher_ratio)
            int_teacher_id = int(teacher_id)
            if 0 <= int_teacher_id <= 9:
                profile["rdsi_teacher{}_ratio".format(int_teacher_id)] = teacher_ratio
        if hist_values:
            profile["rdsi_selected_teacher_switch_ratio"] = (1.0 - torch.stack(hist_values).max()).detach()
    if reliable_means:
        profile["rdsi_teacher_reliable_score"] = torch.stack(reliable_means).mean().detach()
    if gap_means:
        profile["rdsi_knowledge_gap_score"] = torch.stack(gap_means).mean().detach()
    setattr(args, "_rdsi_active_mask", active_mask.detach())
    setattr(args, "_rdsi_residual_alpha_map", residual_alpha.detach())
    setattr(args, "_rdsi_benefit_score", selected_benefit.detach())
    setattr(args, "_rdsi_selected_score", best_score.detach())
    setattr(args, "_rdsi_risk_region", region.detach())
    setattr(args, "_rdsi_hard_core_mask", hard_core_mask.detach())
    setattr(args, "_rdsi_soft_core_mask", soft_core_mask.detach())
    return selected_logits, profile


def _build_refined_teacher_target(
    teacher_prob,
    image,
    label,
    wann_maps,
    args,
    active_fg_mask,
    active_bg_mask,
    region,
    candidate_mask,
    foreground_candidate_mask,
    seed_fg_support,
    seed_bg_support,
    core_mask,
    seed_fg_context,
    teacher_fg_prob,
):
    device = teacher_prob.device
    profile = {}
    num_classes = teacher_prob.shape[1]
    enabled = int(getattr(args, "rgftd_refine_enabled", 0)) == 1
    profile["refine_enabled"] = torch.tensor(1.0 if enabled else 0.0, device=device)
    profile["refine_silent"] = torch.tensor(0.0, device=device)
    if not enabled:
        zero = teacher_prob[:, :1].sum() * 0.0
        for key in [
            "refine_roi_ratio",
            "refine_affinity_mean",
            "refine_teacher_q_kl",
            "refine_q_entropy_mean",
            "refine_q_fg_mass",
            "refine_q_fg_ratio",
            "refine_q_fg_delta",
            "refine_q_seed_precision",
            "refine_q_seed_recall",
            "refine_q_candidate_ratio",
            "refine_q_near_seed_ratio",
            "refine_unsupported_fg_ratio",
            "refine_unsupported_fg_scale",
            "refine_q_core_conflict",
        ]:
            profile[key] = zero.detach()
        return teacher_prob.detach(), False, profile

    support_roi = active_fg_mask | active_bg_mask | seed_fg_context | (candidate_mask & region)
    core_anchor_radius = int(getattr(args, "rgftd_refine_core_anchor_radius", 1))
    core_anchor = core_mask & _dilate_mask(support_roi, core_anchor_radius)
    refine_roi = support_roi | core_anchor | seed_fg_support
    q = teacher_prob.detach().clone()
    supported_fg_region = seed_fg_context | foreground_candidate_mask
    unsupported_fg_region = active_fg_mask & (~supported_fg_region)

    seed_fg_anchor = seed_fg_support & refine_roi
    seed_bg_anchor = seed_bg_support & refine_roi
    if num_classes > 1:
        seed_strength = float(getattr(args, "rgftd_refine_seed_strength", 0.95))
        seed_strength = max(0.0, min(seed_strength, 1.0))
        weak_label_q = _one_hot_label(label, num_classes, q)
        q = torch.where(
            seed_fg_anchor.unsqueeze(1),
            seed_strength * weak_label_q + (1.0 - seed_strength) * q,
            q,
        )
        q = torch.where(seed_bg_anchor.unsqueeze(1), seed_strength * weak_label_q + (1.0 - seed_strength) * q, q)
        q = q.clamp_min(1e-6)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)

    if core_mask.any():
        core_q = _one_hot_label(label, num_classes, q)
        q = torch.where(core_mask.unsqueeze(1), core_q, q)

    locked_mask = core_anchor | seed_fg_anchor | seed_bg_anchor
    locked_q = q.detach()
    q, affinity_mean = _local_affinity_refine(q, image, refine_roi, locked_mask, locked_q, args)

    fg_floor = float(getattr(args, "rgftd_refine_fg_floor", 0.02))
    bg_ceiling = float(getattr(args, "rgftd_refine_bg_ceiling", 0.98))
    fg_floor = max(0.0, min(fg_floor, 0.95))
    bg_ceiling = max(0.0, min(bg_ceiling, 1.0))
    if num_classes > 1:
        fg_region = active_fg_mask | seed_fg_context
        fg_mass = q[:, 1:].sum(dim=1)
        if fg_floor > 0.0:
            need_lift = fg_region & (fg_mass < fg_floor)
            lift = (fg_floor - fg_mass).clamp_min(0.0)
            q[:, 0] = torch.where(need_lift, (q[:, 0] - lift).clamp_min(1e-6), q[:, 0])
            class_mass = q[:, 1:].sum(dim=1, keepdim=True).clamp_min(1e-6)
            q[:, 1:] = torch.where(
                need_lift.unsqueeze(1),
                q[:, 1:] + lift.unsqueeze(1) * q[:, 1:] / class_mass,
                q[:, 1:],
            )
        q[:, 0] = torch.where(refine_roi, q[:, 0].clamp(max=bg_ceiling), q[:, 0])
        q = q.clamp_min(1e-6)
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)
        unsupported_fg_scale = float(getattr(args, "rgftd_refine_unsupported_fg_scale", 0.25))
        unsupported_fg_scale = max(0.0, min(unsupported_fg_scale, 1.0))
        if unsupported_fg_scale < 1.0:
            unsupported = unsupported_fg_region.unsqueeze(1)
            old_fg = q[:, 1:].sum(dim=1, keepdim=True)
            new_fg_classes = q[:, 1:] * unsupported_fg_scale
            removed_fg = old_fg * (1.0 - unsupported_fg_scale)
            q[:, 1:] = torch.where(unsupported, new_fg_classes, q[:, 1:])
            q[:, 0:1] = torch.where(unsupported, q[:, 0:1] + removed_fg, q[:, 0:1])
            q = q.clamp_min(1e-6)
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        unsupported_fg_scale = 1.0

    q = torch.where(locked_mask.unsqueeze(1), locked_q, q)

    q_fg_prob = q[:, 1:].sum(dim=1) if num_classes > 1 else q[:, 0] * 0.0
    q_fg_mass = (q_fg_prob * active_fg_mask.float()).sum()
    active_pixels = (active_fg_mask | active_bg_mask).float().sum().clamp_min(1.0)
    q_fg_ratio = q_fg_mass / active_pixels
    teacher_fg_mass = (teacher_fg_prob * active_fg_mask.float()).sum()
    q_entropy = _normalized_entropy(q)
    q_pred = q.argmax(dim=1)
    q_seed_precision = _masked_mean(q_fg_prob, seed_fg_support & refine_roi)
    q_seed_recall = _masked_ratio((q_fg_prob >= fg_floor) & seed_fg_support & refine_roi, seed_fg_support & refine_roi)
    q_candidate_ratio = _masked_mean(q_fg_prob, candidate_mask & refine_roi)
    q_near_seed_ratio = _masked_mean(q_fg_prob, seed_fg_context & refine_roi)
    unsupported_fg_ratio = _masked_ratio(unsupported_fg_region, active_fg_mask)
    if core_mask.any():
        q_core_conflict = _masked_ratio((q_pred != label) & core_mask, core_mask)
    else:
        q_core_conflict = q[:, :1].sum() * 0.0
    teacher_to_q_kl = F.kl_div(
        torch.log(q.clamp_min(1e-6)),
        teacher_prob,
        reduction="none",
    ).sum(dim=1)
    roi_f = refine_roi.float()
    profile["refine_roi_ratio"] = refine_roi.float().mean().detach()
    profile["refine_affinity_mean"] = affinity_mean.detach()
    profile["refine_teacher_q_kl"] = _masked_mean(teacher_to_q_kl, refine_roi).detach()
    profile["refine_q_entropy_mean"] = _masked_mean(q_entropy, refine_roi).detach()
    profile["refine_q_fg_mass"] = q_fg_mass.detach()
    profile["refine_q_fg_ratio"] = q_fg_ratio.detach()
    profile["refine_q_fg_delta"] = (q_fg_mass - teacher_fg_mass).detach()
    profile["refine_q_seed_precision"] = q_seed_precision.detach()
    profile["refine_q_seed_recall"] = q_seed_recall.detach()
    profile["refine_q_candidate_ratio"] = q_candidate_ratio.detach()
    profile["refine_q_near_seed_ratio"] = q_near_seed_ratio.detach()
    profile["refine_unsupported_fg_ratio"] = unsupported_fg_ratio.detach()
    profile["refine_unsupported_fg_scale"] = torch.tensor(unsupported_fg_scale, device=device)
    profile["refine_q_core_conflict"] = q_core_conflict.detach()

    min_fg_mass = float(getattr(args, "rgftd_refine_min_fg_mass", 1.0))
    min_roi_pixels = float(getattr(args, "rgftd_refine_min_roi_pixels", 1.0))
    silent = (
        float(roi_f.sum().detach().cpu().item()) < min_roi_pixels
        or float(q_fg_mass.detach().cpu().item()) < min_fg_mass
    )
    profile["refine_silent"] = torch.tensor(1.0 if silent else 0.0, device=device)
    return q.detach(), silent, profile


def _classwise_teacher_validation(
    teacher_pred,
    teacher_conf,
    label,
    support_mask,
    core_mask,
    num_classes,
    args,
):
    class_scores = []
    class_score_map = torch.zeros_like(teacher_conf)
    class_profile = {}
    conf_floor = float(getattr(args, "rgftd_teacher_conf_floor", 0.85))
    support_floor = float(getattr(args, "rgftd_teacher_support_agree_floor", 0.70))
    core_floor = float(getattr(args, "rgftd_teacher_core_agree_floor", 0.80))
    class_reliability_min = float(getattr(args, "rgftd_teacher_class_reliability_min", 0.50))

    for class_id in range(1, int(num_classes)):
        support_class = support_mask & (label == class_id)
        core_class = core_mask & (label == class_id)
        support_class_count = float(support_class.float().sum().detach().cpu().item())
        if support_class_count <= 0.0:
            continue

        support_agreement = _masked_ratio((teacher_pred == class_id) & support_class, support_class)
        if float(core_class.float().sum().detach().cpu().item()) > 0.0:
            core_agreement = _masked_ratio((teacher_pred == class_id) & core_class, core_class)
            class_conf_mean = _masked_mean(teacher_conf, core_class)
        else:
            core_agreement = support_agreement
            class_conf_mean = _masked_mean(teacher_conf, support_class)

        support_gate = _normalize_gate(support_agreement, support_floor)
        core_gate = _normalize_gate(core_agreement, core_floor)
        conf_gate = _normalize_gate(class_conf_mean, conf_floor)
        reliability = (support_gate + core_gate + conf_gate) / 3.0

        class_profile["teacher_class{}_agreement".format(class_id)] = support_agreement.detach()
        class_profile["teacher_class{}_core_agreement".format(class_id)] = core_agreement.detach()
        class_profile["teacher_class{}_reliability".format(class_id)] = reliability.detach()

        class_scores.append(reliability)
        class_score_map = class_score_map + (teacher_pred == class_id).float() * reliability

    if class_scores:
        class_score_mean = torch.stack(class_scores, dim=0).mean()
    else:
        class_score_mean = teacher_conf.mean() * 0.0

    return class_score_map, class_score_mean, class_reliability_min, class_profile


def get_rgftd_lambda(iter_num, args):
    forced_lambda = getattr(args, "rgftd_force_lambda", None)
    if forced_lambda is not None:
        return max(float(forced_lambda), 0.0)
    warmup = int(getattr(args, "rgftd_warmup_iters", 800))
    rampup = int(getattr(args, "rgftd_rampup_iters", 800))
    max_lambda = float(getattr(args, "rgftd_lambda", 0.1))
    if int(iter_num) < warmup:
        return 0.0
    if rampup <= 0:
        return max_lambda
    current = max(0, int(iter_num) - warmup)
    if current >= rampup:
        return max_lambda
    phase = 1.0 - float(current) / float(rampup)
    return max_lambda * math.exp(-5.0 * phase * phase)


def rgftd_loss(student_logits, teacher_logits, label, wann_maps, args, iter_num, image=None):
    """Reliability-gated teacher distillation on WANN non-core regions."""
    device = student_logits.device
    profile = zero_rgftd_profile(device)
    lambda_rgftd = get_rgftd_lambda(iter_num, args)
    profile["lambda"] = torch.tensor(lambda_rgftd, device=device)
    profile["lambda_effective"] = torch.tensor(lambda_rgftd, device=device)

    temperature = float(getattr(args, "rgftd_temperature", 1.0))
    temperature = max(temperature, 1e-6)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_prob = F.softmax(student_logits.detach(), dim=1)

    teacher_conf = teacher_prob.max(dim=1)[0]
    teacher_pred = teacher_prob.argmax(dim=1)
    student_conf = student_prob.max(dim=1)[0]
    student_pred = student_prob.argmax(dim=1)
    teacher_entropy = _normalized_entropy(teacher_prob)
    student_entropy = _normalized_entropy(student_prob)
    log_teacher_prob = torch.log(teacher_prob.clamp_min(1e-6))
    log_student_prob = torch.log(student_prob.clamp_min(1e-6))
    kl_teacher_student = (teacher_prob * (log_teacher_prob - log_student_prob)).sum(dim=1)
    kl_student_teacher = (student_prob * (log_student_prob - log_teacher_prob)).sum(dim=1)
    kl_normalizer = max(math.log(float(teacher_prob.shape[1])), 1e-6)
    knowledge_gap_score = (0.5 * (kl_teacher_student + kl_student_teacher) / kl_normalizer).clamp(0.0, 1.0)
    teacher_student_disagreement = (1.0 - (teacher_prob * student_prob).sum(dim=1)).clamp(0.0, 1.0)
    raw_label = label.long()
    num_classes = int(getattr(args, "num_classes", teacher_prob.shape[1]))
    target_label = getattr(wann_maps, "target_label", raw_label).long()
    target_valid = (target_label >= 0) & (target_label < num_classes)
    valid_mask = getattr(wann_maps, "valid_mask", raw_label != num_classes) & target_valid
    raw_support_mask = getattr(wann_maps, "support_mask", valid_mask) & target_valid
    raw_seed_support_mask = getattr(wann_maps, "seed_support_mask", raw_support_mask) & target_valid
    target_support_mask = valid_mask & target_valid
    support_mask = raw_support_mask | target_support_mask
    seed_support_mask = raw_seed_support_mask | target_support_mask
    label = target_label
    core_mask_raw = wann_maps.core_mask & valid_mask
    rdsi_hard_core_mask = _runtime_map(args, "_rdsi_hard_core_mask", teacher_conf, as_bool=True)
    rdsi_soft_core_mask = _runtime_map(args, "_rdsi_soft_core_mask", teacher_conf, as_bool=True)
    if rdsi_hard_core_mask is not None:
        core_mask = rdsi_hard_core_mask & valid_mask
    else:
        hard_core_conf_thresh = float(getattr(args, "rdsi_hard_core_conf_thresh", 0.90))
        hard_core_entropy_thresh = float(getattr(args, "rdsi_hard_core_entropy_thresh", 0.25))
        hard_core_reliability_thresh = float(getattr(args, "rdsi_hard_core_reliability_thresh", 0.65))
        core_mask = (
            core_mask_raw
            & (student_pred == target_label)
            & (student_conf >= hard_core_conf_thresh)
            & (student_entropy <= hard_core_entropy_thresh)
            & (wann_maps.reliability >= hard_core_reliability_thresh)
        )
    if rdsi_soft_core_mask is not None:
        soft_core_mask = rdsi_soft_core_mask & valid_mask
    else:
        soft_core_mask = core_mask_raw & (~core_mask)

    low_r_thresh = float(getattr(args, "rgftd_low_r_thresh", getattr(args, "wann_soft_thresh", 0.25)))
    use_soft_band = int(getattr(args, "rgftd_use_soft_band", 1)) == 1
    teacher_fg_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    min_fg_pixels = int(getattr(args, "rgftd_min_foreground_pixels", 8))
    min_fg_ratio = float(getattr(args, "rgftd_min_foreground_ratio", 0.05))
    teacher_fg_prob_thresh = float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))
    teacher_fg_topk_ratio = float(getattr(args, "rgftd_teacher_fg_topk_ratio", 0.002))
    teacher_fg_topk_min_pixels = int(getattr(args, "rgftd_teacher_fg_topk_min_pixels", min_fg_pixels))
    active_fg_topk_ratio = float(getattr(args, "rgftd_active_fg_topk_ratio", 0.002))
    active_fg_topk_min_pixels = int(getattr(args, "rgftd_active_fg_topk_min_pixels", min_fg_pixels))
    active_fg_topk_max_pixels = int(getattr(args, "rgftd_active_fg_topk_max_pixels", 4096))
    spatial_support_enabled = int(getattr(args, "rgftd_spatial_support_enabled", 1)) == 1
    spatial_support_radius = int(getattr(args, "rgftd_spatial_support_radius", teacher_fg_radius))
    spatial_candidate_weight = float(getattr(args, "rgftd_spatial_candidate_weight", 1.0))
    spatial_near_seed_weight = float(getattr(args, "rgftd_spatial_near_seed_weight", 0.75))
    spatial_far_weight = float(getattr(args, "rgftd_spatial_far_weight", 0.15))
    spatial_candidate_weight = max(0.0, min(spatial_candidate_weight, 1.0))
    spatial_near_seed_weight = max(0.0, min(spatial_near_seed_weight, 1.0))
    spatial_far_weight = max(0.0, min(spatial_far_weight, 1.0))
    max_bg_fg_ratio = float(getattr(args, "rgftd_max_bg_fg_ratio", 1.0))
    allow_bg_without_fg = int(getattr(args, "rgftd_allow_bg_without_fg", 0)) == 1
    lambda_eff_cap = float(getattr(args, "rgftd_lambda_eff_cap", 0.02))
    teacher_student_fg_margin = float(getattr(args, "rgftd_teacher_student_fg_margin", 0.05))
    teacher_bg_conf_thresh = float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98))
    bg_max_fg_prob = float(getattr(args, "rgftd_bg_max_fg_prob", 0.15))
    teacher_support_prob_floor = float(getattr(args, "rgftd_teacher_support_prob_floor", teacher_fg_prob_thresh))
    skip_background_only = int(getattr(args, "rgftd_skip_background_only", 1)) == 1
    validation_enabled = int(getattr(args, "rgftd_teacher_validation_enabled", 0)) == 1

    candidate_mask = getattr(wann_maps, "candidate_mask", None)
    if candidate_mask is None:
        candidate_mask = (~core_mask) & (wann_maps.reliability >= low_r_thresh)
    candidate_mask = (candidate_mask | soft_core_mask) & (~core_mask)
    non_conflict_mask = ~getattr(wann_maps, "low_conflict_mask", torch.zeros_like(valid_mask, dtype=torch.bool))
    reliable_candidate_mask = candidate_mask & non_conflict_mask & (wann_maps.reliability >= low_r_thresh)

    # Re-open non-core pixels, but only around teacher foreground evidence.
    # This keeps the v1 semantics (uncertain / ignore can be distilled) without turning RGFTD into background-only KL.
    non_core_region = wann_maps.ignore_mask | wann_maps.soft_band
    if use_soft_band:
        non_core_region = non_core_region | candidate_mask
    region = (non_core_region | soft_core_mask) & (~core_mask)
    risk_region = region
    preserve_region = core_mask & valid_mask

    teacher_fg = teacher_pred > 0
    teacher_bg = ~teacher_fg
    if teacher_prob.shape[1] > 1:
        teacher_fg_prob = teacher_prob[:, 1:].max(dim=1)[0]
        student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
    else:
        teacher_fg_prob = torch.zeros_like(teacher_conf)
        student_fg_prob = torch.zeros_like(student_conf)
    teacher_bg_prob = teacher_prob[:, 0]
    teacher_fg_margin = teacher_fg_prob - teacher_bg_prob
    seed_support_mask = (
        seed_support_mask
        | (core_mask & target_valid)
        | (reliable_candidate_mask & target_valid)
    )
    fg_support = support_mask & (label > 0)
    bg_support = support_mask & (label == 0)
    seed_fg_support = seed_support_mask & (label > 0)
    seed_bg_support = seed_support_mask & (label == 0)
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
    teacher_accept = teacher_fg_ready | (teacher_bg_conf & teacher_bg_safe)
    teacher_fg_context = _dilate_mask(teacher_fg_ready, teacher_fg_radius)
    teacher_bg_context = teacher_bg_conf & teacher_bg_safe
    rdsi_active_mask = _runtime_map(args, "_rdsi_active_mask", teacher_conf, as_bool=True)
    rdsi_alpha_map = _runtime_map(args, "_rdsi_residual_alpha_map", teacher_conf)
    rdsi_benefit_map = _runtime_map(args, "_rdsi_benefit_score", teacher_conf)
    rdsi_score_map = _runtime_map(args, "_rdsi_selected_score", teacher_conf)
    if rdsi_active_mask is not None:
        teacher_candidate_region = rdsi_active_mask & region
    else:
        region = region & (teacher_fg_context | teacher_bg_context)
        teacher_candidate_region = teacher_accept & region

    core_pixels = core_mask.float().sum().detach()
    core_valid = bool(float(core_pixels.cpu().item()) > 0.0)
    if core_valid:
        teacher_core_agreement = _masked_ratio((teacher_pred == label) & core_mask, core_mask).detach()
        teacher_core_conflict = (1.0 - teacher_core_agreement).detach()
        teacher_core_conf_mean = _masked_mean(teacher_conf, core_mask).detach()
    else:
        teacher_core_agreement = teacher_conf.mean().detach() * 0.0
        teacher_core_conflict = teacher_conf.mean().detach() * 0.0
        teacher_core_conf_mean = teacher_conf.mean().detach() * 0.0
    teacher_support_agreement = _masked_ratio((teacher_pred == label) & support_mask, support_mask).detach()
    teacher_support_conflict = (1.0 - teacher_support_agreement).detach()
    teacher_support_conf_mean = _masked_mean(teacher_conf, support_mask).detach()
    teacher_support_fg_recall = _masked_ratio((teacher_pred > 0) & fg_support, fg_support).detach()
    teacher_support_bg_agreement = _masked_ratio((teacher_pred == 0) & bg_support, bg_support).detach()
    teacher_seed_support_agreement = _masked_ratio((teacher_pred == label) & seed_support_mask, seed_support_mask).detach()
    teacher_seed_support_conflict = (1.0 - teacher_seed_support_agreement).detach()
    teacher_seed_support_conf_mean = _masked_mean(teacher_conf, seed_support_mask).detach()
    teacher_seed_support_fg_recall = _masked_ratio((teacher_pred > 0) & seed_fg_support, seed_fg_support).detach()
    teacher_seed_support_fg_prob_mean = _masked_mean(teacher_fg_prob, seed_fg_support).detach()
    teacher_seed_support_fg_conf_mean = _masked_mean(teacher_conf, seed_fg_support).detach()
    teacher_seed_support_fg_margin_mean = _masked_mean(teacher_fg_margin, seed_fg_support).detach()
    teacher_seed_support_bg_agreement = _masked_ratio((teacher_pred == 0) & seed_bg_support, seed_bg_support).detach()
    seed_fg_support_pixels = seed_fg_support.float().sum().detach()

    class_weight_map = torch.ones_like(teacher_conf)
    class_reliability_mean = teacher_conf.mean() * 0.0
    class_reliability_floor = 0.0
    class_profile = {}
    if validation_enabled and teacher_prob.shape[1] > 1:
        class_weight_map, class_reliability_mean, class_reliability_floor, class_profile = _classwise_teacher_validation(
            teacher_pred=teacher_pred,
            teacher_conf=teacher_conf,
            label=label,
            support_mask=seed_fg_support,
            core_mask=core_mask & (label > 0),
            num_classes=teacher_prob.shape[1],
            args=args,
        )
        if class_profile:
            profile.update(class_profile)
    for class_id in range(1, teacher_prob.shape[1]):
        profile.setdefault("teacher_class{}_agreement".format(class_id), teacher_conf.mean() * 0.0)
        profile.setdefault("teacher_class{}_core_agreement".format(class_id), teacher_conf.mean() * 0.0)
        profile.setdefault("teacher_class{}_reliability".format(class_id), teacher_conf.mean() * 0.0)

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
        weighted_score = (
            score_core_weight * core_gate
            + score_support_weight * support_gate
            + score_class_weight * class_gate
            + score_conf_weight * conf_gate
        )
        weight_sum = score_core_weight + score_support_weight + score_class_weight + score_conf_weight
        teacher_reliability = weighted_score / max(weight_sum, 1e-6)
        core_conflict_veto = bool(
            float(teacher_core_conflict.cpu().item()) >
            float(getattr(args, "rgftd_teacher_max_core_conflict", 0.20))
        )
        release_prob_gate = _normalize_gate(
            teacher_seed_support_fg_prob_mean,
            float(getattr(args, "rgftd_teacher_release_prob_floor", teacher_fg_prob_thresh)),
        )
        release_conf_gate = _normalize_gate(
            teacher_seed_support_fg_margin_mean,
            margin_floor,
        )
        release_class_gate = _normalize_gate(
            class_reliability_mean,
            float(getattr(args, "rgftd_teacher_release_class_floor", class_reliability_floor)),
        )
        if teacher_prob.shape[1] > 2:
            release_raw = release_prob_gate * release_conf_gate * release_class_gate
        else:
            release_raw = release_prob_gate * release_conf_gate
        if float(seed_fg_support.float().sum().detach().cpu().item()) > 0.0:
            release_min = float(getattr(args, "rgftd_teacher_release_min", 0.03))
            max_lambda = float(max(getattr(args, "rgftd_lambda", 0.1), 1e-6))
            release_schedule = float(max(min(lambda_rgftd / max_lambda, 1.0), 0.0))
            if teacher_prob.shape[1] > 2:
                release_quality = release_prob_gate * release_conf_gate * release_class_gate
            else:
                release_quality = release_prob_gate * release_conf_gate
            release_min_effective = release_min * release_quality * release_schedule
            release_factor = release_raw + (1.0 - release_raw) * release_min_effective
        else:
            release_quality = teacher_conf.mean() * 0.0
            release_min_effective = teacher_conf.mean() * 0.0
            release_factor = teacher_conf.mean() * 0.0
        teacher_validation_veto = core_conflict_veto
    else:
        teacher_reliability = torch.tensor(1.0, device=device)
        release_prob_gate = torch.tensor(1.0, device=device)
        release_conf_gate = torch.tensor(1.0, device=device)
        release_class_gate = torch.tensor(1.0, device=device)
        release_raw = torch.tensor(1.0, device=device)
        release_quality = torch.tensor(1.0, device=device)
        release_min_effective = torch.tensor(0.0, device=device)
        release_factor = torch.tensor(1.0, device=device)
        teacher_validation_veto = False
        core_conflict_veto = False

    student_uncertain = (
        (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
        | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
    )
    foreground_correction = teacher_fg_ready & ((teacher_fg_prob - student_fg_prob) >= teacher_student_fg_margin)
    seed_fg_spatial_context = _dilate_mask(seed_fg_support, spatial_support_radius)
    fg_support_context = _dilate_mask(fg_support, spatial_support_radius)
    foreground_candidate_mask = candidate_mask & fg_support_context
    spatial_support_gate = (foreground_candidate_mask | seed_fg_spatial_context) & region
    support_reject_map = (
        risk_region
        & teacher_fg_ready
        & (~seed_fg_spatial_context)
        & (~foreground_candidate_mask)
    )
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

    weak_anchor_mask = seed_support_mask & region
    weak_anchor_agreement = torch.where(
        weak_anchor_mask,
        ((teacher_pred == label) & weak_anchor_mask).float(),
        torch.ones_like(teacher_conf),
    )
    if core_valid:
        core_consistency = (1.0 - teacher_core_conflict).clamp(0.0, 1.0)
    else:
        core_consistency = torch.tensor(1.0, device=device)
    prior_safe = torch.where(
        teacher_fg_ready,
        torch.ones_like(teacher_conf),
        teacher_bg_safe.float(),
    )
    teacher_local_reliability = (
        teacher_conf
        * (1.0 - teacher_entropy).clamp(0.0, 1.0)
        * weak_anchor_agreement
        * teacher_reliability.detach().clamp(0.0, 1.0)
        * core_consistency.detach()
        * prior_safe
    ).clamp(0.0, 1.0)
    student_uncertainty_score = torch.maximum(
        student_entropy,
        (1.0 - student_conf).clamp(0.0, 1.0),
    )
    student_risk_evidence = torch.maximum(student_uncertainty_score, teacher_student_disagreement)
    student_local_risk = (
        (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
        * student_risk_evidence
    ).clamp(0.0, 1.0)
    if rdsi_active_mask is not None:
        if rdsi_score_map is not None:
            release_score = (
                rdsi_score_map.clamp(0.0, 1.0)
                * teacher_candidate_region.float()
            ).clamp(0.0, 1.0)
        elif rdsi_benefit_map is not None:
            release_score = (
                teacher_local_reliability
                * rdsi_benefit_map.clamp(0.0, 1.0)
                * teacher_candidate_region.float()
            ).clamp(0.0, 1.0)
        else:
            release_score = (
                teacher_local_reliability
                * teacher_candidate_region.float()
            ).clamp(0.0, 1.0)
    else:
        release_score = (
            teacher_local_reliability
            * student_local_risk
            * knowledge_gap_score
            * spatial_weight
            * teacher_candidate_region.float()
        ).clamp(0.0, 1.0)
    teacher_release_candidate = teacher_candidate_region & (release_score > 0.0)

    active_fg_pre_spatial = teacher_release_candidate & teacher_fg_ready
    active_fg_pre_budget = active_fg_pre_spatial
    active_bg_pre_budget = teacher_release_candidate & teacher_bg_conf & teacher_bg_safe & (~teacher_fg_ready)
    if rdsi_active_mask is not None:
        active_fg = active_fg_pre_budget
        active_bg = active_bg_pre_budget
    else:
        active_fg = active_fg_pre_spatial
        fg_budget_score = release_score * teacher_fg_ready.float()
        active_fg = _topk_foreground_anchor(
            fg_budget_score,
            active_fg,
            topk_ratio=active_fg_topk_ratio,
            min_pixels=active_fg_topk_min_pixels,
            max_pixels=active_fg_topk_max_pixels,
        )
        active_bg = active_bg_pre_budget
        bg_budget_score = release_score * teacher_bg_prob
        flat_active_fg = active_fg.flatten(1)
        flat_active_bg = active_bg.flatten(1)
        max_bg_pixels = []
        for batch_idx in range(active_bg.shape[0]):
            fg_count = int(flat_active_fg[batch_idx].sum().detach().cpu().item())
            bg_count = int(flat_active_bg[batch_idx].sum().detach().cpu().item())
            if fg_count > 0:
                max_bg_pixels.append(int(math.ceil(float(fg_count) * max(max_bg_fg_ratio, 0.0))))
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
    active = active_fg | active_bg

    region_f = region.float()
    active_f = active.float()
    region_sum = region_f.sum().clamp_min(1.0)
    active_sum = active_f.sum()
    active_fg_mask = active_fg
    active_bg_mask = active_bg
    active_fg_pre_budget_pixels = active_fg_pre_budget.float().sum()
    active_bg_pre_budget_pixels = active_bg_pre_budget.float().sum()
    active_fg_pixels = active_fg_mask.float().sum()
    active_bg_pixels = active_bg_mask.float().sum()
    active_fg_pre_spatial_pixels = active_fg_pre_spatial.float().sum()
    active_fg_hard_spatial_pixels = (active_fg_pre_spatial & spatial_support_gate).float().sum()
    active_total = active_sum.clamp_min(1.0)
    active_fg_ratio = active_fg_pixels / active_total
    active_bg_ratio = active_bg_pixels / active_total
    if float(active_fg_pre_budget_pixels.detach().cpu().item()) > 0.0:
        foreground_budget_ratio = active_fg_pixels / active_fg_pre_budget_pixels.clamp_min(1.0)
    else:
        foreground_budget_ratio = active_fg_pixels * 0.0
    if float(active_bg_pre_budget_pixels.detach().cpu().item()) > 0.0:
        background_budget_ratio = active_bg_pixels / active_bg_pre_budget_pixels.clamp_min(1.0)
    else:
        background_budget_ratio = active_bg_pixels * 0.0
    if float(active_fg_pixels.detach().cpu().item()) > 0.0:
        background_foreground_ratio = active_bg_pixels / active_fg_pixels.clamp_min(1.0)
    else:
        background_foreground_ratio = active_bg_pixels * 0.0
    background_balance_factor = active_fg_pixels / (active_fg_pixels + active_bg_pixels).clamp_min(1.0)
    if float(active_fg_pre_spatial_pixels.detach().cpu().item()) > 0.0:
        active_fg_spatial_keep_ratio = active_fg_hard_spatial_pixels / active_fg_pre_spatial_pixels.clamp_min(1.0)
    else:
        active_fg_spatial_keep_ratio = active_fg_pre_spatial_pixels * 0.0
    seed_fg_context = _dilate_mask(seed_fg_support, teacher_fg_radius)
    active_fg_seed_precision = _masked_ratio(active_fg_mask & seed_fg_support, active_fg_mask)
    active_fg_seed_recall = _masked_ratio(active_fg_mask & seed_fg_support, seed_fg_support)
    active_fg_support_precision = _masked_ratio(active_fg_mask & fg_support, active_fg_mask)
    active_fg_support_recall = _masked_ratio(active_fg_mask & fg_support, fg_support)
    active_fg_candidate_ratio = _masked_ratio(active_fg_mask & candidate_mask, active_fg_mask)
    active_fg_fg_candidate_ratio = _masked_ratio(active_fg_mask & foreground_candidate_mask, active_fg_mask)
    active_fg_near_seed_ratio = _masked_ratio(active_fg_mask & seed_fg_context, active_fg_mask)
    active_fg_pre_budget_seed_precision = _masked_ratio(active_fg_pre_budget & seed_fg_support, active_fg_pre_budget)
    active_fg_pre_budget_candidate_ratio = _masked_ratio(active_fg_pre_budget & candidate_mask, active_fg_pre_budget)
    spatial_loss_scale = _masked_mean(spatial_weight, active_fg_mask).detach()
    if rdsi_active_mask is not None and bool(active_fg_mask.any().detach().cpu().item()):
        lambda_spatial_scale = torch.ones_like(spatial_loss_scale)
    else:
        lambda_spatial_scale = spatial_loss_scale
    foreground_veto = bool(
        rdsi_active_mask is None and skip_background_only and (
            active_fg_pixels.detach().cpu().item() < float(min_fg_pixels)
            or active_fg_ratio.detach().cpu().item() < float(min_fg_ratio)
        )
    )

    profile["candidate_ratio"] = candidate_mask.float().mean().detach()
    profile["region_ratio"] = region_f.mean().detach()
    profile["active_ratio"] = active_f.mean().detach()
    risk_region_f = risk_region.float()
    risk_region_sum = risk_region_f.sum().clamp_min(1.0)
    profile["risk_region_ratio"] = risk_region_f.mean().detach()
    profile["preserve_region_ratio"] = preserve_region.float().mean().detach()
    profile["reject_by_support"] = (support_reject_map.float().sum() / risk_region_sum).detach()
    profile["student_uncertain_ratio"] = ((student_uncertain & region).float().sum() / region_sum).detach()
    profile["teacher_conf_mean"] = (teacher_conf * region_f).sum().detach() / region_sum
    profile["student_conf_mean"] = (student_conf * region_f).sum().detach() / region_sum
    profile["student_entropy_mean"] = (student_entropy * region_f).sum().detach() / region_sum
    profile["teacher_foreground_ratio"] = ((teacher_fg_ready & region).float().sum() / region_sum).detach()
    profile["teacher_reliability"] = teacher_reliability.detach()
    profile["release_factor"] = release_factor.detach()
    profile["release_raw"] = release_raw.detach()
    profile["release_quality"] = release_quality.detach()
    profile["release_min_effective"] = release_min_effective.detach()
    profile["release_prob_gate"] = release_prob_gate.detach()
    profile["release_conf_gate"] = release_conf_gate.detach()
    profile["release_class_gate"] = release_class_gate.detach()
    profile["teacher_core_agreement"] = teacher_core_agreement
    profile["teacher_core_conflict"] = teacher_core_conflict
    profile["teacher_core_valid"] = torch.tensor(1.0 if core_valid else 0.0, device=device)
    profile["teacher_core_conf_mean"] = teacher_core_conf_mean
    profile["teacher_support_agreement"] = teacher_support_agreement
    profile["teacher_support_conflict"] = teacher_support_conflict
    profile["teacher_support_conf_mean"] = teacher_support_conf_mean
    profile["teacher_support_fg_recall"] = teacher_support_fg_recall
    profile["teacher_support_bg_agreement"] = teacher_support_bg_agreement
    profile["teacher_seed_support_agreement"] = teacher_seed_support_agreement
    profile["teacher_seed_support_conflict"] = teacher_seed_support_conflict
    profile["teacher_seed_support_conf_mean"] = teacher_seed_support_conf_mean
    profile["teacher_seed_support_fg_recall"] = teacher_seed_support_fg_recall
    profile["teacher_seed_support_fg_prob_mean"] = teacher_seed_support_fg_prob_mean
    profile["teacher_seed_support_fg_conf_mean"] = teacher_seed_support_fg_conf_mean
    profile["teacher_seed_support_fg_margin_mean"] = teacher_seed_support_fg_margin_mean
    profile["teacher_seed_support_bg_agreement"] = teacher_seed_support_bg_agreement
    profile["seed_fg_support_pixels"] = seed_fg_support_pixels
    profile["core_conflict_veto_ratio"] = torch.tensor(1.0 if teacher_validation_veto else 0.0, device=device)
    if teacher_validation_veto:
        profile["reject_by_core_conflict"] = (
            teacher_candidate_region.float().sum() / risk_region_sum
        ).detach()
    else:
        profile["reject_by_core_conflict"] = teacher_conf.mean().detach() * 0.0
    profile["spatial_support_ratio"] = spatial_support_gate.float().mean().detach()
    profile["foreground_candidate_ratio"] = foreground_candidate_mask.float().mean().detach()
    profile["spatial_weight_mean"] = _masked_mean(spatial_weight, active_fg_mask).detach()
    profile["spatial_weight_candidate_mean"] = _masked_mean(
        spatial_weight,
        active_fg_mask & foreground_candidate_mask,
    ).detach()
    profile["spatial_weight_near_seed_mean"] = _masked_mean(
        spatial_weight,
        active_fg_mask & seed_fg_context,
    ).detach()
    profile["spatial_weight_far_mean"] = _masked_mean(
        spatial_weight,
        active_fg_mask & (~spatial_support_gate),
    ).detach()
    profile["spatial_loss_scale"] = spatial_loss_scale.detach()
    profile["active_foreground_pixels_pre_spatial"] = active_fg_pre_spatial_pixels.detach()
    profile["active_foreground_spatial_keep_ratio"] = active_fg_spatial_keep_ratio.detach()
    profile["foreground_veto_ratio"] = torch.tensor(1.0 if foreground_veto else 0.0, device=device)
    profile["active_foreground_ratio"] = active_fg_ratio.detach()
    profile["active_background_ratio"] = active_bg_ratio.detach()
    profile["active_foreground_pixels_pre_budget"] = active_fg_pre_budget_pixels.detach()
    profile["active_background_pixels_pre_budget"] = active_bg_pre_budget_pixels.detach()
    profile["foreground_budget_ratio"] = foreground_budget_ratio.detach()
    profile["background_budget_ratio"] = background_budget_ratio.detach()
    profile["background_foreground_ratio"] = background_foreground_ratio.detach()
    profile["background_balance_factor"] = background_balance_factor.detach()
    profile["active_foreground_pixels_pre_return"] = active_fg_pixels.detach()
    profile["active_background_pixels_pre_return"] = active_bg_pixels.detach()
    profile["active_foreground_pixels"] = active_fg_pixels.detach()
    profile["active_background_pixels"] = active_bg_pixels.detach()
    profile["active_fg_seed_precision"] = active_fg_seed_precision.detach()
    profile["active_fg_seed_recall"] = active_fg_seed_recall.detach()
    profile["active_fg_support_precision"] = active_fg_support_precision.detach()
    profile["active_fg_support_recall"] = active_fg_support_recall.detach()
    profile["active_fg_candidate_ratio"] = active_fg_candidate_ratio.detach()
    profile["active_fg_fg_candidate_ratio"] = active_fg_fg_candidate_ratio.detach()
    profile["active_fg_near_seed_ratio"] = active_fg_near_seed_ratio.detach()
    profile["active_fg_pre_budget_seed_precision"] = active_fg_pre_budget_seed_precision.detach()
    profile["active_fg_pre_budget_candidate_ratio"] = active_fg_pre_budget_candidate_ratio.detach()
    for class_id in range(1, teacher_prob.shape[1]):
        class_release = ((active_fg & (teacher_pred == class_id)).float().sum() / region_sum).detach()
        profile["teacher_class{}_release_ratio".format(class_id)] = class_release

    if rdsi_active_mask is not None:
        lambda_pre_safety = float(lambda_rgftd)
        core_safety_factor = 1.0
    else:
        lambda_pre_safety = (
            float(lambda_rgftd)
            * float(teacher_reliability.detach().cpu().item())
            * float(release_factor.detach().cpu().item())
        )
    if rdsi_active_mask is None and validation_enabled:
        if core_valid:
            core_conflict_value = float(teacher_core_conflict.detach().cpu().item())
            core_conflict_limit = float(max(getattr(args, "rgftd_teacher_max_core_conflict", 0.20), 1e-6))
            core_safety_factor = max(0.0, min(1.0 - core_conflict_value / core_conflict_limit, 1.0))
        else:
            core_safety_factor = 1.0
    elif rdsi_active_mask is None:
        core_safety_factor = 1.0
    lambda_after_safety = lambda_pre_safety * core_safety_factor
    lambda_effective = lambda_after_safety * float(lambda_spatial_scale.cpu().item())
    if lambda_eff_cap > 0.0:
        lambda_effective = min(lambda_effective, lambda_eff_cap)
    profile["lambda_pre_safety"] = torch.tensor(lambda_pre_safety, device=device)
    profile["lambda_after_safety"] = torch.tensor(lambda_after_safety, device=device)
    profile["core_safety_factor"] = torch.tensor(core_safety_factor, device=device)
    profile["lambda_effective"] = torch.tensor(lambda_effective, device=device)

    return_reason = 0.0
    if float(active_sum.detach().cpu().item()) <= 0.0:
        return_reason = 1.0
    elif foreground_veto:
        return_reason = 2.0
    elif lambda_effective <= 0.0:
        return_reason = 3.0
    profile["return_reason"] = torch.tensor(return_reason, device=device)
    teacher_release_map = active if return_reason <= 0.0 else torch.zeros_like(active, dtype=torch.bool)
    profile["teacher_accept_ratio"] = (
        teacher_release_map.float().sum() / risk_region_sum
    ).detach()
    profile["teacher_reject_ratio"] = (
        (teacher_candidate_region & (~teacher_release_map)).float().sum() / risk_region_sum
    ).detach()
    profile["release_score_mean"] = _masked_mean(release_score, teacher_candidate_region).detach()
    profile["release_score_top"] = _masked_mean(release_score, teacher_release_map).detach()
    profile["teacher_reliable_score"] = _masked_mean(teacher_local_reliability, teacher_candidate_region).detach()
    profile["student_risk_score"] = _masked_mean(student_local_risk, teacher_candidate_region).detach()
    profile["knowledge_gap_score"] = _masked_mean(knowledge_gap_score, teacher_candidate_region).detach()
    profile["selected_gap_mean"] = _masked_mean(knowledge_gap_score, teacher_release_map).detach()
    profile["rejected_gap_mean"] = _masked_mean(
        knowledge_gap_score,
        teacher_candidate_region & (~teacher_release_map),
    ).detach()
    if rdsi_alpha_map is not None:
        profile["rdsi_alpha_mean"] = _masked_mean(rdsi_alpha_map, teacher_release_map).detach()
        profile["rdsi_alpha_top"] = _masked_mean(
            rdsi_alpha_map,
            teacher_release_map & (rdsi_alpha_map >= rdsi_alpha_map.mean()),
        ).detach()
    if rdsi_benefit_map is not None:
        profile["rdsi_benefit_mean"] = _masked_mean(rdsi_benefit_map, teacher_release_map).detach()
        profile["rdsi_benefit_top"] = _masked_mean(
            rdsi_benefit_map,
            teacher_release_map & (rdsi_benefit_map >= rdsi_benefit_map.mean()),
        ).detach()
    if rdsi_score_map is not None:
        profile["rdsi_selected_score_mean"] = _masked_mean(rdsi_score_map, teacher_release_map).detach()
        profile["rdsi_selected_score_top"] = _masked_mean(
            rdsi_score_map,
            teacher_release_map & (rdsi_score_map >= rdsi_score_map.mean()),
        ).detach()
    profile["rdsi_raw_teacher_conf_mean"] = _masked_mean(teacher_conf, teacher_release_map).detach()
    profile["rdsi_raw_teacher_fg_ratio"] = (
        ((teacher_pred > 0) & teacher_release_map).float().sum() / risk_region_sum
    ).detach()
    if foreground_veto:
        profile["reject_by_fg_ratio"] = (
            teacher_candidate_region.float().sum() / risk_region_sum
        ).detach()
    else:
        profile["reject_by_fg_ratio"] = teacher_conf.mean().detach() * 0.0

    if return_reason > 0.0:
        zero_loss = student_logits.sum() * 0.0
        profile["loss"] = zero_loss.detach()
        profile["teacher_active_loss"] = zero_loss.detach()
        return zero_loss, lambda_effective, profile

    refined_target, refine_silent, refine_profile = _build_refined_teacher_target(
        teacher_prob=teacher_prob,
        image=image,
        label=label,
        wann_maps=wann_maps,
        args=args,
        active_fg_mask=active_fg_mask,
        active_bg_mask=active_bg_mask,
        region=region,
        candidate_mask=candidate_mask,
        foreground_candidate_mask=foreground_candidate_mask,
        seed_fg_support=seed_fg_support,
        seed_bg_support=seed_bg_support,
        core_mask=core_mask,
        seed_fg_context=seed_fg_context,
        teacher_fg_prob=teacher_fg_prob,
    )
    profile.update(refine_profile)
    if refine_silent:
        zero_loss = student_logits.sum() * 0.0
        profile["return_reason"] = torch.tensor(4.0, device=device)
        profile["teacher_accept_ratio"] = zero_loss.detach()
        profile["teacher_reject_ratio"] = (
            teacher_candidate_region.float().sum() / risk_region_sum
        ).detach()
        profile["loss"] = zero_loss.detach()
        profile["teacher_active_loss"] = zero_loss.detach()
        return zero_loss, lambda_effective, profile

    if rdsi_alpha_map is not None:
        intervention_alpha = (
            rdsi_alpha_map.clamp(0.0, 1.0)
            * teacher_release_map.float()
        ).unsqueeze(1)
        refined_target = student_prob + intervention_alpha * (refined_target - student_prob)
        refined_target = refined_target.clamp_min(1e-6)
        refined_target = refined_target / refined_target.sum(dim=1, keepdim=True).clamp_min(1e-6)
        target_conf = refined_target.max(dim=1)[0]
        target_entropy = _normalized_entropy(refined_target)
        profile["rdsi_target_conf_mean"] = _masked_mean(target_conf, teacher_release_map).detach()
        profile["rdsi_target_entropy_mean"] = _masked_mean(target_entropy, teacher_release_map).detach()
        if refined_target.shape[1] > 1:
            refined_fg_prob = refined_target[:, 1:].sum(dim=1)
            student_fg_mass = student_prob[:, 1:].sum(dim=1)
            profile["rdsi_q_fg_delta"] = _masked_mean(
                refined_fg_prob - student_fg_mass,
                teacher_release_map,
            ).detach()
        else:
            profile["rdsi_q_fg_delta"] = intervention_alpha.sum().detach() * 0.0

    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    per_pixel_kl = F.kl_div(student_log_prob, refined_target, reduction="none").sum(dim=1) * temperature * temperature
    background_weight = float(getattr(args, "rgftd_background_weight", 0.25))
    if validation_enabled and teacher_prob.shape[1] > 2:
        fg_teacher_weight = (class_weight_map * teacher_conf).clamp(0.0, 1.0)
    else:
        fg_teacher_weight = (0.5 * class_weight_map + 0.5 * teacher_fg_prob).clamp(0.0, 1.0)
    if rdsi_active_mask is not None:
        intervention_region_weight = active.float()
    else:
        intervention_region_weight = (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
    fg_weight = (
        intervention_region_weight
        * active_fg_mask.float()
        * fg_teacher_weight
        * release_score
    )
    bg_suppression = ((bg_max_fg_prob - teacher_fg_prob) / max(bg_max_fg_prob, 1e-6)).clamp(0.0, 1.0)
    bg_weight = (
        intervention_region_weight
        * active_bg_mask.float()
        * release_score
        * bg_suppression
    )
    fg_denom = fg_weight.sum().clamp_min(1.0)
    bg_denom = bg_weight.sum().clamp_min(1.0)
    fg_loss = (per_pixel_kl * fg_weight).sum() / fg_denom
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        bg_loss = (per_pixel_kl * bg_weight).sum() / bg_denom
        bg_loss = bg_loss * background_balance_factor
    else:
        bg_loss = student_logits.sum() * 0.0
    loss = (fg_loss + background_weight * bg_loss) / (1.0 + background_weight if float(active_bg_pixels.detach().cpu().item()) > 0.0 else 1.0)

    profile["loss"] = loss.detach()
    profile["teacher_active_loss"] = (loss.detach() * torch.tensor(lambda_effective, device=device))
    profile["rdsi_loss_raw"] = loss.detach()
    profile["rdsi_loss_weighted"] = profile["teacher_active_loss"].detach()
    profile["kl_mean"] = (per_pixel_kl * active_f).sum().detach() / active_sum.clamp_min(1.0)
    distill_weight = fg_weight + background_weight * bg_weight
    profile["weight_mean"] = distill_weight.mean().detach()
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        profile["background_suppression_mean"] = (
            (bg_suppression * active_bg_mask.float()).sum().detach() / active_bg_pixels.clamp_min(1.0)
        )
    return loss, lambda_effective, profile
