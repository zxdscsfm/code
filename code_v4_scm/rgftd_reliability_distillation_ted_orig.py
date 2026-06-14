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
    "rdsi_selected_spatial_support",
    "rdsi_transfer_compatibility_mean",
    "rdsi_transfer_compatibility_top",
    "rdsi_selected_teacher_mean",
    "rdsi_selected_teacher_switch_ratio",
    "rdsi_teacher_reliable_score",
    "rdsi_knowledge_gap_score",
    "rdsi_risk_region_ratio",
    "rdsi_hard_core_ratio",
    "rdsi_soft_core_ratio",
    "rdsi_fg_deficient_ratio",
    "rdsi_fg_excessive_ratio",
    "rdsi_fg_missing_need",
    "rdsi_fg_excess_need",
    "rdsi_boundary_need",
    "rdsi_candidate_ratio",
    "rdsi_accept_ratio",
    "rdsi_reject_ratio",
    "rdsi_reject_by_core",
    "rdsi_reject_by_seed",
    "rdsi_reject_by_prior",
    "rdsi_reject_by_entropy",
    "rdsi_reject_by_fg_excess",
    "rdsi_reject_by_bg_only",
    "rdsi_reject_by_no_fg_lift",
    "rdsi_foreground_active_ratio",
    "rdsi_background_paired_ratio",
    "rdsi_fg_repair_active_ratio",
    "rdsi_bg_suppress_active_ratio",
    "rdsi_boundary_active_ratio",
    "rdsi_background_pair_ratio",
    "rdsi_fg_repair_score",
    "rdsi_bg_suppress_score",
    "rdsi_boundary_score",
    "rdsi_bg_only_ratio",
    "rdsi_fg_lift_mean",
    "rdsi_fg_lift_top",
    "rdsi_bg_suppression_mean",
    "rdsi_bg_suppression_top",
    "rdsi_benefit_mean",
    "rdsi_benefit_top",
    "rdsi_boundary_support_mean",
    "rdsi_boundary_support_top",
    "rdsi_core_preserving_fg_mean",
    "rdsi_core_preserving_fg_top",
    "rdsi_core_damage_mean",
    "rdsi_core_damage_top",
    "rdsi_seed_conflict_mean",
    "rdsi_unsafe_gap_mean",
    "rdsi_unsafe_gap_top",
    "rdsi_foreground_excess_proxy",
    "rdsi_safe_signal",
    "rdsi_unsafe_signal",
    "rdsi_safe_budget_factor",
    "rdsi_effective_topk_ratio",
    "rdsi_effective_min_pixels",
    "rdsi_core_reopen_ratio",
    "rdsi_core_reopen_active_ratio",
    "rdsi_veto_by_core_damage",
    "rdsi_alpha_mean",
    "rdsi_alpha_top",
    "rdsi_raw_teacher_fg_delta",
    "rdsi_raw_teacher_conf_mean",
    "rdsi_raw_teacher_fg_ratio",
    "rdsi_q_fg_delta",
    "rdsi_fg_repair_q_delta",
    "rdsi_bg_suppress_q_delta",
    "rdsi_boundary_q_delta",
    "rdsi_target_conf_mean",
    "rdsi_target_entropy_mean",
    "rdsi_loss_raw",
    "rdsi_loss_weighted",
    "rdsi_proto_loss",
    "rdsi_proto_weight_mean",
    "rdsi_proto_weight_top",
    "rdsi_proto_cosine",
    "rdsi_proto_region_ratio",
    "rdsi_proto_teacher_entropy",
    "rdsi_proto_teacher_weight_max",
    "rdsi_proto_valid_batches",
    "rdsi_fg_repair_loss",
    "rdsi_bg_suppress_loss",
    "rdsi_boundary_loss",
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
    "rdsi_teacher_scribble_ratio",
    "rdsi_teacher_keypoint_ratio",
    "rdsi_teacher_block_ratio",
    "rdsi_teacher_unknown_ratio",
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
        if torch.is_tensor(topk_ratio):
            batch_topk_ratio = float(topk_ratio[batch_idx].detach().cpu().item())
        elif isinstance(topk_ratio, (list, tuple)):
            batch_topk_ratio = float(topk_ratio[batch_idx])
        else:
            batch_topk_ratio = float(topk_ratio)
        if torch.is_tensor(min_pixels):
            batch_min_pixels = int(math.ceil(float(min_pixels[batch_idx].detach().cpu().item())))
        elif isinstance(min_pixels, (list, tuple)):
            batch_min_pixels = int(math.ceil(float(min_pixels[batch_idx])))
        else:
            batch_min_pixels = int(min_pixels)
        k = max(batch_min_pixels, int(math.ceil(float(valid_count) * batch_topk_ratio)))
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
        if k <= 0:
            continue
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


def _local_range_score(value, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return torch.zeros_like(value)
    kernel = radius * 2 + 1
    x = value.float().unsqueeze(1)
    local_max = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    local_min = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=radius)
    return (local_max[:, 0] - local_min[:, 0]).clamp(0.0, 1.0)


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
    support_mask = raw_support_mask
    seed_support_mask = raw_seed_support_mask
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
    "_rdsi_foreground_active_mask",
    "_rdsi_background_active_mask",
    "_rdsi_fg_repair_active_mask",
    "_rdsi_bg_suppress_active_mask",
    "_rdsi_boundary_active_mask",
    "_rdsi_action_map",
    "_rdsi_residual_alpha_map",
    "_rdsi_teacher_proto_weight_stack",
    "_rdsi_fg_proto_weight_stack",
    "_rdsi_bg_proto_weight_stack",
    "_rdsi_boundary_proto_weight_stack",
    "_rdsi_selected_teacher_stack_index",
    "_rdsi_proto_region_weight",
    "_rdsi_benefit_score",
    "_rdsi_selected_score",
    "_rdsi_fg_repair_score",
    "_rdsi_bg_suppress_score",
    "_rdsi_boundary_score",
    "_rdsi_foreground_lift_score",
    "_rdsi_background_suppression_score",
    "_rdsi_boundary_support",
    "_rdsi_transfer_compatibility",
    "_rdsi_core_preserving_fg_support",
    "_rdsi_core_damage_proxy",
    "_rdsi_seed_conflict",
    "_rdsi_unsafe_gap",
    "_rdsi_foreground_excess_proxy",
    "_rdsi_safe_budget_factor",
    "_rdsi_core_reopen_mask",
    "_rdsi_risk_region",
    "_rdsi_hard_core_mask",
    "_rdsi_soft_core_mask",
]


def _clear_rdsi_runtime_state(args):
    for attr in _RDSI_RUNTIME_ATTRS:
        if hasattr(args, attr):
            delattr(args, attr)


_SUP_TYPE_BUCKETS = ("scribble", "keypoint", "block", "unknown")


def _canonical_rdsi_sup_type(sup_type):
    sup_type = str(sup_type).strip().lower()
    if sup_type == "keypoint":
        return "keypoint"
    if sup_type in ["scribble", "scribble_noisy"] or sup_type.startswith("sparse_scribble_"):
        return "scribble"
    if sup_type in ["box", "block"]:
        return "block"
    return "unknown"


def _rdsi_teacher_sup_type_map(args):
    raw = str(getattr(args, "rdsi_teacher_sup_types", "") or "")
    parts = [part.strip() for part in raw.split(",")]
    return {
        idx: _canonical_rdsi_sup_type(part)
        for idx, part in enumerate(parts)
        if part
    }


def _runtime_map(args, attr, ref, as_bool=False):
    value = getattr(args, attr, None)
    if value is None or not torch.is_tensor(value) or tuple(value.shape) != tuple(ref.shape):
        return None
    value = value.to(device=ref.device)
    if as_bool:
        return value > 0.5 if value.dtype != torch.bool else value
    return value.float()


def _rdsi_transfer_compatibility_stack(student_feature, teacher_feature_list, target_hw, smooth_radius):
    """Build local student-teacher feature compatibility maps for typed prototype transfer."""
    if student_feature is None or teacher_feature_list is None:
        return None
    if not torch.is_tensor(student_feature) or student_feature.dim() != 4:
        raise ValueError("RDSI transfer compatibility expects a 4D student feature map")
    if len(teacher_feature_list) == 0:
        return None

    device = student_feature.device
    student_feature = student_feature.detach().float()
    batch_size, channels, height, width = student_feature.shape
    student_norm = F.normalize(student_feature, dim=1)
    compat_items = []
    for teacher_feature in teacher_feature_list:
        if teacher_feature is None or not torch.is_tensor(teacher_feature) or teacher_feature.dim() != 4:
            raise ValueError("RDSI transfer compatibility expects 4D teacher feature maps")
        teacher_feature = teacher_feature.detach().to(device=device).float()
        if teacher_feature.shape[0] != batch_size or teacher_feature.shape[1] != channels:
            raise ValueError("RDSI transfer compatibility requires matching feature batch/channel dimensions")
        if teacher_feature.shape[-2:] != (height, width):
            teacher_feature = F.interpolate(
                teacher_feature,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        teacher_norm = F.normalize(teacher_feature, dim=1)
        compat = (student_norm * teacher_norm).sum(dim=1).clamp(-1.0, 1.0).clamp_min(0.0)
        radius = int(max(smooth_radius, 0))
        if radius > 0:
            kernel = radius * 2 + 1
            compat = F.avg_pool2d(
                compat.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=radius,
                count_include_pad=False,
            )[:, 0].clamp(0.0, 1.0)
        if tuple(compat.shape[-2:]) != tuple(target_hw):
            compat = F.interpolate(
                compat.unsqueeze(1),
                size=tuple(target_hw),
                mode="bilinear",
                align_corners=False,
            )[:, 0].clamp(0.0, 1.0)
        compat_items.append(compat)
    return torch.stack(compat_items, dim=0)


def _rdsi_intervention_region(wann_maps, student_conf, student_entropy, hard_core_mask, soft_core_mask, args):
    """Return WANN-supported risk pixels eligible for external intervention."""
    low_r_thresh = float(getattr(args, "rgftd_low_r_thresh", getattr(args, "wann_soft_thresh", 0.25)))
    student_conf_thresh = float(getattr(args, "rgftd_student_conf_thresh", 0.80))
    student_entropy_thresh = float(getattr(args, "rgftd_student_entropy_thresh", 0.35))
    zero_bool = torch.zeros_like(hard_core_mask, dtype=torch.bool)
    risk_mask = getattr(wann_maps, "risk_mask", zero_bool)
    low_conflict_mask = getattr(wann_maps, "low_conflict_mask", zero_bool)
    low_reliability_mask = wann_maps.reliability < low_r_thresh
    student_uncertain = (student_conf <= student_conf_thresh) | (student_entropy >= student_entropy_thresh)
    non_core_candidate = (wann_maps.ignore_mask | wann_maps.soft_band | soft_core_mask) & (~hard_core_mask)
    focus_evidence = risk_mask | low_conflict_mask | (student_uncertain & low_reliability_mask)
    return non_core_candidate & focus_evidence


def select_rdsi_teacher_logits(
    student_logits,
    teacher_logits_list,
    teacher_ids,
    label,
    wann_maps,
    args,
    iter_num,
    student_feature=None,
    teacher_feature_list=None,
):
    """Select a region-wise domain-specialist teacher from a cross-client pool."""
    device = student_logits.device
    _clear_rdsi_runtime_state(args)
    profile = {}
    profile["rdsi_enabled"] = torch.tensor(1.0, device=device)
    teacher_sup_type_map = _rdsi_teacher_sup_type_map(args)

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
    candidate_mask = getattr(wann_maps, "candidate_mask", None)
    if candidate_mask is None:
        candidate_mask = (~hard_core_mask) & (wann_maps.reliability >= low_r_thresh)
    candidate_mask = (candidate_mask | soft_core_mask) & (~hard_core_mask)
    non_conflict_mask = ~getattr(wann_maps, "low_conflict_mask", torch.zeros_like(valid_mask, dtype=torch.bool))
    reliable_candidate_mask = candidate_mask & non_conflict_mask & (wann_maps.reliability >= low_r_thresh)

    region = _rdsi_intervention_region(
        wann_maps,
        student_conf,
        student_entropy,
        hard_core_mask,
        soft_core_mask,
        args,
    )

    teacher_fg_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    teacher_fg_prob_thresh = float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))
    teacher_fg_topk_ratio = float(getattr(args, "rgftd_teacher_fg_topk_ratio", 0.002))
    min_fg_pixels = int(getattr(args, "rgftd_min_foreground_pixels", 8))
    teacher_fg_topk_min_pixels = int(getattr(args, "rgftd_teacher_fg_topk_min_pixels", min_fg_pixels))
    teacher_student_fg_margin = float(getattr(args, "rgftd_teacher_student_fg_margin", 0.05))
    teacher_bg_conf_thresh = float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98))
    bg_max_fg_prob = float(getattr(args, "rgftd_bg_max_fg_prob", 0.15))
    max_bg_fg_ratio = float(getattr(args, "rgftd_max_bg_fg_ratio", 1.0))
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
    boundary_radius_arg = int(getattr(args, "rdsi_boundary_radius", -1))
    boundary_radius = max(1, teacher_fg_radius) if boundary_radius_arg < 0 else boundary_radius_arg
    transfer_compatibility_stack = _rdsi_transfer_compatibility_stack(
        student_feature,
        teacher_feature_list,
        student_conf.shape[-2:],
        boundary_radius,
    )
    if transfer_compatibility_stack is None:
        raise ValueError("RDSI transfer compatibility requires student and teacher feature maps")
    else:
        if transfer_compatibility_stack.shape[0] != len(teacher_logits_list):
            raise ValueError("RDSI transfer compatibility must match the teacher logits pool")
        transfer_compatibility_stack = transfer_compatibility_stack.to(
            device=device,
            dtype=student_conf.dtype,
        ).clamp(0.0, 1.0)
    boundary_uncertainty_width = max(float(getattr(args, "rdsi_boundary_uncertainty_width", 0.25)), 1e-6)
    core_damage_veto = max(0.0, min(float(getattr(args, "rdsi_core_damage_veto", 0.30)), 1.0))
    unsafe_gap_scale = max(float(getattr(args, "rdsi_unsafe_gap_scale", 0.40)), 1e-6)
    boundary_weight = max(0.0, float(getattr(args, "rdsi_boundary_support_weight", 0.35)))
    core_preserve_weight = max(0.0, float(getattr(args, "rdsi_core_preserving_fg_weight", 0.25)))
    reliability_weight = max(0.0, float(getattr(args, "rdsi_teacher_reliability_weight", 0.20)))
    risk_weight = max(0.0, float(getattr(args, "rdsi_student_risk_weight", 0.20)))
    core_damage_weight = max(0.0, float(getattr(args, "rdsi_core_damage_weight", 0.45)))
    unsafe_gap_weight = max(0.0, float(getattr(args, "rdsi_unsafe_gap_weight", 0.30)))
    fg_excess_weight = max(0.0, float(getattr(args, "rdsi_foreground_excess_weight", 0.25)))
    safe_budget_gain = max(0.0, float(getattr(args, "rdsi_safe_budget_gain", 1.50)))
    unsafe_budget_decay = max(0.0, float(getattr(args, "rdsi_unsafe_budget_decay", 1.00)))
    max_budget_factor = max(1.0, float(getattr(args, "rdsi_max_budget_factor", 4.0)))

    fg_support = support_mask & (target_label > 0)
    seed_fg_support = seed_support_mask & (target_label > 0)
    seed_fg_spatial_context = _dilate_mask(seed_fg_support, spatial_support_radius)
    fg_support_context = _dilate_mask(fg_support, spatial_support_radius)
    foreground_candidate_mask = reliable_candidate_mask & fg_support_context
    if student_prob.shape[1] > 1:
        student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
    else:
        student_fg_prob = torch.zeros_like(student_conf)
    if student_prob.shape[1] > 1:
        student_fg_mass = student_prob[:, 1:].sum(dim=1)
    else:
        student_fg_mass = torch.zeros_like(student_conf)
    student_boundary_uncertain = (
        1.0 - (student_fg_mass - 0.5).abs() / boundary_uncertainty_width
    ).clamp(0.0, 1.0)
    student_boundary_grad = _local_range_score(student_fg_mass, boundary_radius)
    student_entropy_grad = _local_range_score(student_entropy, boundary_radius)
    wann_boundary_mask = (
        wann_maps.soft_band
        | getattr(wann_maps, "low_conflict_mask", torch.zeros_like(region, dtype=torch.bool))
        | _dilate_mask(seed_fg_support, boundary_radius)
        | _dilate_mask(foreground_candidate_mask, boundary_radius)
    )
    boundary_base = torch.maximum(student_boundary_uncertain, student_boundary_grad)
    boundary_base = torch.maximum(boundary_base, student_entropy)
    boundary_base = torch.maximum(boundary_base, student_entropy_grad)
    boundary_base = torch.maximum(boundary_base, wann_boundary_mask.float())
    boundary_base = boundary_base.clamp(0.0, 1.0)
    student_core_uncertain = (
        (student_conf <= hard_core_conf_thresh)
        | (student_entropy >= hard_core_entropy_thresh)
        | (student_boundary_grad >= float(getattr(args, "rdsi_core_reopen_boundary_floor", 0.20)))
        | getattr(wann_maps, "low_conflict_mask", torch.zeros_like(region, dtype=torch.bool))
    )
    core_reopen_mask = hard_core_mask & student_core_uncertain & valid_mask
    candidate_region = region | core_reopen_mask
    region = candidate_region
    region_f = region.float()
    region_sum = region_f.sum().clamp_min(1.0)
    weak_anchor_mask = seed_support_mask & region
    region_denom = region_f.flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
    if spatial_support_enabled:
        spatial_weight = region.float() * spatial_far_weight
        spatial_weight = torch.maximum(spatial_weight, seed_fg_spatial_context.float() * spatial_near_seed_weight)
        spatial_weight = torch.maximum(spatial_weight, foreground_candidate_mask.float() * spatial_candidate_weight)
        spatial_weight = spatial_weight * region.float()
    else:
        spatial_weight = region.float()
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
    risk_like_mask = (
        getattr(wann_maps, "risk_mask", torch.zeros_like(region, dtype=torch.bool))
        | getattr(wann_maps, "low_conflict_mask", torch.zeros_like(region, dtype=torch.bool))
        | wann_maps.soft_band
        | seed_fg_spatial_context
        | foreground_candidate_mask
        | core_reopen_mask
    )
    typed_support_context = (
        seed_fg_spatial_context
        | foreground_candidate_mask
        | core_reopen_mask
    )
    fg_missing_global_need = (fg_deficit / fg_excess_scale).clamp(0.0, 1.0).expand_as(student_conf)
    fg_missing_need = torch.maximum(
        fg_missing_global_need,
        typed_support_context.float() * (1.0 - student_fg_prob).clamp(0.0, 1.0),
    )
    fg_missing_need = (
        fg_missing_need
        * risk_like_mask.float()
        * region.float()
    ).clamp(0.0, 1.0)
    fg_excess_global_need = (fg_excess_state / fg_excess_scale).clamp(0.0, 1.0).expand_as(student_conf)
    fg_excess_need = torch.maximum(
        fg_excess_global_need,
        (~typed_support_context).float() * student_fg_prob.clamp(0.0, 1.0),
    )
    fg_excess_need = (
        fg_excess_need
        * student_fg_prob.clamp(0.0, 1.0)
        * risk_like_mask.float()
        * region.float()
    ).clamp(0.0, 1.0)
    boundary_need = (
        boundary_base
        * risk_like_mask.float()
        * region.float()
    ).clamp(0.0, 1.0)
    if not bool(region.any().detach().cpu().item()):
        zero_map = torch.zeros_like(student_conf)
        zero_bool = torch.zeros_like(student_conf, dtype=torch.bool)
        zero_proto_weight = torch.zeros(
            (len(teacher_logits_list),) + tuple(student_conf.shape),
            device=device,
            dtype=student_conf.dtype,
        )
        setattr(args, "_rdsi_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_foreground_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_background_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_fg_repair_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_bg_suppress_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_boundary_active_mask", zero_bool.detach())
        setattr(args, "_rdsi_action_map", zero_map.detach())
        setattr(args, "_rdsi_residual_alpha_map", zero_map.detach())
        setattr(args, "_rdsi_teacher_proto_weight_stack", zero_proto_weight.detach())
        setattr(args, "_rdsi_fg_proto_weight_stack", zero_proto_weight.detach())
        setattr(args, "_rdsi_bg_proto_weight_stack", zero_proto_weight.detach())
        setattr(args, "_rdsi_boundary_proto_weight_stack", zero_proto_weight.detach())
        setattr(args, "_rdsi_selected_teacher_stack_index", torch.zeros_like(student_conf, dtype=torch.long).detach())
        setattr(args, "_rdsi_proto_region_weight", zero_map.detach())
        setattr(args, "_rdsi_benefit_score", zero_map.detach())
        setattr(args, "_rdsi_selected_score", zero_map.detach())
        setattr(args, "_rdsi_fg_repair_score", zero_map.detach())
        setattr(args, "_rdsi_bg_suppress_score", zero_map.detach())
        setattr(args, "_rdsi_boundary_score", zero_map.detach())
        setattr(args, "_rdsi_foreground_lift_score", zero_map.detach())
        setattr(args, "_rdsi_background_suppression_score", zero_map.detach())
        setattr(args, "_rdsi_boundary_support", zero_map.detach())
        setattr(args, "_rdsi_transfer_compatibility", zero_map.detach())
        setattr(args, "_rdsi_core_preserving_fg_support", zero_map.detach())
        setattr(args, "_rdsi_core_damage_proxy", zero_map.detach())
        setattr(args, "_rdsi_seed_conflict", zero_map.detach())
        setattr(args, "_rdsi_unsafe_gap", zero_map.detach())
        setattr(args, "_rdsi_foreground_excess_proxy", zero_map.detach())
        setattr(args, "_rdsi_safe_budget_factor", torch.ones(student_conf.shape[0], device=device).detach())
        setattr(args, "_rdsi_core_reopen_mask", zero_bool.detach())
        setattr(args, "_rdsi_risk_region", zero_bool.detach())
        setattr(args, "_rdsi_hard_core_mask", hard_core_mask.detach())
        setattr(args, "_rdsi_soft_core_mask", soft_core_mask.detach())
        profile["rdsi_risk_region_ratio"] = zero_map.mean().detach()
        profile["rdsi_hard_core_ratio"] = hard_core_mask.float().mean().detach()
        profile["rdsi_soft_core_ratio"] = soft_core_mask.float().mean().detach()
        profile["rdsi_fg_missing_need"] = zero_map.mean().detach()
        profile["rdsi_fg_excess_need"] = zero_map.mean().detach()
        profile["rdsi_boundary_need"] = zero_map.mean().detach()
        profile["rdsi_accept_ratio"] = zero_map.mean().detach()
        profile["rdsi_reject_ratio"] = zero_map.mean().detach()
        profile["rdsi_fg_repair_active_ratio"] = zero_map.mean().detach()
        profile["rdsi_bg_suppress_active_ratio"] = zero_map.mean().detach()
        profile["rdsi_boundary_active_ratio"] = zero_map.mean().detach()
        profile["rdsi_background_pair_ratio"] = zero_map.mean().detach()
        profile["rdsi_fg_repair_score"] = zero_map.mean().detach()
        profile["rdsi_bg_suppress_score"] = zero_map.mean().detach()
        profile["rdsi_boundary_score"] = zero_map.mean().detach()
        profile["rdsi_transfer_compatibility_mean"] = zero_map.mean().detach()
        profile["rdsi_transfer_compatibility_top"] = zero_map.mean().detach()
        profile["rdsi_safe_budget_factor"] = zero_map.mean().detach()
        profile["rdsi_effective_topk_ratio"] = zero_map.mean().detach()
        profile["rdsi_effective_min_pixels"] = zero_map.mean().detach()
        return teacher_logits_list[0].detach(), profile

    score_stack_items = []
    benefit_stack_items = []
    reliable_stack_items = []
    gap_stack_items = []
    fg_lift_stack_items = []
    bg_suppression_stack_items = []
    student_risk_stack_items = []
    boundary_support_items = []
    transfer_compatibility_items = []
    core_preserving_fg_items = []
    core_damage_items = []
    seed_conflict_items = []
    unsafe_gap_items = []
    fg_excess_proxy_items = []
    candidate_stack_items = []
    fg_candidate_stack_items = []
    bg_candidate_stack_items = []
    boundary_candidate_stack_items = []
    fg_repair_score_items = []
    bg_suppress_score_items = []
    boundary_score_items = []
    core_reject_items = []
    seed_reject_items = []
    prior_reject_items = []
    entropy_reject_items = []
    fg_excess_reject_items = []
    bg_only_reject_items = []
    no_fg_lift_reject_items = []
    logits_stack_items = []
    teacher_id_values = []
    reliable_means = []
    gap_means = []
    fg_lift_means = []

    for teacher_stack_index, (teacher_id, teacher_logits) in enumerate(zip(teacher_ids, teacher_logits_list)):
        teacher_logits = teacher_logits.detach()
        transfer_compatibility = transfer_compatibility_stack[teacher_stack_index].detach().clamp(0.0, 1.0)
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
        teacher_bg_conf = teacher_bg & (teacher_conf >= teacher_bg_conf_thresh)
        teacher_bg_safe = teacher_fg_prob <= bg_max_fg_prob
        teacher_fg_prob_anchor = (teacher_fg_prob >= teacher_fg_prob_thresh) & region
        teacher_fg_seed = (teacher_fg_conf | teacher_fg_prob_anchor) & region
        teacher_fg_topk_valid = _seed_or_fallback(teacher_fg_seed, region)
        teacher_fg_anchor = _topk_foreground_anchor(
            teacher_fg_prob,
            teacher_fg_topk_valid,
            topk_ratio=teacher_fg_topk_ratio,
            min_pixels=teacher_fg_topk_min_pixels,
        )
        teacher_fg_lift_delta = (teacher_fg_prob - student_fg_prob).clamp_min(0.0)
        teacher_fg_lift_score = (
            teacher_fg_lift_delta / (1.0 - student_fg_prob).clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        teacher_bg_suppression_delta = (student_fg_prob - teacher_fg_prob).clamp_min(0.0)
        teacher_bg_suppression_score = (
            teacher_bg_suppression_delta / student_fg_prob.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        teacher_fg_lift = (teacher_fg_lift_delta >= teacher_student_fg_margin) & region
        teacher_bg_suppression = (
            (teacher_bg_suppression_delta >= teacher_student_fg_margin)
            & (teacher_bg_prob >= teacher_bg_conf_thresh)
            & teacher_bg_safe
            & region
        )
        student_uncertain_for_fg = (
            (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
            | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
        )
        student_foreground_repair = (
            (student_pred == 0)
            | student_uncertain_for_fg
            | (student_fg_prob < teacher_fg_prob_thresh)
            | core_reopen_mask
        ) & region
        teacher_fg_ready = teacher_fg_anchor | teacher_fg_prob_anchor
        teacher_foreground_effective = teacher_fg_ready & teacher_fg_lift & student_foreground_repair
        teacher_background_effective = teacher_bg_conf & teacher_bg_suppression
        teacher_background_only = teacher_background_effective & (~teacher_foreground_effective)
        teacher_fg_context = _dilate_mask(teacher_fg_ready, teacher_fg_radius)
        teacher_bg_context = _dilate_mask(teacher_background_effective, teacher_fg_radius)
        teacher_candidate_region = (
            (teacher_foreground_effective & teacher_fg_context)
            | (teacher_background_effective & teacher_bg_context)
        ) & region

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
        prior_safe = torch.where(
            teacher_fg_ready | teacher_background_effective,
            torch.ones_like(teacher_conf),
            teacher_bg_safe.float(),
        )
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
        prior_benefit_safe = fg_excess_safe * fg_state_safe
        intervention_prior_safe = torch.where(
            teacher_background_effective,
            torch.ones_like(prior_benefit_safe),
            prior_benefit_safe,
        )
        seed_conflict_map = (
            weak_anchor_mask
            & (teacher_pred != target_label)
            & ((teacher_conf >= student_conf) | teacher_fg_ready)
        ).float()
        if student_prob.shape[1] > 1:
            teacher_fg_mass = teacher_prob[:, 1:].sum(dim=1)
            teacher_bg_mass = teacher_prob[:, 0]
            student_bg_mass = student_prob[:, 0]
        else:
            teacher_fg_mass = torch.zeros_like(teacher_conf)
            teacher_bg_mass = torch.zeros_like(teacher_conf)
            student_bg_mass = torch.zeros_like(teacher_conf)
        hard_core_f = hard_core_mask.float()
        core_drop = (
            hard_core_f
            * (target_label > 0).float()
            * (student_fg_mass - teacher_fg_mass).clamp_min(0.0)
        )
        core_bg_lift = (
            hard_core_f
            * (target_label > 0).float()
            * (teacher_bg_mass - student_bg_mass).clamp_min(0.0)
        )
        core_class_flip = (
            hard_core_mask
            & (teacher_pred != target_label)
            & (teacher_conf >= student_conf)
        ).float()
        core_damage_proxy = torch.maximum(core_drop, core_bg_lift)
        core_damage_proxy = torch.maximum(core_damage_proxy, core_class_flip).clamp(0.0, 1.0)
        local_core_damage = _local_range_score(core_damage_proxy, boundary_radius)
        core_damage_proxy = torch.maximum(core_damage_proxy, local_core_damage * core_reopen_mask.float()).clamp(0.0, 1.0)
        core_damage_veto_map = (core_damage_proxy >= core_damage_veto) | (seed_conflict_map > 0.5)
        hard_core_penalty = (core_damage_proxy / max(core_damage_veto, 1e-6)).clamp(0.0, 1.0)
        core_damage_safe = (1.0 - hard_core_penalty).clamp(0.0, 1.0)

        boundary_disagreement = (teacher_student_disagreement * boundary_base).clamp(0.0, 1.0)
        teacher_boundary_grad = _local_range_score(teacher_fg_mass, boundary_radius)
        boundary_alignment = torch.maximum(
            torch.minimum(student_boundary_grad, teacher_boundary_grad),
            boundary_base * wann_boundary_mask.float(),
        ).clamp(0.0, 1.0)
        boundary_support = torch.maximum(boundary_disagreement, boundary_alignment)
        boundary_support = (
            boundary_support
            * region.float()
            * seed_safe
            * core_damage_safe
        ).clamp(0.0, 1.0)
        teacher_local_reliability = (
            teacher_conf
            * (1.0 - teacher_entropy).clamp(0.0, 1.0)
            * weak_anchor_agreement
            * teacher_reliability.detach().clamp(0.0, 1.0)
            * core_consistency.detach()
            * prior_safe
        ).clamp(0.0, 1.0)
        student_risk_evidence = torch.maximum(
            student_entropy,
            (1.0 - student_conf).clamp(0.0, 1.0),
        )
        student_local_risk = (
            (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
            * student_risk_evidence
        ).clamp(0.0, 1.0)
        core_preserving_fg_support = (
            teacher_fg_lift_score
            * teacher_fg_ready.float()
            * student_foreground_repair.float()
            * typed_support_context.float()
            * seed_safe
            * core_damage_safe
        ).clamp(0.0, 1.0)
        bg_suppression_support = (
            teacher_bg_suppression_score
            * teacher_background_effective.float()
            * risk_like_mask.float()
            * seed_safe
            * core_damage_safe
        ).clamp(0.0, 1.0)
        unsafe_gap = (
            knowledge_gap_score
            * (~risk_like_mask).float()
        )
        unsafe_gap = torch.maximum(
            unsafe_gap,
            knowledge_gap_score * hard_core_mask.float() * core_damage_proxy,
        )
        unsafe_gap = (unsafe_gap / unsafe_gap_scale).clamp(0.0, 1.0)
        foreground_excess_proxy = (1.0 - fg_excess_safe).clamp(0.0, 1.0) * teacher_fg_ready.float()
        student_fg_excess_proxy = (fg_excess_state / fg_excess_scale).clamp(0.0, 1.0).expand_as(teacher_conf)
        fg_missing_proxy = torch.maximum(
            (fg_deficit / fg_excess_scale).clamp(0.0, 1.0).expand_as(teacher_conf),
            student_foreground_repair.float() * typed_support_context.float(),
        ).clamp(0.0, 1.0)
        common_negative = (
            core_damage_weight * core_damage_proxy
            + unsafe_gap_weight * unsafe_gap
        )
        fg_negative = common_negative + fg_excess_weight * foreground_excess_proxy
        bg_negative = common_negative
        boundary_negative = common_negative + 0.5 * fg_excess_weight * foreground_excess_proxy
        action_safety = (
            seed_safe
            * entropy_safe
            * core_damage_safe
            * (~core_damage_veto_map).float()
        ).clamp(0.0, 1.0)
        transfer_evidence = transfer_compatibility.detach().clamp(0.0, 1.0)
        reliability_evidence = (0.5 + 0.5 * teacher_local_reliability.detach()).clamp(0.0, 1.0)
        risk_evidence = (0.5 + 0.5 * student_local_risk.detach()).clamp(0.0, 1.0)
        fg_repair_candidate = teacher_foreground_effective & typed_support_context & region & (~core_damage_veto_map)
        bg_suppress_candidate = (
            teacher_background_effective
            & risk_like_mask
            & (student_fg_excess_proxy > 0.0)
            & region
            & (~core_damage_veto_map)
        )
        boundary_candidate = (
            (boundary_support > 0.0)
            & risk_like_mask
            & region
            & (~core_damage_veto_map)
        )
        fg_repair_benefit = (
            core_preserve_weight * core_preserving_fg_support
            + reliability_weight * teacher_local_reliability.detach()
            + risk_weight * student_local_risk.detach()
            + boundary_weight * boundary_support
            - fg_negative
        ).clamp(0.0, 1.0)
        bg_suppress_benefit = (
            core_preserve_weight * bg_suppression_support
            + reliability_weight * teacher_local_reliability.detach()
            + risk_weight * student_local_risk.detach()
            - bg_negative
        ).clamp(0.0, 1.0)
        boundary_benefit = (
            boundary_weight * boundary_support
            + core_preserve_weight * core_preserving_fg_support
            + reliability_weight * teacher_local_reliability.detach()
            + risk_weight * student_local_risk.detach()
            - boundary_negative
        ).clamp(0.0, 1.0)
        fg_repair_score = (
            fg_repair_benefit
            * teacher_fg_lift_score
            * fg_missing_need
            * reliability_evidence
            * risk_evidence
            * spatial_weight
            * action_safety
            * transfer_evidence
            * fg_repair_candidate.float()
        ).clamp(0.0, 1.0)
        bg_suppress_score = (
            bg_suppress_benefit
            * teacher_bg_suppression_score
            * fg_excess_need
            * reliability_evidence
            * risk_evidence
            * spatial_weight
            * action_safety
            * transfer_evidence
            * bg_suppress_candidate.float()
        ).clamp(0.0, 1.0)
        boundary_score = (
            boundary_benefit
            * boundary_support
            * boundary_need
            * reliability_evidence
            * risk_evidence
            * spatial_weight
            * action_safety
            * transfer_evidence
            * boundary_candidate.float()
        ).clamp(0.0, 1.0)
        benefit_positive = (
            boundary_weight * boundary_support
            + core_preserve_weight * core_preserving_fg_support
            + core_preserve_weight * bg_suppression_support
            + reliability_weight * teacher_local_reliability.detach()
            + risk_weight * student_local_risk.detach()
        )
        benefit_negative = common_negative + fg_excess_weight * foreground_excess_proxy
        benefit_score = (benefit_positive - benefit_negative).clamp(0.0, 1.0)
        benefit_score = (
            benefit_score
            * action_safety
            * transfer_evidence
        ).clamp(0.0, 1.0)
        teacher_candidate_region = fg_repair_candidate | bg_suppress_candidate | boundary_candidate
        release_score = torch.maximum(
            torch.maximum(fg_repair_score, bg_suppress_score),
            boundary_score,
        ).clamp(0.0, 1.0)

        score_stack_items.append(release_score)
        benefit_stack_items.append(benefit_score)
        reliable_stack_items.append(teacher_local_reliability)
        gap_stack_items.append(knowledge_gap_score)
        fg_lift_stack_items.append(teacher_fg_lift_score)
        bg_suppression_stack_items.append(teacher_bg_suppression_score)
        student_risk_stack_items.append(student_local_risk)
        boundary_support_items.append(boundary_support)
        transfer_compatibility_items.append(transfer_compatibility)
        core_preserving_fg_items.append(core_preserving_fg_support)
        core_damage_items.append(core_damage_proxy)
        seed_conflict_items.append(seed_conflict_map)
        unsafe_gap_items.append(unsafe_gap)
        fg_excess_proxy_items.append(foreground_excess_proxy)
        candidate_stack_items.append(teacher_candidate_region.float())
        fg_candidate_stack_items.append(fg_repair_candidate.float())
        bg_candidate_stack_items.append(bg_suppress_candidate.float())
        boundary_candidate_stack_items.append(boundary_candidate.float())
        fg_repair_score_items.append(fg_repair_score)
        bg_suppress_score_items.append(bg_suppress_score)
        boundary_score_items.append(boundary_score)
        core_reject_items.append((teacher_candidate_region.float() * torch.maximum(1.0 - core_safe, core_damage_proxy)).clamp(0.0, 1.0))
        seed_reject_items.append((teacher_candidate_region & weak_anchor_mask & (teacher_pred != target_label)).float())
        prior_reject_items.append((teacher_candidate_region.float() * torch.maximum(1.0 - intervention_prior_safe, foreground_excess_proxy)).clamp(0.0, 1.0))
        entropy_reject_items.append((teacher_candidate_region.float() * (1.0 - entropy_safe)).clamp(0.0, 1.0))
        fg_excess_reject_items.append((teacher_candidate_region.float() * (1.0 - fg_excess_safe)).clamp(0.0, 1.0))
        bg_only_reject_items.append(teacher_background_only.float())
        no_fg_lift_reject_items.append((teacher_fg_ready & region & (~teacher_fg_lift)).float())
        logits_stack_items.append(teacher_logits)
        teacher_id_values.append(float(teacher_id))
        reliable_means.append(_masked_mean(teacher_local_reliability, teacher_candidate_region).detach())
        gap_means.append(_masked_mean(knowledge_gap_score, teacher_candidate_region).detach())
        fg_lift_means.append(_masked_mean(teacher_fg_lift_score, teacher_candidate_region).detach())

    score_stack = torch.stack(score_stack_items, dim=0)
    benefit_stack = torch.stack(benefit_stack_items, dim=0)
    reliable_stack = torch.stack(reliable_stack_items, dim=0)
    gap_stack = torch.stack(gap_stack_items, dim=0)
    fg_lift_stack = torch.stack(fg_lift_stack_items, dim=0)
    bg_suppression_stack = torch.stack(bg_suppression_stack_items, dim=0)
    student_risk_stack = torch.stack(student_risk_stack_items, dim=0)
    boundary_support_stack = torch.stack(boundary_support_items, dim=0)
    transfer_compatibility_stack = torch.stack(transfer_compatibility_items, dim=0)
    core_preserving_fg_stack = torch.stack(core_preserving_fg_items, dim=0)
    core_damage_stack = torch.stack(core_damage_items, dim=0)
    seed_conflict_stack = torch.stack(seed_conflict_items, dim=0)
    unsafe_gap_stack = torch.stack(unsafe_gap_items, dim=0)
    fg_excess_proxy_stack = torch.stack(fg_excess_proxy_items, dim=0)
    candidate_stack = torch.stack(candidate_stack_items, dim=0)
    fg_candidate_stack = torch.stack(fg_candidate_stack_items, dim=0)
    bg_candidate_stack = torch.stack(bg_candidate_stack_items, dim=0)
    boundary_candidate_stack = torch.stack(boundary_candidate_stack_items, dim=0)
    fg_repair_score_stack = torch.stack(fg_repair_score_items, dim=0)
    bg_suppress_score_stack = torch.stack(bg_suppress_score_items, dim=0)
    boundary_score_stack = torch.stack(boundary_score_items, dim=0)
    core_reject_stack = torch.stack(core_reject_items, dim=0)
    seed_reject_stack = torch.stack(seed_reject_items, dim=0)
    prior_reject_stack = torch.stack(prior_reject_items, dim=0)
    entropy_reject_stack = torch.stack(entropy_reject_items, dim=0)
    fg_excess_reject_stack = torch.stack(fg_excess_reject_items, dim=0)
    bg_only_reject_stack = torch.stack(bg_only_reject_items, dim=0)
    no_fg_lift_reject_stack = torch.stack(no_fg_lift_reject_items, dim=0)
    logits_stack = torch.stack(logits_stack_items, dim=0)
    best_score, best_index = score_stack.max(dim=0)
    teacher_id_tensor = torch.tensor(teacher_id_values, device=device, dtype=torch.float32)

    raw_active_mask = (candidate_stack.max(dim=0).values > 0.5) & region
    raw_active_f = raw_active_mask.float()
    raw_active_denom = raw_active_f.flatten(1).sum(dim=1).clamp_min(1.0)
    safe_signal = (
        (
            boundary_support_stack.max(dim=0).values
            + core_preserving_fg_stack.max(dim=0).values
            + benefit_stack.max(dim=0).values
            + transfer_compatibility_stack.max(dim=0).values
        )
        * raw_active_f
    ).flatten(1).sum(dim=1) / (4.0 * raw_active_denom)
    unsafe_signal = (
        (
            core_damage_stack.max(dim=0).values
            + seed_conflict_stack.max(dim=0).values
            + unsafe_gap_stack.max(dim=0).values
            + fg_excess_proxy_stack.max(dim=0).values
        )
        * raw_active_f
    ).flatten(1).sum(dim=1) / (4.0 * raw_active_denom)
    safe_budget_factor = (
        1.0
        + safe_budget_gain * safe_signal
        - unsafe_budget_decay * unsafe_signal
    ).clamp(0.0, max_budget_factor)
    effective_topk_ratio = (benefit_topk_ratio * safe_budget_factor).clamp(0.0, 1.0)
    effective_min_pixels = torch.ceil(
        torch.tensor(float(max(benefit_topk_min_pixels, 0)), device=device) * safe_budget_factor
    )
    if benefit_topk_max_pixels > 0:
        effective_max_pixels = torch.ceil(
            torch.tensor(float(benefit_topk_max_pixels), device=device) * safe_budget_factor
        )
    else:
        effective_max_pixels = None
    fg_proto_score, fg_proto_index = fg_repair_score_stack.max(dim=0)
    fg_proto_candidate = fg_candidate_stack.gather(0, fg_proto_index.unsqueeze(0)).squeeze(0) > 0.5
    bg_proto_score_raw, bg_proto_index_raw = bg_suppress_score_stack.max(dim=0)
    bg_proto_candidate_raw = bg_candidate_stack.gather(0, bg_proto_index_raw.unsqueeze(0)).squeeze(0) > 0.5
    bg_proto_score = bg_proto_score_raw
    bg_proto_index = bg_proto_index_raw
    bg_proto_candidate = bg_proto_candidate_raw
    boundary_proto_score, boundary_proto_index = boundary_score_stack.max(dim=0)
    boundary_proto_candidate = boundary_candidate_stack.gather(0, boundary_proto_index.unsqueeze(0)).squeeze(0) > 0.5

    fg_proto_pre_mask = (fg_proto_score > score_floor) & fg_proto_candidate & region
    bg_proto_pre_mask = (bg_proto_score > score_floor) & bg_proto_candidate & region
    boundary_proto_pre_mask = (boundary_proto_score > score_floor) & boundary_proto_candidate & region
    typed_candidate_mask = fg_proto_pre_mask | bg_proto_pre_mask | boundary_proto_pre_mask
    fg_action_score = fg_proto_score * fg_proto_pre_mask.float()
    bg_action_score = bg_proto_score * bg_proto_pre_mask.float()
    boundary_action_score = boundary_proto_score * boundary_proto_pre_mask.float()

    fg_need_mass = (fg_missing_need * fg_proto_pre_mask.float()).flatten(1).sum(dim=1)
    bg_need_mass = (fg_excess_need * bg_proto_pre_mask.float()).flatten(1).sum(dim=1)
    boundary_need_mass = (boundary_need * boundary_proto_pre_mask.float()).flatten(1).sum(dim=1)
    typed_need_mass = (fg_need_mass + bg_need_mass + boundary_need_mass).clamp_min(1e-6)
    fg_budget_share = (fg_need_mass / typed_need_mass).clamp(0.0, 1.0)
    bg_budget_share = (bg_need_mass / typed_need_mass).clamp(0.0, 1.0)
    boundary_budget_share = (boundary_need_mass / typed_need_mass).clamp(0.0, 1.0)

    def _typed_budget(max_pixels, share):
        if max_pixels is None:
            return None
        return torch.ceil(max_pixels.float() * share)

    fg_budget_mask = _topk_foreground_anchor(
        fg_action_score,
        fg_proto_pre_mask,
        topk_ratio=effective_topk_ratio * fg_budget_share,
        min_pixels=torch.ceil(effective_min_pixels.float() * fg_budget_share),
        max_pixels=_typed_budget(effective_max_pixels, fg_budget_share),
    ) & fg_proto_pre_mask
    bg_budget_mask = _topk_foreground_anchor(
        bg_action_score,
        bg_proto_pre_mask,
        topk_ratio=effective_topk_ratio * bg_budget_share,
        min_pixels=torch.ceil(effective_min_pixels.float() * bg_budget_share),
        max_pixels=_typed_budget(effective_max_pixels, bg_budget_share),
    ) & bg_proto_pre_mask
    boundary_budget_mask = _topk_foreground_anchor(
        boundary_action_score,
        boundary_proto_pre_mask,
        topk_ratio=effective_topk_ratio * boundary_budget_share,
        min_pixels=torch.ceil(effective_min_pixels.float() * boundary_budget_share),
        max_pixels=_typed_budget(effective_max_pixels, boundary_budget_share),
    ) & boundary_proto_pre_mask

    fg_budget_score = fg_action_score * fg_budget_mask.float()
    bg_budget_score = bg_action_score * bg_budget_mask.float()
    boundary_budget_score = boundary_action_score * boundary_budget_mask.float()
    fg_action_need = fg_missing_need * fg_budget_mask.float()
    bg_action_need = fg_excess_need * bg_budget_mask.float()
    boundary_action_need = boundary_need * boundary_budget_mask.float()
    fg_repair_active_mask = (
        fg_budget_mask
        & (fg_action_need >= bg_action_need)
        & (fg_action_need >= boundary_action_need)
    )
    bg_suppress_active_mask = (
        bg_budget_mask
        & (bg_action_need > fg_action_need)
        & (bg_action_need >= boundary_action_need)
    )
    boundary_active_mask = (
        boundary_budget_mask
        & (boundary_action_need > fg_action_need)
        & (boundary_action_need > bg_action_need)
    )
    typed_active_mask = fg_repair_active_mask | bg_suppress_active_mask | boundary_active_mask
    foreground_active_mask = fg_repair_active_mask
    bg_suppress_direct_active_mask = bg_suppress_active_mask
    background_active_mask = bg_suppress_direct_active_mask
    active_mask = foreground_active_mask | background_active_mask | boundary_active_mask

    background_pair_score_items = []
    for teacher_stack_index, teacher_id in enumerate(teacher_id_values):
        teacher_fg_active = fg_repair_active_mask & (teacher_id_tensor[fg_proto_index] == float(teacher_id))
        teacher_fg_alpha = (
            residual_alpha_max
            * fg_proto_score
            * teacher_fg_active.float()
        ).clamp(0.0, residual_alpha_max)
        teacher_alpha_context = F.max_pool2d(
            teacher_fg_alpha.unsqueeze(1),
            kernel_size=teacher_fg_radius * 2 + 1,
            stride=1,
            padding=teacher_fg_radius,
        )[:, 0] if teacher_fg_radius > 0 else teacher_fg_alpha
        teacher_fg_active_context = _dilate_mask(teacher_fg_active, teacher_fg_radius)
        teacher_prob_for_bg = F.softmax(logits_stack[teacher_stack_index] / temperature, dim=1)
        teacher_conf_for_bg = teacher_prob_for_bg.max(dim=1)[0]
        teacher_pred_for_bg = teacher_prob_for_bg.argmax(dim=1)
        if teacher_prob_for_bg.shape[1] > 1:
            teacher_fg_prob_for_bg = teacher_prob_for_bg[:, 1:].max(dim=1)[0]
        else:
            teacher_fg_prob_for_bg = torch.zeros_like(teacher_conf_for_bg)
        teacher_bg_prob_for_bg = teacher_prob_for_bg[:, 0]
        teacher_bg_pair = (
            teacher_fg_active_context
            & region
            & (~foreground_active_mask)
            & (~background_active_mask)
            & (~boundary_active_mask)
            & (teacher_pred_for_bg == 0)
            & (teacher_conf_for_bg >= teacher_bg_conf_thresh)
            & (teacher_fg_prob_for_bg <= bg_max_fg_prob)
        )
        teacher_bg_core_damage = core_damage_stack[teacher_stack_index]
        teacher_bg_seed_conflict = seed_conflict_stack[teacher_stack_index]
        teacher_bg_unsafe_gap = unsafe_gap_stack[teacher_stack_index]
        teacher_bg_pair = (
            teacher_bg_pair
            & (teacher_bg_core_damage < core_damage_veto)
            & (teacher_bg_seed_conflict <= 0.5)
        )
        teacher_bg_safe_score = (
            (1.0 - teacher_bg_core_damage.clamp(0.0, 1.0))
            * (1.0 - teacher_bg_seed_conflict.clamp(0.0, 1.0))
            * (1.0 - teacher_bg_unsafe_gap.clamp(0.0, 1.0))
        ).clamp(0.0, 1.0)
        background_pair_score_items.append(
            teacher_bg_prob_for_bg
            * teacher_alpha_context
            * teacher_bg_pair.float()
            * teacher_bg_safe_score
        )
    background_pair_score_stack = torch.stack(background_pair_score_items, dim=0)
    background_pair_score, background_pair_index = background_pair_score_stack.max(dim=0)
    raw_background_pair_mask = (
        (background_pair_score > 0.0)
        & region
        & (~foreground_active_mask)
        & (~background_active_mask)
        & (~boundary_active_mask)
    )
    flat_fg_active = foreground_active_mask.flatten(1)
    flat_bg_pair = raw_background_pair_mask.flatten(1)
    max_bg_pixels = []
    for batch_idx in range(raw_background_pair_mask.shape[0]):
        fg_count = int(flat_fg_active[batch_idx].sum().detach().cpu().item())
        bg_count = int(flat_bg_pair[batch_idx].sum().detach().cpu().item())
        if fg_count > 0 and bg_count > 0:
            max_bg_pixels.append(int(math.ceil(float(fg_count) * max(max_bg_fg_ratio, 0.0))))
        else:
            max_bg_pixels.append(0)
    if max(max_bg_pixels) <= 0:
        background_pair_active_mask = torch.zeros_like(foreground_active_mask, dtype=torch.bool)
    else:
        background_pair_active_mask = _topk_foreground_anchor(
            background_pair_score,
            raw_background_pair_mask,
            topk_ratio=1.0,
            min_pixels=0,
            max_pixels=max_bg_pixels,
        ) & raw_background_pair_mask
    background_active_mask = bg_suppress_direct_active_mask | background_pair_active_mask
    bg_suppress_active_mask = background_active_mask
    active_mask = foreground_active_mask | background_active_mask | boundary_active_mask

    typed_selected_stack_index = torch.where(
        fg_repair_active_mask,
        fg_proto_index,
        torch.where(
            background_pair_active_mask,
            background_pair_index,
            torch.where(bg_suppress_direct_active_mask, bg_proto_index, boundary_proto_index),
        ),
    )
    selected_stack_index = torch.where(active_mask, typed_selected_stack_index, best_index).detach()
    typed_score_map = torch.where(
        fg_repair_active_mask,
        fg_proto_score,
        torch.where(
            background_pair_active_mask,
            background_pair_score,
            torch.where(bg_suppress_direct_active_mask, bg_proto_score, boundary_proto_score),
        ),
    )
    selected_score_map = torch.where(active_mask, typed_score_map, best_score).detach()
    fg_repair_alpha = (
        residual_alpha_max
        * fg_proto_score
        * fg_repair_active_mask.float()
    ).clamp(0.0, residual_alpha_max)
    bg_suppress_alpha = (
        residual_alpha_max
        * bg_proto_score
        * bg_suppress_direct_active_mask.float()
    ).clamp(0.0, residual_alpha_max)
    bg_suppress_alpha = torch.where(
        background_pair_active_mask,
        (background_pair_score * background_pair_active_mask.float()).clamp(0.0, residual_alpha_max),
        bg_suppress_alpha,
    )
    boundary_alpha = (
        residual_alpha_max
        * boundary_proto_score
        * boundary_active_mask.float()
    ).clamp(0.0, residual_alpha_max)
    residual_alpha = (fg_repair_alpha + bg_suppress_alpha + boundary_alpha).clamp(0.0, residual_alpha_max)
    selected_action_map = torch.zeros_like(best_score)
    selected_action_map = torch.where(fg_repair_active_mask, torch.ones_like(selected_action_map), selected_action_map)
    selected_action_map = torch.where(
        bg_suppress_active_mask,
        torch.ones_like(selected_action_map) * 2.0,
        selected_action_map,
    )
    selected_action_map = torch.where(
        boundary_active_mask,
        torch.ones_like(selected_action_map) * 3.0,
        selected_action_map,
    )
    typed_gather_index = selected_stack_index.unsqueeze(0).unsqueeze(2).expand(
        1,
        logits_stack.shape[1],
        logits_stack.shape[2],
        logits_stack.shape[3],
        logits_stack.shape[4],
    )
    selected_logits = logits_stack.gather(0, typed_gather_index).squeeze(0).detach()
    selected_teacher_id = teacher_id_tensor[selected_stack_index]

    def _gather_selected_map(stack):
        return stack.gather(0, selected_stack_index.unsqueeze(0)).squeeze(0).detach()

    selected_benefit = _gather_selected_map(benefit_stack)
    selected_reliable = _gather_selected_map(reliable_stack)
    selected_gap = _gather_selected_map(gap_stack)
    selected_fg_lift = _gather_selected_map(fg_lift_stack)
    selected_bg_suppression = _gather_selected_map(bg_suppression_stack)
    selected_student_risk = _gather_selected_map(student_risk_stack)
    selected_boundary_support = _gather_selected_map(boundary_support_stack)
    selected_transfer_compatibility = _gather_selected_map(transfer_compatibility_stack)
    selected_core_preserving_fg = _gather_selected_map(core_preserving_fg_stack)
    selected_core_damage = _gather_selected_map(core_damage_stack)
    selected_seed_conflict = _gather_selected_map(seed_conflict_stack)
    selected_unsafe_gap = _gather_selected_map(unsafe_gap_stack)
    selected_fg_excess_proxy = _gather_selected_map(fg_excess_proxy_stack)
    selected_core_reject = _gather_selected_map(core_reject_stack)
    selected_seed_reject = _gather_selected_map(seed_reject_stack)
    selected_prior_reject = _gather_selected_map(prior_reject_stack)
    selected_entropy_reject = _gather_selected_map(entropy_reject_stack)
    selected_fg_excess_reject = _gather_selected_map(fg_excess_reject_stack)
    selected_fg_repair_score = fg_proto_score.detach()
    selected_bg_suppress_score = torch.where(
        background_pair_active_mask,
        background_pair_score,
        bg_proto_score,
    ).detach()
    selected_boundary_score = boundary_proto_score.detach()
    raw_active_mask = active_mask

    def _scatter_proto_weight(index, weight):
        scattered = torch.zeros_like(score_stack)
        scattered.scatter_(0, index.unsqueeze(0), weight.unsqueeze(0))
        return scattered

    fg_proto_weight_stack = _scatter_proto_weight(
        fg_proto_index,
        (fg_proto_score * fg_repair_active_mask.float()).detach(),
    )
    bg_proto_weight_stack = _scatter_proto_weight(
        bg_proto_index,
        (bg_proto_score * bg_suppress_direct_active_mask.float()).detach(),
    )
    bg_pair_proto_weight_stack = _scatter_proto_weight(
        background_pair_index,
        (background_pair_score * background_pair_active_mask.float()).detach(),
    )
    bg_proto_weight_stack = bg_proto_weight_stack + bg_pair_proto_weight_stack
    boundary_proto_weight_stack = _scatter_proto_weight(
        boundary_proto_index,
        (boundary_proto_score * boundary_active_mask.float()).detach(),
    )
    proto_weight_stack = fg_proto_weight_stack + bg_proto_weight_stack + boundary_proto_weight_stack
    proto_score_mass = proto_weight_stack.sum(dim=0).detach()
    selected_teacher_prob = F.softmax(selected_logits / temperature, dim=1)

    active_sum = active_mask.float().sum().clamp_min(1.0)
    candidate_any = (candidate_stack.max(dim=0).values > 0.5) & region
    bg_only_any = (bg_only_reject_stack.max(dim=0).values > 0.5) & region
    no_fg_lift_any = (no_fg_lift_reject_stack.max(dim=0).values > 0.5) & region
    if score_stack.shape[0] > 1:
        top2 = torch.topk(score_stack, k=2, dim=0).values
        best_vs_second = (top2[0] - top2[1]).clamp_min(0.0)
    else:
        best_vs_second = best_score
    profile["rdsi_teacher_compete_count"] = torch.tensor(float(len(teacher_logits_list)), device=device)
    profile["rdsi_candidate_teacher_count"] = torch.tensor(float(len(teacher_logits_list)), device=device)
    profile["rdsi_risk_region_ratio"] = region.float().mean().detach()
    profile["rdsi_hard_core_ratio"] = hard_core_mask.float().mean().detach()
    profile["rdsi_soft_core_ratio"] = soft_core_mask.float().mean().detach()
    profile["rdsi_fg_deficient_ratio"] = (fg_deficient_region & region).float().mean().detach()
    profile["rdsi_fg_excessive_ratio"] = (fg_excessive_region & region).float().mean().detach()
    profile["rdsi_fg_missing_need"] = _masked_mean(fg_missing_need, region).detach()
    profile["rdsi_fg_excess_need"] = _masked_mean(fg_excess_need, region).detach()
    profile["rdsi_boundary_need"] = _masked_mean(boundary_need, region).detach()
    profile["rdsi_candidate_ratio"] = (candidate_any.float().sum() / region_sum).detach()
    profile["rdsi_accept_ratio"] = (active_mask.float().sum() / region_sum).detach()
    profile["rdsi_reject_ratio"] = ((candidate_any & (~active_mask)).float().sum() / region_sum).detach()
    profile["rdsi_reject_by_core"] = (selected_core_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_seed"] = (selected_seed_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_prior"] = (selected_prior_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_entropy"] = (selected_entropy_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_fg_excess"] = (selected_fg_excess_reject.sum() / region_sum).detach()
    profile["rdsi_reject_by_bg_only"] = (bg_only_any.float().sum() / region_sum).detach()
    profile["rdsi_reject_by_no_fg_lift"] = (no_fg_lift_any.float().sum() / region_sum).detach()
    profile["rdsi_multi_teacher_active"] = active_mask.float().mean().detach()
    profile["rdsi_foreground_active_ratio"] = foreground_active_mask.float().mean().detach()
    profile["rdsi_background_paired_ratio"] = background_pair_active_mask.float().mean().detach()
    profile["rdsi_fg_repair_active_ratio"] = fg_repair_active_mask.float().mean().detach()
    profile["rdsi_bg_suppress_active_ratio"] = bg_suppress_active_mask.float().mean().detach()
    profile["rdsi_boundary_active_ratio"] = boundary_active_mask.float().mean().detach()
    profile["rdsi_background_pair_ratio"] = background_pair_active_mask.float().mean().detach()
    profile["rdsi_bg_only_ratio"] = torch.zeros_like(profile["rdsi_multi_teacher_active"]).detach()
    profile["rdsi_best_vs_second_gap"] = _masked_mean(best_vs_second, active_mask).detach()
    profile["rdsi_selected_score_mean"] = _masked_mean(selected_score_map, active_mask).detach()
    profile["rdsi_selected_score_top"] = _masked_mean(
        selected_score_map,
        active_mask & (selected_score_map >= selected_score_map.mean()),
    ).detach()
    profile["rdsi_score_benefit_mean"] = _masked_mean(
        selected_score_map,
        active_mask,
    ).detach()
    profile["rdsi_selected_teacher_reliable"] = _masked_mean(selected_reliable, active_mask).detach()
    profile["rdsi_selected_teacher_benefit"] = _masked_mean(selected_score_map, active_mask).detach()
    profile["rdsi_selected_teacher_gap"] = _masked_mean(selected_gap, active_mask).detach()
    profile["rdsi_selected_student_risk"] = _masked_mean(selected_student_risk, active_mask).detach()
    profile["rdsi_selected_spatial_support"] = _masked_mean(spatial_weight, active_mask).detach()
    selected_transfer_compatibility_mean = _masked_mean(
        selected_transfer_compatibility,
        active_mask,
    )
    profile["rdsi_transfer_compatibility_mean"] = selected_transfer_compatibility_mean.detach()
    profile["rdsi_transfer_compatibility_top"] = _masked_mean(
        selected_transfer_compatibility,
        active_mask & (selected_transfer_compatibility >= selected_transfer_compatibility_mean),
    ).detach()
    profile["rdsi_fg_repair_score"] = _masked_mean(selected_fg_repair_score, fg_repair_active_mask).detach()
    profile["rdsi_bg_suppress_score"] = _masked_mean(selected_bg_suppress_score, bg_suppress_active_mask).detach()
    profile["rdsi_boundary_score"] = _masked_mean(selected_boundary_score, boundary_active_mask).detach()
    profile["rdsi_fg_lift_mean"] = _masked_mean(selected_fg_lift, fg_repair_active_mask).detach()
    profile["rdsi_fg_lift_top"] = _masked_mean(
        selected_fg_lift,
        fg_repair_active_mask & (selected_fg_lift >= selected_fg_lift.mean()),
    ).detach()
    profile["rdsi_bg_suppression_mean"] = _masked_mean(selected_bg_suppression, bg_suppress_active_mask).detach()
    profile["rdsi_bg_suppression_top"] = _masked_mean(
        selected_bg_suppression,
        bg_suppress_active_mask & (selected_bg_suppression >= selected_bg_suppression.mean()),
    ).detach()
    profile["rdsi_benefit_mean"] = _masked_mean(selected_score_map, active_mask).detach()
    profile["rdsi_benefit_top"] = _masked_mean(
        selected_score_map,
        active_mask & (selected_score_map >= selected_score_map.mean()),
    ).detach()
    profile["rdsi_boundary_support_mean"] = _masked_mean(selected_boundary_support, boundary_active_mask).detach()
    profile["rdsi_boundary_support_top"] = _masked_mean(
        selected_boundary_support,
        boundary_active_mask & (selected_boundary_support >= selected_boundary_support.mean()),
    ).detach()
    profile["rdsi_core_preserving_fg_mean"] = _masked_mean(selected_core_preserving_fg, fg_repair_active_mask).detach()
    profile["rdsi_core_preserving_fg_top"] = _masked_mean(
        selected_core_preserving_fg,
        fg_repair_active_mask & (selected_core_preserving_fg >= selected_core_preserving_fg.mean()),
    ).detach()
    profile["rdsi_core_damage_mean"] = _masked_mean(selected_core_damage, raw_active_mask).detach()
    profile["rdsi_core_damage_top"] = _masked_mean(selected_core_damage, foreground_active_mask).detach()
    profile["rdsi_seed_conflict_mean"] = _masked_mean(selected_seed_conflict, raw_active_mask).detach()
    profile["rdsi_unsafe_gap_mean"] = _masked_mean(selected_unsafe_gap, raw_active_mask).detach()
    profile["rdsi_unsafe_gap_top"] = _masked_mean(selected_unsafe_gap, active_mask).detach()
    profile["rdsi_foreground_excess_proxy"] = _masked_mean(selected_fg_excess_proxy, raw_active_mask).detach()
    profile["rdsi_safe_signal"] = safe_signal.mean().detach()
    profile["rdsi_unsafe_signal"] = unsafe_signal.mean().detach()
    profile["rdsi_safe_budget_factor"] = safe_budget_factor.mean().detach()
    profile["rdsi_effective_topk_ratio"] = effective_topk_ratio.mean().detach()
    profile["rdsi_effective_min_pixels"] = effective_min_pixels.float().mean().detach()
    profile["rdsi_core_reopen_ratio"] = core_reopen_mask.float().mean().detach()
    profile["rdsi_core_reopen_active_ratio"] = (foreground_active_mask & core_reopen_mask).float().mean().detach()
    candidate_veto_map = (
        ((core_damage_stack >= core_damage_veto) | (seed_conflict_stack > 0.5)).float()
        * candidate_stack
    ).max(dim=0).values > 0.5
    profile["rdsi_veto_by_core_damage"] = _masked_ratio(
        candidate_veto_map,
        candidate_any,
    ).detach()
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
    zero_ratio = active_sum.detach() * 0.0
    sup_type_ratios = {bucket: zero_ratio for bucket in _SUP_TYPE_BUCKETS}
    if bool(active_mask.any().detach().cpu().item()):
        hist_values = []
        for teacher_id in teacher_id_values:
            teacher_ratio = (((selected_teacher_id == float(teacher_id)) & active_mask).float().sum() / active_sum).detach()
            hist_values.append(teacher_ratio)
            int_teacher_id = int(teacher_id)
            if 0 <= int_teacher_id <= 9:
                profile["rdsi_teacher{}_ratio".format(int_teacher_id)] = teacher_ratio
            sup_type = teacher_sup_type_map.get(int_teacher_id, "unknown")
            sup_type_ratios[sup_type] = sup_type_ratios.get(sup_type, zero_ratio) + teacher_ratio
        if hist_values:
            profile["rdsi_selected_teacher_switch_ratio"] = (1.0 - torch.stack(hist_values).max()).detach()
    for sup_type, sup_ratio in sup_type_ratios.items():
        profile["rdsi_teacher_{}_ratio".format(sup_type)] = sup_ratio.detach()
    if reliable_means:
        profile["rdsi_teacher_reliable_score"] = torch.stack(reliable_means).mean().detach()
    if gap_means:
        profile["rdsi_knowledge_gap_score"] = torch.stack(gap_means).mean().detach()
    if fg_lift_means:
        profile["rdsi_fg_lift_mean"] = _masked_mean(selected_fg_lift, fg_repair_active_mask).detach()
    setattr(args, "_rdsi_active_mask", active_mask.detach())
    setattr(args, "_rdsi_foreground_active_mask", foreground_active_mask.detach())
    setattr(args, "_rdsi_background_active_mask", background_active_mask.detach())
    setattr(args, "_rdsi_fg_repair_active_mask", fg_repair_active_mask.detach())
    setattr(args, "_rdsi_bg_suppress_active_mask", bg_suppress_active_mask.detach())
    setattr(args, "_rdsi_boundary_active_mask", boundary_active_mask.detach())
    setattr(args, "_rdsi_action_map", selected_action_map.detach())
    setattr(args, "_rdsi_residual_alpha_map", residual_alpha.detach())
    setattr(args, "_rdsi_teacher_proto_weight_stack", proto_weight_stack.detach())
    setattr(args, "_rdsi_fg_proto_weight_stack", fg_proto_weight_stack.detach())
    setattr(args, "_rdsi_bg_proto_weight_stack", bg_proto_weight_stack.detach())
    setattr(args, "_rdsi_boundary_proto_weight_stack", boundary_proto_weight_stack.detach())
    setattr(args, "_rdsi_selected_teacher_stack_index", selected_stack_index.detach())
    setattr(args, "_rdsi_proto_region_weight", proto_score_mass.detach())
    setattr(args, "_rdsi_benefit_score", selected_score_map.detach())
    setattr(args, "_rdsi_selected_score", selected_score_map.detach())
    setattr(args, "_rdsi_fg_repair_score", selected_fg_repair_score.detach())
    setattr(args, "_rdsi_bg_suppress_score", selected_bg_suppress_score.detach())
    setattr(args, "_rdsi_boundary_score", selected_boundary_score.detach())
    setattr(args, "_rdsi_foreground_lift_score", selected_fg_lift.detach())
    setattr(args, "_rdsi_background_suppression_score", selected_bg_suppression.detach())
    setattr(args, "_rdsi_boundary_support", selected_boundary_support.detach())
    setattr(args, "_rdsi_transfer_compatibility", selected_transfer_compatibility.detach())
    setattr(args, "_rdsi_core_preserving_fg_support", selected_core_preserving_fg.detach())
    setattr(args, "_rdsi_core_damage_proxy", selected_core_damage.detach())
    setattr(args, "_rdsi_seed_conflict", selected_seed_conflict.detach())
    setattr(args, "_rdsi_unsafe_gap", selected_unsafe_gap.detach())
    setattr(args, "_rdsi_foreground_excess_proxy", selected_fg_excess_proxy.detach())
    setattr(args, "_rdsi_safe_budget_factor", safe_budget_factor.detach())
    setattr(args, "_rdsi_core_reopen_mask", core_reopen_mask.detach())
    setattr(args, "_rdsi_risk_region", region.detach())
    setattr(args, "_rdsi_hard_core_mask", hard_core_mask.detach())
    setattr(args, "_rdsi_soft_core_mask", soft_core_mask.detach())
    return selected_logits, profile


def rdsi_feature_prototype_loss(student_feature, teacher_feature_list, args, iter_num):
    """Transfer RDSI-selected regional knowledge through feature prototypes."""
    device = student_feature.device
    zero_loss = student_feature.sum() * 0.0
    profile = zero_rgftd_profile(device)
    profile["rdsi_enabled"] = torch.tensor(1.0, device=device)

    if teacher_feature_list is None or len(teacher_feature_list) == 0:
        raise ValueError("RDSI feature-prototype transfer requires a non-empty teacher feature list")

    fg_weight_stack = getattr(args, "_rdsi_fg_proto_weight_stack", None)
    bg_weight_stack = getattr(args, "_rdsi_bg_proto_weight_stack", None)
    boundary_weight_stack = getattr(args, "_rdsi_boundary_proto_weight_stack", None)
    if not all(torch.is_tensor(x) for x in [fg_weight_stack, bg_weight_stack, boundary_weight_stack]):
        raise ValueError("RDSI typed prototype weights must be produced before feature-prototype transfer")
    for typed_weight_stack in [fg_weight_stack, bg_weight_stack, boundary_weight_stack]:
        if typed_weight_stack.dim() != 4 or typed_weight_stack.shape[0] != len(teacher_feature_list):
            raise ValueError("RDSI typed prototype weight stack does not match teacher feature list")

    if student_feature.dim() != 4:
        raise ValueError("RDSI feature-prototype transfer expects a 4D student feature map")

    batch_size, channels, height, width = student_feature.shape
    teacher_features = []
    for teacher_feature in teacher_feature_list:
        if teacher_feature is None or not torch.is_tensor(teacher_feature) or teacher_feature.dim() != 4:
            raise ValueError("RDSI feature-prototype transfer expects 4D teacher feature maps")
        teacher_feature = teacher_feature.detach().to(device=device, dtype=student_feature.dtype)
        if teacher_feature.shape[0] != batch_size or teacher_feature.shape[1] != channels:
            raise ValueError("RDSI teacher feature batch/channel dimensions must match student features")
        if teacher_feature.shape[-2:] != (height, width):
            teacher_feature = F.interpolate(
                teacher_feature,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        teacher_features.append(teacher_feature)

    def _prepare_weight_stack(weight_stack, name):
        weight_stack = weight_stack.detach().to(device=device, dtype=student_feature.dtype).clamp_min(0.0)
        if weight_stack.shape[1] != batch_size:
            raise ValueError("RDSI {} batch dimension must match student features".format(name))
        if weight_stack.shape[-2:] != (height, width):
            flat_weight = weight_stack.reshape(-1, 1, weight_stack.shape[-2], weight_stack.shape[-1])
            flat_weight = F.interpolate(flat_weight, size=(height, width), mode="bilinear", align_corners=False)
            weight_stack = flat_weight.reshape(weight_stack.shape[0], batch_size, height, width).clamp_min(0.0)
        return weight_stack

    fg_weight_stack = _prepare_weight_stack(fg_weight_stack, "foreground prototype weights")
    bg_weight_stack = _prepare_weight_stack(bg_weight_stack, "background prototype weights")
    boundary_weight_stack = _prepare_weight_stack(boundary_weight_stack, "boundary prototype weights")
    weight_stack = fg_weight_stack + bg_weight_stack + boundary_weight_stack

    region_weight = weight_stack.sum(dim=0).clamp_min(0.0)
    region_mass = region_weight.flatten(1).sum(dim=1)
    valid_batch = region_mass > 1e-6

    lambda_raw = float(get_rgftd_lambda(iter_num, args))
    lambda_cap = float(getattr(args, "rgftd_lambda_eff_cap", lambda_raw))
    if lambda_cap > 0.0:
        lambda_effective = min(lambda_raw, lambda_cap)
    else:
        lambda_effective = lambda_raw
    profile["lambda"] = torch.tensor(lambda_raw, device=device)
    profile["lambda_pre_safety"] = torch.tensor(lambda_raw, device=device)
    profile["lambda_after_safety"] = torch.tensor(lambda_effective, device=device)
    profile["lambda_effective"] = torch.tensor(lambda_effective, device=device)

    if not bool(valid_batch.any().detach().cpu().item()):
        profile["lambda_after_safety"] = torch.tensor(0.0, device=device)
        profile["lambda_effective"] = torch.tensor(0.0, device=device)
        return zero_loss, 0.0, profile

    teacher_stack = torch.stack(teacher_features, dim=0)
    teacher_stack = F.normalize(teacher_stack, dim=2)
    student_norm = F.normalize(student_feature, dim=1)

    def _prototype_term(action_weight_stack):
        action_region_weight = action_weight_stack.sum(dim=0).clamp_min(0.0)
        action_mass = action_region_weight.flatten(1).sum(dim=1)
        action_valid = action_mass > 1e-6
        if not bool(action_valid.any().detach().cpu().item()):
            zero_cosine = torch.zeros(batch_size, device=device, dtype=student_feature.dtype)
            return zero_loss, zero_cosine, action_valid, action_region_weight, action_mass
        action_teacher_weight = action_weight_stack / action_region_weight.unsqueeze(0).clamp_min(1e-6)
        action_teacher_mix = (teacher_stack * action_teacher_weight.unsqueeze(2)).sum(dim=0).detach()
        action_teacher_mix = F.normalize(action_teacher_mix, dim=1)
        action_region_weight_4d = action_region_weight.unsqueeze(1)
        action_denom = action_mass.clamp_min(1e-6).view(batch_size, 1)
        action_positive_pixels = (action_region_weight > 0.0).float().flatten(1).sum(dim=1).clamp_min(1.0)
        action_strength = (action_mass / action_positive_pixels).clamp(0.0, 1.0)
        action_student_proto = (student_norm * action_region_weight_4d).flatten(2).sum(dim=2) / action_denom
        action_teacher_proto = (action_teacher_mix * action_region_weight_4d).flatten(2).sum(dim=2) / action_denom
        action_student_proto = F.normalize(action_student_proto, dim=1)
        action_teacher_proto = F.normalize(action_teacher_proto.detach(), dim=1)
        action_cosine = (action_student_proto * action_teacher_proto).sum(dim=1).clamp(-1.0, 1.0)
        action_valid_float = action_valid.float()
        action_loss = (
            (1.0 - action_cosine) * action_strength * action_valid_float
        ).sum() / action_valid_float.sum().clamp_min(1.0)
        return action_loss, action_cosine, action_valid, action_region_weight, action_mass

    fg_loss, fg_cosine, fg_valid, fg_region_weight, fg_mass = _prototype_term(fg_weight_stack)
    bg_loss, bg_cosine, bg_valid, bg_region_weight, bg_mass = _prototype_term(bg_weight_stack)
    boundary_loss, boundary_cosine, boundary_valid, boundary_region_weight, boundary_mass = _prototype_term(
        boundary_weight_stack
    )

    action_losses = []
    action_cosines = []
    for action_loss, action_cosine, action_valid in [
        (fg_loss, fg_cosine, fg_valid),
        (bg_loss, bg_cosine, bg_valid),
        (boundary_loss, boundary_cosine, boundary_valid),
    ]:
        if bool(action_valid.any().detach().cpu().item()):
            action_losses.append(action_loss)
            valid_f = action_valid.float()
            action_cosines.append((action_cosine * valid_f).sum() / valid_f.sum().clamp_min(1.0))
    loss = torch.stack(action_losses).mean()
    valid_float = valid_batch.float()
    proto_cosine = torch.stack(action_cosines).mean().detach()

    teacher_mix_weight = weight_stack / region_weight.unsqueeze(0).clamp_min(1e-6)
    teacher_prob = teacher_mix_weight.clamp_min(1e-6)
    teacher_prob = teacher_prob / teacher_prob.sum(dim=0, keepdim=True).clamp_min(1e-6)
    if teacher_prob.shape[0] > 1:
        teacher_entropy = -(teacher_prob * torch.log(teacher_prob)).sum(dim=0)
        teacher_entropy = teacher_entropy / math.log(float(teacher_prob.shape[0]))
    else:
        teacher_entropy = torch.zeros_like(region_weight)
    teacher_weight_max = teacher_prob.max(dim=0).values
    positive_region = region_weight > 0.0

    profile["loss"] = loss.detach()
    profile["lambda"] = torch.tensor(lambda_raw, device=device)
    profile["lambda_pre_safety"] = torch.tensor(lambda_raw, device=device)
    profile["lambda_effective"] = torch.tensor(lambda_effective, device=device)
    profile["lambda_after_safety"] = torch.tensor(lambda_effective, device=device)
    profile["teacher_active_loss"] = (loss * lambda_effective).detach()
    profile["rdsi_loss_raw"] = loss.detach()
    profile["rdsi_loss_weighted"] = profile["teacher_active_loss"].detach()
    profile["rdsi_proto_loss"] = loss.detach()
    profile["rdsi_proto_weight_mean"] = region_weight.mean().detach()
    profile["rdsi_proto_weight_top"] = _masked_mean(region_weight, positive_region).detach()
    profile["rdsi_proto_cosine"] = proto_cosine.detach()
    profile["rdsi_proto_region_ratio"] = positive_region.float().mean().detach()
    profile["rdsi_proto_teacher_entropy"] = _masked_mean(teacher_entropy, positive_region).detach()
    profile["rdsi_proto_teacher_weight_max"] = _masked_mean(teacher_weight_max, positive_region).detach()
    profile["rdsi_proto_valid_batches"] = valid_float.mean().detach()
    profile["rdsi_fg_repair_loss"] = fg_loss.detach()
    profile["rdsi_bg_suppress_loss"] = bg_loss.detach()
    profile["rdsi_boundary_loss"] = boundary_loss.detach()
    profile["region_ratio"] = positive_region.float().mean().detach()
    profile["active_ratio"] = positive_region.float().mean().detach()
    risk_region = _runtime_map(args, "_rdsi_risk_region", region_weight, as_bool=True)
    if risk_region is not None:
        profile["risk_region_ratio"] = risk_region.float().mean().detach()
    else:
        profile["risk_region_ratio"] = positive_region.float().mean().detach()
    return loss, lambda_effective, profile


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
    preserve_core_mask = core_mask & (~active_fg_mask) & (~active_bg_mask)
    core_anchor_radius = int(getattr(args, "rgftd_refine_core_anchor_radius", 1))
    core_anchor = preserve_core_mask & _dilate_mask(support_roi, core_anchor_radius)
    refine_roi = support_roi | core_anchor | seed_fg_support
    q = teacher_prob.detach().clone()
    supported_fg_region = seed_fg_context | foreground_candidate_mask | active_fg_mask
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

    if preserve_core_mask.any():
        core_q = _one_hot_label(label, num_classes, q)
        q = torch.where(preserve_core_mask.unsqueeze(1), core_q, q)

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
    if preserve_core_mask.any():
        q_core_conflict = _masked_ratio((q_pred != label) & preserve_core_mask, preserve_core_mask)
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
    support_mask = raw_support_mask
    seed_support_mask = raw_seed_support_mask
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

    region = _rdsi_intervention_region(
        wann_maps,
        student_conf,
        student_entropy,
        core_mask,
        soft_core_mask,
        args,
    )
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
    student_uncertain_for_fg = (
        (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
        | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
    )
    student_foreground_repair = (
        (student_pred == 0)
        | student_uncertain_for_fg
        | (student_fg_prob < teacher_fg_prob_thresh)
    ) & region
    teacher_fg_lift_score = (
        (teacher_fg_prob - student_fg_prob).clamp_min(0.0)
        / (1.0 - student_fg_prob).clamp_min(1e-6)
    ).clamp(0.0, 1.0)
    teacher_fg_repair_ready = (
        teacher_fg_ready
        & ((teacher_fg_prob - student_fg_prob) >= teacher_student_fg_margin)
        & student_foreground_repair
    )
    teacher_accept = teacher_fg_repair_ready | (teacher_bg_conf & teacher_bg_safe)
    teacher_fg_context = _dilate_mask(teacher_fg_ready, teacher_fg_radius)
    teacher_bg_context = teacher_bg_conf & teacher_bg_safe
    rdsi_active_mask = _runtime_map(args, "_rdsi_active_mask", teacher_conf, as_bool=True)
    rdsi_foreground_active_mask = _runtime_map(args, "_rdsi_foreground_active_mask", teacher_conf, as_bool=True)
    rdsi_background_active_mask = _runtime_map(args, "_rdsi_background_active_mask", teacher_conf, as_bool=True)
    rdsi_fg_repair_active_mask = _runtime_map(args, "_rdsi_fg_repair_active_mask", teacher_conf, as_bool=True)
    rdsi_bg_suppress_active_mask = _runtime_map(args, "_rdsi_bg_suppress_active_mask", teacher_conf, as_bool=True)
    rdsi_boundary_active_mask = _runtime_map(args, "_rdsi_boundary_active_mask", teacher_conf, as_bool=True)
    rdsi_action_map = _runtime_map(args, "_rdsi_action_map", teacher_conf)
    rdsi_alpha_map = _runtime_map(args, "_rdsi_residual_alpha_map", teacher_conf)
    rdsi_benefit_map = _runtime_map(args, "_rdsi_benefit_score", teacher_conf)
    rdsi_score_map = _runtime_map(args, "_rdsi_selected_score", teacher_conf)
    rdsi_fg_repair_score_map = _runtime_map(args, "_rdsi_fg_repair_score", teacher_conf)
    rdsi_bg_suppress_score_map = _runtime_map(args, "_rdsi_bg_suppress_score", teacher_conf)
    rdsi_boundary_score_map = _runtime_map(args, "_rdsi_boundary_score", teacher_conf)
    rdsi_fg_lift_map = _runtime_map(args, "_rdsi_foreground_lift_score", teacher_conf)
    rdsi_bg_suppression_map = _runtime_map(args, "_rdsi_background_suppression_score", teacher_conf)
    rdsi_boundary_support_map = _runtime_map(args, "_rdsi_boundary_support", teacher_conf)
    rdsi_core_preserving_fg_map = _runtime_map(args, "_rdsi_core_preserving_fg_support", teacher_conf)
    rdsi_core_damage_map = _runtime_map(args, "_rdsi_core_damage_proxy", teacher_conf)
    rdsi_seed_conflict_map = _runtime_map(args, "_rdsi_seed_conflict", teacher_conf)
    rdsi_unsafe_gap_map = _runtime_map(args, "_rdsi_unsafe_gap", teacher_conf)
    rdsi_fg_excess_proxy_map = _runtime_map(args, "_rdsi_foreground_excess_proxy", teacher_conf)
    rdsi_core_reopen_mask = _runtime_map(args, "_rdsi_core_reopen_mask", teacher_conf, as_bool=True)
    if rdsi_core_reopen_mask is not None:
        region = region | rdsi_core_reopen_mask
        risk_region = risk_region | rdsi_core_reopen_mask
        student_foreground_repair = student_foreground_repair | rdsi_core_reopen_mask
        teacher_fg_repair_ready = teacher_fg_repair_ready | (
            rdsi_core_reopen_mask
            & teacher_fg_ready
            & ((teacher_fg_prob - student_fg_prob) >= teacher_student_fg_margin)
        )
        teacher_accept = teacher_fg_repair_ready | (teacher_bg_conf & teacher_bg_safe)
    if rdsi_active_mask is not None:
        if rdsi_fg_repair_active_mask is None:
            rdsi_fg_repair_active_mask = (
                rdsi_foreground_active_mask & region
                if rdsi_foreground_active_mask is not None
                else rdsi_active_mask & region
            )
        if rdsi_bg_suppress_active_mask is None:
            rdsi_bg_suppress_active_mask = (
                rdsi_background_active_mask & region
                if rdsi_background_active_mask is not None
                else torch.zeros_like(region, dtype=torch.bool)
            )
        if rdsi_boundary_active_mask is None:
            rdsi_boundary_active_mask = torch.zeros_like(region, dtype=torch.bool)
        rdsi_foreground_region = (
            rdsi_foreground_active_mask & region
            if rdsi_foreground_active_mask is not None
            else (rdsi_fg_repair_active_mask | rdsi_boundary_active_mask) & region
        )
        rdsi_background_region = (
            rdsi_background_active_mask & region
            if rdsi_background_active_mask is not None
            else rdsi_bg_suppress_active_mask & region
        )
        teacher_candidate_region = (rdsi_foreground_region | rdsi_background_region) & region
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
    foreground_correction = teacher_fg_repair_ready
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
    student_risk_evidence = torch.maximum(
        student_entropy,
        (1.0 - student_conf).clamp(0.0, 1.0),
    )
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
        if rdsi_background_active_mask is not None and rdsi_alpha_map is not None:
            alpha_floor = max(float(getattr(args, "rdsi_residual_alpha", 0.35)), 1e-6)
            paired_bg_score = (rdsi_alpha_map / alpha_floor).clamp(0.0, 1.0)
            paired_bg_mask = rdsi_background_active_mask & teacher_candidate_region
            release_score = torch.where(
                paired_bg_mask,
                torch.maximum(release_score, paired_bg_score),
                release_score,
            )
    else:
        release_score = (
            teacher_local_reliability
            * student_local_risk
            * teacher_fg_lift_score
            * spatial_weight
            * teacher_candidate_region.float()
        ).clamp(0.0, 1.0)
    teacher_release_candidate = teacher_candidate_region & (release_score > 0.0)

    if rdsi_active_mask is not None:
        active_fg_pre_spatial = teacher_release_candidate & (
            rdsi_fg_repair_active_mask | rdsi_boundary_active_mask
        )
        active_fg_pre_budget = active_fg_pre_spatial
        active_bg_pre_budget = teacher_release_candidate & rdsi_bg_suppress_active_mask
        active_fg = active_fg_pre_budget
        active_bg = active_bg_pre_budget
    else:
        active_fg_pre_spatial = teacher_release_candidate & teacher_fg_ready
        active_fg_pre_budget = active_fg_pre_spatial
        active_bg_pre_budget = teacher_release_candidate & teacher_bg_conf & teacher_bg_safe & (~teacher_fg_ready)
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
    if rdsi_active_mask is not None:
        active_fg_repair_mask = active & rdsi_fg_repair_active_mask
        active_bg_suppress_mask = active & rdsi_bg_suppress_active_mask
        active_boundary_mask = active & rdsi_boundary_active_mask
        active_fg_mask = active_fg_repair_mask | active_boundary_mask
        active_bg_mask = active_bg_suppress_mask
    else:
        active_fg_repair_mask = active_fg_mask
        active_bg_suppress_mask = active_bg_mask
        active_boundary_mask = torch.zeros_like(active, dtype=torch.bool)
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
    if rdsi_active_mask is not None:
        profile["rdsi_foreground_active_ratio"] = active_fg_mask.float().mean().detach()
        profile["rdsi_background_paired_ratio"] = active_bg_mask.float().mean().detach()
        profile["rdsi_fg_repair_active_ratio"] = active_fg_repair_mask.float().mean().detach()
        profile["rdsi_bg_suppress_active_ratio"] = active_bg_suppress_mask.float().mean().detach()
        profile["rdsi_boundary_active_ratio"] = active_boundary_mask.float().mean().detach()
        profile["rdsi_bg_only_ratio"] = (
            (active_bg_mask & (~active_fg_mask)).float().sum() / active_sum.clamp_min(1.0)
        ).detach() if float(active_fg_pixels.detach().cpu().item()) <= 0.0 else teacher_conf.mean().detach() * 0.0
        if rdsi_fg_repair_score_map is not None:
            profile["rdsi_fg_repair_score"] = _masked_mean(rdsi_fg_repair_score_map, active_fg_repair_mask).detach()
        if rdsi_bg_suppress_score_map is not None:
            profile["rdsi_bg_suppress_score"] = _masked_mean(rdsi_bg_suppress_score_map, active_bg_suppress_mask).detach()
        if rdsi_boundary_score_map is not None:
            profile["rdsi_boundary_score"] = _masked_mean(rdsi_boundary_score_map, active_boundary_mask).detach()
        if rdsi_fg_lift_map is not None:
            profile["rdsi_fg_lift_mean"] = _masked_mean(rdsi_fg_lift_map, active_fg_repair_mask).detach()
            profile["rdsi_fg_lift_top"] = _masked_mean(
                rdsi_fg_lift_map,
                active_fg_repair_mask & (rdsi_fg_lift_map >= rdsi_fg_lift_map.mean()),
            ).detach()
        if rdsi_bg_suppression_map is not None:
            profile["rdsi_bg_suppression_mean"] = _masked_mean(rdsi_bg_suppression_map, active_bg_suppress_mask).detach()
            profile["rdsi_bg_suppression_top"] = _masked_mean(
                rdsi_bg_suppression_map,
                active_bg_suppress_mask & (rdsi_bg_suppression_map >= rdsi_bg_suppression_map.mean()),
            ).detach()
        if rdsi_boundary_support_map is not None:
            profile["rdsi_boundary_support_mean"] = _masked_mean(rdsi_boundary_support_map, active_boundary_mask).detach()
            profile["rdsi_boundary_support_top"] = _masked_mean(
                rdsi_boundary_support_map,
                active_boundary_mask & (rdsi_boundary_support_map >= rdsi_boundary_support_map.mean()),
            ).detach()
        if rdsi_core_preserving_fg_map is not None:
            profile["rdsi_core_preserving_fg_mean"] = _masked_mean(rdsi_core_preserving_fg_map, active_fg_repair_mask).detach()
            profile["rdsi_core_preserving_fg_top"] = _masked_mean(
                rdsi_core_preserving_fg_map,
                active_fg_repair_mask & (rdsi_core_preserving_fg_map >= rdsi_core_preserving_fg_map.mean()),
            ).detach()
        if rdsi_core_damage_map is not None:
            profile["rdsi_core_damage_mean"] = _masked_mean(rdsi_core_damage_map, teacher_candidate_region).detach()
            profile["rdsi_core_damage_top"] = _masked_mean(rdsi_core_damage_map, active_fg_mask).detach()
            profile["rdsi_veto_by_core_damage"] = _masked_ratio(
                rdsi_core_damage_map >= float(getattr(args, "rdsi_core_damage_veto", 0.30)),
                teacher_candidate_region,
            ).detach()
        if rdsi_seed_conflict_map is not None:
            profile["rdsi_seed_conflict_mean"] = _masked_mean(rdsi_seed_conflict_map, teacher_candidate_region).detach()
        if rdsi_unsafe_gap_map is not None:
            profile["rdsi_unsafe_gap_mean"] = _masked_mean(rdsi_unsafe_gap_map, teacher_candidate_region).detach()
            profile["rdsi_unsafe_gap_top"] = _masked_mean(rdsi_unsafe_gap_map, active_fg_mask).detach()
        if rdsi_fg_excess_proxy_map is not None:
            profile["rdsi_foreground_excess_proxy"] = _masked_mean(rdsi_fg_excess_proxy_map, teacher_candidate_region).detach()
        if rdsi_core_reopen_mask is not None:
            profile["rdsi_core_reopen_ratio"] = rdsi_core_reopen_mask.float().mean().detach()
            profile["rdsi_core_reopen_active_ratio"] = (active_fg_mask & rdsi_core_reopen_mask).float().mean().detach()
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
    if rdsi_boundary_support_map is not None:
        profile["rdsi_boundary_support_mean"] = _masked_mean(rdsi_boundary_support_map, teacher_release_map).detach()
        profile["rdsi_boundary_support_top"] = _masked_mean(
            rdsi_boundary_support_map,
            teacher_release_map & (rdsi_boundary_support_map >= rdsi_boundary_support_map.mean()),
        ).detach()
    if rdsi_core_preserving_fg_map is not None:
        profile["rdsi_core_preserving_fg_mean"] = _masked_mean(rdsi_core_preserving_fg_map, teacher_release_map).detach()
        profile["rdsi_core_preserving_fg_top"] = _masked_mean(
            rdsi_core_preserving_fg_map,
            teacher_release_map & (rdsi_core_preserving_fg_map >= rdsi_core_preserving_fg_map.mean()),
        ).detach()
    if rdsi_core_damage_map is not None:
        profile["rdsi_core_damage_mean"] = _masked_mean(rdsi_core_damage_map, teacher_candidate_region).detach()
        profile["rdsi_core_damage_top"] = _masked_mean(rdsi_core_damage_map, teacher_release_map).detach()
    if rdsi_seed_conflict_map is not None:
        profile["rdsi_seed_conflict_mean"] = _masked_mean(rdsi_seed_conflict_map, teacher_candidate_region).detach()
    if rdsi_unsafe_gap_map is not None:
        profile["rdsi_unsafe_gap_mean"] = _masked_mean(rdsi_unsafe_gap_map, teacher_candidate_region).detach()
        profile["rdsi_unsafe_gap_top"] = _masked_mean(rdsi_unsafe_gap_map, teacher_release_map).detach()
    if rdsi_fg_excess_proxy_map is not None:
        profile["rdsi_foreground_excess_proxy"] = _masked_mean(rdsi_fg_excess_proxy_map, teacher_candidate_region).detach()
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
        alpha_map = rdsi_alpha_map.clamp(0.0, 1.0) * teacher_release_map.float()
        intervention_alpha = alpha_map.unsqueeze(1)
        if refined_target.shape[1] > 1:
            raw_refined_target = refined_target
            student_fg_mass = student_prob[:, 1:].sum(dim=1)
            target_fg_mass = raw_refined_target[:, 1:].sum(dim=1)
            raw_fg_delta = target_fg_mass - student_fg_mass
            fg_repair_map = active_fg_repair_mask & teacher_release_map
            bg_suppress_map = active_bg_suppress_mask & teacher_release_map
            boundary_map = active_boundary_mask & teacher_release_map
            allowed_fg_delta = (
                raw_fg_delta.clamp_min(0.0) * fg_repair_map.float()
                + raw_fg_delta.clamp_max(0.0) * bg_suppress_map.float()
                + raw_fg_delta * boundary_map.float()
            )
            new_fg_mass = (student_fg_mass + alpha_map * allowed_fg_delta).clamp(1e-6, 1.0 - 1e-6)
            target_fg_dist = raw_refined_target[:, 1:] / target_fg_mass.unsqueeze(1).clamp_min(1e-6)
            student_fg_dist = student_prob[:, 1:] / student_fg_mass.unsqueeze(1).clamp_min(1e-6)
            active_delta = allowed_fg_delta.abs() > 1e-8
            fg_dist = torch.where(
                (allowed_fg_delta >= 0.0).unsqueeze(1),
                target_fg_dist,
                student_fg_dist,
            )
            fg_dist = torch.where(active_delta.unsqueeze(1), fg_dist, student_fg_dist)
            refined_target = student_prob.clone()
            refined_target[:, 1:] = new_fg_mass.unsqueeze(1) * fg_dist
            refined_target[:, 0] = 1.0 - new_fg_mass
            refined_target = torch.where(
                teacher_release_map.unsqueeze(1),
                refined_target,
                student_prob,
            )
            profile["rdsi_fg_repair_q_delta"] = _masked_mean(
                new_fg_mass - student_fg_mass,
                fg_repair_map,
            ).detach()
            profile["rdsi_bg_suppress_q_delta"] = _masked_mean(
                new_fg_mass - student_fg_mass,
                bg_suppress_map,
            ).detach()
            profile["rdsi_boundary_q_delta"] = _masked_mean(
                new_fg_mass - student_fg_mass,
                boundary_map,
            ).detach()
        else:
            refined_target = student_prob + intervention_alpha * (refined_target - student_prob)
            profile["rdsi_fg_repair_q_delta"] = intervention_alpha.sum().detach() * 0.0
            profile["rdsi_bg_suppress_q_delta"] = intervention_alpha.sum().detach() * 0.0
            profile["rdsi_boundary_q_delta"] = intervention_alpha.sum().detach() * 0.0
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
    if rdsi_active_mask is not None:
        rdsi_alpha_weight = rdsi_alpha_map.clamp(0.0, 1.0) if rdsi_alpha_map is not None else torch.ones_like(teacher_conf)
        rdsi_benefit_weight = rdsi_benefit_map.clamp(0.0, 1.0) if rdsi_benefit_map is not None else release_score
        rdsi_reliable_weight = teacher_local_reliability.clamp(0.0, 1.0)
        bg_suppression = ((bg_max_fg_prob - teacher_fg_prob) / max(bg_max_fg_prob, 1e-6)).clamp(0.0, 1.0)
        rdsi_loss_weight = (
            intervention_region_weight
            * torch.maximum(release_score, rdsi_benefit_weight)
            * torch.maximum(rdsi_alpha_weight, rdsi_benefit_weight)
            * rdsi_reliable_weight
        ).clamp(0.0, 1.0)
        fg_weight = rdsi_loss_weight * active_fg_repair_mask.float()
        bg_weight = rdsi_loss_weight * active_bg_suppress_mask.float()
        boundary_weight_map = rdsi_loss_weight * active_boundary_mask.float()
    else:
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
        boundary_weight_map = torch.zeros_like(bg_weight)
    fg_denom = fg_weight.sum().clamp_min(1.0)
    bg_denom = bg_weight.sum().clamp_min(1.0)
    boundary_denom = boundary_weight_map.sum().clamp_min(1.0)
    fg_loss = (per_pixel_kl * fg_weight).sum() / fg_denom
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        bg_loss = (per_pixel_kl * bg_weight).sum() / bg_denom
    else:
        bg_loss = student_logits.sum() * 0.0
    if float(active_boundary_mask.float().sum().detach().cpu().item()) > 0.0:
        boundary_loss = (per_pixel_kl * boundary_weight_map).sum() / boundary_denom
    else:
        boundary_loss = student_logits.sum() * 0.0
    if rdsi_active_mask is not None:
        distill_weight = fg_weight + background_weight * bg_weight + boundary_weight_map
        loss = (per_pixel_kl * distill_weight).sum() / distill_weight.sum().clamp_min(1.0)
    else:
        bg_loss_balanced = bg_loss * background_balance_factor if float(active_bg_pixels.detach().cpu().item()) > 0.0 else bg_loss
        loss = (
            (fg_loss + background_weight * bg_loss_balanced)
            / (1.0 + background_weight if float(active_bg_pixels.detach().cpu().item()) > 0.0 else 1.0)
        )

    profile["loss"] = loss.detach()
    profile["teacher_active_loss"] = (loss.detach() * torch.tensor(lambda_effective, device=device))
    profile["rdsi_loss_raw"] = loss.detach()
    profile["rdsi_loss_weighted"] = profile["teacher_active_loss"].detach()
    profile["rdsi_fg_repair_loss"] = fg_loss.detach()
    profile["rdsi_bg_suppress_loss"] = bg_loss.detach()
    profile["rdsi_boundary_loss"] = boundary_loss.detach()
    profile["kl_mean"] = (per_pixel_kl * active_f).sum().detach() / active_sum.clamp_min(1.0)
    distill_weight = fg_weight + background_weight * bg_weight + boundary_weight_map
    profile["weight_mean"] = distill_weight.mean().detach()
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        profile["background_suppression_mean"] = (
            (bg_suppression * active_bg_mask.float()).sum().detach() / active_bg_pixels.clamp_min(1.0)
        )
    return loss, lambda_effective, profile
