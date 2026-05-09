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
    "candidate_ratio",
    "active_ratio",
    "region_ratio",
    "teacher_accept_ratio",
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
    "core_conflict_veto_ratio",
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
    "background_suppression_mean",
    "return_reason",
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


def _normalize_gate(score, floor):
    floor = float(max(min(floor, 0.999999), 0.0))
    return ((score - floor) / max(1.0 - floor, 1e-6)).clamp(0.0, 1.0)


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


def rgftd_loss(student_logits, teacher_logits, label, wann_maps, args, iter_num):
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
    student_entropy = _normalized_entropy(student_prob)
    label = label.long()
    valid_mask = getattr(wann_maps, "valid_mask", label != int(getattr(args, "num_classes", teacher_prob.shape[1])))
    support_mask = getattr(wann_maps, "support_mask", valid_mask)
    seed_support_mask = getattr(wann_maps, "seed_support_mask", support_mask)
    core_mask = wann_maps.core_mask & valid_mask
    fg_support = support_mask & (label > 0)
    bg_support = support_mask & (label == 0)
    seed_fg_support = seed_support_mask & (label > 0)
    seed_bg_support = seed_support_mask & (label == 0)

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
        candidate_mask = (~wann_maps.core_mask) & (wann_maps.reliability >= low_r_thresh)
    candidate_mask = candidate_mask & (~wann_maps.core_mask)

    # Re-open non-core pixels, but only around teacher foreground evidence.
    # This keeps the v1 semantics (uncertain / ignore can be distilled) without turning RGFTD into background-only KL.
    non_core_region = wann_maps.ignore_mask | wann_maps.soft_band
    if use_soft_band:
        non_core_region = non_core_region | candidate_mask
    region = non_core_region & (~wann_maps.core_mask)

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
    teacher_bg_context = teacher_fg_context & teacher_bg_conf
    region = region & (teacher_fg_context | teacher_bg_context)

    teacher_core_agreement = _masked_ratio((teacher_pred == label) & core_mask, core_mask).detach()
    teacher_core_conflict = (1.0 - teacher_core_agreement).detach()
    teacher_core_conf_mean = _masked_mean(teacher_conf, core_mask).detach()
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
        core_gate = _normalize_gate(
            teacher_core_agreement,
            float(getattr(args, "rgftd_teacher_core_agree_floor", 0.80)),
        )
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
    active_fg = region & teacher_fg_ready & (student_uncertain | foreground_correction)
    active_fg_pre_budget = active_fg
    fg_budget_score = teacher_fg_prob * student_entropy * (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
    active_fg = _topk_foreground_anchor(
        fg_budget_score,
        active_fg,
        topk_ratio=active_fg_topk_ratio,
        min_pixels=active_fg_topk_min_pixels,
        max_pixels=active_fg_topk_max_pixels,
        preserve_mask=foreground_correction & active_fg,
    )
    active_bg = region & teacher_bg_conf & teacher_bg_safe & (~teacher_fg_ready) & student_uncertain
    active_bg_pre_budget = active_bg
    bg_budget_score = teacher_bg_prob * student_entropy * (1.0 - wann_maps.reliability).clamp(0.0, 1.0)
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
    foreground_veto = bool(
        skip_background_only and (
            active_fg_pixels.detach().cpu().item() < float(min_fg_pixels)
            or active_fg_ratio.detach().cpu().item() < float(min_fg_ratio)
        )
    )

    profile["candidate_ratio"] = candidate_mask.float().mean().detach()
    profile["region_ratio"] = region_f.mean().detach()
    profile["active_ratio"] = active_f.mean().detach()
    profile["teacher_accept_ratio"] = ((teacher_accept & region).float().sum() / region_sum).detach()
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
    profile["core_conflict_veto_ratio"] = torch.tensor(1.0 if teacher_validation_veto else 0.0, device=device)
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
    for class_id in range(1, teacher_prob.shape[1]):
        class_release = ((active_fg & (teacher_pred == class_id)).float().sum() / region_sum).detach()
        profile["teacher_class{}_release_ratio".format(class_id)] = class_release

    lambda_pre_safety = (
        float(lambda_rgftd)
        * float(teacher_reliability.detach().cpu().item())
        * float(release_factor.detach().cpu().item())
    )
    if validation_enabled:
        core_conflict_value = float(teacher_core_conflict.detach().cpu().item())
        core_conflict_limit = float(max(getattr(args, "rgftd_teacher_max_core_conflict", 0.20), 1e-6))
        core_safety_factor = max(0.0, min(1.0 - core_conflict_value / core_conflict_limit, 1.0))
    else:
        core_safety_factor = 1.0
    lambda_after_safety = lambda_pre_safety * core_safety_factor
    lambda_effective = lambda_after_safety
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

    if return_reason > 0.0:
        zero_loss = student_logits.sum() * 0.0
        profile["loss"] = zero_loss.detach()
        return zero_loss, lambda_effective, profile

    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    per_pixel_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * temperature * temperature
    background_weight = float(getattr(args, "rgftd_background_weight", 0.25))
    if validation_enabled and teacher_prob.shape[1] > 2:
        fg_teacher_weight = (class_weight_map * teacher_conf).clamp(0.0, 1.0)
    else:
        fg_teacher_weight = (0.5 * class_weight_map + 0.5 * teacher_fg_prob).clamp(0.0, 1.0)
    fg_weight = (1.0 - wann_maps.reliability).clamp(0.0, 1.0) * active_fg_mask.float() * fg_teacher_weight * teacher_reliability
    bg_suppression = ((bg_max_fg_prob - teacher_fg_prob) / max(bg_max_fg_prob, 1e-6)).clamp(0.0, 1.0)
    bg_weight = (1.0 - wann_maps.reliability).clamp(0.0, 1.0) * active_bg_mask.float() * teacher_reliability * bg_suppression
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
    profile["kl_mean"] = (per_pixel_kl * active_f).sum().detach() / active_sum.clamp_min(1.0)
    distill_weight = fg_weight + background_weight * bg_weight
    profile["weight_mean"] = distill_weight.mean().detach()
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        profile["background_suppression_mean"] = (
            (bg_suppression * active_bg_mask.float()).sum().detach() / active_bg_pixels.clamp_min(1.0)
        )
    return loss, lambda_effective, profile
