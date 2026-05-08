# -*- coding:utf-8 -*-
import math

import torch
import torch.nn.functional as F


RGFTD_PROFILE_KEYS = [
    "loss",
    "lambda",
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
    "active_foreground_ratio",
    "active_background_ratio",
    "foreground_veto_ratio",
    "active_foreground_pixels",
    "active_background_pixels",
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


def _topk_foreground_anchor(score, valid_mask, topk_ratio, min_pixels):
    anchors = torch.zeros_like(valid_mask, dtype=torch.bool)
    flat_score = score.flatten(1)
    flat_valid = valid_mask.flatten(1)
    for batch_idx in range(score.shape[0]):
        valid_count = int(flat_valid[batch_idx].sum().detach().cpu().item())
        if valid_count <= 0:
            continue
        k = max(int(min_pixels), int(math.ceil(float(valid_count) * float(topk_ratio))))
        k = min(k, valid_count)
        masked_score = flat_score[batch_idx].masked_fill(~flat_valid[batch_idx], -1.0)
        topk_index = torch.topk(masked_score, k=k, largest=True).indices
        anchors.flatten(1)[batch_idx, topk_index] = True
    return anchors & valid_mask


def _seed_or_fallback(seed_mask, fallback_mask):
    valid_mask = fallback_mask.clone()
    for batch_idx in range(seed_mask.shape[0]):
        if seed_mask[batch_idx].any():
            valid_mask[batch_idx] = seed_mask[batch_idx]
    return valid_mask & fallback_mask


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


def rgftd_loss(student_logits, teacher_logits, wann_maps, args, iter_num):
    """Reliability-gated teacher distillation on WANN non-core regions."""
    device = student_logits.device
    profile = zero_rgftd_profile(device)
    lambda_rgftd = get_rgftd_lambda(iter_num, args)
    profile["lambda"] = torch.tensor(lambda_rgftd, device=device)

    temperature = float(getattr(args, "rgftd_temperature", 1.0))
    temperature = max(temperature, 1e-6)
    teacher_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    student_prob = F.softmax(student_logits.detach(), dim=1)

    teacher_conf = teacher_prob.max(dim=1)[0]
    teacher_pred = teacher_prob.argmax(dim=1)
    student_conf = student_prob.max(dim=1)[0]
    student_entropy = _normalized_entropy(student_prob)

    low_r_thresh = float(getattr(args, "rgftd_low_r_thresh", getattr(args, "wann_soft_thresh", 0.25)))
    use_soft_band = int(getattr(args, "rgftd_use_soft_band", 1)) == 1
    teacher_fg_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    min_fg_pixels = int(getattr(args, "rgftd_min_foreground_pixels", 8))
    min_fg_ratio = float(getattr(args, "rgftd_min_foreground_ratio", 0.05))
    teacher_fg_prob_thresh = float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))
    teacher_fg_topk_ratio = float(getattr(args, "rgftd_teacher_fg_topk_ratio", 0.002))
    teacher_fg_topk_min_pixels = int(getattr(args, "rgftd_teacher_fg_topk_min_pixels", min_fg_pixels))
    teacher_student_fg_margin = float(getattr(args, "rgftd_teacher_student_fg_margin", 0.05))
    teacher_bg_conf_thresh = float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98))
    skip_background_only = int(getattr(args, "rgftd_skip_background_only", 1)) == 1

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
    teacher_bg_conf = teacher_bg & (teacher_conf >= teacher_bg_conf_thresh)
    teacher_accept = teacher_fg_anchor | teacher_bg_conf
    teacher_fg_context = _dilate_mask(teacher_fg_anchor, teacher_fg_radius)
    teacher_bg_context = teacher_fg_context & teacher_bg_conf
    region = region & (teacher_fg_context | teacher_bg_context)

    student_uncertain = (
        (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
        | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
    )
    low_reliability = wann_maps.reliability < low_r_thresh
    foreground_correction = teacher_fg_anchor & ((teacher_fg_prob - student_fg_prob) >= teacher_student_fg_margin)
    active_fg = region & teacher_fg_anchor & (student_uncertain | foreground_correction | low_reliability)
    active_bg = region & teacher_bg_conf & (~teacher_fg_anchor) & student_uncertain
    active = active_fg | active_bg

    region_f = region.float()
    active_f = active.float()
    region_sum = region_f.sum().clamp_min(1.0)
    active_sum = active_f.sum()
    active_fg_mask = active_fg
    active_bg_mask = active_bg
    active_fg_pixels = active_fg_mask.float().sum()
    active_bg_pixels = active_bg_mask.float().sum()
    active_total = active_sum.clamp_min(1.0)
    active_fg_ratio = active_fg_pixels / active_total
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
    profile["teacher_foreground_ratio"] = ((teacher_fg_anchor & region).float().sum() / region_sum).detach()
    profile["foreground_veto_ratio"] = torch.tensor(1.0 if foreground_veto else 0.0, device=device)
    profile["active_foreground_ratio"] = active_fg_ratio.detach()

    if float(active_sum.detach().cpu().item()) <= 0.0 or foreground_veto:
        zero_loss = student_logits.sum() * 0.0
        profile["loss"] = zero_loss.detach()
        return zero_loss, lambda_rgftd, profile

    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    per_pixel_kl = F.kl_div(student_log_prob, teacher_prob, reduction="none").sum(dim=1) * temperature * temperature
    background_weight = float(getattr(args, "rgftd_background_weight", 0.25))
    fg_weight = (1.0 - wann_maps.reliability).clamp(0.0, 1.0) * active_fg_mask.float()
    bg_weight = (1.0 - wann_maps.reliability).clamp(0.0, 1.0) * active_bg_mask.float()
    fg_denom = fg_weight.sum().clamp_min(1.0)
    bg_denom = bg_weight.sum().clamp_min(1.0)
    fg_loss = (per_pixel_kl * fg_weight).sum() / fg_denom
    if float(active_bg_pixels.detach().cpu().item()) > 0.0:
        bg_loss = (per_pixel_kl * bg_weight).sum() / bg_denom
    else:
        bg_loss = student_logits.sum() * 0.0
    loss = (fg_loss + background_weight * bg_loss) / (1.0 + background_weight if float(active_bg_pixels.detach().cpu().item()) > 0.0 else 1.0)

    profile["loss"] = loss.detach()
    profile["kl_mean"] = (per_pixel_kl * active_f).sum().detach() / active_sum.clamp_min(1.0)
    distill_weight = fg_weight + background_weight * bg_weight
    profile["weight_mean"] = distill_weight.mean().detach()
    profile["active_background_ratio"] = (active_bg_pixels / active_sum.clamp_min(1.0)).detach()
    profile["active_foreground_pixels"] = active_fg_pixels.detach()
    profile["active_background_pixels"] = active_bg_pixels.detach()
    return loss, lambda_rgftd, profile
