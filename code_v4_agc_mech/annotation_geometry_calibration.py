# -*- coding:utf-8 -*-
import math

import torch
import torch.nn.functional as F


def _zero_like_scalar(logits):
    return logits.sum() * 0.0


def _safe_weighted_mean(value, mask, weight=None):
    mask_f = mask.float()
    if weight is None:
        weight_f = mask_f
    else:
        weight_f = weight.float() * mask_f
    denom = weight_f.sum().clamp_min(1.0)
    return (value * weight_f).sum() / denom


def _mask_ratio(mask, ref):
    return mask.float().mean() if mask is not None else ref.sum() * 0.0


def _dilate(mask, radius):
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _erode(mask, radius):
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(1)
    y = -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _sigmoid_rampup(current, rampup_length):
    rampup_length = int(max(rampup_length, 0))
    if rampup_length == 0:
        return 1.0
    current = float(max(0, min(int(current), rampup_length)))
    phase = 1.0 - current / float(rampup_length)
    return float(math.exp(-5.0 * phase * phase))


def zero_acg_profile(device):
    zero = torch.tensor(0.0, device=device)
    return {
        "enabled": zero,
        "lambda": zero,
        "loss": zero,
        "core_miss_loss": zero,
        "support_miss_loss": zero,
        "range_loss": zero,
        "range_under_loss": zero,
        "range_over_loss": zero,
        "unsupported_leak_loss": zero,
        "boundary_loss": zero,
        "boundary_cons_loss": zero,
        "boundary_smooth_loss": zero,
        "shape_contrast_loss": zero,
        "core_ratio": zero,
        "support_ratio": zero,
        "unsupported_ratio": zero,
        "boundary_ratio": zero,
        "shape_ring_ratio": zero,
        "shape_anchor_ratio": zero,
        "core_mass": zero,
        "envelope_mass": zero,
        "lower_mass": zero,
        "upper_mass": zero,
        "uncertain_mass": zero,
        "reliability_mean": zero,
        "pred_fg_mass": zero,
        "pred_fg_core_mean": zero,
        "pred_fg_support_mean": zero,
        "pred_fg_unsupported_mean": zero,
        "pred_fg_shape_ring_mean": zero,
        "pred_fg_shape_anchor_mean": zero,
        "core_weight_scale": zero,
        "support_weight_scale": zero,
        "range_weight_scale": zero,
        "leak_weight_scale": zero,
        "boundary_weight_scale": zero,
        "shape_weight_scale": zero,
        "fg_violation_share": zero,
        "range_violation_share": zero,
        "leak_violation_share": zero,
        "boundary_violation_share": zero,
        "shape_violation_share": zero,
        "nwr_fg_loss": zero,
        "nwr_seed_fg_loss": zero,
        "nwr_context_fg_loss": zero,
        "nwr_bg_loss": zero,
        "nwr_context_fg_target": zero,
        "nwr_context_fg_upper_target": zero,
        "nwr_context_fg_margin_gap": zero,
        "nwr_context_fg_over_gap": zero,
        "nwr_fg_weight_mass": zero,
        "nwr_seed_fg_weight_mass": zero,
        "nwr_context_fg_weight_mass": zero,
        "nwr_bg_weight_mass": zero,
        "nwr_fg_region_ratio": zero,
        "nwr_seed_fg_region_ratio": zero,
        "nwr_context_fg_region_ratio": zero,
        "nwr_bg_region_ratio": zero,
        "nwr_region_count": zero,
        "nwr_fg_prior": zero,
        "nwr_seed_fg_prior": zero,
        "nwr_context_fg_prior": zero,
        "nwr_bg_prior": zero,
        "nwr_prior_mode_id": zero,
        "nwr_pred_fg_on_fg": zero,
        "nwr_pred_fg_on_seed_fg": zero,
        "nwr_pred_fg_on_context_fg": zero,
        "nwr_pred_fg_on_bg": zero,
        "agc_enabled": zero,
        "agc_mode": zero,
        "agc_active": zero,
        "agc_new_context_ratio": zero,
        "agc_new_context_to_target_ratio": zero,
        "agc_new_context_fg_prob_mean": zero,
        "agc_cap_hit_ratio": zero,
        "agc_from_candidate_ratio": zero,
        "agc_from_connected_ratio": zero,
        "agc_completed_context_ratio": zero,
        "agc_pred_fg_on_completed_context": zero,
        "agc_spatial_rule_id": zero,
        "agc_disable_cap": zero,
        "agc_disable_conf_gate": zero,
        "agc_complete_as_seed": zero,
    }


def _normalized_region_mean(value, region, weight):
    weight_f = torch.nan_to_num(weight.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weight_f = weight_f.clamp_min(0.0) * region.float()
    denom = weight_f.sum()
    numer = (value * weight_f).sum()
    return torch.where(denom > 0.0, numer / denom.clamp_min(1e-6), value.sum() * 0.0), denom


def _foreground_probability(cur_logits, num_classes):
    prob = torch.softmax(cur_logits, dim=1)
    if int(num_classes) <= 1:
        fg_prob = prob[:, 0]
    else:
        fg_prob = prob[:, 1:int(num_classes)].sum(dim=1)
    return torch.nan_to_num(fg_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _topk_mask_per_image(score, candidate, max_target_mult, max_image_ratio, target_fg):
    candidate = candidate.bool()
    flat_score = score.flatten(1)
    flat_candidate = candidate.flatten(1)
    flat_target = target_fg.flatten(1)
    out = torch.zeros_like(flat_candidate)
    pre_cap_total = 0
    cap_hit = False
    for idx in range(flat_candidate.shape[0]):
        cand_idx = torch.nonzero(flat_candidate[idx], as_tuple=False).flatten()
        pre_cap_total += int(cand_idx.numel())
        if cand_idx.numel() == 0:
            continue
        image_pixels = int(flat_candidate.shape[1])
        target_pixels = float(flat_target[idx].sum().detach().item())
        cap_by_target = int(round(float(max_target_mult) * target_pixels))
        cap_by_image = int(round(float(max_image_ratio) * image_pixels))
        max_new_pixels = max(0, min(cap_by_target, cap_by_image))
        if cand_idx.numel() > max_new_pixels:
            cap_hit = True
        if max_new_pixels <= 0:
            continue
        k = min(max_new_pixels, cand_idx.numel())
        cand_score = flat_score[idx, cand_idx]
        keep = torch.topk(cand_score, k=k, largest=True, sorted=False).indices
        out[idx, cand_idx[keep]] = True
    return out.view_as(candidate), pre_cap_total, cap_hit


def _connected_to_anchor(mask, anchor, max_steps):
    mask = mask.bool()
    keep = mask & anchor.bool()
    if int(max_steps) <= 0 or not keep.any():
        return keep

    prev_count = -1
    for _ in range(int(max_steps)):
        keep = _dilate(keep, 1) & mask
        cur_count = int(keep.detach().sum().item())
        if cur_count == prev_count:
            break
        prev_count = cur_count
    return keep


def _agc_ramp(iter_num, args):
    start = int(getattr(args, "agc_start_iter", 1000))
    ramp = int(getattr(args, "agc_ramp_iters", 800))
    if int(iter_num) < start:
        return 0.0
    return _sigmoid_rampup(int(iter_num) - start, ramp)


def _build_agc_completion(fg_prob, target_fg, target_bg, seed_fg, context_fg, wann_maps, args, iter_num):
    device = fg_prob.device
    zero = fg_prob.sum() * 0.0
    profile = {
        "agc_enabled": torch.tensor(float(int(getattr(args, "agc_enabled", 0)) == 1), device=device),
        "agc_mode": torch.tensor(1.0 if str(getattr(args, "agc_mode", "conservative")).lower() == "moderate" else 0.0, device=device),
        "agc_active": zero,
        "agc_new_context_ratio": zero,
        "agc_new_context_to_target_ratio": zero,
        "agc_new_context_fg_prob_mean": zero,
        "agc_cap_hit_ratio": zero,
        "agc_from_candidate_ratio": zero,
        "agc_from_connected_ratio": zero,
        "agc_completed_context_ratio": context_fg.float().mean().detach(),
        "agc_pred_fg_on_completed_context": _safe_weighted_mean(fg_prob.detach(), context_fg).detach(),
        "agc_spatial_rule_id": zero,
        "agc_disable_cap": torch.tensor(float(int(getattr(args, "agc_disable_cap", 0)) == 1), device=device),
        "agc_disable_conf_gate": torch.tensor(float(int(getattr(args, "agc_disable_conf_gate", 0)) == 1), device=device),
        "agc_complete_as_seed": torch.tensor(float(int(getattr(args, "agc_complete_as_seed", 0)) == 1), device=device),
    }
    if int(getattr(args, "agc_enabled", 0)) != 1:
        return seed_fg, context_fg, target_fg, None, None, profile

    ramp = _agc_ramp(iter_num, args)
    if ramp <= 0.0:
        return seed_fg, context_fg, target_fg, None, None, profile

    mode = str(getattr(args, "agc_mode", "conservative")).lower()
    if mode == "moderate":
        tau = float(getattr(args, "agc_moderate_tau", 0.65))
        max_target_mult = float(getattr(args, "agc_moderate_max_target_mult", 1.5))
        max_image_ratio = float(getattr(args, "agc_moderate_max_image_ratio", 0.08))
        completion_weight = float(getattr(args, "agc_moderate_weight", 0.35))
        completion_reliability = float(getattr(args, "agc_moderate_reliability", 0.55))
        require_mode = "or"
    else:
        tau = float(getattr(args, "agc_conservative_tau", 0.75))
        max_target_mult = float(getattr(args, "agc_conservative_max_target_mult", 0.5))
        max_image_ratio = float(getattr(args, "agc_conservative_max_image_ratio", 0.03))
        completion_weight = float(getattr(args, "agc_conservative_weight", 0.5))
        completion_reliability = float(getattr(args, "agc_conservative_reliability", 0.65))
        require_mode = "and"

    candidate = wann_maps.candidate_mask.bool()
    support_fg = wann_maps.support_mask.bool() & target_fg
    anchor_fg = seed_fg | support_fg
    pred_candidate = fg_prob.detach() >= tau
    confidence_gate = torch.ones_like(pred_candidate, dtype=torch.bool)
    if int(getattr(args, "agc_disable_conf_gate", 0)) != 1:
        confidence_gate = pred_candidate
    connected_source = pred_candidate
    if int(getattr(args, "agc_disable_conf_gate", 0)) == 1:
        connected_source = torch.ones_like(pred_candidate, dtype=torch.bool)
    connected = _connected_to_anchor(connected_source, anchor_fg, int(getattr(args, "agc_connect_steps", 64)))
    spatial_rule = str(getattr(args, "agc_spatial_rule", "mode")).lower()
    spatial_rule_id = 0.0
    if spatial_rule == "confidence_only":
        spatial_safe = torch.ones_like(pred_candidate, dtype=torch.bool)
        spatial_rule_id = 1.0
    elif spatial_rule == "candidate_only":
        spatial_safe = candidate
        spatial_rule_id = 2.0
    elif spatial_rule == "connected_only":
        spatial_safe = connected
        spatial_rule_id = 3.0
    elif spatial_rule == "candidate_or_connected":
        spatial_safe = candidate | connected
        spatial_rule_id = 4.0
    elif spatial_rule == "candidate_and_connected":
        spatial_safe = candidate & connected
        spatial_rule_id = 5.0
    elif require_mode == "or":
        spatial_safe = candidate | connected
    else:
        spatial_safe = candidate & connected

    hard_bg = target_bg & wann_maps.seed_support_mask.bool()
    safe = confidence_gate & spatial_safe & (~target_fg) & (~hard_bg)
    target_pixels = float(target_fg.detach().sum().item())
    if int(getattr(args, "agc_disable_cap", 0)) == 1:
        pre_cap_pixels = int(safe.detach().sum().item())
        cap_hit = False
    else:
        safe, pre_cap_pixels, cap_hit = _topk_mask_per_image(
            fg_prob.detach(), safe, max_target_mult, max_image_ratio, target_fg
        )

    if int(getattr(args, "agc_complete_as_seed", 0)) == 1:
        completed_seed = seed_fg | safe
        completed_context = context_fg
    else:
        completed_seed = seed_fg
        completed_context = context_fg | safe
    completed_target_fg = completed_seed | completed_context
    completion_weight_map = torch.zeros_like(fg_prob.detach())
    completion_reliability_map = torch.zeros_like(fg_prob.detach())
    if safe.any():
        completion_weight_map[safe] = completion_weight * ramp
        completion_reliability_map[safe] = completion_reliability * ramp

    new_pixels = int(safe.detach().sum().item())
    denom_target = max(target_pixels, 1.0)
    profile.update({
        "agc_active": torch.tensor(float(ramp), device=device),
        "agc_new_context_ratio": safe.float().mean().detach(),
        "agc_new_context_to_target_ratio": torch.tensor(float(new_pixels) / denom_target, device=device),
        "agc_new_context_fg_prob_mean": _safe_weighted_mean(fg_prob.detach(), safe).detach(),
        "agc_cap_hit_ratio": torch.tensor(1.0 if cap_hit else 0.0, device=device),
        "agc_from_candidate_ratio": _safe_weighted_mean(candidate.float(), safe).detach(),
        "agc_from_connected_ratio": _safe_weighted_mean(connected.float(), safe).detach(),
        "agc_completed_context_ratio": completed_context.float().mean().detach(),
        "agc_pred_fg_on_completed_context": _safe_weighted_mean(fg_prob.detach(), completed_context).detach(),
        "agc_spatial_rule_id": torch.tensor(float(spatial_rule_id), device=device),
        "agc_disable_cap": torch.tensor(float(int(getattr(args, "agc_disable_cap", 0)) == 1), device=device),
        "agc_disable_conf_gate": torch.tensor(float(int(getattr(args, "agc_disable_conf_gate", 0)) == 1), device=device),
        "agc_complete_as_seed": torch.tensor(float(int(getattr(args, "agc_complete_as_seed", 0)) == 1), device=device),
    })
    return completed_seed, completed_context, completed_target_fg, completion_weight_map, completion_reliability_map, profile


def _context_soft_foreground_loss(cur_logits, num_classes, context_fg, geometry_weight, reliability, args):
    fg_prob = _foreground_probability(cur_logits, num_classes)
    rel = torch.nan_to_num(reliability.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    margin_floor = float(getattr(args, "acg_context_margin_floor", 0.35))
    margin_ceiling = float(getattr(args, "acg_context_margin_ceiling", 0.85))
    margin_ceiling = max(margin_floor, margin_ceiling)
    band_width = float(getattr(args, "acg_context_band_width", 0.15))
    over_weight = float(getattr(args, "acg_context_band_over_weight", 0.25))
    lower_target = rel.clamp(margin_floor, margin_ceiling)
    upper_target = (lower_target + band_width).clamp(max=margin_ceiling)
    under_gap = (lower_target - fg_prob).clamp_min(0.0)
    over_gap = (fg_prob - upper_target).clamp_min(0.0)
    loss_map = (
        under_gap.detach() * (-torch.log(fg_prob.clamp_min(1e-6)))
        + over_weight * over_gap.detach() * (-torch.log((1.0 - fg_prob).clamp_min(1e-6)))
    )
    context_loss, context_weight_mass = _normalized_region_mean(loss_map, context_fg, geometry_weight)
    context_target, _ = _normalized_region_mean(lower_target, context_fg, geometry_weight)
    context_upper_target, _ = _normalized_region_mean(upper_target, context_fg, geometry_weight)
    context_gap, _ = _normalized_region_mean(under_gap, context_fg, geometry_weight)
    context_over_gap, _ = _normalized_region_mean(over_gap, context_fg, geometry_weight)
    return context_loss, context_weight_mass, context_target, context_upper_target, context_gap, context_over_gap


def _normalized_acg_components(
    cur_logits,
    target_label,
    num_classes,
    seed_fg,
    context_fg,
    target_fg,
    target_bg,
    geometry_weight,
    reliability,
    args,
):
    safe_target = target_label.clamp(0, max(int(num_classes) - 1, 0))
    ce_map = F.cross_entropy(cur_logits, safe_target, reduction="none")
    ce_map = torch.nan_to_num(ce_map, nan=0.0, posinf=0.0, neginf=0.0)
    seed_fg_loss, seed_fg_weight_mass = _normalized_region_mean(ce_map, seed_fg, geometry_weight)
    context_fg_loss, context_fg_weight_mass = _normalized_region_mean(ce_map, context_fg, geometry_weight)
    context_fg_target = context_fg_loss.detach() * 0.0
    context_fg_upper_target = context_fg_loss.detach() * 0.0
    context_fg_margin_gap = context_fg_loss.detach() * 0.0
    context_fg_over_gap = context_fg_loss.detach() * 0.0
    if int(getattr(args, "acg_context_soft_foreground", 0)) == 1:
        (
            context_fg_loss,
            context_fg_weight_mass,
            context_fg_target,
            context_fg_upper_target,
            context_fg_margin_gap,
            context_fg_over_gap,
        ) = (
            _context_soft_foreground_loss(cur_logits, num_classes, context_fg, geometry_weight, reliability, args)
        )
    fg_loss, fg_weight_mass = _normalized_region_mean(ce_map, target_fg, geometry_weight)
    bg_loss, bg_weight_mass = _normalized_region_mean(ce_map, target_bg, geometry_weight)
    seed_fg_present = (seed_fg_weight_mass > 0.0).float()
    context_fg_present = (context_fg_weight_mass > 0.0).float()
    bg_present = (bg_weight_mass > 0.0).float()
    region_count = (seed_fg_present + context_fg_present + bg_present).clamp_min(1.0)
    prior_mode = str(getattr(args, "acg_prior_mode", "sqrt")).lower()
    prior_mode_id = 0.0
    if prior_mode == "uniform":
        seed_fg_prior_score = seed_fg_present
        context_fg_prior_score = context_fg_present
        bg_prior_score = bg_present
        prior_mode_id = 2.0
    elif prior_mode == "mass":
        seed_fg_prior_score = seed_fg_weight_mass.clamp_min(0.0) * seed_fg_present
        context_fg_prior_score = context_fg_weight_mass.clamp_min(0.0) * context_fg_present
        bg_prior_score = bg_weight_mass.clamp_min(0.0) * bg_present
        prior_mode_id = 1.0
    else:
        seed_fg_prior_score = torch.sqrt(seed_fg_weight_mass.clamp_min(0.0)) * seed_fg_present
        context_fg_prior_score = torch.sqrt(context_fg_weight_mass.clamp_min(0.0)) * context_fg_present
        bg_prior_score = torch.sqrt(bg_weight_mass.clamp_min(0.0)) * bg_present
        if prior_mode == "raw_mean":
            prior_mode_id = 3.0
    prior_sum = seed_fg_prior_score + context_fg_prior_score + bg_prior_score
    seed_fg_prior = torch.where(prior_sum > 0.0, seed_fg_prior_score / prior_sum.clamp_min(1e-6), seed_fg_present * 0.0)
    context_fg_prior = torch.where(prior_sum > 0.0, context_fg_prior_score / prior_sum.clamp_min(1e-6), context_fg_present * 0.0)
    bg_prior = torch.where(prior_sum > 0.0, bg_prior_score / prior_sum.clamp_min(1e-6), bg_present * 0.0)
    fg_prior = seed_fg_prior + context_fg_prior
    calibrated_fg_loss = torch.where(
        fg_prior > 0.0,
        (seed_fg_prior * seed_fg_loss + context_fg_prior * context_fg_loss) / fg_prior.clamp_min(1e-6),
        fg_loss,
    )
    if prior_mode == "raw_mean":
        union_region = seed_fg | context_fg | target_bg
        total_loss, _ = _normalized_region_mean(ce_map, union_region, geometry_weight)
    else:
        total_loss = seed_fg_prior * seed_fg_loss + context_fg_prior * context_fg_loss + bg_prior * bg_loss
    return {
        "total_loss": total_loss,
        "fg_loss": calibrated_fg_loss,
        "raw_fg_loss": fg_loss,
        "seed_fg_loss": seed_fg_loss,
        "context_fg_loss": context_fg_loss,
        "bg_loss": bg_loss,
        "context_fg_target": context_fg_target,
        "context_fg_upper_target": context_fg_upper_target,
        "context_fg_margin_gap": context_fg_margin_gap,
        "context_fg_over_gap": context_fg_over_gap,
        "fg_weight_mass": fg_weight_mass,
        "seed_fg_weight_mass": seed_fg_weight_mass,
        "context_fg_weight_mass": context_fg_weight_mass,
        "bg_weight_mass": bg_weight_mass,
        "region_count": region_count,
        "fg_prior": fg_prior,
        "seed_fg_prior": seed_fg_prior,
        "context_fg_prior": context_fg_prior,
        "bg_prior": bg_prior,
        "prior_mode_id": torch.tensor(float(prior_mode_id), device=cur_logits.device),
    }


def acg_loss(logits, aux_logits, wann_maps, args, iter_num):
    device = logits.device
    profile = zero_acg_profile(device)
    if wann_maps is None:
        return _zero_like_scalar(logits), 0.0, profile

    num_classes = int(getattr(args, "num_classes", logits.shape[1]))
    prob = torch.softmax(logits, dim=1)
    if num_classes <= 1:
        fg_prob = prob[:, 0]
    else:
        fg_prob = prob[:, 1:int(num_classes)].sum(dim=1)
    fg_prob = torch.nan_to_num(fg_prob, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    target_label = wann_maps.target_label.long()
    valid_target = target_label != int(num_classes)
    target_fg = valid_target & (target_label > 0) & (target_label < int(num_classes))
    target_bg = valid_target & (target_label == 0)
    seed_fg = target_fg & wann_maps.seed_support_mask.bool()
    context_fg = target_fg & (~seed_fg)

    core_weight = torch.nan_to_num(wann_maps.core_weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    soft_weight = torch.nan_to_num(wann_maps.soft_weight.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    geometry_weight = torch.maximum(core_weight, soft_weight).clamp_min(0.0)
    reliability = torch.nan_to_num(wann_maps.reliability.detach(), nan=0.0, posinf=0.0, neginf=0.0)

    (
        seed_fg,
        context_fg,
        target_fg,
        agc_weight_map,
        agc_reliability_map,
        agc_profile,
    ) = _build_agc_completion(fg_prob, target_fg, target_bg, seed_fg, context_fg, wann_maps, args, iter_num)
    target_bg = target_bg & (~target_fg)
    if agc_weight_map is not None:
        geometry_weight = torch.maximum(geometry_weight, agc_weight_map).clamp_min(0.0)
    if agc_reliability_map is not None:
        reliability = torch.maximum(reliability, agc_reliability_map).clamp(0.0, 1.0)

    acg_parts = _normalized_acg_components(
        logits, target_label, num_classes, seed_fg, context_fg, target_fg, target_bg, geometry_weight, reliability, args
    )
    if aux_logits is not None:
        aux_parts = _normalized_acg_components(
            aux_logits, target_label, num_classes, seed_fg, context_fg, target_fg, target_bg, geometry_weight, reliability, args
        )
        for key in [
            "total_loss",
            "fg_loss",
            "raw_fg_loss",
            "seed_fg_loss",
            "context_fg_loss",
            "bg_loss",
            "context_fg_target",
            "context_fg_upper_target",
            "context_fg_margin_gap",
            "context_fg_over_gap",
        ]:
            acg_parts[key] = 0.5 * (acg_parts[key] + aux_parts[key])

    total_loss = acg_parts["total_loss"]
    fg_loss = acg_parts["fg_loss"]
    seed_fg_loss = acg_parts["seed_fg_loss"]
    context_fg_loss = acg_parts["context_fg_loss"]
    bg_loss = acg_parts["bg_loss"]
    context_fg_target = acg_parts["context_fg_target"]
    context_fg_upper_target = acg_parts["context_fg_upper_target"]
    context_fg_margin_gap = acg_parts["context_fg_margin_gap"]
    context_fg_over_gap = acg_parts["context_fg_over_gap"]
    fg_weight_mass = acg_parts["fg_weight_mass"]
    seed_fg_weight_mass = acg_parts["seed_fg_weight_mass"]
    context_fg_weight_mass = acg_parts["context_fg_weight_mass"]
    bg_weight_mass = acg_parts["bg_weight_mass"]
    region_count = acg_parts["region_count"]
    fg_prior = acg_parts["fg_prior"]
    seed_fg_prior = acg_parts["seed_fg_prior"]
    context_fg_prior = acg_parts["context_fg_prior"]
    bg_prior = acg_parts["bg_prior"]
    prior_mode_id = acg_parts["prior_mode_id"]

    pred_fg_mass = fg_prob.flatten(1).mean(dim=1)
    reliability_mean = reliability.flatten(1).mean(dim=1)
    fg_region_ratio = target_fg.float().mean()
    seed_fg_region_ratio = seed_fg.float().mean()
    context_fg_region_ratio = context_fg.float().mean()
    bg_region_ratio = target_bg.float().mean()
    fg_weight_density = fg_weight_mass / float(max(target_fg.numel(), 1))
    bg_weight_density = bg_weight_mass / float(max(target_bg.numel(), 1))
    pred_fg_on_fg = _safe_weighted_mean(fg_prob.detach(), target_fg, geometry_weight).detach()
    pred_fg_on_seed_fg = _safe_weighted_mean(fg_prob.detach(), seed_fg, geometry_weight).detach()
    pred_fg_on_context_fg = _safe_weighted_mean(fg_prob.detach(), context_fg, geometry_weight).detach()
    pred_fg_on_bg = _safe_weighted_mean(fg_prob.detach(), target_bg, geometry_weight).detach()

    lambda_acg = 1.0

    profile.update({
        "enabled": torch.tensor(1.0, device=device),
        "lambda": torch.tensor(float(lambda_acg), device=device),
        "loss": total_loss.detach(),
        "core_miss_loss": fg_loss.detach(),
        "support_miss_loss": bg_loss.detach(),
        "core_ratio": fg_region_ratio.detach(),
        "support_ratio": bg_region_ratio.detach(),
        "core_mass": fg_weight_density.detach(),
        "envelope_mass": (fg_weight_density + bg_weight_density).detach(),
        "lower_mass": fg_weight_density.detach(),
        "upper_mass": bg_weight_density.detach(),
        "reliability_mean": reliability_mean.mean().detach(),
        "pred_fg_mass": fg_prob.mean().detach(),
        "pred_fg_core_mean": pred_fg_on_fg.detach(),
        "pred_fg_support_mean": pred_fg_on_bg.detach(),
        "nwr_fg_loss": fg_loss.detach(),
        "nwr_seed_fg_loss": seed_fg_loss.detach(),
        "nwr_context_fg_loss": context_fg_loss.detach(),
        "nwr_bg_loss": bg_loss.detach(),
        "nwr_context_fg_target": context_fg_target.detach(),
        "nwr_context_fg_upper_target": context_fg_upper_target.detach(),
        "nwr_context_fg_margin_gap": context_fg_margin_gap.detach(),
        "nwr_context_fg_over_gap": context_fg_over_gap.detach(),
        "nwr_fg_weight_mass": fg_weight_mass.detach(),
        "nwr_seed_fg_weight_mass": seed_fg_weight_mass.detach(),
        "nwr_context_fg_weight_mass": context_fg_weight_mass.detach(),
        "nwr_bg_weight_mass": bg_weight_mass.detach(),
        "nwr_fg_region_ratio": fg_region_ratio.detach(),
        "nwr_seed_fg_region_ratio": seed_fg_region_ratio.detach(),
        "nwr_context_fg_region_ratio": context_fg_region_ratio.detach(),
        "nwr_bg_region_ratio": bg_region_ratio.detach(),
        "nwr_region_count": region_count.detach(),
        "nwr_fg_prior": fg_prior.detach(),
        "nwr_seed_fg_prior": seed_fg_prior.detach(),
        "nwr_context_fg_prior": context_fg_prior.detach(),
        "nwr_bg_prior": bg_prior.detach(),
        "nwr_prior_mode_id": prior_mode_id.detach(),
        "nwr_pred_fg_on_fg": pred_fg_on_fg.detach(),
        "nwr_pred_fg_on_seed_fg": pred_fg_on_seed_fg.detach(),
        "nwr_pred_fg_on_context_fg": pred_fg_on_context_fg.detach(),
        "nwr_pred_fg_on_bg": pred_fg_on_bg.detach(),
    })
    profile.update({key: value.detach() for key, value in agc_profile.items()})
    return total_loss, lambda_acg, profile
