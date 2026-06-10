# -*- coding:utf-8 -*-
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class WannMaps:
    target_label: torch.Tensor
    core_mask: torch.Tensor
    soft_band: torch.Tensor
    ignore_mask: torch.Tensor
    reliability: torch.Tensor
    core_weight: torch.Tensor
    soft_weight: torch.Tensor
    valid_mask: torch.Tensor
    support_mask: torch.Tensor
    seed_support_mask: torch.Tensor
    candidate_mask: torch.Tensor
    profile: dict


def _odd_kernel(radius):
    radius = int(max(radius, 0))
    return radius * 2 + 1


def _dilate(mask, radius):
    if radius <= 0:
        return mask
    kernel = _odd_kernel(radius)
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _erode(mask, radius):
    if radius <= 0:
        return mask
    kernel = _odd_kernel(radius)
    x = mask.float().unsqueeze(1)
    y = -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _normalize_image(image):
    if image.dim() == 3:
        image = image.unsqueeze(1)
    gray = image.float().mean(dim=1)
    gray = torch.nan_to_num(gray, nan=0.0, posinf=0.0, neginf=0.0)
    flat = gray.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1, unbiased=False).clamp_min(1e-6).view(-1, 1, 1)
    return (gray - mean) / std


def _support_mean_abs_distance(gray, label, support, num_classes):
    b, h, w = gray.shape
    dist = torch.zeros_like(gray)
    for idx in range(b):
        class_dists = []
        for class_id in range(int(num_classes)):
            cur_support = support[idx] & (label[idx] == class_id)
            if cur_support.any():
                support_values = gray[idx][cur_support]
                center = support_values.mean()
                if support_values.numel() <= 1:
                    scale = support_values.new_tensor(0.25)
                else:
                    scale = support_values.std(unbiased=False).clamp_min(0.25)
                class_dists.append(torch.abs(gray[idx] - center) / scale)
        if not class_dists:
            dist[idx].fill_(1.0)
        else:
            stacked = torch.stack(class_dists, dim=0)
            dist[idx] = torch.min(stacked, dim=0)[0]
    return dist


def _prediction_ambiguity(logits, aux_logits, num_classes):
    prob = torch.softmax(logits, dim=1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    entropy_norm = torch.log(torch.tensor(float(num_classes), device=logits.device)).clamp_min(1e-6)
    entropy = entropy / entropy_norm

    if aux_logits is None:
        agreement = torch.ones_like(entropy)
    else:
        aux_prob = torch.softmax(aux_logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        aux_pred = torch.argmax(aux_prob, dim=1)
        agreement = (pred == aux_pred).float()
    return entropy.clamp(0.0, 1.0), agreement


def _texture_stability(gray, kernel_size, temp):
    kernel_size = int(max(kernel_size, 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    x = gray.unsqueeze(1)
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding)
    local_var = (mean_sq - mean * mean).clamp_min(0.0)[:, 0]
    return torch.exp(-local_var / float(max(temp, 1e-6)))


def _radius_for_sup_type(sup_type, args):
    sup_type = str(sup_type).lower()
    if sup_type == "keypoint":
        return int(getattr(args, "wann_keypoint_soft_radius", 2))
    if sup_type in ["scribble", "scribble_noisy"]:
        return int(getattr(args, "wann_scribble_soft_radius", 4))
    if sup_type in ["box", "block"]:
        return int(getattr(args, "wann_box_soft_radius", 2))
    return int(getattr(args, "wann_mask_soft_radius", 1))


def _build_seed_support_mask(label, valid, sup_type, num_classes, args):
    sup_type = str(sup_type).lower()
    if _is_sparse_seed_sup_type(sup_type):
        return valid
    if not _is_block_like_sup_type(sup_type):
        return valid

    radius_name = "wann_seed_support_{}_erode_radius".format(sup_type)
    radius = int(getattr(args, radius_name, getattr(args, "wann_seed_support_erode_radius", 1)))
    if radius <= 0:
        return valid

    seed_support = torch.zeros_like(valid)
    for class_id in range(int(num_classes)):
        class_mask = valid & (label == class_id)
        if not class_mask.any():
            continue
        class_seed = _erode(class_mask, radius)
        if not class_seed.any():
            class_seed = class_mask
        seed_support = seed_support | class_seed
    return seed_support


def _is_sparse_seed_sup_type(sup_type):
    sup_type = str(sup_type).lower()
    return (
        sup_type == "keypoint"
        or sup_type in ["scribble", "scribble_noisy"]
        or sup_type.startswith("sparse_scribble_")
    )


def _is_block_like_sup_type(sup_type):
    return str(sup_type).lower() in ["box", "block"]


def _is_sparse_seed_protocol(sup_type, seed_support, args):
    if int(getattr(args, "wann_sparse_adaptive_core", 0)) != 1:
        return False
    if not _is_sparse_seed_sup_type(sup_type):
        return False
    return bool(seed_support.detach().any().item())


def _topk_ratio_mask(score, candidate, target_ratio, max_ratio):
    target_ratio = float(max(target_ratio, 0.0))
    max_ratio = float(max(max_ratio, 0.0))
    if target_ratio <= 0.0 or not candidate.any():
        return torch.zeros_like(candidate)

    flat_score = score.flatten(1)
    flat_candidate = candidate.flatten(1)
    out = torch.zeros_like(flat_candidate)
    num_pixels = flat_candidate.shape[1]
    target_k = int(round(target_ratio * num_pixels))
    max_k = int(round(max_ratio * num_pixels)) if max_ratio > 0.0 else target_k
    target_k = max(1, min(target_k, max_k, num_pixels))

    for idx in range(flat_candidate.shape[0]):
        cand_idx = torch.nonzero(flat_candidate[idx], as_tuple=False).flatten()
        if cand_idx.numel() == 0:
            continue
        k = min(target_k, cand_idx.numel())
        cand_score = flat_score[idx, cand_idx]
        keep = torch.topk(cand_score, k=k, largest=True, sorted=False).indices
        out[idx, cand_idx[keep]] = True
    return out.view_as(candidate)


def _build_sparse_support_label(label, support, radius, num_classes):
    target = torch.full_like(label, int(num_classes))
    target[support] = label[support]
    hit_count = torch.zeros_like(label, dtype=torch.int16)
    candidate = torch.zeros_like(support)

    for class_id in range(int(num_classes)):
        class_support = support & (label == class_id)
        if not class_support.any():
            continue
        class_context = _dilate(class_support, radius)
        assignable = (hit_count == 0) & class_context
        target[assignable] = class_id
        hit_count = hit_count + class_context.to(hit_count.dtype)
        candidate = candidate | class_context

    ambiguous = hit_count > 1
    target[ambiguous & (~support)] = int(num_classes)
    candidate = candidate & (~ambiguous | support)
    return target, candidate


def _build_unambiguous_class_targets(label, support, radius, num_classes):
    target = torch.full_like(label, int(num_classes))
    target[support] = label[support]
    hit_count = torch.zeros_like(label, dtype=torch.int16)
    foreground_context = torch.zeros_like(support)

    for class_id in range(1, int(num_classes)):
        class_support = support & (label == class_id)
        if not class_support.any():
            continue
        class_context = _dilate(class_support, radius)
        target[(hit_count == 0) & class_context] = class_id
        hit_count = hit_count + class_context.to(hit_count.dtype)
        foreground_context = foreground_context | class_context

    ambiguous = hit_count > 1
    target[ambiguous & (~support)] = int(num_classes)
    candidate = support | (foreground_context & (~ambiguous))
    return target, candidate


def build_wann_maps(image, label, logits, aux_logits, sup_type, img_class, num_classes, iter_num, args, ref_logits=None):
    del img_class  # Class-aware handling is task-structure driven, not dataset-name driven.
    label = label.long()
    valid = label != int(num_classes)
    support = valid
    seed_support = _build_seed_support_mask(label, valid, sup_type, num_classes, args)

    soft_radius = _radius_for_sup_type(sup_type, args)
    target_label = label
    sparse_seed_sup_type = _is_sparse_seed_sup_type(sup_type)
    block_like_protocol = _is_block_like_sup_type(sup_type)
    if sparse_seed_sup_type:
        target_label, support_dilated = _build_sparse_support_label(
            label, support, soft_radius, num_classes
        )
    else:
        support_dilated = _dilate(support, soft_radius)
    class_aware = (
        int(num_classes) > 2
        and int(getattr(args, "rdsi_class_aware", 0)) == 1
    )
    if class_aware:
        target_label, support_dilated = _build_unambiguous_class_targets(
            label, support, soft_radius, num_classes
        )

    annotation_score = torch.zeros_like(label, dtype=torch.float32)
    annotation_score[support_dilated] = float(getattr(args, "wann_dilated_support_score", 0.55))
    annotation_score[support] = 1.0

    gray = _normalize_image(image)
    appearance_dist = _support_mean_abs_distance(gray, label, support, num_classes)
    appearance_score = torch.exp(-appearance_dist / float(getattr(args, "wann_appearance_temp", 1.5)))
    texture_score = _texture_stability(
        gray,
        kernel_size=int(getattr(args, "wann_texture_kernel_size", 5)),
        temp=float(getattr(args, "wann_texture_temp", 1.0)),
    )
    texture_weight = float(getattr(args, "wann_texture_weight", 0.25))
    appearance_score = appearance_score * (texture_weight * texture_score + (1.0 - texture_weight))

    entropy, agreement = _prediction_ambiguity(logits.detach(), aux_logits.detach() if aux_logits is not None else None, num_classes)
    if ref_logits is None:
        ref_agreement = torch.ones_like(entropy)
    else:
        pred = torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1)
        ref_pred = torch.argmax(torch.softmax(ref_logits.detach(), dim=1), dim=1)
        ref_agreement = (pred == ref_pred).float()
    if int(iter_num) < int(getattr(args, "wann_pred_start_iter", 0)):
        ambiguity_score = torch.ones_like(entropy)
    else:
        entropy_weight = float(getattr(args, "wann_entropy_weight", 0.5))
        agreement_weight = float(getattr(args, "wann_agreement_weight", 0.5))
        global_agreement_weight = float(getattr(args, "wann_global_agreement_weight", 0.5))
        ambiguity_score = (1.0 - entropy_weight * entropy).clamp(0.0, 1.0)
        ambiguity_score = ambiguity_score * (agreement_weight * agreement + (1.0 - agreement_weight))
        ambiguity_score = ambiguity_score * (global_agreement_weight * ref_agreement + (1.0 - global_agreement_weight))

    appearance_score = torch.nan_to_num(appearance_score, nan=0.0, posinf=0.0, neginf=0.0)
    ambiguity_score = torch.nan_to_num(ambiguity_score, nan=0.0, posinf=0.0, neginf=0.0)
    reliability = annotation_score * appearance_score * ambiguity_score
    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=0.0, neginf=0.0)
    reliability = reliability.clamp(0.0, float(getattr(args, "wann_r_max", 1.2)))

    core_thresh = float(getattr(args, "wann_core_thresh", 0.65))
    soft_thresh = float(getattr(args, "wann_soft_thresh", 0.25))
    target_valid = (target_label >= 0) & (target_label < int(num_classes))
    core_mask = target_valid & (reliability >= core_thresh)
    pred_prob = torch.softmax(logits.detach(), dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    max_prob = pred_prob.max(dim=1)[0]
    seed_fg = seed_support & (target_label > 0) & (target_label < int(num_classes))
    sparse_seed_protocol = _is_sparse_seed_protocol(sup_type, seed_support, args)
    if sparse_seed_protocol:
        core_candidate_mask = support_dilated & target_valid
    elif block_like_protocol:
        core_candidate_mask = seed_support & target_valid
    else:
        core_candidate_mask = target_valid

    if sparse_seed_protocol:
        min_core_ratio = float(getattr(args, "wann_sparse_min_core_ratio", 0.0))
        max_core_ratio = float(getattr(args, "wann_sparse_max_core_ratio", min_core_ratio))
        if max_core_ratio <= 0.0:
            max_core_ratio = min_core_ratio
        max_core_ratio = max(min_core_ratio, max_core_ratio)
        min_reliability = float(getattr(args, "wann_sparse_core_min_reliability", 0.0))
        core_candidate = core_candidate_mask & (reliability >= min_reliability)
        core_candidate = core_candidate | core_mask
        adaptive_core = _topk_ratio_mask(
            reliability * max_prob,
            core_candidate,
            min_core_ratio,
            max_core_ratio,
        )
        core_mask = core_mask | adaptive_core
    else:
        core_candidate = core_candidate_mask

    soft_band = target_valid & (~core_mask) & (reliability >= soft_thresh)
    ignore_mask = ~(core_mask | soft_band)

    core_weight = torch.zeros_like(reliability)
    core_min = float(getattr(args, "wann_core_min_weight", 0.8))
    core_weight[core_mask] = reliability[core_mask].clamp_min(core_min)
    core_weight = torch.nan_to_num(core_weight, nan=0.0, posinf=0.0, neginf=0.0)

    soft_weight = torch.zeros_like(reliability)
    soft_weight[soft_band] = reliability[soft_band]
    soft_weight = torch.nan_to_num(soft_weight, nan=0.0, posinf=0.0, neginf=0.0)

    valid_or_candidate = support_dilated
    if valid_or_candidate.any():
        effective_mass = reliability[valid_or_candidate].mean()
    else:
        effective_mass = reliability.mean() * 0.0

    low_mask = reliability < soft_thresh
    low_confident_mask = low_mask & (max_prob >= float(getattr(args, "wann_low_confident_thresh", 0.95)))
    low_agree_mask = low_mask & (agreement >= 0.5) & (ref_agreement >= 0.5)
    low_conflict_mask = low_confident_mask & ((agreement < 0.5) | (ref_agreement < 0.5))
    risk_mask = low_confident_mask | low_conflict_mask
    profile = {
        "effective_supervision_mass": effective_mass.detach(),
        "core_ratio": core_mask.float().mean().detach(),
        "soft_ratio": soft_band.float().mean().detach(),
        "low_weight_ratio": low_mask.float().mean().detach(),
        "seed_support_ratio": seed_support.float().mean().detach(),
        "core_candidate_ratio": core_candidate_mask.float().mean().detach(),
        "sparse_seed_protocol": reliability.new_tensor(1.0 if sparse_seed_protocol else 0.0).detach(),
        "block_like_protocol": reliability.new_tensor(1.0 if block_like_protocol else 0.0).detach(),
        "mean_reliability": reliability.mean().detach(),
        "entropy_low_r": entropy[low_mask].mean().detach() if low_mask.any() else entropy.mean().detach() * 0.0,
        "max_prob_low_r": max_prob[low_mask].mean().detach()
        if low_mask.any() else reliability.mean().detach() * 0.0,
        "foreground_ratio_low_r": (pred_label > 0).float()[low_mask].mean().detach()
        if low_mask.any() else reliability.mean().detach() * 0.0,
    }

    maps = WannMaps(
        target_label=target_label,
        core_mask=core_mask,
        soft_band=soft_band,
        ignore_mask=ignore_mask,
        reliability=reliability,
        core_weight=core_weight,
        soft_weight=soft_weight,
        valid_mask=valid,
        support_mask=support,
        seed_support_mask=seed_support,
        candidate_mask=(support_dilated | risk_mask),
        profile=profile,
    )
    maps.class_aware = torch.tensor(1.0 if class_aware else 0.0, device=label.device)
    maps.low_confident_mask = low_confident_mask
    maps.low_agree_mask = low_agree_mask
    maps.low_conflict_mask = low_conflict_mask
    maps.risk_mask = risk_mask
    return maps


def weighted_ce_loss(logits, target, weight, ignore_index):
    per_pixel = F.cross_entropy(logits, target.long(), reduction="none", ignore_index=ignore_index)
    valid_weight = weight * (target != ignore_index).float()
    denom = valid_weight.sum().clamp_min(1.0)
    return (per_pixel * valid_weight).sum() / denom


def soft_band_loss(logits, aux_logits, target, maps, ignore_index):
    prob = torch.softmax(logits, dim=1)
    weight = maps.soft_weight
    denom = weight.sum().clamp_min(1.0)

    if aux_logits is None:
        loss = logits.sum() * 0.0
    else:
        aux_prob = torch.softmax(aux_logits, dim=1)
        consistency = ((prob - aux_prob.detach()) ** 2).sum(dim=1)
        loss = (consistency * weight).sum() / denom

    valid_soft = maps.soft_band & (target != ignore_index)
    if valid_soft.any():
        ce = F.cross_entropy(logits, target.long(), reduction="none", ignore_index=ignore_index)
        valid_weight = weight * valid_soft.float()
        loss = loss + (ce * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    return loss


def consistency_loss(logits, aux_logits, mask):
    if aux_logits is None:
        return logits.sum() * 0.0
    prob = torch.softmax(logits, dim=1)
    aux_prob = torch.softmax(aux_logits, dim=1)
    weight = mask.float()
    denom = weight.sum().clamp_min(1.0)
    return (((prob - aux_prob.detach()) ** 2).sum(dim=1) * weight).sum() / denom
