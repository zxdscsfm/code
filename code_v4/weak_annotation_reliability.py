# -*- coding:utf-8 -*-
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class WannMaps:
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
    flat = gray.flatten(1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1).clamp_min(1e-6).view(-1, 1, 1)
    return (gray - mean) / std


def _support_mean_abs_distance(gray, label, support, num_classes):
    b, h, w = gray.shape
    dist = torch.zeros_like(gray)
    for idx in range(b):
        class_dists = []
        for class_id in range(int(num_classes)):
            cur_support = support[idx] & (label[idx] == class_id)
            if cur_support.any():
                center = gray[idx][cur_support].mean()
                scale = gray[idx][cur_support].std().clamp_min(0.25)
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
    if sup_type not in ["box", "block"]:
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


def build_wann_maps(image, label, logits, aux_logits, sup_type, img_class, num_classes, iter_num, args, ref_logits=None):
    del img_class  # The current implementation is annotation-profile driven.
    label = label.long()
    valid = label != int(num_classes)
    support = valid
    seed_support = _build_seed_support_mask(label, valid, sup_type, num_classes, args)

    soft_radius = _radius_for_sup_type(sup_type, args)
    support_dilated = _dilate(support, soft_radius)

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

    reliability = annotation_score * appearance_score * ambiguity_score
    reliability = reliability.clamp(0.0, float(getattr(args, "wann_r_max", 1.2)))

    core_thresh = float(getattr(args, "wann_core_thresh", 0.65))
    soft_thresh = float(getattr(args, "wann_soft_thresh", 0.25))
    core_mask = valid & (reliability >= core_thresh)
    soft_band = (~core_mask) & (reliability >= soft_thresh)
    ignore_mask = ~(core_mask | soft_band)

    core_weight = torch.zeros_like(reliability)
    core_min = float(getattr(args, "wann_core_min_weight", 0.8))
    core_weight[core_mask] = reliability[core_mask].clamp_min(core_min)

    soft_weight = torch.zeros_like(reliability)
    soft_weight[soft_band] = reliability[soft_band]

    valid_or_candidate = support_dilated
    if valid_or_candidate.any():
        effective_mass = reliability[valid_or_candidate].mean()
    else:
        effective_mass = reliability.mean() * 0.0

    low_mask = reliability < soft_thresh
    profile = {
        "effective_supervision_mass": effective_mass.detach(),
        "core_ratio": core_mask.float().mean().detach(),
        "soft_ratio": soft_band.float().mean().detach(),
        "low_weight_ratio": low_mask.float().mean().detach(),
        "mean_reliability": reliability.mean().detach(),
        "entropy_low_r": entropy[low_mask].mean().detach() if low_mask.any() else entropy.mean().detach() * 0.0,
        "max_prob_low_r": torch.softmax(logits.detach(), dim=1).max(dim=1)[0][low_mask].mean().detach()
        if low_mask.any() else reliability.mean().detach() * 0.0,
        "foreground_ratio_low_r": (torch.argmax(torch.softmax(logits.detach(), dim=1), dim=1) > 0).float()[low_mask].mean().detach()
        if low_mask.any() else reliability.mean().detach() * 0.0,
    }

    return WannMaps(
        core_mask=core_mask,
        soft_band=soft_band,
        ignore_mask=ignore_mask,
        reliability=reliability,
        core_weight=core_weight,
        soft_weight=soft_weight,
        valid_mask=valid,
        support_mask=support,
        seed_support_mask=seed_support,
        candidate_mask=support_dilated,
        profile=profile,
    )


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
