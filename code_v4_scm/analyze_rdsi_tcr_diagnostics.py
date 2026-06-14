# -*- coding:utf-8 -*-
"""Offline diagnostics for Typed Candidate Recall (TCR).

This script does not train or change model weights. It uses full GT only for
offline measurement. Observable masks/scores are computed from WANN maps,
student predictions, teacher predictions, and feature transfer compatibility.
"""

import argparse
import json
import math
import os
from contextlib import contextmanager

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory
from weak_annotation_reliability import build_wann_maps
from rgftd_reliability_distillation import select_rdsi_teacher_logits


PROSTATE_CLIENTS = ["client1", "client2", "client3", "client4", "client5", "client6"]
POLYP_CLIENTS = ["client1", "client2", "client3", "client4"]
FAZ_CLIENTS = ["client1", "client2", "client3", "client4", "client5"]


def default_setup(img_class):
    if img_class == "prostate":
        return PROSTATE_CLIENTS, 2, 1, 6
    if img_class == "polyp":
        return POLYP_CLIENTS, 2, 3, 4
    if img_class == "faz":
        return FAZ_CLIENTS, 2, 1, 5
    raise ValueError("Unsupported img_class: {}".format(img_class))


def parse_client_sup_types(args, num_clients):
    raw = str(getattr(args, "client_sup_types", "") or "").strip()
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
        if len(values) != num_clients:
            raise ValueError("--client_sup_types expects {} values, got {}".format(num_clients, len(values)))
        return values
    return ["scribble" for _ in range(num_clients)]


def get_client_sup_type(args, cid):
    values = getattr(args, "client_sup_type_list", None)
    if values is None:
        return "scribble"
    return values[int(cid)]


@contextmanager
def cpu_cuda_noop():
    old_module_cuda = torch.nn.Module.cuda
    old_tensor_cuda = torch.Tensor.cuda
    old_module_to = torch.nn.Module.to
    old_tensor_to = torch.Tensor.to

    def module_cuda(self, device=None):
        del device
        return self

    def tensor_cuda(self, device=None, non_blocking=False, memory_format=torch.preserve_format):
        del device, non_blocking, memory_format
        return self

    def is_cuda_target(args, kwargs):
        if args:
            first = args[0]
            if isinstance(first, str) and first.startswith("cuda"):
                return True
            if isinstance(first, torch.device) and first.type == "cuda":
                return True
        device = kwargs.get("device", None)
        if isinstance(device, str) and device.startswith("cuda"):
            return True
        if isinstance(device, torch.device) and device.type == "cuda":
            return True
        return False

    def module_to(self, *args, **kwargs):
        if is_cuda_target(args, kwargs):
            return self
        return old_module_to(self, *args, **kwargs)

    def tensor_to(self, *args, **kwargs):
        if is_cuda_target(args, kwargs):
            return self
        return old_tensor_to(self, *args, **kwargs)

    torch.nn.Module.cuda = module_cuda
    torch.Tensor.cuda = tensor_cuda
    torch.nn.Module.to = module_to
    torch.Tensor.to = tensor_to
    try:
        yield
    finally:
        torch.nn.Module.cuda = old_module_cuda
        torch.Tensor.cuda = old_tensor_cuda
        torch.nn.Module.to = old_module_to
        torch.Tensor.to = old_tensor_to


def primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def auxiliary_logits(model_out):
    if isinstance(model_out, (tuple, list)) and len(model_out) > 8:
        return model_out[8]
    return None


def rdsi_feature(model_out):
    if not isinstance(model_out, (tuple, list)) or len(model_out) < 3:
        raise ValueError("RDSI feature requires model output with de1 at index 2")
    feature = model_out[2]
    if not torch.is_tensor(feature) or feature.dim() != 4:
        raise ValueError("RDSI feature must be a 4D tensor")
    return feature


def to_image_tensor(image, img_class):
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    image = image.float()
    if img_class in ["prostate", "faz"]:
        if image.dim() == 3:
            image = image.unsqueeze(1)
    elif img_class in ["polyp", "odoc"]:
        if image.dim() == 3:
            image = image.unsqueeze(0)
    return image


def load_full_gt(root_path, rel_path):
    h5_path = os.path.join(root_path, rel_path)
    with h5py.File(h5_path, "r") as h5f:
        if "mask" not in h5f:
            raise KeyError("Full GT mask not found in {}".format(h5_path))
        return torch.from_numpy(h5f["mask"][:].astype(np.int64))


def make_model_args(args, cid):
    class Obj:
        pass
    obj = Obj()
    obj.model = args.model
    obj.in_chns = args.in_chns
    obj.num_classes = args.num_classes
    obj.min_num_clients = args.min_num_clients
    obj.cid = int(cid)
    obj.prompt = args.prompt
    obj.attention = args.attention
    obj.sup_type = get_client_sup_type(args, cid)
    obj.label_prompt = args.label_prompt
    obj.img_size = args.img_size
    obj.img_class = args.img_class
    return obj


def load_model(args, cid, checkpoint_path):
    model_args = make_model_args(args, cid)
    if args.device.type == "cpu":
        with cpu_cuda_noop():
            model = net_factory(
                model_args,
                net_type=args.model,
                in_chns=args.in_chns,
                class_num=args.num_classes,
            )
    else:
        model = net_factory(
            model_args,
            net_type=args.model,
            in_chns=args.in_chns,
            class_num=args.num_classes,
        )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def resolve_checkpoints(args, num_clients):
    paths = []
    for cid in range(num_clients):
        name = args.checkpoint_pattern.format(cid=cid, model=args.model)
        path = os.path.join(args.snapshot_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def normalized_entropy(prob):
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    return (entropy / max(math.log(float(prob.shape[1])), 1e-6)).clamp(0.0, 1.0)


def local_range_score(value, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return torch.zeros_like(value)
    max_v = F.max_pool2d(value.unsqueeze(1), 2 * radius + 1, stride=1, padding=radius)[:, 0]
    min_v = -F.max_pool2d((-value).unsqueeze(1), 2 * radius + 1, stride=1, padding=radius)[:, 0]
    return (max_v - min_v).clamp(0.0, 1.0)


def dilate(mask, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return mask
    out = F.max_pool2d(mask.float().unsqueeze(1), 2 * radius + 1, stride=1, padding=radius)
    return out[:, 0] > 0.5


def resize_like(mask_or_score, target_hw, mode="nearest"):
    if mask_or_score.shape[-2:] == target_hw:
        return mask_or_score
    x = mask_or_score.unsqueeze(1).float()
    if mode == "nearest":
        y = F.interpolate(x, size=target_hw, mode="nearest")[:, 0]
    else:
        y = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)[:, 0]
    return y


def masked_mean(value, mask):
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return 0.0
    return float((value.float() * mask_f).sum().detach().cpu().item() / denom.detach().cpu().item())


def ratio(numer_mask, denom_mask):
    denom = float(denom_mask.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 0.0
    return float((numer_mask.float() * denom_mask.float()).sum().detach().cpu().item() / denom)


def topk_mask(score, valid_mask, k):
    out = torch.zeros_like(valid_mask, dtype=torch.bool)
    if not isinstance(k, (list, tuple)):
        if k <= 0:
            return out
    bsz = score.shape[0]
    flat_score = score.flatten(1)
    flat_valid = valid_mask.flatten(1)
    flat_out = out.flatten(1)
    for b in range(bsz):
        valid_idx = torch.nonzero(flat_valid[b], as_tuple=False).flatten()
        if valid_idx.numel() <= 0:
            continue
        kk = min(int(k[b] if isinstance(k, (list, tuple)) else k), int(valid_idx.numel()))
        if kk <= 0:
            continue
        vals = flat_score[b, valid_idx]
        pick = torch.topk(vals, kk, largest=True).indices
        flat_out[b, valid_idx[pick]] = True
    return out


def extract_runtime_mask(args, name, shape, device):
    value = getattr(args, name, None)
    if torch.is_tensor(value):
        return value.to(device=device)
    return torch.zeros(shape, dtype=torch.bool, device=device)


def feature_transfer_compatibility(student_feature, teacher_features, target_hw, radius):
    student_feature = student_feature.detach().float()
    student_norm = F.normalize(student_feature, dim=1)
    compat = []
    for teacher_feature in teacher_features:
        tf = teacher_feature.detach().to(device=student_feature.device).float()
        if tf.shape[-2:] != student_feature.shape[-2:]:
            tf = F.interpolate(tf, size=student_feature.shape[-2:], mode="bilinear", align_corners=False)
        teacher_norm = F.normalize(tf, dim=1)
        score = (student_norm * teacher_norm).sum(dim=1).clamp(-1.0, 1.0)
        score = ((score + 1.0) * 0.5).clamp(0.0, 1.0)
        if radius > 0:
            score = F.avg_pool2d(score.unsqueeze(1), 2 * radius + 1, stride=1, padding=radius)[:, 0]
        score = resize_like(score, target_hw, mode="bilinear").clamp(0.0, 1.0)
        compat.append(score)
    return torch.stack(compat, dim=0)


def build_tcr_selection(args, student_logits, teacher_logits_list, teacher_features, student_feature, label, maps, gt):
    device = student_logits.device
    student_prob = F.softmax(student_logits.detach(), dim=1)
    student_conf = student_prob.max(dim=1)[0]
    student_entropy = normalized_entropy(student_prob)
    student_pred = student_prob.argmax(dim=1)
    if student_prob.shape[1] > 1:
        student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
        student_fg_mass = student_prob[:, 1:].sum(dim=1)
    else:
        student_fg_prob = torch.zeros_like(student_conf)
        student_fg_mass = torch.zeros_like(student_conf)

    valid_mask = getattr(maps, "valid_mask", torch.ones_like(student_pred, dtype=torch.bool))
    target_label = getattr(maps, "target_label", label)
    seed_support_mask = getattr(maps, "seed_support_mask", torch.zeros_like(student_pred, dtype=torch.bool))
    support_mask = getattr(maps, "support_mask", torch.zeros_like(student_pred, dtype=torch.bool))
    hard_core = getattr(args, "_rdsi_hard_core_mask", maps.core_mask & valid_mask)
    soft_core = getattr(args, "_rdsi_soft_core_mask", maps.core_mask & (~hard_core) & valid_mask)
    risk_region = getattr(args, "_rdsi_risk_region", None)
    if not torch.is_tensor(risk_region):
        risk_region = ((maps.ignore_mask | maps.soft_band) & (~hard_core) & valid_mask) | soft_core
    risk_region = risk_region.bool()

    radius = int(getattr(args, "rdsi_boundary_radius", -1))
    teacher_radius = int(getattr(args, "rgftd_teacher_foreground_radius", 2))
    boundary_radius = max(1, teacher_radius) if radius < 0 else radius
    transfer_stack = feature_transfer_compatibility(
        student_feature,
        teacher_features,
        student_conf.shape[-2:],
        boundary_radius,
    )

    weak_anchor = seed_support_mask & risk_region
    fg_seed = seed_support_mask & (target_label > 0)
    fg_support = support_mask & (target_label > 0)
    support_context = dilate(fg_seed | fg_support, int(getattr(args, "rgftd_spatial_support_radius", teacher_radius)))
    risk_like = (
        risk_region
        | getattr(maps, "soft_band", torch.zeros_like(risk_region))
        | getattr(maps, "low_conflict_mask", torch.zeros_like(risk_region))
        | support_context
    ) & valid_mask & (~hard_core)
    boundary_base = torch.maximum(
        (1.0 - (student_fg_mass - 0.5).abs() / max(float(getattr(args, "rdsi_boundary_uncertainty_width", 0.25)), 1e-6)).clamp(0.0, 1.0),
        local_range_score(student_fg_mass, boundary_radius),
    )
    boundary_base = torch.maximum(boundary_base, student_entropy).clamp(0.0, 1.0)
    student_uncertain = (
        (student_conf <= float(getattr(args, "rgftd_student_conf_thresh", 0.80)))
        | (student_entropy >= float(getattr(args, "rgftd_student_entropy_thresh", 0.35)))
    )
    student_foreground_repair = (
        (student_pred == 0)
        | student_uncertain
        | (student_fg_prob < float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35)))
    ) & risk_like
    student_fg_excess = (student_pred > 0) & risk_like

    fg_scores = []
    bg_scores = []
    bd_scores = []
    fg_indices = []
    bg_indices = []
    bd_indices = []
    core_damage_maps = []
    seed_conflict_maps = []
    teacher_pred_maps = []

    for t_idx, teacher_logits in enumerate(teacher_logits_list):
        teacher_prob = F.softmax(teacher_logits.detach() / max(float(getattr(args, "rgftd_temperature", 1.0)), 1e-6), dim=1)
        teacher_conf = teacher_prob.max(dim=1)[0]
        teacher_pred = teacher_prob.argmax(dim=1)
        teacher_entropy = normalized_entropy(teacher_prob)
        if teacher_prob.shape[1] > 1:
            teacher_fg_prob = teacher_prob[:, 1:].max(dim=1)[0]
            teacher_fg_mass = teacher_prob[:, 1:].sum(dim=1)
        else:
            teacher_fg_prob = torch.zeros_like(teacher_conf)
            teacher_fg_mass = torch.zeros_like(teacher_conf)
        teacher_bg_prob = teacher_prob[:, 0]
        teacher_fg_ready = (
            ((teacher_pred > 0) & (teacher_conf >= float(getattr(args, "rgftd_teacher_conf_thresh", 0.90))))
            | ((teacher_fg_prob >= float(getattr(args, "rgftd_teacher_fg_prob_thresh", 0.35))) & risk_like)
        )
        teacher_bg_ready = (
            (teacher_pred == 0)
            & (teacher_bg_prob >= float(getattr(args, "rgftd_teacher_bg_conf_thresh", 0.98)))
            & (teacher_fg_prob <= float(getattr(args, "rgftd_bg_max_fg_prob", 0.15)))
        )
        weak_agree = torch.where(
            weak_anchor,
            ((teacher_pred == target_label) & weak_anchor).float(),
            torch.ones_like(teacher_conf),
        )
        hard_core_f = hard_core.float()
        if student_prob.shape[1] > 1:
            student_bg_mass = student_prob[:, 0]
            teacher_bg_mass = teacher_prob[:, 0]
            core_drop = hard_core_f * (target_label > 0).float() * (student_fg_mass - teacher_fg_mass).clamp_min(0.0)
            core_bg_lift = hard_core_f * (target_label > 0).float() * (teacher_bg_mass - student_bg_mass).clamp_min(0.0)
        else:
            core_drop = torch.zeros_like(student_conf)
            core_bg_lift = torch.zeros_like(student_conf)
        core_flip = (hard_core & (teacher_pred != target_label) & (teacher_conf >= student_conf)).float()
        core_damage = torch.maximum(torch.maximum(core_drop, core_bg_lift), core_flip).clamp(0.0, 1.0)
        seed_conflict = (weak_anchor & (teacher_pred != target_label) & (teacher_conf >= student_conf)).float()
        core_safe = (1.0 - core_damage / max(float(getattr(args, "rdsi_core_damage_veto", 0.30)), 1e-6)).clamp(0.0, 1.0)
        safety = weak_agree * core_safe * (seed_conflict <= 0.5).float()
        transfer = transfer_stack[t_idx].detach().clamp(0.0, 1.0)
        reliability = (teacher_conf * (1.0 - teacher_entropy).clamp(0.0, 1.0)).clamp(0.0, 1.0)
        risk_evidence = ((1.0 - maps.reliability).clamp(0.0, 1.0) * torch.maximum(student_entropy, 1.0 - student_conf)).clamp(0.0, 1.0)
        fg_lift = ((teacher_fg_prob - student_fg_prob).clamp_min(0.0) / (1.0 - student_fg_prob).clamp_min(1e-6)).clamp(0.0, 1.0)
        bg_suppress = ((student_fg_prob - teacher_fg_prob).clamp_min(0.0) / student_fg_prob.clamp_min(1e-6)).clamp(0.0, 1.0)
        boundary_support = torch.maximum(
            (1.0 - (teacher_prob * student_prob).sum(dim=1)).clamp(0.0, 1.0) * boundary_base,
            torch.minimum(local_range_score(teacher_fg_mass, boundary_radius), local_range_score(student_fg_mass, boundary_radius)),
        ).clamp(0.0, 1.0)
        common = reliability * (0.5 + 0.5 * risk_evidence) * transfer * safety * risk_like.float()
        fg_score = (
            common
            * fg_lift
            * teacher_fg_ready.float()
            * student_foreground_repair.float()
        ).clamp(0.0, 1.0)
        bg_score = (
            common
            * bg_suppress
            * teacher_bg_ready.float()
            * student_fg_excess.float()
        ).clamp(0.0, 1.0)
        bd_score = (
            common
            * boundary_support
            * boundary_base
            * (~student_foreground_repair).float()
            * (~student_fg_excess).float()
        ).clamp(0.0, 1.0)
        fg_scores.append(fg_score)
        bg_scores.append(bg_score)
        bd_scores.append(bd_score)
        fg_indices.append(torch.full_like(student_pred, t_idx, dtype=torch.long))
        bg_indices.append(torch.full_like(student_pred, t_idx, dtype=torch.long))
        bd_indices.append(torch.full_like(student_pred, t_idx, dtype=torch.long))
        core_damage_maps.append(core_damage)
        seed_conflict_maps.append(seed_conflict)
        teacher_pred_maps.append(teacher_pred)

    fg_stack = torch.stack(fg_scores, dim=0)
    bg_stack = torch.stack(bg_scores, dim=0)
    bd_stack = torch.stack(bd_scores, dim=0)
    core_stack = torch.stack(core_damage_maps, dim=0)
    seed_stack = torch.stack(seed_conflict_maps, dim=0)
    pred_stack = torch.stack(teacher_pred_maps, dim=0)
    fg_score, fg_index = fg_stack.max(dim=0)
    bg_score, bg_index = bg_stack.max(dim=0)
    bd_score, bd_index = bd_stack.max(dim=0)

    # Fair comparison: keep the same number of active pixels as current RDSITP
    # for each sample, but allocate it by typed score mass before selecting.
    current_active = extract_runtime_mask(args, "_rdsi_active_mask", student_pred.shape, device)
    current_count = current_active.flatten(1).sum(dim=1).long().tolist()
    fg_mass = fg_score.flatten(1).sum(dim=1)
    bg_mass = bg_score.flatten(1).sum(dim=1)
    bd_mass = bd_score.flatten(1).sum(dim=1)
    total_mass = (fg_mass + bg_mass + bd_mass).clamp_min(1e-6)
    fg_k = [int(round(current_count[i] * float((fg_mass[i] / total_mass[i]).detach().cpu().item()))) for i in range(student_pred.shape[0])]
    bg_k = [int(round(current_count[i] * float((bg_mass[i] / total_mass[i]).detach().cpu().item()))) for i in range(student_pred.shape[0])]
    bd_k = [max(0, current_count[i] - fg_k[i] - bg_k[i]) for i in range(student_pred.shape[0])]

    fg_mask = topk_mask(fg_score, (fg_score > 0.0) & risk_like, fg_k)
    bg_mask = topk_mask(bg_score, (bg_score > 0.0) & risk_like & (~fg_mask), bg_k)
    bd_mask = topk_mask(bd_score, (bd_score > 0.0) & risk_like & (~fg_mask) & (~bg_mask), bd_k)
    active = fg_mask | bg_mask | bd_mask
    selected_index = torch.where(fg_mask, fg_index, torch.where(bg_mask, bg_index, bd_index))
    gathered_core = core_stack.gather(0, selected_index.unsqueeze(0)).squeeze(0)
    gathered_seed = seed_stack.gather(0, selected_index.unsqueeze(0)).squeeze(0)
    gathered_pred = pred_stack.gather(0, selected_index.unsqueeze(0)).squeeze(0)
    selected_score = torch.where(fg_mask, fg_score, torch.where(bg_mask, bg_score, bd_score))
    return {
        "tcr_active": active,
        "tcr_fg": fg_mask,
        "tcr_bg": bg_mask,
        "tcr_bd": bd_mask,
        "tcr_selected_teacher_pred": gathered_pred,
        "tcr_selected_core_damage": gathered_core,
        "tcr_selected_seed_conflict": gathered_seed,
        "tcr_selected_score": selected_score,
        "risk_region": risk_region,
    }


def summarize_selection(prefix, selected, selected_fg, selected_bg, selected_bd, selected_teacher_pred, gt, student_pred, risk_region, core_region, tcr_core_damage=None, tcr_seed_conflict=None):
    gt_fg = gt > 0
    student_fg = student_pred > 0
    fg_missing = risk_region & gt_fg & (~student_fg)
    fg_excess = risk_region & (~gt_fg) & student_fg
    boundary_truth = risk_region & dilate(gt_fg, 2) & dilate(~gt_fg, 2) & (~fg_missing) & (~fg_excess)
    teacher_correct = selected & (selected_teacher_pred == gt)
    out = {
        "{}_active_ratio".format(prefix): float(selected.float().mean().detach().cpu().item()),
        "{}_fg_active_ratio".format(prefix): float(selected_fg.float().mean().detach().cpu().item()),
        "{}_bg_active_ratio".format(prefix): float(selected_bg.float().mean().detach().cpu().item()),
        "{}_bd_active_ratio".format(prefix): float(selected_bd.float().mean().detach().cpu().item()),
        "{}_fg_missing_recall".format(prefix): ratio(selected_fg & fg_missing, fg_missing),
        "{}_fg_excess_recall".format(prefix): ratio(selected_bg & fg_excess, fg_excess),
        "{}_boundary_recall".format(prefix): ratio(selected_bd & boundary_truth, boundary_truth),
        "{}_fg_precision".format(prefix): ratio(selected_fg & fg_missing, selected_fg),
        "{}_bg_precision".format(prefix): ratio(selected_bg & fg_excess, selected_bg),
        "{}_bd_precision".format(prefix): ratio(selected_bd & boundary_truth, selected_bd),
        "{}_teacher_correct_on_active".format(prefix): ratio(teacher_correct, selected),
        "{}_teacher_correct_on_fg".format(prefix): ratio((selected_teacher_pred > 0) & gt_fg & selected_fg, selected_fg),
        "{}_teacher_correct_on_bg".format(prefix): ratio((selected_teacher_pred == 0) & (~gt_fg) & selected_bg, selected_bg),
        "{}_risk_error_coverage".format(prefix): ratio(selected & (student_pred != gt), risk_region & (student_pred != gt)),
        "{}_selected_error_precision".format(prefix): ratio(selected & (student_pred != gt), selected),
        "{}_risk_pixels".format(prefix): int(risk_region.float().sum().detach().cpu().item()),
        "{}_fg_missing_pixels".format(prefix): int(fg_missing.float().sum().detach().cpu().item()),
        "{}_fg_excess_pixels".format(prefix): int(fg_excess.float().sum().detach().cpu().item()),
    }
    if tcr_core_damage is not None:
        out["{}_core_damage_selected".format(prefix)] = masked_mean(tcr_core_damage, selected)
    if tcr_seed_conflict is not None:
        out["{}_seed_conflict_selected".format(prefix)] = masked_mean(tcr_seed_conflict, selected)
    if bool(core_region.any().detach().cpu().item()):
        out["{}_core_error_student".format(prefix)] = ratio(student_pred != gt, core_region)
    else:
        out["{}_core_error_student".format(prefix)] = 0.0
    return out


def corr_summary(df, x, y):
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3:
        return {"pearson": 0.0, "spearman": 0.0, "n": int(len(sub))}
    return {
        "pearson": float(sub[x].corr(sub[y], method="pearson")),
        "spearman": float(sub[x].corr(sub[y], method="spearman")),
        "n": int(len(sub)),
    }


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--img_class", choices=["prostate", "polyp", "faz"], required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint_pattern", default="client_{cid}_async_{model}_best_model.pth")
    parser.add_argument("--client_sup_types", default="")
    parser.add_argument("--max_cases_per_client", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--analysis_iter", type=int, default=1600)
    parser.add_argument("--exclude_same_client", type=int, default=1)
    parser.add_argument("--model", default="unet_univ5")
    parser.add_argument("--prompt", default="universal")
    parser.add_argument("--attention", default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=384)

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
    parser.add_argument("--rgftd_teacher_fg_topk_ratio", type=float, default=0.002)
    parser.add_argument("--rgftd_min_foreground_pixels", type=int, default=8)
    parser.add_argument("--rgftd_teacher_fg_topk_min_pixels", type=int, default=8)
    parser.add_argument("--rgftd_teacher_student_fg_margin", type=float, default=0.05)
    parser.add_argument("--rgftd_teacher_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rgftd_teacher_bg_conf_thresh", type=float, default=0.98)
    parser.add_argument("--rgftd_bg_max_fg_prob", type=float, default=0.15)
    parser.add_argument("--rgftd_max_bg_fg_ratio", type=float, default=1.0)
    parser.add_argument("--rgftd_spatial_support_enabled", type=int, default=1)
    parser.add_argument("--rgftd_spatial_support_radius", type=int, default=2)
    parser.add_argument("--rgftd_spatial_candidate_weight", type=float, default=1.0)
    parser.add_argument("--rgftd_spatial_near_seed_weight", type=float, default=0.75)
    parser.add_argument("--rgftd_spatial_far_weight", type=float, default=0.15)
    parser.add_argument("--rgftd_teacher_validation_enabled", type=int, default=0)
    parser.add_argument("--rgftd_student_conf_thresh", type=float, default=0.80)
    parser.add_argument("--rgftd_student_entropy_thresh", type=float, default=0.35)

    parser.add_argument("--rdsi_entropy_increase_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_entropy_increase_scale", type=float, default=0.35)
    parser.add_argument("--rdsi_fg_excess_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_fg_excess_scale", type=float, default=0.20)
    parser.add_argument("--rdsi_residual_alpha", type=float, default=0.35)
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
            raise RuntimeError("--device cuda requested but CUDA is not available")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    clients, num_classes, in_chns, min_num_clients = default_setup(args.img_class)
    args.num_classes = num_classes
    args.in_chns = in_chns
    args.min_num_clients = min_num_clients
    args.client_sup_type_list = parse_client_sup_types(args, len(clients))
    out_dir = args.output_dir or os.path.join(args.snapshot_path, "rdsi_tcr_diagnostics")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_paths = resolve_checkpoints(args, len(clients))
    models = [load_model(args, cid, path) for cid, path in enumerate(checkpoint_paths)]
    rows = []

    with torch.no_grad():
        for target_cid, client in enumerate(clients):
            dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=client,
                sup_type=get_client_sup_type(args, target_cid),
                img_class=args.img_class,
            )
            base_sample_list = dataset.sample_list
            if args.max_cases_per_client > 0:
                keep = list(range(min(args.max_cases_per_client, len(dataset))))
                dataset = Subset(dataset, keep)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            processed = 0
            for batch in loader:
                images = to_image_tensor(batch["image"], args.img_class).to(args.device)
                weak_label = batch["label"].long().to(args.device)
                idxs = batch["idx"].tolist()
                rel_paths = [base_sample_list[int(idx)] for idx in idxs]
                gt = torch.stack([load_full_gt(args.root_path, path) for path in rel_paths], dim=0).long().to(args.device)

                student_out = models[target_cid](images)
                student_logits = primary_logits(student_out)
                student_feature = rdsi_feature(student_out)
                student_aux = auxiliary_logits(student_out)
                maps = build_wann_maps(
                    image=images,
                    label=weak_label,
                    logits=student_logits,
                    aux_logits=student_aux,
                    sup_type=get_client_sup_type(args, target_cid),
                    img_class=args.img_class,
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )
                teacher_logits_list = []
                teacher_feature_list = []
                teacher_ids = []
                teacher_model_cids = []
                for teacher_cid, model in enumerate(models):
                    if int(args.exclude_same_client) == 1 and teacher_cid == target_cid:
                        continue
                    teacher_out = model(images)
                    teacher_logits_list.append(primary_logits(teacher_out).detach())
                    teacher_feature_list.append(rdsi_feature(teacher_out).detach())
                    teacher_ids.append(teacher_cid)
                    teacher_model_cids.append(teacher_cid)
                if not teacher_logits_list:
                    continue

                _, profile = select_rdsi_teacher_logits(
                    student_logits,
                    teacher_logits_list,
                    teacher_ids,
                    weak_label,
                    maps,
                    args,
                    args.analysis_iter,
                    student_feature=student_feature,
                    teacher_feature_list=teacher_feature_list,
                )
                student_prob = F.softmax(student_logits.detach(), dim=1)
                student_pred = student_prob.argmax(dim=1)
                selected_stack_index = getattr(args, "_rdsi_selected_teacher_stack_index")
                pred_stack = torch.stack(
                    [F.softmax(logits.detach(), dim=1).argmax(dim=1) for logits in teacher_logits_list],
                    dim=0,
                )
                current_teacher_pred = pred_stack.gather(0, selected_stack_index.unsqueeze(0)).squeeze(0)
                current_active = extract_runtime_mask(args, "_rdsi_active_mask", student_pred.shape, args.device)
                current_fg = extract_runtime_mask(args, "_rdsi_fg_repair_active_mask", student_pred.shape, args.device)
                current_bg = extract_runtime_mask(args, "_rdsi_bg_suppress_active_mask", student_pred.shape, args.device)
                current_bd = extract_runtime_mask(args, "_rdsi_boundary_active_mask", student_pred.shape, args.device)
                current_core_damage = getattr(args, "_rdsi_core_damage_proxy", torch.zeros_like(student_pred, dtype=torch.float32))
                current_seed_conflict = getattr(args, "_rdsi_seed_conflict", torch.zeros_like(student_pred, dtype=torch.float32))

                tcr = build_tcr_selection(
                    args,
                    student_logits,
                    teacher_logits_list,
                    teacher_feature_list,
                    student_feature,
                    weak_label,
                    maps,
                    gt,
                )
                risk_region = tcr["risk_region"]
                core_region = getattr(args, "_rdsi_hard_core_mask", maps.core_mask)
                for b, rel_path in enumerate(rel_paths):
                    row = {
                        "img_class": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "case": rel_path,
                        "sample_idx": int(idxs[b]),
                    }
                    row.update(summarize_selection(
                        "current",
                        current_active[b:b + 1],
                        current_fg[b:b + 1],
                        current_bg[b:b + 1],
                        current_bd[b:b + 1],
                        current_teacher_pred[b:b + 1],
                        gt[b:b + 1],
                        student_pred[b:b + 1],
                        risk_region[b:b + 1],
                        core_region[b:b + 1],
                        current_core_damage[b:b + 1],
                        current_seed_conflict[b:b + 1],
                    ))
                    row.update(summarize_selection(
                        "tcr",
                        tcr["tcr_active"][b:b + 1],
                        tcr["tcr_fg"][b:b + 1],
                        tcr["tcr_bg"][b:b + 1],
                        tcr["tcr_bd"][b:b + 1],
                        tcr["tcr_selected_teacher_pred"][b:b + 1],
                        gt[b:b + 1],
                        student_pred[b:b + 1],
                        risk_region[b:b + 1],
                        core_region[b:b + 1],
                        tcr["tcr_selected_core_damage"][b:b + 1],
                        tcr["tcr_selected_seed_conflict"][b:b + 1],
                    ))
                    row["current_rdsi_accept_ratio"] = float(profile.get("rdsi_accept_ratio", torch.tensor(0.0)).detach().cpu().item())
                    row["current_rdsi_reject_ratio"] = float(profile.get("rdsi_reject_ratio", torch.tensor(0.0)).detach().cpu().item())
                    row["current_rdsi_fg_purity"] = float(profile.get("rdsi_fg_missing_purity", torch.tensor(0.0)).detach().cpu().item())
                    row["current_rdsi_bg_purity"] = float(profile.get("rdsi_fg_excess_purity", torch.tensor(0.0)).detach().cpu().item())
                    row["current_rdsi_bd_purity"] = float(profile.get("rdsi_boundary_purity", torch.tensor(0.0)).detach().cpu().item())
                    rows.append(row)
                processed += len(idxs)
                print("processed {} target_client={} samples={}".format(args.img_class, client, processed), flush=True)

    df = pd.DataFrame(rows)
    sample_csv = os.path.join(out_dir, "tcr_vs_current_samples.csv")
    df.to_csv(sample_csv, index=False)
    metric_cols = [col for col in df.columns if col.startswith("current_") or col.startswith("tcr_")]
    summary = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "analysis_iter": args.analysis_iter,
        "num_samples": int(len(df)),
        "files": {"sample_csv": sample_csv},
    }
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary[col + "_mean"] = float(df[col].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(df) else 0.0
    for name in ["fg_missing_recall", "fg_excess_recall", "boundary_recall", "teacher_correct_on_active", "selected_error_precision", "risk_error_coverage", "core_damage_selected"]:
        c = "current_" + name
        t = "tcr_" + name
        if c in df and t in df:
            summary["delta_tcr_minus_current_" + name] = float((df[t] - df[c]).replace([np.inf, -np.inf], np.nan).dropna().mean())
    if "current_rdsi_fg_purity" in df and "tcr_fg_missing_recall" in df:
        summary["correlations"] = {
            "current_fg_purity_vs_tcr_fg_missing_recall": corr_summary(df, "current_rdsi_fg_purity", "tcr_fg_missing_recall"),
            "current_fg_purity_vs_current_fg_missing_recall": corr_summary(df, "current_rdsi_fg_purity", "current_fg_missing_recall"),
        }
    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_json(os.path.join(out_dir, "summary_flat.json"), orient="records", force_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
