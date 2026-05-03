# -*- coding:utf-8 -*-
import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss, MSELoss, L1Loss
from info_nce import InfoNCE
from torch.utils.data import DataLoader
from torchvision import transforms

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from collections import OrderedDict
from logging import DEBUG, INFO
import timeit
import copy
from torch.cuda.amp import autocast, GradScaler

from dataloaders import utils
from dataloaders.dataset import (
    BaseDataSets,
    RandomGenerator,
    infer_unlabeled_value_from_label,
)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds
from utils.gate_crf_loss import ModelLossSemsegGatedCRF
from flower_common_v4 import (BaseClient, MyModel, fit_metrics_aggregation_fn, TreeEnergyLoss, MScaleRecurveTreeEnergyLoss, evaluate, get_evaluate_fn,
                        get_strategy, MyServer, VAL_METRICS, get_fedrep_local_keys, get_evaluate_metrics_aggregation_fn,
                        get_bn_stats, inject_prototype_bank_into_config, PROTOTYPE_CLASS_NAMES)

writer = None


def parse_geometry_weight_list(weight_str, expected_len):
    values = [float(x) for x in str(weight_str).split(',') if x.strip() != '']
    if len(values) != expected_len:
        raise ValueError(
            'Expected {} geometry weights, got {} from {}'.format(
                expected_len, len(values), weight_str
            )
        )
    return values


def build_geometry_weight_map(geometry_bin, weight_values):
    weight_tensor = torch.tensor(weight_values, device=geometry_bin.device, dtype=torch.float32)
    geometry_bin = geometry_bin.long().clamp(0, len(weight_values) - 1)
    return weight_tensor[geometry_bin]


def build_linear_geometry_weight_list(num_bins, alpha=0.8, floor=0.2):
    num_bins = int(num_bins)
    alpha = float(alpha)
    floor = float(floor)
    if num_bins <= 0:
        raise ValueError('num_bins must be positive, got {}'.format(num_bins))
    if num_bins == 1:
        return [1.0]
    values = []
    for bin_idx in range(num_bins):
        progress = float(bin_idx) / float(max(num_bins - 1, 1))
        value = 1.0 - (alpha * progress)
        value = max(floor, min(1.0, value))
        values.append(float(value))
    return values


def weighted_pseudo_dice_loss(inputs, target, pixel_weight=None):
    smooth = 1e-5
    num_classes = inputs.shape[1]
    if pixel_weight is None:
        pixel_weight = torch.ones_like(target, dtype=inputs.dtype, device=inputs.device)
    else:
        pixel_weight = pixel_weight.to(device=inputs.device, dtype=inputs.dtype)

    loss = 0.0
    for class_idx in range(num_classes):
        target_mask = (target == class_idx).to(dtype=inputs.dtype)
        score = inputs[:, class_idx]
        intersect = torch.sum(score * target_mask * pixel_weight)
        target_sum = torch.sum(target_mask * pixel_weight)
        score_sum = torch.sum(score * score * pixel_weight)
        dice = (2.0 * intersect + smooth) / (score_sum + target_sum + smooth)
        loss += 1.0 - dice
    return loss / num_classes


def weighted_reverse_kl_loss(logits, target_prob, pixel_weight=None):
    target_prob = target_prob.detach().to(device=logits.device, dtype=logits.dtype).clamp_min(1e-8)
    log_pred = F.log_softmax(logits, dim=1)
    loss_map = (target_prob * (target_prob.log() - log_pred)).sum(dim=1)
    if pixel_weight is None:
        return loss_map.mean()
    pixel_weight = pixel_weight.to(device=logits.device, dtype=logits.dtype)
    weight_sum = pixel_weight.sum()
    if weight_sum.item() <= 0:
        return logits.new_tensor(0.0)
    return (loss_map * pixel_weight).sum() / weight_sum


def morphology_boundary(binary_mask, kernel_size=3):
    if kernel_size <= 1:
        return binary_mask.bool()
    mask = binary_mask.float().unsqueeze(1)
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=padding)
    boundary = (dilated - eroded) > 0
    return boundary.squeeze(1)


def boundary_dice_loss(pred_prob, target_mask, valid_mask=None):
    pred_prob = pred_prob.float()
    target_mask = target_mask.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(target_mask, dtype=pred_prob.dtype, device=pred_prob.device)
    else:
        valid_mask = valid_mask.to(device=pred_prob.device, dtype=pred_prob.dtype)
    valid_sum = valid_mask.sum()
    if valid_sum.item() <= 0:
        return pred_prob.new_tensor(0.0)
    smooth = 1e-5
    intersect = torch.sum(pred_prob * target_mask * valid_mask)
    pred_sum = torch.sum(pred_prob * valid_mask)
    target_sum = torch.sum(target_mask * valid_mask)
    dice = (2.0 * intersect + smooth) / (pred_sum + target_sum + smooth)
    return 1.0 - dice


def compute_adaptive_pseudo_reliability(local_prob, global_prob, gamma_prob, gamma_conf):
    local_prob = local_prob.detach()
    global_prob = global_prob.detach()

    conf_local, pred_local = torch.max(local_prob, dim=1)
    conf_global, pred_global = torch.max(global_prob, dim=1)
    prob_gap = 0.5 * torch.abs(local_prob - global_prob).sum(dim=1)
    conf_gap = torch.abs(conf_local - conf_global)
    agree_lg = pred_local == pred_global

    score = torch.exp(-float(gamma_prob) * prob_gap) * torch.exp(-float(gamma_conf) * conf_gap)
    score = score * (0.5 + 0.5 * agree_lg.float())

    return {
        'local_prob': local_prob,
        'global_prob': global_prob,
        'conf_local': conf_local.clamp(0.0, 1.0),
        'conf_global': conf_global.clamp(0.0, 1.0),
        'prob_gap': prob_gap.clamp(0.0, 1.0),
        'conf_gap': conf_gap.clamp(0.0, 1.0),
        'agree_lg': agree_lg,
        'score': score.clamp(0.0, 1.0),
    }


def compute_masked_class_prior(prob_map, valid_mask):
    valid_weight = valid_mask.unsqueeze(1).float()
    denom = valid_weight.sum().clamp_min(1.0)
    return (prob_map * valid_weight).sum(dim=(0, 2, 3)) / denom


def resize_long_mask(mask_tensor, size_hw):
    return F.interpolate(mask_tensor.unsqueeze(1).float(), size=size_hw, mode='nearest').squeeze(1).long()


def decode_prototype_bank_from_config(config, device):
    proto_tensors = []
    for class_name in PROTOTYPE_CLASS_NAMES:
        key = f"proto_bank_{class_name}"
        if key not in config:
            return None
        proto_np = fl.common.bytes_to_ndarray(config[key]).astype(np.float32)
        proto_tensors.append(torch.from_numpy(proto_np).to(device))
    return torch.stack(proto_tensors, dim=0)


def masked_cross_entropy_loss(logits, target, valid_mask, ignore_index):
    if valid_mask.sum().item() <= 0:
        return logits.new_tensor(0.0)
    target_masked = target.clone()
    target_masked[~valid_mask] = ignore_index
    return F.cross_entropy(logits, target_masked.long(), ignore_index=ignore_index)


def compute_local_weak_prototypes(model_wrapper, batches, args):
    model_wrapper.eval()
    accum = {class_name: None for class_name in PROTOTYPE_CLASS_NAMES}
    counts = {class_name: 0 for class_name in PROTOTYPE_CLASS_NAMES}

    with torch.no_grad():
        for sampled_batch in batches:
            if args.img_class in ['faz', 'prostate']:
                volume_batch = sampled_batch['image'].unsqueeze(1).cuda()
                label_batch = sampled_batch['label'].cuda()
            else:
                volume_batch = sampled_batch['image'].cuda()
                label_batch = sampled_batch['label'].cuda()

            feature = model_wrapper.model.encoder(volume_batch)[-1]
            n, c, h, w = feature.shape
            label_low = resize_long_mask(label_batch, (h, w))
            feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, c)
            label_flat = label_low.reshape(-1)
            for class_idx, class_name in enumerate(PROTOTYPE_CLASS_NAMES):
                class_mask = label_flat == class_idx
                class_count = int(class_mask.sum().item())
                if class_count <= 0:
                    continue
                class_sum = feature_flat[class_mask].sum(dim=0)
                if accum[class_name] is None:
                    accum[class_name] = class_sum
                else:
                    accum[class_name] += class_sum
                counts[class_name] += class_count

    prototypes = {}
    for class_name in PROTOTYPE_CLASS_NAMES:
        if counts[class_name] <= 0 or accum[class_name] is None:
            prototypes[class_name] = None
        else:
            prototypes[class_name] = (accum[class_name] / float(counts[class_name])).detach().cpu().numpy().astype(np.float32)
    model_wrapper.train()
    return prototypes, counts


def select_top_margin_mask(raw_encoder_feature, pseudo_probs, weak_label_batch, prototype_bank, ignore_index, topk_ratio):
    n, c, h, w = raw_encoder_feature.shape
    probs_low = F.interpolate(pseudo_probs, size=(h, w), mode='bilinear', align_corners=False)
    pred_low = torch.argmax(probs_low, dim=1)
    weak_low = resize_long_mask(weak_label_batch, (h, w))
    candidate_mask = (weak_low == ignore_index) & (pred_low > 0)
    if candidate_mask.sum().item() <= 0:
        zero_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
        return zero_mask, pred_low, probs_low, raw_encoder_feature.new_tensor(0.0), 0, 0

    feature_flat = raw_encoder_feature.permute(0, 2, 3, 1).reshape(-1, c)
    feature_flat = F.normalize(feature_flat, dim=1)
    proto_bank = F.normalize(prototype_bank.to(device=raw_encoder_feature.device, dtype=raw_encoder_feature.dtype), dim=1)
    sims = torch.matmul(feature_flat, proto_bank.t()).reshape(n, h, w, -1)

    pred_sim = sims.gather(dim=3, index=pred_low.unsqueeze(-1)).squeeze(-1)
    other_sims = sims.clone()
    other_sims.scatter_(3, pred_low.unsqueeze(-1), float('-inf'))
    other_max = other_sims.max(dim=3).values
    margin = pred_sim - other_max

    selected_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    candidate_idx = candidate_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    candidate_margin = margin.reshape(-1)[candidate_idx]
    k = max(1, int(np.ceil(candidate_idx.numel() * float(topk_ratio))))
    topk_values, topk_pos = torch.topk(candidate_margin, k=k, largest=True)
    selected_flat = selected_mask.reshape(-1)
    selected_flat[candidate_idx[topk_pos]] = True
    selected_mask = selected_flat.view_as(candidate_mask)
    margin_mean = topk_values.mean() if topk_values.numel() > 0 else raw_encoder_feature.new_tensor(0.0)
    return selected_mask, pred_low, probs_low, margin_mean, int(candidate_idx.numel()), int(k)


def build_local_seed_bank(raw_encoder_feature, weak_label_batch, corrected_prob, score_map, ignore_index, reliable_score_min):
    n, c, h, w = raw_encoder_feature.shape
    weak_low = resize_long_mask(weak_label_batch, (h, w))
    corrected_low = F.interpolate(corrected_prob, size=(h, w), mode='bilinear', align_corners=False)
    score_low = F.interpolate(score_map.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)
    pred_low = torch.argmax(corrected_low, dim=1)
    conf_low = torch.max(corrected_low, dim=1).values
    unlabeled_low = weak_low == ignore_index

    feature_flat = F.normalize(raw_encoder_feature.permute(0, 2, 3, 1).reshape(-1, c), dim=1)
    weak_flat = weak_low.reshape(-1)
    pred_flat = pred_low.reshape(-1)
    conf_flat = conf_low.reshape(-1)
    score_flat = score_low.reshape(-1)
    unlabeled_flat = unlabeled_low.reshape(-1)

    prototype_list = []
    seed_prob_low = corrected_low.new_zeros((n, corrected_low.shape[1] - 1, h, w))
    reliable_seed_mask = {}
    fg_seed_valid = torch.zeros(corrected_low.shape[1] - 1, device=corrected_low.device, dtype=torch.bool)

    for class_idx in range(corrected_low.shape[1]):
        class_mask = weak_flat == class_idx
        if class_idx > 0:
            reliable_mask = (
                unlabeled_flat
                & (pred_flat == class_idx)
                & (conf_flat >= 0.75)
                & (score_flat >= reliable_score_min)
            )
            class_mask = class_mask | reliable_mask
            reliable_seed_mask[class_idx] = reliable_mask.view(n, h, w)
        if class_mask.sum().item() <= 0 and class_idx > 0:
            fallback_mask = (
                unlabeled_flat
                & (pred_flat == class_idx)
                & (conf_flat >= 0.90)
            )
            class_mask = class_mask | fallback_mask
            reliable_seed_mask[class_idx] = fallback_mask.view(n, h, w)
        if class_mask.sum().item() <= 0:
            prototype_list.append(None)
            continue
        class_proto = feature_flat[class_mask].mean(dim=0)
        prototype_list.append(F.normalize(class_proto, dim=0))

    if prototype_list[0] is None:
        prototype_list[0] = F.normalize(feature_flat.mean(dim=0), dim=0)
    for class_idx in range(1, corrected_low.shape[1]):
        if prototype_list[class_idx] is None:
            prototype_list[class_idx] = prototype_list[0].clone()
        reliable_mask = reliable_seed_mask.get(class_idx, torch.zeros((n, h, w), device=corrected_low.device, dtype=torch.bool))
        weak_seed_mask = weak_low == class_idx
        seed_mask = weak_seed_mask | reliable_mask
        fg_seed_valid[class_idx - 1] = bool(seed_mask.any().item())
        seed_prob_low[:, class_idx - 1] = corrected_low[:, class_idx] * seed_mask.float()

    proto_bank = torch.stack(prototype_list, dim=0).to(device=raw_encoder_feature.device, dtype=raw_encoder_feature.dtype)
    return proto_bank, seed_prob_low, fg_seed_valid


def propagate_local_seed_prob(raw_encoder_feature, seed_prob_low, kernel_size, affinity_temp):
    if kernel_size <= 1:
        return seed_prob_low
    pad = kernel_size // 2
    normalized_feature = F.normalize(raw_encoder_feature, dim=1)
    b, c, h, w = normalized_feature.shape
    feature_patch = F.unfold(normalized_feature, kernel_size=kernel_size, padding=pad).view(
        b, c, kernel_size * kernel_size, h, w
    )
    center_feature = normalized_feature.unsqueeze(2)
    affinity = (center_feature * feature_patch).sum(dim=1)
    affinity = torch.relu(affinity)
    affinity = torch.softmax(affinity * affinity_temp, dim=1)

    seed_patch = F.unfold(seed_prob_low, kernel_size=kernel_size, padding=pad).view(
        b, seed_prob_low.shape[1], kernel_size * kernel_size, h, w
    )
    propagated = (affinity.unsqueeze(1) * seed_patch).sum(dim=2)
    norm = propagated.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return propagated / norm


def build_seed_support_guided_prob(
    raw_encoder_feature,
    weak_label_batch,
    corrected_prob,
    score_map,
    ignore_index,
    reliable_score_min,
    kernel_size,
    affinity_temp,
    blend_alpha,
):
    n, _, h, w = raw_encoder_feature.shape
    proto_bank, seed_prob_low, fg_seed_valid = build_local_seed_bank(
        raw_encoder_feature,
        weak_label_batch,
        corrected_prob,
        score_map,
        ignore_index,
        reliable_score_min,
    )
    if not bool(fg_seed_valid.any().item()):
        zero_support = corrected_prob.new_zeros(corrected_prob.shape[0], corrected_prob.shape[2], corrected_prob.shape[3])
        return corrected_prob, zero_support, zero_support

    feature_flat = F.normalize(raw_encoder_feature.permute(0, 2, 3, 1).reshape(-1, raw_encoder_feature.shape[1]), dim=1)
    proto_bank = F.normalize(proto_bank, dim=1)
    proto_sims = torch.matmul(feature_flat, proto_bank.t()).reshape(n, h, w, -1).permute(0, 3, 1, 2)
    proto_fg_logits = proto_sims[:, 1:] - proto_sims[:, 0:1]
    fg_valid_mask = fg_seed_valid.view(1, -1, 1, 1)
    proto_fg_logits = proto_fg_logits.masked_fill(~fg_valid_mask, -1e4)
    proto_fg_prob = torch.softmax(proto_fg_logits, dim=1) * fg_valid_mask.float()
    proto_fg_prob = proto_fg_prob / proto_fg_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    proto_support_margin = torch.max(proto_fg_logits, dim=1).values
    proto_support_strength = torch.sigmoid(4.0 * proto_support_margin)

    propagated_fg_prob = propagate_local_seed_prob(
        raw_encoder_feature,
        seed_prob_low,
        kernel_size=kernel_size,
        affinity_temp=affinity_temp,
    )
    propagated_fg_strength = torch.max(propagated_fg_prob, dim=1).values
    support_strength_low = torch.maximum(proto_support_strength, propagated_fg_strength)

    fused_fg_low = 0.5 * proto_fg_prob + 0.5 * propagated_fg_prob
    base_fg_low = F.interpolate(corrected_prob[:, 1:], size=(h, w), mode='bilinear', align_corners=False)
    support_fg_low = fused_fg_low * support_strength_low.unsqueeze(1)
    blended_fg_low = blend_alpha * base_fg_low + (1.0 - blend_alpha) * support_fg_low
    blended_fg = F.interpolate(blended_fg_low, size=corrected_prob.shape[-2:], mode='bilinear', align_corners=False)
    support_strength = F.interpolate(
        support_strength_low.unsqueeze(1),
        size=corrected_prob.shape[-2:],
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)
    propagated_fg_strength = F.interpolate(
        propagated_fg_strength.unsqueeze(1),
        size=corrected_prob.shape[-2:],
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)

    final_prob = corrected_prob.clone()
    final_prob[:, 1:] = blended_fg.clamp_min(0.0)
    fg_sum = final_prob[:, 1:].sum(dim=1, keepdim=True).clamp(max=1.0 - 1e-6)
    bg_prob = (1.0 - fg_sum).clamp_min(1e-6)
    final_prob = torch.cat([bg_prob, final_prob[:, 1:]], dim=1)
    final_prob = final_prob / final_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

    return final_prob, support_strength, propagated_fg_strength


def compute_linear_cka(feature_a, feature_b, eps=1e-8):
    if feature_a.dim() > 2:
        feature_a = feature_a.reshape(feature_a.shape[0], -1)
    if feature_b.dim() > 2:
        feature_b = feature_b.reshape(feature_b.shape[0], -1)
    feature_a = feature_a.float()
    feature_b = feature_b.float()
    gram_a = torch.matmul(feature_a, feature_a.t())
    gram_b = torch.matmul(feature_b, feature_b.t())
    n = gram_a.shape[0]
    if n <= 1:
        return feature_a.new_tensor(1.0)
    identity = torch.eye(n, device=gram_a.device, dtype=gram_a.dtype)
    ones = torch.ones((n, n), device=gram_a.device, dtype=gram_a.dtype) / float(n)
    center = identity - ones
    gram_a = torch.matmul(torch.matmul(center, gram_a), center)
    gram_b = torch.matmul(torch.matmul(center, gram_b), center)
    hsic = torch.sum(gram_a * gram_b)
    norm_a = torch.sqrt(torch.sum(gram_a * gram_a))
    norm_b = torch.sqrt(torch.sum(gram_b * gram_b))
    return hsic / (norm_a * norm_b + eps)


def compute_teacher_student_diagnostics(student_outputs, teacher_outputs, student_feature, teacher_feature):
    student_logits = student_outputs.reshape(student_outputs.shape[0], -1).float()
    teacher_logits = teacher_outputs.reshape(teacher_outputs.shape[0], -1).float()
    logit_cosine = F.cosine_similarity(student_logits, teacher_logits, dim=1).mean()
    student_pred = torch.argmax(student_outputs.detach(), dim=1)
    teacher_pred = torch.argmax(teacher_outputs.detach(), dim=1)
    prediction_disagreement = (student_pred != teacher_pred).float().mean()
    cka_down3 = compute_linear_cka(student_feature[-2].detach(), teacher_feature[-2].detach())
    cka_down4 = compute_linear_cka(student_feature[-1].detach(), teacher_feature[-1].detach())
    return {
        'logit_cosine': float(logit_cosine.detach().item()),
        'prediction_disagreement': float(prediction_disagreement.detach().item()),
        'cka_down3': float(cka_down3.detach().item()),
        'cka_down4': float(cka_down4.detach().item()),
    }



class MyClient(BaseClient):

    def __init__(self, args, model, trainloader, valloader, amp=False):
        super(MyClient, self).__init__(args, model, trainloader, valloader)
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()
        self.best_performance = 0.0
        self.optimizer = None
        self.student_current_lr = self.current_lr
        self.teacher_current_lr = self.current_lr
        self._init_adaptive_pl_state()
        self._load_adaptive_pl_state()
        self._init_trustgeo_state()
        self._init_dg_state()
        self._init_support_bonus_state()

    def _adaptive_pl_is_enabled(self):
        return bool(getattr(self.args, 'adaptive_pl_enabled', 0)) and self.args.strategy in ['FedUniV2', 'FedUniV2.1']

    def _trustgeo_is_enabled(self):
        return bool(getattr(self.args, 'trustgeo_enabled', 0)) and bool(getattr(self.args, 'geometry_guided', 0))

    def _support_bonus_is_enabled(self):
        return bool(getattr(self.args, 'support_bonus_enabled', 0)) and bool(getattr(self.args, 'geometry_guided', 0))

    def _dg_is_enabled(self):
        return bool(getattr(self.args, 'dg_enabled', 0)) and bool(getattr(self.args, 'geometry_guided', 0))

    def _build_default_geometry_weight_values(self):
        return parse_geometry_weight_list(
            getattr(self.args, 'geometry_pseudo_weights', '1.0,0.8,0.5,0.2'),
            int(getattr(self.args, 'geometry_num_bins', 4)),
        )

    def _get_dg_cache_path(self):
        return os.path.join(self.args.snapshot_path, 'dg_dataset_audit.json')

    def _get_dg_local_audit_path(self, client_id=None):
        if client_id is None:
            client_id = self.cid
        return os.path.join(self.args.snapshot_path, 'dg_client_{}_audit.json'.format(int(client_id)))

    def _subsample_dg_cases(self, cases, max_cases):
        if max_cases <= 0 or len(cases) <= max_cases:
            return list(cases)
        keep_indices = np.linspace(0, len(cases) - 1, num=max_cases, dtype=np.int64)
        keep_indices = sorted(set(int(x) for x in keep_indices.tolist()))
        return [cases[idx] for idx in keep_indices]

    def _get_dg_cache_key(self):
        return {
            'dg_impl_version': 'v1-dg-local-audit-v2',
            'img_class': self.args.img_class,
            'root_path': os.path.abspath(self.args.root_path),
            'min_num_clients': int(self.args.min_num_clients),
            'dg_max_audit_samples': int(getattr(self.args, 'dg_max_audit_samples', 64)),
            'dg_max_prop_samples': int(getattr(self.args, 'dg_max_prop_samples', 16)),
            'dg_local_std_kernel': int(getattr(self.args, 'dg_local_std_kernel', 15)),
            'dg_far_radius': float(getattr(self.args, 'dg_far_radius', 24.0)),
            'dg_sep_near_radius': float(getattr(self.args, 'dg_sep_near_radius', 8.0)),
            'dg_sep_mid_radius': float(getattr(self.args, 'dg_sep_mid_radius', 24.0)),
            'dg_sep_far_radius': float(getattr(self.args, 'dg_sep_far_radius', 48.0)),
            'dg_min_fg_pixels': int(getattr(self.args, 'dg_min_fg_pixels', 8)),
            'dg_min_ring_pixels': int(getattr(self.args, 'dg_min_ring_pixels', 64)),
            'dg_min_non_seed_pixels': int(getattr(self.args, 'dg_min_non_seed_pixels', 128)),
            'dg_hom_tau': float(getattr(self.args, 'dg_hom_tau', 0.75)),
            'dg_sep_tau': float(getattr(self.args, 'dg_sep_tau', 0.25)),
            'dg_prop_seed_keep_ratio1': float(getattr(self.args, 'dg_prop_seed_keep_ratio1', 0.85)),
            'dg_prop_seed_keep_ratio2': float(getattr(self.args, 'dg_prop_seed_keep_ratio2', 0.70)),
            'dg_prop_min_seed_pixels': int(getattr(self.args, 'dg_prop_min_seed_pixels', 4)),
            'dg_prop_min_keep_pixels': int(getattr(self.args, 'dg_prop_min_keep_pixels', 2)),
            'dg_prop_min_region_pixels': int(getattr(self.args, 'dg_prop_min_region_pixels', 16)),
            'dg_prop_dist_std_scale': float(getattr(self.args, 'dg_prop_dist_std_scale', 1.0)),
            'dg_dataset_w_sep': float(getattr(self.args, 'dg_dataset_w_sep', 0.60)),
            'dg_dataset_w_hom': float(getattr(self.args, 'dg_dataset_w_hom', 0.30)),
            'dg_dataset_w_prop': float(getattr(self.args, 'dg_dataset_w_prop', 0.10)),
        }

    def _write_json_atomic(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = '{}.tmp.{}.{}'.format(path, os.getpid(), int(time.time() * 1000))
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)

    def _load_json_if_matching(self, path, cache_key):
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception:
            return None
        if payload.get('cache_key') != cache_key:
            return None
        return payload

    def _iter_local_dg_cases(self):
        dataset = getattr(self.trainloader, 'dataset', None)
        data_list = getattr(dataset, 'data_list', None)
        if not data_list:
            return []
        cases = []
        for sample_idx, sample in enumerate(data_list):
            image = sample.get('image', None)
            label = sample.get('label', None)
            if image is None or label is None:
                continue
            cases.append({
                'case_tag': 'client{}_sample{}'.format(self.cid, sample_idx),
                'image': np.asarray(image, dtype=np.float32),
                'label': np.asarray(label),
            })
        return cases

    def _extract_labeled_and_fg_masks(self, label):
        label = np.asarray(label)
        unlabeled_value = infer_unlabeled_value_from_label(label, self.args.img_class)
        if unlabeled_value is None:
            labeled_mask = np.ones_like(label, dtype=bool)
        else:
            labeled_mask = label != unlabeled_value
        fg_mask = np.logical_and(labeled_mask, label > 0)
        return labeled_mask, fg_mask

    def _prepare_dg_feature_maps(self, image):
        image = np.asarray(image, dtype=np.float32)
        if image.ndim == 2:
            image_chw = image[None, ...]
        elif image.ndim == 3:
            if image.shape[0] <= 4:
                image_chw = image
            elif image.shape[-1] <= 4:
                image_chw = np.transpose(image, (2, 0, 1))
            else:
                image_chw = image[None, ...]
        else:
            raise ValueError('Unsupported audit image shape: {}'.format(image.shape))

        image_chw = image_chw.astype(np.float32, copy=False)
        norm_channels = []
        for channel in image_chw:
            mean_val = float(channel.mean())
            std_val = float(channel.std())
            norm_channels.append((channel - mean_val) / max(std_val, 1e-6))
        image_norm = np.stack(norm_channels, axis=0)
        gray = image_norm.mean(axis=0)

        grad_x = ndimage.sobel(gray, axis=0, mode='reflect')
        grad_y = ndimage.sobel(gray, axis=1, mode='reflect')
        grad_mag = np.sqrt(np.maximum((grad_x ** 2) + (grad_y ** 2), 0.0)).astype(np.float32)

        local_std_kernel = int(getattr(self.args, 'dg_local_std_kernel', 15))
        if local_std_kernel <= 0:
            local_std_kernel = 15
        if local_std_kernel % 2 == 0:
            local_std_kernel += 1
        mean_map = ndimage.uniform_filter(gray, size=local_std_kernel, mode='reflect')
        sq_mean_map = ndimage.uniform_filter(gray * gray, size=local_std_kernel, mode='reflect')
        var_map = np.maximum(sq_mean_map - (mean_map * mean_map), 0.0)
        local_std = np.sqrt(var_map).astype(np.float32)

        feature_stack = np.stack([gray.astype(np.float32), grad_mag, local_std], axis=0)
        for feat_idx in range(feature_stack.shape[0]):
            feat = feature_stack[feat_idx]
            feat_mean = float(feat.mean())
            feat_std = float(feat.std())
            feature_stack[feat_idx] = (feat - feat_mean) / max(feat_std, 1e-6)
        return gray.astype(np.float32), local_std, feature_stack.astype(np.float32)

    def _compute_dg_homogeneity_score(self, local_std_map, support_mask):
        far_radius = float(getattr(self.args, 'dg_far_radius', 24.0))
        if support_mask.any():
            dist_to_seed = ndimage.distance_transform_edt(~support_mask)
            non_seed_mask = dist_to_seed >= far_radius
        else:
            non_seed_mask = np.ones_like(support_mask, dtype=bool)
        min_pixels = int(getattr(self.args, 'dg_min_non_seed_pixels', 128))
        if int(non_seed_mask.sum()) < min_pixels:
            if support_mask.any():
                dist_to_seed = ndimage.distance_transform_edt(~support_mask)
                non_seed_mask = dist_to_seed >= max(8.0, 0.5 * far_radius)
        if int(non_seed_mask.sum()) < min_pixels:
            return 0.5
        mean_local_std = float(local_std_map[non_seed_mask].mean())
        hom_tau = float(getattr(self.args, 'dg_hom_tau', 0.75))
        return float(np.exp(-mean_local_std / max(hom_tau, 1e-6)))

    def _compute_dg_separability_score(self, feature_stack, support_mask):
        min_fg_pixels = int(getattr(self.args, 'dg_min_fg_pixels', 8))
        if int(support_mask.sum()) < min_fg_pixels:
            return 0.5
        dist_to_seed = ndimage.distance_transform_edt(~support_mask)
        near_radius = float(getattr(self.args, 'dg_sep_near_radius', 8.0))
        mid_radius = float(getattr(self.args, 'dg_sep_mid_radius', 24.0))
        far_radius = float(getattr(self.args, 'dg_sep_far_radius', 48.0))
        near_mask = np.logical_and(dist_to_seed > 0.0, dist_to_seed <= near_radius)
        mid_mask = np.logical_and(dist_to_seed > near_radius, dist_to_seed <= mid_radius)
        far_mask = np.logical_and(dist_to_seed > mid_radius, dist_to_seed <= far_radius)
        min_ring_pixels = int(getattr(self.args, 'dg_min_ring_pixels', 64))
        if min(int(near_mask.sum()), int(mid_mask.sum()), int(far_mask.sum())) < min_ring_pixels:
            return 0.5

        seed_feat = feature_stack[:, support_mask].mean(axis=1)
        near_feat = feature_stack[:, near_mask].mean(axis=1)
        mid_feat = feature_stack[:, mid_mask].mean(axis=1)
        far_feat = feature_stack[:, far_mask].mean(axis=1)
        d_near = float(np.linalg.norm(near_feat - seed_feat))
        d_mid = float(np.linalg.norm(mid_feat - seed_feat))
        d_far = float(np.linalg.norm(far_feat - seed_feat))
        sep_tau = float(getattr(self.args, 'dg_sep_tau', 0.25))
        score_nm = 1.0 / (1.0 + np.exp(-(d_mid - d_near) / max(sep_tau, 1e-6)))
        score_mf = 1.0 / (1.0 + np.exp(-(d_far - d_mid) / max(sep_tau, 1e-6)))
        return float(0.5 * (score_nm + score_mf))

    def _perturb_dg_support_mask(self, support_mask, keep_ratio, rng):
        support_mask = np.asarray(support_mask, dtype=bool)
        coords = np.argwhere(support_mask)
        if len(coords) <= int(getattr(self.args, 'dg_prop_min_seed_pixels', 4)):
            return support_mask.copy()
        keep_count = max(
            int(getattr(self.args, 'dg_prop_min_keep_pixels', 2)),
            int(round(len(coords) * float(keep_ratio))),
        )
        keep_count = min(keep_count, len(coords))
        perm = rng.permutation(len(coords))
        keep_coords = coords[perm[:keep_count]]
        perturbed = np.zeros_like(support_mask, dtype=bool)
        if len(keep_coords) > 0:
            perturbed[tuple(keep_coords.T)] = True
        return perturbed

    def _binary_iou(self, mask_a, mask_b):
        mask_a = np.asarray(mask_a, dtype=bool)
        mask_b = np.asarray(mask_b, dtype=bool)
        union = int(np.logical_or(mask_a, mask_b).sum())
        if union <= 0:
            return 0.0
        inter = int(np.logical_and(mask_a, mask_b).sum())
        return float(inter) / float(union)

    def _build_dg_support_region(self, feature_stack, support_mask):
        support_mask = np.asarray(support_mask, dtype=bool)
        if int(support_mask.sum()) < int(getattr(self.args, 'dg_prop_min_seed_pixels', 4)):
            return np.zeros_like(support_mask, dtype=bool)
        support_feat = feature_stack[:, support_mask]
        support_mean = support_feat.mean(axis=1)
        support_dist = np.linalg.norm(support_feat.T - support_mean[None, :], axis=1)
        dist_map = np.linalg.norm(feature_stack - support_mean[:, None, None], axis=0)
        dist_thr = float(np.median(support_dist) + (float(getattr(self.args, 'dg_prop_dist_std_scale', 1.0)) * np.std(support_dist)))
        region_mask = dist_map <= max(dist_thr, 1e-6)
        return np.asarray(region_mask, dtype=bool)

    def _compute_dg_propagation_score(self, feature_stack, support_mask, case_tag):
        min_fg_pixels = int(getattr(self.args, 'dg_min_fg_pixels', 8))
        if int(support_mask.sum()) < min_fg_pixels:
            return 0.5

        base_fg = self._build_dg_support_region(feature_stack, support_mask)
        if int(base_fg.sum()) < int(getattr(self.args, 'dg_prop_min_region_pixels', 16)):
            return 0.0

        base_seed = int(sum(ord(ch) for ch in str(case_tag)) % (2 ** 32 - 1))
        rng = np.random.RandomState(base_seed)
        keep_ratio_1 = float(getattr(self.args, 'dg_prop_seed_keep_ratio1', 0.85))
        keep_ratio_2 = float(getattr(self.args, 'dg_prop_seed_keep_ratio2', 0.70))
        mask_p1 = self._perturb_dg_support_mask(support_mask, keep_ratio_1, rng)
        mask_p2 = self._perturb_dg_support_mask(support_mask, keep_ratio_2, rng)
        fg_p1 = self._build_dg_support_region(feature_stack, mask_p1)
        fg_p2 = self._build_dg_support_region(feature_stack, mask_p2)
        iou_scores = [
            self._binary_iou(base_fg, fg_p1),
            self._binary_iou(base_fg, fg_p2),
            self._binary_iou(fg_p1, fg_p2),
        ]
        return float(np.mean(iou_scores))

    def _compute_local_dg_audit(self):
        cases = self._iter_local_dg_cases()
        if not cases:
            return {
                'homogeneity': 0.5,
                'separability': 0.5,
                'propagation': 0.5,
                'strength': 0.5,
                'num_cases': 0,
                'num_prop_cases': 0,
            }
        audit_cases = self._subsample_dg_cases(cases, int(getattr(self.args, 'dg_max_audit_samples', 64)))
        prop_case_tags = {
            case['case_tag'] for case in self._subsample_dg_cases(audit_cases, int(getattr(self.args, 'dg_max_prop_samples', 16)))
        }
        homogeneity_scores = []
        separability_scores = []
        propagation_scores = []
        for case in audit_cases:
            image = np.asarray(case['image'], dtype=np.float32)
            label = np.asarray(case['label'])
            support_mask, fg_mask = self._extract_labeled_and_fg_masks(label)
            if int(support_mask.sum()) <= 0:
                continue
            _, local_std_map, feature_stack = self._prepare_dg_feature_maps(image)
            homogeneity_scores.append(self._compute_dg_homogeneity_score(local_std_map, support_mask))
            separability_scores.append(self._compute_dg_separability_score(feature_stack, support_mask))
            if case['case_tag'] in prop_case_tags:
                propagation_scores.append(self._compute_dg_propagation_score(feature_stack, support_mask, case_tag=case['case_tag']))

        homogeneity = float(np.median(np.asarray(homogeneity_scores, dtype=np.float32))) if homogeneity_scores else 0.5
        separability = float(np.median(np.asarray(separability_scores, dtype=np.float32))) if separability_scores else 0.5
        propagation = float(np.median(np.asarray(propagation_scores, dtype=np.float32))) if propagation_scores else 0.5
        w_sep = float(getattr(self.args, 'dg_dataset_w_sep', 0.60))
        w_hom = float(getattr(self.args, 'dg_dataset_w_hom', 0.30))
        w_prop = float(getattr(self.args, 'dg_dataset_w_prop', 0.10))
        strength = float(np.clip((w_sep * separability) + (w_hom * homogeneity) + (w_prop * propagation), 0.0, 1.0))
        return {
            'homogeneity': homogeneity,
            'separability': separability,
            'propagation': propagation,
            'strength': strength,
            'num_cases': len(audit_cases),
            'num_prop_cases': len(propagation_scores),
        }

    def _aggregate_dg_dataset_audit(self, local_audits):
        if not local_audits:
            return {
                'homogeneity': 0.5,
                'separability': 0.5,
                'propagation': 0.5,
                'strength': 0.5,
                'num_cases': 0,
                'num_prop_cases': 0,
                'num_client_audits': 0,
            }
        homogeneity = float(np.median(np.asarray([x['homogeneity'] for x in local_audits], dtype=np.float32)))
        separability = float(np.median(np.asarray([x['separability'] for x in local_audits], dtype=np.float32)))
        propagation = float(np.median(np.asarray([x['propagation'] for x in local_audits], dtype=np.float32)))
        w_sep = float(getattr(self.args, 'dg_dataset_w_sep', 0.60))
        w_hom = float(getattr(self.args, 'dg_dataset_w_hom', 0.30))
        w_prop = float(getattr(self.args, 'dg_dataset_w_prop', 0.10))
        strength = float(np.clip((w_sep * separability) + (w_hom * homogeneity) + (w_prop * propagation), 0.0, 1.0))
        return {
            'homogeneity': homogeneity,
            'separability': separability,
            'propagation': propagation,
            'strength': strength,
            'num_cases': int(sum(int(x.get('num_cases', 0)) for x in local_audits)),
            'num_prop_cases': int(sum(int(x.get('num_prop_cases', 0)) for x in local_audits)),
            'num_client_audits': int(len(local_audits)),
        }

    def _load_all_dg_local_audits_with_wait(self, cache_key):
        expected_clients = int(getattr(self.args, 'min_num_clients', 1))
        timeout_sec = float(getattr(self.args, 'dg_audit_wait_timeout_sec', 300.0))
        poll_sec = float(getattr(self.args, 'dg_audit_wait_poll_sec', 2.0))
        start_time = time.time()
        last_loaded = -1
        while True:
            loaded = []
            for client_id in range(expected_clients):
                payload = self._load_json_if_matching(self._get_dg_local_audit_path(client_id), cache_key)
                if payload is None or not isinstance(payload.get('audit'), dict):
                    continue
                loaded.append(payload['audit'])
            if len(loaded) != last_loaded:
                log(INFO, 'Client {} dg waiting: loaded {}/{} local audits'.format(self.cid, len(loaded), expected_clients))
                last_loaded = len(loaded)
            if len(loaded) >= expected_clients:
                return loaded, True
            if (time.time() - start_time) >= timeout_sec:
                return loaded, False
            time.sleep(max(poll_sec, 0.1))

    def _load_or_compute_dg_dataset_audit(self):
        cache_path = self._get_dg_cache_path()
        cache_key = self._get_dg_cache_key()
        cache_obj = self._load_json_if_matching(cache_path, cache_key)
        if cache_obj is not None and isinstance(cache_obj.get('audit'), dict):
            return cache_obj['audit']

        local_audit_path = self._get_dg_local_audit_path()
        local_payload = self._load_json_if_matching(local_audit_path, cache_key)
        if local_payload is None or not isinstance(local_payload.get('audit'), dict):
            local_audit = self._compute_local_dg_audit()
            local_payload = {
                'cache_key': cache_key,
                'client_id': int(self.cid),
                'local_num_cases': int(len(self._iter_local_dg_cases())),
                'audit': local_audit,
            }
            try:
                self._write_json_atomic(local_audit_path, local_payload)
            except Exception:
                pass

        local_audits, all_ready = self._load_all_dg_local_audits_with_wait(cache_key)
        audit = self._aggregate_dg_dataset_audit(local_audits)
        try:
            if all_ready:
                self._write_json_atomic(cache_path, {'cache_key': cache_key, 'audit': audit})
        except Exception:
            pass
        return audit

    def _iter_train_supervision_labels(self):
        dataset = getattr(self.trainloader, 'dataset', None)
        data_list = getattr(dataset, 'data_list', None)
        if not data_list:
            return []

        train_labels = []
        for sample in data_list:
            if 'label' not in sample:
                continue
            train_labels.append(np.asarray(sample['label']))
        return train_labels

    def _compute_weak_label_structure_stats(self, dilation_radius, density_kernel=None, mask_mode='foreground'):
        train_labels = self._iter_train_supervision_labels()
        if not train_labels:
            return 0.0, 0.0, 0.0

        dilation_radius = int(dilation_radius)
        if dilation_radius > 0:
            dilation_kernel_size = (2 * dilation_radius) + 1
            dilation_structure = np.ones((dilation_kernel_size, dilation_kernel_size), dtype=np.uint8)
        else:
            dilation_structure = None

        if density_kernel is not None:
            density_kernel = max(1, int(density_kernel))
            if density_kernel % 2 == 0:
                density_kernel += 1

        coverages = []
        dilated_coverages = []
        densities = []
        for label in train_labels:
            unlabeled_value = infer_unlabeled_value_from_label(label, self.args.img_class)
            if unlabeled_value is None:
                labeled_mask = np.ones_like(label, dtype=bool)
            else:
                labeled_mask = label != unlabeled_value
            fg_mask = np.logical_and(labeled_mask, label > 0)
            if mask_mode == 'foreground':
                support_mask = fg_mask
            elif mask_mode == 'labeled':
                support_mask = labeled_mask
            else:
                raise ValueError('Unsupported mask_mode: {}'.format(mask_mode))

            coverages.append(float(support_mask.mean()))
            if dilation_structure is not None and support_mask.any():
                dilated_support_mask = ndimage.binary_dilation(support_mask, structure=dilation_structure)
            else:
                dilated_support_mask = support_mask
            dilated_coverages.append(float(dilated_support_mask.mean()))
            if density_kernel is not None and support_mask.any():
                density_map = ndimage.uniform_filter(support_mask.astype(np.float32), size=density_kernel, mode='constant')
                densities.append(float(np.median(density_map[support_mask])))
            else:
                densities.append(0.0)

        coverage_client = float(np.median(np.asarray(coverages, dtype=np.float32)))
        dilated_coverage_client = float(np.median(np.asarray(dilated_coverages, dtype=np.float32)))
        density_client = float(np.median(np.asarray(densities, dtype=np.float32)))
        return coverage_client, dilated_coverage_client, density_client

    def _compute_support_bonus_client_stats(self):
        fg_coverage_client, dilated_fg_coverage_client, density_client = self._compute_weak_label_structure_stats(
            dilation_radius=int(getattr(self.args, 'support_bonus_dilation_radius', 5)),
            density_kernel=int(getattr(self.args, 'support_bonus_density_kernel', 11)),
            mask_mode='foreground',
        )
        tau_c = float(getattr(self.args, 'support_bonus_tau_c', 0.05))
        tau_d = float(getattr(self.args, 'support_bonus_tau_d', 0.20))
        coverage_score = 1.0 - np.exp(-fg_coverage_client / max(tau_c, 1e-8))
        dilated_score = 1.0 - np.exp(-dilated_fg_coverage_client / max(tau_d, 1e-8))
        density_score = float(np.clip(density_client, 0.0, 1.0))
        trust_score = float(np.clip((0.5 * coverage_score) + (0.3 * dilated_score) + (0.2 * density_score), 0.0, 1.0))
        bonus_strength = float(getattr(self.args, 'support_bonus_lambda', 0.10)) * trust_score
        return fg_coverage_client, dilated_fg_coverage_client, density_client, bonus_strength

    def _init_support_bonus_state(self):
        self.support_bonus_fg_coverage = 0.0
        self.support_bonus_dilated_fg_coverage = 0.0
        self.support_bonus_density = 0.0
        self.support_bonus_strength = 0.0
        if not self._support_bonus_is_enabled():
            return
        (
            self.support_bonus_fg_coverage,
            self.support_bonus_dilated_fg_coverage,
            self.support_bonus_density,
            self.support_bonus_strength,
        ) = self._compute_support_bonus_client_stats()
        log(
            INFO,
            'Client {} support-bonus: fg_coverage={:.6f}, dilated_fg_coverage={:.6f}, density={:.6f}, bonus_strength={:.6f}'.format(
                self.cid,
                self.support_bonus_fg_coverage,
                self.support_bonus_dilated_fg_coverage,
                self.support_bonus_density,
                self.support_bonus_strength,
            ),
        )

    def _compute_trustgeo_client_stats(self):
        fg_coverage_client, dilated_fg_coverage_client, fg_density_client = self._compute_weak_label_structure_stats(
            dilation_radius=int(getattr(self.args, 'trustgeo_dilation_radius', 5)),
            density_kernel=int(getattr(self.args, 'trustgeo_density_kernel', 11)),
            mask_mode='foreground',
        )
        support_coverage_client, dilated_support_coverage_client, support_density_client = self._compute_weak_label_structure_stats(
            dilation_radius=int(getattr(self.args, 'trustgeo_dilation_radius', 5)),
            density_kernel=int(getattr(self.args, 'trustgeo_density_kernel', 11)),
            mask_mode='labeled',
        )

        c_low = float(getattr(self.args, 'trustgeo_c_low', 0.0015))
        c_high = float(getattr(self.args, 'trustgeo_c_high', 0.0400))
        d_low = float(getattr(self.args, 'trustgeo_d_low', 0.0060))
        d_high = float(getattr(self.args, 'trustgeo_d_high', 0.0900))
        support_low = float(getattr(self.args, 'trustgeo_support_low', 0.0100))
        support_high = float(getattr(self.args, 'trustgeo_support_high', 0.1500))
        support_mod_min = float(getattr(self.args, 'trustgeo_support_mod_min', 0.85))
        w_c = float(getattr(self.args, 'trustgeo_w_c', 0.70))
        w_d = float(getattr(self.args, 'trustgeo_w_d', 0.30))

        coverage_score = float(np.clip((fg_coverage_client - c_low) / max(c_high - c_low, 1e-8), 0.0, 1.0))
        dilated_score = float(np.clip((dilated_fg_coverage_client - d_low) / max(d_high - d_low, 1e-8), 0.0, 1.0))
        support_score = float(np.clip((support_coverage_client - support_low) / max(support_high - support_low, 1e-8), 0.0, 1.0))
        support_mod = float(np.clip(support_mod_min + ((1.0 - support_mod_min) * support_score), support_mod_min, 1.0))

        # Foreground coverage is the primary trust signal; labeled support only
        # acts as a weak modulation so that large annotated support cannot
        # "rescue" a client whose foreground anchors are intrinsically too sparse.
        if fg_coverage_client <= c_low:
            trust_score = 0.0
            base_score = 0.0
        else:
            base_score = float(np.clip((w_c * coverage_score) + (w_d * dilated_score), 0.0, 1.0))
            trust_score = base_score * support_mod
        return (
            fg_coverage_client,
            dilated_fg_coverage_client,
            fg_density_client,
            support_coverage_client,
            dilated_support_coverage_client,
            support_density_client,
            coverage_score,
            dilated_score,
            support_score,
            support_mod,
            base_score,
            trust_score,
        )

    def _init_trustgeo_state(self):
        self.trustgeo_fg_coverage = 0.0
        self.trustgeo_dilated_fg_coverage = 0.0
        self.trustgeo_fg_density = 0.0
        self.trustgeo_support_coverage = 0.0
        self.trustgeo_dilated_support_coverage = 0.0
        self.trustgeo_support_density = 0.0
        self.trustgeo_coverage_score = 0.0
        self.trustgeo_dilated_score = 0.0
        self.trustgeo_support_score = 0.0
        self.trustgeo_support_mod = 1.0
        self.trustgeo_base_strength = 0.0
        self.trustgeo_strength = 0.0
        self.trustgeo_prior_weights = self._build_default_geometry_weight_values()
        if not self._trustgeo_is_enabled():
            return
        self.trustgeo_prior_weights = build_linear_geometry_weight_list(
            int(getattr(self.args, 'geometry_num_bins', 4)),
            alpha=float(getattr(self.args, 'trustgeo_prior_alpha', 0.8)),
            floor=float(getattr(self.args, 'trustgeo_prior_floor', 0.2)),
        )
        (
            self.trustgeo_fg_coverage,
            self.trustgeo_dilated_fg_coverage,
            self.trustgeo_fg_density,
            self.trustgeo_support_coverage,
            self.trustgeo_dilated_support_coverage,
            self.trustgeo_support_density,
            self.trustgeo_coverage_score,
            self.trustgeo_dilated_score,
            self.trustgeo_support_score,
            self.trustgeo_support_mod,
            self.trustgeo_base_strength,
            self.trustgeo_strength,
        ) = self._compute_trustgeo_client_stats()
        log(
            INFO,
            'Client {} trustgeo: fg_coverage={:.6f}, dilated_fg_coverage={:.6f}, fg_density={:.6f}, support_coverage={:.6f}, dilated_support_coverage={:.6f}, support_density={:.6f}, s_c={:.4f}, s_d={:.4f}, support_score={:.4f}, support_mod={:.4f}, base_lambda={:.6f}, lambda_client={:.6f}, prior_weights={}'.format(
                self.cid,
                self.trustgeo_fg_coverage,
                self.trustgeo_dilated_fg_coverage,
                self.trustgeo_fg_density,
                self.trustgeo_support_coverage,
                self.trustgeo_dilated_support_coverage,
                self.trustgeo_support_density,
                self.trustgeo_coverage_score,
                self.trustgeo_dilated_score,
                self.trustgeo_support_score,
                self.trustgeo_support_mod,
                self.trustgeo_base_strength,
                self.trustgeo_strength,
                [round(x, 4) for x in self.trustgeo_prior_weights],
            ),
        )

    def _init_dg_state(self):
        self.dg_dataset_homogeneity = 0.5
        self.dg_dataset_separability = 0.5
        self.dg_dataset_propagation = 0.5
        self.dg_dataset_strength = 0.5
        self.dg_dataset_num_cases = 0
        self.dg_dataset_num_prop_cases = 0
        self.dg_dataset_num_client_audits = 0
        self.dg_fg_coverage = 0.0
        self.dg_dilated_fg_coverage = 0.0
        self.dg_fg_density = 0.0
        self.dg_support_coverage = 0.0
        self.dg_dilated_support_coverage = 0.0
        self.dg_support_density = 0.0
        self.dg_coverage_score = 0.0
        self.dg_dilated_score = 0.0
        self.dg_support_score = 0.0
        self.dg_support_mod = 1.0
        self.dg_client_base_strength = 0.0
        self.dg_client_strength = 0.0
        self.dg_strength = 0.0
        self.dg_prior_weights = self._build_default_geometry_weight_values()
        if not self._dg_is_enabled():
            return
        self.dg_prior_weights = build_linear_geometry_weight_list(
            int(getattr(self.args, 'geometry_num_bins', 4)),
            alpha=float(getattr(self.args, 'trustgeo_prior_alpha', 0.8)),
            floor=float(getattr(self.args, 'trustgeo_prior_floor', 0.2)),
        )
        audit = self._load_or_compute_dg_dataset_audit()
        self.dg_dataset_homogeneity = float(audit.get('homogeneity', 0.5))
        self.dg_dataset_separability = float(audit.get('separability', 0.5))
        self.dg_dataset_propagation = float(audit.get('propagation', 0.5))
        self.dg_dataset_strength = float(audit.get('strength', 0.5))
        self.dg_dataset_num_cases = int(audit.get('num_cases', 0))
        self.dg_dataset_num_prop_cases = int(audit.get('num_prop_cases', 0))
        self.dg_dataset_num_client_audits = int(audit.get('num_client_audits', 0))
        (
            self.dg_fg_coverage,
            self.dg_dilated_fg_coverage,
            self.dg_fg_density,
            self.dg_support_coverage,
            self.dg_dilated_support_coverage,
            self.dg_support_density,
            self.dg_coverage_score,
            self.dg_dilated_score,
            self.dg_support_score,
            self.dg_support_mod,
            self.dg_client_base_strength,
            self.dg_client_strength,
        ) = self._compute_trustgeo_client_stats()
        self.dg_strength = float(np.clip(self.dg_dataset_strength * self.dg_client_strength, 0.0, 1.0))
        log(
            INFO,
            'Client {} dg: g_dataset={:.6f} (sep={:.4f}, hom={:.4f}, prop={:.4f}, cases={}, prop_cases={}, client_audits={}), '
            'e_client={:.6f}, e_client_base={:.6f}, lambda_client={:.6f}, fg_coverage={:.6f}, dilated_fg_coverage={:.6f}, '
            's_c={:.4f}, s_d={:.4f}, support_mod={:.4f}, prior_weights={}'.format(
                self.cid,
                self.dg_dataset_strength,
                self.dg_dataset_separability,
                self.dg_dataset_homogeneity,
                self.dg_dataset_propagation,
                self.dg_dataset_num_cases,
                self.dg_dataset_num_prop_cases,
                self.dg_dataset_num_client_audits,
                self.dg_client_strength,
                self.dg_client_base_strength,
                self.dg_strength,
                self.dg_fg_coverage,
                self.dg_dilated_fg_coverage,
                self.dg_coverage_score,
                self.dg_dilated_score,
                self.dg_support_mod,
                [round(x, 4) for x in self.dg_prior_weights],
            ),
        )

    def _build_support_bonus_weight_map(self, label_batch, target_batch=None):
        weight_map = torch.ones_like(label_batch, dtype=torch.float32, device=label_batch.device)
        if target_batch is None:
            return weight_map
        support_radius = int(getattr(self.args, 'support_bonus_support_radius', 5))
        bonus_strength = float(np.clip(self.support_bonus_strength, 0.0, 1.0))
        if support_radius <= 0 or bonus_strength <= 0.0:
            return weight_map

        unlabeled_mask = (label_batch == self.args.num_classes)
        target_batch = target_batch.to(device=label_batch.device, dtype=label_batch.dtype)
        support_mask = torch.zeros_like(unlabeled_mask, dtype=torch.bool)
        kernel_size = (2 * support_radius) + 1

        # Make the bonus class-aware: a pixel only receives extra weight if its
        # pseudo target matches the nearby weak foreground seed class.
        for class_idx in range(1, int(self.args.num_classes)):
            class_seed_mask = label_batch == class_idx
            if not bool(class_seed_mask.any().item()):
                continue
            dilated_seed = F.max_pool2d(
                class_seed_mask.float().unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=support_radius,
            ).squeeze(1) > 0.0
            class_support = unlabeled_mask & dilated_seed & (target_batch == class_idx)
            support_mask = support_mask | class_support
        return weight_map + (bonus_strength * support_mask.float())

    def _build_effective_geometry_weight_map(self, geometry_bin_batch, label_batch=None, target_batch=None, apply_support_bonus=True):
        if self._support_bonus_is_enabled():
            if label_batch is None:
                if geometry_bin_batch is None:
                    return None
                return torch.ones_like(geometry_bin_batch, dtype=torch.float32, device=geometry_bin_batch.device)
            if not apply_support_bonus:
                return torch.ones_like(label_batch, dtype=torch.float32, device=label_batch.device)
            return self._build_support_bonus_weight_map(label_batch, target_batch=target_batch)
        if geometry_bin_batch is None:
            return None
        if self._dg_is_enabled():
            prior_weight_map = build_geometry_weight_map(geometry_bin_batch, self.dg_prior_weights)
            strength = float(np.clip(self.dg_strength, 0.0, 1.0))
            return 1.0 + (strength * (prior_weight_map - 1.0))
        if self._trustgeo_is_enabled():
            prior_weight_map = build_geometry_weight_map(geometry_bin_batch, self.trustgeo_prior_weights)
            strength = float(np.clip(self.trustgeo_strength, 0.0, 1.0))
            return 1.0 + (strength * (prior_weight_map - 1.0))
        geometry_weight_values = self._build_default_geometry_weight_values()
        return build_geometry_weight_map(geometry_bin_batch, geometry_weight_values)

    def _init_adaptive_pl_state(self):
        num_bins = int(getattr(self.args, 'geometry_num_bins', 4))
        tau_init = float(getattr(self.args, 'adaptive_pl_tau_init', 0.55))
        self.adaptive_pl_tau = np.full(num_bins, tau_init, dtype=np.float32)
        self.adaptive_pl_lg_agreement_ema = 0.0
        self.adaptive_pl_prob_gap_ema = 0.0
        self.adaptive_pl_conf_gap_ema = 0.0
        self.adaptive_pl_last_observed_accept = np.full(num_bins, np.nan, dtype=np.float32)
        self.adaptive_pl_last_bin_counts = np.zeros(num_bins, dtype=np.int64)
        self.adaptive_pl_last_score_mean = np.full(num_bins, np.nan, dtype=np.float32)
        self.adaptive_pl_last_hard_ratio = 1.0
        self.adaptive_pl_last_loss_hard = 0.0
        self.adaptive_pl_last_loss_soft = 0.0
        self.adaptive_pl_last_loss_total = 0.0
        self.adaptive_pl_last_loss_total_backbone = 0.0
        self.adaptive_pl_last_loss_aux = 0.0
        self.adaptive_pl_last_lg_class_agree_ratio = 1.0
        self.adaptive_pl_last_mean_prob_gap = 0.0
        self.adaptive_pl_last_mean_conf_gap = 0.0
        self.adaptive_pl_last_both_uncertain_ratio = 0.0
        self.adaptive_pl_both_uncertain_ema = 0.0
        self.adaptive_pl_last_boundary_loss_oc = 0.0
        self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
        self.adaptive_pl_last_seed_support_mean = 0.0
        self.adaptive_pl_last_seed_support_soft_ratio = 0.0
        self.adaptive_pl_last_propagated_fg_mean = 0.0
        self.adaptive_pl_prior_gap_ema = 0.0
        self.adaptive_pl_last_prior_gap = 0.0
        self.adaptive_pl_last_client_risk = 0.0
        self.adaptive_pl_last_loss_risk = 0.0
        self.adaptive_pl_release_score_ema = 0.0
        self.adaptive_pl_preserve_score_ema = 0.0
        self.adaptive_pl_correction_score_ema = 0.0
        self.adaptive_pl_last_stage_progress = 0.0
        self.adaptive_pl_last_release_score = 0.0
        self.adaptive_pl_last_preserve_score = 0.0
        self.adaptive_pl_last_correction_score = 0.0
        self.adaptive_pl_last_effective_correction = 0.0
        self.adaptive_pl_last_hard_control = 0.0
        self.adaptive_pl_last_calibration_control = 0.0
        self.adaptive_pl_last_regime_code = 0.0
        self.adaptive_pl_release_streak = 0.0
        self.adaptive_pl_last_release_ready = 0.0
        self.adaptive_pl_last_high_risk = 0.0
        self.adaptive_pl_gate_release_armed = 0.0
        self.adaptive_pl_hard_ratio_ema = 1.0
        self.adaptive_pl_last_w6_tail_active = 0.0
        self.adaptive_pl_last_w6_tail_strength = 0.0
        self.adaptive_pl_last_w6_tail_votes = 0.0
        self.adaptive_pl_prior_gap_ref_ema = 0.0
        self.adaptive_pl_prior_gap_ref_dev_ema = 0.01
        self.adaptive_pl_prob_gap_ref_ema = 0.0
        self.adaptive_pl_prob_gap_ref_dev_ema = 0.01
        self.adaptive_pl_conf_gap_ref_ema = 0.0
        self.adaptive_pl_conf_gap_ref_dev_ema = 0.01
        self.adaptive_pl_agreement_ref_ema = 1.0
        self.adaptive_pl_agreement_ref_dev_ema = 0.01
        self.adaptive_pl_hard_ratio_ref_ema = 1.0
        self.adaptive_pl_hard_ratio_ref_dev_ema = 0.01
        self.adaptive_pl_last_risk_term_prior = 0.0
        self.adaptive_pl_last_risk_term_prob = 0.0
        self.adaptive_pl_last_risk_term_conf = 0.0
        self.adaptive_pl_last_risk_term_disagree = 0.0
        self.adaptive_pl_last_risk_term_hard = 0.0
        self.adaptive_pl_last_risk_term_credit = 0.0

    def _get_adaptive_pl_state_path(self, tag='latest'):
        return os.path.join(
            self.args.snapshot_path,
            'client_{}_adaptive_pl_{}.pth'.format(self.cid, tag),
        )

    def _load_adaptive_pl_state(self):
        if not self._adaptive_pl_is_enabled():
            return
        if int(getattr(self.args, 'adaptive_pl_resume_state', 0)) != 1:
            return
        state_path = self._get_adaptive_pl_state_path(tag='latest')
        if not os.path.exists(state_path):
            return
        state = torch.load(state_path, map_location='cpu')
        tau = state.get('adaptive_pl_tau')
        if tau is not None and len(tau) == len(self.adaptive_pl_tau):
            self.adaptive_pl_tau = np.asarray(tau, dtype=np.float32)
        self.current_iter = int(state.get('current_iter', self.current_iter))
        self.current_lr = self._compute_decay_lr(self.current_iter)
        self.adaptive_pl_lg_agreement_ema = float(
            state.get('adaptive_pl_lg_agreement_ema', state.get('adaptive_pl_agreement_ema', self.adaptive_pl_lg_agreement_ema))
        )
        self.adaptive_pl_prob_gap_ema = float(state.get('adaptive_pl_prob_gap_ema', self.adaptive_pl_prob_gap_ema))
        self.adaptive_pl_conf_gap_ema = float(state.get('adaptive_pl_conf_gap_ema', self.adaptive_pl_conf_gap_ema))
        self.adaptive_pl_last_observed_accept = np.asarray(
            state.get('adaptive_pl_last_observed_accept', self.adaptive_pl_last_observed_accept),
            dtype=np.float32,
        )
        self.adaptive_pl_last_bin_counts = np.asarray(
            state.get('adaptive_pl_last_bin_counts', self.adaptive_pl_last_bin_counts),
            dtype=np.int64,
        )
        self.adaptive_pl_last_score_mean = np.asarray(
            state.get('adaptive_pl_last_score_mean', self.adaptive_pl_last_score_mean),
            dtype=np.float32,
        )
        self.adaptive_pl_last_hard_ratio = float(state.get('adaptive_pl_last_hard_ratio', self.adaptive_pl_last_hard_ratio))
        self.adaptive_pl_last_loss_hard = float(state.get('adaptive_pl_last_loss_hard', self.adaptive_pl_last_loss_hard))
        self.adaptive_pl_last_loss_soft = float(state.get('adaptive_pl_last_loss_soft', self.adaptive_pl_last_loss_soft))
        self.adaptive_pl_last_loss_total = float(state.get('adaptive_pl_last_loss_total', self.adaptive_pl_last_loss_total))
        self.adaptive_pl_last_loss_total_backbone = float(state.get('adaptive_pl_last_loss_total_backbone', self.adaptive_pl_last_loss_total_backbone))
        self.adaptive_pl_last_loss_aux = float(state.get('adaptive_pl_last_loss_aux', self.adaptive_pl_last_loss_aux))
        self.adaptive_pl_last_lg_class_agree_ratio = float(
            state.get('adaptive_pl_last_lg_class_agree_ratio', self.adaptive_pl_last_lg_class_agree_ratio)
        )
        self.adaptive_pl_last_mean_prob_gap = float(
            state.get('adaptive_pl_last_mean_prob_gap', self.adaptive_pl_last_mean_prob_gap)
        )
        self.adaptive_pl_last_mean_conf_gap = float(
            state.get('adaptive_pl_last_mean_conf_gap', self.adaptive_pl_last_mean_conf_gap)
        )
        self.adaptive_pl_last_both_uncertain_ratio = float(
            state.get('adaptive_pl_last_both_uncertain_ratio', self.adaptive_pl_last_both_uncertain_ratio)
        )
        self.adaptive_pl_both_uncertain_ema = float(
            state.get('adaptive_pl_both_uncertain_ema', self.adaptive_pl_both_uncertain_ema)
        )
        self.adaptive_pl_last_boundary_loss_oc = float(
            state.get('adaptive_pl_last_boundary_loss_oc', self.adaptive_pl_last_boundary_loss_oc)
        )
        self.adaptive_pl_last_ring_valid_oc_ratio = float(
            state.get('adaptive_pl_last_ring_valid_oc_ratio', self.adaptive_pl_last_ring_valid_oc_ratio)
        )
        self.adaptive_pl_last_seed_support_mean = float(
            state.get('adaptive_pl_last_seed_support_mean', self.adaptive_pl_last_seed_support_mean)
        )
        self.adaptive_pl_last_seed_support_soft_ratio = float(
            state.get(
                'adaptive_pl_last_seed_support_soft_ratio',
                state.get('adaptive_pl_last_seed_support_fg_ratio', self.adaptive_pl_last_seed_support_soft_ratio)
            )
        )
        self.adaptive_pl_last_propagated_fg_mean = float(
            state.get('adaptive_pl_last_propagated_fg_mean', self.adaptive_pl_last_propagated_fg_mean)
        )
        self.adaptive_pl_prior_gap_ema = float(
            state.get('adaptive_pl_prior_gap_ema', self.adaptive_pl_prior_gap_ema)
        )
        self.adaptive_pl_last_prior_gap = float(
            state.get('adaptive_pl_last_prior_gap', self.adaptive_pl_last_prior_gap)
        )
        self.adaptive_pl_last_client_risk = float(
            state.get('adaptive_pl_last_client_risk', self.adaptive_pl_last_client_risk)
        )
        self.adaptive_pl_last_loss_risk = float(
            state.get('adaptive_pl_last_loss_risk', self.adaptive_pl_last_loss_risk)
        )
        self.adaptive_pl_release_score_ema = float(
            state.get('adaptive_pl_release_score_ema', self.adaptive_pl_release_score_ema)
        )
        self.adaptive_pl_preserve_score_ema = float(
            state.get('adaptive_pl_preserve_score_ema', self.adaptive_pl_preserve_score_ema)
        )
        self.adaptive_pl_correction_score_ema = float(
            state.get('adaptive_pl_correction_score_ema', self.adaptive_pl_correction_score_ema)
        )
        self.adaptive_pl_last_stage_progress = float(
            state.get('adaptive_pl_last_stage_progress', self.adaptive_pl_last_stage_progress)
        )
        self.adaptive_pl_last_release_score = float(
            state.get('adaptive_pl_last_release_score', self.adaptive_pl_last_release_score)
        )
        self.adaptive_pl_last_preserve_score = float(
            state.get('adaptive_pl_last_preserve_score', self.adaptive_pl_last_preserve_score)
        )
        self.adaptive_pl_last_correction_score = float(
            state.get('adaptive_pl_last_correction_score', self.adaptive_pl_last_correction_score)
        )
        self.adaptive_pl_last_effective_correction = float(
            state.get('adaptive_pl_last_effective_correction', self.adaptive_pl_last_effective_correction)
        )
        self.adaptive_pl_last_hard_control = float(
            state.get('adaptive_pl_last_hard_control', self.adaptive_pl_last_hard_control)
        )
        self.adaptive_pl_last_calibration_control = float(
            state.get('adaptive_pl_last_calibration_control', self.adaptive_pl_last_calibration_control)
        )
        self.adaptive_pl_last_regime_code = float(
            state.get('adaptive_pl_last_regime_code', self.adaptive_pl_last_regime_code)
        )
        self.adaptive_pl_release_streak = float(
            state.get('adaptive_pl_release_streak', self.adaptive_pl_release_streak)
        )
        self.adaptive_pl_last_release_ready = float(
            state.get('adaptive_pl_last_release_ready', self.adaptive_pl_last_release_ready)
        )
        self.adaptive_pl_last_high_risk = float(
            state.get('adaptive_pl_last_high_risk', self.adaptive_pl_last_high_risk)
        )
        self.adaptive_pl_gate_release_armed = float(
            state.get('adaptive_pl_gate_release_armed', self.adaptive_pl_gate_release_armed)
        )
        self.adaptive_pl_hard_ratio_ema = float(
            state.get('adaptive_pl_hard_ratio_ema', self.adaptive_pl_hard_ratio_ema)
        )
        self.adaptive_pl_last_w6_tail_active = float(
            state.get('adaptive_pl_last_w6_tail_active', self.adaptive_pl_last_w6_tail_active)
        )
        self.adaptive_pl_last_w6_tail_strength = float(
            state.get('adaptive_pl_last_w6_tail_strength', self.adaptive_pl_last_w6_tail_strength)
        )
        self.adaptive_pl_last_w6_tail_votes = float(
            state.get('adaptive_pl_last_w6_tail_votes', self.adaptive_pl_last_w6_tail_votes)
        )
        self.adaptive_pl_prior_gap_ref_ema = float(
            state.get('adaptive_pl_prior_gap_ref_ema', self.adaptive_pl_prior_gap_ref_ema)
        )
        self.adaptive_pl_prior_gap_ref_dev_ema = float(
            state.get('adaptive_pl_prior_gap_ref_dev_ema', self.adaptive_pl_prior_gap_ref_dev_ema)
        )
        self.adaptive_pl_prob_gap_ref_ema = float(
            state.get('adaptive_pl_prob_gap_ref_ema', self.adaptive_pl_prob_gap_ref_ema)
        )
        self.adaptive_pl_prob_gap_ref_dev_ema = float(
            state.get('adaptive_pl_prob_gap_ref_dev_ema', self.adaptive_pl_prob_gap_ref_dev_ema)
        )
        self.adaptive_pl_conf_gap_ref_ema = float(
            state.get('adaptive_pl_conf_gap_ref_ema', self.adaptive_pl_conf_gap_ref_ema)
        )
        self.adaptive_pl_conf_gap_ref_dev_ema = float(
            state.get('adaptive_pl_conf_gap_ref_dev_ema', self.adaptive_pl_conf_gap_ref_dev_ema)
        )
        self.adaptive_pl_agreement_ref_ema = float(
            state.get('adaptive_pl_agreement_ref_ema', self.adaptive_pl_agreement_ref_ema)
        )
        self.adaptive_pl_agreement_ref_dev_ema = float(
            state.get('adaptive_pl_agreement_ref_dev_ema', self.adaptive_pl_agreement_ref_dev_ema)
        )
        self.adaptive_pl_hard_ratio_ref_ema = float(
            state.get('adaptive_pl_hard_ratio_ref_ema', self.adaptive_pl_hard_ratio_ref_ema)
        )
        self.adaptive_pl_hard_ratio_ref_dev_ema = float(
            state.get('adaptive_pl_hard_ratio_ref_dev_ema', self.adaptive_pl_hard_ratio_ref_dev_ema)
        )
        self.adaptive_pl_last_risk_term_prior = float(
            state.get('adaptive_pl_last_risk_term_prior', self.adaptive_pl_last_risk_term_prior)
        )
        self.adaptive_pl_last_risk_term_prob = float(
            state.get('adaptive_pl_last_risk_term_prob', self.adaptive_pl_last_risk_term_prob)
        )
        self.adaptive_pl_last_risk_term_conf = float(
            state.get('adaptive_pl_last_risk_term_conf', self.adaptive_pl_last_risk_term_conf)
        )
        self.adaptive_pl_last_risk_term_disagree = float(
            state.get('adaptive_pl_last_risk_term_disagree', self.adaptive_pl_last_risk_term_disagree)
        )
        self.adaptive_pl_last_risk_term_hard = float(
            state.get('adaptive_pl_last_risk_term_hard', self.adaptive_pl_last_risk_term_hard)
        )
        self.adaptive_pl_last_risk_term_credit = float(
            state.get('adaptive_pl_last_risk_term_credit', self.adaptive_pl_last_risk_term_credit)
        )

    def _save_adaptive_pl_state(self, tag='latest'):
        if not self._adaptive_pl_is_enabled():
            return
        state = {
            'current_iter': int(self.current_iter),
            'adaptive_pl_tau': self.adaptive_pl_tau.astype(np.float32),
            'adaptive_pl_lg_agreement_ema': float(self.adaptive_pl_lg_agreement_ema),
            'adaptive_pl_prob_gap_ema': float(self.adaptive_pl_prob_gap_ema),
            'adaptive_pl_conf_gap_ema': float(self.adaptive_pl_conf_gap_ema),
            'adaptive_pl_last_observed_accept': self.adaptive_pl_last_observed_accept.astype(np.float32),
            'adaptive_pl_last_bin_counts': self.adaptive_pl_last_bin_counts.astype(np.int64),
            'adaptive_pl_last_score_mean': self.adaptive_pl_last_score_mean.astype(np.float32),
            'adaptive_pl_last_hard_ratio': float(self.adaptive_pl_last_hard_ratio),
            'adaptive_pl_last_loss_hard': float(self.adaptive_pl_last_loss_hard),
            'adaptive_pl_last_loss_soft': float(self.adaptive_pl_last_loss_soft),
            'adaptive_pl_last_loss_total': float(self.adaptive_pl_last_loss_total),
            'adaptive_pl_last_loss_total_backbone': float(self.adaptive_pl_last_loss_total_backbone),
            'adaptive_pl_last_loss_aux': float(self.adaptive_pl_last_loss_aux),
            'adaptive_pl_last_lg_class_agree_ratio': float(self.adaptive_pl_last_lg_class_agree_ratio),
            'adaptive_pl_last_mean_prob_gap': float(self.adaptive_pl_last_mean_prob_gap),
            'adaptive_pl_last_mean_conf_gap': float(self.adaptive_pl_last_mean_conf_gap),
            'adaptive_pl_last_both_uncertain_ratio': float(self.adaptive_pl_last_both_uncertain_ratio),
            'adaptive_pl_both_uncertain_ema': float(self.adaptive_pl_both_uncertain_ema),
            'adaptive_pl_last_boundary_loss_oc': float(self.adaptive_pl_last_boundary_loss_oc),
            'adaptive_pl_last_ring_valid_oc_ratio': float(self.adaptive_pl_last_ring_valid_oc_ratio),
            'adaptive_pl_last_seed_support_mean': float(self.adaptive_pl_last_seed_support_mean),
            'adaptive_pl_last_seed_support_soft_ratio': float(self.adaptive_pl_last_seed_support_soft_ratio),
            'adaptive_pl_last_propagated_fg_mean': float(self.adaptive_pl_last_propagated_fg_mean),
            'adaptive_pl_prior_gap_ema': float(self.adaptive_pl_prior_gap_ema),
            'adaptive_pl_last_prior_gap': float(self.adaptive_pl_last_prior_gap),
            'adaptive_pl_last_client_risk': float(self.adaptive_pl_last_client_risk),
            'adaptive_pl_last_loss_risk': float(self.adaptive_pl_last_loss_risk),
            'adaptive_pl_release_score_ema': float(self.adaptive_pl_release_score_ema),
            'adaptive_pl_preserve_score_ema': float(self.adaptive_pl_preserve_score_ema),
            'adaptive_pl_correction_score_ema': float(self.adaptive_pl_correction_score_ema),
            'adaptive_pl_last_stage_progress': float(self.adaptive_pl_last_stage_progress),
            'adaptive_pl_last_release_score': float(self.adaptive_pl_last_release_score),
            'adaptive_pl_last_preserve_score': float(self.adaptive_pl_last_preserve_score),
            'adaptive_pl_last_correction_score': float(self.adaptive_pl_last_correction_score),
            'adaptive_pl_last_effective_correction': float(self.adaptive_pl_last_effective_correction),
            'adaptive_pl_last_hard_control': float(self.adaptive_pl_last_hard_control),
            'adaptive_pl_last_calibration_control': float(self.adaptive_pl_last_calibration_control),
            'adaptive_pl_last_regime_code': float(self.adaptive_pl_last_regime_code),
            'adaptive_pl_release_streak': float(self.adaptive_pl_release_streak),
            'adaptive_pl_last_release_ready': float(self.adaptive_pl_last_release_ready),
            'adaptive_pl_last_high_risk': float(self.adaptive_pl_last_high_risk),
            'adaptive_pl_gate_release_armed': float(self.adaptive_pl_gate_release_armed),
            'adaptive_pl_hard_ratio_ema': float(self.adaptive_pl_hard_ratio_ema),
            'adaptive_pl_last_w6_tail_active': float(self.adaptive_pl_last_w6_tail_active),
            'adaptive_pl_last_w6_tail_strength': float(self.adaptive_pl_last_w6_tail_strength),
            'adaptive_pl_last_w6_tail_votes': float(self.adaptive_pl_last_w6_tail_votes),
            'adaptive_pl_prior_gap_ref_ema': float(self.adaptive_pl_prior_gap_ref_ema),
            'adaptive_pl_prior_gap_ref_dev_ema': float(self.adaptive_pl_prior_gap_ref_dev_ema),
            'adaptive_pl_prob_gap_ref_ema': float(self.adaptive_pl_prob_gap_ref_ema),
            'adaptive_pl_prob_gap_ref_dev_ema': float(self.adaptive_pl_prob_gap_ref_dev_ema),
            'adaptive_pl_conf_gap_ref_ema': float(self.adaptive_pl_conf_gap_ref_ema),
            'adaptive_pl_conf_gap_ref_dev_ema': float(self.adaptive_pl_conf_gap_ref_dev_ema),
            'adaptive_pl_agreement_ref_ema': float(self.adaptive_pl_agreement_ref_ema),
            'adaptive_pl_agreement_ref_dev_ema': float(self.adaptive_pl_agreement_ref_dev_ema),
            'adaptive_pl_hard_ratio_ref_ema': float(self.adaptive_pl_hard_ratio_ref_ema),
            'adaptive_pl_hard_ratio_ref_dev_ema': float(self.adaptive_pl_hard_ratio_ref_dev_ema),
            'adaptive_pl_last_risk_term_prior': float(self.adaptive_pl_last_risk_term_prior),
            'adaptive_pl_last_risk_term_prob': float(self.adaptive_pl_last_risk_term_prob),
            'adaptive_pl_last_risk_term_conf': float(self.adaptive_pl_last_risk_term_conf),
            'adaptive_pl_last_risk_term_disagree': float(self.adaptive_pl_last_risk_term_disagree),
            'adaptive_pl_last_risk_term_hard': float(self.adaptive_pl_last_risk_term_hard),
            'adaptive_pl_last_risk_term_credit': float(self.adaptive_pl_last_risk_term_credit),
        }
        torch.save(state, self._get_adaptive_pl_state_path(tag=tag))

    def _capture_adaptive_pl_runtime_state(self):
        state = {}
        for key, value in self.__dict__.items():
            if not key.startswith('adaptive_pl_'):
                continue
            if isinstance(value, np.ndarray):
                state[key] = value.copy()
            else:
                state[key] = copy.deepcopy(value)
        return state

    def _restore_adaptive_pl_runtime_state(self, state):
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                setattr(self, key, value.copy())
            else:
                setattr(self, key, copy.deepcopy(value))

    def _maybe_log_adaptive_pl_state(self):
        global writer
        if writer is None or not self._adaptive_pl_is_enabled():
            return
        w5_soft_only_enabled = bool(getattr(self.args, 'risk_calibration_w5_soft_only_enabled', 0))
        log_interval = int(getattr(self.args, 'adaptive_pl_log_interval', 50))
        if log_interval <= 0 or self.current_iter % log_interval != 0:
            return
        if self._dg_is_enabled():
            writer.add_scalar('client_{}/dg/dataset_homogeneity'.format(self.cid), float(self.dg_dataset_homogeneity), self.current_iter)
            writer.add_scalar('client_{}/dg/dataset_separability'.format(self.cid), float(self.dg_dataset_separability), self.current_iter)
            writer.add_scalar('client_{}/dg/dataset_propagation'.format(self.cid), float(self.dg_dataset_propagation), self.current_iter)
            writer.add_scalar('client_{}/dg/dataset_strength'.format(self.cid), float(self.dg_dataset_strength), self.current_iter)
            writer.add_scalar('client_{}/dg/dataset_num_client_audits'.format(self.cid), float(self.dg_dataset_num_client_audits), self.current_iter)
            writer.add_scalar('client_{}/dg/client_base_strength'.format(self.cid), float(self.dg_client_base_strength), self.current_iter)
            writer.add_scalar('client_{}/dg/client_strength'.format(self.cid), float(self.dg_client_strength), self.current_iter)
            writer.add_scalar('client_{}/dg/strength'.format(self.cid), float(self.dg_strength), self.current_iter)
            writer.add_scalar('client_{}/dg/fg_coverage'.format(self.cid), float(self.dg_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/dg/dilated_fg_coverage'.format(self.cid), float(self.dg_dilated_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/dg/s_c'.format(self.cid), float(self.dg_coverage_score), self.current_iter)
            writer.add_scalar('client_{}/dg/s_d'.format(self.cid), float(self.dg_dilated_score), self.current_iter)
        if self._trustgeo_is_enabled():
            writer.add_scalar('client_{}/trustgeo/fg_coverage'.format(self.cid), float(self.trustgeo_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/dilated_fg_coverage'.format(self.cid), float(self.trustgeo_dilated_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/fg_density'.format(self.cid), float(self.trustgeo_fg_density), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/support_coverage'.format(self.cid), float(self.trustgeo_support_coverage), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/dilated_support_coverage'.format(self.cid), float(self.trustgeo_dilated_support_coverage), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/support_density'.format(self.cid), float(self.trustgeo_support_density), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/s_c'.format(self.cid), float(self.trustgeo_coverage_score), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/s_d'.format(self.cid), float(self.trustgeo_dilated_score), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/support_score'.format(self.cid), float(self.trustgeo_support_score), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/support_mod'.format(self.cid), float(self.trustgeo_support_mod), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/base_strength'.format(self.cid), float(self.trustgeo_base_strength), self.current_iter)
            writer.add_scalar('client_{}/trustgeo/strength'.format(self.cid), float(self.trustgeo_strength), self.current_iter)
        if self._support_bonus_is_enabled():
            writer.add_scalar('client_{}/support_bonus/fg_coverage'.format(self.cid), float(self.support_bonus_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/support_bonus/dilated_fg_coverage'.format(self.cid), float(self.support_bonus_dilated_fg_coverage), self.current_iter)
            writer.add_scalar('client_{}/support_bonus/density'.format(self.cid), float(self.support_bonus_density), self.current_iter)
            writer.add_scalar('client_{}/support_bonus/strength'.format(self.cid), float(self.support_bonus_strength), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/lg_class_agree_ratio'.format(self.cid), float(self.adaptive_pl_last_lg_class_agree_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/lg_class_agree_ema'.format(self.cid), float(self.adaptive_pl_lg_agreement_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_prob_gap'.format(self.cid), float(self.adaptive_pl_last_mean_prob_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_prob_gap_ema'.format(self.cid), float(self.adaptive_pl_prob_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_conf_gap'.format(self.cid), float(self.adaptive_pl_last_mean_conf_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_conf_gap_ema'.format(self.cid), float(self.adaptive_pl_conf_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/both_uncertain_ratio'.format(self.cid), float(self.adaptive_pl_last_both_uncertain_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/both_uncertain_ema'.format(self.cid), float(self.adaptive_pl_both_uncertain_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/hard_ratio'.format(self.cid), float(self.adaptive_pl_last_hard_ratio), self.current_iter)
        if bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
            writer.add_scalar('client_{}/adaptive_pl/loss_hard_backbone'.format(self.cid), float(self.adaptive_pl_last_loss_hard), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_soft_backbone'.format(self.cid), float(self.adaptive_pl_last_loss_soft), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_total_backbone'.format(self.cid), float(self.adaptive_pl_last_loss_total_backbone), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_aux_corr'.format(self.cid), float(self.adaptive_pl_last_loss_aux), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_total_v1'.format(self.cid), float(self.adaptive_pl_last_loss_total), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/boundary_loss_oc_backbone'.format(self.cid), float(self.adaptive_pl_last_boundary_loss_oc), self.current_iter)
        else:
            writer.add_scalar('client_{}/adaptive_pl/loss_hard'.format(self.cid), float(self.adaptive_pl_last_loss_hard), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_soft'.format(self.cid), float(self.adaptive_pl_last_loss_soft), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_total'.format(self.cid), float(self.adaptive_pl_last_loss_total), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/loss_risk'.format(self.cid), float(self.adaptive_pl_last_loss_risk), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/boundary_loss_oc'.format(self.cid), float(self.adaptive_pl_last_boundary_loss_oc), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/ring_valid_oc_ratio'.format(self.cid), float(self.adaptive_pl_last_ring_valid_oc_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/seed_support_mean'.format(self.cid), float(self.adaptive_pl_last_seed_support_mean), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/seed_support_soft_ratio'.format(self.cid), float(self.adaptive_pl_last_seed_support_soft_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/propagated_fg_mean'.format(self.cid), float(self.adaptive_pl_last_propagated_fg_mean), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/prior_gap'.format(self.cid), float(self.adaptive_pl_last_prior_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/prior_gap_ema'.format(self.cid), float(self.adaptive_pl_prior_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/client_risk'.format(self.cid), float(self.adaptive_pl_last_client_risk), self.current_iter)
        if bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
            writer.add_scalar('client_{}/adaptive_pl/loss_aux_corr_weighted'.format(self.cid), float(self.adaptive_pl_last_loss_risk), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/stage_progress'.format(self.cid), float(self.adaptive_pl_last_stage_progress), self.current_iter)
        if (not w5_soft_only_enabled) and (not bool(getattr(self.args, 'risk_calibration_v1_enabled', 0))):
            writer.add_scalar('client_{}/adaptive_pl/release_score'.format(self.cid), float(self.adaptive_pl_last_release_score), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/preserve_score'.format(self.cid), float(self.adaptive_pl_last_preserve_score), self.current_iter)
            writer.add_scalar('client_{}/adaptive_pl/correction_score'.format(self.cid), float(self.adaptive_pl_last_correction_score), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/effective_correction'.format(self.cid), float(self.adaptive_pl_last_effective_correction), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/hard_control'.format(self.cid), float(self.adaptive_pl_last_hard_control), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/calibration_control'.format(self.cid), float(self.adaptive_pl_last_calibration_control), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/hard_ratio_ema'.format(self.cid), float(self.adaptive_pl_hard_ratio_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/w6_tail_active'.format(self.cid), float(self.adaptive_pl_last_w6_tail_active), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/w6_tail_strength'.format(self.cid), float(self.adaptive_pl_last_w6_tail_strength), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/w6_tail_votes'.format(self.cid), float(self.adaptive_pl_last_w6_tail_votes), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_prior'.format(self.cid), float(self.adaptive_pl_last_risk_term_prior), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_prob'.format(self.cid), float(self.adaptive_pl_last_risk_term_prob), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_conf'.format(self.cid), float(self.adaptive_pl_last_risk_term_conf), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_disagree'.format(self.cid), float(self.adaptive_pl_last_risk_term_disagree), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_hard'.format(self.cid), float(self.adaptive_pl_last_risk_term_hard), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/risk_term_credit'.format(self.cid), float(self.adaptive_pl_last_risk_term_credit), self.current_iter)
        for bin_idx, tau_val in enumerate(self.adaptive_pl_tau):
            writer.add_scalar('client_{}/adaptive_pl/tau_bin_{}'.format(self.cid, bin_idx), float(tau_val), self.current_iter)
            observed_accept = self.adaptive_pl_last_observed_accept[bin_idx]
            if not np.isnan(observed_accept):
                writer.add_scalar(
                    'client_{}/adaptive_pl/observed_accept_bin_{}'.format(self.cid, bin_idx),
                    float(observed_accept),
                    self.current_iter,
                )
            score_mean = self.adaptive_pl_last_score_mean[bin_idx]
            if not np.isnan(score_mean):
                writer.add_scalar(
                    'client_{}/adaptive_pl/score_mean_bin_{}'.format(self.cid, bin_idx),
                    float(score_mean),
                    self.current_iter,
                )
            writer.add_scalar(
                'client_{}/adaptive_pl/bin_count_{}'.format(self.cid, bin_idx),
                int(self.adaptive_pl_last_bin_counts[bin_idx]),
                self.current_iter,
            )

    def _risk_ref_cfg(self):
        momentum = float(getattr(self.args, 'risk_calibration_ref_ema_momentum', 0.97))
        scale_ratio = float(getattr(self.args, 'risk_calibration_ref_scale_ratio', 0.15))
        scale_eps = float(getattr(self.args, 'risk_calibration_ref_scale_eps', 1e-3))
        z_clip = float(getattr(self.args, 'risk_calibration_ref_z_clip', 3.0))
        hard_weight = float(getattr(self.args, 'risk_calibration_hard_ratio_weight', 0.10))
        return momentum, scale_ratio, scale_eps, z_clip, hard_weight

    def _update_ref_stat(self, value, mean_attr, dev_attr, momentum):
        value = float(value)
        ref_mean = float(getattr(self, mean_attr))
        ref_dev = float(getattr(self, dev_attr))
        new_mean = (momentum * ref_mean) + ((1.0 - momentum) * value)
        new_dev = (momentum * ref_dev) + ((1.0 - momentum) * abs(value - new_mean))
        setattr(self, mean_attr, float(new_mean))
        setattr(self, dev_attr, float(max(new_dev, 0.0)))

    def _relative_high_risk_term(self, value, ref_mean, ref_dev, scale_ratio, scale_eps, z_clip):
        scale = max(float(ref_dev), (abs(float(ref_mean)) * scale_ratio) + scale_eps)
        raw = (float(value) - float(ref_mean)) / max(scale, 1e-8)
        return float(np.clip(raw / max(z_clip, 1e-8), 0.0, 1.0))

    def _relative_low_risk_term(self, value, ref_mean, ref_dev, scale_ratio, scale_eps, z_clip):
        scale = max(float(ref_dev), (abs(float(ref_mean)) * scale_ratio) + scale_eps)
        raw = (float(ref_mean) - float(value)) / max(scale, 1e-8)
        return float(np.clip(raw / max(z_clip, 1e-8), 0.0, 1.0))

    def _compute_relative_client_risk(self):
        momentum, scale_ratio, scale_eps, z_clip, hard_weight = self._risk_ref_cfg()
        prior_term = self._relative_high_risk_term(
            self.adaptive_pl_prior_gap_ema,
            self.adaptive_pl_prior_gap_ref_ema,
            self.adaptive_pl_prior_gap_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )
        prob_term = self._relative_high_risk_term(
            self.adaptive_pl_prob_gap_ema,
            self.adaptive_pl_prob_gap_ref_ema,
            self.adaptive_pl_prob_gap_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )
        conf_term = self._relative_high_risk_term(
            self.adaptive_pl_conf_gap_ema,
            self.adaptive_pl_conf_gap_ref_ema,
            self.adaptive_pl_conf_gap_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )
        disagree_term = self._relative_low_risk_term(
            self.adaptive_pl_lg_agreement_ema,
            self.adaptive_pl_agreement_ref_ema,
            self.adaptive_pl_agreement_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )
        hard_low_term = self._relative_low_risk_term(
            self.adaptive_pl_hard_ratio_ema,
            self.adaptive_pl_hard_ratio_ref_ema,
            self.adaptive_pl_hard_ratio_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )
        agreement_credit = self._relative_high_risk_term(
            self.adaptive_pl_lg_agreement_ema,
            self.adaptive_pl_agreement_ref_ema,
            self.adaptive_pl_agreement_ref_dev_ema,
            scale_ratio,
            scale_eps,
            z_clip,
        )

        self._update_ref_stat(self.adaptive_pl_prior_gap_ema, 'adaptive_pl_prior_gap_ref_ema', 'adaptive_pl_prior_gap_ref_dev_ema', momentum)
        self._update_ref_stat(self.adaptive_pl_prob_gap_ema, 'adaptive_pl_prob_gap_ref_ema', 'adaptive_pl_prob_gap_ref_dev_ema', momentum)
        self._update_ref_stat(self.adaptive_pl_conf_gap_ema, 'adaptive_pl_conf_gap_ref_ema', 'adaptive_pl_conf_gap_ref_dev_ema', momentum)
        self._update_ref_stat(self.adaptive_pl_lg_agreement_ema, 'adaptive_pl_agreement_ref_ema', 'adaptive_pl_agreement_ref_dev_ema', momentum)
        self._update_ref_stat(self.adaptive_pl_hard_ratio_ema, 'adaptive_pl_hard_ratio_ref_ema', 'adaptive_pl_hard_ratio_ref_dev_ema', momentum)

        self.adaptive_pl_last_risk_term_prior = prior_term
        self.adaptive_pl_last_risk_term_prob = prob_term
        self.adaptive_pl_last_risk_term_conf = conf_term
        self.adaptive_pl_last_risk_term_disagree = disagree_term
        self.adaptive_pl_last_risk_term_hard = hard_low_term
        self.adaptive_pl_last_risk_term_credit = agreement_credit

        return {
            'prior': prior_term,
            'prob': prob_term,
            'conf': conf_term,
            'disagree': disagree_term,
            'hard': hard_low_term,
            'credit': agreement_credit,
            'hard_weight': hard_weight,
        }

    def _extract_univ_outputs(self, model_out):
        if self.args.model == 'unet_univ5':
            return model_out[0], model_out[8]
        if self.args.model in ['unet_univ3', 'unet_univ4', 'unet_univ2']:
            return model_out[0], model_out[8]
        raise NotImplementedError('Adaptive pseudo-labeling only supports FedUniV2/FedUniV2.1 UNet Univ models.')

    def _build_global_pseudo_label_mix(self, global_pl_model, volume_batch, pseudo_alpha):
        if global_pl_model is None:
            return None
        with self._preserve_global_rng_state():
            with torch.no_grad():
                with autocast(enabled=self.amp):
                    global_out = global_pl_model(volume_batch)
                    global_outputs, global_outputs_auxiliary = self._extract_univ_outputs(global_out)
                    global_outputs_soft = torch.softmax(global_outputs, dim=1)
                    global_outputs_soft_auxiliary = torch.softmax(global_outputs_auxiliary, dim=1)
                    return (
                        pseudo_alpha * global_outputs_soft.detach()
                        + (1.0 - pseudo_alpha) * global_outputs_soft_auxiliary.detach()
                    )

    def _compute_adaptive_pl_loss_w1p_pure(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                           pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch):
        if global_pseudo_label_mix is None:
            global_pseudo_label_mix = pseudo_label_mix.detach()
        reliability = compute_adaptive_pseudo_reliability(
            pseudo_label_mix,
            global_pseudo_label_mix,
            gamma_prob=float(getattr(self.args, 'adaptive_pl_gamma_prob', 4.0)),
            gamma_conf=float(getattr(self.args, 'adaptive_pl_gamma_conf', 3.0)),
        )
        local_prob = reliability['local_prob']
        global_prob = reliability['global_prob']
        conf_global = reliability['conf_global']
        prob_gap = reliability['prob_gap']
        conf_gap = reliability['conf_gap']
        agree_lg = reliability['agree_lg']
        score = reliability['score']
        tau_values = torch.tensor(self.adaptive_pl_tau, device=geometry_bin_batch.device, dtype=score.dtype)
        tau_map = tau_values[geometry_bin_batch.long().clamp(0, len(self.adaptive_pl_tau) - 1)]
        unlabeled_mask = label_batch == self.args.num_classes

        ema_momentum = 0.9
        min_pixels = int(getattr(self.args, 'adaptive_pl_min_pixels_per_bin', 64))
        warmup_iters = int(getattr(self.args, 'adaptive_pl_warmup_iters', 800))
        tau_update_enabled = bool(getattr(self.args, 'adaptive_pl_tau_update', 1))
        adaptive_active = self.current_iter >= warmup_iters
        target_accept = float(getattr(self.args, 'adaptive_pl_target_accept', 0.35))
        tau_global_min = float(getattr(self.args, 'adaptive_pl_global_min_conf', 0.6))
        blend_kappa = float(getattr(self.args, 'adaptive_pl_blend_kappa', 2.0))
        boundary_lambda = float(getattr(self.args, 'adaptive_pl_boundary_lambda', 0.0))
        boundary_kernel_size = int(getattr(self.args, 'adaptive_pl_boundary_kernel_size', 3))
        boundary_warmup_iters = int(getattr(self.args, 'adaptive_pl_boundary_warmup_iters', warmup_iters))
        outer_beta = float(getattr(self.args, 'beta', 1.0))

        valid_mask = unlabeled_mask.bool()
        if valid_mask.any():
            lg_agree_mean = float(agree_lg[valid_mask].float().mean().item())
            prob_gap_mean = float(prob_gap[valid_mask].mean().item())
            conf_gap_mean = float(conf_gap[valid_mask].mean().item())
            self.adaptive_pl_lg_agreement_ema = (ema_momentum * self.adaptive_pl_lg_agreement_ema) + ((1.0 - ema_momentum) * lg_agree_mean)
            self.adaptive_pl_prob_gap_ema = (ema_momentum * self.adaptive_pl_prob_gap_ema) + ((1.0 - ema_momentum) * prob_gap_mean)
            self.adaptive_pl_conf_gap_ema = (ema_momentum * self.adaptive_pl_conf_gap_ema) + ((1.0 - ema_momentum) * conf_gap_mean)
            self.adaptive_pl_last_lg_class_agree_ratio = lg_agree_mean
            self.adaptive_pl_last_mean_prob_gap = prob_gap_mean
            self.adaptive_pl_last_mean_conf_gap = conf_gap_mean
            local_prior = compute_masked_class_prior(local_prob, valid_mask)
            global_prior = compute_masked_class_prior(global_prob, valid_mask)
            prior_gap = float(torch.mean(torch.abs(local_prior[1:] - global_prior[1:])).item())
            self.adaptive_pl_prior_gap_ema = (ema_momentum * self.adaptive_pl_prior_gap_ema) + ((1.0 - ema_momentum) * prior_gap)
            self.adaptive_pl_last_prior_gap = prior_gap
        else:
            self.adaptive_pl_last_lg_class_agree_ratio = 0.0
            self.adaptive_pl_last_mean_prob_gap = 0.0
            self.adaptive_pl_last_mean_conf_gap = 0.0
            self.adaptive_pl_last_prior_gap = 0.0

        self.adaptive_pl_last_client_risk = 0.0
        self.adaptive_pl_release_score_ema = 0.0
        self.adaptive_pl_preserve_score_ema = 0.0
        self.adaptive_pl_correction_score_ema = 0.0
        self.adaptive_pl_last_stage_progress = 0.0
        self.adaptive_pl_last_release_score = 0.0
        self.adaptive_pl_last_preserve_score = 0.0
        self.adaptive_pl_last_correction_score = 0.0
        self.adaptive_pl_last_effective_correction = 0.0
        self.adaptive_pl_last_hard_control = 0.0
        self.adaptive_pl_last_calibration_control = 0.0
        self.adaptive_pl_last_regime_code = 0.0
        self.adaptive_pl_release_streak = 0.0
        self.adaptive_pl_last_release_ready = 0.0
        self.adaptive_pl_last_high_risk = 0.0
        self.adaptive_pl_gate_release_armed = 0.0
        self.adaptive_pl_last_w6_tail_active = 0.0
        self.adaptive_pl_last_w6_tail_strength = 0.0
        self.adaptive_pl_last_w6_tail_votes = 0.0

        self.adaptive_pl_last_observed_accept[:] = np.nan
        self.adaptive_pl_last_score_mean[:] = np.nan
        self.adaptive_pl_last_bin_counts[:] = 0
        for bin_idx in range(len(self.adaptive_pl_tau)):
            bin_mask = (geometry_bin_batch == bin_idx) & valid_mask
            bin_count = int(bin_mask.sum().item())
            self.adaptive_pl_last_bin_counts[bin_idx] = bin_count
            if bin_count <= 0:
                continue
            score_bin = score[bin_mask]
            self.adaptive_pl_last_score_mean[bin_idx] = float(score_bin.mean().item())
            if bin_count < min_pixels:
                continue
            observed_accept = float((score_bin >= self.adaptive_pl_tau[bin_idx]).float().mean().item())
            self.adaptive_pl_last_observed_accept[bin_idx] = observed_accept
            if adaptive_active and tau_update_enabled:
                updated_tau = self.adaptive_pl_tau[bin_idx] + 0.01 * (observed_accept - target_accept)
                self.adaptive_pl_tau[bin_idx] = float(np.clip(updated_tau, 0.3, 0.9))

        if adaptive_active:
            both_uncertain = valid_mask & (score < tau_map) & (conf_global < tau_global_min)
            hard_mask = valid_mask & (~both_uncertain) & (score >= tau_map)
            soft_mask = valid_mask & (~both_uncertain) & (score < tau_map)
        else:
            both_uncertain = torch.zeros_like(score, dtype=torch.bool)
            hard_mask = valid_mask
            soft_mask = torch.zeros_like(score, dtype=torch.bool)

        lambda_dynamic = torch.exp(-blend_kappa * conf_gap).clamp(1e-6, 1.0)
        corrected_prob = lambda_dynamic.unsqueeze(1) * local_prob + (1.0 - lambda_dynamic).unsqueeze(1) * global_prob
        hard_target = torch.argmax(corrected_prob, dim=1)
        supervision_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=hard_target.detach(),
            apply_support_bonus=True,
        )

        hard_weight = supervision_weight_map * hard_mask.float() * valid_mask.float()
        soft_weight = supervision_weight_map * soft_mask.float() * valid_mask.float()
        loss_hard_1 = weighted_pseudo_dice_loss(outputs_soft, hard_target, hard_weight)
        loss_hard_2 = weighted_pseudo_dice_loss(outputs_soft_auxiliary, hard_target, hard_weight)
        loss_soft_1 = weighted_reverse_kl_loss(outputs, corrected_prob, soft_weight)
        loss_soft_2 = weighted_reverse_kl_loss(outputs_auxiliary, corrected_prob, soft_weight)
        loss_hard = 0.5 * (loss_hard_1 + loss_hard_2)
        loss_soft = 0.5 * (loss_soft_1 + loss_soft_2)
        total_loss = loss_hard + float(getattr(self.args, 'adaptive_pl_soft_lambda', 0.2)) * loss_soft
        valid_count = float(valid_mask.float().sum().item())

        loss_boundary_oc = outputs.new_tensor(0.0)
        if self.current_iter >= boundary_warmup_iters and boundary_lambda > 0.0:
            with torch.no_grad():
                pl_target = torch.argmax(corrected_prob.detach(), dim=1)
                target_mask_oc = pl_target == 1
                ring_oc = morphology_boundary(target_mask_oc, kernel_size=boundary_kernel_size)
                ring_valid_oc = ring_oc & (geometry_bin_batch <= 2) & valid_mask
            if ring_valid_oc.any():
                pred_prob_oc = F.softmax(outputs, dim=1)[:, 1]
                loss_boundary_oc = boundary_dice_loss(
                    pred_prob_oc,
                    target_mask_oc.float(),
                    ring_valid_oc.float(),
                )
                effective_boundary_scale = boundary_lambda / max(outer_beta, 1e-8)
                total_loss = total_loss + (effective_boundary_scale * loss_boundary_oc)
                if valid_count > 0:
                    self.adaptive_pl_last_ring_valid_oc_ratio = float(
                        ring_valid_oc.float().sum().item() / valid_count
                    )
                else:
                    self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
            else:
                self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
        else:
            self.adaptive_pl_last_ring_valid_oc_ratio = 0.0

        if valid_count > 0:
            hard_ratio_current = float(hard_mask.float().sum().item() / valid_count)
            self.adaptive_pl_last_hard_ratio = hard_ratio_current
            self.adaptive_pl_hard_ratio_ema = (ema_momentum * self.adaptive_pl_hard_ratio_ema) + ((1.0 - ema_momentum) * hard_ratio_current)
            self.adaptive_pl_last_both_uncertain_ratio = float(both_uncertain.float().sum().item() / valid_count)
            self.adaptive_pl_both_uncertain_ema = (
                ema_momentum * self.adaptive_pl_both_uncertain_ema
            ) + ((1.0 - ema_momentum) * self.adaptive_pl_last_both_uncertain_ratio)
        else:
            self.adaptive_pl_last_hard_ratio = 0.0
            self.adaptive_pl_hard_ratio_ema = ema_momentum * self.adaptive_pl_hard_ratio_ema
            self.adaptive_pl_last_both_uncertain_ratio = 0.0
            self.adaptive_pl_both_uncertain_ema = ema_momentum * self.adaptive_pl_both_uncertain_ema
        self.adaptive_pl_last_seed_support_mean = 0.0
        self.adaptive_pl_last_seed_support_soft_ratio = 0.0
        self.adaptive_pl_last_propagated_fg_mean = 0.0
        self.adaptive_pl_last_loss_hard = float(loss_hard.detach().item())
        self.adaptive_pl_last_loss_soft = float(loss_soft.detach().item())
        self.adaptive_pl_last_loss_total = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_total_backbone = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_aux = 0.0
        self.adaptive_pl_last_loss_risk = 0.0
        self.adaptive_pl_last_boundary_loss_oc = float(loss_boundary_oc.detach().item())

        return {
            'loss_total': total_loss,
            'loss_hard': loss_hard,
            'loss_soft': loss_soft,
            'loss_risk': outputs.new_tensor(0.0),
            'loss_boundary_oc': loss_boundary_oc,
            'hard_target': hard_target,
            'mixed_prob': corrected_prob,
            'score': score,
            'seed_support_map': score.new_zeros(score.shape),
            'hard_ratio': self.adaptive_pl_last_hard_ratio,
            'client_risk': 0.0,
            'correction_strength': 0.0,
        }

    def _compute_adaptive_pl_loss_w5_from_w1p(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                              pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch):
        if global_pseudo_label_mix is None:
            global_pseudo_label_mix = pseudo_label_mix.detach()
        reliability = compute_adaptive_pseudo_reliability(
            pseudo_label_mix,
            global_pseudo_label_mix,
            gamma_prob=float(getattr(self.args, 'adaptive_pl_gamma_prob', 4.0)),
            gamma_conf=float(getattr(self.args, 'adaptive_pl_gamma_conf', 3.0)),
        )
        local_prob = reliability['local_prob']
        global_prob = reliability['global_prob']
        conf_global = reliability['conf_global']
        prob_gap = reliability['prob_gap']
        conf_gap = reliability['conf_gap']
        agree_lg = reliability['agree_lg']
        score = reliability['score']
        risk_geometry_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=None,
            apply_support_bonus=False,
        )
        tau_values = torch.tensor(self.adaptive_pl_tau, device=geometry_bin_batch.device, dtype=score.dtype)
        tau_map = tau_values[geometry_bin_batch.long().clamp(0, len(self.adaptive_pl_tau) - 1)]
        unlabeled_mask = label_batch == self.args.num_classes

        ema_momentum = 0.9
        min_pixels = int(getattr(self.args, 'adaptive_pl_min_pixels_per_bin', 64))
        warmup_iters = int(getattr(self.args, 'adaptive_pl_warmup_iters', 800))
        tau_update_enabled = bool(getattr(self.args, 'adaptive_pl_tau_update', 1))
        adaptive_active = self.current_iter >= warmup_iters
        target_accept = float(getattr(self.args, 'adaptive_pl_target_accept', 0.35))
        tau_global_min = float(getattr(self.args, 'adaptive_pl_global_min_conf', 0.6))
        blend_kappa = float(getattr(self.args, 'adaptive_pl_blend_kappa', 2.0))
        boundary_lambda = float(getattr(self.args, 'adaptive_pl_boundary_lambda', 0.0))
        boundary_kernel_size = int(getattr(self.args, 'adaptive_pl_boundary_kernel_size', 3))
        boundary_warmup_iters = int(getattr(self.args, 'adaptive_pl_boundary_warmup_iters', warmup_iters))
        outer_beta = float(getattr(self.args, 'beta', 1.0))

        risk_calibration_enabled = bool(getattr(self.args, 'risk_calibration_enabled', 0))
        risk_prior_power = float(getattr(self.args, 'risk_calibration_prior_power', 1.0))
        risk_prior_clip_min = float(getattr(self.args, 'risk_calibration_prior_clip_min', 0.5))
        risk_prior_clip_max = float(getattr(self.args, 'risk_calibration_prior_clip_max', 1.5))
        risk_soft_boost = float(getattr(self.args, 'risk_calibration_soft_boost', 0.5))
        risk_global_soft_lambda = float(getattr(self.args, 'risk_calibration_global_soft_lambda', 0.15))
        risk_agreement_credit = float(getattr(self.args, 'risk_calibration_risk_agreement_credit', 0.20))
        w5_start_iter = int(getattr(self.args, 'risk_calibration_w5_start_iter', warmup_iters))
        w5_peak_iter = int(getattr(self.args, 'risk_calibration_w5_peak_iter', max(warmup_iters + 1, warmup_iters + 600)))
        w5_end_iter = int(getattr(self.args, 'risk_calibration_w5_end_iter', max(warmup_iters + 2, warmup_iters + 1800)))
        w5_min_correction = float(getattr(self.args, 'risk_calibration_w5_min_correction', 0.10))
        w5_risk_scale = float(getattr(self.args, 'risk_calibration_w5_risk_scale', 0.40))
        w6_late_tail_enabled = bool(getattr(self.args, 'risk_calibration_w6_late_tail_enabled', 0))
        w6_tail_start_iter = int(getattr(self.args, 'risk_calibration_w6_tail_start_iter', -1))
        w6_tail_min_votes = int(getattr(self.args, 'risk_calibration_w6_tail_min_votes', 2))
        w6_tail_risk_threshold = float(getattr(self.args, 'risk_calibration_w6_tail_risk_threshold', 0.55))
        w6_tail_prior_gap_threshold = float(getattr(self.args, 'risk_calibration_w6_tail_prior_gap_threshold', 0.025))
        w6_tail_hard_ratio_threshold = float(getattr(self.args, 'risk_calibration_w6_tail_hard_ratio_threshold', 0.50))
        w6_tail_peak_fraction = float(getattr(self.args, 'risk_calibration_w6_tail_peak_fraction', 0.30))
        w6_tail_min_correction = float(getattr(self.args, 'risk_calibration_w6_tail_min_correction', 0.03))
        w6_tail_risk_scale = float(getattr(self.args, 'risk_calibration_w6_tail_risk_scale', 0.12))

        valid_mask = unlabeled_mask.bool()
        if valid_mask.any():
            lg_agree_mean = float(agree_lg[valid_mask].float().mean().item())
            prob_gap_mean = float(prob_gap[valid_mask].mean().item())
            conf_gap_mean = float(conf_gap[valid_mask].mean().item())
            self.adaptive_pl_lg_agreement_ema = (ema_momentum * self.adaptive_pl_lg_agreement_ema) + ((1.0 - ema_momentum) * lg_agree_mean)
            self.adaptive_pl_prob_gap_ema = (ema_momentum * self.adaptive_pl_prob_gap_ema) + ((1.0 - ema_momentum) * prob_gap_mean)
            self.adaptive_pl_conf_gap_ema = (ema_momentum * self.adaptive_pl_conf_gap_ema) + ((1.0 - ema_momentum) * conf_gap_mean)
            self.adaptive_pl_last_lg_class_agree_ratio = lg_agree_mean
            self.adaptive_pl_last_mean_prob_gap = prob_gap_mean
            self.adaptive_pl_last_mean_conf_gap = conf_gap_mean
            local_prior = compute_masked_class_prior(local_prob, valid_mask)
            global_prior = compute_masked_class_prior(global_prob, valid_mask)
            prior_gap = float(torch.mean(torch.abs(local_prior[1:] - global_prior[1:])).item())
            self.adaptive_pl_prior_gap_ema = (ema_momentum * self.adaptive_pl_prior_gap_ema) + ((1.0 - ema_momentum) * prior_gap)
            self.adaptive_pl_last_prior_gap = prior_gap
        else:
            self.adaptive_pl_last_lg_class_agree_ratio = 0.0
            self.adaptive_pl_last_mean_prob_gap = 0.0
            self.adaptive_pl_last_mean_conf_gap = 0.0
            local_prior = None
            global_prior = None
            self.adaptive_pl_last_prior_gap = 0.0

        if risk_calibration_enabled and valid_mask.any():
            risk_terms = self._compute_relative_client_risk()
            hard_weight = float(np.clip(risk_terms['hard_weight'], 0.0, 0.5))
            base_weights = {
                'prior': 0.40,
                'prob': 0.20,
                'conf': 0.10,
                'disagree': 0.30,
            }
            weight_scale = max(1e-6, 1.0 - hard_weight)
            risk_score = (
                weight_scale * base_weights['prior'] * risk_terms['prior']
                + weight_scale * base_weights['prob'] * risk_terms['prob']
                + weight_scale * base_weights['conf'] * risk_terms['conf']
                + weight_scale * base_weights['disagree'] * risk_terms['disagree']
                + hard_weight * risk_terms['hard']
                - (risk_agreement_credit * risk_terms['credit'])
            )
            risk_score = float(np.clip(risk_score, 0.0, 1.0))
        else:
            self.adaptive_pl_last_risk_term_prior = 0.0
            self.adaptive_pl_last_risk_term_prob = 0.0
            self.adaptive_pl_last_risk_term_conf = 0.0
            self.adaptive_pl_last_risk_term_disagree = 0.0
            self.adaptive_pl_last_risk_term_hard = 0.0
            self.adaptive_pl_last_risk_term_credit = 0.0
            risk_score = 0.0
        self.adaptive_pl_last_client_risk = risk_score

        w5_start_iter = max(warmup_iters, w5_start_iter)
        w5_peak_iter = max(w5_start_iter + 1, w5_peak_iter)
        w5_end_iter = max(w5_peak_iter + 1, w5_end_iter)
        if self.current_iter < w5_start_iter:
            stage_progress = 0.0
        elif self.current_iter < w5_peak_iter:
            stage_progress = float(np.clip(
                (self.current_iter - w5_start_iter) / float(w5_peak_iter - w5_start_iter),
                0.0,
                1.0,
            ))
        elif self.current_iter < w5_end_iter:
            stage_progress = float(np.clip(
                (w5_end_iter - self.current_iter) / float(w5_end_iter - w5_peak_iter),
                0.0,
                1.0,
            ))
        else:
            stage_progress = 0.0
        calibration_target = float(np.clip(w5_min_correction + (w5_risk_scale * risk_score), 0.0, 1.0))
        calibration_control = float(np.clip(stage_progress * calibration_target, 0.0, 1.0))
        tail_start_iter = w5_end_iter if w6_tail_start_iter < 0 else w6_tail_start_iter
        tail_strength = 0.0
        tail_active = 0.0
        weak_votes = 0
        hard_control = 0.0

        self.adaptive_pl_last_observed_accept[:] = np.nan
        self.adaptive_pl_last_score_mean[:] = np.nan
        self.adaptive_pl_last_bin_counts[:] = 0
        for bin_idx in range(len(self.adaptive_pl_tau)):
            bin_mask = (geometry_bin_batch == bin_idx) & valid_mask
            bin_count = int(bin_mask.sum().item())
            self.adaptive_pl_last_bin_counts[bin_idx] = bin_count
            if bin_count <= 0:
                continue
            score_bin = score[bin_mask]
            self.adaptive_pl_last_score_mean[bin_idx] = float(score_bin.mean().item())
            if bin_count < min_pixels:
                continue
            observed_accept = float((score_bin >= self.adaptive_pl_tau[bin_idx]).float().mean().item())
            self.adaptive_pl_last_observed_accept[bin_idx] = observed_accept
            if adaptive_active and tau_update_enabled:
                updated_tau = self.adaptive_pl_tau[bin_idx] + 0.01 * (observed_accept - target_accept)
                self.adaptive_pl_tau[bin_idx] = float(np.clip(updated_tau, 0.3, 0.9))

        if adaptive_active:
            both_uncertain = valid_mask & (score < tau_map) & (conf_global < tau_global_min)
            hard_mask = valid_mask & (~both_uncertain) & (score >= tau_map)
            soft_mask = valid_mask & (~both_uncertain) & (score < tau_map)
        else:
            both_uncertain = torch.zeros_like(score, dtype=torch.bool)
            hard_mask = valid_mask
            soft_mask = torch.zeros_like(score, dtype=torch.bool)

        valid_count = float(valid_mask.float().sum().item())
        if valid_count > 0:
            hard_ratio_current = float(hard_mask.float().sum().item() / valid_count)
            hard_ratio_gate = (
                ema_momentum * self.adaptive_pl_hard_ratio_ema
            ) + ((1.0 - ema_momentum) * hard_ratio_current)
        else:
            hard_ratio_current = 0.0
            hard_ratio_gate = ema_momentum * self.adaptive_pl_hard_ratio_ema

        if valid_count > 0:
            if risk_score >= w6_tail_risk_threshold:
                weak_votes += 1
            if self.adaptive_pl_prior_gap_ema >= w6_tail_prior_gap_threshold:
                weak_votes += 1
            if hard_ratio_gate <= w6_tail_hard_ratio_threshold:
                weak_votes += 1

            if (
                w6_late_tail_enabled
                and self.current_iter >= tail_start_iter
                and weak_votes >= w6_tail_min_votes
            ):
                tail_active = 1.0
                tail_floor = float(np.clip(w6_tail_min_correction + (w6_tail_risk_scale * risk_score), 0.0, 1.0))
                tail_cap = float(np.clip(w6_tail_peak_fraction * calibration_target, 0.0, 1.0))
                tail_strength = min(tail_floor, tail_cap)
                calibration_control = max(calibration_control, tail_strength)

        self.adaptive_pl_last_stage_progress = stage_progress
        self.adaptive_pl_last_release_score = 0.0
        self.adaptive_pl_last_preserve_score = calibration_control
        self.adaptive_pl_last_correction_score = risk_score
        self.adaptive_pl_last_effective_correction = calibration_control
        self.adaptive_pl_last_hard_control = hard_control
        self.adaptive_pl_last_calibration_control = calibration_control
        self.adaptive_pl_last_regime_code = 0.0
        self.adaptive_pl_release_streak = 0.0
        self.adaptive_pl_last_release_ready = 0.0
        self.adaptive_pl_last_high_risk = 0.0
        self.adaptive_pl_gate_release_armed = 0.0
        self.adaptive_pl_last_w6_tail_active = tail_active
        self.adaptive_pl_last_w6_tail_strength = tail_strength
        self.adaptive_pl_last_w6_tail_votes = float(weak_votes)
        self.adaptive_pl_release_score_ema = ema_momentum * self.adaptive_pl_release_score_ema
        self.adaptive_pl_preserve_score_ema = (
            ema_momentum * self.adaptive_pl_preserve_score_ema
        ) + ((1.0 - ema_momentum) * calibration_control)
        self.adaptive_pl_correction_score_ema = (
            ema_momentum * self.adaptive_pl_correction_score_ema
        ) + ((1.0 - ema_momentum) * risk_score)

        lambda_dynamic = torch.exp(-blend_kappa * conf_gap).clamp(1e-6, 1.0)
        corrected_prob = lambda_dynamic.unsqueeze(1) * local_prob + (1.0 - lambda_dynamic).unsqueeze(1) * global_prob
        corrected_prob_soft = corrected_prob
        if risk_calibration_enabled and valid_mask.any():
            prior_scale = ((global_prior + 1e-6) / (local_prior + 1e-6)).clamp(
                min=risk_prior_clip_min,
                max=risk_prior_clip_max,
            )
            prior_scale = torch.pow(prior_scale, risk_prior_power * calibration_control)
            corrected_prob_soft = corrected_prob_soft * prior_scale.view(1, -1, 1, 1)
            corrected_prob_soft = corrected_prob_soft / corrected_prob_soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
        hard_target = torch.argmax(corrected_prob, dim=1)
        soft_target = torch.argmax(corrected_prob_soft.detach(), dim=1)
        hard_supervision_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=hard_target.detach(),
            apply_support_bonus=True,
        )
        soft_supervision_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=soft_target,
            apply_support_bonus=True,
        )

        hard_weight = hard_supervision_weight_map * hard_mask.float() * valid_mask.float()
        soft_weight = soft_supervision_weight_map * soft_mask.float() * valid_mask.float()
        risk_weight = risk_geometry_weight_map * both_uncertain.float() * valid_mask.float()
        loss_hard_1 = weighted_pseudo_dice_loss(outputs_soft, hard_target, hard_weight)
        loss_hard_2 = weighted_pseudo_dice_loss(outputs_soft_auxiliary, hard_target, hard_weight)
        loss_soft_1 = weighted_reverse_kl_loss(outputs, corrected_prob_soft, soft_weight)
        loss_soft_2 = weighted_reverse_kl_loss(outputs_auxiliary, corrected_prob_soft, soft_weight)
        loss_risk_1 = weighted_reverse_kl_loss(outputs, global_prob, risk_weight)
        loss_risk_2 = weighted_reverse_kl_loss(outputs_auxiliary, global_prob, risk_weight)
        loss_hard = 0.5 * (loss_hard_1 + loss_hard_2)
        loss_soft = (1.0 + (risk_soft_boost * calibration_control)) * 0.5 * (loss_soft_1 + loss_soft_2)
        loss_risk = risk_score * 0.5 * (loss_risk_1 + loss_risk_2)
        total_loss = loss_hard + float(getattr(self.args, 'adaptive_pl_soft_lambda', 0.2)) * loss_soft
        if risk_calibration_enabled and adaptive_active:
            total_loss = total_loss + (risk_global_soft_lambda * calibration_control * loss_risk)

        loss_boundary_oc = outputs.new_tensor(0.0)
        if self.current_iter >= boundary_warmup_iters and boundary_lambda > 0.0:
            with torch.no_grad():
                pl_target = torch.argmax(corrected_prob_soft.detach(), dim=1)
                target_mask_oc = pl_target == 1
                ring_oc = morphology_boundary(target_mask_oc, kernel_size=boundary_kernel_size)
                ring_valid_oc = ring_oc & (geometry_bin_batch <= 2) & valid_mask
            if ring_valid_oc.any():
                pred_prob_oc = F.softmax(outputs, dim=1)[:, 1]
                loss_boundary_oc = boundary_dice_loss(
                    pred_prob_oc,
                    target_mask_oc.float(),
                    ring_valid_oc.float(),
                )
                effective_boundary_scale = boundary_lambda / max(outer_beta, 1e-8)
                total_loss = total_loss + (effective_boundary_scale * loss_boundary_oc)
                if valid_count > 0:
                    self.adaptive_pl_last_ring_valid_oc_ratio = float(
                        ring_valid_oc.float().sum().item() / valid_count
                    )
                else:
                    self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
            else:
                self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
        else:
            self.adaptive_pl_last_ring_valid_oc_ratio = 0.0

        if valid_count > 0:
            self.adaptive_pl_last_hard_ratio = hard_ratio_current
            self.adaptive_pl_hard_ratio_ema = hard_ratio_gate
            self.adaptive_pl_last_both_uncertain_ratio = float(both_uncertain.float().sum().item() / valid_count)
            self.adaptive_pl_both_uncertain_ema = (
                ema_momentum * self.adaptive_pl_both_uncertain_ema
            ) + ((1.0 - ema_momentum) * self.adaptive_pl_last_both_uncertain_ratio)
        else:
            self.adaptive_pl_last_hard_ratio = 0.0
            self.adaptive_pl_hard_ratio_ema = ema_momentum * self.adaptive_pl_hard_ratio_ema
            self.adaptive_pl_last_both_uncertain_ratio = 0.0
            self.adaptive_pl_both_uncertain_ema = ema_momentum * self.adaptive_pl_both_uncertain_ema
        self.adaptive_pl_last_seed_support_mean = 0.0
        self.adaptive_pl_last_seed_support_soft_ratio = 0.0
        self.adaptive_pl_last_propagated_fg_mean = 0.0
        self.adaptive_pl_last_loss_hard = float(loss_hard.detach().item())
        self.adaptive_pl_last_loss_soft = float(loss_soft.detach().item())
        self.adaptive_pl_last_loss_total = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_total_backbone = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_aux = float((risk_global_soft_lambda * calibration_control * loss_risk).detach().item()) if (risk_calibration_enabled and adaptive_active) else 0.0
        self.adaptive_pl_last_loss_risk = float(loss_risk.detach().item())
        self.adaptive_pl_last_boundary_loss_oc = float(loss_boundary_oc.detach().item())

        return {
            'loss_total': total_loss,
            'loss_hard': loss_hard,
            'loss_soft': loss_soft,
            'loss_risk': loss_risk,
            'loss_boundary_oc': loss_boundary_oc,
            'hard_target': hard_target,
            'mixed_prob': corrected_prob_soft,
            'score': score,
            'seed_support_map': score.new_zeros(score.shape),
            'hard_ratio': self.adaptive_pl_last_hard_ratio,
            'client_risk': self.adaptive_pl_last_client_risk,
            'correction_strength': calibration_control,
        }

    def _compute_adaptive_pl_loss_v1_backbone_aux(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                                  pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch,
                                                  raw_encoder_feature=None):
        # V1 keeps W1' as the always-on backbone and uses the older
        # risk statistics only as a lightweight auxiliary objective.
        base_result = self._compute_adaptive_pl_loss_w1p_pure(
            outputs=outputs,
            outputs_auxiliary=outputs_auxiliary,
            outputs_soft=outputs_soft,
            outputs_soft_auxiliary=outputs_soft_auxiliary,
            pseudo_label_mix=pseudo_label_mix,
            global_pseudo_label_mix=global_pseudo_label_mix,
            geometry_bin_batch=geometry_bin_batch,
            label_batch=label_batch,
        )

        if global_pseudo_label_mix is None:
            global_pseudo_label_mix = pseudo_label_mix.detach()

        reliability = compute_adaptive_pseudo_reliability(
            pseudo_label_mix,
            global_pseudo_label_mix,
            gamma_prob=float(getattr(self.args, 'adaptive_pl_gamma_prob', 4.0)),
            gamma_conf=float(getattr(self.args, 'adaptive_pl_gamma_conf', 3.0)),
        )
        global_prob = reliability['global_prob']
        conf_global = reliability['conf_global']
        score = reliability['score']
        risk_geometry_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=None,
            apply_support_bonus=False,
        )
        tau_values = torch.tensor(self.adaptive_pl_tau, device=geometry_bin_batch.device, dtype=score.dtype)
        tau_map = tau_values[geometry_bin_batch.long().clamp(0, len(self.adaptive_pl_tau) - 1)]
        valid_mask = (label_batch == self.args.num_classes).bool()
        adaptive_active = self.current_iter >= int(getattr(self.args, 'adaptive_pl_warmup_iters', 800))
        tau_global_min = float(getattr(self.args, 'adaptive_pl_global_min_conf', 0.6))
        risk_calibration_enabled = bool(getattr(self.args, 'risk_calibration_enabled', 0))
        risk_agreement_credit = float(getattr(self.args, 'risk_calibration_risk_agreement_credit', 0.20))
        risk_global_soft_lambda = float(getattr(self.args, 'risk_calibration_global_soft_lambda', 0.15))
        if bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
            aux_max_weight = float(getattr(self.args, 'risk_calibration_v1_aux_max_weight', 0.30))
        else:
            aux_max_weight = float(getattr(self.args, 'risk_calibration_w7_aux_max_weight', 0.30))

        client_risk = 0.0
        correction_strength = 0.0
        aux_loss = outputs.new_tensor(0.0)
        self.adaptive_pl_last_risk_term_prior = 0.0
        self.adaptive_pl_last_risk_term_prob = 0.0
        self.adaptive_pl_last_risk_term_conf = 0.0
        self.adaptive_pl_last_risk_term_disagree = 0.0
        self.adaptive_pl_last_risk_term_hard = 0.0
        self.adaptive_pl_last_risk_term_credit = 0.0
        if risk_calibration_enabled and adaptive_active and bool(valid_mask.any().item()):
            risk_terms = self._compute_relative_client_risk()
            hard_weight = float(np.clip(risk_terms['hard_weight'], 0.0, 0.5))
            base_weights = {
                'prior': 0.40,
                'prob': 0.20,
                'conf': 0.10,
                'disagree': 0.30,
            }
            weight_scale = max(1e-6, 1.0 - hard_weight)
            client_risk = float(np.clip(
                weight_scale * base_weights['prior'] * risk_terms['prior']
                + weight_scale * base_weights['prob'] * risk_terms['prob']
                + weight_scale * base_weights['conf'] * risk_terms['conf']
                + weight_scale * base_weights['disagree'] * risk_terms['disagree']
                + hard_weight * risk_terms['hard']
                - (risk_agreement_credit * risk_terms['credit']),
                0.0,
                1.0,
            ))
            correction_strength = float(np.clip(aux_max_weight * client_risk, 0.0, aux_max_weight))

            both_uncertain = valid_mask & (score < tau_map) & (conf_global < tau_global_min)
            risk_weight = risk_geometry_weight_map * both_uncertain.float() * valid_mask.float()
            if risk_weight.sum().item() > 0 and correction_strength > 0.0:
                loss_corr_1 = weighted_reverse_kl_loss(outputs, global_prob, risk_weight)
                loss_corr_2 = weighted_reverse_kl_loss(outputs_auxiliary, global_prob, risk_weight)
                aux_loss = risk_global_soft_lambda * correction_strength * (0.5 * (loss_corr_1 + loss_corr_2))

        merged_result = dict(base_result)
        merged_result['loss_total'] = base_result['loss_total'] + aux_loss
        merged_result['loss_risk'] = aux_loss
        merged_result['client_risk'] = client_risk
        merged_result['correction_strength'] = correction_strength

        self.adaptive_pl_last_client_risk = client_risk
        self.adaptive_pl_last_loss_total = float(merged_result['loss_total'].detach().item())
        self.adaptive_pl_last_loss_total_backbone = float(base_result['loss_total'].detach().item())
        self.adaptive_pl_last_loss_aux = float(aux_loss.detach().item())
        self.adaptive_pl_last_loss_risk = float(aux_loss.detach().item())
        self.adaptive_pl_last_effective_correction = correction_strength
        self.adaptive_pl_last_calibration_control = correction_strength
        self.adaptive_pl_last_preserve_score = 0.0
        self.adaptive_pl_last_correction_score = 0.0
        self.adaptive_pl_last_hard_control = 0.0
        self.adaptive_pl_last_release_score = 0.0
        self.adaptive_pl_last_regime_code = 0.0
        self.adaptive_pl_last_release_ready = 0.0
        self.adaptive_pl_last_high_risk = 0.0
        self.adaptive_pl_release_streak = 0.0
        self.adaptive_pl_gate_release_armed = 0.0
        return merged_result

    def _compute_adaptive_pl_loss_w7_backbone_aux(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                                  pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch,
                                                  raw_encoder_feature=None):
        # W7 keeps its original two-stage semantics: early risk-calibrated path,
        # then a hard return to pure W1' after the switch iteration.
        switch_iter = int(getattr(self.args, 'risk_calibration_w7_switch_iter', 800))
        if self.current_iter >= switch_iter:
            return self._compute_adaptive_pl_loss_w1p_pure(
                outputs=outputs,
                outputs_auxiliary=outputs_auxiliary,
                outputs_soft=outputs_soft,
                outputs_soft_auxiliary=outputs_soft_auxiliary,
                pseudo_label_mix=pseudo_label_mix,
                global_pseudo_label_mix=global_pseudo_label_mix,
                geometry_bin_batch=geometry_bin_batch,
                label_batch=label_batch,
            )

        prev_w7_enabled = getattr(self.args, 'risk_calibration_w7_enabled', 0)
        try:
            self.args.risk_calibration_w7_enabled = 0
            return self._compute_adaptive_pl_loss(
                outputs=outputs,
                outputs_auxiliary=outputs_auxiliary,
                outputs_soft=outputs_soft,
                outputs_soft_auxiliary=outputs_soft_auxiliary,
                pseudo_label_mix=pseudo_label_mix,
                global_pseudo_label_mix=global_pseudo_label_mix,
                geometry_bin_batch=geometry_bin_batch,
                label_batch=label_batch,
                raw_encoder_feature=raw_encoder_feature,
            )
        finally:
            self.args.risk_calibration_w7_enabled = prev_w7_enabled

    def _compute_adaptive_pl_loss(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                  pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch,
                                  raw_encoder_feature=None):
        if bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
            return self._compute_adaptive_pl_loss_v1_backbone_aux(
                outputs=outputs,
                outputs_auxiliary=outputs_auxiliary,
                outputs_soft=outputs_soft,
                outputs_soft_auxiliary=outputs_soft_auxiliary,
                pseudo_label_mix=pseudo_label_mix,
                global_pseudo_label_mix=global_pseudo_label_mix,
                geometry_bin_batch=geometry_bin_batch,
                label_batch=label_batch,
                raw_encoder_feature=raw_encoder_feature,
            )
        if bool(getattr(self.args, 'risk_calibration_w7_enabled', 0)):
            return self._compute_adaptive_pl_loss_w7_backbone_aux(
                outputs=outputs,
                outputs_auxiliary=outputs_auxiliary,
                outputs_soft=outputs_soft,
                outputs_soft_auxiliary=outputs_soft_auxiliary,
                pseudo_label_mix=pseudo_label_mix,
                global_pseudo_label_mix=global_pseudo_label_mix,
                geometry_bin_batch=geometry_bin_batch,
                label_batch=label_batch,
                raw_encoder_feature=raw_encoder_feature,
            )
        if bool(getattr(self.args, 'risk_calibration_w5_soft_only_enabled', 0)):
            return self._compute_adaptive_pl_loss_w5_from_w1p(
                outputs=outputs,
                outputs_auxiliary=outputs_auxiliary,
                outputs_soft=outputs_soft,
                outputs_soft_auxiliary=outputs_soft_auxiliary,
                pseudo_label_mix=pseudo_label_mix,
                global_pseudo_label_mix=global_pseudo_label_mix,
                geometry_bin_batch=geometry_bin_batch,
                label_batch=label_batch,
            )
        if global_pseudo_label_mix is None:
            global_pseudo_label_mix = pseudo_label_mix.detach()
        reliability = compute_adaptive_pseudo_reliability(
            pseudo_label_mix,
            global_pseudo_label_mix,
            gamma_prob=float(getattr(self.args, 'adaptive_pl_gamma_prob', 4.0)),
            gamma_conf=float(getattr(self.args, 'adaptive_pl_gamma_conf', 3.0)),
        )
        local_prob = reliability['local_prob']
        global_prob = reliability['global_prob']
        conf_global = reliability['conf_global']
        prob_gap = reliability['prob_gap']
        conf_gap = reliability['conf_gap']
        agree_lg = reliability['agree_lg']
        score = reliability['score']
        risk_geometry_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=None,
            apply_support_bonus=False,
        )
        tau_values = torch.tensor(self.adaptive_pl_tau, device=geometry_bin_batch.device, dtype=score.dtype)
        tau_map = tau_values[geometry_bin_batch.long().clamp(0, len(self.adaptive_pl_tau) - 1)]
        unlabeled_mask = label_batch == self.args.num_classes

        ema_momentum = 0.9
        min_pixels = int(getattr(self.args, 'adaptive_pl_min_pixels_per_bin', 64))
        warmup_iters = int(getattr(self.args, 'adaptive_pl_warmup_iters', 800))
        tau_update_enabled = bool(getattr(self.args, 'adaptive_pl_tau_update', 1))
        adaptive_active = self.current_iter >= warmup_iters
        target_accept = float(getattr(self.args, 'adaptive_pl_target_accept', 0.35))
        tau_global_min = float(getattr(self.args, 'adaptive_pl_global_min_conf', 0.6))
        blend_kappa = float(getattr(self.args, 'adaptive_pl_blend_kappa', 2.0))
        boundary_lambda = float(getattr(self.args, 'adaptive_pl_boundary_lambda', 0.0))
        boundary_kernel_size = int(getattr(self.args, 'adaptive_pl_boundary_kernel_size', 3))
        boundary_warmup_iters = int(getattr(self.args, 'adaptive_pl_boundary_warmup_iters', warmup_iters))
        seed_support_enabled = bool(getattr(self.args, 'seed_support_enabled', 0))
        seed_support_warmup_iters = int(getattr(self.args, 'seed_support_warmup_iters', warmup_iters))
        seed_support_reliable_score_min = float(getattr(self.args, 'seed_support_reliable_score_min', 0.7))
        seed_support_kernel_size = int(getattr(self.args, 'seed_support_kernel_size', 5))
        seed_support_affinity_temp = float(getattr(self.args, 'seed_support_affinity_temp', 8.0))
        seed_support_blend_alpha = float(getattr(self.args, 'seed_support_blend_alpha', 0.5))
        seed_support_soft_tau = float(getattr(self.args, 'seed_support_soft_tau', 0.20))
        risk_calibration_enabled = bool(getattr(self.args, 'risk_calibration_enabled', 0))
        risk_tau_lambda = float(getattr(self.args, 'risk_calibration_tau_lambda', 0.12))
        risk_conf_lambda = float(getattr(self.args, 'risk_calibration_conf_lambda', 0.08))
        risk_prior_power = float(getattr(self.args, 'risk_calibration_prior_power', 1.0))
        risk_prior_clip_min = float(getattr(self.args, 'risk_calibration_prior_clip_min', 0.5))
        risk_prior_clip_max = float(getattr(self.args, 'risk_calibration_prior_clip_max', 1.5))
        risk_hard_dampen = float(getattr(self.args, 'risk_calibration_hard_dampen', 0.5))
        risk_soft_boost = float(getattr(self.args, 'risk_calibration_soft_boost', 0.5))
        risk_global_soft_lambda = float(getattr(self.args, 'risk_calibration_global_soft_lambda', 0.15))
        stateful_release_enabled = bool(getattr(self.args, 'risk_calibration_stateful_release_enabled', 0))
        release_start_iter = int(getattr(self.args, 'risk_calibration_release_start_iter', warmup_iters))
        release_full_iter = int(getattr(self.args, 'risk_calibration_release_full_iter', max(warmup_iters + 1, warmup_iters + 1000)))
        regime_ema_momentum = float(getattr(self.args, 'risk_calibration_regime_ema_momentum', 0.9))
        release_risk_threshold = float(getattr(self.args, 'risk_calibration_release_risk_threshold', 0.35))
        release_agreement_threshold = float(getattr(self.args, 'risk_calibration_release_agreement_threshold', 0.90))
        release_prior_gap_threshold = float(getattr(self.args, 'risk_calibration_release_prior_gap_threshold', 0.015))
        release_uncertain_threshold = float(getattr(self.args, 'risk_calibration_release_uncertain_threshold', 0.02))
        preserve_risk_threshold = float(getattr(self.args, 'risk_calibration_preserve_risk_threshold', 0.65))
        preserve_agreement_threshold = float(getattr(self.args, 'risk_calibration_preserve_agreement_threshold', 0.92))
        preserve_prior_gap_threshold = float(getattr(self.args, 'risk_calibration_preserve_prior_gap_threshold', 0.025))
        preserve_uncertain_threshold = float(getattr(self.args, 'risk_calibration_preserve_uncertain_threshold', 0.04))
        preserve_hard_ratio_threshold = float(getattr(self.args, 'risk_calibration_preserve_hard_ratio_threshold', 0.60))
        correction_risk_threshold = float(getattr(self.args, 'risk_calibration_correction_risk_threshold', 0.75))
        correction_prior_gap_threshold = float(getattr(self.args, 'risk_calibration_correction_prior_gap_threshold', 0.03))
        correction_agreement_threshold = float(getattr(self.args, 'risk_calibration_correction_agreement_threshold', 0.88))
        correction_uncertain_threshold = float(getattr(self.args, 'risk_calibration_correction_uncertain_threshold', 0.06))
        risk_agreement_credit = float(getattr(self.args, 'risk_calibration_risk_agreement_credit', 0.20))
        release_min_correction = float(getattr(self.args, 'risk_calibration_release_min_correction', 0.02))
        release_max_correction = float(getattr(self.args, 'risk_calibration_release_max_correction', 0.12))
        preserve_min_correction = float(getattr(self.args, 'risk_calibration_preserve_min_correction', 0.25))
        preserve_max_correction = float(getattr(self.args, 'risk_calibration_preserve_max_correction', 0.55))
        correction_min_correction = float(getattr(self.args, 'risk_calibration_correction_min_correction', 0.75))
        regime_hysteresis = float(getattr(self.args, 'risk_calibration_regime_hysteresis', 0.05))
        anneal_end_iter = int(getattr(self.args, 'risk_calibration_anneal_end_iter', -1))
        hard_min_correction = float(getattr(self.args, 'risk_calibration_hard_min_correction', 0.05))
        hard_risk_scale = float(getattr(self.args, 'risk_calibration_hard_risk_scale', 0.15))
        calibration_min_correction = float(getattr(self.args, 'risk_calibration_calibration_min_correction', 0.10))
        calibration_risk_scale = float(getattr(self.args, 'risk_calibration_calibration_risk_scale', 0.40))
        w5_soft_only_enabled = bool(getattr(self.args, 'risk_calibration_w5_soft_only_enabled', 0))
        w5_start_iter = int(getattr(self.args, 'risk_calibration_w5_start_iter', warmup_iters))
        w5_peak_iter = int(getattr(self.args, 'risk_calibration_w5_peak_iter', max(warmup_iters + 1, warmup_iters + 600)))
        w5_end_iter = int(getattr(self.args, 'risk_calibration_w5_end_iter', max(warmup_iters + 2, warmup_iters + 1800)))
        w5_min_correction = float(getattr(self.args, 'risk_calibration_w5_min_correction', 0.10))
        w5_risk_scale = float(getattr(self.args, 'risk_calibration_w5_risk_scale', 0.40))
        outer_beta = float(getattr(self.args, 'beta', 1.0))

        valid_mask = unlabeled_mask.bool()
        if valid_mask.any():
            lg_agree_mean = float(agree_lg[valid_mask].float().mean().item())
            prob_gap_mean = float(prob_gap[valid_mask].mean().item())
            conf_gap_mean = float(conf_gap[valid_mask].mean().item())
            self.adaptive_pl_lg_agreement_ema = (ema_momentum * self.adaptive_pl_lg_agreement_ema) + ((1.0 - ema_momentum) * lg_agree_mean)
            self.adaptive_pl_prob_gap_ema = (ema_momentum * self.adaptive_pl_prob_gap_ema) + ((1.0 - ema_momentum) * prob_gap_mean)
            self.adaptive_pl_conf_gap_ema = (ema_momentum * self.adaptive_pl_conf_gap_ema) + ((1.0 - ema_momentum) * conf_gap_mean)
            self.adaptive_pl_last_lg_class_agree_ratio = lg_agree_mean
            self.adaptive_pl_last_mean_prob_gap = prob_gap_mean
            self.adaptive_pl_last_mean_conf_gap = conf_gap_mean
            local_prior = compute_masked_class_prior(local_prob, valid_mask)
            global_prior = compute_masked_class_prior(global_prob, valid_mask)
            prior_gap = float(torch.mean(torch.abs(local_prior[1:] - global_prior[1:])).item())
            self.adaptive_pl_prior_gap_ema = (ema_momentum * self.adaptive_pl_prior_gap_ema) + ((1.0 - ema_momentum) * prior_gap)
            self.adaptive_pl_last_prior_gap = prior_gap
        else:
            self.adaptive_pl_last_lg_class_agree_ratio = 0.0
            self.adaptive_pl_last_mean_prob_gap = 0.0
            self.adaptive_pl_last_mean_conf_gap = 0.0
            local_prior = None
            global_prior = None
            self.adaptive_pl_last_prior_gap = 0.0

        if risk_calibration_enabled and valid_mask.any():
            risk_terms = self._compute_relative_client_risk()
            hard_weight = float(np.clip(risk_terms['hard_weight'], 0.0, 0.5))
            base_weights = {
                'prior': 0.40,
                'prob': 0.20,
                'conf': 0.10,
                'disagree': 0.30,
            }
            weight_scale = max(1e-6, 1.0 - hard_weight)
            risk_score = (
                weight_scale * base_weights['prior'] * risk_terms['prior']
                + weight_scale * base_weights['prob'] * risk_terms['prob']
                + weight_scale * base_weights['conf'] * risk_terms['conf']
                + weight_scale * base_weights['disagree'] * risk_terms['disagree']
                + hard_weight * risk_terms['hard']
                - (risk_agreement_credit * risk_terms['credit'])
            )
            risk_score = float(np.clip(risk_score, 0.0, 1.0))
        else:
            self.adaptive_pl_last_risk_term_prior = 0.0
            self.adaptive_pl_last_risk_term_prob = 0.0
            self.adaptive_pl_last_risk_term_conf = 0.0
            self.adaptive_pl_last_risk_term_disagree = 0.0
            self.adaptive_pl_last_risk_term_hard = 0.0
            self.adaptive_pl_last_risk_term_credit = 0.0
            risk_score = 0.0
        self.adaptive_pl_last_client_risk = risk_score

        stage_progress = 0.0
        release_score = 0.0
        preserve_score = 0.0
        correction_score = 0.0
        effective_correction = 0.0
        hard_control = 0.0
        calibration_control = 0.0
        regime_code = 0.0
        release_ready_flag = 0.0
        high_risk_flag = 0.0
        if risk_calibration_enabled:
            effective_correction = 1.0
            regime_code = 0.0
        if risk_calibration_enabled and w5_soft_only_enabled:
            w5_start_iter = max(warmup_iters, w5_start_iter)
            w5_peak_iter = max(w5_start_iter + 1, w5_peak_iter)
            w5_end_iter = max(w5_peak_iter + 1, w5_end_iter)
            if self.current_iter < w5_start_iter:
                stage_progress = 0.0
            elif self.current_iter < w5_peak_iter:
                stage_progress = float(np.clip(
                    (self.current_iter - w5_start_iter) / float(w5_peak_iter - w5_start_iter),
                    0.0,
                    1.0,
                ))
            elif self.current_iter < w5_end_iter:
                stage_progress = float(np.clip(
                    (w5_end_iter - self.current_iter) / float(w5_end_iter - w5_peak_iter),
                    0.0,
                    1.0,
                ))
            else:
                stage_progress = 0.0
            soft_target = float(np.clip(w5_min_correction + (w5_risk_scale * risk_score), 0.0, 1.0))
            calibration_control = float(np.clip(stage_progress * soft_target, 0.0, 1.0))
            hard_control = 0.0
            release_score = 0.0
            preserve_score = calibration_control
            correction_score = risk_score
            effective_correction = calibration_control
            self.adaptive_pl_release_score_ema = regime_ema_momentum * self.adaptive_pl_release_score_ema
            self.adaptive_pl_preserve_score_ema = (
                regime_ema_momentum * self.adaptive_pl_preserve_score_ema
            ) + ((1.0 - regime_ema_momentum) * calibration_control)
            self.adaptive_pl_correction_score_ema = (
                regime_ema_momentum * self.adaptive_pl_correction_score_ema
            ) + ((1.0 - regime_ema_momentum) * risk_score)
            regime_code = 0.0
            self.adaptive_pl_release_streak = 0.0
            self.adaptive_pl_gate_release_armed = 0.0
        elif risk_calibration_enabled and stateful_release_enabled:
            anneal_start_iter = max(warmup_iters, release_start_iter)
            if anneal_end_iter < 0:
                anneal_end_iter = release_full_iter
            anneal_end_iter = max(anneal_start_iter + 1, anneal_end_iter)
            if self.current_iter >= anneal_start_iter:
                # Replace discrete W4.2 routing with continuous late-stage annealing.
                stage_progress = float(np.clip(
                    (self.current_iter - anneal_start_iter) / float(anneal_end_iter - anneal_start_iter),
                    0.0,
                    1.0,
                ))
            hard_target = float(np.clip(hard_min_correction + (hard_risk_scale * risk_score), 0.0, 1.0))
            calibration_target = float(np.clip(calibration_min_correction + (calibration_risk_scale * risk_score), 0.0, 1.0))
            # Keep strong correction before anneal starts, then gradually inject risk-aware late controls.
            hard_control = float(np.clip(((1.0 - stage_progress) * 1.0) + (stage_progress * hard_target), 0.0, 1.0))
            calibration_control = float(np.clip(((1.0 - stage_progress) * 1.0) + (stage_progress * calibration_target), 0.0, 1.0))
            release_score = hard_control
            preserve_score = calibration_control
            correction_score = risk_score
            effective_correction = calibration_control
            self.adaptive_pl_release_score_ema = (
                regime_ema_momentum * self.adaptive_pl_release_score_ema
            ) + ((1.0 - regime_ema_momentum) * hard_control)
            self.adaptive_pl_preserve_score_ema = (
                regime_ema_momentum * self.adaptive_pl_preserve_score_ema
            ) + ((1.0 - regime_ema_momentum) * calibration_control)
            self.adaptive_pl_correction_score_ema = (
                regime_ema_momentum * self.adaptive_pl_correction_score_ema
            ) + ((1.0 - regime_ema_momentum) * risk_score)
            regime_code = 0.0
            self.adaptive_pl_release_streak = 0.0
            self.adaptive_pl_gate_release_armed = 0.0
        elif risk_calibration_enabled:
            self.adaptive_pl_release_score_ema = regime_ema_momentum * self.adaptive_pl_release_score_ema
            self.adaptive_pl_preserve_score_ema = regime_ema_momentum * self.adaptive_pl_preserve_score_ema
            self.adaptive_pl_correction_score_ema = (
                regime_ema_momentum * self.adaptive_pl_correction_score_ema
            ) + ((1.0 - regime_ema_momentum) * risk_score)
            release_score = 0.0
            preserve_score = 0.0
            correction_score = 1.0
            effective_correction = 1.0
            hard_control = 1.0
            calibration_control = 1.0
            regime_code = 0.0
            self.adaptive_pl_release_streak = 0.0
            self.adaptive_pl_gate_release_armed = 0.0
        else:
            self.adaptive_pl_release_score_ema = regime_ema_momentum * self.adaptive_pl_release_score_ema
            self.adaptive_pl_preserve_score_ema = regime_ema_momentum * self.adaptive_pl_preserve_score_ema
            self.adaptive_pl_correction_score_ema = regime_ema_momentum * self.adaptive_pl_correction_score_ema
            self.adaptive_pl_release_streak = 0.0
            self.adaptive_pl_gate_release_armed = 0.0
        self.adaptive_pl_last_stage_progress = stage_progress
        self.adaptive_pl_last_release_score = release_score
        self.adaptive_pl_last_preserve_score = preserve_score
        self.adaptive_pl_last_correction_score = correction_score
        self.adaptive_pl_last_effective_correction = effective_correction
        self.adaptive_pl_last_hard_control = hard_control
        self.adaptive_pl_last_calibration_control = calibration_control
        self.adaptive_pl_last_regime_code = regime_code
        self.adaptive_pl_last_release_ready = release_ready_flag
        self.adaptive_pl_last_high_risk = high_risk_flag

        self.adaptive_pl_last_observed_accept[:] = np.nan
        self.adaptive_pl_last_score_mean[:] = np.nan
        self.adaptive_pl_last_bin_counts[:] = 0
        for bin_idx in range(len(self.adaptive_pl_tau)):
            bin_mask = (geometry_bin_batch == bin_idx) & valid_mask
            bin_count = int(bin_mask.sum().item())
            self.adaptive_pl_last_bin_counts[bin_idx] = bin_count
            if bin_count <= 0:
                continue
            score_bin = score[bin_mask]
            self.adaptive_pl_last_score_mean[bin_idx] = float(score_bin.mean().item())
            if bin_count < min_pixels:
                continue
            observed_accept = float((score_bin >= self.adaptive_pl_tau[bin_idx]).float().mean().item())
            self.adaptive_pl_last_observed_accept[bin_idx] = observed_accept
            if adaptive_active and tau_update_enabled:
                updated_tau = self.adaptive_pl_tau[bin_idx] + 0.01 * (observed_accept - target_accept)
                self.adaptive_pl_tau[bin_idx] = float(np.clip(updated_tau, 0.3, 0.9))

        lambda_dynamic = torch.exp(-blend_kappa * conf_gap).clamp(1e-6, 1.0)
        corrected_prob = lambda_dynamic.unsqueeze(1) * local_prob + (1.0 - lambda_dynamic).unsqueeze(1) * global_prob
        corrected_prob_soft = corrected_prob
        if risk_calibration_enabled and valid_mask.any():
            prior_scale = ((global_prior + 1e-6) / (local_prior + 1e-6)).clamp(
                min=risk_prior_clip_min,
                max=risk_prior_clip_max,
            )
            prior_scale = torch.pow(prior_scale, risk_prior_power * calibration_control)
            if w5_soft_only_enabled:
                corrected_prob_soft = corrected_prob_soft * prior_scale.view(1, -1, 1, 1)
                corrected_prob_soft = corrected_prob_soft / corrected_prob_soft.sum(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                corrected_prob = corrected_prob * prior_scale.view(1, -1, 1, 1)
                corrected_prob = corrected_prob / corrected_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
                corrected_prob_soft = corrected_prob
        seed_support_map = score.new_zeros(score.shape)
        propagated_fg_map = score.new_zeros(score.shape)
        soft_assist_mask = torch.zeros_like(score, dtype=torch.bool)
        seed_support_active = (
            seed_support_enabled
            and raw_encoder_feature is not None
            and self.current_iter >= seed_support_warmup_iters
        )
        if seed_support_active:
            seed_guided_prob, seed_support_map, propagated_fg_map = build_seed_support_guided_prob(
                raw_encoder_feature=raw_encoder_feature,
                weak_label_batch=label_batch,
                corrected_prob=corrected_prob_soft if w5_soft_only_enabled else corrected_prob,
                score_map=score,
                ignore_index=self.args.num_classes,
                reliable_score_min=seed_support_reliable_score_min,
                kernel_size=seed_support_kernel_size,
                affinity_temp=seed_support_affinity_temp,
                blend_alpha=seed_support_blend_alpha,
            )
            if not bool((seed_support_map > 0).any().item()):
                seed_support_active = False

        tau_map_effective = tau_map
        conf_global_cutoff = tau_global_min
        hard_client_scale = 1.0
        soft_client_scale = 1.0
        if risk_calibration_enabled and adaptive_active:
            tau_map_effective = (tau_map + (risk_tau_lambda * hard_control)).clamp(0.3, 0.95)
            conf_global_cutoff = min(0.95, tau_global_min + (risk_conf_lambda * hard_control))
            hard_client_scale = max(0.0, 1.0 - (risk_hard_dampen * hard_control))
            soft_client_scale = 1.0 + (risk_soft_boost * calibration_control)

        if adaptive_active:
            both_uncertain = valid_mask & (score < tau_map_effective) & (conf_global < conf_global_cutoff)
            hard_mask = valid_mask & (~both_uncertain) & (score >= tau_map_effective)
            soft_mask = valid_mask & (~both_uncertain) & (score < tau_map_effective)
        else:
            both_uncertain = torch.zeros_like(score, dtype=torch.bool)
            hard_mask = valid_mask
            soft_mask = torch.zeros_like(score, dtype=torch.bool)

        if seed_support_active:
            # Keep W1' gating intact; seed support is only allowed to refine soft targets.
            soft_assist_mask = soft_mask & (seed_support_map >= seed_support_soft_tau)
            if bool(soft_assist_mask.any().item()):
                corrected_prob_soft = torch.where(soft_assist_mask.unsqueeze(1), seed_guided_prob, corrected_prob_soft)
            else:
                seed_support_active = False

        hard_target = torch.argmax(corrected_prob, dim=1)
        soft_target = torch.argmax(corrected_prob_soft.detach(), dim=1)
        hard_supervision_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=hard_target.detach(),
            apply_support_bonus=True,
        )
        soft_supervision_weight_map = self._build_effective_geometry_weight_map(
            geometry_bin_batch,
            label_batch=label_batch,
            target_batch=soft_target,
            apply_support_bonus=True,
        )

        hard_weight = hard_supervision_weight_map * hard_mask.float() * valid_mask.float()
        soft_weight = soft_supervision_weight_map * soft_mask.float() * valid_mask.float()
        risk_weight = risk_geometry_weight_map * both_uncertain.float() * valid_mask.float()
        loss_hard_1 = weighted_pseudo_dice_loss(outputs_soft, hard_target, hard_weight)
        loss_hard_2 = weighted_pseudo_dice_loss(outputs_soft_auxiliary, hard_target, hard_weight)
        loss_soft_1 = weighted_reverse_kl_loss(outputs, corrected_prob_soft, soft_weight)
        loss_soft_2 = weighted_reverse_kl_loss(outputs_auxiliary, corrected_prob_soft, soft_weight)
        loss_risk_1 = weighted_reverse_kl_loss(outputs, global_prob, risk_weight)
        loss_risk_2 = weighted_reverse_kl_loss(outputs_auxiliary, global_prob, risk_weight)
        loss_hard = hard_client_scale * 0.5 * (loss_hard_1 + loss_hard_2)
        loss_soft = soft_client_scale * 0.5 * (loss_soft_1 + loss_soft_2)
        loss_risk = risk_score * 0.5 * (loss_risk_1 + loss_risk_2)
        total_loss = loss_hard + float(getattr(self.args, 'adaptive_pl_soft_lambda', 0.2)) * loss_soft
        if risk_calibration_enabled and adaptive_active:
            total_loss = total_loss + (risk_global_soft_lambda * calibration_control * loss_risk)
        valid_count = float(valid_mask.float().sum().item())

        loss_boundary_oc = outputs.new_tensor(0.0)
        if self.current_iter >= boundary_warmup_iters and boundary_lambda > 0.0:
            with torch.no_grad():
                pl_target = torch.argmax(corrected_prob_soft.detach(), dim=1)
                target_mask_oc = pl_target == 1
                ring_oc = morphology_boundary(target_mask_oc, kernel_size=boundary_kernel_size)
                ring_valid_oc = ring_oc & (geometry_bin_batch <= 2) & valid_mask
            if ring_valid_oc.any():
                pred_prob_oc = F.softmax(outputs, dim=1)[:, 1]
                loss_boundary_oc = boundary_dice_loss(
                    pred_prob_oc,
                    target_mask_oc.float(),
                    ring_valid_oc.float(),
                )
                # The whole adaptive PL term is later multiplied by outer beta.
                # Compensate here so the effective boundary weight matches boundary_lambda.
                effective_boundary_scale = boundary_lambda / max(outer_beta, 1e-8)
                total_loss = total_loss + (effective_boundary_scale * loss_boundary_oc)
                if valid_count > 0:
                    self.adaptive_pl_last_ring_valid_oc_ratio = float(
                        ring_valid_oc.float().sum().item() / valid_count
                    )
                else:
                    self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
            else:
                self.adaptive_pl_last_ring_valid_oc_ratio = 0.0
        else:
            self.adaptive_pl_last_ring_valid_oc_ratio = 0.0

        if valid_count > 0:
            self.adaptive_pl_last_hard_ratio = float(hard_mask.float().sum().item() / valid_count)
            self.adaptive_pl_last_both_uncertain_ratio = float(both_uncertain.float().sum().item() / valid_count)
            self.adaptive_pl_both_uncertain_ema = (
                regime_ema_momentum * self.adaptive_pl_both_uncertain_ema
            ) + ((1.0 - regime_ema_momentum) * self.adaptive_pl_last_both_uncertain_ratio)
            if seed_support_active:
                self.adaptive_pl_last_seed_support_mean = float(seed_support_map[valid_mask].mean().item())
                self.adaptive_pl_last_seed_support_soft_ratio = float(soft_assist_mask.float().sum().item() / valid_count)
                self.adaptive_pl_last_propagated_fg_mean = float(propagated_fg_map[valid_mask].mean().item())
            else:
                self.adaptive_pl_last_seed_support_mean = 0.0
                self.adaptive_pl_last_seed_support_soft_ratio = 0.0
                self.adaptive_pl_last_propagated_fg_mean = 0.0
        else:
            self.adaptive_pl_last_hard_ratio = 0.0
            self.adaptive_pl_last_both_uncertain_ratio = 0.0
            self.adaptive_pl_both_uncertain_ema = regime_ema_momentum * self.adaptive_pl_both_uncertain_ema
            self.adaptive_pl_last_seed_support_mean = 0.0
            self.adaptive_pl_last_seed_support_soft_ratio = 0.0
            self.adaptive_pl_last_propagated_fg_mean = 0.0
        self.adaptive_pl_last_loss_hard = float(loss_hard.detach().item())
        self.adaptive_pl_last_loss_soft = float(loss_soft.detach().item())
        self.adaptive_pl_last_loss_total = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_total_backbone = float(total_loss.detach().item())
        self.adaptive_pl_last_loss_aux = float((risk_global_soft_lambda * calibration_control * loss_risk).detach().item()) if (risk_calibration_enabled and adaptive_active) else 0.0
        self.adaptive_pl_last_loss_risk = float(loss_risk.detach().item())
        self.adaptive_pl_last_boundary_loss_oc = float(loss_boundary_oc.detach().item())

        return {
            'loss_total': total_loss,
            'loss_hard': loss_hard,
            'loss_soft': loss_soft,
            'loss_risk': loss_risk,
            'loss_boundary_oc': loss_boundary_oc,
            'hard_target': hard_target,
            'mixed_prob': corrected_prob_soft,
            'score': score,
            'seed_support_map': seed_support_map,
            'hard_ratio': self.adaptive_pl_last_hard_ratio,
            'client_risk': self.adaptive_pl_last_client_risk,
        }

    def _build_optimizer(self, target_model, lr=None):
        optimizer_lr = self.current_lr if lr is None else lr
        if self.args.strategy == 'FedRep':
            local_keys = get_fedrep_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
            decay_params, nondecay_params = [], []
            for name, param in target_model.named_parameters():
                if 'bias' in name or (name not in local_keys):
                    nondecay_params += [param]
                else:
                    decay_params += [param]
            optimize_params = [{'params': decay_params, 'weight_decay': 0.0001},
                            {'params': nondecay_params, 'weight_decay': 0}]
            optimizer = optim.SGD(optimize_params, lr=optimizer_lr,
                                momentum=0.9)
        else:
            optimizer = optim.AdamW(target_model.parameters(), lr=optimizer_lr, betas=(0.9, 0.999),
                                    eps=1e-8, weight_decay=1e-2, amsgrad=False)
        return optimizer

    def _set_optimizer_lr(self, optimizer, lr_value):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_value

    def _compute_decay_lr(self, iter_idx):
        progress = min(max(float(iter_idx) / float(self.args.max_iterations), 0.0), 1.0)
        return self.args.base_lr * (1.0 - progress) ** 0.9

    def _get_student_teacher_student_lr(self, iter_idx):
        lr_mode = getattr(self.args, 'student_teacher_student_lr_mode', 'global_decay')
        if lr_mode == 'fixed':
            student_lr = float(getattr(self.args, 'student_teacher_student_lr_value', -1.0))
            if student_lr <= 0.0:
                student_lr = float(self.args.base_lr)
            return student_lr
        return self._compute_decay_lr(iter_idx)

    def _get_student_teacher_kd_lambda(self, iter_idx):
        base_lambda = float(getattr(self.args, 'student_teacher_kd_lambda', 0.2))
        schedule = getattr(self.args, 'student_teacher_kd_schedule', 'constant')
        if schedule == 'cutoff':
            cutoff_iter = int(getattr(self.args, 'student_teacher_kd_cutoff_iter', self.args.max_iterations))
            return base_lambda if int(iter_idx) < cutoff_iter else 0.0
        if schedule == 'linear_decay':
            decay_start = int(getattr(self.args, 'student_teacher_kd_decay_start_iter', 0))
            decay_end = int(getattr(self.args, 'student_teacher_kd_decay_end_iter', self.args.max_iterations))
            if int(iter_idx) <= decay_start:
                return base_lambda
            if int(iter_idx) >= decay_end:
                return 0.0
            span = max(1, decay_end - decay_start)
            ratio = 1.0 - float(iter_idx - decay_start) / float(span)
            return base_lambda * max(0.0, min(1.0, ratio))
        return base_lambda

    def _apply_student_photometric_asymmetry(self, volume_batch):
        if int(getattr(self.args, 'student_teacher_asym_photometric', 0)) != 1:
            return volume_batch

        augmented_batch = volume_batch.clone()
        batch_size = augmented_batch.shape[0]
        view_shape = [batch_size] + [1] * (augmented_batch.dim() - 1)
        data_min = augmented_batch.amin(dim=tuple(range(2, augmented_batch.dim())), keepdim=True)
        data_max = augmented_batch.amax(dim=tuple(range(2, augmented_batch.dim())), keepdim=True)

        brightness = float(getattr(self.args, 'student_teacher_asym_brightness', 0.2))
        contrast = float(getattr(self.args, 'student_teacher_asym_contrast', 0.2))
        gamma = float(getattr(self.args, 'student_teacher_asym_gamma', 0.2))
        noise_std = float(getattr(self.args, 'student_teacher_asym_noise_std', 0.05))
        blur_prob = float(getattr(self.args, 'student_teacher_asym_blur_prob', 0.3))
        blur_kernel = int(getattr(self.args, 'student_teacher_asym_blur_kernel', 3))

        if brightness > 0.0:
            brightness_delta = (torch.rand(view_shape, device=augmented_batch.device, dtype=augmented_batch.dtype) * 2.0 - 1.0) * brightness
            augmented_batch = augmented_batch + brightness_delta

        if contrast > 0.0:
            contrast_scale = 1.0 + (torch.rand(view_shape, device=augmented_batch.device, dtype=augmented_batch.dtype) * 2.0 - 1.0) * contrast
            channel_mean = augmented_batch.mean(dim=tuple(range(2, augmented_batch.dim())), keepdim=True)
            augmented_batch = (augmented_batch - channel_mean) * contrast_scale + channel_mean

        if gamma > 0.0:
            gamma_scale = 1.0 + (torch.rand(view_shape, device=augmented_batch.device, dtype=augmented_batch.dtype) * 2.0 - 1.0) * gamma
            gamma_scale = gamma_scale.clamp(min=0.5)
            denom = (data_max - data_min).clamp_min(1e-6)
            normalized_batch = ((augmented_batch - data_min) / denom).clamp(0.0, 1.0)
            augmented_batch = normalized_batch.pow(gamma_scale) * denom + data_min

        if noise_std > 0.0:
            augmented_batch = augmented_batch + torch.randn_like(augmented_batch) * noise_std

        if blur_prob > 0.0 and blur_kernel > 1:
            if blur_kernel % 2 == 0:
                blur_kernel += 1
            blurred_batch = F.avg_pool2d(augmented_batch, kernel_size=blur_kernel, stride=1, padding=blur_kernel // 2)
            blur_mask = (torch.rand(view_shape, device=augmented_batch.device) < blur_prob)
            augmented_batch = torch.where(blur_mask, blurred_batch, augmented_batch)

        return augmented_batch.clamp(min=data_min, max=data_max)

    def _compute_student_teacher_kd_loss(self, student_outputs, teacher_outputs, kd_temperature):
        kd_map = F.kl_div(
            F.log_softmax(student_outputs / kd_temperature, dim=1),
            F.softmax(teacher_outputs / kd_temperature, dim=1),
            reduction='none',
        ) * (kd_temperature * kd_temperature)
        kd_loss_map = kd_map.sum(dim=1)
        kd_active_ratio = 1.0
        if int(getattr(self.args, 'student_teacher_kd_veto', 0)) == 1:
            weight_mode = getattr(self.args, 'student_teacher_kd_weight_mode', 'hard_confidence_veto')
            student_probs = torch.softmax(student_outputs.detach(), dim=1)
            if weight_mode == 'soft_entropy_disagreement':
                student_entropy = -(student_probs * torch.log(student_probs.clamp_min(1e-8))).sum(dim=1)
                entropy_norm = student_entropy / np.log(student_outputs.shape[1])
                teacher_pred = torch.argmax(teacher_outputs.detach(), dim=1)
                student_pred = torch.argmax(student_probs, dim=1)
                disagreement_mask = (student_pred != teacher_pred).float()
                kd_weight = torch.maximum(entropy_norm.clamp(0.0, 1.0), disagreement_mask)
                kd_active_ratio = float(kd_weight.mean().item())
                kd_weight_sum = kd_weight.sum()
                if kd_weight_sum.item() > 0:
                    return ((kd_loss_map * kd_weight).sum() / kd_weight_sum), kd_active_ratio
                return kd_loss_map.new_tensor(0.0), kd_active_ratio

            confidence_threshold = float(getattr(self.args, 'student_teacher_kd_veto_confidence', 0.85))
            student_confidence = student_probs.max(dim=1).values
            kd_mask = student_confidence < confidence_threshold
            kd_active_ratio = float(kd_mask.float().mean().item())
            if kd_mask.any():
                return kd_loss_map[kd_mask].mean(), kd_active_ratio
            return kd_loss_map.new_tensor(0.0), kd_active_ratio
        return kd_loss_map.mean(), kd_active_ratio

    def _compute_student_ema_loss(self, student_outputs, student_volume_batch):
        if not (bool(getattr(self.args, 'student_teacher_ema_enabled', 0)) and hasattr(self.model, 'student_ema_model')):
            return student_outputs.new_tensor(0.0)
        with torch.no_grad():
            ema_out = self.model.student_ema_model(student_volume_batch)
        ema_outputs = ema_out[0]
        return F.mse_loss(torch.softmax(student_outputs, dim=1), torch.softmax(ema_outputs, dim=1))

    def _update_student_ema_model(self):
        if not (bool(getattr(self.args, 'student_teacher_ema_enabled', 0)) and hasattr(self.model, 'student_ema_model')):
            return
        decay = float(getattr(self.args, 'student_teacher_ema_decay', 0.99))
        student_state_dict = self.model.model.state_dict()
        ema_state_dict = self.model.student_ema_model.state_dict()
        updated_state_dict = OrderedDict()
        for key, ema_tensor in ema_state_dict.items():
            if key not in student_state_dict:
                updated_state_dict[key] = ema_tensor
                continue
            student_tensor = student_state_dict[key].detach().to(ema_tensor.device)
            if torch.is_floating_point(ema_tensor):
                updated_state_dict[key] = decay * ema_tensor + (1.0 - decay) * student_tensor
            else:
                updated_state_dict[key] = student_tensor
        self.model.student_ema_model.load_state_dict(updated_state_dict, strict=False)
        self.model.student_ema_model.eval()
        for param in self.model.student_ema_model.parameters():
            param.requires_grad = False

    @contextmanager
    def _preserve_global_rng_state(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = torch.cuda.get_rng_state_all()
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

    def _get_round_optimizer(self, target_model):
        persistent_optimizer = bool(getattr(self.args, 'student_teacher_v1', 0)) and bool(getattr(self.args, 'student_teacher_persistent', 1))
        if persistent_optimizer:
            if self.optimizer is None:
                self.optimizer = self._build_optimizer(target_model)
            return self.optimizer
        self.optimizer = self._build_optimizer(target_model)
        return self.optimizer

    def _compute_feduniv2_losses(self, target_model, volume_batch, label_batch, geometry_bin_batch, config,
                                 ce_loss, gatecrf_loss, loss_gatedcrf_kernels_desc, loss_gatedcrf_radius,
                                 kl_loss, dice_loss, previous_prompt_state, prototype_bank,
                                 proto_filter_enabled, proto_filter_warmup, proto_filter_topk,
                                 global_pl_model=None):
        with autocast(enabled=self.amp):
            out = target_model(volume_batch)
            if self.args.model == 'unet_univ5':
                outputs, feature, de1, de2, de3, de4, prompts, fuse_feature, outputs_auxiliary, distribution_prompts, uni_prompts = out
            elif self.args.model in ['unet_univ3', 'unet_univ4', 'unet_univ2']:
                outputs, feature, de1, de2, de3, de4, prompts, fuse_feature, outputs_auxiliary = out
                distribution_prompts = prompts
                uni_prompts = prompts[self.args.cid].unsqueeze(0)
            else:
                raise NotImplementedError('student_teacher_teacher_track only supports FedUniV2/FedUniV2.1 UNet Univ models.')
            raw_encoder_feature = fuse_feature[:, : (fuse_feature.shape[1] - prompts[self.args.cid].shape[1])]

            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_soft_auxiliary = torch.softmax(outputs_auxiliary, dim=1)
            loss_ce_seg = ce_loss(outputs, label_batch[:].long())
            loss_ce_auxiliary = ce_loss(outputs_auxiliary, label_batch[:].long())
            loss_ce = 0.5 * (loss_ce_seg + loss_ce_auxiliary)

            loss_total = loss_ce
            out_gatedcrf = gatecrf_loss(
                outputs_soft,
                loss_gatedcrf_kernels_desc,
                loss_gatedcrf_radius,
                volume_batch,
                self.args.img_size,
                self.args.img_size
            )["loss"]
            loss_total = loss_total + 0.1 * out_gatedcrf

            loss_uni = torch.tensor(0.0).cuda()
            for other_client in range(self.args.min_num_clients):
                if other_client == self.args.cid:
                    continue
                loss_uni += kl_loss(prompts[self.args.cid], prompts[other_client].detach())
            loss_uni = -loss_uni / (self.args.min_num_clients - 1)
            loss_total = torch.add(loss_total, loss_uni, alpha=self.args.alpha)

            pseudo_alpha = np.random.uniform(0, 1)
            pseudo_label_mix = pseudo_alpha * outputs_soft.detach() + (1 - pseudo_alpha) * outputs_soft_auxiliary.detach()
            global_pseudo_label_mix = self._build_global_pseudo_label_mix(global_pl_model, volume_batch, pseudo_alpha)
            pseudo_label = torch.argmax(pseudo_label_mix, dim=1)

            use_proto_filter = (
                proto_filter_enabled
                and prototype_bank is not None
                and int(config.get('iter_global', 0)) > proto_filter_warmup
            )
            proto_filter_loss = torch.tensor(0.0).cuda()
            proto_filter_selected_ratio = 0.0
            proto_filter_candidate_count = 0
            proto_filter_selected_count = 0
            proto_filter_margin_mean = 0.0
            if use_proto_filter:
                raw_encoder_feature = fuse_feature[:, : (fuse_feature.shape[1] - prompts[self.args.cid].shape[1])]
                selected_low_mask, _, _, margin_mean, candidate_count, selected_count = select_top_margin_mask(
                    raw_encoder_feature=raw_encoder_feature,
                    pseudo_probs=0.5 * (outputs_soft.detach() + outputs_soft_auxiliary.detach()),
                    weak_label_batch=label_batch,
                    prototype_bank=prototype_bank,
                    ignore_index=self.args.num_classes,
                    topk_ratio=proto_filter_topk,
                )
                selected_full_mask = F.interpolate(
                    selected_low_mask.unsqueeze(1).float(),
                    size=outputs.shape[-2:],
                    mode='nearest',
                ).squeeze(1).bool()
                loss_pls_1 = masked_cross_entropy_loss(
                    outputs,
                    pseudo_label,
                    selected_full_mask,
                    ignore_index=self.args.num_classes,
                )
                loss_pls_2 = masked_cross_entropy_loss(
                    outputs_auxiliary,
                    pseudo_label,
                    selected_full_mask,
                    ignore_index=self.args.num_classes,
                )
                loss_pls = (loss_pls_1 + loss_pls_2) / 2
                proto_filter_loss = loss_pls.detach()
                proto_filter_candidate_count = candidate_count
                proto_filter_selected_count = selected_count
                proto_filter_selected_ratio = float(selected_count / max(candidate_count, 1))
                proto_filter_margin_mean = float(margin_mean.detach().item()) if torch.is_tensor(margin_mean) else float(margin_mean)
            else:
                geometry_weight_map = None
                if getattr(self.args, 'geometry_guided', 0) == 1 and geometry_bin_batch is not None:
                    geometry_weight_map = self._build_effective_geometry_weight_map(
                        geometry_bin_batch,
                        label_batch=label_batch,
                        target_batch=pseudo_label,
                        apply_support_bonus=True,
                    )
                if self._adaptive_pl_is_enabled() and geometry_weight_map is not None:
                    adaptive_pl_result = self._compute_adaptive_pl_loss(
                        outputs=outputs,
                        outputs_auxiliary=outputs_auxiliary,
                        outputs_soft=outputs_soft,
                        outputs_soft_auxiliary=outputs_soft_auxiliary,
                        pseudo_label_mix=pseudo_label_mix,
                        global_pseudo_label_mix=global_pseudo_label_mix,
                        geometry_bin_batch=geometry_bin_batch,
                        label_batch=label_batch,
                        raw_encoder_feature=raw_encoder_feature,
                    )
                    loss_pls = adaptive_pl_result['loss_total']
                    pseudo_label = adaptive_pl_result['hard_target']
                elif geometry_weight_map is None:
                    loss_pls_1 = dice_loss(outputs_soft, pseudo_label.unsqueeze(1))
                    loss_pls_2 = dice_loss(outputs_soft_auxiliary, pseudo_label.unsqueeze(1))
                    loss_pls = (loss_pls_1 + loss_pls_2) / 2
                else:
                    loss_pls_1 = weighted_pseudo_dice_loss(outputs_soft, pseudo_label, geometry_weight_map)
                    loss_pls_2 = weighted_pseudo_dice_loss(outputs_soft_auxiliary, pseudo_label, geometry_weight_map)
                    loss_pls = (loss_pls_1 + loss_pls_2) / 2
            loss_total = torch.add(loss_total, loss_pls, alpha=self.args.beta)

        distance_prompt = None
        distance_prompt_distribution = None
        distance_prompt_uni = None
        if previous_prompt_state is not None:
            prev_prompt, prev_prompt_dis, prev_prompt_uni = previous_prompt_state
            distance_prompt = F.l1_loss(prev_prompt.detach(), prompts[self.args.cid].detach())
            distance_prompt_distribution = F.l1_loss(prev_prompt_dis.detach(), distribution_prompts[self.args.cid].detach())
            distance_prompt_uni = F.l1_loss(prev_prompt_uni.detach(), uni_prompts.detach())
        next_prompt_state = (
            prompts[self.args.cid].detach(),
            distribution_prompts[self.args.cid].detach(),
            uni_prompts.detach(),
        )
        return {
            'loss_total': loss_total,
            'loss_ce': loss_ce,
            'loss_uni': loss_uni,
            'loss_pls': loss_pls,
            'outputs': outputs,
            'outputs_auxiliary': outputs_auxiliary,
            'feature': feature,
            'prompts': prompts,
            'distribution_prompts': distribution_prompts,
            'uni_prompts': uni_prompts,
            'pseudo_label': pseudo_label,
            'next_prompt_state': next_prompt_state,
            'distance_prompt': distance_prompt,
            'distance_prompt_distribution': distance_prompt_distribution,
            'distance_prompt_uni': distance_prompt_uni,
            'proto_filter_loss': proto_filter_loss,
            'proto_filter_selected_ratio': proto_filter_selected_ratio,
            'proto_filter_candidate_count': proto_filter_candidate_count,
            'proto_filter_selected_count': proto_filter_selected_count,
            'proto_filter_margin_mean': proto_filter_margin_mean,
        }

    def _train_student_teacher_teachertrack(self, config):
        self.model.train()
        self.model.teacher_model.train()
        for param in self.model.teacher_model.parameters():
            param.requires_grad = True

        student_optimizer = self._get_round_optimizer(self.model.model)
        teacher_optimizer = self._build_optimizer(
            self.model.teacher_model,
            lr=self._compute_decay_lr(self.current_iter),
        )

        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes)
        gatecrf_loss = ModelLossSemsegGatedCRF()
        dice_loss = losses.pDLoss(self.args.num_classes, ignore_index=self.args.num_classes)
        kl_loss = KLDivLoss()
        l1_loss = L1Loss()

        log(INFO, '{} iterations per epoch'.format(len(self.trainloader)))

        loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        loss_gatedcrf_radius = 5
        previous_prompts = None
        previous_prompts_dis = None
        previous_prompts_uni = None
        proto_filter_loss = torch.tensor(0.0).cuda()
        proto_filter_selected_ratio = 0.0
        proto_filter_candidate_count = 0
        proto_filter_selected_count = 0
        proto_filter_margin_mean = 0.0
        loss_kd = torch.tensor(0.0).cuda()
        loss_ema = torch.tensor(0.0).cuda()
        kd_active_ratio = 1.0
        teacher_student_diag = {
            'logit_cosine': 1.0,
            'prediction_disagreement': 0.0,
            'cka_down3': 1.0,
            'cka_down4': 1.0,
        }
        prototype_bank = None
        proto_filter_enabled = (
            getattr(self.args, 'prototype_filter_enabled', 0) == 1
            and self.args.strategy in ['FedUniV2', 'FedUniV2.1']
        )
        proto_filter_warmup = int(getattr(self.args, 'prototype_filter_warmup_rounds', 50))
        proto_filter_topk = float(getattr(self.args, 'prototype_filter_topk_ratio', 0.25))
        if proto_filter_enabled and int(config.get('proto_bank_ready', 0)) == 1:
            prototype_bank = decode_prototype_bank_from_config(
                config,
                device=next(self.model.model.parameters()).device,
            )
        global_pl_model = None
        if self._adaptive_pl_is_enabled():
            global_pl_model = copy.deepcopy(self.model)
            global_pl_model.eval()
            for param in global_pl_model.parameters():
                param.requires_grad = False
        global_teacher_pl_model = None
        global_student_pl_model = None
        if self._adaptive_pl_is_enabled():
            global_teacher_pl_model = copy.deepcopy(self.model.teacher_model)
            global_teacher_pl_model.eval()
            for param in global_teacher_pl_model.parameters():
                param.requires_grad = False
            global_student_pl_model = copy.deepcopy(self.model.model)
            global_student_pl_model.eval()
            for param in global_student_pl_model.parameters():
                param.requires_grad = False

        for i_iter in range(config['iters']):
            if self.current_iter % len(self.trainloader) == 0:
                print('generating sampled batches......')
                self.sampled_batches.clear()
                for _, sampled_batch in enumerate(self.trainloader):
                    self.sampled_batches.append(sampled_batch)

            self.teacher_current_lr = self._compute_decay_lr(self.current_iter)
            self.student_current_lr = self._get_student_teacher_student_lr(self.current_iter)
            self._set_optimizer_lr(teacher_optimizer, self.teacher_current_lr)
            self._set_optimizer_lr(student_optimizer, self.student_current_lr)
            self.current_lr = self.student_current_lr

            idx = self.current_iter % len(self.trainloader)
            sampled_batch = self.sampled_batches[idx]

            if self.args.img_class in ['faz', 'prostate']:
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            else:
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            geometry_bin_batch = None
            if getattr(self.args, 'geometry_guided', 0) == 1 and 'geometry_bin' in sampled_batch:
                geometry_bin_batch = sampled_batch['geometry_bin'].cuda().long()

            teacher_result = self._compute_feduniv2_losses(
                self.model.teacher_model,
                volume_batch,
                label_batch,
                geometry_bin_batch,
                config,
                ce_loss,
                gatecrf_loss,
                loss_gatedcrf_kernels_desc,
                loss_gatedcrf_radius,
                kl_loss,
                dice_loss,
                previous_prompt_state=None,
                prototype_bank=prototype_bank,
                proto_filter_enabled=proto_filter_enabled,
                proto_filter_warmup=proto_filter_warmup,
                proto_filter_topk=proto_filter_topk,
                global_pl_model=global_teacher_pl_model,
            )
            teacher_optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(teacher_result['loss_total']).backward()
                self.scaler.step(teacher_optimizer)
                self.scaler.update()
            else:
                teacher_result['loss_total'].backward()
                teacher_optimizer.step()

            self.model.teacher_model.eval()
            with torch.no_grad():
                teacher_eval_out = self.model.teacher_model(volume_batch)
            teacher_outputs = teacher_eval_out[0]
            teacher_feature = teacher_eval_out[1]
            # Keep teacher stochasticity on the baseline RNG path. Student-only augmentation
            # must happen after both teacher forwards, otherwise it perturbs teacher dropout/
            # prompt sampling randomness and changes the teacher trajectory.
            with self._preserve_global_rng_state():
                student_volume_batch = self._apply_student_photometric_asymmetry(volume_batch)
                student_result = self._compute_feduniv2_losses(
                    self.model.model,
                    student_volume_batch,
                    label_batch,
                    geometry_bin_batch,
                    config,
                    ce_loss,
                    gatecrf_loss,
                    loss_gatedcrf_kernels_desc,
                    loss_gatedcrf_radius,
                    kl_loss,
                    dice_loss,
                    previous_prompt_state=(previous_prompts, previous_prompts_dis, previous_prompts_uni) if previous_prompts is not None else None,
                    prototype_bank=prototype_bank,
                    proto_filter_enabled=proto_filter_enabled,
                    proto_filter_warmup=proto_filter_warmup,
                    proto_filter_topk=proto_filter_topk,
                    global_pl_model=global_student_pl_model,
                )
                loss_ema = self._compute_student_ema_loss(student_result['outputs'], student_volume_batch)
            kd_temperature = float(getattr(self.args, 'student_teacher_kd_temperature', 2.0))
            current_kd_lambda = self._get_student_teacher_kd_lambda(self.current_iter)
            loss_kd, kd_active_ratio = self._compute_student_teacher_kd_loss(
                student_result['outputs'],
                teacher_outputs,
                kd_temperature,
            )
            current_ema_lambda = float(getattr(self.args, 'student_teacher_ema_lambda', 0.05))
            student_loss_total = student_result['loss_total'] + current_kd_lambda * loss_kd + current_ema_lambda * loss_ema

            student_optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(student_loss_total).backward()
                self.scaler.step(student_optimizer)
                self.scaler.update()
            else:
                student_loss_total.backward()
                student_optimizer.step()
            self._update_student_ema_model()

            previous_prompts, previous_prompts_dis, previous_prompts_uni = student_result['next_prompt_state']
            teacher_student_diag = compute_teacher_student_diagnostics(
                student_result['outputs'].detach(),
                teacher_outputs.detach(),
                student_result['feature'],
                teacher_feature,
            )

            loss = student_loss_total
            loss_ce = student_result['loss_ce']
            loss_uni = student_result['loss_uni']
            loss_pls = student_result['loss_pls']
            outputs = student_result['outputs']
            outputs_auxiliary = student_result['outputs_auxiliary']
            prompts = student_result['prompts']
            pseudo_label = student_result['pseudo_label']
            proto_filter_loss = student_result['proto_filter_loss']
            proto_filter_selected_ratio = student_result['proto_filter_selected_ratio']
            proto_filter_candidate_count = student_result['proto_filter_candidate_count']
            proto_filter_selected_count = student_result['proto_filter_selected_count']
            proto_filter_margin_mean = student_result['proto_filter_margin_mean']

            self.current_iter = self.current_iter + 1
            log(INFO, 'client %d : iteration %d : lr: %f, loss : %f, loss_ce: %f' % (
                self.cid, self.current_iter, self.current_lr, loss.item(), loss_ce.item()))
            self.model.teacher_model.train()

        image = volume_batch[1, :, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        outputs_vis = outputs_vis[1, ...] * 50
        labs = label_batch[1, ...].unsqueeze(0) * 50
        if self.args.img_class in ['odoc', 'polyp']:
            outputs_vis, labs = outputs_vis.repeat(3, 1, 1), labs.repeat(3, 1, 1)

        metrics_ = {
            'client_{}_lr'.format(self.cid): self.current_lr,
            'client_{}_student_lr'.format(self.cid): self.student_current_lr,
            'client_{}_teacher_lr'.format(self.cid): self.teacher_current_lr,
            'client_{}_effective_kd_lambda'.format(self.cid): float(current_kd_lambda),
            'client_{}_kd_active_ratio'.format(self.cid): float(kd_active_ratio),
            'client_{}_student_refreshed'.format(self.cid): float(getattr(self.model, 'student_refreshed_this_round', False)),
            'client_{}_refresh_skip_ratio'.format(self.cid): float(getattr(self.model, 'refresh_skip_ratio', 0.0)),
            'client_{}_refresh_veto_teacher_drop'.format(self.cid): float(getattr(self.model, 'refresh_skipped_due_to_teacher_drop', False)),
            'client_{}_refresh_veto_student_best'.format(self.cid): float(getattr(self.model, 'refresh_skipped_due_to_student_best', False)),
            'client_{}_refresh_veto_applied'.format(self.cid): float(
                getattr(self.model, 'last_refresh_veto_reason', 'none') in ['teacher_drop', 'student_best']
            ),
            'client_{}_teacher_improvement_rate'.format(self.cid): float(getattr(self.model, 'teacher_improvement_rate', 0.0)),
            'client_{}_total_loss'.format(self.cid): loss.item(),
            'client_{}_loss_ce'.format(self.cid): loss_ce.item(),
            'client_{}_Image'.format(self.cid): fl.common.ndarray_to_bytes(image.cpu().numpy()),
            'client_{}_Prediction'.format(self.cid): fl.common.ndarray_to_bytes(outputs_vis.cpu().numpy()),
            'client_{}_GroundTruth'.format(self.cid): fl.common.ndarray_to_bytes(labs.cpu().numpy()),
            'client_{}_loss_uni'.format(self.cid): loss_uni.item(),
            'client_{}_loss_pls'.format(self.cid): loss_pls.item(),
            'client_{}_loss_ema'.format(self.cid): float(loss_ema.item()),
            'client_{}_prompts'.format(self.cid): fl.common.ndarray_to_bytes(prompts[self.cid].detach().cpu().numpy()),
            'client_{}_Prediction2'.format(self.cid): fl.common.ndarray_to_bytes(
                torch.argmax(torch.softmax(outputs_auxiliary, dim=1), dim=1, keepdim=True)[1, ...].repeat(3, 1, 1).cpu().numpy()
                if self.args.img_class in ['odoc', 'polyp']
                else torch.argmax(torch.softmax(outputs_auxiliary, dim=1), dim=1, keepdim=True)[1, ...].cpu().numpy()
            ),
            'client_{}_Pseudo'.format(self.cid): fl.common.ndarray_to_bytes(
                pseudo_label[1, ...].unsqueeze(0).repeat(3, 1, 1).mul(50).cpu().numpy()
                if self.args.img_class in ['odoc', 'polyp']
                else pseudo_label[1, ...].unsqueeze(0).mul(50).cpu().numpy()
            ),
            'client_{}_loss_kd'.format(self.cid): float(loss_kd.item()),
            'client_{}_teacher_student_logit_cosine'.format(self.cid): float(teacher_student_diag['logit_cosine']),
            'client_{}_teacher_student_prediction_disagreement'.format(self.cid): float(teacher_student_diag['prediction_disagreement']),
            'client_{}_teacher_student_cka_down3'.format(self.cid): float(teacher_student_diag['cka_down3']),
            'client_{}_teacher_student_cka_down4'.format(self.cid): float(teacher_student_diag['cka_down4']),
        }
        if student_result['distance_prompt'] is not None:
            metrics_['client_{}_distance_prompt'.format(self.cid)] = student_result['distance_prompt'].item()
            metrics_['client_{}_distance_prompt_distribution'.format(self.cid)] = student_result['distance_prompt_distribution'].item()
            metrics_['client_{}_distance_prompt_uni'.format(self.cid)] = student_result['distance_prompt_uni'].item()
        if proto_filter_enabled:
            prototype_batches = self.sampled_batches if len(self.sampled_batches) > 0 else [sampled_batch for sampled_batch in self.trainloader]
            local_prototypes, local_counts = compute_local_weak_prototypes(self.model, prototype_batches, self.args)
            for class_name in PROTOTYPE_CLASS_NAMES:
                metrics_[f'client_{self.cid}_proto_count_{class_name}'] = int(local_counts[class_name])
                if local_prototypes[class_name] is not None:
                    metrics_[f'client_{self.cid}_proto_vec_{class_name}'] = fl.common.ndarray_to_bytes(local_prototypes[class_name])
            metrics_[f'client_{self.cid}_proto_filter_loss'] = float(proto_filter_loss.item())
            metrics_[f'client_{self.cid}_proto_filter_selected_ratio'] = float(proto_filter_selected_ratio)
            metrics_[f'client_{self.cid}_proto_filter_candidate_count'] = int(proto_filter_candidate_count)
            metrics_[f'client_{self.cid}_proto_filter_selected_count'] = int(proto_filter_selected_count)
            metrics_[f'client_{self.cid}_proto_filter_margin_mean'] = float(proto_filter_margin_mean)
        return loss.item(), metrics_


    def _train(self, config):
        if getattr(self.args, 'student_teacher_v1', 0) == 1 and getattr(self.args, 'student_teacher_teacher_track', 0) == 1:
            return self._train_student_teacher_teachertrack(config)

        self.model.train()

        optimizer = self._get_round_optimizer(self.model.model)

        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes)
        tree_loss = TreeEnergyLoss()
        tree_loss_muti = MScaleRecurveTreeEnergyLoss()
        gatecrf_loss = ModelLossSemsegGatedCRF()
        dice_loss = losses.pDLoss(self.args.num_classes, ignore_index=self.args.num_classes)
        mse_loss = MSELoss()
        l1_loss = L1Loss()
        kl_loss = KLDivLoss()
        info_nce_loss = InfoNCE(negative_mode='paired')

        # writer = SummaryWriter(snapshot_path + '/log')
        log(INFO, '{} iterations per epoch'.format(len(self.trainloader)))

        if self.args.strategy == 'FedProx':
            server_model = copy.deepcopy(self.model)

        uncertainty_list = []
        loss_gatedcrf_kernels_desc = [{"weight": 1, "xy": 6, "rgb": 0.1}]
        loss_gatedcrf_radius = 5
        previous_prompts = None
        previous_prompts_dis = None
        previous_prompts_uni = None
        proto_filter_loss = torch.tensor(0.0).cuda()
        proto_filter_selected_ratio = 0.0
        proto_filter_candidate_count = 0
        proto_filter_selected_count = 0
        proto_filter_margin_mean = 0.0
        adaptive_pl_loss_hard = torch.tensor(0.0).cuda()
        adaptive_pl_loss_soft = torch.tensor(0.0).cuda()
        adaptive_pl_hard_ratio = 1.0
        loss_kd = torch.tensor(0.0).cuda()
        teacher_student_diag = {
            'logit_cosine': 1.0,
            'prediction_disagreement': 0.0,
            'cka_down3': 1.0,
            'cka_down4': 1.0,
        }
        prototype_bank = None
        proto_filter_enabled = (
            getattr(self.args, 'prototype_filter_enabled', 0) == 1
            and self.args.strategy in ['FedUniV2', 'FedUniV2.1']
        )
        proto_filter_warmup = int(getattr(self.args, 'prototype_filter_warmup_rounds', 50))
        proto_filter_topk = float(getattr(self.args, 'prototype_filter_topk_ratio', 0.25))
        if proto_filter_enabled and int(config.get('proto_bank_ready', 0)) == 1:
            prototype_bank = decode_prototype_bank_from_config(
                config,
                device=next(self.model.parameters()).device,
            )
        global_pl_model = None
        if self._adaptive_pl_is_enabled():
            global_pl_model = copy.deepcopy(self.model)
            global_pl_model.eval()
            for param in global_pl_model.parameters():
                param.requires_grad = False
        
        for i_iter in range(config['iters']):
            # genearate sampled batches
            if self.current_iter % len(self.trainloader) == 0:
                print('generating sampled batches......')
                self.sampled_batches.clear()
                for i_batch, sampled_batch in enumerate(self.trainloader):
                    self.sampled_batches.append(sampled_batch)

            idx = self.current_iter % len(self.trainloader)
            sampled_batch = self.sampled_batches[idx]
            # print(self.current_iter, i_iter, idx)

            if self.args.img_class in ['faz', 'prostate']:
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            geometry_bin_batch = None
            if getattr(self.args, 'geometry_guided', 0) == 1 and 'geometry_bin' in sampled_batch:
                geometry_bin_batch = sampled_batch['geometry_bin'].cuda().long()

            # gradient settings
            if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
                local_keys = ['decoder.out_conv.weight', 'decoder.out_conv.bias']

            if self.args.strategy in ['FedRep', 'FedLC', 'FedALALC', 'FedAPLC']:
                if i_iter < self.args.iters - self.args.rep_iters:
                    for name, param in self.model.named_parameters():
                        if name.replace('model.', '') in local_keys:
                            # print(self.args.cid, name, True)
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                else:
                    for name, param in self.model.named_parameters():
                        if name.replace('model.', '') in local_keys:
                            # print(self.args.cid, name, False)
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

            ## training procedures
            with autocast(enabled=self.amp):
                # forward
                out = self.model(volume_batch)
                if self.args.model == 'fcnet':
                    high_feats, outputs = out
                elif self.args.model in ['deeplabv3plus', 'treefcn']:
                    outputs, _, high_feats = out
                elif self.args.model == 'unet_head':
                    outputs, feature, de1, de2, de3, de4, aux_output = out
                    high_feats = aux_output
                elif self.args.model == 'unet_multihead':
                    outputs, feature, de1, de2, de3, de4, aux_output1, aux_output2, aux_output3 = out
                    high_feats = aux_output1
                elif self.args.model == 'unet_lc':
                    outputs, feature, de1, de2, de3, de4, heatmaps, aux_output = out
                elif self.args.model == 'unet_uni':
                    outputs, feature, de1, de2, de3, de4, prompts, fuse_feature = out
                elif self.args.model == 'unet_univ2':
                    outputs, feature, de1, de2, de3, de4, prompts, fuse_feature, outputs_auxiliary = out
                elif self.args.model == 'unet_univ3' or self.args.model == 'unet_univ4':
                    outputs, feature, de1, de2, de3, de4, prompts, fuse_feature, outputs_auxiliary = out
                elif self.args.model == 'unet_univ5':
                    outputs, feature, de1, de2, de3, de4, prompts, fuse_feature, outputs_auxiliary, distribution_prompts, uni_prompts = out
                else:
                    outputs, feature, de1, de2, de3, de4 = out
                raw_encoder_feature = None
                if self.args.model in ['unet_univ2', 'unet_univ3', 'unet_univ4', 'unet_univ5']:
                    raw_encoder_feature = fuse_feature[:, : (fuse_feature.shape[1] - prompts[self.args.cid].shape[1])]

                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce_seg = ce_loss(outputs, label_batch[:].long())
                loss_ce_auxiliary = ce_loss(outputs_auxiliary, label_batch[:].long())
                loss_ce = 0.5 * (loss_ce_seg + loss_ce_auxiliary)
# TreeEnergyLoss
                # unlabeled_RoIs = (sampled_batch['label'] == self.args.num_classes)
                # unlabeled_RoIs = unlabeled_RoIs.cuda()
                # if self.args.img_class == 'faz':
                #     three_channel = volume_batch.repeat(1, 3, 1, 1)
                # elif self.args.img_class == 'odoc' or self.args.img_class =='polyp':
                #     three_channel = volume_batch
                # three_channel = three_channel.cuda()
                # out_tree_loss, heatmaps_tree = tree_loss(outputs, three_channel, high_feats, unlabeled_RoIs, self.args.tree_loss_weight)
                
                loss = loss_ce
                # loss = loss_ce

                # calculate strategy-specific metrics
                if self.args.strategy == 'FedProx' and i_iter > 0:
                    w_diff = torch.tensor(0.).cuda()
                    for w, w_t in zip(server_model.parameters(), self.model.parameters()):
                        w_diff += torch.pow(torch.norm(w - w_t), 2)
                    w_diff = torch.sqrt(w_diff)
                    loss_prox = self.args.mu / 2. * w_diff
                    loss += loss_prox

                if self.args.strategy in ['MetaFed', 'FedAN']:
                    rot_times = random.randrange(0, 4)
                    rotated_volume_batch = torch.rot90(volume_batch, rot_times, [2, 3])
                    T = 8
                    _, _, w, h = volume_batch.shape
                    volume_batch_r = rotated_volume_batch.repeat(2, 1, 1, 1)
                    stride = volume_batch_r.shape[0] // 2
                    preds = torch.zeros([stride * T, self.args.num_classes, w, h]).cuda()
                    for i in range(T // 2):
                        ema_inputs = volume_batch_r + \
                            torch.clamp(torch.randn_like(
                                volume_batch_r) * 0.1, -0.2, 0.2)
                        with torch.no_grad():
                            preds[2 * stride * i:2 * stride *
                                (i + 1)] = self.model(ema_inputs)[0]
                    preds = F.softmax(preds, dim=1)
                    preds = preds.reshape(T, stride, self.args.num_classes, w, h)
                    preds = torch.mean(preds, dim=0)
                    uncertainty = -1.0 * \
                        torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
                    uncertainty_list.append(torch.mean(uncertainty).item())

                if self.args.strategy == 'MetaFed':
                    if self.model.meta_flag and (self.current_iter + 1) > self.args.init_iters:
                        self.model.teacher_model.eval()
                        teacher_out = self.model.teacher_model(volume_batch)
                        teacher_feature = teacher_out[1]
                        loss_meta_feature = torch.tensor(0.0).cuda()
                        for i_feature in range(len(feature)):
                            loss_meta_feature += F.mse_loss(feature[i_feature], teacher_feature[i_feature]).detach()
                        teacher_outputs = teacher_out[0]
                        teacher_outputs_soft = torch.softmax(teacher_outputs, dim=1)
                        # loss_meta = F.cross_entropy(outputs_soft.log(), teacher_outputs_soft).detach()
                        loss_meta = F.kl_div(outputs_soft.log(), teacher_outputs_soft).detach()
                        loss = torch.add(loss, loss_meta, alpha=self.model.lam)
                        loss = torch.add(loss, loss_meta_feature, alpha=self.args.beta)
                        lam = self.model.lam
                        # print(self.current_iter + 1, loss, loss_ce, loss_meta)
                    else:
                        loss_meta = torch.tensor(0.0)
                        lam = 0.0

                if (self.args.strategy in ['FedAP', 'FedAPLC'] and (self.current_iter + 1) == self.args.iters) \
                    or (self.args.strategy == 'MetaFed' and (self.current_iter + 1) == (self.args.common_iters + self.args.iters)):
                    print(self.args.cid, 'get_bn_stats', self.current_iter + 1)
                    bnm, bnv = get_bn_stats(self.args, self.model.model, self.trainloader)

                if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
                    loss_lc = 0
                    for other_client in range(self.args.min_num_clients):
                        if other_client == self.args.cid:
                           continue 
                        with torch.no_grad():
                            _heatmaps = self.model(volume_batch, other_client)[-2]
                            # print(heatmaps[-1].shape, _heatmaps[-1].shape)
                            loss_lc += mse_loss(heatmaps[-1], _heatmaps[-1].detach())
                            # loss_lc += kl_loss(heatmaps[-1], _heatmaps[-1].detach())
                            # loss_lc += mse_loss(heatmaps[-1], _heatmaps[-1])
                    loss_lc = -loss_lc / (self.args.min_num_clients - 1)
                    loss = torch.add(loss, loss_lc, alpha=self.args.alpha)

                if self.args.strategy == 'FedUni':
                    loss_uni = torch.tensor(0.0).cuda()
                    # MSE, KL
                    for other_client in range(self.args.min_num_clients):
                        if other_client == self.args.cid:
                           continue 
                        # print(fuse_feature.shape, prompts.shape)
                        # loss_uni += mse_loss(prompts[:, self.args.cid], prompts[:, other_client].detach())
                        loss_uni += kl_loss(prompts[:, self.args.cid], prompts[:, other_client].detach())
                    loss_uni = -loss_uni / (self.args.min_num_clients - 1)
                    # loss_uni = -torch.log(loss_uni / (self.args.min_num_clients - 1))
                    loss = torch.add(loss, loss_uni, alpha=self.args.alpha)

                if self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
                    loss_uni = torch.tensor(0.0).cuda()
                    out_gatedcrf = gatecrf_loss(
                      outputs_soft,
                      loss_gatedcrf_kernels_desc,
                      loss_gatedcrf_radius,
                      volume_batch,
                      self.args.img_size,
                      self.args.img_size
                  )["loss"]
                    loss = loss + 0.1 * out_gatedcrf
                    # MSE, KL
                    for other_client in range(self.args.min_num_clients):
                        if other_client == self.args.cid:
                           continue 
                        # print(fuse_feature.shape, prompts.shape)
                        # loss_uni += mse_loss(prompts[:, self.args.cid], prompts[:, other_client].detach())
                        loss_uni += kl_loss(prompts[self.args.cid], prompts[other_client].detach())
                    loss_uni = -loss_uni / (self.args.min_num_clients - 1)
                    # loss_uni = -torch.log(loss_uni / (self.args.min_num_clients - 1))
                    loss = torch.add(loss, loss_uni, alpha=self.args.alpha)
                    # dual branches
                    outputs_soft_auxiliary = torch.softmax(outputs_auxiliary, dim=1)
                    pseudo_alpha = np.random.uniform(0, 1)
                    pseudo_label_mix = pseudo_alpha * outputs_soft.detach() + (1 - pseudo_alpha) * outputs_soft_auxiliary.detach()
                    global_pseudo_label_mix = self._build_global_pseudo_label_mix(global_pl_model, volume_batch, pseudo_alpha)
                    pseudo_label = torch.argmax(pseudo_label_mix, dim=1)
                    use_proto_filter = (
                        proto_filter_enabled
                        and prototype_bank is not None
                        and int(config.get('iter_global', 0)) > proto_filter_warmup
                    )
                    if use_proto_filter:
                        raw_encoder_feature = fuse_feature[:, : (fuse_feature.shape[1] - prompts[self.args.cid].shape[1])]
                        selected_low_mask, _, _, margin_mean, candidate_count, selected_count = select_top_margin_mask(
                            raw_encoder_feature=raw_encoder_feature,
                            pseudo_probs=0.5 * (outputs_soft.detach() + outputs_soft_auxiliary.detach()),
                            weak_label_batch=label_batch,
                            prototype_bank=prototype_bank,
                            ignore_index=self.args.num_classes,
                            topk_ratio=proto_filter_topk,
                        )
                        selected_full_mask = F.interpolate(
                            selected_low_mask.unsqueeze(1).float(),
                            size=outputs.shape[-2:],
                            mode='nearest',
                        ).squeeze(1).bool()
                        loss_pls_1 = masked_cross_entropy_loss(
                            outputs,
                            pseudo_label,
                            selected_full_mask,
                            ignore_index=self.args.num_classes,
                        )
                        loss_pls_2 = masked_cross_entropy_loss(
                            outputs_auxiliary,
                            pseudo_label,
                            selected_full_mask,
                            ignore_index=self.args.num_classes,
                        )
                        loss_pls = (loss_pls_1 + loss_pls_2) / 2
                        proto_filter_loss = loss_pls.detach()
                        proto_filter_candidate_count = candidate_count
                        proto_filter_selected_count = selected_count
                        proto_filter_selected_ratio = float(selected_count / max(candidate_count, 1))
                        proto_filter_margin_mean = float(margin_mean.detach().item()) if torch.is_tensor(margin_mean) else float(margin_mean)
                    else:
                        geometry_weight_map = None
                        adaptive_pl_applied = False
                        if getattr(self.args, 'geometry_guided', 0) == 1 and geometry_bin_batch is not None:
                            if self._adaptive_pl_is_enabled():
                                adaptive_pl_result = self._compute_adaptive_pl_loss(
                                    outputs=outputs,
                                    outputs_auxiliary=outputs_auxiliary,
                                    outputs_soft=outputs_soft,
                                    outputs_soft_auxiliary=outputs_soft_auxiliary,
                                    pseudo_label_mix=pseudo_label_mix,
                                    global_pseudo_label_mix=global_pseudo_label_mix,
                                    geometry_bin_batch=geometry_bin_batch,
                                    label_batch=label_batch,
                                    raw_encoder_feature=raw_encoder_feature,
                                )
                                loss_pls = adaptive_pl_result['loss_total']
                                adaptive_pl_loss_hard = adaptive_pl_result['loss_hard'].detach()
                                adaptive_pl_loss_soft = adaptive_pl_result['loss_soft'].detach()
                                adaptive_pl_hard_ratio = adaptive_pl_result['hard_ratio']
                                pseudo_label = adaptive_pl_result['hard_target']
                                adaptive_pl_applied = True
                            else:
                                geometry_weight_map = self._build_effective_geometry_weight_map(
                                    geometry_bin_batch,
                                    label_batch=label_batch,
                                    target_batch=pseudo_label,
                                    apply_support_bonus=True,
                                )
                        if adaptive_pl_applied:
                            pass
                        elif geometry_weight_map is None:
                            loss_pls_1 = dice_loss(outputs_soft, pseudo_label.unsqueeze(1))
                            loss_pls_2 = dice_loss(outputs_soft_auxiliary, pseudo_label.unsqueeze(1))
                            loss_pls = (loss_pls_1 + loss_pls_2) / 2
                        else:
                            loss_pls_1 = weighted_pseudo_dice_loss(outputs_soft, pseudo_label, geometry_weight_map)
                            loss_pls_2 = weighted_pseudo_dice_loss(outputs_soft_auxiliary, pseudo_label, geometry_weight_map)
                            loss_pls = (loss_pls_1 + loss_pls_2) / 2
                        proto_filter_loss = torch.tensor(0.0).cuda()
                        proto_filter_candidate_count = 0
                        proto_filter_selected_count = 0
                        proto_filter_selected_ratio = 0.0
                        proto_filter_margin_mean = 0.0
                    loss = torch.add(loss, loss_pls, alpha=self.args.beta)
                    if i_iter > 0:
                        distance_prompts = l1_loss(previous_prompts.detach(), prompts[self.args.cid].clone().detach())
                        distance_prompts_dis = l1_loss(previous_prompts_dis.detach(), distribution_prompts[self.args.cid].clone().detach())
                        distance_prompts_uni = l1_loss(previous_prompts_uni.detach(), uni_prompts.clone().detach())
                    previous_prompts = prompts[self.args.cid].detach()
                    previous_prompts_dis = distribution_prompts[self.args.cid].detach()
                    previous_prompts_uni = uni_prompts.detach()

                if getattr(self.args, 'student_teacher_v1', 0) == 1:
                    with torch.no_grad():
                        teacher_out = self.model.teacher_model(volume_batch)
                    teacher_outputs = teacher_out[0]
                    teacher_feature = teacher_out[1]
                    kd_temperature = float(getattr(self.args, 'student_teacher_kd_temperature', 2.0))
                    loss_kd, _ = self._compute_student_teacher_kd_loss(
                        outputs,
                        teacher_outputs,
                        kd_temperature,
                    )
                    loss = loss + float(getattr(self.args, 'student_teacher_kd_lambda', 0.2)) * loss_kd
                    teacher_student_diag = compute_teacher_student_diagnostics(
                        outputs.detach(),
                        teacher_outputs.detach(),
                        feature,
                        teacher_feature,
                    )

            # update model parameters
            optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
                

            self.current_iter = self.current_iter + 1
            log(INFO, 'client %d : iteration %d : lr: %f, loss : %f, loss_ce: %f' % (self.cid, self.current_iter, self.current_lr, loss.item(), loss_ce.item()))
            self._maybe_log_adaptive_pl_state()

            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_

        self._save_adaptive_pl_state(tag='latest')

        # pack general metrics
        image = volume_batch[1, :, :, :]
        image = (image - image.min()) / (image.max() - image.min())
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        outputs = outputs[1, ...] * 50
        labs = label_batch[1, ...].unsqueeze(0) * 50
        if self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
            outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)

        metrics_ = {
            'client_{}_lr'.format(self.cid): self.current_lr,
            'client_{}_total_loss'.format(self.cid): loss.item(),
            'client_{}_loss_ce'.format(self.cid): loss_ce.item(),
            'client_{}_Image'.format(self.cid): fl.common.ndarray_to_bytes(image.cpu().numpy()),
            'client_{}_Prediction'.format(self.cid): fl.common.ndarray_to_bytes(outputs.cpu().numpy()),
            'client_{}_GroundTruth'.format(self.cid):fl.common.ndarray_to_bytes(labs.cpu().numpy()),
        }
        if proto_filter_enabled:
            prototype_batches = self.sampled_batches if len(self.sampled_batches) > 0 else [sampled_batch for sampled_batch in self.trainloader]
            local_prototypes, local_counts = compute_local_weak_prototypes(self.model, prototype_batches, self.args)
            for class_name in PROTOTYPE_CLASS_NAMES:
                metrics_[f'client_{self.cid}_proto_count_{class_name}'] = int(local_counts[class_name])
                if local_prototypes[class_name] is not None:
                    metrics_[f'client_{self.cid}_proto_vec_{class_name}'] = fl.common.ndarray_to_bytes(local_prototypes[class_name])
            metrics_[f'client_{self.cid}_proto_filter_loss'] = float(proto_filter_loss.item())
            metrics_[f'client_{self.cid}_proto_filter_selected_ratio'] = float(proto_filter_selected_ratio)
            metrics_[f'client_{self.cid}_proto_filter_candidate_count'] = int(proto_filter_candidate_count)
            metrics_[f'client_{self.cid}_proto_filter_selected_count'] = int(proto_filter_selected_count)
            metrics_[f'client_{self.cid}_proto_filter_margin_mean'] = float(proto_filter_margin_mean)

        # pack strategy-specific metrics
        if self.args.strategy == 'FedProx':
            metrics_['client_{}_loss_prox'.format(self.cid)] = loss_prox.item()
        if self.args.strategy == 'MetaFed':
            metrics_['client_{}_loss_meta'.format(self.cid)] = loss_meta.item()
            metrics_['client_{}_loss_meta_feature'.format(self.cid)] = loss_meta_feature.item()
            metrics_['client_{}_lam'.format(self.cid)] = lam

        if self.args.strategy in ['MetaFed', 'FedAN']:
            overall_uncertainty = np.mean(uncertainty_list)
            metrics_['client_{}_uncertainty'.format(self.cid)] = overall_uncertainty

        if (self.args.strategy in ['FedAP', 'FedAPLC'] and self.current_iter == self.args.iters) \
            or (self.args.strategy == 'MetaFed' and self.current_iter == (self.args.common_iters + self.args.iters)):
            metrics_['client_{}_bnl'.format(self.cid)] = len(bnm)
            print(self.cid, bnm[0].shape, bnv[0].shape, len(bnm))
            for i_bn in range(len(bnm)):
                metrics_['client_{}_bnm_{}'.format(self.cid, i_bn)] = fl.common.ndarray_to_bytes(bnm[i_bn])
                metrics_['client_{}_bnv_{}'.format(self.cid, i_bn)] =  fl.common.ndarray_to_bytes(bnv[i_bn])

        if self.args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
            metrics_['client_{}_loss_lc'.format(self.cid)] = loss_lc.item()

        if self.args.strategy == 'FedUni':
            metrics_['client_{}_loss_uni'.format(self.cid)] = loss_uni.item()

        if self.args.strategy in ['FedUniV2', 'FedUniV2.1']:
            metrics_['client_{}_loss_uni'.format(self.cid)] = loss_uni.item()
            metrics_['client_{}_loss_pls'.format(self.cid)] = loss_pls.item()
            if self._dg_is_enabled():
                metrics_['client_{}_dg_dataset_homogeneity'.format(self.cid)] = float(self.dg_dataset_homogeneity)
                metrics_['client_{}_dg_dataset_separability'.format(self.cid)] = float(self.dg_dataset_separability)
                metrics_['client_{}_dg_dataset_propagation'.format(self.cid)] = float(self.dg_dataset_propagation)
                metrics_['client_{}_dg_dataset_strength'.format(self.cid)] = float(self.dg_dataset_strength)
                metrics_['client_{}_dg_dataset_num_client_audits'.format(self.cid)] = float(self.dg_dataset_num_client_audits)
                metrics_['client_{}_dg_client_base_strength'.format(self.cid)] = float(self.dg_client_base_strength)
                metrics_['client_{}_dg_client_strength'.format(self.cid)] = float(self.dg_client_strength)
                metrics_['client_{}_dg_strength'.format(self.cid)] = float(self.dg_strength)
                metrics_['client_{}_dg_fg_coverage'.format(self.cid)] = float(self.dg_fg_coverage)
                metrics_['client_{}_dg_dilated_fg_coverage'.format(self.cid)] = float(self.dg_dilated_fg_coverage)
                metrics_['client_{}_dg_s_c'.format(self.cid)] = float(self.dg_coverage_score)
                metrics_['client_{}_dg_s_d'.format(self.cid)] = float(self.dg_dilated_score)
            if self._trustgeo_is_enabled():
                metrics_['client_{}_trustgeo_strength'.format(self.cid)] = float(self.trustgeo_strength)
                metrics_['client_{}_trustgeo_fg_coverage'.format(self.cid)] = float(self.trustgeo_fg_coverage)
                metrics_['client_{}_trustgeo_dilated_fg_coverage'.format(self.cid)] = float(self.trustgeo_dilated_fg_coverage)
                metrics_['client_{}_trustgeo_fg_density'.format(self.cid)] = float(self.trustgeo_fg_density)
                metrics_['client_{}_trustgeo_support_coverage'.format(self.cid)] = float(self.trustgeo_support_coverage)
                metrics_['client_{}_trustgeo_dilated_support_coverage'.format(self.cid)] = float(self.trustgeo_dilated_support_coverage)
                metrics_['client_{}_trustgeo_support_density'.format(self.cid)] = float(self.trustgeo_support_density)
                metrics_['client_{}_trustgeo_s_c'.format(self.cid)] = float(self.trustgeo_coverage_score)
                metrics_['client_{}_trustgeo_s_d'.format(self.cid)] = float(self.trustgeo_dilated_score)
                metrics_['client_{}_trustgeo_support_score'.format(self.cid)] = float(self.trustgeo_support_score)
                metrics_['client_{}_trustgeo_support_mod'.format(self.cid)] = float(self.trustgeo_support_mod)
                metrics_['client_{}_trustgeo_base_strength'.format(self.cid)] = float(self.trustgeo_base_strength)
            if self._support_bonus_is_enabled():
                metrics_['client_{}_support_bonus_strength'.format(self.cid)] = float(self.support_bonus_strength)
                metrics_['client_{}_support_bonus_fg_coverage'.format(self.cid)] = float(self.support_bonus_fg_coverage)
                metrics_['client_{}_support_bonus_dilated_fg_coverage'.format(self.cid)] = float(self.support_bonus_dilated_fg_coverage)
                metrics_['client_{}_support_bonus_density'.format(self.cid)] = float(self.support_bonus_density)
            if self._adaptive_pl_is_enabled():
                w5_soft_only_enabled = bool(getattr(self.args, 'risk_calibration_w5_soft_only_enabled', 0))
                if bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
                    metrics_['client_{}_adaptive_pl_loss_hard_backbone'.format(self.cid)] = float(adaptive_pl_loss_hard.item())
                    metrics_['client_{}_adaptive_pl_loss_soft_backbone'.format(self.cid)] = float(adaptive_pl_loss_soft.item())
                    metrics_['client_{}_adaptive_pl_loss_total_backbone'.format(self.cid)] = float(self.adaptive_pl_last_loss_total_backbone)
                    metrics_['client_{}_adaptive_pl_loss_aux_corr'.format(self.cid)] = float(self.adaptive_pl_last_loss_aux)
                    metrics_['client_{}_adaptive_pl_loss_total_v1'.format(self.cid)] = float(self.adaptive_pl_last_loss_total)
                    metrics_['client_{}_adaptive_pl_boundary_loss_oc_backbone'.format(self.cid)] = float(self.adaptive_pl_last_boundary_loss_oc)
                else:
                    metrics_['client_{}_adaptive_pl_loss_hard'.format(self.cid)] = float(adaptive_pl_loss_hard.item())
                    metrics_['client_{}_adaptive_pl_loss_soft'.format(self.cid)] = float(adaptive_pl_loss_soft.item())
                    metrics_['client_{}_adaptive_pl_loss_total'.format(self.cid)] = float(self.adaptive_pl_last_loss_total)
                    metrics_['client_{}_adaptive_pl_loss_risk'.format(self.cid)] = float(self.adaptive_pl_last_loss_risk)
                metrics_['client_{}_adaptive_pl_hard_ratio'.format(self.cid)] = float(adaptive_pl_hard_ratio)
                metrics_['client_{}_adaptive_pl_lg_class_agree_ratio'.format(self.cid)] = float(self.adaptive_pl_last_lg_class_agree_ratio)
                metrics_['client_{}_adaptive_pl_lg_class_agree_ema'.format(self.cid)] = float(self.adaptive_pl_lg_agreement_ema)
                metrics_['client_{}_adaptive_pl_mean_prob_gap'.format(self.cid)] = float(self.adaptive_pl_last_mean_prob_gap)
                metrics_['client_{}_adaptive_pl_mean_conf_gap'.format(self.cid)] = float(self.adaptive_pl_last_mean_conf_gap)
                metrics_['client_{}_adaptive_pl_both_uncertain_ratio'.format(self.cid)] = float(self.adaptive_pl_last_both_uncertain_ratio)
                metrics_['client_{}_adaptive_pl_both_uncertain_ema'.format(self.cid)] = float(self.adaptive_pl_both_uncertain_ema)
                if not bool(getattr(self.args, 'risk_calibration_v1_enabled', 0)):
                    metrics_['client_{}_adaptive_pl_boundary_loss_oc'.format(self.cid)] = float(self.adaptive_pl_last_boundary_loss_oc)
                metrics_['client_{}_adaptive_pl_ring_valid_oc_ratio'.format(self.cid)] = float(self.adaptive_pl_last_ring_valid_oc_ratio)
                metrics_['client_{}_adaptive_pl_prior_gap'.format(self.cid)] = float(self.adaptive_pl_last_prior_gap)
                metrics_['client_{}_adaptive_pl_client_risk'.format(self.cid)] = float(self.adaptive_pl_last_client_risk)
                metrics_['client_{}_adaptive_pl_stage_progress'.format(self.cid)] = float(self.adaptive_pl_last_stage_progress)
                if (not w5_soft_only_enabled) and (not bool(getattr(self.args, 'risk_calibration_v1_enabled', 0))):
                    metrics_['client_{}_adaptive_pl_release_score'.format(self.cid)] = float(self.adaptive_pl_last_release_score)
                    metrics_['client_{}_adaptive_pl_preserve_score'.format(self.cid)] = float(self.adaptive_pl_last_preserve_score)
                    metrics_['client_{}_adaptive_pl_correction_score'.format(self.cid)] = float(self.adaptive_pl_last_correction_score)
                metrics_['client_{}_adaptive_pl_effective_correction'.format(self.cid)] = float(self.adaptive_pl_last_effective_correction)
                metrics_['client_{}_adaptive_pl_hard_control'.format(self.cid)] = float(self.adaptive_pl_last_hard_control)
                metrics_['client_{}_adaptive_pl_calibration_control'.format(self.cid)] = float(self.adaptive_pl_last_calibration_control)
                metrics_['client_{}_adaptive_pl_hard_ratio_ema'.format(self.cid)] = float(self.adaptive_pl_hard_ratio_ema)
                metrics_['client_{}_adaptive_pl_w6_tail_active'.format(self.cid)] = float(self.adaptive_pl_last_w6_tail_active)
                metrics_['client_{}_adaptive_pl_w6_tail_strength'.format(self.cid)] = float(self.adaptive_pl_last_w6_tail_strength)
                metrics_['client_{}_adaptive_pl_w6_tail_votes'.format(self.cid)] = float(self.adaptive_pl_last_w6_tail_votes)
                for bin_idx, tau_val in enumerate(self.adaptive_pl_tau):
                    metrics_['client_{}_adaptive_pl_tau_bin_{}'.format(self.cid, bin_idx)] = float(tau_val)
                    metrics_['client_{}_adaptive_pl_bin_count_{}'.format(self.cid, bin_idx)] = int(self.adaptive_pl_last_bin_counts[bin_idx])
                    if not np.isnan(self.adaptive_pl_last_observed_accept[bin_idx]):
                        metrics_['client_{}_adaptive_pl_accept_bin_{}'.format(self.cid, bin_idx)] = float(self.adaptive_pl_last_observed_accept[bin_idx])
                    if not np.isnan(self.adaptive_pl_last_score_mean[bin_idx]):
                        metrics_['client_{}_adaptive_pl_score_bin_{}'.format(self.cid, bin_idx)] = float(self.adaptive_pl_last_score_mean[bin_idx])
            metrics_['client_{}_prompts'.format(self.cid)] = fl.common.ndarray_to_bytes(prompts[self.cid].detach().cpu().numpy())
            if i_iter > 0:
                metrics_['client_{}_distance_prompt'.format(self.cid)] = distance_prompts.item()
                metrics_['client_{}_distance_prompt_distribution'.format(self.cid)] = distance_prompts_dis.item()
                metrics_['client_{}_distance_prompt_uni'.format(self.cid)] = distance_prompts_uni.item()
            outputs_auxiliary = torch.argmax(torch.softmax(outputs_auxiliary, dim=1), dim=1, keepdim=True)
            outputs_auxiliary = outputs_auxiliary[1, ...] * 50
            pseudo_labs = pseudo_label[1, ...].unsqueeze(0) * 50
            if self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                outputs_auxiliary, pseudo_labs = outputs_auxiliary.repeat(3, 1, 1), pseudo_labs.repeat(3, 1, 1)
            metrics_['client_{}_Prediction2'.format(self.cid)] = fl.common.ndarray_to_bytes(outputs_auxiliary.cpu().numpy())
            metrics_['client_{}_Pseudo'.format(self.cid)] = fl.common.ndarray_to_bytes(pseudo_labs.cpu().numpy())

        if getattr(self.args, 'student_teacher_v1', 0) == 1:
            metrics_['client_{}_loss_kd'.format(self.cid)] = float(loss_kd.item())
            metrics_['client_{}_teacher_student_logit_cosine'.format(self.cid)] = float(teacher_student_diag['logit_cosine'])
            metrics_['client_{}_teacher_student_prediction_disagreement'.format(self.cid)] = float(teacher_student_diag['prediction_disagreement'])
            metrics_['client_{}_teacher_student_cka_down3'.format(self.cid)] = float(teacher_student_diag['cka_down3'])
            metrics_['client_{}_teacher_student_cka_down4'.format(self.cid)] = float(teacher_student_diag['cka_down4'])

        return loss.item(), metrics_


from flower_common import PretrainDataset
from torchvision.utils import make_grid
from tqdm import tqdm
def pretrain_model(args, writer, worker_init_fn):
    db_train = PretrainDataset(args.root_path, args.sup_type_list, transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    model.cuda()

    if args.amp:
        scaler = GradScaler()
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=args.num_classes)
    dice_loss = losses.pDLoss(args.num_classes,ignore_index=args.num_classes)
    iter_num = 0
    max_epoch = args.pretrain_iters // len(trainloader) + 1
    iterator = tqdm(range(max_epoch), ncols=70)
    model.train()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            if args.img_class in ['faz', 'prostate']:
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif args.img_class == 'odoc' or args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
    
            with autocast(enabled=(args.amp == 1)):
                outputs = model(volume_batch)[0]
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss = loss_ce

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            iter_num = iter_num + 1
            log(INFO, 'pretrain : iteration %d : lr: %f, loss : %f, loss_ce: %f' % (iter_num, args.base_lr, loss.item(), loss_ce.item()))


            if iter_num % args.iters == 0:
                image = volume_batch[1, :, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                outputs = outputs[1, ...] * 50
                labs = label_batch[1, ...].unsqueeze(0) * 50
                if args.img_class == 'odoc' or args.img_class == 'polyp':
                    outputs, labs = outputs.repeat(3, 1, 1), labs.repeat(3, 1, 1)

                image_list = np.array([image.cpu().numpy(), outputs.cpu().numpy(), labs.cpu().numpy()])
                writer.add_image('pretrain/grid_image', make_grid(torch.tensor(image_list)), iter_num)

            if iter_num >= args.pretrain_iters:
                save_mode_path = os.path.join(args.snapshot_path, 'fedap_pretrain.pth')
                torch.save(model.state_dict(), save_mode_path)
                log(INFO, 'save model to {}'.format(save_mode_path))
                iterator.close()
                break

    return save_mode_path, model.state_dict()


def main():
    global writer
    parser = argparse.ArgumentParser()
    ## flower related arguments
    parser.add_argument('--server_address', type=str,
                        default='[::]:8080', help='gRPC server address (default: [::]:8080)')
    parser.add_argument('--gpu', type=int,
                        required=True, help='GPU index')
    parser.add_argument('--role', type=str,
                        required=True, help='Role')
    # server
    parser.add_argument('--iters', type=int,
                        default=20, help='Number of iters (default: 20)')
    parser.add_argument('--eval_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--tsne_iters', type=int,
                        default=200, help='Number of iters (default: 200)')
    parser.add_argument('--rep_iters', type=int,
                        default=12, help='Number of iters (default: 12)')
    parser.add_argument('--sample_fraction', type=float,
                        default=1.0, help='Fraction of available clients used for fit/evaluate (default: 1.0)')
    parser.add_argument('--min_num_clients', type=int,
                        default=2, help='Minimum number of available clients required (default: 2)')
    parser.add_argument('--strategy', type=str,
                        default='FedAvg', help='Federated learning algorithm (default: FedAvg)')
    parser.add_argument('--mu', type=float, default=1e-3,
                        help='The hyper parameter for FedProx')
    parser.add_argument('--lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN')
    parser.add_argument('--sort_lam', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--sort_beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The hyper parameter for MetaFed/FedAN/FedUniV2/FedUniV2.1')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The hyper parameter for FedLC/FedALALC/FedAPLC/FedUni/FedUniV2/FedUniV2.1')
    parser.add_argument('--ala_threshold', type=float, default=0.1,
                        help='Convergence threshold for ALA/FedUniV2 personalization')
    parser.add_argument('--ala_num_pre_loss', type=int, default=10,
                        help='Window size used to judge ALA/FedUniV2 personalization convergence')
    parser.add_argument('--ala_max_init_epochs', type=int, default=10,
                        help='Maximum epochs allowed during the initial ALA/FedUniV2 personalization phase')
    parser.add_argument('--init_iters', type=int,
                        default=100, help='Number of iters (default: 100)')
    parser.add_argument('--common_iters', type=int,
                        default=400, help='Number of iters (default: 400)')
    parser.add_argument('--pretrain_iters', type=int,
                        default=150, help='Number of iters (default: 150)')
    parser.add_argument('--pretrain', type=int,  default=0,
                        help='pretrain')
    parser.add_argument('--img_size', type=int,  default=384,
                        help='image size')
    parser.add_argument('--model_momentum', type=float, default=0.5,
                        help='The hyper parameter for FedAP/FedAPLC')
    parser.add_argument('--prompt', type=str, default='universal',
                        help='Prompt type for FedUniV2/FedUniV2.1')
    parser.add_argument('--attention', type=str, default='dual',
                        help='Attention type for FedUniV2/FedUniV2.1')
    parser.add_argument('--dual_init', type=str, default='random',
                        help='Dual branch initiallization for FedUniV2/FedUniV2.1')
    parser.add_argument('--label_prompt', type=int, default=0,
                        help='Whether use label prompt for FedUniV2/FedUniV2.1')
    # client
    parser.add_argument('--cid', type=int, default=0, help='Client CID (no default)')

    ## WSL4MIS related arguments
    parser.add_argument('--root_path', type=str,
                    default='../data/FAZ_h5', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='faz_pCE', help='experiment_name')
    parser.add_argument('--client', type=str,
                        default='client1', help='cross validation')
    parser.add_argument('--sup_type', type=str,
                        default='mask', help='supervision label type(scr ; label ; scr_n ; keypoint ; block)')
    parser.add_argument('--sup_type_list', nargs='+', type=str,
                        help='supervision label type')
    parser.add_argument('--model', type=str,
                        default='unet', help='model_name')
    parser.add_argument('--num_classes', type=int,  default=2,
                        help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='batch_size per gpu')
    parser.add_argument('--in_chns', type=int, default=1,
                        help='image channel')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--amp', type=int,  default=0,
                        help='whether use amp training')
    parser.add_argument('--base_lr', type=float,  default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list,  default=[256, 256],
                        help='patch size of network input')
    parser.add_argument('--img_class', type=str,
                        default='faz', help='the img class(odoc or faz)')
    parser.add_argument('--seed', type=int,  default=2022, help='random seed')
    parser.add_argument('--geometry_guided', type=int, default=0,
                        help='Enable geometry-guided pseudo-label weighting')
    parser.add_argument('--geometry_num_bins', type=int, default=4,
                        help='Number of geometry bins')
    parser.add_argument('--geometry_near_radius', type=float, default=8.0,
                        help='Near-radius threshold for rule-based geometry bins')
    parser.add_argument('--geometry_mid_radius', type=float, default=24.0,
                        help='Mid-radius threshold for rule-based geometry bins')
    parser.add_argument('--geometry_pseudo_weights', type=str, default='1.0,0.8,0.5,0.2',
                        help='Comma-separated pseudo-label weights for geometry bins 0..K-1')
    parser.add_argument('--trustgeo_enabled', type=int, default=0,
                        help='Enable static client-level geometry qualification to interpolate between v1B and a fixed geometry prior.')
    parser.add_argument('--trustgeo_dilation_radius', type=int, default=5,
                        help='Pixel radius used when computing trustgeo weak-label structure statistics.')
    parser.add_argument('--trustgeo_density_kernel', type=int, default=11,
                        help='Odd kernel size used to estimate local weak-label densities for trustgeo diagnostics.')
    parser.add_argument('--trustgeo_c_low', type=float, default=0.0015,
                        help='Lower raw foreground-coverage bound for piecewise-linear trustgeo qualification.')
    parser.add_argument('--trustgeo_c_high', type=float, default=0.0400,
                        help='Upper raw foreground-coverage bound for piecewise-linear trustgeo qualification.')
    parser.add_argument('--trustgeo_d_low', type=float, default=0.0060,
                        help='Lower dilated foreground-coverage bound for piecewise-linear trustgeo qualification.')
    parser.add_argument('--trustgeo_d_high', type=float, default=0.0900,
                        help='Upper dilated foreground-coverage bound for piecewise-linear trustgeo qualification.')
    parser.add_argument('--trustgeo_support_low', type=float, default=0.0100,
                        help='Lower labeled-support-coverage bound used by the weak trustgeo modulation term.')
    parser.add_argument('--trustgeo_support_high', type=float, default=0.1500,
                        help='Upper labeled-support-coverage bound used by the weak trustgeo modulation term.')
    parser.add_argument('--trustgeo_support_mod_min', type=float, default=0.85,
                        help='Minimum multiplicative modulation applied from labeled support to the foreground-driven trustgeo score.')
    parser.add_argument('--trustgeo_w_c', type=float, default=0.70,
                        help='Weight assigned to raw foreground coverage in trustgeo qualification.')
    parser.add_argument('--trustgeo_w_d', type=float, default=0.30,
                        help='Weight assigned to dilated foreground coverage in trustgeo qualification.')
    parser.add_argument('--trustgeo_prior_alpha', type=float, default=0.8,
                        help='Linear decay strength for the fixed geometry prior used by trustgeo.')
    parser.add_argument('--trustgeo_prior_floor', type=float, default=0.2,
                        help='Minimum bin weight floor for the fixed geometry prior used by trustgeo.')
    parser.add_argument('--dg_enabled', type=int, default=0,
                        help='Enable dataset-level geometry audit that gates client-level geometry qualification.')
    parser.add_argument('--dg_max_audit_samples', type=int, default=64,
                        help='Maximum number of train cases used for the dataset-level geometry audit.')
    parser.add_argument('--dg_max_prop_samples', type=int, default=16,
                        help='Maximum number of train cases used for the weak propagation stability probe.')
    parser.add_argument('--dg_local_std_kernel', type=int, default=15,
                        help='Odd kernel size used for local texture-homogeneity estimation in dataset audit.')
    parser.add_argument('--dg_far_radius', type=float, default=24.0,
                        help='Distance threshold that defines the non-seed region for homogeneity audit.')
    parser.add_argument('--dg_sep_near_radius', type=float, default=8.0,
                        help='Near-ring radius used by the appearance separability audit.')
    parser.add_argument('--dg_sep_mid_radius', type=float, default=24.0,
                        help='Mid-ring radius used by the appearance separability audit.')
    parser.add_argument('--dg_sep_far_radius', type=float, default=48.0,
                        help='Far-ring radius used by the appearance separability audit.')
    parser.add_argument('--dg_min_fg_pixels', type=int, default=8,
                        help='Minimum weak-foreground pixels required to score a case in dataset audit.')
    parser.add_argument('--dg_min_ring_pixels', type=int, default=64,
                        help='Minimum ring pixels required for a valid appearance-separability score.')
    parser.add_argument('--dg_min_non_seed_pixels', type=int, default=128,
                        help='Minimum non-seed pixels required for a valid homogeneity score.')
    parser.add_argument('--dg_hom_tau', type=float, default=0.75,
                        help='Scale parameter used to convert non-seed texture roughness into a homogeneity score.')
    parser.add_argument('--dg_sep_tau', type=float, default=0.25,
                        help='Scale parameter used in the monotonic appearance-separability logistic score.')
    parser.add_argument('--dg_prop_seed_keep_ratio1', type=float, default=0.85,
                        help='First seed-retention ratio used in the weak propagation stability probe.')
    parser.add_argument('--dg_prop_seed_keep_ratio2', type=float, default=0.70,
                        help='Second seed-retention ratio used in the weak propagation stability probe.')
    parser.add_argument('--dg_prop_min_seed_pixels', type=int, default=4,
                        help='Minimum pixels per foreground seed class before seed perturbation is applied.')
    parser.add_argument('--dg_prop_min_keep_pixels', type=int, default=2,
                        help='Minimum pixels retained for each foreground seed class in seed perturbation.')
    parser.add_argument('--dg_prop_min_region_pixels', type=int, default=16,
                        help='Minimum propagated foreground pixels required for a non-zero propagation stability score.')
    parser.add_argument('--dg_prop_dist_std_scale', type=float, default=1.0,
                        help='Std multiplier used by the support-conditioned propagation probe when converting support-feature spread into a growth threshold.')
    parser.add_argument('--dg_dataset_w_sep', type=float, default=0.60,
                        help='Weight of appearance separability in the dataset-level geometry gate.')
    parser.add_argument('--dg_dataset_w_hom', type=float, default=0.30,
                        help='Weight of non-seed homogeneity in the dataset-level geometry gate.')
    parser.add_argument('--dg_dataset_w_prop', type=float, default=0.10,
                        help='Weight of propagation stability in the dataset-level geometry gate.')
    parser.add_argument('--dg_audit_wait_timeout_sec', type=float, default=300.0,
                        help='Maximum time each client waits for all local DG audit summaries before falling back to partial aggregation.')
    parser.add_argument('--dg_audit_wait_poll_sec', type=float, default=2.0,
                        help='Polling interval used while waiting for peer DG audit summaries.')
    parser.add_argument('--support_bonus_enabled', type=int, default=0,
                        help='Enable conservative local support bonus on top of the v1B path.')
    parser.add_argument('--support_bonus_tau_c', type=float, default=0.05,
                        help='Saturation constant for raw weak-foreground coverage in support bonus scoring.')
    parser.add_argument('--support_bonus_tau_d', type=float, default=0.20,
                        help='Saturation constant for dilated weak-foreground coverage in support bonus scoring.')
    parser.add_argument('--support_bonus_dilation_radius', type=int, default=5,
                        help='Pixel radius used when computing dilated weak-foreground support coverage.')
    parser.add_argument('--support_bonus_density_kernel', type=int, default=11,
                        help='Odd kernel size used to estimate local weak-label density for support bonus scoring.')
    parser.add_argument('--support_bonus_support_radius', type=int, default=5,
                        help='Pixel radius used to define the local support region for support bonus weighting.')
    parser.add_argument('--support_bonus_lambda', type=float, default=0.10,
                        help='Maximum extra weight applied inside the local support region, scaled by the client trust score.')
    parser.add_argument('--adaptive_pl_enabled', type=int, default=0,
                        help='Enable adaptive tau-gated pseudo-label selection on top of geometry-guided weighting.')
    parser.add_argument('--adaptive_pl_tau_init', type=float, default=0.55,
                        help='Initial per-bin pseudo-label threshold.')
    parser.add_argument('--adaptive_pl_tau_update', type=int, default=1,
                        help='Whether per-bin tau is updated online.')
    parser.add_argument('--adaptive_pl_resume_state', type=int, default=0,
                        help='Resume adaptive tau/EMA sidecar state from the current exp directory.')
    parser.add_argument('--adaptive_pl_target_accept', type=float, default=0.35,
                        help='Target accepted-pixel ratio used by tau updates.')
    parser.add_argument('--adaptive_pl_warmup_iters', type=int, default=800,
                        help='Warmup iterations before hard/soft pseudo-label gating becomes active.')
    parser.add_argument('--adaptive_pl_soft_lambda', type=float, default=0.2,
                        help='Weight of reverse-KL soft pseudo-label supervision.')
    parser.add_argument('--adaptive_pl_min_pixels_per_bin', type=int, default=64,
                        help='Minimum number of pixels required before a bin can update tau.')
    parser.add_argument('--adaptive_pl_log_interval', type=int, default=50,
                        help='TensorBoard logging interval for adaptive pseudo-label statistics.')
    parser.add_argument('--adaptive_pl_gamma_prob', type=float, default=4.0,
                        help='Exponential decay factor applied to the local-global probability gap.')
    parser.add_argument('--adaptive_pl_gamma_conf', type=float, default=3.0,
                        help='Exponential decay factor applied to the local-global confidence gap.')
    parser.add_argument('--adaptive_pl_blend_kappa', type=float, default=2.0,
                        help='Blend sensitivity for corrected local-global pseudo-probabilities.')
    parser.add_argument('--adaptive_pl_global_min_conf', type=float, default=0.6,
                        help='Minimum global confidence required before low-score pixels receive soft PL supervision.')
    parser.add_argument('--adaptive_pl_boundary_lambda', type=float, default=0.0,
                        help='Weight of OC-only boundary supervision added on top of adaptive PL.')
    parser.add_argument('--adaptive_pl_boundary_kernel_size', type=int, default=3,
                        help='Kernel size used to build the OC boundary ring.')
    parser.add_argument('--adaptive_pl_boundary_warmup_iters', type=int, default=800,
                        help='Warmup iterations before OC boundary supervision becomes active.')
    parser.add_argument('--seed_support_enabled', type=int, default=0,
                        help='Enable local seed support and feature-affinity propagation on top of adaptive PL.')
    parser.add_argument('--seed_support_warmup_iters', type=int, default=800,
                        help='Warmup iterations before seed-support-guided adaptive PL becomes active.')
    parser.add_argument('--seed_support_reliable_score_min', type=float, default=0.7,
                        help='Minimum adaptive PL score required before unlabeled pseudo-seeds can enter the local seed bank.')
    parser.add_argument('--seed_support_kernel_size', type=int, default=5,
                        help='Neighborhood kernel size used by feature-affinity seed propagation.')
    parser.add_argument('--seed_support_affinity_temp', type=float, default=8.0,
                        help='Softmax temperature for local feature-affinity propagation.')
    parser.add_argument('--seed_support_blend_alpha', type=float, default=0.5,
                        help='Blend ratio between corrected local-global PL and propagated local seed support.')
    parser.add_argument('--seed_support_hard_tau', type=float, default=0.35,
                        help='Reserved compatibility threshold from the earlier W2_mid design; unused in the current W2_lite gating.')
    parser.add_argument('--seed_support_soft_tau', type=float, default=0.20,
                        help='Minimum seed-support strength required before a pixel can receive soft PL.')
    parser.add_argument('--risk_calibration_enabled', type=int, default=0,
                        help='Enable client-risk-aware prior correction, stricter gating, and global-anchor soft supervision.')
    parser.add_argument('--risk_calibration_tau_lambda', type=float, default=0.12,
                        help='Additional tau margin applied to higher-risk clients.')
    parser.add_argument('--risk_calibration_conf_lambda', type=float, default=0.08,
                        help='Additional minimum global-confidence margin for both-uncertain filtering on higher-risk clients.')
    parser.add_argument('--risk_calibration_prior_power', type=float, default=1.0,
                        help='Exponent applied to class-prior correction strength under client risk.')
    parser.add_argument('--risk_calibration_prior_clip_min', type=float, default=0.5,
                        help='Minimum class-prior correction multiplier.')
    parser.add_argument('--risk_calibration_prior_clip_max', type=float, default=1.5,
                        help='Maximum class-prior correction multiplier.')
    parser.add_argument('--risk_calibration_hard_dampen', type=float, default=0.5,
                        help='How strongly hard pseudo-label weights are damped for higher-risk clients.')
    parser.add_argument('--risk_calibration_soft_boost', type=float, default=0.5,
                        help='How strongly soft pseudo-label weights are boosted for higher-risk clients.')
    parser.add_argument('--risk_calibration_global_soft_lambda', type=float, default=0.15,
                        help='Weight of conservative global-anchor soft supervision on both-uncertain pixels.')
    parser.add_argument('--risk_calibration_stateful_release_enabled', type=int, default=0,
                        help='Enable W4 staged release: early unified correction, late state-based release/correction routing.')
    parser.add_argument('--risk_calibration_release_start_iter', type=int, default=800,
                        help='Global iteration where W4 release routing starts annealing away from W3-style unified correction.')
    parser.add_argument('--risk_calibration_release_full_iter', type=int, default=1800,
                        help='Global iteration where W4 release annealing reaches full strength.')
    parser.add_argument('--risk_calibration_regime_ema_momentum', type=float, default=0.9,
                        help='EMA momentum used to smooth W4 release/correction regime scores.')
    parser.add_argument('--risk_calibration_release_risk_threshold', type=float, default=0.35,
                        help='Clients below this risk level become eligible for the W4 release regime.')
    parser.add_argument('--risk_calibration_release_agreement_threshold', type=float, default=0.90,
                        help='Clients above this local-global agreement become eligible for the W4 release regime.')
    parser.add_argument('--risk_calibration_release_prior_gap_threshold', type=float, default=0.015,
                        help='Clients below this prior-gap EMA become eligible for the W4 release regime.')
    parser.add_argument('--risk_calibration_release_uncertain_threshold', type=float, default=0.02,
                        help='Clients below this both-uncertain EMA become eligible for the W4 release regime.')
    parser.add_argument('--risk_calibration_preserve_risk_threshold', type=float, default=0.65,
                        help='Clients below this risk level can leave strict correction and enter preserve.')
    parser.add_argument('--risk_calibration_preserve_agreement_threshold', type=float, default=0.92,
                        help='Clients above this agreement can enter the W4 preserve regime.')
    parser.add_argument('--risk_calibration_preserve_prior_gap_threshold', type=float, default=0.025,
                        help='Clients below this prior-gap EMA can enter the W4 preserve regime.')
    parser.add_argument('--risk_calibration_preserve_uncertain_threshold', type=float, default=0.04,
                        help='Clients below this both-uncertain EMA can enter the W4 preserve regime.')
    parser.add_argument('--risk_calibration_preserve_hard_ratio_threshold', type=float, default=0.60,
                        help='Clients above this hard-ratio can enter the W4 preserve regime.')
    parser.add_argument('--risk_calibration_correction_risk_threshold', type=float, default=0.75,
                        help='Clients above this risk level stay in the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_prior_gap_threshold', type=float, default=0.03,
                        help='Clients above this prior-gap EMA stay in the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_agreement_threshold', type=float, default=0.88,
                        help='Clients below this agreement are pushed toward the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_uncertain_threshold', type=float, default=0.06,
                        help='Clients above this both-uncertain EMA are pushed toward the W4 correction regime.')
    parser.add_argument('--risk_calibration_risk_agreement_credit', type=float, default=0.20,
                        help='Amount of risk reduction granted to clients with persistently high agreement.')
    parser.add_argument('--risk_calibration_ref_ema_momentum', type=float, default=0.97,
                        help='EMA momentum for per-client risk reference statistics (mean/deviation baselines).')
    parser.add_argument('--risk_calibration_ref_scale_ratio', type=float, default=0.15,
                        help='Scale floor ratio used by relative risk normalization against per-client reference means.')
    parser.add_argument('--risk_calibration_ref_scale_eps', type=float, default=1e-3,
                        help='Absolute scale floor used by relative risk normalization against per-client references.')
    parser.add_argument('--risk_calibration_ref_z_clip', type=float, default=3.0,
                        help='Z-score clip for relative risk terms before mapping to [0, 1].')
    parser.add_argument('--risk_calibration_hard_ratio_weight', type=float, default=0.10,
                        help='Risk-score weight assigned to low hard-ratio relative deviation.')
    parser.add_argument('--risk_calibration_release_min_correction', type=float, default=0.02,
                        help='Minimum correction strength kept for clients in release.')
    parser.add_argument('--risk_calibration_release_max_correction', type=float, default=0.12,
                        help='Maximum correction strength kept for clients in release.')
    parser.add_argument('--risk_calibration_preserve_min_correction', type=float, default=0.25,
                        help='Minimum correction strength kept for clients in preserve.')
    parser.add_argument('--risk_calibration_preserve_max_correction', type=float, default=0.55,
                        help='Maximum correction strength kept for clients in preserve.')
    parser.add_argument('--risk_calibration_correction_min_correction', type=float, default=0.75,
                        help='Minimum correction strength kept for clients in correction after release starts.')
    parser.add_argument('--risk_calibration_regime_hysteresis', type=float, default=0.05,
                        help='Margin used when mapping continuous W4 scores to correction/preserve/release regime codes.')
    parser.add_argument('--risk_calibration_anneal_end_iter', type=int, default=-1,
                        help='End iter of the late-stage continuous annealing. <0 falls back to risk_calibration_release_full_iter.')
    parser.add_argument('--risk_calibration_hard_min_correction', type=float, default=0.05,
                        help='Late-stage floor for the hard-expansion control channel before multiplying client risk.')
    parser.add_argument('--risk_calibration_hard_risk_scale', type=float, default=0.15,
                        help='Late-stage risk scale for the hard-expansion control channel.')
    parser.add_argument('--risk_calibration_calibration_min_correction', type=float, default=0.10,
                        help='Late-stage floor for the calibration channel before multiplying client risk.')
    parser.add_argument('--risk_calibration_calibration_risk_scale', type=float, default=0.40,
                        help='Late-stage risk scale for the calibration channel.')
    parser.add_argument('--risk_calibration_routing_mode', type=str, default='score',
                        help='W4 routing mode: score keeps W4.1 score competition; gated enables W4.2 sequential gate.')
    parser.add_argument('--risk_calibration_gate_release_start_iter', type=int, default=1200,
                        help='W4.2 post-mid-stage gate start. Before this iter keep W4.1-style score routing; after this iter use gated sequential routing.')
    parser.add_argument('--risk_calibration_release_streak_required', type=int, default=3,
                        help='Minimum consecutive release-ready checks required before entering release in W4.2.')
    parser.add_argument('--risk_calibration_release_streak_decay', type=int, default=1,
                        help='How much release streak decays per step when release-ready condition is not satisfied.')
    parser.add_argument('--risk_calibration_release_hard_ratio_threshold', type=float, default=0.60,
                        help='Additional hard-ratio gate for W4.2 release trigger.')
    parser.add_argument('--risk_calibration_w7_enabled', type=int, default=0,
                        help="Enable W7 two-stage mode: use the early risk-calibrated path before switch, then hard-return to pure W1'.")
    parser.add_argument('--risk_calibration_w7_switch_iter', type=int, default=800,
                        help="Hard switch iter for W7. Iter >= switch uses pure W1' adaptive pseudo-label path.")
    parser.add_argument('--risk_calibration_w7_aux_max_weight', type=float, default=0.30,
                        help="Reserved compatibility weight for deprecated post-W7 backbone+aux experiments. Unused by the restored W7 route.")
    parser.add_argument('--risk_calibration_v1_enabled', type=int, default=0,
                        help="Enable V1 mode: keep W1' as backbone and blend a light risk-calibrated auxiliary branch.")
    parser.add_argument('--risk_calibration_v1_aux_max_weight', type=float, default=0.30,
                        help="Maximum weight assigned to the auxiliary correction branch when V1 mode is enabled.")
    parser.add_argument('--risk_calibration_w5_soft_only_enabled', type=int, default=0,
                        help='Enable W5 mode: keep W1 hard branch untouched and apply risk calibration only on soft/uncertain branch.')
    parser.add_argument('--risk_calibration_w5_start_iter', type=int, default=800,
                        help='W5 soft-only calibration start iteration.')
    parser.add_argument('--risk_calibration_w5_peak_iter', type=int, default=1400,
                        help='W5 soft-only calibration peak iteration.')
    parser.add_argument('--risk_calibration_w5_end_iter', type=int, default=2400,
                        help='W5 soft-only calibration end iteration.')
    parser.add_argument('--risk_calibration_w5_min_correction', type=float, default=0.10,
                        help='W5 floor for soft-branch calibration intensity before multiplying client risk.')
    parser.add_argument('--risk_calibration_w5_risk_scale', type=float, default=0.40,
                        help='W5 soft-branch risk scale.')
    parser.add_argument('--risk_calibration_w6_late_tail_enabled', type=int, default=0,
                        help='Enable W6 late-tail retention: after W5 window, keep weak-state clients on a light soft calibration tail.')
    parser.add_argument('--risk_calibration_w6_tail_start_iter', type=int, default=-1,
                        help='W6 late-tail start iter. -1 means start at risk_calibration_w5_end_iter.')
    parser.add_argument('--risk_calibration_w6_tail_min_votes', type=int, default=2,
                        help='W6 weak-state vote threshold across {risk, prior-gap, hard-ratio} indicators.')
    parser.add_argument('--risk_calibration_w6_tail_risk_threshold', type=float, default=0.55,
                        help='W6 weak-state risk threshold (>=).')
    parser.add_argument('--risk_calibration_w6_tail_prior_gap_threshold', type=float, default=0.025,
                        help='W6 weak-state prior-gap threshold (>=).')
    parser.add_argument('--risk_calibration_w6_tail_hard_ratio_threshold', type=float, default=0.50,
                        help='W6 weak-state hard-ratio threshold (<=).')
    parser.add_argument('--risk_calibration_w6_tail_peak_fraction', type=float, default=0.30,
                        help='W6 max tail strength as a fraction of W5 calibration_target.')
    parser.add_argument('--risk_calibration_w6_tail_min_correction', type=float, default=0.03,
                        help='W6 tail floor before risk scaling.')
    parser.add_argument('--risk_calibration_w6_tail_risk_scale', type=float, default=0.12,
                        help='W6 tail risk scaling factor.')
    parser.add_argument('--max_train_samples_per_client', type=int, default=0,
                        help='Cap the number of training samples loaded per client. 0 keeps the full dataset.')
    parser.add_argument('--max_val_samples_per_client', type=int, default=0,
                        help='Cap the number of validation samples loaded per client. 0 keeps the full dataset.')
    parser.add_argument('--train_sample_ratio', type=float, default=0.0,
                        help='If > 0, load each client train split proportionally to its original size.')
    parser.add_argument('--val_sample_ratio', type=float, default=0.0,
                        help='If > 0, load each client val split proportionally to its original size.')
    parser.add_argument('--train_sample_floor', type=int, default=0,
                        help='Minimum number of train samples kept per client when using train_sample_ratio.')
    parser.add_argument('--val_sample_floor', type=int, default=0,
                        help='Minimum number of val samples kept per client when using val_sample_ratio.')
    parser.add_argument('--prototype_filter_enabled', type=int, default=0,
                        help='Enable prototype-margin-guided pseudo-label filtering.')
    parser.add_argument('--prototype_filter_warmup_rounds', type=int, default=50,
                        help='Warmup iter_global before enabling prototype filtering.')
    parser.add_argument('--prototype_filter_topk_ratio', type=float, default=0.25,
                        help='Relative top-k ratio for prototype-margin selection among unlabeled foreground pixels.')
    parser.add_argument('--student_teacher_v1', type=int, default=0,
                        help='Enable shared-teacher + local-student V1 parameter flow.')
    parser.add_argument('--student_teacher_teacher_track', type=int, default=0,
                        help='Train/upload teacher on an independent FedAvg track while keeping student local-only.')
    parser.add_argument('--student_teacher_persistent', type=int, default=1,
                        help='Whether the local student and optimizer persist across rounds.')
    parser.add_argument('--student_teacher_kd_lambda', type=float, default=0.2,
                        help='KD loss weight for teacher -> student consistency.')
    parser.add_argument('--student_teacher_kd_temperature', type=float, default=2.0,
                        help='KD temperature for teacher -> student consistency.')
    parser.add_argument('--student_teacher_student_lr_mode', type=str, default='global_decay',
                        help='Student LR schedule in teacher-track mode: global_decay or fixed.')
    parser.add_argument('--student_teacher_student_lr_value', type=float, default=-1.0,
                        help='Fixed student LR when student_teacher_student_lr_mode=fixed. <=0 means base_lr.')
    parser.add_argument('--student_teacher_kd_schedule', type=str, default='constant',
                        help='KD schedule in teacher-track mode: constant, linear_decay, or cutoff.')
    parser.add_argument('--student_teacher_kd_cutoff_iter', type=int, default=-1,
                        help='When using cutoff schedule, KD becomes 0 after this iter. <0 means max_iterations.')
    parser.add_argument('--student_teacher_kd_decay_start_iter', type=int, default=0,
                        help='When using linear_decay, KD starts decaying after this iter.')
    parser.add_argument('--student_teacher_kd_decay_end_iter', type=int, default=-1,
                        help='When using linear_decay, KD reaches 0 at this iter. <0 means max_iterations.')
    parser.add_argument('--student_teacher_refresh_mode', type=str, default='none',
                        help='Student refresh mode in teacher-track: none, soft_ema, or hard_reset.')
    parser.add_argument('--student_teacher_refresh_interval', type=int, default=100,
                        help='Refresh student every K global iterations.')
    parser.add_argument('--student_teacher_refresh_alpha', type=float, default=0.2,
                        help='EMA alpha for soft refresh: student = (1-alpha) * student + alpha * teacher.')
    parser.add_argument('--student_teacher_kd_veto', type=int, default=0,
                        help='Enable confidence-based KD veto so confident student pixels stop following teacher.')
    parser.add_argument('--student_teacher_kd_veto_confidence', type=float, default=0.85,
                        help='KD is only applied on pixels where max student probability is below this threshold.')
    parser.add_argument('--student_teacher_kd_weight_mode', type=str, default='hard_confidence_veto',
                        help='KD weighting mode when kd_veto is enabled: hard_confidence_veto or soft_entropy_disagreement.')
    parser.add_argument('--student_teacher_refresh_veto', type=int, default=0,
                        help='Enable refresh veto checks before applying periodic teacher -> student refresh.')
    parser.add_argument('--student_teacher_refresh_skip_on_teacher_drop', type=int, default=1,
                        help='When refresh veto is on, skip refresh if teacher val_mean_dice dropped at last eval.')
    parser.add_argument('--student_teacher_refresh_skip_on_student_best', type=int, default=1,
                        help='When refresh veto is on, skip refresh if student just hit a new local best at last eval.')
    parser.add_argument('--student_teacher_refresh_teacher_drop_window', type=int, default=3,
                        help='Window size used to estimate teacher improvement slope for refresh veto.')
    parser.add_argument('--student_teacher_refresh_teacher_drop_epsilon', type=float, default=0.0,
                        help='Teacher is treated as dropping only when the recent slope is below -epsilon.')
    parser.add_argument('--student_teacher_refresh_skip_history_window', type=int, default=32,
                        help='Window size used to report moving-average refresh skip ratio.')
    parser.add_argument('--student_teacher_asym_photometric', type=int, default=0,
                        help='Use teacher-clean / student-photometric asymmetry without geometric label distortion.')
    parser.add_argument('--student_teacher_asym_brightness', type=float, default=0.2,
                        help='Brightness jitter range for student photometric asymmetry.')
    parser.add_argument('--student_teacher_asym_contrast', type=float, default=0.2,
                        help='Contrast jitter range for student photometric asymmetry.')
    parser.add_argument('--student_teacher_asym_gamma', type=float, default=0.2,
                        help='Gamma jitter range for student photometric asymmetry.')
    parser.add_argument('--student_teacher_asym_noise_std', type=float, default=0.05,
                        help='Gaussian noise std for student photometric asymmetry.')
    parser.add_argument('--student_teacher_asym_blur_prob', type=float, default=0.3,
                        help='Probability of applying blur to each sample in student photometric asymmetry.')
    parser.add_argument('--student_teacher_asym_blur_kernel', type=int, default=3,
                        help='Blur kernel size for student photometric asymmetry.')
    parser.add_argument('--student_teacher_ema_enabled', type=int, default=0,
                        help='Enable a local student EMA branch as an auxiliary consistency target.')
    parser.add_argument('--student_teacher_ema_decay', type=float, default=0.99,
                        help='EMA decay for the local student momentum branch.')
    parser.add_argument('--student_teacher_ema_lambda', type=float, default=0.05,
                        help='Consistency loss weight for the local student EMA branch.')
    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = '../model/{}'.format(
        args.exp)
    os.makedirs(snapshot_path, exist_ok=True)
    setattr(args, 'snapshot_path', snapshot_path)

    # Check arguments
    assert args.iters > 0
    assert args.eval_iters > 0 and (args.eval_iters % args.iters == 0)
    assert args.max_iterations > 0 and (args.max_iterations % args.eval_iters == 0)
    assert args.geometry_guided in [0, 1]
    assert args.trustgeo_enabled in [0, 1]
    assert args.dg_enabled in [0, 1]
    assert args.support_bonus_enabled in [0, 1]
    assert args.adaptive_pl_enabled in [0, 1]
    assert args.adaptive_pl_tau_update in [0, 1]
    assert args.adaptive_pl_resume_state in [0, 1]
    assert args.geometry_num_bins == 4
    assert args.geometry_near_radius >= 0
    assert args.geometry_mid_radius >= args.geometry_near_radius
    _geometry_weights = parse_geometry_weight_list(args.geometry_pseudo_weights, args.geometry_num_bins)
    assert all(x >= 0 for x in _geometry_weights)
    assert args.trustgeo_dilation_radius >= 0
    assert args.trustgeo_density_kernel > 0 and (args.trustgeo_density_kernel % 2 == 1)
    assert 0.0 <= args.trustgeo_c_low < args.trustgeo_c_high <= 1.0
    assert 0.0 <= args.trustgeo_d_low < args.trustgeo_d_high <= 1.0
    assert 0.0 <= args.trustgeo_support_low < args.trustgeo_support_high <= 1.0
    assert 0.0 < args.trustgeo_support_mod_min <= 1.0
    assert args.trustgeo_w_c >= 0.0 and args.trustgeo_w_d >= 0.0
    assert abs((args.trustgeo_w_c + args.trustgeo_w_d) - 1.0) < 1e-6
    assert args.trustgeo_prior_alpha >= 0.0
    assert 0.0 <= args.trustgeo_prior_floor <= 1.0
    _trustgeo_prior_weights = build_linear_geometry_weight_list(
        args.geometry_num_bins,
        alpha=args.trustgeo_prior_alpha,
        floor=args.trustgeo_prior_floor,
    )
    assert len(_trustgeo_prior_weights) == args.geometry_num_bins
    assert args.dg_max_audit_samples >= 0
    assert args.dg_max_prop_samples >= 0
    assert args.dg_local_std_kernel > 0 and (args.dg_local_std_kernel % 2 == 1)
    assert args.dg_far_radius >= 0.0
    assert 0.0 <= args.dg_sep_near_radius < args.dg_sep_mid_radius < args.dg_sep_far_radius
    assert args.dg_min_fg_pixels >= 1
    assert args.dg_min_ring_pixels >= 1
    assert args.dg_min_non_seed_pixels >= 1
    assert args.dg_hom_tau > 0.0
    assert args.dg_sep_tau > 0.0
    assert 0.0 < args.dg_prop_seed_keep_ratio2 <= args.dg_prop_seed_keep_ratio1 <= 1.0
    assert args.dg_prop_min_seed_pixels >= 1
    assert args.dg_prop_min_keep_pixels >= 1
    assert args.dg_prop_min_region_pixels >= 1
    assert args.dg_prop_dist_std_scale > 0.0
    assert args.dg_dataset_w_sep >= 0.0 and args.dg_dataset_w_hom >= 0.0 and args.dg_dataset_w_prop >= 0.0
    assert abs((args.dg_dataset_w_sep + args.dg_dataset_w_hom + args.dg_dataset_w_prop) - 1.0) < 1e-6
    assert args.dg_audit_wait_timeout_sec >= 0.0
    assert args.dg_audit_wait_poll_sec > 0.0
    assert args.support_bonus_tau_c > 0.0
    assert args.support_bonus_tau_d > 0.0
    assert args.support_bonus_dilation_radius >= 0
    assert args.support_bonus_density_kernel > 0 and (args.support_bonus_density_kernel % 2 == 1)
    assert args.support_bonus_support_radius >= 0
    assert args.support_bonus_lambda >= 0.0
    assert 0.0 <= args.adaptive_pl_tau_init <= 1.0
    assert 0.0 <= args.adaptive_pl_target_accept <= 1.0
    assert args.adaptive_pl_warmup_iters >= 0
    assert args.adaptive_pl_soft_lambda >= 0.0
    assert args.adaptive_pl_min_pixels_per_bin >= 1
    assert args.adaptive_pl_log_interval >= 1
    assert args.adaptive_pl_gamma_prob >= 0.0
    assert args.adaptive_pl_gamma_conf >= 0.0
    assert args.adaptive_pl_blend_kappa >= 0.0
    assert 0.0 <= args.adaptive_pl_global_min_conf <= 1.0
    assert args.seed_support_enabled in [0, 1]
    assert args.seed_support_warmup_iters >= 0
    assert 0.0 <= args.seed_support_reliable_score_min <= 1.0
    assert args.seed_support_kernel_size > 0 and (args.seed_support_kernel_size % 2 == 1)
    assert args.seed_support_affinity_temp > 0.0
    assert 0.0 <= args.seed_support_blend_alpha <= 1.0
    assert 0.0 <= args.seed_support_hard_tau <= 1.0
    assert 0.0 <= args.seed_support_soft_tau <= 1.0
    assert args.seed_support_hard_tau >= args.seed_support_soft_tau
    assert args.risk_calibration_enabled in [0, 1]
    assert args.risk_calibration_tau_lambda >= 0.0
    assert args.risk_calibration_conf_lambda >= 0.0
    assert args.risk_calibration_prior_power >= 0.0
    assert 0.0 < args.risk_calibration_prior_clip_min <= args.risk_calibration_prior_clip_max
    assert args.risk_calibration_hard_dampen >= 0.0
    assert args.risk_calibration_soft_boost >= 0.0
    assert args.risk_calibration_global_soft_lambda >= 0.0
    assert args.risk_calibration_stateful_release_enabled in [0, 1]
    assert 0.0 <= args.risk_calibration_regime_ema_momentum < 1.0
    assert 0.0 <= args.risk_calibration_release_risk_threshold <= 1.0
    assert 0.0 <= args.risk_calibration_release_agreement_threshold <= 1.0
    assert args.risk_calibration_release_prior_gap_threshold >= 0.0
    assert args.risk_calibration_release_uncertain_threshold >= 0.0
    assert 0.0 <= args.risk_calibration_preserve_risk_threshold <= 1.0
    assert 0.0 <= args.risk_calibration_preserve_agreement_threshold <= 1.0
    assert args.risk_calibration_preserve_prior_gap_threshold >= 0.0
    assert args.risk_calibration_preserve_uncertain_threshold >= 0.0
    assert 0.0 <= args.risk_calibration_preserve_hard_ratio_threshold <= 1.0
    assert 0.0 <= args.risk_calibration_correction_risk_threshold <= 1.0
    assert args.risk_calibration_correction_prior_gap_threshold >= 0.0
    assert 0.0 <= args.risk_calibration_correction_agreement_threshold <= 1.0
    assert args.risk_calibration_correction_uncertain_threshold >= 0.0
    assert args.risk_calibration_risk_agreement_credit >= 0.0
    assert 0.0 <= args.risk_calibration_ref_ema_momentum < 1.0
    assert args.risk_calibration_ref_scale_ratio >= 0.0
    assert args.risk_calibration_ref_scale_eps > 0.0
    assert args.risk_calibration_ref_z_clip > 0.0
    assert 0.0 <= args.risk_calibration_hard_ratio_weight <= 0.5
    assert 0.0 <= args.risk_calibration_release_min_correction <= args.risk_calibration_release_max_correction <= 1.0
    assert 0.0 <= args.risk_calibration_preserve_min_correction <= args.risk_calibration_preserve_max_correction <= 1.0
    assert 0.0 <= args.risk_calibration_correction_min_correction <= 1.0
    assert args.risk_calibration_regime_hysteresis >= 0.0
    assert 0.0 <= args.risk_calibration_hard_min_correction <= 1.0
    assert args.risk_calibration_hard_risk_scale >= 0.0
    assert 0.0 <= args.risk_calibration_calibration_min_correction <= 1.0
    assert args.risk_calibration_calibration_risk_scale >= 0.0
    if args.risk_calibration_stateful_release_enabled == 1:
        assert args.risk_calibration_release_start_iter >= 0
        assert args.risk_calibration_release_full_iter > args.risk_calibration_release_start_iter
        assert args.risk_calibration_release_risk_threshold <= args.risk_calibration_preserve_risk_threshold <= args.risk_calibration_correction_risk_threshold
        assert args.risk_calibration_release_agreement_threshold >= args.risk_calibration_preserve_agreement_threshold >= args.risk_calibration_correction_agreement_threshold
        assert args.risk_calibration_release_prior_gap_threshold <= args.risk_calibration_preserve_prior_gap_threshold <= args.risk_calibration_correction_prior_gap_threshold
        assert args.risk_calibration_release_uncertain_threshold <= args.risk_calibration_preserve_uncertain_threshold <= args.risk_calibration_correction_uncertain_threshold
        assert args.risk_calibration_release_min_correction <= args.risk_calibration_release_max_correction <= args.risk_calibration_preserve_min_correction <= args.risk_calibration_preserve_max_correction <= args.risk_calibration_correction_min_correction
        assert args.risk_calibration_anneal_end_iter == -1 or args.risk_calibration_anneal_end_iter > args.risk_calibration_release_start_iter
        assert args.risk_calibration_routing_mode.lower() in ['score', 'gated']
        assert args.risk_calibration_gate_release_start_iter >= 0
        if args.risk_calibration_routing_mode.lower() == 'gated':
            assert args.risk_calibration_gate_release_start_iter >= args.risk_calibration_release_start_iter
        assert args.risk_calibration_release_streak_required >= 1
        assert args.risk_calibration_release_streak_decay >= 1
        assert 0.0 <= args.risk_calibration_release_hard_ratio_threshold <= 1.0
    assert args.risk_calibration_w7_enabled in [0, 1]
    assert args.risk_calibration_w7_switch_iter >= 0
    assert 0.0 <= args.risk_calibration_w7_aux_max_weight <= 1.0
    assert args.risk_calibration_v1_enabled in [0, 1]
    assert 0.0 <= args.risk_calibration_v1_aux_max_weight <= 1.0
    assert args.risk_calibration_w5_soft_only_enabled in [0, 1]
    assert args.risk_calibration_w5_start_iter >= 0
    assert args.risk_calibration_w5_peak_iter > args.risk_calibration_w5_start_iter
    assert args.risk_calibration_w5_end_iter > args.risk_calibration_w5_peak_iter
    assert 0.0 <= args.risk_calibration_w5_min_correction <= 1.0
    assert args.risk_calibration_w5_risk_scale >= 0.0
    assert args.risk_calibration_w6_late_tail_enabled in [0, 1]
    assert args.risk_calibration_w6_tail_start_iter >= -1
    assert args.risk_calibration_w6_tail_min_votes >= 1
    assert args.risk_calibration_w6_tail_min_votes <= 3
    assert 0.0 <= args.risk_calibration_w6_tail_risk_threshold <= 1.0
    assert args.risk_calibration_w6_tail_prior_gap_threshold >= 0.0
    assert 0.0 <= args.risk_calibration_w6_tail_hard_ratio_threshold <= 1.0
    assert 0.0 <= args.risk_calibration_w6_tail_peak_fraction <= 1.0
    assert 0.0 <= args.risk_calibration_w6_tail_min_correction <= 1.0
    assert args.risk_calibration_w6_tail_risk_scale >= 0.0
    if args.risk_calibration_w5_soft_only_enabled == 1:
        assert args.risk_calibration_enabled == 1
        assert args.risk_calibration_stateful_release_enabled == 0
        assert args.seed_support_enabled == 0
        assert args.risk_calibration_tau_lambda == 0.0
        assert args.risk_calibration_conf_lambda == 0.0
        assert args.risk_calibration_hard_dampen == 0.0
    if args.risk_calibration_v1_enabled == 1:
        assert args.risk_calibration_enabled == 1
        assert args.risk_calibration_w5_soft_only_enabled == 0
        assert args.risk_calibration_w6_late_tail_enabled == 0
        assert args.risk_calibration_w7_enabled == 0
        assert args.risk_calibration_v1_aux_max_weight >= 0.0
    if args.risk_calibration_w7_enabled == 1:
        assert args.risk_calibration_enabled == 1
        assert args.risk_calibration_v1_enabled == 0
        assert args.risk_calibration_w5_soft_only_enabled == 0
        assert args.risk_calibration_w6_late_tail_enabled == 0
        assert args.risk_calibration_w7_aux_max_weight >= 0.0
        assert args.risk_calibration_w7_switch_iter >= args.adaptive_pl_warmup_iters
    if args.risk_calibration_w6_late_tail_enabled == 1:
        assert args.risk_calibration_w5_soft_only_enabled == 1
        if args.risk_calibration_w6_tail_start_iter >= 0:
            assert args.risk_calibration_w6_tail_start_iter >= args.risk_calibration_w5_end_iter
    if args.adaptive_pl_enabled == 1:
        assert args.geometry_guided == 1
    if args.trustgeo_enabled == 1:
        assert args.geometry_guided == 1
    if args.dg_enabled == 1:
        assert args.geometry_guided == 1
        assert args.trustgeo_enabled == 0
    if args.support_bonus_enabled == 1:
        assert args.geometry_guided == 1
        assert args.trustgeo_enabled == 0
        assert args.dg_enabled == 0
    if args.seed_support_enabled == 1:
        assert args.adaptive_pl_enabled == 1
    assert args.ala_num_pre_loss > 0
    assert args.ala_max_init_epochs > 0
    assert args.max_train_samples_per_client >= 0
    assert args.max_val_samples_per_client >= 0
    assert 0.0 <= args.train_sample_ratio <= 1.0
    assert 0.0 <= args.val_sample_ratio <= 1.0
    assert args.train_sample_floor >= 0
    assert args.val_sample_floor >= 0
    assert args.prototype_filter_enabled in [0, 1]
    assert args.prototype_filter_warmup_rounds >= 0
    assert 0.0 < args.prototype_filter_topk_ratio <= 1.0
    assert not (args.adaptive_pl_enabled == 1 and args.prototype_filter_enabled == 1)
    assert args.student_teacher_v1 in [0, 1]
    assert args.student_teacher_teacher_track in [0, 1]
    assert args.student_teacher_persistent in [0, 1]
    assert args.student_teacher_kd_lambda >= 0.0
    assert args.student_teacher_kd_temperature > 0.0
    assert args.student_teacher_student_lr_mode in ['global_decay', 'fixed']
    assert args.student_teacher_student_lr_value == -1.0 or args.student_teacher_student_lr_value > 0.0
    assert args.student_teacher_kd_schedule in ['constant', 'linear_decay', 'cutoff']
    assert args.student_teacher_kd_cutoff_iter == -1 or args.student_teacher_kd_cutoff_iter >= 0
    assert args.student_teacher_kd_decay_start_iter >= 0
    assert args.student_teacher_kd_decay_end_iter == -1 or args.student_teacher_kd_decay_end_iter >= 0
    assert args.student_teacher_refresh_mode in ['none', 'soft_ema', 'hard_reset']
    assert args.student_teacher_refresh_interval > 0
    assert 0.0 <= args.student_teacher_refresh_alpha <= 1.0
    assert args.student_teacher_kd_veto in [0, 1]
    assert 0.0 <= args.student_teacher_kd_veto_confidence <= 1.0
    assert args.student_teacher_kd_weight_mode in ['hard_confidence_veto', 'soft_entropy_disagreement']
    assert args.student_teacher_refresh_veto in [0, 1]
    assert args.student_teacher_refresh_skip_on_teacher_drop in [0, 1]
    assert args.student_teacher_refresh_skip_on_student_best in [0, 1]
    assert args.student_teacher_refresh_teacher_drop_window >= 2
    assert args.student_teacher_refresh_teacher_drop_epsilon >= 0.0
    assert args.student_teacher_refresh_skip_history_window >= 1
    assert args.student_teacher_asym_photometric in [0, 1]
    assert args.student_teacher_asym_brightness >= 0.0
    assert args.student_teacher_asym_contrast >= 0.0
    assert args.student_teacher_asym_gamma >= 0.0
    assert args.student_teacher_asym_noise_std >= 0.0
    assert 0.0 <= args.student_teacher_asym_blur_prob <= 1.0
    assert args.student_teacher_asym_blur_kernel > 0
    assert args.student_teacher_ema_enabled in [0, 1]
    assert 0.0 <= args.student_teacher_ema_decay < 1.0
    assert args.student_teacher_ema_lambda >= 0.0
    if args.student_teacher_kd_cutoff_iter < 0:
        args.student_teacher_kd_cutoff_iter = args.max_iterations
    if args.student_teacher_kd_decay_end_iter < 0:
        args.student_teacher_kd_decay_end_iter = args.max_iterations
    assert args.student_teacher_kd_decay_end_iter >= args.student_teacher_kd_decay_start_iter

    if args.strategy in ['FedLC', 'FedALALC', 'FedAPLC', 'FedUni', 'FedUniV2', 'FedUniV2.1']:
        assert args.tsne_iters > 0 and (args.tsne_iters % args.iters == 0)

    if args.strategy == 'FedRep':
        assert args.iters > args.rep_iters
    elif args.strategy == 'MetaFed':
        assert args.eval_iters == args.iters
        assert args.init_iters > args.eval_iters and (args.init_iters % args.eval_iters == 0)
        assert args.common_iters > args.init_iters and (args.common_iters % args.eval_iters == 0)
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedAP':
        assert args.pretrain_iters > args.eval_iters and (args.pretrain_iters % args.eval_iters == 0)
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedALA':
        assert args.model in ['unet', 'unet_head', 'unet_multihead']
    elif args.strategy == 'FedLC':
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']
    elif args.strategy == 'FedALALC':
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']
    elif args.strategy == 'FedAPLC':
        assert args.pretrain_iters > args.eval_iters and (args.pretrain_iters % args.eval_iters == 0)
        assert args.iters > args.rep_iters
        assert args.model in ['unet_lc']
    elif args.strategy == 'FedUni':
        assert args.model in ['unet_uni']
    elif args.strategy in ['FedUniV2', 'FedUniV2.1']:
        assert args.model in ['unet_univ2', 'unet_univ3', 'unet_univ4', 'unet_univ5']
        assert args.prompt in ['universal', 'onehot']
        assert args.attention in ['dual', 'sab', 'cab', 'multi']
        assert args.dual_init in ['random', 'adjacent', 'nearest', 'aggregated']

    if args.student_teacher_v1 == 1:
        assert args.strategy in ['FedUniV2', 'FedUniV2.1']
        if args.student_teacher_teacher_track == 1:
            assert args.student_teacher_persistent == 1

    assert args.role in ['server', 'client']
    assert args.img_class in ['odoc', 'faz', 'polyp', 'prostate']
    if args.img_class in ['faz', 'prostate']:
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']
    else:
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']

    # Configure logger
    if args.role == 'server':
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree(
            '.',
            snapshot_path + '/code',
            ignore=shutil.ignore_patterns('.git', '__pycache__', 'bilateralfilter', 'SegmentationAug'),
        )
        fl.common.logger.configure('server', filename=os.path.join(snapshot_path, 'server.log'))
        writer = SummaryWriter(snapshot_path + '/log')
    else:
        fl.common.logger.configure('client_{}'.format(args.cid), filename=os.path.join(snapshot_path, 'client_{}.log'.format(args.cid)))
        writer = SummaryWriter(os.path.join(snapshot_path, 'client_{}_log'.format(args.cid)))

    log(INFO, 'Arguments: {}'.format(args))

    # Load model and data
    db_train = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(
            args.patch_size,
            img_class=args.img_class,
            geometry_guided=bool(getattr(args, 'geometry_guided', 0)),
            geometry_near_radius=getattr(args, 'geometry_near_radius', 8.0),
            geometry_mid_radius=getattr(args, 'geometry_mid_radius', 24.0),
        )
    ]), client=args.client, sup_type=args.sup_type, img_class=args.img_class,
        geometry_guided=bool(getattr(args, 'geometry_guided', 0)),
        geometry_near_radius=getattr(args, 'geometry_near_radius', 8.0),
        geometry_mid_radius=getattr(args, 'geometry_mid_radius', 24.0),
        max_train_samples_per_client=args.max_train_samples_per_client,
        max_val_samples_per_client=args.max_val_samples_per_client,
        train_sample_ratio=args.train_sample_ratio,
        val_sample_ratio=args.val_sample_ratio,
        train_sample_floor=args.train_sample_floor,
        val_sample_floor=args.val_sample_floor)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split='val', img_class=args.img_class,
                          geometry_guided=bool(getattr(args, 'geometry_guided', 0)),
                          geometry_near_radius=getattr(args, 'geometry_near_radius', 8.0),
                          geometry_mid_radius=getattr(args, 'geometry_mid_radius', 24.0),
                          max_train_samples_per_client=args.max_train_samples_per_client,
                          max_val_samples_per_client=args.max_val_samples_per_client,
                          train_sample_ratio=args.train_sample_ratio,
                          val_sample_ratio=args.val_sample_ratio,
                          train_sample_floor=args.train_sample_floor,
                          val_sample_floor=args.val_sample_floor)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    drop_last = True if args.strategy in ['FedUni', 'FedUniV2', 'FedUniV2.1'] else False
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=drop_last)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.strategy in ['FedLC', 'FedALALC', 'FedAPLC']:
        class MyModelV2(MyModel):
            def forward(self, x, emb_idx=None):
                return self.model(x, emb_idx)
        model = MyModelV2(args, net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes), trainloader, valloader)
    else:
        model = MyModel(args, net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes), trainloader, valloader)
    model.cuda()

    if args.role == 'server':
        prototype_bank_state = {}
        def fit_config(server_round):
            config = {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'fit'
            }
            return inject_prototype_bank_into_config(config, prototype_bank_state)

        def evaluate_config_fn(server_round):
            config = {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'evaluate'
            }
            return config

        # Create strategy
        kwargs = {
            "fraction_fit": args.sample_fraction,
            "min_fit_clients": args.min_num_clients,
            "min_available_clients": args.min_num_clients,
            "evaluate_fn": get_evaluate_fn(args, valloader, amp=(args.amp == 1)),
            "on_fit_config_fn": fit_config,
            "on_evaluate_config_fn": evaluate_config_fn,
            "fit_metrics_aggregation_fn": fit_metrics_aggregation_fn,
            "evaluate_metrics_aggregation_fn": get_evaluate_metrics_aggregation_fn(args, val_metrics=VAL_METRICS),
            "accept_failures": False
        
        }
        '''weights = [val.cpu().numpy() for _, val in model.model.state_dict().items()]
        initial_parameters = fl.common.ndarrays_to_parameters(weights)
        if args.strategy == 'FedAdagrad':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'tau': 1e-9,
                        'initial_parameters': initial_parameters})
        elif args.strategy == 'FedAdam':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'beta_1': 0.9,
                        'beta_2': 0.99, 'tau': 1e-9, 'initial_parameters': initial_parameters})
        elif args.strategy == 'FedYogi':
            kwargs.update({'eta': 5e-3, 'eta_l': 5e-3, 'beta_1': 0.9,
                        'beta_2': 0.99, 'tau': 1e-9, 'initial_parameters': initial_parameters})'''
        if args.strategy in ['FedAP', 'FedAPLC']:
            if args.pretrain:
                start_time = time.time()
                writer = SummaryWriter(snapshot_path + '/pretrain_log')
                pretrain_path, pretrain_state_dict = pretrain_model(args, writer, worker_init_fn)
                print(time.time() - start_time)
                return
            else:
                pretrain_state_dict = torch.load('{}/fedap_pretrain.pth'.format(args.snapshot_path), map_location='cpu')
                weights = [val.cpu().numpy() for _, val in pretrain_state_dict.items()]
                initial_parameters = fl.common.ndarrays_to_parameters(weights)
                kwargs.update({'iters': args.iters, 'model_momentum': args.model_momentum, 'initial_parameters': initial_parameters})

        if args.strategy == 'FedAN':
            kwargs.update({'iters': args.iters, 'lam': args.lam, 'beta': args.beta})

        strategy = get_strategy(args.strategy, **kwargs)
        # Start server
        state_dict_keys = model.model.state_dict().keys()
        train_scalar_metrics = ['lr', 'total_loss', 'loss_ce']
        if args.student_teacher_v1 == 1:
            train_scalar_metrics += [
                'loss_kd',
                'effective_kd_lambda',
                'kd_active_ratio',
                'teacher_student_logit_cosine',
                'teacher_student_prediction_disagreement',
                'teacher_student_cka_down3',
                'teacher_student_cka_down4',
            ]
            if args.student_teacher_teacher_track == 1:
                train_scalar_metrics += [
                    'student_lr',
                    'teacher_lr',
                    'student_refreshed',
                    'refresh_skip_ratio',
                    'refresh_veto_teacher_drop',
                    'refresh_veto_student_best',
                    'refresh_veto_applied',
                    'teacher_improvement_rate',
                    'loss_ema',
                ]
        train_image_metrics = ['Image', 'Prediction', 'GroundTruth']
        if args.strategy in ['FedUniV2', 'FedUniV2.1']:
            train_image_metrics += ['Prediction2', 'Pseudo']
        val_metrics = VAL_METRICS
        server = MyServer(
            args=args, writer=writer, state_dict_keys=state_dict_keys, train_scalar_metrics=train_scalar_metrics,
            train_image_metrics=train_image_metrics, val_metrics=val_metrics, client_manager=SimpleClientManager(), strategy=strategy,
            prototype_bank_state=prototype_bank_state,
        )
        fl.server.start_server(
            server_address=args.server_address,
            server=server,
            config=ServerConfig(num_rounds=args.max_iterations, round_timeout=None)
        )
    else:
        client = MyClient(args, model, trainloader, valloader, amp=(args.amp == 1))
        fl.client.start_client(server_address=args.server_address, client=client)



if __name__ == '__main__':
    main()
