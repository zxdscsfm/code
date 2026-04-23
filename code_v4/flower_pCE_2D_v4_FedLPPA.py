# -*- coding:utf-8 -*-
import argparse
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
from dataloaders.dataset import BaseDataSets, RandomGenerator
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
            if args.img_class == 'faz':
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

    def _adaptive_pl_is_enabled(self):
        return bool(getattr(self.args, 'adaptive_pl_enabled', 0)) and self.args.strategy in ['FedUniV2', 'FedUniV2.1']

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
        self.adaptive_pl_last_regime_code = 0.0
        self.adaptive_pl_release_streak = 0.0
        self.adaptive_pl_last_release_ready = 0.0
        self.adaptive_pl_last_high_risk = 0.0
        self.adaptive_pl_gate_release_armed = 0.0

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
            'adaptive_pl_last_regime_code': float(self.adaptive_pl_last_regime_code),
            'adaptive_pl_release_streak': float(self.adaptive_pl_release_streak),
            'adaptive_pl_last_release_ready': float(self.adaptive_pl_last_release_ready),
            'adaptive_pl_last_high_risk': float(self.adaptive_pl_last_high_risk),
            'adaptive_pl_gate_release_armed': float(self.adaptive_pl_gate_release_armed),
        }
        torch.save(state, self._get_adaptive_pl_state_path(tag=tag))

    def _maybe_log_adaptive_pl_state(self):
        global writer
        if writer is None or not self._adaptive_pl_is_enabled():
            return
        log_interval = int(getattr(self.args, 'adaptive_pl_log_interval', 50))
        if log_interval <= 0 or self.current_iter % log_interval != 0:
            return
        writer.add_scalar('client_{}/adaptive_pl/lg_class_agree_ratio'.format(self.cid), float(self.adaptive_pl_last_lg_class_agree_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/lg_class_agree_ema'.format(self.cid), float(self.adaptive_pl_lg_agreement_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_prob_gap'.format(self.cid), float(self.adaptive_pl_last_mean_prob_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_prob_gap_ema'.format(self.cid), float(self.adaptive_pl_prob_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_conf_gap'.format(self.cid), float(self.adaptive_pl_last_mean_conf_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/mean_conf_gap_ema'.format(self.cid), float(self.adaptive_pl_conf_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/both_uncertain_ratio'.format(self.cid), float(self.adaptive_pl_last_both_uncertain_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/both_uncertain_ema'.format(self.cid), float(self.adaptive_pl_both_uncertain_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/hard_ratio'.format(self.cid), float(self.adaptive_pl_last_hard_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/loss_hard'.format(self.cid), float(self.adaptive_pl_last_loss_hard), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/loss_soft'.format(self.cid), float(self.adaptive_pl_last_loss_soft), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/boundary_loss_oc'.format(self.cid), float(self.adaptive_pl_last_boundary_loss_oc), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/ring_valid_oc_ratio'.format(self.cid), float(self.adaptive_pl_last_ring_valid_oc_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/seed_support_mean'.format(self.cid), float(self.adaptive_pl_last_seed_support_mean), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/seed_support_soft_ratio'.format(self.cid), float(self.adaptive_pl_last_seed_support_soft_ratio), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/propagated_fg_mean'.format(self.cid), float(self.adaptive_pl_last_propagated_fg_mean), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/prior_gap'.format(self.cid), float(self.adaptive_pl_last_prior_gap), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/prior_gap_ema'.format(self.cid), float(self.adaptive_pl_prior_gap_ema), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/client_risk'.format(self.cid), float(self.adaptive_pl_last_client_risk), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/loss_risk'.format(self.cid), float(self.adaptive_pl_last_loss_risk), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/stage_progress'.format(self.cid), float(self.adaptive_pl_last_stage_progress), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/release_score'.format(self.cid), float(self.adaptive_pl_last_release_score), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/preserve_score'.format(self.cid), float(self.adaptive_pl_last_preserve_score), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/correction_score'.format(self.cid), float(self.adaptive_pl_last_correction_score), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/effective_correction'.format(self.cid), float(self.adaptive_pl_last_effective_correction), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/regime_code'.format(self.cid), float(self.adaptive_pl_last_regime_code), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/release_streak'.format(self.cid), float(self.adaptive_pl_release_streak), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/release_ready'.format(self.cid), float(self.adaptive_pl_last_release_ready), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/high_risk_flag'.format(self.cid), float(self.adaptive_pl_last_high_risk), self.current_iter)
        writer.add_scalar('client_{}/adaptive_pl/gate_release_armed'.format(self.cid), float(self.adaptive_pl_gate_release_armed), self.current_iter)
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

    def _compute_adaptive_pl_loss(self, outputs, outputs_auxiliary, outputs_soft, outputs_soft_auxiliary,
                                  pseudo_label_mix, global_pseudo_label_mix, geometry_bin_batch, label_batch,
                                  raw_encoder_feature=None):
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
        geometry_weight_values = parse_geometry_weight_list(
            getattr(self.args, 'geometry_pseudo_weights', '1.0,0.8,0.5,0.2'),
            int(getattr(self.args, 'geometry_num_bins', 4)),
        )
        geometry_weight_map = build_geometry_weight_map(geometry_bin_batch, geometry_weight_values)
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
        correction_risk_threshold = float(getattr(self.args, 'risk_calibration_correction_risk_threshold', 0.55))
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
        routing_mode = str(getattr(self.args, 'risk_calibration_routing_mode', 'score')).lower()
        gate_release_start_iter = int(getattr(self.args, 'risk_calibration_gate_release_start_iter', max(release_start_iter, warmup_iters + 400)))
        release_streak_required = int(getattr(self.args, 'risk_calibration_release_streak_required', 3))
        release_streak_decay = int(getattr(self.args, 'risk_calibration_release_streak_decay', 1))
        release_hard_ratio_threshold = float(getattr(self.args, 'risk_calibration_release_hard_ratio_threshold', preserve_hard_ratio_threshold))
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
            prior_risk_term = float(np.clip(self.adaptive_pl_prior_gap_ema / 0.05, 0.0, 1.0))
            prob_risk_term = float(np.clip(self.adaptive_pl_prob_gap_ema / 0.03, 0.0, 1.0))
            conf_risk_term = float(np.clip(self.adaptive_pl_conf_gap_ema / 0.03, 0.0, 1.0))
            disagree_risk_term = float(np.clip(
                (1.0 - self.adaptive_pl_lg_agreement_ema) / max(1e-6, 1.0 - correction_agreement_threshold),
                0.0,
                1.0,
            ))
            agreement_credit = float(np.clip(
                (self.adaptive_pl_lg_agreement_ema - preserve_agreement_threshold) / max(1e-6, 1.0 - preserve_agreement_threshold),
                0.0,
                1.0,
            ))
            # Keep prior-gap as the main risk source, use probability/confidence gaps as auxiliaries,
            # and explicitly reward high agreement so recovered clients can leave strict correction.
            risk_score = (
                0.40 * prior_risk_term
                + 0.20 * prob_risk_term
                + 0.10 * conf_risk_term
                + 0.30 * disagree_risk_term
                - (risk_agreement_credit * agreement_credit)
            )
            risk_score = float(np.clip(risk_score, 0.0, 1.0))
        else:
            risk_score = 0.0
        self.adaptive_pl_last_client_risk = risk_score

        stage_progress = 0.0
        release_score = 0.0
        preserve_score = 0.0
        correction_score = 0.0
        effective_correction = 0.0
        regime_code = 0.0
        release_ready_flag = 0.0
        high_risk_flag = 0.0
        if risk_calibration_enabled:
            effective_correction = 1.0
            regime_code = 0.0
        if risk_calibration_enabled and stateful_release_enabled:
            release_start_iter = max(warmup_iters, release_start_iter)
            gate_release_start_iter = max(release_start_iter, gate_release_start_iter)
            release_full_iter = max(release_start_iter + 1, release_full_iter)
            release_streak_required = max(1, release_streak_required)
            release_streak_decay = max(1, release_streak_decay)
            release_uncertain_term = float(np.clip(
                (release_uncertain_threshold - self.adaptive_pl_both_uncertain_ema) / max(release_uncertain_threshold, 1e-6),
                0.0,
                1.0,
            ))
            release_risk_term = float(np.clip(
                (release_risk_threshold - risk_score) / max(release_risk_threshold, 1e-6),
                0.0,
                1.0,
            ))
            release_agreement_term = float(np.clip(
                (self.adaptive_pl_lg_agreement_ema - release_agreement_threshold) / max(1e-6, 1.0 - release_agreement_threshold),
                0.0,
                1.0,
            ))
            release_prior_term = float(np.clip(
                (release_prior_gap_threshold - self.adaptive_pl_prior_gap_ema) / max(release_prior_gap_threshold, 1e-6),
                0.0,
                1.0,
            ))
            release_raw = min(release_risk_term, release_agreement_term, release_prior_term, release_uncertain_term)
            preserve_risk_term = float(np.clip(
                (preserve_risk_threshold - risk_score) / max(preserve_risk_threshold, 1e-6),
                0.0,
                1.0,
            ))
            preserve_agreement_term = float(np.clip(
                (self.adaptive_pl_lg_agreement_ema - preserve_agreement_threshold) / max(1e-6, 1.0 - preserve_agreement_threshold),
                0.0,
                1.0,
            ))
            preserve_prior_term = float(np.clip(
                (preserve_prior_gap_threshold - self.adaptive_pl_prior_gap_ema) / max(preserve_prior_gap_threshold, 1e-6),
                0.0,
                1.0,
            ))
            preserve_uncertain_term = float(np.clip(
                (preserve_uncertain_threshold - self.adaptive_pl_both_uncertain_ema) / max(preserve_uncertain_threshold, 1e-6),
                0.0,
                1.0,
            ))
            preserve_hard_term = float(np.clip(
                (self.adaptive_pl_last_hard_ratio - preserve_hard_ratio_threshold) / max(1e-6, 1.0 - preserve_hard_ratio_threshold),
                0.0,
                1.0,
            ))
            preserve_raw = min(
                preserve_risk_term,
                preserve_agreement_term,
                preserve_prior_term,
                preserve_uncertain_term,
                preserve_hard_term,
            )
            correction_risk_term = float(np.clip(
                (risk_score - correction_risk_threshold) / max(1e-6, 1.0 - correction_risk_threshold),
                0.0,
                1.0,
            ))
            correction_prior_term = float(np.clip(
                (self.adaptive_pl_prior_gap_ema - correction_prior_gap_threshold) / max(correction_prior_gap_threshold, 1e-6),
                0.0,
                1.0,
            ))
            correction_agreement_term = float(np.clip(
                (correction_agreement_threshold - self.adaptive_pl_lg_agreement_ema) / max(correction_agreement_threshold, 1e-6),
                0.0,
                1.0,
            ))
            correction_uncertain_term = float(np.clip(
                (self.adaptive_pl_both_uncertain_ema - correction_uncertain_threshold) / max(correction_uncertain_threshold, 1e-6),
                0.0,
                1.0,
            ))
            correction_raw = max(
                correction_risk_term,
                correction_prior_term,
                correction_agreement_term,
                correction_uncertain_term,
            )
            self.adaptive_pl_release_score_ema = (
                regime_ema_momentum * self.adaptive_pl_release_score_ema
            ) + ((1.0 - regime_ema_momentum) * release_raw)
            self.adaptive_pl_preserve_score_ema = (
                regime_ema_momentum * self.adaptive_pl_preserve_score_ema
            ) + ((1.0 - regime_ema_momentum) * preserve_raw)
            self.adaptive_pl_correction_score_ema = (
                regime_ema_momentum * self.adaptive_pl_correction_score_ema
            ) + ((1.0 - regime_ema_momentum) * correction_raw)

            if self.current_iter >= release_start_iter:
                # W4.1: early unified correction before T1, then route clients into
                # correction / preserve / release instead of keeping a single conservative path.
                stage_progress = float(np.clip(
                    (self.current_iter - release_start_iter) / float(release_full_iter - release_start_iter),
                    0.0,
                    1.0,
                ))
                release_score = float(np.clip(stage_progress * self.adaptive_pl_release_score_ema, 0.0, 1.0))
                preserve_score = float(np.clip(stage_progress * self.adaptive_pl_preserve_score_ema, 0.0, 1.0))
                correction_score = float(np.clip(stage_progress * self.adaptive_pl_correction_score_ema, 0.0, 1.0))
                prev_regime = int(np.clip(round(self.adaptive_pl_last_regime_code), 0, 2))
                correction_target = float(np.clip(max(correction_min_correction, correction_score), 0.0, 1.0))
                preserve_target = float(np.clip(
                    preserve_min_correction + (1.0 - preserve_score) * (preserve_max_correction - preserve_min_correction),
                    0.0,
                    1.0,
                ))
                release_target = float(np.clip(
                    release_min_correction + (1.0 - release_score) * (release_max_correction - release_min_correction),
                    0.0,
                    1.0,
                ))

                if routing_mode != 'gated' or self.current_iter < gate_release_start_iter:
                    # Keep W4.1 behavior before post-1200 W4.2 gate.
                    candidate_scores = [correction_score, preserve_score, release_score]
                    best_regime = int(np.argmax(candidate_scores))
                    if candidate_scores[prev_regime] + regime_hysteresis >= candidate_scores[best_regime]:
                        regime_code = float(prev_regime)
                    else:
                        regime_code = float(best_regime)
                    regime_target = correction_target
                    if int(regime_code) == 1:
                        regime_target = preserve_target
                    elif int(regime_code) == 2:
                        regime_target = release_target
                    effective_correction = float(np.clip(
                        ((1.0 - stage_progress) * 1.0) + (stage_progress * regime_target),
                        0.0,
                        1.0,
                    ))
                    self.adaptive_pl_release_streak = max(
                        0.0,
                        self.adaptive_pl_release_streak - float(release_streak_decay),
                    )
                    self.adaptive_pl_gate_release_armed = 0.0
                else:
                    # W4.2: gated sequential routing (correction -> release -> preserve)
                    high_risk = bool(
                        (risk_score >= correction_risk_threshold)
                        or (self.adaptive_pl_prior_gap_ema >= correction_prior_gap_threshold)
                        or (self.adaptive_pl_lg_agreement_ema <= correction_agreement_threshold)
                        or (self.adaptive_pl_both_uncertain_ema >= correction_uncertain_threshold)
                    )
                    release_ready = bool(
                        (risk_score <= release_risk_threshold)
                        and (self.adaptive_pl_lg_agreement_ema >= release_agreement_threshold)
                        and (self.adaptive_pl_prior_gap_ema <= release_prior_gap_threshold)
                        and (self.adaptive_pl_both_uncertain_ema <= release_uncertain_threshold)
                        and (self.adaptive_pl_last_hard_ratio >= release_hard_ratio_threshold)
                    )
                    release_keep_ready = bool(
                        (risk_score <= preserve_risk_threshold)
                        and (self.adaptive_pl_lg_agreement_ema >= preserve_agreement_threshold)
                        and (self.adaptive_pl_prior_gap_ema <= preserve_prior_gap_threshold)
                        and (self.adaptive_pl_both_uncertain_ema <= preserve_uncertain_threshold)
                        and (self.adaptive_pl_last_hard_ratio >= preserve_hard_ratio_threshold)
                    )
                    high_risk_flag = 1.0 if high_risk else 0.0
                    release_ready_flag = 1.0 if release_ready else 0.0

                    if high_risk:
                        self.adaptive_pl_release_streak = 0.0
                        self.adaptive_pl_gate_release_armed = 0.0
                        regime_code = 0.0
                    else:
                        if release_ready:
                            self.adaptive_pl_release_streak = min(
                                float(release_streak_required + 8),
                                self.adaptive_pl_release_streak + 1.0,
                            )
                        elif self.adaptive_pl_gate_release_armed >= 1.0 and prev_regime == 2 and release_keep_ready:
                            self.adaptive_pl_release_streak = max(
                                self.adaptive_pl_release_streak,
                                float(max(0, release_streak_required - 1)),
                            )
                        else:
                            self.adaptive_pl_release_streak = max(
                                0.0,
                                self.adaptive_pl_release_streak - float(release_streak_decay),
                            )

                        release_triggered = self.adaptive_pl_release_streak >= float(release_streak_required)
                        if (not release_triggered) and self.adaptive_pl_gate_release_armed >= 1.0 and prev_regime == 2 and release_keep_ready:
                            release_triggered = self.adaptive_pl_release_streak >= float(max(0, release_streak_required - 1))
                        regime_code = 2.0 if release_triggered else 1.0
                        if release_triggered:
                            self.adaptive_pl_gate_release_armed = 1.0

                    regime_target = correction_target
                    if int(regime_code) == 1:
                        regime_target = preserve_target
                    elif int(regime_code) == 2:
                        regime_target = release_target
                    effective_correction = regime_target
            else:
                release_score = 0.0
                preserve_score = 0.0
                correction_score = 1.0
                effective_correction = 1.0
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
        if risk_calibration_enabled and valid_mask.any():
            prior_scale = ((global_prior + 1e-6) / (local_prior + 1e-6)).clamp(
                min=risk_prior_clip_min,
                max=risk_prior_clip_max,
            )
            prior_scale = torch.pow(prior_scale, risk_prior_power * risk_score * effective_correction)
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
                corrected_prob=corrected_prob,
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
            tau_map_effective = (tau_map + (risk_tau_lambda * risk_score * effective_correction)).clamp(0.3, 0.95)
            conf_global_cutoff = min(0.95, tau_global_min + (risk_conf_lambda * risk_score * effective_correction))
            hard_client_scale = max(0.0, 1.0 - (risk_hard_dampen * risk_score * effective_correction))
            soft_client_scale = 1.0 + (risk_soft_boost * risk_score * effective_correction)

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

        hard_weight = geometry_weight_map * hard_mask.float() * valid_mask.float()
        soft_weight = geometry_weight_map * soft_mask.float() * valid_mask.float()
        risk_weight = geometry_weight_map * both_uncertain.float() * valid_mask.float()
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
            total_loss = total_loss + (risk_global_soft_lambda * effective_correction * loss_risk)
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
                    geometry_weight_values = parse_geometry_weight_list(
                        getattr(self.args, 'geometry_pseudo_weights', '1.0,0.8,0.5,0.2'),
                        int(getattr(self.args, 'geometry_num_bins', 4)),
                    )
                    geometry_weight_map = build_geometry_weight_map(
                        geometry_bin_batch,
                        geometry_weight_values,
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

            if self.args.img_class == 'faz':
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

            if self.args.img_class == 'faz':
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
                                geometry_weight_values = parse_geometry_weight_list(
                                    getattr(self.args, 'geometry_pseudo_weights', '1.0,0.8,0.5,0.2'),
                                    int(getattr(self.args, 'geometry_num_bins', 4)),
                                )
                                geometry_weight_map = build_geometry_weight_map(
                                    geometry_bin_batch,
                                    geometry_weight_values,
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
            if self._adaptive_pl_is_enabled():
                metrics_['client_{}_adaptive_pl_loss_hard'.format(self.cid)] = float(adaptive_pl_loss_hard.item())
                metrics_['client_{}_adaptive_pl_loss_soft'.format(self.cid)] = float(adaptive_pl_loss_soft.item())
                metrics_['client_{}_adaptive_pl_hard_ratio'.format(self.cid)] = float(adaptive_pl_hard_ratio)
                metrics_['client_{}_adaptive_pl_lg_class_agree_ratio'.format(self.cid)] = float(self.adaptive_pl_last_lg_class_agree_ratio)
                metrics_['client_{}_adaptive_pl_lg_class_agree_ema'.format(self.cid)] = float(self.adaptive_pl_lg_agreement_ema)
                metrics_['client_{}_adaptive_pl_mean_prob_gap'.format(self.cid)] = float(self.adaptive_pl_last_mean_prob_gap)
                metrics_['client_{}_adaptive_pl_mean_conf_gap'.format(self.cid)] = float(self.adaptive_pl_last_mean_conf_gap)
                metrics_['client_{}_adaptive_pl_both_uncertain_ratio'.format(self.cid)] = float(self.adaptive_pl_last_both_uncertain_ratio)
                metrics_['client_{}_adaptive_pl_both_uncertain_ema'.format(self.cid)] = float(self.adaptive_pl_both_uncertain_ema)
                metrics_['client_{}_adaptive_pl_boundary_loss_oc'.format(self.cid)] = float(self.adaptive_pl_last_boundary_loss_oc)
                metrics_['client_{}_adaptive_pl_ring_valid_oc_ratio'.format(self.cid)] = float(self.adaptive_pl_last_ring_valid_oc_ratio)
                metrics_['client_{}_adaptive_pl_prior_gap'.format(self.cid)] = float(self.adaptive_pl_last_prior_gap)
                metrics_['client_{}_adaptive_pl_client_risk'.format(self.cid)] = float(self.adaptive_pl_last_client_risk)
                metrics_['client_{}_adaptive_pl_loss_risk'.format(self.cid)] = float(self.adaptive_pl_last_loss_risk)
                metrics_['client_{}_adaptive_pl_stage_progress'.format(self.cid)] = float(self.adaptive_pl_last_stage_progress)
                metrics_['client_{}_adaptive_pl_release_score'.format(self.cid)] = float(self.adaptive_pl_last_release_score)
                metrics_['client_{}_adaptive_pl_preserve_score'.format(self.cid)] = float(self.adaptive_pl_last_preserve_score)
                metrics_['client_{}_adaptive_pl_correction_score'.format(self.cid)] = float(self.adaptive_pl_last_correction_score)
                metrics_['client_{}_adaptive_pl_effective_correction'.format(self.cid)] = float(self.adaptive_pl_last_effective_correction)
                metrics_['client_{}_adaptive_pl_regime_code'.format(self.cid)] = float(self.adaptive_pl_last_regime_code)
                metrics_['client_{}_adaptive_pl_release_streak'.format(self.cid)] = float(self.adaptive_pl_release_streak)
                metrics_['client_{}_adaptive_pl_release_ready'.format(self.cid)] = float(self.adaptive_pl_last_release_ready)
                metrics_['client_{}_adaptive_pl_high_risk_flag'.format(self.cid)] = float(self.adaptive_pl_last_high_risk)
                metrics_['client_{}_adaptive_pl_gate_release_armed'.format(self.cid)] = float(self.adaptive_pl_gate_release_armed)
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
            if args.img_class == 'faz':
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
    parser.add_argument('--risk_calibration_correction_risk_threshold', type=float, default=0.55,
                        help='Clients above this risk level stay in the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_prior_gap_threshold', type=float, default=0.03,
                        help='Clients above this prior-gap EMA stay in the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_agreement_threshold', type=float, default=0.88,
                        help='Clients below this agreement are pushed toward the W4 correction regime.')
    parser.add_argument('--risk_calibration_correction_uncertain_threshold', type=float, default=0.06,
                        help='Clients above this both-uncertain EMA are pushed toward the W4 correction regime.')
    parser.add_argument('--risk_calibration_risk_agreement_credit', type=float, default=0.20,
                        help='Amount of risk reduction granted to clients with persistently high agreement.')
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
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    setattr(args, 'snapshot_path', snapshot_path)

    # Check arguments
    assert args.iters > 0
    assert args.eval_iters > 0 and (args.eval_iters % args.iters == 0)
    assert args.max_iterations > 0 and (args.max_iterations % args.eval_iters == 0)
    assert args.geometry_guided in [0, 1]
    assert args.adaptive_pl_enabled in [0, 1]
    assert args.adaptive_pl_tau_update in [0, 1]
    assert args.adaptive_pl_resume_state in [0, 1]
    assert args.geometry_num_bins == 4
    assert args.geometry_near_radius >= 0
    assert args.geometry_mid_radius >= args.geometry_near_radius
    _geometry_weights = parse_geometry_weight_list(args.geometry_pseudo_weights, args.geometry_num_bins)
    assert all(x >= 0 for x in _geometry_weights)
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
    assert args.risk_calibration_release_start_iter >= 0
    assert args.risk_calibration_release_full_iter > args.risk_calibration_release_start_iter
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
    assert 0.0 <= args.risk_calibration_release_min_correction <= args.risk_calibration_release_max_correction <= 1.0
    assert 0.0 <= args.risk_calibration_preserve_min_correction <= args.risk_calibration_preserve_max_correction <= 1.0
    assert 0.0 <= args.risk_calibration_correction_min_correction <= 1.0
    assert args.risk_calibration_release_risk_threshold <= args.risk_calibration_preserve_risk_threshold <= args.risk_calibration_correction_risk_threshold
    assert args.risk_calibration_release_agreement_threshold >= args.risk_calibration_preserve_agreement_threshold >= args.risk_calibration_correction_agreement_threshold
    assert args.risk_calibration_release_prior_gap_threshold <= args.risk_calibration_preserve_prior_gap_threshold <= args.risk_calibration_correction_prior_gap_threshold
    assert args.risk_calibration_release_uncertain_threshold <= args.risk_calibration_preserve_uncertain_threshold <= args.risk_calibration_correction_uncertain_threshold
    assert args.risk_calibration_release_min_correction <= args.risk_calibration_release_max_correction <= args.risk_calibration_preserve_min_correction <= args.risk_calibration_preserve_max_correction <= args.risk_calibration_correction_min_correction
    assert args.risk_calibration_regime_hysteresis >= 0.0
    assert args.risk_calibration_routing_mode.lower() in ['score', 'gated']
    assert args.risk_calibration_gate_release_start_iter >= 0
    if args.risk_calibration_routing_mode.lower() == 'gated':
        assert args.risk_calibration_gate_release_start_iter >= args.risk_calibration_release_start_iter
    assert args.risk_calibration_release_streak_required >= 1
    assert args.risk_calibration_release_streak_decay >= 1
    assert 0.0 <= args.risk_calibration_release_hard_ratio_threshold <= 1.0
    if args.adaptive_pl_enabled == 1:
        assert args.geometry_guided == 1
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
    assert args.img_class in ['odoc', 'faz', 'polyp']
    if args.img_class == 'faz':
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
