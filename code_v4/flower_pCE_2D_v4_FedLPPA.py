# -*- coding:utf-8 -*-
import argparse
import logging
import os
import random
import shutil
import sys
import time

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
                        get_bn_stats)
from weak_annotation_reliability import (
    build_wann_maps,
    consistency_loss as wann_consistency_loss,
    soft_band_loss as wann_soft_band_loss,
    weighted_ce_loss as wann_weighted_ce_loss,
)
from rgftd_reliability_distillation import get_rgftd_lambda, rgftd_loss, zero_rgftd_profile


def _primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def _average_rgftd_profiles(profile_a, profile_b):
    return {
        key: 0.5 * (profile_a[key] + profile_b[key])
        for key in profile_a.keys()
    }


def _scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _rgftd_class_ids_from_profile(rgftd_profile):
    class_ids = []
    if rgftd_profile is None:
        return class_ids
    for key in rgftd_profile.keys():
        if not key.startswith('teacher_class') or not key.endswith('_reliability'):
            continue
        middle = key[len('teacher_class'):-len('_reliability')]
        if middle.isdigit():
            class_ids.append(int(middle))
    return sorted(set(class_ids))


def _format_wann_rgftd_log(cid, iter_num, wann_maps, lambda_rgftd, rgftd_profile):
    parts = [
        'client %d : iteration %d : WANN/RGFTD' % (cid, iter_num),
    ]
    if wann_maps is not None:
        wann_profile = wann_maps.profile
        parts.extend([
            'wann_mass=%.4f' % _scalar_float(wann_profile.get('effective_supervision_mass', 0.0)),
            'wann_core=%.4f' % _scalar_float(wann_profile.get('core_ratio', 0.0)),
            'wann_soft=%.4f' % _scalar_float(wann_profile.get('soft_ratio', 0.0)),
            'wann_low=%.4f' % _scalar_float(wann_profile.get('low_weight_ratio', 0.0)),
            'wann_low_maxp=%.4f' % _scalar_float(wann_profile.get('max_prob_low_r', 0.0)),
        ])
    if rgftd_profile is not None:
        parts.extend([
            'rgftd_lambda=%.6f' % float(lambda_rgftd),
            'rgftd_lambda_pre=%.6f' % _scalar_float(rgftd_profile.get('lambda_pre_safety', 0.0)),
            'rgftd_lambda_safe=%.6f' % _scalar_float(rgftd_profile.get('lambda_after_safety', 0.0)),
            'rgftd_lambda_eff=%.6f' % _scalar_float(rgftd_profile.get('lambda_effective', lambda_rgftd)),
            'rgftd_core_safe=%.6f' % _scalar_float(rgftd_profile.get('core_safety_factor', 0.0)),
            'rgftd_relf=%.6f' % _scalar_float(rgftd_profile.get('release_factor', 0.0)),
            'rgftd_relq=%.6f' % _scalar_float(rgftd_profile.get('release_quality', 0.0)),
            'rgftd_relmin=%.6f' % _scalar_float(rgftd_profile.get('release_min_effective', 0.0)),
            'rgftd_loss=%.6f' % _scalar_float(rgftd_profile.get('loss', 0.0)),
            'rgftd_region=%.6f' % _scalar_float(rgftd_profile.get('region_ratio', 0.0)),
            'rgftd_active=%.6f' % _scalar_float(rgftd_profile.get('active_ratio', 0.0)),
            'rgftd_t_rel=%.6f' % _scalar_float(rgftd_profile.get('teacher_reliability', 0.0)),
            'rgftd_t_core=%.6f' % _scalar_float(rgftd_profile.get('teacher_core_agreement', 0.0)),
            'rgftd_t_core_cf=%.6f' % _scalar_float(rgftd_profile.get('teacher_core_conflict', 0.0)),
            'rgftd_t_sup=%.6f' % _scalar_float(rgftd_profile.get('teacher_support_agreement', 0.0)),
            'rgftd_t_fg_sup=%.6f' % _scalar_float(rgftd_profile.get('teacher_support_fg_recall', 0.0)),
            'rgftd_t_bg_sup=%.6f' % _scalar_float(rgftd_profile.get('teacher_support_bg_agreement', 0.0)),
            'rgftd_t_seed_sup=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_agreement', 0.0)),
            'rgftd_t_seed_fg=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_fg_recall', 0.0)),
            'rgftd_t_seed_fgp=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_fg_prob_mean', 0.0)),
            'rgftd_t_seed_fgc=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_fg_conf_mean', 0.0)),
            'rgftd_t_seed_fgm=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_fg_margin_mean', 0.0)),
            'rgftd_t_seed_bg=%.6f' % _scalar_float(rgftd_profile.get('teacher_seed_support_bg_agreement', 0.0)),
            'rgftd_t_fg=%.6f' % _scalar_float(rgftd_profile.get('teacher_foreground_ratio', 0.0)),
            'rgftd_a_fg=%.6f' % _scalar_float(rgftd_profile.get('active_foreground_ratio', 0.0)),
            'rgftd_a_bg=%.6f' % _scalar_float(rgftd_profile.get('active_background_ratio', 0.0)),
            'rgftd_bg_sup=%.6f' % _scalar_float(rgftd_profile.get('background_suppression_mean', 0.0)),
            'rgftd_veto=%.1f' % _scalar_float(rgftd_profile.get('foreground_veto_ratio', 0.0)),
            'rgftd_core_veto=%.1f' % _scalar_float(rgftd_profile.get('core_conflict_veto_ratio', 0.0)),
            'rgftd_fg_pre_px=%.1f' % _scalar_float(rgftd_profile.get('active_foreground_pixels_pre_budget', 0.0)),
            'rgftd_fg_budget=%.6f' % _scalar_float(rgftd_profile.get('foreground_budget_ratio', 0.0)),
            'rgftd_fg_px_pre=%.1f' % _scalar_float(rgftd_profile.get('active_foreground_pixels_pre_return', 0.0)),
            'rgftd_fg_px=%.1f' % _scalar_float(rgftd_profile.get('active_foreground_pixels', 0.0)),
            'rgftd_bg_pre_px=%.1f' % _scalar_float(rgftd_profile.get('active_background_pixels_pre_budget', 0.0)),
            'rgftd_bg_budget=%.6f' % _scalar_float(rgftd_profile.get('background_budget_ratio', 0.0)),
            'rgftd_bg_px_pre=%.1f' % _scalar_float(rgftd_profile.get('active_background_pixels_pre_return', 0.0)),
            'rgftd_bg_px=%.1f' % _scalar_float(rgftd_profile.get('active_background_pixels', 0.0)),
            'rgftd_bgfg=%.6f' % _scalar_float(rgftd_profile.get('background_foreground_ratio', 0.0)),
            'rgftd_bg_bal=%.6f' % _scalar_float(rgftd_profile.get('background_balance_factor', 0.0)),
            'rgftd_ret=%.1f' % _scalar_float(rgftd_profile.get('return_reason', 0.0)),
        ])
        for class_id in _rgftd_class_ids_from_profile(rgftd_profile):
            parts.extend([
                'rgftd_t_c%d=%.6f' % (
                    class_id,
                    _scalar_float(rgftd_profile.get('teacher_class{}_reliability'.format(class_id), 0.0)),
                ),
                'rgftd_t_c%d_sup=%.6f' % (
                    class_id,
                    _scalar_float(rgftd_profile.get('teacher_class{}_agreement'.format(class_id), 0.0)),
                ),
                'rgftd_t_c%d_core=%.6f' % (
                    class_id,
                    _scalar_float(rgftd_profile.get('teacher_class{}_core_agreement'.format(class_id), 0.0)),
                ),
                'rgftd_a_c%d=%.6f' % (
                    class_id,
                    _scalar_float(rgftd_profile.get('teacher_class{}_release_ratio'.format(class_id), 0.0)),
                ),
            ])
    return ', '.join(parts)


class MyClient(BaseClient):

    def __init__(self, args, model, trainloader, valloader, amp=False):
        super(MyClient, self).__init__(args, model, trainloader, valloader)
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()
        self.best_performance = 0.0


    def _train(self, config):
        self.model.train()

        # optimizer
        if self.args.strategy == 'FedRep':
            local_keys = get_fedrep_local_keys(self.args.model, self.args.in_chns, self.args.num_classes)
            decay_params, nondecay_params = [], []
            for name, param in self.model.named_parameters():
                if 'bias' in name or (name.replace('model.', '') not in local_keys):
                    nondecay_params += [param]
                else:
                    decay_params += [param]
            optimize_params = [{'params': decay_params, 'weight_decay': 0.0001},
                            {'params': nondecay_params, 'weight_decay': 0}]
            optimizer = optim.SGD(optimize_params, lr=self.current_lr,
                                momentum=0.9)
        else:
            
            optimizer = optim.AdamW(self.model.parameters(),lr=self.current_lr,betas=(0.9, 0.999),eps=1e-8,weight_decay=1e-2, amsgrad=False)

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
        wann_ref_model = None
        if int(getattr(self.args, 'wann_enabled', 0)) == 1:
            wann_ref_model = copy.deepcopy(self.model).cuda()
            wann_ref_model.eval()
            for param in wann_ref_model.parameters():
                param.requires_grad = False
        rgftd_teacher_model = None
        if int(getattr(self.args, 'rgftd_enabled', 0)) == 1:
            teacher_state_dict = getattr(self.model, 'rgftd_teacher_state_dict', None)
            if teacher_state_dict is not None:
                rgftd_teacher_model = copy.deepcopy(self.model.model).cuda()
                rgftd_teacher_model.load_state_dict(teacher_state_dict, strict=False)
                rgftd_teacher_model.eval()
                for param in rgftd_teacher_model.parameters():
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

            if self.args.img_class == 'faz' or self.args.img_class == 'prostate':
                volume_batch, label_batch = sampled_batch['image'].unsqueeze(1), sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            elif self.args.img_class == 'odoc' or self.args.img_class == 'polyp':
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

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

                outputs_soft = torch.softmax(outputs, dim=1)
                if int(getattr(self.args, 'wann_enabled', 0)) == 1:
                    with torch.no_grad():
                        ref_out = wann_ref_model(volume_batch)
                        if self.args.model == 'fcnet':
                            _, ref_logits = ref_out
                        elif self.args.model in ['deeplabv3plus', 'treefcn']:
                            ref_logits = ref_out[0]
                        elif self.args.model in [
                            'unet_head', 'unet_multihead', 'unet_lc', 'unet_uni',
                            'unet_univ2', 'unet_univ3', 'unet_univ4', 'unet_univ5'
                        ]:
                            ref_logits = ref_out[0]
                        else:
                            ref_logits = ref_out[0]
                    wann_maps = build_wann_maps(
                        image=volume_batch,
                        label=label_batch,
                        logits=outputs,
                        aux_logits=outputs_auxiliary,
                        sup_type=self.args.sup_type,
                        img_class=self.args.img_class,
                        num_classes=self.args.num_classes,
                        iter_num=self.current_iter,
                        args=self.args,
                        ref_logits=ref_logits,
                    )
                    loss_ce_seg = wann_weighted_ce_loss(
                        outputs, label_batch, wann_maps.core_weight, ignore_index=self.args.num_classes
                    )
                    loss_ce_auxiliary = wann_weighted_ce_loss(
                        outputs_auxiliary, label_batch, wann_maps.core_weight, ignore_index=self.args.num_classes
                    )
                    loss_hard = 0.5 * (loss_ce_seg + loss_ce_auxiliary)
                    lambda_soft = float(getattr(self.args, 'wann_soft_lambda', 0.2)) * ramps.sigmoid_rampup(
                        self.current_iter, int(getattr(self.args, 'wann_soft_rampup_iters', 800))
                    )
                    lambda_cons = float(getattr(self.args, 'wann_cons_lambda', 0.05)) * ramps.sigmoid_rampup(
                        self.current_iter, int(getattr(self.args, 'wann_cons_rampup_iters', 800))
                    )
                    loss_soft = wann_soft_band_loss(
                        outputs, outputs_auxiliary, label_batch, wann_maps, ignore_index=self.args.num_classes
                    )
                    loss_cons = wann_consistency_loss(outputs, outputs_auxiliary, wann_maps.ignore_mask)
                    loss_ce = loss_hard + lambda_soft * loss_soft + lambda_cons * loss_cons
                    loss_rgftd = outputs.sum() * 0.0
                    loss_rgftd_seg = outputs.sum() * 0.0
                    loss_rgftd_aux = outputs.sum() * 0.0
                    lambda_rgftd_raw = 0.0
                    lambda_rgftd = 0.0
                    rgftd_profile = zero_rgftd_profile(outputs.device)
                    if int(getattr(self.args, 'rgftd_enabled', 0)) == 1 and rgftd_teacher_model is not None:
                        lambda_rgftd_raw = get_rgftd_lambda(self.current_iter, self.args)
                        if float(lambda_rgftd_raw) > 0.0:
                            with torch.no_grad():
                                teacher_out = rgftd_teacher_model(volume_batch)
                                teacher_logits = _primary_logits(teacher_out)
                            loss_rgftd_seg, lambda_rgftd_seg, rgftd_profile_seg = rgftd_loss(
                                outputs, teacher_logits, label_batch, wann_maps, self.args, self.current_iter
                            )
                            loss_rgftd_aux, lambda_rgftd_aux, rgftd_profile_aux = rgftd_loss(
                                outputs_auxiliary, teacher_logits, label_batch, wann_maps, self.args, self.current_iter
                            )
                            loss_rgftd = 0.5 * (loss_rgftd_seg + loss_rgftd_aux)
                            lambda_rgftd = 0.5 * (float(lambda_rgftd_seg) + float(lambda_rgftd_aux))
                            rgftd_profile = _average_rgftd_profiles(rgftd_profile_seg, rgftd_profile_aux)
                            loss_ce = loss_ce + float(lambda_rgftd) * loss_rgftd
                else:
                    wann_maps = None
                    lambda_soft = 0.0
                    lambda_cons = 0.0
                    lambda_rgftd_raw = 0.0
                    lambda_rgftd = 0.0
                    loss_hard = torch.tensor(0.0).cuda()
                    loss_soft = torch.tensor(0.0).cuda()
                    loss_cons = torch.tensor(0.0).cuda()
                    loss_rgftd = torch.tensor(0.0).cuda()
                    loss_rgftd_seg = torch.tensor(0.0).cuda()
                    loss_rgftd_aux = torch.tensor(0.0).cuda()
                    rgftd_profile = zero_rgftd_profile(loss_rgftd.device)
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
                    # pseudo_alpha = 0.75 # np.random.random()
                    pseudo_alpha = np.random.uniform(0, 1)
                    pseudo_label = pseudo_alpha * outputs_soft.detach() + (1 - pseudo_alpha) * outputs_soft_auxiliary.detach()
                    pseudo_label = torch.argmax(pseudo_label, dim=1)
                    # print(self.cid, pseudo_alpha, torch.unique(pseudo_label))
                    loss_pls_1 = dice_loss(outputs_soft, pseudo_label.unsqueeze(1))
                    loss_pls_2 = dice_loss(outputs_soft_auxiliary, pseudo_label.unsqueeze(1))
                    loss_pls = (loss_pls_1 + loss_pls_2) / 2
                    loss = torch.add(loss, loss_pls, alpha=self.args.beta)
                    if i_iter > 0:
                        distance_prompts = l1_loss(previous_prompts.detach(), prompts[self.args.cid].clone().detach())
                        distance_prompts_dis = l1_loss(previous_prompts_dis.detach(), distribution_prompts[self.args.cid].clone().detach())
                        distance_prompts_uni = l1_loss(previous_prompts_uni.detach(), uni_prompts.clone().detach())
                    previous_prompts = prompts[self.args.cid].detach()
                    previous_prompts_dis = distribution_prompts[self.args.cid].detach()
                    previous_prompts_uni = uni_prompts.detach()

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
            if int(getattr(self.args, 'wann_enabled', 0)) == 1 and self.current_iter % int(self.args.eval_iters) == 0:
                log(INFO, _format_wann_rgftd_log(
                    self.cid,
                    self.current_iter,
                    wann_maps,
                    lambda_rgftd_raw,
                    rgftd_profile,
                ))

            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_

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
            if int(getattr(self.args, 'ala_max_epochs', 0)) > 0:
                metrics_['client_{}_ala_epochs'.format(self.cid)] = float(getattr(self.model, 'ala_last_epochs', 0))
                metrics_['client_{}_ala_final_std'.format(self.cid)] = float(getattr(self.model, 'ala_last_std', 0.0))
                metrics_['client_{}_ala_hit_max_epochs'.format(self.cid)] = float(getattr(self.model, 'ala_hit_max_epochs', 0))

        if int(getattr(self.args, 'wann_enabled', 0)) == 1 and wann_maps is not None:
            metrics_['client_{}_wann_loss_hard'.format(self.cid)] = loss_hard.item()
            metrics_['client_{}_wann_loss_soft'.format(self.cid)] = loss_soft.item()
            metrics_['client_{}_wann_loss_cons'.format(self.cid)] = loss_cons.item()
            metrics_['client_{}_wann_lambda_soft'.format(self.cid)] = float(lambda_soft)
            metrics_['client_{}_wann_lambda_cons'.format(self.cid)] = float(lambda_cons)
            for key, value in wann_maps.profile.items():
                metrics_['client_{}_wann_{}'.format(self.cid, key)] = float(value.detach().cpu().item())

        if int(getattr(self.args, 'rgftd_enabled', 0)) == 1:
            metrics_['client_{}_rgftd_loss'.format(self.cid)] = float(loss_rgftd.detach().cpu().item())
            metrics_['client_{}_rgftd_loss_seg'.format(self.cid)] = float(loss_rgftd_seg.detach().cpu().item())
            metrics_['client_{}_rgftd_loss_aux'.format(self.cid)] = float(loss_rgftd_aux.detach().cpu().item())
            metrics_['client_{}_rgftd_lambda'.format(self.cid)] = float(lambda_rgftd_raw)
            for key, value in rgftd_profile.items():
                if key in ['loss', 'lambda']:
                    continue
                metrics_['client_{}_rgftd_{}'.format(self.cid, key)] = float(value.detach().cpu().item())
            for class_id in range(1, self.args.num_classes):
                for suffix in ['agreement', 'core_agreement', 'reliability', 'release_ratio']:
                    profile_key = 'teacher_class{}_{}'.format(class_id, suffix)
                    metrics_key = 'client_{}_rgftd_{}'.format(self.cid, profile_key)
                    if metrics_key not in metrics_:
                        metrics_[metrics_key] = float(_scalar_float(rgftd_profile.get(profile_key, 0.0)))

        return loss.item(), metrics_


from flower_common import PretrainDataset
from torchvision.utils import make_grid
from tqdm import tqdm
def pretrain_model(args, writer, worker_init_fn):
    db_train = PretrainDataset(args.root_path, args.sup_type_list, transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]))
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

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
            if args.img_class == 'faz' or args.img_class == 'prostate':
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
    parser = argparse.ArgumentParser()
    ## flower related arguments
    parser.add_argument('--server_address', type=str,
                        default='[::]:8080', help='gRPC server address (default: [::]:8080)')
    parser.add_argument('--gpu', type=int,
                        required=True, help='GPU index')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
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
    parser.add_argument('--ala_max_epochs', type=int, default=0,
                        help='Maximum ALA initialization epochs; <=0 keeps the original unbounded behavior')
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
    parser.add_argument('--wann_enabled', type=int, default=0,
                        help='Enable weak annotation reliability reconstruction')
    parser.add_argument('--wann_core_thresh', type=float, default=0.65,
                        help='Reliability threshold for hard-supervised core pixels')
    parser.add_argument('--wann_soft_thresh', type=float, default=0.25,
                        help='Reliability threshold for soft-band pixels')
    parser.add_argument('--wann_core_min_weight', type=float, default=0.8,
                        help='Minimum hard-supervision weight for core pixels')
    parser.add_argument('--wann_r_max', type=float, default=1.2,
                        help='Maximum reliability weight')
    parser.add_argument('--wann_dilated_support_score', type=float, default=0.55,
                        help='Annotation score assigned to non-core pixels near weak support')
    parser.add_argument('--wann_appearance_temp', type=float, default=1.5,
                        help='Temperature for image-appearance consistency')
    parser.add_argument('--wann_texture_kernel_size', type=int, default=5,
                        help='Local window size for WANN texture ambiguity penalty')
    parser.add_argument('--wann_texture_temp', type=float, default=1.0,
                        help='Temperature for WANN local texture ambiguity penalty')
    parser.add_argument('--wann_texture_weight', type=float, default=0.25,
                        help='Weight of WANN local texture stability in appearance reliability')
    parser.add_argument('--wann_pred_start_iter', type=int, default=800,
                        help='Iteration to start using prediction ambiguity in WANN')
    parser.add_argument('--wann_entropy_weight', type=float, default=0.5,
                        help='Entropy penalty weight in WANN reliability')
    parser.add_argument('--wann_agreement_weight', type=float, default=0.5,
                        help='Main-aux agreement weight in WANN reliability')
    parser.add_argument('--wann_global_agreement_weight', type=float, default=0.5,
                        help='Current-reference prediction agreement weight in WANN reliability')
    parser.add_argument('--wann_keypoint_soft_radius', type=int, default=2,
                        help='Soft-band radius for keypoint annotations')
    parser.add_argument('--wann_scribble_soft_radius', type=int, default=4,
                        help='Soft-band radius for scribble annotations')
    parser.add_argument('--wann_box_soft_radius', type=int, default=2,
                        help='Soft-band radius for box/block annotations')
    parser.add_argument('--wann_mask_soft_radius', type=int, default=1,
                        help='Soft-band radius for mask annotations')
    parser.add_argument('--wann_seed_support_erode_radius', type=int, default=1,
                        help='Default erosion radius used to build conservative seed support for box/block supervision')
    parser.add_argument('--wann_seed_support_box_erode_radius', type=int, default=1,
                        help='Erosion radius used to build conservative seed support for box supervision')
    parser.add_argument('--wann_seed_support_block_erode_radius', type=int, default=1,
                        help='Erosion radius used to build conservative seed support for block supervision')
    parser.add_argument('--wann_soft_lambda', type=float, default=0.2,
                        help='Maximum WANN soft-band loss weight')
    parser.add_argument('--wann_cons_lambda', type=float, default=0.05,
                        help='Maximum WANN ignore-region consistency weight')
    parser.add_argument('--wann_soft_rampup_iters', type=int, default=800,
                        help='Rampup iterations for WANN soft-band loss')
    parser.add_argument('--wann_cons_rampup_iters', type=int, default=800,
                        help='Rampup iterations for WANN consistency loss')
    parser.add_argument('--rgftd_enabled', type=int, default=0,
                        help='Enable reliability-gated federated teacher distillation')
    parser.add_argument('--rgftd_lambda', type=float, default=0.1,
                        help='Maximum RGFTD distillation loss weight')
    parser.add_argument('--rgftd_warmup_iters', type=int, default=800,
                        help='Iterations before RGFTD loss is released')
    parser.add_argument('--rgftd_rampup_iters', type=int, default=800,
                        help='Rampup iterations after RGFTD warmup')
    parser.add_argument('--rgftd_teacher_ema_decay', type=float, default=0.99,
                        help='Server EMA decay for RGFTD teacher')
    parser.add_argument('--rgftd_teacher_conf_thresh', type=float, default=0.90,
                        help='Teacher confidence threshold for RGFTD pixels')
    parser.add_argument('--rgftd_teacher_foreground_radius', type=int, default=2,
                        help='Foreground dilation radius for RGFTD teacher gating')
    parser.add_argument('--rgftd_min_foreground_pixels', type=int, default=8,
                        help='Minimum active foreground pixels required to keep RGFTD active')
    parser.add_argument('--rgftd_min_foreground_ratio', type=float, default=0.05,
                        help='Minimum active foreground ratio required to keep RGFTD active')
    parser.add_argument('--rgftd_teacher_fg_prob_thresh', type=float, default=0.35,
                        help='Teacher foreground probability threshold for RGFTD foreground anchoring')
    parser.add_argument('--rgftd_teacher_fg_topk_ratio', type=float, default=0.002,
                        help='Top-k ratio for fallback RGFTD foreground anchors inside WANN non-core regions')
    parser.add_argument('--rgftd_teacher_fg_topk_min_pixels', type=int, default=8,
                        help='Minimum fallback foreground-anchor pixels for RGFTD')
    parser.add_argument('--rgftd_teacher_student_fg_margin', type=float, default=0.05,
                        help='Teacher-student foreground probability margin for RGFTD foreground correction')
    parser.add_argument('--rgftd_teacher_bg_conf_thresh', type=float, default=0.98,
                        help='Teacher confidence threshold for accepted background pixels')
    parser.add_argument('--rgftd_bg_max_fg_prob', type=float, default=0.15,
                        help='Maximum teacher foreground probability allowed for accepted background pixels')
    parser.add_argument('--rgftd_student_conf_thresh', type=float, default=0.80,
                        help='Student maximum-confidence threshold for uncertainty gate')
    parser.add_argument('--rgftd_student_entropy_thresh', type=float, default=0.35,
                        help='Student normalized-entropy threshold for uncertainty gate')
    parser.add_argument('--rgftd_low_r_thresh', type=float, default=0.25,
                        help='WANN reliability threshold for RGFTD low-reliability region')
    parser.add_argument('--rgftd_temperature', type=float, default=1.0,
                        help='Temperature for RGFTD KL distillation')
    parser.add_argument('--rgftd_use_soft_band', type=int, default=0,
                        help='Whether RGFTD can also use WANN soft-band pixels')
    parser.add_argument('--rgftd_background_weight', type=float, default=0.25,
                        help='Relative weight for high-confidence teacher background pixels')
    parser.add_argument('--rgftd_skip_background_only', type=int, default=1,
                        help='Skip RGFTD when no active foreground pixels are present')
    parser.add_argument('--rgftd_teacher_validation_enabled', type=int, default=0,
                        help='Enable RGFTD-v2 teacher reliability validation and release veto')
    parser.add_argument('--rgftd_teacher_reliability_min', type=float, default=0.55,
                        help='Legacy reliability threshold kept for RGFTD-v2 audit compatibility')
    parser.add_argument('--rgftd_teacher_core_agree_floor', type=float, default=0.80,
                        help='Agreement floor for teacher validation on WANN core pixels')
    parser.add_argument('--rgftd_teacher_support_agree_floor', type=float, default=0.70,
                        help='Agreement floor for teacher validation on weak-support foreground pixels')
    parser.add_argument('--rgftd_teacher_support_prob_floor', type=float, default=0.35,
                        help='Foreground-probability floor used in RGFTD-v2 teacher reliability support gate')
    parser.add_argument('--rgftd_teacher_conf_floor', type=float, default=0.85,
                        help='Confidence floor used when building RGFTD-v2 teacher reliability')
    parser.add_argument('--rgftd_teacher_class_reliability_min', type=float, default=0.50,
                        help='Minimum class-wise reliability required to release a teacher class')
    parser.add_argument('--rgftd_teacher_max_core_conflict', type=float, default=0.20,
                        help='Maximum tolerated teacher conflict ratio on WANN core before veto')
    parser.add_argument('--rgftd_teacher_score_core_weight', type=float, default=0.35,
                        help='Weight of core agreement in teacher reliability score')
    parser.add_argument('--rgftd_teacher_score_support_weight', type=float, default=0.30,
                        help='Weight of support foreground agreement in teacher reliability score')
    parser.add_argument('--rgftd_teacher_score_class_weight', type=float, default=0.25,
                        help='Weight of class-wise validation in teacher reliability score')
    parser.add_argument('--rgftd_teacher_score_conf_weight', type=float, default=0.10,
                        help='Weight of teacher confidence in teacher reliability score')
    parser.add_argument('--rgftd_teacher_release_prob_floor', type=float, default=0.35,
                        help='Foreground-probability floor for soft RGFTD-v2 release')
    parser.add_argument('--rgftd_teacher_release_margin_floor', type=float, default=0.05,
                        help='Foreground-vs-background margin floor for soft RGFTD-v2 release')
    parser.add_argument('--rgftd_teacher_release_class_floor', type=float, default=0.50,
                        help='Class-reliability floor for soft RGFTD-v2 release')
    parser.add_argument('--rgftd_teacher_release_min', type=float, default=0.03,
                        help='Maximum adaptive floor for RGFTD-v2 release after quality and schedule scaling')
    parser.add_argument('--rgftd_active_fg_topk_ratio', type=float, default=0.002,
                        help='Top-k ratio used to cap active RGFTD foreground pixels after gating')
    parser.add_argument('--rgftd_active_fg_topk_min_pixels', type=int, default=8,
                        help='Minimum active RGFTD foreground pixels kept after budget pruning')
    parser.add_argument('--rgftd_active_fg_topk_max_pixels', type=int, default=4096,
                        help='Maximum active RGFTD foreground pixels kept after budget pruning')
    parser.add_argument('--rgftd_max_bg_fg_ratio', type=float, default=1.0,
                        help='Maximum allowed active background-to-foreground pixel ratio for RGFTD')
    parser.add_argument('--rgftd_allow_bg_without_fg', type=int, default=0,
                        help='Whether RGFTD may keep background pixels when no foreground pixels survive')
    parser.add_argument('--rgftd_lambda_eff_cap', type=float, default=0.02,
                        help='Upper bound on effective RGFTD lambda after release and safety scaling')
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

    if args.strategy in ['FedLC', 'FedALALC', 'FedAPLC', 'FedUni', 'FedUniV2', 'FedUniV2.1']:
        assert args.tsne_iters >= 0
        if args.tsne_iters > 0:
            assert args.tsne_iters % args.iters == 0

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

    assert args.role in ['server', 'client']
    assert args.img_class in ['odoc', 'faz', 'polyp', 'prostate']
    if args.img_class == 'faz':
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']
    else:
        assert args.sup_type in ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']
    assert args.wann_enabled in [0, 1]
    assert args.ala_max_epochs >= 0
    assert 0.0 <= args.wann_soft_thresh <= args.wann_core_thresh <= args.wann_r_max
    assert 0.0 <= args.wann_core_min_weight <= args.wann_r_max
    assert 0.0 <= args.wann_dilated_support_score <= 1.0
    assert args.wann_appearance_temp > 0.0
    assert args.wann_texture_kernel_size >= 1
    assert args.wann_texture_temp > 0.0
    assert 0.0 <= args.wann_texture_weight <= 1.0
    assert args.wann_pred_start_iter >= 0
    assert 0.0 <= args.wann_entropy_weight <= 1.0
    assert 0.0 <= args.wann_agreement_weight <= 1.0
    assert 0.0 <= args.wann_global_agreement_weight <= 1.0
    assert args.wann_keypoint_soft_radius >= 0
    assert args.wann_scribble_soft_radius >= 0
    assert args.wann_box_soft_radius >= 0
    assert args.wann_mask_soft_radius >= 0
    assert args.wann_seed_support_erode_radius >= 0
    assert args.wann_seed_support_box_erode_radius >= 0
    assert args.wann_seed_support_block_erode_radius >= 0
    assert args.wann_soft_lambda >= 0.0
    assert args.wann_cons_lambda >= 0.0
    assert args.wann_soft_rampup_iters >= 0
    assert args.wann_cons_rampup_iters >= 0
    assert args.rgftd_enabled in [0, 1]
    if args.rgftd_enabled == 1:
        assert args.wann_enabled == 1
        assert args.strategy in ['FedUniV2', 'FedUniV2.1']
    assert args.rgftd_lambda >= 0.0
    assert args.rgftd_warmup_iters >= 0
    assert args.rgftd_rampup_iters >= 0
    assert 0.0 <= args.rgftd_teacher_ema_decay < 1.0
    assert 0.0 <= args.rgftd_teacher_conf_thresh <= 1.0
    assert args.rgftd_teacher_foreground_radius >= 0
    assert args.rgftd_min_foreground_pixels >= 0
    assert 0.0 <= args.rgftd_min_foreground_ratio <= 1.0
    assert 0.0 <= args.rgftd_teacher_fg_prob_thresh <= 1.0
    assert 0.0 <= args.rgftd_teacher_fg_topk_ratio <= 1.0
    assert args.rgftd_teacher_fg_topk_min_pixels >= 0
    assert args.rgftd_teacher_student_fg_margin >= 0.0
    assert 0.0 <= args.rgftd_teacher_bg_conf_thresh <= 1.0
    assert 0.0 <= args.rgftd_bg_max_fg_prob <= 1.0
    assert 0.0 <= args.rgftd_student_conf_thresh <= 1.0
    assert 0.0 <= args.rgftd_student_entropy_thresh <= 1.0
    assert 0.0 <= args.rgftd_low_r_thresh <= args.wann_r_max
    assert args.rgftd_temperature > 0.0
    assert args.rgftd_use_soft_band in [0, 1]
    assert 0.0 <= args.rgftd_background_weight <= 1.0
    assert args.rgftd_skip_background_only in [0, 1]
    assert args.rgftd_teacher_validation_enabled in [0, 1]
    assert 0.0 <= args.rgftd_teacher_reliability_min <= 1.0
    assert 0.0 <= args.rgftd_teacher_core_agree_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_support_agree_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_support_prob_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_conf_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_class_reliability_min <= 1.0
    assert 0.0 <= args.rgftd_teacher_max_core_conflict <= 1.0
    assert 0.0 <= args.rgftd_teacher_release_prob_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_release_margin_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_release_class_floor <= 1.0
    assert 0.0 <= args.rgftd_teacher_release_min <= 1.0
    assert 0.0 <= args.rgftd_active_fg_topk_ratio <= 1.0
    assert args.rgftd_active_fg_topk_min_pixels >= 0
    assert args.rgftd_active_fg_topk_max_pixels >= 0
    assert args.rgftd_active_fg_topk_max_pixels == 0 or args.rgftd_active_fg_topk_max_pixels >= args.rgftd_active_fg_topk_min_pixels
    assert args.rgftd_max_bg_fg_ratio >= 0.0
    assert args.rgftd_allow_bg_without_fg in [0, 1]
    assert args.rgftd_lambda_eff_cap >= 0.0
    assert args.rgftd_teacher_score_core_weight >= 0.0
    assert args.rgftd_teacher_score_support_weight >= 0.0
    assert args.rgftd_teacher_score_class_weight >= 0.0
    assert args.rgftd_teacher_score_conf_weight >= 0.0

    # Configure logger
    if args.role == 'server':
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code',
                        shutil.ignore_patterns(['.git', '__pycache__']))
        fl.common.logger.configure('server', filename=os.path.join(snapshot_path, 'server.log'))
        writer = SummaryWriter(snapshot_path + '/log')
    else:
        fl.common.logger.configure('client_{}'.format(args.cid), filename=os.path.join(snapshot_path, 'client_{}.log'.format(args.cid)))

    log(INFO, 'Arguments: {}'.format(args))

    # Load model and data
    db_train = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type, img_class=args.img_class)
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split='val', img_class=args.img_class)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    drop_last = True if args.strategy in ['FedUni', 'FedUniV2', 'FedUniV2.1'] else False
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=drop_last)
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
        def fit_config(server_round):
            config = {
                'iter_global': server_round,
                'iters': args.iters,
                'eval_iters': args.eval_iters,
                'batch_size': args.batch_size,
                'stage': 'fit'
            }
            return config

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
        if args.wann_enabled == 1:
            train_scalar_metrics += [
                'wann_loss_hard',
                'wann_loss_soft',
                'wann_loss_cons',
                'wann_lambda_soft',
                'wann_lambda_cons',
                'wann_effective_supervision_mass',
                'wann_core_ratio',
                'wann_soft_ratio',
                'wann_low_weight_ratio',
                'wann_mean_reliability',
                'wann_entropy_low_r',
                'wann_max_prob_low_r',
                'wann_foreground_ratio_low_r',
                'wann_update_norm',
                'wann_update_cos_loo',
                'wann_update_conflict',
            ]
        if args.rgftd_enabled == 1:
            train_scalar_metrics += [
                'rgftd_loss',
                'rgftd_loss_seg',
                'rgftd_loss_aux',
                'rgftd_lambda',
                'rgftd_lambda_pre_safety',
                'rgftd_lambda_after_safety',
                'rgftd_lambda_effective',
                'rgftd_core_safety_factor',
                'rgftd_candidate_ratio',
                'rgftd_active_ratio',
                'rgftd_region_ratio',
                'rgftd_teacher_accept_ratio',
                'rgftd_student_uncertain_ratio',
                'rgftd_teacher_conf_mean',
                'rgftd_teacher_reliability',
                'rgftd_teacher_core_agreement',
                'rgftd_teacher_core_conflict',
                'rgftd_teacher_core_conf_mean',
                'rgftd_teacher_support_agreement',
                'rgftd_teacher_support_conflict',
                'rgftd_teacher_support_conf_mean',
                'rgftd_teacher_support_fg_recall',
                'rgftd_teacher_support_bg_agreement',
                'rgftd_teacher_seed_support_agreement',
                'rgftd_teacher_seed_support_conflict',
                'rgftd_teacher_seed_support_conf_mean',
                'rgftd_teacher_seed_support_fg_recall',
                'rgftd_teacher_seed_support_fg_prob_mean',
                'rgftd_teacher_seed_support_fg_conf_mean',
                'rgftd_teacher_seed_support_fg_margin_mean',
                'rgftd_teacher_seed_support_bg_agreement',
                'rgftd_core_conflict_veto_ratio',
                'rgftd_student_conf_mean',
                'rgftd_student_entropy_mean',
                'rgftd_kl_mean',
                'rgftd_weight_mean',
                'rgftd_release_factor',
                'rgftd_release_raw',
                'rgftd_release_quality',
                'rgftd_release_min_effective',
                'rgftd_release_prob_gate',
                'rgftd_release_conf_gate',
                'rgftd_release_class_gate',
                'rgftd_teacher_foreground_ratio',
                'rgftd_active_foreground_ratio',
                'rgftd_active_background_ratio',
                'rgftd_foreground_veto_ratio',
                'rgftd_active_foreground_pixels_pre_budget',
                'rgftd_active_background_pixels_pre_budget',
                'rgftd_foreground_budget_ratio',
                'rgftd_background_budget_ratio',
                'rgftd_background_foreground_ratio',
                'rgftd_background_balance_factor',
                'rgftd_active_foreground_pixels_pre_return',
                'rgftd_active_background_pixels_pre_return',
                'rgftd_active_foreground_pixels',
                'rgftd_active_background_pixels',
                'rgftd_background_suppression_mean',
                'rgftd_return_reason',
            ]
            for class_id in range(1, args.num_classes):
                train_scalar_metrics += [
                    'rgftd_teacher_class{}_agreement'.format(class_id),
                    'rgftd_teacher_class{}_core_agreement'.format(class_id),
                    'rgftd_teacher_class{}_reliability'.format(class_id),
                    'rgftd_teacher_class{}_release_ratio'.format(class_id),
                ]
        if args.ala_max_epochs > 0:
            train_scalar_metrics += [
                'ala_epochs',
                'ala_final_std',
                'ala_hit_max_epochs',
            ]
        train_image_metrics = ['Image', 'Prediction', 'GroundTruth']
        if args.strategy in ['FedUniV2', 'FedUniV2.1']:
            train_image_metrics += ['Prediction2', 'Pseudo']
        val_metrics = VAL_METRICS
        server = MyServer(
            args=args, writer=writer, state_dict_keys=state_dict_keys, train_scalar_metrics=train_scalar_metrics,
            train_image_metrics=train_image_metrics, val_metrics=val_metrics, client_manager=SimpleClientManager(), strategy=strategy
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
