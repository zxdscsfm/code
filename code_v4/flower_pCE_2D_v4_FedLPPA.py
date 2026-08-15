# -*- coding:utf-8 -*-
import argparse
import io
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import flwr as fl
from flwr.common.logger import log
from flwr.server import ServerConfig


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_figure(self, *args, **kwargs):
        pass

    def close(self):
        pass


def build_summary_writer(args, log_dir):
    if int(getattr(args, 'disable_tensorboard', 0)) == 1:
        return NullSummaryWriter()
    return SummaryWriter(log_dir)
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
from annotation_geometry_calibration import (
    acg_loss,
    complete_wann_maps_with_agc,
    zero_acg_profile,
)
from rgftd_reliability_distillation import (
    get_rgftd_lambda,
    rdsi_feature_prototype_loss,
    rgftd_loss,
    select_rdsi_teacher_logits,
    zero_rgftd_profile,
)


def _primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def _aux_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (list, tuple)) and len(model_out) >= 9:
        return model_out[8]
    return model_out[0]


def _rdsi_feature(model_out):
    if not isinstance(model_out, (list, tuple)) or len(model_out) < 3:
        raise ValueError('RDSI feature transfer requires a model output with decoder feature de1 at index 2')
    feature = model_out[2]
    if not torch.is_tensor(feature) or feature.dim() != 4:
        raise ValueError('RDSI feature transfer requires a 4D decoder feature map')
    return feature


def _average_rgftd_profiles(profile_a, profile_b):
    return {
        key: 0.5 * (
            profile_a.get(key, torch.tensor(0.0, device=next(iter(profile_a.values())).device))
            + profile_b.get(key, torch.tensor(0.0, device=next(iter(profile_b.values())).device))
        )
        for key in set(profile_a.keys()) | set(profile_b.keys())
    }


def _scalar_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _normalize_gate_scalar(score, floor):
    floor = float(max(min(floor, 0.999999), 0.0))
    return max(0.0, min((float(score) - floor) / max(1.0 - floor, 1e-6), 1.0))


def _pack_rgftd_stable_snapshot(arrays):
    buffer = io.BytesIO()
    payload = {'arr_{}'.format(idx): array for idx, array in enumerate(arrays)}
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()


def _default_rgftd_v3_status():
    return {
        'v3_pool_size': 0.0,
        'v3_pool_stable': 0.0,
        'v3_no_teacher': 1.0,
        'v3_fallback_teacher': 0.0,
        'v3_teacher_source': 0.0,
        'v3_selected_teacher': -1.0,
        'v3_selected_score': 0.0,
        'v3_selected_stable_score': 0.0,
        'v3_best_failed_teacher': -1.0,
        'v3_best_failed_score': 0.0,
        'v3_routing_score': 0.0,
        'v3_audit_seed_fg_prob_mean': 0.0,
        'v3_audit_seed_fg_margin_mean': 0.0,
        'v3_audit_seed_fg_recall': 0.0,
        'v3_audit_core_conflict': 0.0,
        'v3_audit_core_valid': 0.0,
        'v3_audit_teacher_reliability': 0.0,
        'v3_audit_release_factor': 0.0,
        'v3_lease_active': 0.0,
        'v3_lease_age': 0.0,
        'v3_benefit_score': 1.0,
        'v3_window_score': 1.0,
        'v3_stability_score': 1.0,
        'v3_teacher_renew': 0.0,
        'v3_teacher_revoke': 0.0,
        'v3_teacher_decay': 0.0,
        'v3_cap_current': 0.0,
        'v3_cap_scale': 1.0,
        'v3_ret0_ratio': 0.0,
        'v3_cap_hit_ratio': 0.0,
        'v3_bgfg_window': 0.0,
        'v3_lowmaxp_delta': 0.0,
        'v3_wann_mass_delta': 0.0,
        'stable_valid': 0.0,
        'stable_score': 0.0,
        'stable_best_score': 0.0,
        'stable_updated': 0.0,
    }


def _attach_rgftd_v3_status(profile, status, device):
    merged = dict(profile)
    full_status = _default_rgftd_v3_status()
    if status is not None:
        full_status.update(status)
    for key, value in full_status.items():
        merged[key] = torch.tensor(float(value), device=device)
    return merged


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


def _format_wann_rgftd_log(cid, iter_num, wann_maps, lambda_rgftd, rgftd_profile, acg_profile=None):
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
            'wann_seed=%.4f' % _scalar_float(wann_profile.get('seed_support_ratio', 0.0)),
            'wann_core_cand=%.4f' % _scalar_float(wann_profile.get('core_candidate_ratio', 0.0)),
            'wann_sparse_proto=%.1f' % _scalar_float(wann_profile.get('sparse_seed_protocol', 0.0)),
            'wann_block_proto=%.1f' % _scalar_float(wann_profile.get('block_like_protocol', 0.0)),
            'wann_low_maxp=%.4f' % _scalar_float(wann_profile.get('max_prob_low_r', 0.0)),
        ])
    if acg_profile is not None:
        parts.extend([
            'acg_enabled=%.1f' % _scalar_float(acg_profile.get('enabled', 0.0)),
            'acg_lambda=%.6f' % _scalar_float(acg_profile.get('lambda', 0.0)),
            'acg_loss=%.6f' % _scalar_float(acg_profile.get('loss', 0.0)),
            'acg_core_miss=%.6f' % _scalar_float(acg_profile.get('core_miss_loss', 0.0)),
            'acg_support_miss=%.6f' % _scalar_float(acg_profile.get('support_miss_loss', 0.0)),
            'acg_range=%.6f' % _scalar_float(acg_profile.get('range_loss', 0.0)),
            'acg_range_under=%.6f' % _scalar_float(acg_profile.get('range_under_loss', 0.0)),
            'acg_range_over=%.6f' % _scalar_float(acg_profile.get('range_over_loss', 0.0)),
            'acg_leak=%.6f' % _scalar_float(acg_profile.get('unsupported_leak_loss', 0.0)),
            'acg_boundary=%.6f' % _scalar_float(acg_profile.get('boundary_loss', 0.0)),
            'acg_shape=%.6f' % _scalar_float(acg_profile.get('shape_contrast_loss', 0.0)),
            'acg_core_ratio=%.6f' % _scalar_float(acg_profile.get('core_ratio', 0.0)),
            'acg_support_ratio=%.6f' % _scalar_float(acg_profile.get('support_ratio', 0.0)),
            'acg_unsupported_ratio=%.6f' % _scalar_float(acg_profile.get('unsupported_ratio', 0.0)),
            'acg_boundary_ratio=%.6f' % _scalar_float(acg_profile.get('boundary_ratio', 0.0)),
            'acg_shape_ring_ratio=%.6f' % _scalar_float(acg_profile.get('shape_ring_ratio', 0.0)),
            'acg_shape_anchor_ratio=%.6f' % _scalar_float(acg_profile.get('shape_anchor_ratio', 0.0)),
            'acg_core_mass=%.6f' % _scalar_float(acg_profile.get('core_mass', 0.0)),
            'acg_envelope_mass=%.6f' % _scalar_float(acg_profile.get('envelope_mass', 0.0)),
            'acg_lower_mass=%.6f' % _scalar_float(acg_profile.get('lower_mass', 0.0)),
            'acg_upper_mass=%.6f' % _scalar_float(acg_profile.get('upper_mass', 0.0)),
            'acg_uncertain_mass=%.6f' % _scalar_float(acg_profile.get('uncertain_mass', 0.0)),
            'acg_rel_mean=%.6f' % _scalar_float(acg_profile.get('reliability_mean', 0.0)),
            'acg_pred_fg=%.6f' % _scalar_float(acg_profile.get('pred_fg_mass', 0.0)),
            'acg_shape_ring_fg=%.6f' % _scalar_float(acg_profile.get('pred_fg_shape_ring_mean', 0.0)),
            'acg_shape_anchor_fg=%.6f' % _scalar_float(acg_profile.get('pred_fg_shape_anchor_mean', 0.0)),
            'acg_core_w=%.6f' % _scalar_float(acg_profile.get('core_weight_scale', 0.0)),
            'acg_support_w=%.6f' % _scalar_float(acg_profile.get('support_weight_scale', 0.0)),
            'acg_range_w=%.6f' % _scalar_float(acg_profile.get('range_weight_scale', 0.0)),
            'acg_leak_w=%.6f' % _scalar_float(acg_profile.get('leak_weight_scale', 0.0)),
            'acg_boundary_w=%.6f' % _scalar_float(acg_profile.get('boundary_weight_scale', 0.0)),
            'acg_shape_w=%.6f' % _scalar_float(acg_profile.get('shape_weight_scale', 0.0)),
            'acg_fg_share=%.6f' % _scalar_float(acg_profile.get('fg_violation_share', 0.0)),
            'acg_range_share=%.6f' % _scalar_float(acg_profile.get('range_violation_share', 0.0)),
            'acg_leak_share=%.6f' % _scalar_float(acg_profile.get('leak_violation_share', 0.0)),
            'acg_boundary_share=%.6f' % _scalar_float(acg_profile.get('boundary_violation_share', 0.0)),
            'acg_shape_share=%.6f' % _scalar_float(acg_profile.get('shape_violation_share', 0.0)),
            'acg_nwr_fg_loss=%.6f' % _scalar_float(acg_profile.get('nwr_fg_loss', 0.0)),
            'acg_nwr_seed_fg_loss=%.6f' % _scalar_float(acg_profile.get('nwr_seed_fg_loss', 0.0)),
            'acg_nwr_context_fg_loss=%.6f' % _scalar_float(acg_profile.get('nwr_context_fg_loss', 0.0)),
            'acg_nwr_bg_loss=%.6f' % _scalar_float(acg_profile.get('nwr_bg_loss', 0.0)),
            'acg_nwr_context_fg_target=%.6f' % _scalar_float(acg_profile.get('nwr_context_fg_target', 0.0)),
            'acg_nwr_context_fg_upper_target=%.6f' % _scalar_float(
                acg_profile.get('nwr_context_fg_upper_target', 0.0)
            ),
            'acg_nwr_context_fg_margin_gap=%.6f' % _scalar_float(
                acg_profile.get('nwr_context_fg_margin_gap', 0.0)
            ),
            'acg_nwr_context_fg_over_gap=%.6f' % _scalar_float(
                acg_profile.get('nwr_context_fg_over_gap', 0.0)
            ),
            'acg_nwr_fg_mass=%.6f' % _scalar_float(acg_profile.get('nwr_fg_weight_mass', 0.0)),
            'acg_nwr_seed_fg_mass=%.6f' % _scalar_float(acg_profile.get('nwr_seed_fg_weight_mass', 0.0)),
            'acg_nwr_context_fg_mass=%.6f' % _scalar_float(acg_profile.get('nwr_context_fg_weight_mass', 0.0)),
            'acg_nwr_bg_mass=%.6f' % _scalar_float(acg_profile.get('nwr_bg_weight_mass', 0.0)),
            'acg_nwr_fg_ratio=%.6f' % _scalar_float(acg_profile.get('nwr_fg_region_ratio', 0.0)),
            'acg_nwr_seed_fg_ratio=%.6f' % _scalar_float(acg_profile.get('nwr_seed_fg_region_ratio', 0.0)),
            'acg_nwr_context_fg_ratio=%.6f' % _scalar_float(acg_profile.get('nwr_context_fg_region_ratio', 0.0)),
            'acg_nwr_bg_ratio=%.6f' % _scalar_float(acg_profile.get('nwr_bg_region_ratio', 0.0)),
            'acg_nwr_regions=%.1f' % _scalar_float(acg_profile.get('nwr_region_count', 0.0)),
            'acg_nwr_fg_prior=%.6f' % _scalar_float(acg_profile.get('nwr_fg_prior', 0.0)),
            'acg_nwr_seed_fg_prior=%.6f' % _scalar_float(acg_profile.get('nwr_seed_fg_prior', 0.0)),
            'acg_nwr_context_fg_prior=%.6f' % _scalar_float(acg_profile.get('nwr_context_fg_prior', 0.0)),
            'acg_nwr_bg_prior=%.6f' % _scalar_float(acg_profile.get('nwr_bg_prior', 0.0)),
            'acg_nwr_prior_mode_id=%.1f' % _scalar_float(acg_profile.get('nwr_prior_mode_id', 0.0)),
            'acg_nwr_pred_fg=%.6f' % _scalar_float(acg_profile.get('nwr_pred_fg_on_fg', 0.0)),
            'acg_nwr_pred_seed_fg=%.6f' % _scalar_float(acg_profile.get('nwr_pred_fg_on_seed_fg', 0.0)),
            'acg_nwr_pred_context_fg=%.6f' % _scalar_float(acg_profile.get('nwr_pred_fg_on_context_fg', 0.0)),
            'acg_nwr_pred_bg=%.6f' % _scalar_float(acg_profile.get('nwr_pred_fg_on_bg', 0.0)),
            'agc_spatial_rule_id=%.1f' % _scalar_float(acg_profile.get('agc_spatial_rule_id', 0.0)),
            'agc_disable_cap=%.1f' % _scalar_float(acg_profile.get('agc_disable_cap', 0.0)),
            'agc_disable_conf_gate=%.1f' % _scalar_float(acg_profile.get('agc_disable_conf_gate', 0.0)),
            'agc_complete_as_seed=%.1f' % _scalar_float(acg_profile.get('agc_complete_as_seed', 0.0)),
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
            'rgftd_teacher_active_loss=%.6f' % _scalar_float(rgftd_profile.get('teacher_active_loss', 0.0)),
            'rgftd_region=%.6f' % _scalar_float(rgftd_profile.get('region_ratio', 0.0)),
            'rgftd_risk_region_ratio=%.6f' % _scalar_float(rgftd_profile.get('risk_region_ratio', 0.0)),
            'rgftd_preserve_region_ratio=%.6f' % _scalar_float(rgftd_profile.get('preserve_region_ratio', 0.0)),
            'rgftd_active=%.6f' % _scalar_float(rgftd_profile.get('active_ratio', 0.0)),
            'rgftd_teacher_accept_ratio=%.6f' % _scalar_float(rgftd_profile.get('teacher_accept_ratio', 0.0)),
            'rgftd_teacher_reject_ratio=%.6f' % _scalar_float(rgftd_profile.get('teacher_reject_ratio', 0.0)),
            'rgftd_reject_by_support=%.6f' % _scalar_float(rgftd_profile.get('reject_by_support', 0.0)),
            'rgftd_reject_by_core_conflict=%.6f' % _scalar_float(rgftd_profile.get('reject_by_core_conflict', 0.0)),
            'rgftd_reject_by_fg_ratio=%.6f' % _scalar_float(rgftd_profile.get('reject_by_fg_ratio', 0.0)),
            'rgftd_release_score_mean=%.6f' % _scalar_float(rgftd_profile.get('release_score_mean', 0.0)),
            'rgftd_release_score_top=%.6f' % _scalar_float(rgftd_profile.get('release_score_top', 0.0)),
            'rdsi_enabled=%.1f' % _scalar_float(rgftd_profile.get('rdsi_enabled', 0.0)),
            'rdsi_compete=%.1f' % _scalar_float(rgftd_profile.get('rdsi_teacher_compete_count', 0.0)),
            'rdsi_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_multi_teacher_active', 0.0)),
            'rdsi_gap12=%.6f' % _scalar_float(rgftd_profile.get('rdsi_best_vs_second_gap', 0.0)),
            'rdsi_score=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_score_mean', 0.0)),
            'rdsi_score_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_score_top', 0.0)),
            'rdsi_score_benefit=%.6f' % _scalar_float(rgftd_profile.get('rdsi_score_benefit_mean', 0.0)),
            'rdsi_tid_mean=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_teacher_mean', 0.0)),
            'rdsi_switch=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_teacher_switch_ratio', 0.0)),
            'rdsi_trel=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher_reliable_score', 0.0)),
            'rdsi_sel_trel=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_teacher_reliable', 0.0)),
            'rdsi_sel_benefit=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_teacher_benefit', 0.0)),
            'rdsi_sel_gap=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_teacher_gap', 0.0)),
            'rdsi_sel_risk=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_student_risk', 0.0)),
            'rdsi_sel_spatial=%.6f' % _scalar_float(rgftd_profile.get('rdsi_selected_spatial_support', 0.0)),
            'rdsi_transfer=%.6f' % _scalar_float(rgftd_profile.get('rdsi_transfer_compatibility_mean', 0.0)),
            'rdsi_transfer_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_transfer_compatibility_top', 0.0)),
            'rdsi_gap=%.6f' % _scalar_float(rgftd_profile.get('rdsi_knowledge_gap_score', 0.0)),
            'rdsi_risk=%.6f' % _scalar_float(rgftd_profile.get('rdsi_risk_region_ratio', 0.0)),
            'rdsi_hcore=%.6f' % _scalar_float(rgftd_profile.get('rdsi_hard_core_ratio', 0.0)),
            'rdsi_softcore=%.6f' % _scalar_float(rgftd_profile.get('rdsi_soft_core_ratio', 0.0)),
            'rdsi_fgdef=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_deficient_ratio', 0.0)),
            'rdsi_fgex=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_excessive_ratio', 0.0)),
            'rdsi_fg_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_missing_need', 0.0)),
            'rdsi_bg_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_excess_need', 0.0)),
            'rdsi_bd_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_need', 0.0)),
            'rdsi_cand=%.6f' % _scalar_float(rgftd_profile.get('rdsi_candidate_ratio', 0.0)),
            'rdsi_accept=%.6f' % _scalar_float(rgftd_profile.get('rdsi_accept_ratio', 0.0)),
            'rdsi_reject=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_ratio', 0.0)),
            'rdsi_rej_core=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_core', 0.0)),
            'rdsi_rej_seed=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_seed', 0.0)),
            'rdsi_rej_prior=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_prior', 0.0)),
            'rdsi_rej_ent=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_entropy', 0.0)),
            'rdsi_rej_fgex=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_fg_excess', 0.0)),
            'rdsi_rej_bgonly=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_bg_only', 0.0)),
            'rdsi_rej_nofglift=%.6f' % _scalar_float(rgftd_profile.get('rdsi_reject_by_no_fg_lift', 0.0)),
            'rdsi_fg_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_foreground_active_ratio', 0.0)),
            'rdsi_bg_pair=%.6f' % _scalar_float(rgftd_profile.get('rdsi_background_paired_ratio', 0.0)),
            'rdsi_fg_repair_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_repair_active_ratio', 0.0)),
            'rdsi_bg_suppress_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppress_active_ratio', 0.0)),
            'rdsi_boundary_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_active_ratio', 0.0)),
            'rdsi_bg_pair_ctx=%.6f' % _scalar_float(rgftd_profile.get('rdsi_background_pair_ratio', 0.0)),
            'rdsi_bg_only=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_only_ratio', 0.0)),
            'rdsi_fg_repair_score=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_repair_score', 0.0)),
            'rdsi_bg_suppress_score=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppress_score', 0.0)),
            'rdsi_boundary_score=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_score', 0.0)),
            'rdsi_fg_lift=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_lift_mean', 0.0)),
            'rdsi_fg_lift_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_lift_top', 0.0)),
            'rdsi_bg_suppress=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppression_mean', 0.0)),
            'rdsi_bg_suppress_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppression_top', 0.0)),
            'rdsi_benefit=%.6f' % _scalar_float(rgftd_profile.get('rdsi_benefit_mean', 0.0)),
            'rdsi_benefit_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_benefit_top', 0.0)),
            'rdsi_boundary=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_support_mean', 0.0)),
            'rdsi_boundary_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_support_top', 0.0)),
            'rdsi_core_pres_fg=%.6f' % _scalar_float(rgftd_profile.get('rdsi_core_preserving_fg_mean', 0.0)),
            'rdsi_core_damage=%.6f' % _scalar_float(rgftd_profile.get('rdsi_core_damage_mean', 0.0)),
            'rdsi_seed_conflict=%.6f' % _scalar_float(rgftd_profile.get('rdsi_seed_conflict_mean', 0.0)),
            'rdsi_unsafe_gap=%.6f' % _scalar_float(rgftd_profile.get('rdsi_unsafe_gap_mean', 0.0)),
            'rdsi_fg_excess_proxy=%.6f' % _scalar_float(rgftd_profile.get('rdsi_foreground_excess_proxy', 0.0)),
            'rdsi_safe_signal=%.6f' % _scalar_float(rgftd_profile.get('rdsi_safe_signal', 0.0)),
            'rdsi_unsafe_signal=%.6f' % _scalar_float(rgftd_profile.get('rdsi_unsafe_signal', 0.0)),
            'rdsi_budget_factor=%.6f' % _scalar_float(rgftd_profile.get('rdsi_safe_budget_factor', 0.0)),
            'rdsi_budget_target=%.6f' % _scalar_float(rgftd_profile.get('rdsi_budget_target_ratio', 0.0)),
            'rdsi_recv_fg_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_fg_need', 0.0)),
            'rdsi_recv_bg_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_bg_need', 0.0)),
            'rdsi_recv_bd_need=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_boundary_need', 0.0)),
            'rdsi_recv_fg_prior=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_fg_prior', 0.0)),
            'rdsi_recv_bg_prior=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_bg_prior', 0.0)),
            'rdsi_recv_bd_prior=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_boundary_prior', 0.0)),
            'rdsi_recv_fg_share=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_fg_budget_share', 0.0)),
            'rdsi_recv_bg_share=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_bg_budget_share', 0.0)),
            'rdsi_recv_bd_share=%.6f' % _scalar_float(rgftd_profile.get('rdsi_receiver_boundary_budget_share', 0.0)),
            'rdsi_fg_budget_demand=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_budget_demand', 0.0)),
            'rdsi_bg_budget_demand=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_budget_demand', 0.0)),
            'rdsi_bd_budget_demand=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_budget_demand', 0.0)),
            'rdsi_fg_budget_alloc=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_budget_alloc', 0.0)),
            'rdsi_bg_budget_alloc=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_budget_alloc', 0.0)),
            'rdsi_bd_budget_alloc=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_budget_alloc', 0.0)),
            'rdsi_fg_budget_unused=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_budget_unused', 0.0)),
            'rdsi_bg_budget_unused=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_budget_unused', 0.0)),
            'rdsi_bd_budget_unused=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_budget_unused', 0.0)),
            'rdsi_budget_reflow=%.6f' % _scalar_float(rgftd_profile.get('rdsi_budget_reflow_ratio', 0.0)),
            'rdsi_quality=%.6f' % _scalar_float(rgftd_profile.get('rdsi_quality_mean', 0.0)),
            'rdsi_eff_topk=%.6f' % _scalar_float(rgftd_profile.get('rdsi_effective_topk_ratio', 0.0)),
            'rdsi_eff_minpx=%.1f' % _scalar_float(rgftd_profile.get('rdsi_effective_min_pixels', 0.0)),
            'rdsi_core_reopen=%.6f' % _scalar_float(rgftd_profile.get('rdsi_core_reopen_ratio', 0.0)),
            'rdsi_core_reopen_active=%.6f' % _scalar_float(rgftd_profile.get('rdsi_core_reopen_active_ratio', 0.0)),
            'rdsi_core_damage_veto=%.6f' % _scalar_float(rgftd_profile.get('rdsi_veto_by_core_damage', 0.0)),
            'rdsi_alpha=%.6f' % _scalar_float(rgftd_profile.get('rdsi_alpha_mean', 0.0)),
            'rdsi_alpha_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_alpha_top', 0.0)),
            'rdsi_raw_fg_delta=%.6f' % _scalar_float(rgftd_profile.get('rdsi_raw_teacher_fg_delta', 0.0)),
            'rdsi_raw_conf=%.6f' % _scalar_float(rgftd_profile.get('rdsi_raw_teacher_conf_mean', 0.0)),
            'rdsi_raw_fg=%.6f' % _scalar_float(rgftd_profile.get('rdsi_raw_teacher_fg_ratio', 0.0)),
            'rdsi_q_fg_delta=%.6f' % _scalar_float(rgftd_profile.get('rdsi_q_fg_delta', 0.0)),
            'rdsi_fg_repair_q_delta=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_repair_q_delta', 0.0)),
            'rdsi_bg_suppress_q_delta=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppress_q_delta', 0.0)),
            'rdsi_boundary_q_delta=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_q_delta', 0.0)),
            'rdsi_target_conf=%.6f' % _scalar_float(rgftd_profile.get('rdsi_target_conf_mean', 0.0)),
            'rdsi_target_ent=%.6f' % _scalar_float(rgftd_profile.get('rdsi_target_entropy_mean', 0.0)),
            'rdsi_loss_raw=%.6f' % _scalar_float(rgftd_profile.get('rdsi_loss_raw', 0.0)),
            'rdsi_loss_w=%.6f' % _scalar_float(rgftd_profile.get('rdsi_loss_weighted', 0.0)),
            'rdsi_proto_loss=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_loss', 0.0)),
            'rdsi_proto_w=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_weight_mean', 0.0)),
            'rdsi_proto_top=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_weight_top', 0.0)),
            'rdsi_proto_cos=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_cosine', 0.0)),
            'rdsi_proto_region=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_region_ratio', 0.0)),
            'rdsi_proto_tent=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_teacher_entropy', 0.0)),
            'rdsi_proto_tmax=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_teacher_weight_max', 0.0)),
            'rdsi_proto_valid=%.6f' % _scalar_float(rgftd_profile.get('rdsi_proto_valid_batches', 0.0)),
            'rdsi_fg_repair_loss=%.6f' % _scalar_float(rgftd_profile.get('rdsi_fg_repair_loss', 0.0)),
            'rdsi_bg_suppress_loss=%.6f' % _scalar_float(rgftd_profile.get('rdsi_bg_suppress_loss', 0.0)),
            'rdsi_boundary_loss=%.6f' % _scalar_float(rgftd_profile.get('rdsi_boundary_loss', 0.0)),
            'rdsi_t0=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher0_ratio', 0.0)),
            'rdsi_t1=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher1_ratio', 0.0)),
            'rdsi_t2=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher2_ratio', 0.0)),
            'rdsi_t3=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher3_ratio', 0.0)),
            'rdsi_t4=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher4_ratio', 0.0)),
            'rdsi_t5=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher5_ratio', 0.0)),
            'rdsi_t_scribble=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher_scribble_ratio', 0.0)),
            'rdsi_t_keypoint=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher_keypoint_ratio', 0.0)),
            'rdsi_t_block=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher_block_ratio', 0.0)),
            'rdsi_t_unknown=%.6f' % _scalar_float(rgftd_profile.get('rdsi_teacher_unknown_ratio', 0.0)),
            'rgftd_teacher_reliable_score=%.6f' % _scalar_float(rgftd_profile.get('teacher_reliable_score', 0.0)),
            'rgftd_student_risk_score=%.6f' % _scalar_float(rgftd_profile.get('student_risk_score', 0.0)),
            'rgftd_knowledge_gap_score=%.6f' % _scalar_float(rgftd_profile.get('knowledge_gap_score', 0.0)),
            'rgftd_selected_gap_mean=%.6f' % _scalar_float(rgftd_profile.get('selected_gap_mean', 0.0)),
            'rgftd_rejected_gap_mean=%.6f' % _scalar_float(rgftd_profile.get('rejected_gap_mean', 0.0)),
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
            'rgftd_spatial=%.6f' % _scalar_float(rgftd_profile.get('spatial_support_ratio', 0.0)),
            'rgftd_spw=%.6f' % _scalar_float(rgftd_profile.get('spatial_weight_mean', 0.0)),
            'rgftd_spw_cand=%.6f' % _scalar_float(rgftd_profile.get('spatial_weight_candidate_mean', 0.0)),
            'rgftd_spw_near=%.6f' % _scalar_float(rgftd_profile.get('spatial_weight_near_seed_mean', 0.0)),
            'rgftd_spw_far=%.6f' % _scalar_float(rgftd_profile.get('spatial_weight_far_mean', 0.0)),
            'rgftd_sp_loss=%.6f' % _scalar_float(rgftd_profile.get('spatial_loss_scale', 0.0)),
            'rgftd_fg_pre_sp=%.1f' % _scalar_float(rgftd_profile.get('active_foreground_pixels_pre_spatial', 0.0)),
            'rgftd_fg_sp_keep=%.6f' % _scalar_float(rgftd_profile.get('active_foreground_spatial_keep_ratio', 0.0)),
            'rgftd_fg_pre_px=%.1f' % _scalar_float(rgftd_profile.get('active_foreground_pixels_pre_budget', 0.0)),
            'rgftd_fg_budget=%.6f' % _scalar_float(rgftd_profile.get('foreground_budget_ratio', 0.0)),
            'rgftd_fg_cand=%.6f' % _scalar_float(rgftd_profile.get('active_fg_candidate_ratio', 0.0)),
            'rgftd_fg_fgcand=%.6f' % _scalar_float(rgftd_profile.get('active_fg_fg_candidate_ratio', 0.0)),
            'rgftd_fg_near_seed=%.6f' % _scalar_float(rgftd_profile.get('active_fg_near_seed_ratio', 0.0)),
            'rgftd_fg_seed_p=%.6f' % _scalar_float(rgftd_profile.get('active_fg_seed_precision', 0.0)),
            'rgftd_ref=%.1f' % _scalar_float(rgftd_profile.get('refine_enabled', 0.0)),
            'rgftd_ref_silent=%.1f' % _scalar_float(rgftd_profile.get('refine_silent', 0.0)),
            'rgftd_ref_roi=%.6f' % _scalar_float(rgftd_profile.get('refine_roi_ratio', 0.0)),
            'rgftd_ref_aff=%.6f' % _scalar_float(rgftd_profile.get('refine_affinity_mean', 0.0)),
            'rgftd_tq_kl=%.6f' % _scalar_float(rgftd_profile.get('refine_teacher_q_kl', 0.0)),
            'rgftd_q_ent=%.6f' % _scalar_float(rgftd_profile.get('refine_q_entropy_mean', 0.0)),
            'rgftd_q_fg_mass=%.6f' % _scalar_float(rgftd_profile.get('refine_q_fg_mass', 0.0)),
            'rgftd_q_fg_ratio=%.6f' % _scalar_float(rgftd_profile.get('refine_q_fg_ratio', 0.0)),
            'rgftd_q_fg_delta=%.6f' % _scalar_float(rgftd_profile.get('refine_q_fg_delta', 0.0)),
            'rgftd_q_seed_p=%.6f' % _scalar_float(rgftd_profile.get('refine_q_seed_precision', 0.0)),
            'rgftd_q_seed_r=%.6f' % _scalar_float(rgftd_profile.get('refine_q_seed_recall', 0.0)),
            'rgftd_q_cand=%.6f' % _scalar_float(rgftd_profile.get('refine_q_candidate_ratio', 0.0)),
            'rgftd_q_near=%.6f' % _scalar_float(rgftd_profile.get('refine_q_near_seed_ratio', 0.0)),
            'rgftd_q_unsup_fg=%.6f' % _scalar_float(rgftd_profile.get('refine_unsupported_fg_ratio', 0.0)),
            'rgftd_q_unsup_s=%.6f' % _scalar_float(rgftd_profile.get('refine_unsupported_fg_scale', 0.0)),
            'rgftd_q_core_cf=%.6f' % _scalar_float(rgftd_profile.get('refine_q_core_conflict', 0.0)),
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
        if 'v3_pool_size' in rgftd_profile:
            parts.extend([
                'rgftd_v3_pool=%.1f' % _scalar_float(rgftd_profile.get('v3_pool_size', 0.0)),
                'rgftd_v3_pool_stable=%.1f' % _scalar_float(rgftd_profile.get('v3_pool_stable', 0.0)),
                'rgftd_v3_silent=%.1f' % _scalar_float(rgftd_profile.get('v3_no_teacher', 1.0)),
                'rgftd_v3_fb=%.1f' % _scalar_float(rgftd_profile.get('v3_fallback_teacher', 0.0)),
                'rgftd_v3_src=%.1f' % _scalar_float(rgftd_profile.get('v3_teacher_source', 0.0)),
                'rgftd_v3_tid=%.1f' % _scalar_float(rgftd_profile.get('v3_selected_teacher', -1.0)),
                'rgftd_v3_score=%.6f' % _scalar_float(rgftd_profile.get('v3_selected_score', 0.0)),
                'rgftd_v3_sel_stable=%.6f' % _scalar_float(rgftd_profile.get('v3_selected_stable_score', 0.0)),
                'rgftd_v3_fail_tid=%.1f' % _scalar_float(rgftd_profile.get('v3_best_failed_teacher', -1.0)),
                'rgftd_v3_fail_score=%.6f' % _scalar_float(rgftd_profile.get('v3_best_failed_score', 0.0)),
                'rgftd_v3_seed_fgp=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_seed_fg_prob_mean', 0.0)),
                'rgftd_v3_seed_fgm=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_seed_fg_margin_mean', 0.0)),
                'rgftd_v3_seed_fg=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_seed_fg_recall', 0.0)),
                'rgftd_v3_core_cf=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_core_conflict', 0.0)),
                'rgftd_v3_core_valid=%.1f' % _scalar_float(rgftd_profile.get('v3_audit_core_valid', 0.0)),
                'rgftd_v3_trel=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_teacher_reliability', 0.0)),
                'rgftd_v3_relf=%.6f' % _scalar_float(rgftd_profile.get('v3_audit_release_factor', 0.0)),
                'rgftd_v3_lease=%.1f' % _scalar_float(rgftd_profile.get('v3_lease_active', 0.0)),
                'rgftd_v3_lease_age=%.1f' % _scalar_float(rgftd_profile.get('v3_lease_age', 0.0)),
                'rgftd_v3_benefit=%.6f' % _scalar_float(rgftd_profile.get('v3_benefit_score', 1.0)),
                'rgftd_v3_win=%.6f' % _scalar_float(rgftd_profile.get('v3_window_score', 1.0)),
                'rgftd_v3_stab=%.6f' % _scalar_float(rgftd_profile.get('v3_stability_score', 1.0)),
                'rgftd_v3_renew=%.1f' % _scalar_float(rgftd_profile.get('v3_teacher_renew', 0.0)),
                'rgftd_v3_revoke=%.1f' % _scalar_float(rgftd_profile.get('v3_teacher_revoke', 0.0)),
                'rgftd_v3_decay=%.1f' % _scalar_float(rgftd_profile.get('v3_teacher_decay', 0.0)),
                'rgftd_v3_cap=%.6f' % _scalar_float(rgftd_profile.get('v3_cap_current', 0.0)),
                'rgftd_v3_cap_scale=%.6f' % _scalar_float(rgftd_profile.get('v3_cap_scale', 1.0)),
                'rgftd_v3_ret0=%.6f' % _scalar_float(rgftd_profile.get('v3_ret0_ratio', 0.0)),
                'rgftd_v3_cap_hit=%.6f' % _scalar_float(rgftd_profile.get('v3_cap_hit_ratio', 0.0)),
                'rgftd_v3_bgfg_win=%.6f' % _scalar_float(rgftd_profile.get('v3_bgfg_window', 0.0)),
                'rgftd_v3_lowmaxp_d=%.6f' % _scalar_float(rgftd_profile.get('v3_lowmaxp_delta', 0.0)),
                'rgftd_v3_mass_d=%.6f' % _scalar_float(rgftd_profile.get('v3_wann_mass_delta', 0.0)),
                'rgftd_stable_valid=%.1f' % _scalar_float(rgftd_profile.get('stable_valid', 0.0)),
                'rgftd_stable_score=%.6f' % _scalar_float(rgftd_profile.get('stable_score', 0.0)),
                'rgftd_stable_best=%.6f' % _scalar_float(rgftd_profile.get('stable_best_score', 0.0)),
                'rgftd_stable_updated=%.1f' % _scalar_float(rgftd_profile.get('stable_updated', 0.0)),
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

    def __init__(self, args, model, trainloader, valloader, meta_valloader=None, amp=False):
        super(MyClient, self).__init__(args, model, trainloader, valloader)
        self.meta_valloader = meta_valloader
        self.amp = amp
        if self.amp:
            self.scaler = GradScaler()
        self.best_performance = 0.0
        self.rgftd_v3_last_audit_iter = -1
        self.rgftd_v3_selected_teacher_id = None
        self.rgftd_v3_status = _default_rgftd_v3_status()
        self.rgftd_v3_active_key = None
        self.rgftd_v3_lease_start_iter = -1
        self.rgftd_v3_benefit_scores = {}
        self.rgftd_v3_stability_scores = {}
        self.rgftd_v3_last_window_scores = {}
        self.rgftd_v3_last_wann_mass = {}
        self.rgftd_v3_last_wann_lowmaxp = {}
        self.rgftd_stable_best_score = 0.0
        self.rgftd_stable_valid = 0.0
        self.rgftd_stable_updated = 0.0
        self.rgftd_stable_last_score = 0.0
        self.rgftd_stable_last_wann_mass = None
        self.rgftd_stable_last_wann_lowmaxp = None
        self.rgftd_stable_snapshot_arrays = None
        self.rgftd_stable_pending_upload = False

    def _validate(self, config):
        if config.get('stage', '') == 'learnable_nwr_meta':
            return self._validate_learnable_nwr_meta(config)
        return super()._validate(config)

    def _validate_learnable_nwr_meta(self, config):
        del config
        if self.meta_valloader is None:
            raise RuntimeError('learnable NWR meta-validation loader is required')
        self.model.eval()
        ce_loss = CrossEntropyLoss(ignore_index=self.args.num_classes, reduction='sum')
        total_loss = 0.0
        total_pixels = 0.0
        with torch.no_grad():
            for sampled_batch in self.meta_valloader:
                volume_batch, label_batch = self._move_batch_to_cuda(sampled_batch)
                logits = _primary_logits(self.model(volume_batch))
                valid = label_batch != self.args.num_classes
                loss = ce_loss(logits, label_batch.long())
                total_loss += float(loss.detach().cpu().item())
                total_pixels += float(valid.float().sum().detach().cpu().item())
        meta_loss = total_loss / max(total_pixels, 1.0)
        metrics_ = {
            'client_{}_learnable_nwr_meta_loss'.format(self.cid): meta_loss,
        }
        return meta_loss, metrics_

    def _rgftd_v3_enabled(self):
        return (
            int(getattr(self.args, 'rgftd_enabled', 0)) == 1
            and int(getattr(self.args, 'rgftd_v3_enabled', 0)) == 1
        )

    def _rgftd_v3_stable_enabled(self):
        return self._rgftd_v3_enabled() and int(getattr(self.args, 'rgftd_v3_stable_teacher_enabled', 0)) == 1

    def _rdsi_enabled(self):
        return self._rgftd_v3_enabled() and int(getattr(self.args, 'rdsi_enabled', 1)) == 1

    def _move_batch_to_cuda(self, sampled_batch):
        if self.args.img_class == 'faz' or self.args.img_class == 'prostate':
            volume_batch = sampled_batch['image'].unsqueeze(1).cuda()
            label_batch = sampled_batch['label'].cuda()
        elif self.args.img_class == 'odoc' or self.args.img_class == 'odoc_binary' or self.args.img_class == 'polyp' or self.args.img_class == 'isic' or self.args.img_class == 'busi' or self.args.img_class == 'tn3k' or self.args.img_class == 'duts' or self.args.img_class == 'glas' or self.args.img_class == 'ebhiseg':
            volume_batch = sampled_batch['image'].cuda()
            label_batch = sampled_batch['label'].cuda()
        else:
            raise ValueError('Unsupported img_class: {}'.format(self.args.img_class))
        return volume_batch, label_batch

    def _build_teacher_model_from_state_dict(self, teacher_state_dict):
        teacher_model = copy.deepcopy(self.model.model).cuda()
        teacher_model.load_state_dict(teacher_state_dict, strict=False)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        return teacher_model

    def _rgftd_v3_candidate_teacher_ids(self):
        teacher_state_dicts = getattr(self.model, 'rgftd_teacher_state_dicts', {})
        teacher_valids = getattr(self.model, 'rgftd_teacher_valids', {})
        candidate_ids = []
        for teacher_id in sorted(teacher_state_dicts.keys()):
            if int(teacher_id) == int(self.cid):
                continue
            if self._rgftd_v3_stable_enabled() and float(teacher_valids.get(teacher_id, 0.0)) <= 0.5:
                continue
            candidate_ids.append(teacher_id)
        return candidate_ids

    def _rgftd_v3_pair_key(self, teacher_source, teacher_id):
        return '{}:{}'.format(int(teacher_source), int(teacher_id))

    def _rgftd_v3_get_score(self, storage, key, default=1.0):
        return float(storage.get(key, default))

    def _rgftd_v3_cap_scale_for_score(self, benefit_score):
        if int(getattr(self.args, 'rgftd_v3_benefit_enabled', 1)) != 1:
            return 1.0
        good_thresh = float(getattr(self.args, 'rgftd_v3_benefit_good_thresh', 0.70))
        decay_thresh = float(getattr(self.args, 'rgftd_v3_benefit_decay_thresh', 0.50))
        min_scale = float(getattr(self.args, 'rgftd_v3_min_cap_scale', 0.25))
        if benefit_score >= good_thresh:
            return 1.0
        if benefit_score >= decay_thresh:
            return float(getattr(self.args, 'rgftd_v3_decay_cap_scale', 0.50))
        return min_scale

    def _rgftd_v3_routing_benefit(self, key):
        benefit = self._rgftd_v3_get_score(self.rgftd_v3_benefit_scores, key, 1.0)
        floor = float(getattr(self.args, 'rgftd_v3_routing_benefit_floor', 0.25))
        return max(benefit, floor)

    def _rgftd_v3_base_cap(self, fallback_teacher=False):
        if fallback_teacher:
            return float(getattr(
                self.args,
                'rgftd_v3_fallback_lambda_eff_cap',
                min(float(getattr(self.args, 'rgftd_lambda_eff_cap', 0.02)), 0.01),
            ))
        return float(getattr(self.args, 'rgftd_lambda_eff_cap', 0.02))

    def _rgftd_args_for_teacher_source(self, fallback_teacher=False, cap_scale=1.0):
        cap_scale = max(float(cap_scale), 0.0)
        if not fallback_teacher:
            if abs(cap_scale - 1.0) < 1e-8:
                return self.args
            routed_args = copy.copy(self.args)
            routed_args.rgftd_lambda_eff_cap = self._rgftd_v3_base_cap(False) * cap_scale
            return routed_args
        fallback_args = copy.copy(self.args)
        fallback_cap = self._rgftd_v3_base_cap(True) * cap_scale
        fallback_args.rgftd_lambda_eff_cap = fallback_cap
        return fallback_args

    def _rgftd_stable_profile_score(self, profile, wann_maps):
        prob_floor = float(getattr(self.args, 'rgftd_stable_seed_prob_floor', 0.35))
        margin_floor = float(getattr(self.args, 'rgftd_stable_seed_margin_floor', 0.05))
        max_conflict = float(max(getattr(self.args, 'rgftd_stable_max_core_conflict', 0.20), 1e-6))
        seed_fg_pixels = _scalar_float(profile.get('seed_fg_support_pixels', 0.0))
        seed_prob = _scalar_float(profile.get('teacher_seed_support_fg_prob_mean', 0.0))
        seed_margin = _scalar_float(profile.get('teacher_seed_support_fg_margin_mean', 0.0))
        core_valid = _scalar_float(profile.get('teacher_core_valid', 0.0))
        core_conflict = _scalar_float(profile.get('teacher_core_conflict', 0.0))
        if seed_fg_pixels <= 0.0:
            return 0.0, {
                'prob_gate': 0.0,
                'margin_gate': 0.0,
                'core_gate': 0.0,
                'wann_gate': 0.0,
            }

        prob_gate = _normalize_gate_scalar(seed_prob, prob_floor)
        margin_gate = _normalize_gate_scalar(seed_margin, margin_floor)
        core_gate = 1.0
        if core_valid > 0.0:
            core_gate = max(0.0, min(1.0 - core_conflict / max_conflict, 1.0))

        wann_mass = _scalar_float(wann_maps.profile.get('effective_supervision_mass', 0.0))
        wann_lowmaxp = _scalar_float(wann_maps.profile.get('max_prob_low_r', 0.0))
        mass_drop_thresh = float(max(getattr(self.args, 'rgftd_stable_wann_mass_drop_thresh', 0.05), 1e-6))
        lowmaxp_delta_thresh = float(max(getattr(self.args, 'rgftd_stable_lowmaxp_delta_thresh', 0.02), 1e-6))
        mass_gate = 1.0
        lowmaxp_gate = 1.0
        if self.rgftd_stable_last_wann_mass is not None:
            mass_drop = max(0.0, self.rgftd_stable_last_wann_mass - wann_mass)
            mass_gate = max(0.0, min(1.0 - mass_drop / mass_drop_thresh, 1.0))
        if self.rgftd_stable_last_wann_lowmaxp is not None:
            lowmaxp_delta = max(0.0, wann_lowmaxp - self.rgftd_stable_last_wann_lowmaxp)
            lowmaxp_gate = max(0.0, min(1.0 - lowmaxp_delta / lowmaxp_delta_thresh, 1.0))
        self.rgftd_stable_last_wann_mass = wann_mass
        self.rgftd_stable_last_wann_lowmaxp = wann_lowmaxp
        wann_gate = min(mass_gate, lowmaxp_gate)
        score = prob_gate * margin_gate * core_gate * wann_gate
        return score, {
            'prob_gate': prob_gate,
            'margin_gate': margin_gate,
            'core_gate': core_gate,
            'wann_gate': wann_gate,
        }

    def _rgftd_update_stable_teacher(self, outputs, label_batch, wann_maps):
        if not self._rgftd_v3_stable_enabled():
            return
        audit_start = int(getattr(self.args, 'rgftd_v3_audit_start_iters', getattr(self.args, 'rgftd_warmup_iters', 800)))
        if self.current_iter < audit_start:
            return
        stable_args = copy.copy(self.args)
        stable_args.rgftd_refine_enabled = 0
        _, _, stable_profile = rgftd_loss(
            outputs.detach(),
            outputs.detach(),
            label_batch,
            wann_maps,
            stable_args,
            max(self.current_iter, int(getattr(self.args, 'rgftd_warmup_iters', 800)) + 1),
            image=None,
        )
        score, _ = self._rgftd_stable_profile_score(stable_profile, wann_maps)
        self.rgftd_stable_last_score = float(score)
        min_score = float(getattr(self.args, 'rgftd_stable_min_score', 0.05))
        update_margin = float(getattr(self.args, 'rgftd_stable_update_margin', 0.01))
        min_prob = float(getattr(self.args, 'rgftd_stable_seed_prob_floor', 0.35))
        min_margin = float(getattr(self.args, 'rgftd_stable_seed_margin_floor', 0.05))
        max_conflict = float(getattr(self.args, 'rgftd_stable_max_core_conflict', 0.20))
        seed_prob = _scalar_float(stable_profile.get('teacher_seed_support_fg_prob_mean', 0.0))
        seed_margin = _scalar_float(stable_profile.get('teacher_seed_support_fg_margin_mean', 0.0))
        core_valid = _scalar_float(stable_profile.get('teacher_core_valid', 0.0))
        core_conflict = _scalar_float(stable_profile.get('teacher_core_conflict', 0.0))
        core_safe = core_valid <= 0.0 or core_conflict <= max_conflict
        should_update = (
            score >= min_score
            and score > self.rgftd_stable_best_score + update_margin
            and seed_prob >= min_prob
            and seed_margin >= min_margin
            and core_safe
        )
        if should_update:
            self.rgftd_stable_best_score = float(score)
            self.rgftd_stable_valid = 1.0
            self.rgftd_stable_updated = 1.0
            self.rgftd_stable_pending_upload = True
            self.rgftd_stable_snapshot_arrays = [
                tensor.detach().cpu().numpy().copy()
                for tensor in self.model.model.state_dict().values()
            ]

    def _rgftd_v3_fill_runtime_status(self, status, teacher_source, teacher_id, cap_current):
        status = dict(status)
        key = self._rgftd_v3_pair_key(teacher_source, teacher_id)
        if self.rgftd_v3_active_key != key:
            self.rgftd_v3_active_key = key
            self.rgftd_v3_lease_start_iter = self.current_iter
        benefit_score = self._rgftd_v3_get_score(self.rgftd_v3_benefit_scores, key, 1.0)
        stability_score = self._rgftd_v3_get_score(self.rgftd_v3_stability_scores, key, 1.0)
        window_score = self._rgftd_v3_get_score(self.rgftd_v3_last_window_scores, key, 1.0)
        revoke_thresh = float(getattr(self.args, 'rgftd_v3_benefit_revoke_thresh', 0.35))
        decay_thresh = float(getattr(self.args, 'rgftd_v3_benefit_decay_thresh', 0.50))
        teacher_revoke = 1.0 if benefit_score < revoke_thresh else 0.0
        teacher_decay = 1.0 if revoke_thresh <= benefit_score < decay_thresh else 0.0
        if int(teacher_source) == 2:
            teacher_revoke = 0.0
            teacher_decay = 1.0 if benefit_score < decay_thresh else 0.0
        status.update({
            'v3_lease_active': 0.0,
            'v3_lease_age': 0.0,
            'v3_benefit_score': benefit_score,
            'v3_window_score': window_score,
            'v3_stability_score': stability_score,
            'v3_teacher_renew': 1.0 if benefit_score >= float(getattr(self.args, 'rgftd_v3_benefit_good_thresh', 0.70)) else 0.0,
            'v3_teacher_revoke': teacher_revoke,
            'v3_teacher_decay': teacher_decay,
            'v3_cap_current': float(cap_current),
            'v3_cap_scale': self._rgftd_v3_cap_scale_for_score(benefit_score),
        })
        return status

    def _rgftd_v3_update_online_benefit(self, status, rgftd_profile, wann_maps):
        if int(getattr(self.args, 'rgftd_v3_benefit_enabled', 1)) != 1:
            return status
        source = int(float(status.get('v3_teacher_source', 0.0)))
        teacher_id = int(float(status.get('v3_selected_teacher', -1.0)))
        if source not in [1, 2]:
            return status

        key = self._rgftd_v3_pair_key(source, teacher_id)
        ret = _scalar_float(rgftd_profile.get('return_reason', 3.0))
        ret0 = 1.0 if ret <= 0.0 else 0.0
        fg_px = _scalar_float(rgftd_profile.get('active_foreground_pixels', 0.0))
        bgfg = _scalar_float(rgftd_profile.get('background_foreground_ratio', 0.0))
        lambda_eff = _scalar_float(rgftd_profile.get('lambda_effective', 0.0))
        cap_current = max(float(status.get('v3_cap_current', 0.0)), 0.0)
        cap_hit = 1.0 if cap_current > 0.0 and lambda_eff >= 0.95 * cap_current else 0.0

        wann_mass = 0.0
        wann_lowmaxp = 0.0
        if wann_maps is not None:
            wann_profile = wann_maps.profile
            wann_mass = _scalar_float(wann_profile.get('effective_supervision_mass', 0.0))
            wann_lowmaxp = _scalar_float(wann_profile.get('max_prob_low_r', 0.0))
        last_mass = self.rgftd_v3_last_wann_mass.get(key, None)
        last_lowmaxp = self.rgftd_v3_last_wann_lowmaxp.get(key, None)
        if last_mass is None:
            mass_delta = 0.0
        else:
            mass_delta = wann_mass - last_mass
        if last_lowmaxp is None:
            lowmaxp_delta = 0.0
        else:
            lowmaxp_delta = wann_lowmaxp - last_lowmaxp
        self.rgftd_v3_last_wann_mass[key] = wann_mass
        self.rgftd_v3_last_wann_lowmaxp[key] = wann_lowmaxp

        min_fg = float(getattr(self.args, 'rgftd_min_foreground_pixels', 8))
        if ret0 > 0.0 and fg_px >= min_fg:
            release_score = 1.0
        elif ret0 > 0.0:
            release_score = 0.65
        elif ret == 4.0:
            release_score = 0.55
        elif ret in [1.0, 2.0]:
            release_score = 0.45
        else:
            release_score = 0.30

        stability = 1.0
        bgfg_bad = bgfg > float(getattr(self.args, 'rgftd_v3_bgfg_warn_thresh', 0.75))
        lowmaxp_bad = lowmaxp_delta > float(getattr(self.args, 'rgftd_v3_lowmaxp_delta_thresh', 0.02))
        mass_bad = mass_delta < -float(getattr(self.args, 'rgftd_v3_wann_mass_drop_thresh', 0.05))
        if bgfg_bad:
            stability -= 0.20
        if cap_hit > 0.0 and (bgfg_bad or lowmaxp_bad or mass_bad):
            stability -= float(getattr(self.args, 'rgftd_v3_cap_hit_penalty', 0.20))
        if lowmaxp_bad:
            stability -= 0.20
        if mass_bad:
            stability -= 0.20
        stability = max(0.0, min(stability, 1.0))
        window_score = max(0.0, min(release_score * stability, 1.0))

        momentum = float(getattr(self.args, 'rgftd_v3_benefit_momentum', 0.80))
        momentum = max(0.0, min(momentum, 0.999))
        prev_benefit = self._rgftd_v3_get_score(self.rgftd_v3_benefit_scores, key, 1.0)
        prev_stability = self._rgftd_v3_get_score(self.rgftd_v3_stability_scores, key, 1.0)
        benefit = momentum * prev_benefit + (1.0 - momentum) * window_score
        stability_ema = momentum * prev_stability + (1.0 - momentum) * stability
        self.rgftd_v3_benefit_scores[key] = benefit
        self.rgftd_v3_stability_scores[key] = stability_ema
        self.rgftd_v3_last_window_scores[key] = window_score

        revoke_thresh = float(getattr(self.args, 'rgftd_v3_benefit_revoke_thresh', 0.35))
        decay_thresh = float(getattr(self.args, 'rgftd_v3_benefit_decay_thresh', 0.50))
        good_thresh = float(getattr(self.args, 'rgftd_v3_benefit_good_thresh', 0.70))
        teacher_revoke = 1.0 if benefit < revoke_thresh else 0.0
        teacher_decay = 1.0 if revoke_thresh <= benefit < decay_thresh else 0.0
        if source == 2:
            teacher_revoke = 0.0
            teacher_decay = 1.0 if benefit < decay_thresh else 0.0
        status.update({
            'v3_benefit_score': benefit,
            'v3_window_score': window_score,
            'v3_stability_score': stability_ema,
            'v3_teacher_renew': 1.0 if benefit >= good_thresh else 0.0,
            'v3_teacher_revoke': teacher_revoke,
            'v3_teacher_decay': teacher_decay,
            'v3_cap_scale': self._rgftd_v3_cap_scale_for_score(benefit),
            'v3_ret0_ratio': ret0,
            'v3_cap_hit_ratio': cap_hit,
            'v3_bgfg_window': bgfg,
            'v3_lowmaxp_delta': lowmaxp_delta,
            'v3_wann_mass_delta': mass_delta,
        })
        return status

    def _should_run_rgftd_v3_audit(self):
        if not self._rgftd_v3_enabled():
            return False
        audit_start = int(getattr(self.args, 'rgftd_v3_audit_start_iters', getattr(self.args, 'rgftd_warmup_iters', 800)))
        if self.current_iter < audit_start:
            return False
        audit_interval = int(getattr(self.args, 'rgftd_v3_audit_interval_iters', 1000))
        if self.rgftd_v3_last_audit_iter < 0:
            return True
        last_status = getattr(self, 'rgftd_v3_status', _default_rgftd_v3_status())
        empty_pool_retry = int(getattr(self.args, 'rgftd_v3_empty_pool_retry_iters', 10))
        if (
            empty_pool_retry > 0
            and float(last_status.get('v3_no_teacher', 1.0)) > 0.5
            and float(last_status.get('v3_pool_size', 0.0)) <= 0.0
        ):
            return (self.current_iter - self.rgftd_v3_last_audit_iter) >= empty_pool_retry
        if audit_interval <= 0:
            return False
        return (self.current_iter - self.rgftd_v3_last_audit_iter) >= audit_interval

    def _run_rgftd_v3_routing_audit(self, wann_ref_model):
        status = _default_rgftd_v3_status()
        teacher_state_dicts = getattr(self.model, 'rgftd_teacher_state_dicts', {})
        teacher_valids = getattr(self.model, 'rgftd_teacher_valids', {})
        teacher_scores = getattr(self.model, 'rgftd_teacher_scores', {})
        if not teacher_state_dicts:
            self.rgftd_v3_selected_teacher_id = None
            self.rgftd_v3_last_audit_iter = self.current_iter
            self.rgftd_v3_status = status
            return status

        candidate_ids = [
            teacher_id for teacher_id in sorted(teacher_state_dicts.keys())
            if int(teacher_id) != int(self.cid)
            and (
                not self._rgftd_v3_stable_enabled()
                or float(teacher_valids.get(teacher_id, 0.0)) > 0.5
            )
        ]
        if not candidate_ids:
            self.rgftd_v3_selected_teacher_id = None
            self.rgftd_v3_last_audit_iter = self.current_iter
            self.rgftd_v3_status = status
            return status

        audit_batches = max(int(getattr(self.args, 'rgftd_v3_audit_batches', 4)), 1)
        teacher_probe_model = copy.deepcopy(self.model.model).cuda()
        teacher_probe_model.eval()
        for param in teacher_probe_model.parameters():
            param.requires_grad = False

        profile_keys = [
            'teacher_seed_support_fg_prob_mean',
            'teacher_seed_support_fg_margin_mean',
            'teacher_seed_support_fg_recall',
            'teacher_core_conflict',
            'teacher_core_valid',
            'teacher_reliability',
            'release_factor',
        ]
        fg_weighted_profile_keys = [
            'teacher_seed_support_fg_prob_mean',
            'teacher_seed_support_fg_margin_mean',
            'teacher_seed_support_fg_recall',
        ]
        core_weighted_profile_keys = [
            'teacher_core_conflict',
            'teacher_core_valid',
        ]
        teacher_sums = {
            teacher_id: {key: 0.0 for key in profile_keys}
            for teacher_id in candidate_ids
        }
        teacher_weights = {
            teacher_id: {key: 0.0 for key in profile_keys}
            for teacher_id in candidate_ids
        }

        previous_mode = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch_idx, sampled_batch in enumerate(self.trainloader):
                if batch_idx >= audit_batches:
                    break
                volume_batch, label_batch = self._move_batch_to_cuda(sampled_batch)
                student_out = self.model(volume_batch)
                student_logits = _primary_logits(student_out)
                student_aux_logits = _aux_logits(student_out)
                ref_out = wann_ref_model(volume_batch)
                ref_logits = _primary_logits(ref_out)
                wann_maps = build_wann_maps(
                    image=volume_batch,
                    label=label_batch,
                    logits=student_logits,
                    aux_logits=student_aux_logits,
                    sup_type=self.args.sup_type,
                    img_class=self.args.img_class,
                    num_classes=self.args.num_classes,
                    iter_num=self.current_iter,
                    args=self.args,
                    ref_logits=ref_logits,
                )
                for teacher_id in candidate_ids:
                    teacher_probe_model.load_state_dict(teacher_state_dicts[teacher_id], strict=False)
                    teacher_logits = _primary_logits(teacher_probe_model(volume_batch))
                    _, _, audit_profile = rgftd_loss(
                        student_logits,
                        teacher_logits,
                        label_batch,
                        wann_maps,
                        self.args,
                        self.current_iter,
                        image=volume_batch,
                    )
                    for key in profile_keys:
                        metric_value = _scalar_float(audit_profile.get(key, 0.0))
                        if key in fg_weighted_profile_keys:
                            metric_weight = _scalar_float(audit_profile.get('seed_fg_support_pixels', 0.0))
                        elif key in core_weighted_profile_keys:
                            metric_weight = _scalar_float(audit_profile.get('teacher_core_valid', 0.0))
                        else:
                            metric_weight = 1.0
                        if metric_weight <= 0.0:
                            continue
                        teacher_sums[teacher_id][key] += metric_value * metric_weight
                        teacher_weights[teacher_id][key] += metric_weight
        if previous_mode:
            self.model.train()

        prob_floor = float(getattr(self.args, 'rgftd_teacher_release_prob_floor', getattr(self.args, 'rgftd_teacher_fg_prob_thresh', 0.35)))
        margin_floor = float(getattr(self.args, 'rgftd_teacher_release_margin_floor', 0.05))
        max_core_conflict = float(max(getattr(self.args, 'rgftd_teacher_max_core_conflict', 0.20), 1e-6))
        score_thresh = float(getattr(self.args, 'rgftd_v3_audit_score_thresh', 0.05))
        reliability_min = float(getattr(self.args, 'rgftd_v3_audit_teacher_reliability_min', 0.30))

        passed = []
        best_teacher_status = dict(status)
        for teacher_id in candidate_ids:
            mean_profile = {
                key: (
                    teacher_sums[teacher_id][key] / teacher_weights[teacher_id][key]
                    if teacher_weights[teacher_id][key] > 0.0 else 0.0
                )
                for key in profile_keys
            }
            foreground_gate = (
                _normalize_gate_scalar(mean_profile['teacher_seed_support_fg_prob_mean'], prob_floor)
                * _normalize_gate_scalar(mean_profile['teacher_seed_support_fg_margin_mean'], margin_floor)
            )
            if mean_profile['teacher_core_valid'] > 0.0:
                safety_gate = max(0.0, min(1.0 - mean_profile['teacher_core_conflict'] / max_core_conflict, 1.0))
            else:
                safety_gate = 1.0
            audit_score = foreground_gate * safety_gate
            pair_key = self._rgftd_v3_pair_key(1, teacher_id)
            benefit_score = self._rgftd_v3_routing_benefit(pair_key)
            stability_score = self._rgftd_v3_get_score(self.rgftd_v3_stability_scores, pair_key, 1.0)
            routing_score = audit_score * benefit_score * stability_score
            recover_thresh = float(getattr(self.args, 'rgftd_v3_recover_audit_score_thresh', 0.60))
            if (
                audit_score >= recover_thresh
                and self._rgftd_v3_get_score(self.rgftd_v3_benefit_scores, pair_key, 1.0) < benefit_score
            ):
                self.rgftd_v3_benefit_scores[pair_key] = benefit_score
            teacher_status = {
                'v3_pool_size': 0.0,
                'v3_pool_stable': 0.0,
                'v3_no_teacher': 0.0,
                'v3_fallback_teacher': 0.0,
                'v3_teacher_source': 1.0,
                'v3_selected_teacher': float(teacher_id),
                'v3_selected_score': float(routing_score),
                'v3_selected_stable_score': float(teacher_scores.get(teacher_id, 0.0)),
                'v3_best_failed_teacher': -1.0,
                'v3_best_failed_score': 0.0,
                'v3_routing_score': float(routing_score),
                'v3_audit_seed_fg_prob_mean': float(mean_profile['teacher_seed_support_fg_prob_mean']),
                'v3_audit_seed_fg_margin_mean': float(mean_profile['teacher_seed_support_fg_margin_mean']),
                'v3_audit_seed_fg_recall': float(mean_profile['teacher_seed_support_fg_recall']),
                'v3_audit_core_conflict': float(mean_profile['teacher_core_conflict']),
                'v3_audit_core_valid': float(mean_profile['teacher_core_valid']),
                'v3_audit_teacher_reliability': float(mean_profile['teacher_reliability']),
                'v3_audit_release_factor': float(mean_profile['release_factor']),
                'v3_benefit_score': float(benefit_score),
                'v3_stability_score': float(stability_score),
            }
            if teacher_id == candidate_ids[0] or routing_score > best_teacher_status['v3_selected_score']:
                best_teacher_status = teacher_status
            if (
                routing_score >= score_thresh
                and mean_profile['teacher_reliability'] >= reliability_min
                and (
                    mean_profile['teacher_core_valid'] <= 0.0
                    or mean_profile['teacher_core_conflict'] <= max_core_conflict
                )
            ):
                passed.append((routing_score, teacher_id, teacher_status))

        passed.sort(key=lambda item: item[0], reverse=True)
        top_k = int(getattr(self.args, 'rgftd_v3_teacher_pool_topk', 1))
        if top_k != 1:
            raise ValueError('RGFTD-v3 first full version currently only supports top_k=1')

        if passed:
            _, selected_teacher_id, selected_status = passed[0]
            selected_status['v3_pool_size'] = float(len(passed))
            selected_status['v3_pool_stable'] = float(len(passed)) if self._rgftd_v3_stable_enabled() else 0.0
            selected_status['v3_no_teacher'] = 0.0
            self.rgftd_v3_selected_teacher_id = selected_teacher_id
            status = selected_status
        else:
            status = best_teacher_status
            status['v3_pool_size'] = 0.0
            status['v3_no_teacher'] = 1.0
            status['v3_selected_teacher'] = -1.0
            status['v3_selected_score'] = 0.0
            status['v3_best_failed_teacher'] = best_teacher_status['v3_selected_teacher']
            status['v3_best_failed_score'] = best_teacher_status['v3_selected_score']
            self.rgftd_v3_selected_teacher_id = None

        self.rgftd_v3_last_audit_iter = self.current_iter
        self.rgftd_v3_status = status
        return status


    def _train(self, config):
        self.model.train()
        if self._rgftd_v3_stable_enabled():
            self.rgftd_stable_updated = 0.0
            self.rgftd_stable_pending_upload = False

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
        rgftd_teacher_state_items = []
        rgftd_pool_probe_model = None
        rgftd_teacher_args = self.args
        rgftd_v3_status = _default_rgftd_v3_status()
        if int(getattr(self.args, 'rgftd_enabled', 0)) == 1:
            if self._rgftd_v3_enabled():
                if self._should_run_rgftd_v3_audit():
                    rgftd_v3_status = self._run_rgftd_v3_routing_audit(wann_ref_model)
                else:
                    rgftd_v3_status = dict(getattr(self, 'rgftd_v3_status', _default_rgftd_v3_status()))
                teacher_state_dicts = getattr(self.model, 'rgftd_teacher_state_dicts', {})
                selected_teacher_id = getattr(self, 'rgftd_v3_selected_teacher_id', None)
                rdsi_candidate_ids = self._rgftd_v3_candidate_teacher_ids() if self._rdsi_enabled() else []
                if self._rdsi_enabled():
                    if rdsi_candidate_ids:
                        for teacher_id in rdsi_candidate_ids:
                            rgftd_teacher_state_items.append((teacher_id, teacher_state_dicts[teacher_id]))
                        rgftd_v3_status['v3_no_teacher'] = 0.0
                        rgftd_v3_status['v3_fallback_teacher'] = 0.0
                        rgftd_v3_status['v3_teacher_source'] = 3.0
                        rgftd_v3_status['v3_pool_size'] = float(len(rdsi_candidate_ids))
                        rgftd_v3_status['v3_pool_stable'] = float(len(rdsi_candidate_ids)) if self._rgftd_v3_stable_enabled() else 0.0
                        rgftd_v3_status['v3_selected_teacher'] = -3.0
                        rgftd_v3_status['v3_selected_score'] = 0.0
                        rgftd_pool_probe_model = self._build_teacher_model_from_state_dict(
                            teacher_state_dicts[rdsi_candidate_ids[0]]
                        )
                    else:
                        rgftd_v3_status['v3_no_teacher'] = 1.0
                        rgftd_v3_status['v3_fallback_teacher'] = 0.0
                        rgftd_v3_status['v3_teacher_source'] = 0.0
                        rgftd_v3_status['v3_selected_teacher'] = -1.0
                elif selected_teacher_id is not None and selected_teacher_id in teacher_state_dicts:
                    pair_key = self._rgftd_v3_pair_key(1, selected_teacher_id)
                    benefit_score = self._rgftd_v3_get_score(self.rgftd_v3_benefit_scores, pair_key, 1.0)
                    revoke_thresh = float(getattr(self.args, 'rgftd_v3_benefit_revoke_thresh', 0.35))
                    if int(getattr(self.args, 'rgftd_v3_benefit_enabled', 1)) == 1 and benefit_score < revoke_thresh:
                        rgftd_v3_status['v3_no_teacher'] = 1.0
                        rgftd_v3_status['v3_fallback_teacher'] = 0.0
                        rgftd_v3_status['v3_teacher_source'] = 0.0
                        rgftd_v3_status['v3_selected_teacher'] = -1.0
                        rgftd_v3_status['v3_teacher_revoke'] = 1.0
                        self.rgftd_v3_selected_teacher_id = None
                    else:
                        cap_scale = self._rgftd_v3_cap_scale_for_score(benefit_score)
                        cap_current = self._rgftd_v3_base_cap(False) * cap_scale
                        rgftd_teacher_args = self._rgftd_args_for_teacher_source(
                            fallback_teacher=False,
                            cap_scale=cap_scale,
                        )
                        rgftd_teacher_model = self._build_teacher_model_from_state_dict(
                            teacher_state_dicts[selected_teacher_id]
                        )
                        rgftd_v3_status['v3_no_teacher'] = 0.0
                        rgftd_v3_status['v3_fallback_teacher'] = 0.0
                        rgftd_v3_status['v3_teacher_source'] = 1.0
                        rgftd_v3_status = self._rgftd_v3_fill_runtime_status(
                            rgftd_v3_status,
                            1,
                            selected_teacher_id,
                            cap_current,
                        )
                else:
                    fallback_state_dict = getattr(self.model, 'rgftd_teacher_state_dict', None)
                    fallback_flag = int(getattr(self.args, 'rgftd_v3_server_ema_fallback', -1))
                    if fallback_flag < 0:
                        fallback_allowed = not self._rgftd_v3_stable_enabled()
                    else:
                        fallback_allowed = fallback_flag == 1
                    if fallback_allowed and fallback_state_dict is not None:
                        pair_key = self._rgftd_v3_pair_key(2, -2)
                        benefit_score = self._rgftd_v3_routing_benefit(pair_key)
                        cap_scale = self._rgftd_v3_cap_scale_for_score(benefit_score)
                        cap_current = self._rgftd_v3_base_cap(True) * cap_scale
                        rgftd_teacher_model = self._build_teacher_model_from_state_dict(fallback_state_dict)
                        rgftd_teacher_args = self._rgftd_args_for_teacher_source(
                            fallback_teacher=True,
                            cap_scale=cap_scale,
                        )
                        rgftd_v3_status['v3_no_teacher'] = 0.0
                        rgftd_v3_status['v3_fallback_teacher'] = 1.0
                        rgftd_v3_status['v3_teacher_source'] = 2.0
                        rgftd_v3_status['v3_selected_teacher'] = -2.0
                        rgftd_v3_status['v3_selected_score'] = 0.0
                        rgftd_v3_status = self._rgftd_v3_fill_runtime_status(
                            rgftd_v3_status,
                            2,
                            -2,
                            cap_current,
                        )
                    else:
                        rgftd_v3_status['v3_no_teacher'] = 1.0
                        rgftd_v3_status['v3_fallback_teacher'] = 0.0
                        rgftd_v3_status['v3_teacher_source'] = 0.0
                        rgftd_v3_status['v3_selected_teacher'] = -1.0
            else:
                teacher_state_dict = getattr(self.model, 'rgftd_teacher_state_dict', None)
                if teacher_state_dict is not None:
                    rgftd_teacher_model = self._build_teacher_model_from_state_dict(teacher_state_dict)
        
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
            elif self.args.img_class == 'odoc' or self.args.img_class == 'odoc_binary' or self.args.img_class == 'polyp' or self.args.img_class == 'isic' or self.args.img_class == 'busi' or self.args.img_class == 'tn3k' or self.args.img_class == 'duts' or self.args.img_class == 'glas' or self.args.img_class == 'ebhiseg':
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
                    agc_only_profile = zero_acg_profile(outputs.device)
                    if (
                        int(getattr(self.args, 'agc_enabled', 0)) == 1
                        and int(getattr(self.args, 'acg_enabled', 0)) == 0
                    ):
                        wann_maps, agc_only_profile = complete_wann_maps_with_agc(
                            outputs,
                            wann_maps,
                            self.args,
                            self.current_iter,
                        )
                    wann_target_label = getattr(wann_maps, 'target_label', label_batch)
                    loss_ce_seg = wann_weighted_ce_loss(
                        outputs, wann_target_label, wann_maps.core_weight, ignore_index=self.args.num_classes
                    )
                    loss_ce_auxiliary = wann_weighted_ce_loss(
                        outputs_auxiliary, wann_target_label, wann_maps.core_weight, ignore_index=self.args.num_classes
                    )
                    loss_hard = 0.5 * (loss_ce_seg + loss_ce_auxiliary)
                    core_ratio = _scalar_float(wann_maps.profile.get('core_ratio', 0.0))
                    target_core_ratio = float(getattr(self.args, 'wann_target_core_ratio', 0.0))
                    core_deficit = max(0.0, target_core_ratio - core_ratio)
                    soft_boost = 1.0 + float(getattr(self.args, 'wann_core_deficit_soft_boost', 0.0)) * core_deficit
                    lambda_soft = float(getattr(self.args, 'wann_soft_lambda', 0.2)) * ramps.sigmoid_rampup(
                        self.current_iter, int(getattr(self.args, 'wann_soft_rampup_iters', 800))
                    ) * soft_boost
                    lambda_cons = float(getattr(self.args, 'wann_cons_lambda', 0.05)) * ramps.sigmoid_rampup(
                        self.current_iter, int(getattr(self.args, 'wann_cons_rampup_iters', 800))
                    )
                    loss_soft = wann_soft_band_loss(
                        outputs, outputs_auxiliary, label_batch, wann_maps, ignore_index=self.args.num_classes
                    )
                    loss_cons = wann_consistency_loss(outputs, outputs_auxiliary, wann_maps.ignore_mask)
                    loss_acg = outputs.sum() * 0.0
                    lambda_acg = 0.0
                    acg_profile = zero_acg_profile(outputs.device)
                    if int(getattr(self.args, 'acg_enabled', 0)) == 1:
                        loss_acg, lambda_acg, acg_profile = acg_loss(
                            outputs,
                            outputs_auxiliary,
                            wann_maps,
                            self.args,
                            self.current_iter,
                        )
                        loss_hard = loss_acg
                    loss_ce = loss_hard + lambda_soft * loss_soft + lambda_cons * loss_cons
                    loss_rgftd = outputs.sum() * 0.0
                    loss_rgftd_seg = outputs.sum() * 0.0
                    loss_rgftd_aux = outputs.sum() * 0.0
                    lambda_rgftd_raw = 0.0
                    lambda_rgftd = 0.0
                    rgftd_profile = zero_rgftd_profile(outputs.device)
                    if int(getattr(self.args, 'rgftd_enabled', 0)) == 1:
                        lambda_rgftd_raw = get_rgftd_lambda(self.current_iter, self.args)
                    has_rgftd_teacher = rgftd_teacher_model is not None or len(rgftd_teacher_state_items) > 0
                    if int(getattr(self.args, 'rgftd_enabled', 0)) == 1 and has_rgftd_teacher:
                        light_audit_trigger = (
                            core_ratio < float(getattr(self.args, 'rgftd_light_audit_core_ratio_thresh', 0.05))
                            and _scalar_float(wann_maps.profile.get('max_prob_low_r', 0.0)) >
                            float(getattr(self.args, 'rgftd_light_audit_low_maxp_thresh', 0.95))
                        )
                        light_audit_active = (
                            float(lambda_rgftd_raw) <= 0.0
                            and int(getattr(self.args, 'rgftd_light_audit_enabled', 0)) == 1
                            and int(self.current_iter) >= int(getattr(self.args, 'rgftd_light_audit_start_iters', 0))
                            and light_audit_trigger
                        )
                        if light_audit_active:
                            rgftd_v3_status['v3_light_audit_active'] = 1.0
                        if float(lambda_rgftd_raw) > 0.0:
                            rdsi_profile = None
                            if len(rgftd_teacher_state_items) > 0:
                                if rgftd_pool_probe_model is None:
                                    rgftd_pool_probe_model = self._build_teacher_model_from_state_dict(
                                        rgftd_teacher_state_items[0][1]
                                    )
                                teacher_logits_list = []
                                teacher_feature_list = []
                                teacher_ids = []
                                with torch.no_grad():
                                    for teacher_id, teacher_state_dict in rgftd_teacher_state_items:
                                        rgftd_pool_probe_model.load_state_dict(teacher_state_dict, strict=False)
                                        teacher_out = rgftd_pool_probe_model(volume_batch)
                                        teacher_logits_list.append(_primary_logits(teacher_out).detach())
                                        teacher_feature_list.append(_rdsi_feature(teacher_out).detach())
                                        teacher_ids.append(teacher_id)
                                    rgftd_teacher_args = copy.copy(rgftd_teacher_args)
                                    selected_logits, rdsi_profile = select_rdsi_teacher_logits(
                                        outputs,
                                        teacher_logits_list,
                                        teacher_ids,
                                        label_batch,
                                        wann_maps,
                                        rgftd_teacher_args,
                                        self.current_iter,
                                        student_feature=de1,
                                        teacher_feature_list=teacher_feature_list,
                                    )
                                loss_rgftd_seg, lambda_rgftd_seg, rgftd_profile_seg = rgftd_loss(
                                    outputs,
                                    selected_logits,
                                    label_batch,
                                    wann_maps,
                                    rgftd_teacher_args,
                                    self.current_iter,
                                    image=volume_batch,
                                )
                                loss_rgftd_aux, lambda_rgftd_aux, rgftd_profile_aux = rgftd_loss(
                                    outputs_auxiliary,
                                    selected_logits,
                                    label_batch,
                                    wann_maps,
                                    rgftd_teacher_args,
                                    self.current_iter,
                                    image=volume_batch,
                                )
                                loss_rgftd_proto, lambda_rgftd_proto, rgftd_profile_proto = rdsi_feature_prototype_loss(
                                    de1,
                                    teacher_feature_list,
                                    rgftd_teacher_args,
                                    self.current_iter,
                                )
                                loss_rgftd = 0.5 * (loss_rgftd_seg + loss_rgftd_aux) + loss_rgftd_proto
                                loss_rgftd_weighted = (
                                    0.5 * (
                                        float(lambda_rgftd_seg) * loss_rgftd_seg
                                        + float(lambda_rgftd_aux) * loss_rgftd_aux
                                    )
                                    + float(lambda_rgftd_proto) * loss_rgftd_proto
                                )
                                lambda_rgftd = (
                                    float(lambda_rgftd_seg)
                                    + float(lambda_rgftd_aux)
                                    + float(lambda_rgftd_proto)
                                ) / 3.0
                                rgftd_profile = _average_rgftd_profiles(rgftd_profile_seg, rgftd_profile_aux)
                                for proto_key, proto_value in rgftd_profile_proto.items():
                                    if str(proto_key).startswith('rdsi_proto'):
                                        rgftd_profile[proto_key] = proto_value
                            else:
                                with torch.no_grad():
                                    teacher_out = rgftd_teacher_model(volume_batch)
                                    teacher_logits = _primary_logits(teacher_out)
                                loss_rgftd_seg, lambda_rgftd_seg, rgftd_profile_seg = rgftd_loss(
                                    outputs,
                                    teacher_logits,
                                    label_batch,
                                    wann_maps,
                                    rgftd_teacher_args,
                                    self.current_iter,
                                    image=volume_batch,
                                )
                                loss_rgftd_aux, lambda_rgftd_aux, rgftd_profile_aux = rgftd_loss(
                                    outputs_auxiliary,
                                    teacher_logits,
                                    label_batch,
                                    wann_maps,
                                    rgftd_teacher_args,
                                    self.current_iter,
                                    image=volume_batch,
                                )
                                loss_rgftd = 0.5 * (loss_rgftd_seg + loss_rgftd_aux)
                                loss_rgftd_weighted = 0.5 * (
                                    float(lambda_rgftd_seg) * loss_rgftd_seg
                                    + float(lambda_rgftd_aux) * loss_rgftd_aux
                                )
                                lambda_rgftd = 0.5 * (float(lambda_rgftd_seg) + float(lambda_rgftd_aux))
                                rgftd_profile = _average_rgftd_profiles(rgftd_profile_seg, rgftd_profile_aux)
                            if rdsi_profile is not None:
                                for rdsi_key, rdsi_value in rdsi_profile.items():
                                    if str(rdsi_key).startswith('rdsi_'):
                                        rgftd_profile[rdsi_key] = rdsi_value
                                rdsi_generic_map = {
                                    'candidate_ratio': 'rdsi_candidate_ratio',
                                    'teacher_accept_ratio': 'rdsi_accept_ratio',
                                    'teacher_reject_ratio': 'rdsi_reject_ratio',
                                    'reject_by_core_conflict': 'rdsi_reject_by_core',
                                    'reject_by_support': 'rdsi_reject_by_seed',
                                    'reject_by_fg_ratio': 'rdsi_reject_by_prior',
                                    'release_score_mean': 'rdsi_selected_score_mean',
                                    'release_score_top': 'rdsi_selected_score_top',
                                }
                                for generic_key, rdsi_key in rdsi_generic_map.items():
                                    if rdsi_key in rdsi_profile:
                                        rgftd_profile[generic_key] = rdsi_profile[rdsi_key]
                            rgftd_profile['teacher_active_loss'] = loss_rgftd_weighted.detach()
                            rgftd_profile['rdsi_loss_raw'] = loss_rgftd.detach()
                            rgftd_profile['rdsi_loss_weighted'] = rgftd_profile['teacher_active_loss'].detach()
                            loss_ce = loss_ce + loss_rgftd_weighted
                            if self._rgftd_v3_enabled():
                                rgftd_v3_status = self._rgftd_v3_update_online_benefit(
                                    rgftd_v3_status,
                                    rgftd_profile,
                                    wann_maps,
                                )
                    self._rgftd_update_stable_teacher(outputs, label_batch, wann_maps)
                    rgftd_v3_status['stable_valid'] = float(self.rgftd_stable_valid)
                    rgftd_v3_status['stable_score'] = float(self.rgftd_stable_last_score)
                    rgftd_v3_status['stable_best_score'] = float(self.rgftd_stable_best_score)
                    rgftd_v3_status['stable_updated'] = float(self.rgftd_stable_updated)
                    rgftd_profile = _attach_rgftd_v3_status(rgftd_profile, rgftd_v3_status, outputs.device)
                else:
                    wann_maps = None
                    lambda_soft = 0.0
                    lambda_cons = 0.0
                    lambda_rgftd_raw = 0.0
                    lambda_rgftd = 0.0
                    loss_hard = torch.tensor(0.0).cuda()
                    loss_soft = torch.tensor(0.0).cuda()
                    loss_cons = torch.tensor(0.0).cuda()
                    loss_acg = torch.tensor(0.0).cuda()
                    lambda_acg = 0.0
                    acg_profile = zero_acg_profile(loss_acg.device)
                    loss_rgftd = torch.tensor(0.0).cuda()
                    loss_rgftd_seg = torch.tensor(0.0).cuda()
                    loss_rgftd_aux = torch.tensor(0.0).cuda()
                    rgftd_profile = zero_rgftd_profile(loss_rgftd.device)
                    rgftd_profile = _attach_rgftd_v3_status(rgftd_profile, rgftd_v3_status, loss_rgftd.device)
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
            max_grad_norm = float(getattr(self.args, 'max_grad_norm', 0.0))
            skip_update = not torch.isfinite(loss.detach()).all().item()
            if skip_update:
                log(logging.WARNING, 'client %d : iteration %d : non-finite loss detected before backward' % (
                    self.cid, self.current_iter + 1
                ))
                loss = torch.nan_to_num(loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                loss_ce = torch.nan_to_num(loss_ce.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            elif self.amp:
                self.scaler.scale(loss).backward()
                if max_grad_norm > 0.0:
                    self.scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    if not torch.isfinite(grad_norm.detach()).all().item():
                        log(logging.WARNING, 'client %d : iteration %d : non-finite grad norm detected before optimizer step' % (
                            self.cid, self.current_iter + 1
                        ))
                        optimizer.zero_grad()
                        skip_update = True
                if not skip_update:
                    self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if max_grad_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    if not torch.isfinite(grad_norm.detach()).all().item():
                        log(logging.WARNING, 'client %d : iteration %d : non-finite grad norm detected before optimizer step' % (
                            self.cid, self.current_iter + 1
                        ))
                        optimizer.zero_grad()
                        skip_update = True
                if not skip_update:
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
                    acg_profile,
                ))

            lr_ = self.args.base_lr * (1.0 - self.current_iter / self.args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            self.current_lr = lr_

        # pack general metrics
        sample_idx = 1 if volume_batch.shape[0] > 1 else 0
        image = volume_batch[sample_idx, :, :, :]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
        outputs = outputs[sample_idx, ...] * 50
        labs = label_batch[sample_idx, ...].unsqueeze(0) * 50
        if self.args.img_class == 'odoc' or self.args.img_class == 'odoc_binary' or self.args.img_class == 'polyp' or self.args.img_class == 'isic' or self.args.img_class == 'busi' or self.args.img_class == 'tn3k' or self.args.img_class == 'duts' or self.args.img_class == 'glas' or self.args.img_class == 'ebhiseg':
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
            if self.args.img_class == 'odoc' or self.args.img_class == 'odoc_binary' or self.args.img_class == 'polyp' or self.args.img_class == 'isic' or self.args.img_class == 'busi' or self.args.img_class == 'tn3k' or self.args.img_class == 'duts' or self.args.img_class == 'glas' or self.args.img_class == 'ebhiseg':
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

        if int(getattr(self.args, 'acg_enabled', 0)) == 1:
            metrics_['client_{}_acg_loss'.format(self.cid)] = float(loss_acg.detach().cpu().item())
            metrics_['client_{}_acg_lambda'.format(self.cid)] = float(lambda_acg)
            for key, value in acg_profile.items():
                metrics_['client_{}_acg_{}'.format(self.cid, key)] = float(value.detach().cpu().item())

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
            if (
                self._rgftd_v3_stable_enabled()
                and self.rgftd_stable_pending_upload
                and self.rgftd_stable_snapshot_arrays is not None
            ):
                metrics_['client_{}_rgftd_stable_snapshot'.format(self.cid)] = _pack_rgftd_stable_snapshot(
                    self.rgftd_stable_snapshot_arrays
                )

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
            elif args.img_class == 'odoc' or args.img_class == 'odoc_binary' or args.img_class == 'polyp' or args.img_class == 'isic' or args.img_class == 'busi' or args.img_class == 'tn3k' or args.img_class == 'duts' or args.img_class == 'glas' or args.img_class == 'ebhiseg':
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
                sample_idx = 1 if volume_batch.shape[0] > 1 else 0
                image = volume_batch[sample_idx, :, :, :]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                outputs = outputs[sample_idx, ...] * 50
                labs = label_batch[sample_idx, ...].unsqueeze(0) * 50
                if args.img_class == 'odoc' or args.img_class == 'odoc_binary' or args.img_class == 'polyp' or args.img_class == 'isic' or args.img_class == 'busi' or args.img_class == 'tn3k' or args.img_class == 'duts' or args.img_class == 'glas' or args.img_class == 'ebhiseg':
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
    parser.add_argument('--disable_ala', type=int, default=0,
                        help='Skip ALA/FedALA local interpolation when set to 1')
    parser.add_argument('--disable_tensorboard', type=int, default=0,
                        help='Disable TensorBoard event writing when set to 1')
    parser.add_argument('--learnable_nwr_enabled', type=int, default=0,
                        help='Enable direct learnable NWR client aggregation')
    parser.add_argument('--learnable_nwr_hidden_dim', type=int, default=16,
                        help='Hidden dimension of the shared learnable NWR MLP')
    parser.add_argument('--learnable_nwr_lr', type=float, default=1e-3,
                        help='Learning rate for the server-side learnable NWR MLP')
    parser.add_argument('--learnable_nwr_tau', type=float, default=1.0,
                        help='Softmax temperature for direct learnable NWR aggregation weights')
    parser.add_argument('--learnable_nwr_meta_temp', type=float, default=0.05,
                        help='Temperature for converting marginal meta-loss contributions into learnable NWR targets')
    parser.add_argument('--learnable_nwr_meta_fraction', type=float, default=0.1,
                        help='Fraction of each client training set reserved for learnable NWR meta-validation')
    parser.add_argument('--save_code_snapshot', type=int, default=1,
                        help='Copy code into snapshot_path/code when set to 1')
    parser.add_argument('--save_checkpoint_copies', type=int, default=1,
                        help='Save per-iteration checkpoint copies when set to 1; best_model files are still updated')
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
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='clip gradient norm before optimizer step; <=0 disables clipping')
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
    parser.add_argument('--wann_sparse_adaptive_core', type=int, default=0,
                        help='Use source-free adaptive core quota for sparse scribble support')
    parser.add_argument('--wann_sparse_min_core_ratio', type=float, default=0.0,
                        help='Minimum per-image hard core ratio under sparse scribble support')
    parser.add_argument('--wann_sparse_max_core_ratio', type=float, default=0.0,
                        help='Maximum per-image hard core ratio under sparse scribble support')
    parser.add_argument('--wann_sparse_core_min_reliability', type=float, default=0.25,
                        help='Minimum reliability for sparse adaptive core candidates')
    parser.add_argument('--wann_low_confident_thresh', type=float, default=0.95,
                        help='Confidence threshold for high-confidence low-reliability WANN risk pixels')
    parser.add_argument('--wann_target_core_ratio', type=float, default=0.0,
                        help='Target core ratio used to scale soft supervision when WANN core is deficient')
    parser.add_argument('--wann_core_deficit_soft_boost', type=float, default=0.0,
                        help='Multiplier slope for soft-band supervision when WANN hard core is deficient')
    parser.add_argument('--acg_enabled', type=int, default=0,
                        help='Enable annotation geometry calibration via normalized weak risk')
    parser.add_argument('--acg_lambda', type=float, default=0.08,
                        help='Legacy compatibility argument; ACG-NWR replaces hard weak risk when enabled')
    parser.add_argument('--acg_warmup_iters', type=int, default=800,
                        help='Legacy compatibility argument for previous ACG variants')
    parser.add_argument('--acg_context_soft_foreground', type=int, default=0,
                        help='Use seed-anchored soft foreground margin for WANN context foreground risk')
    parser.add_argument('--acg_context_margin_floor', type=float, default=0.35,
                        help='Minimum foreground probability target for ACG context foreground')
    parser.add_argument('--acg_context_margin_ceiling', type=float, default=0.85,
                        help='Maximum foreground probability target for ACG context foreground')
    parser.add_argument('--acg_context_band_width', type=float, default=0.15,
                        help='Reliability-bounded context foreground band width above the lower target')
    parser.add_argument('--acg_context_band_over_weight', type=float, default=0.25,
                        help='Relative penalty for context foreground probability above the reliability band')
    parser.add_argument('--acg_prior_mode', type=str, default='sqrt',
                        choices=['sqrt', 'mass', 'uniform', 'raw_mean'],
                        help='Mechanism-control prior for seed/context/background risk calibration')
    parser.add_argument('--agc_enabled', type=int, default=0,
                        help='Enable anchored geometry completion between WANN and SCM/ACG')
    parser.add_argument('--agc_mode', type=str, default='conservative',
                        choices=['conservative', 'moderate'],
                        help='AGC preset: conservative is precision-first, moderate is recall-releasing')
    parser.add_argument('--agc_spatial_rule', type=str, default='mode',
                        choices=['mode', 'confidence_only', 'candidate_only', 'connected_only',
                                 'candidate_or_connected', 'candidate_and_connected'],
                        help='Ablation switch for AGC spatial anchoring; mode preserves the preset behavior')
    parser.add_argument('--agc_disable_cap', type=int, default=0,
                        help='Ablation switch that disables the AGC top-k/area cap when set to 1')
    parser.add_argument('--agc_disable_conf_gate', type=int, default=0,
                        help='Ablation switch that removes the confidence threshold from AGC completion')
    parser.add_argument('--agc_complete_as_seed', type=int, default=0,
                        help='Ablation switch that assigns AGC completion pixels to seed foreground instead of context foreground')
    parser.add_argument('--agc_start_iter', type=int, default=1000,
                        help='Iteration to start AGC completed-context release')
    parser.add_argument('--agc_ramp_iters', type=int, default=800,
                        help='Sigmoid ramp iterations after AGC start')
    parser.add_argument('--agc_connect_steps', type=int, default=64,
                        help='Maximum geodesic dilation steps for seed-connected model foreground')
    parser.add_argument('--agc_conservative_tau', type=float, default=0.75,
                        help='Foreground probability threshold for conservative AGC')
    parser.add_argument('--agc_conservative_max_target_mult', type=float, default=0.5,
                        help='Conservative AGC cap as a multiple of original WANN target foreground pixels')
    parser.add_argument('--agc_conservative_max_image_ratio', type=float, default=0.03,
                        help='Conservative AGC cap as an image-area ratio')
    parser.add_argument('--agc_conservative_weight', type=float, default=0.5,
                        help='Geometry weight assigned to conservative AGC completed context')
    parser.add_argument('--agc_conservative_reliability', type=float, default=0.65,
                        help='Reliability target assigned to conservative AGC completed context')
    parser.add_argument('--agc_moderate_tau', type=float, default=0.65,
                        help='Foreground probability threshold for moderate AGC')
    parser.add_argument('--agc_moderate_max_target_mult', type=float, default=1.5,
                        help='Moderate AGC cap as a multiple of original WANN target foreground pixels')
    parser.add_argument('--agc_moderate_max_image_ratio', type=float, default=0.08,
                        help='Moderate AGC cap as an image-area ratio')
    parser.add_argument('--agc_moderate_weight', type=float, default=0.35,
                        help='Geometry weight assigned to moderate AGC completed context')
    parser.add_argument('--agc_moderate_reliability', type=float, default=0.55,
                        help='Reliability target assigned to moderate AGC completed context')
    parser.add_argument('--rgftd_enabled', type=int, default=0,
                        help='Enable reliability-gated federated teacher distillation')
    parser.add_argument('--rgftd_lambda', type=float, default=0.1,
                        help='Maximum RGFTD distillation loss weight')
    parser.add_argument('--rgftd_warmup_iters', type=int, default=800,
                        help='Iterations before RGFTD loss is released')
    parser.add_argument('--rgftd_rampup_iters', type=int, default=800,
                        help='Rampup iterations after RGFTD warmup')
    parser.add_argument('--rgftd_light_audit_enabled', type=int, default=0,
                        help='Enable WANN-risk-triggered light RGFTD before the main release schedule')
    parser.add_argument('--rgftd_light_audit_start_iters', type=int, default=0,
                        help='Iteration to start WANN-risk-triggered light RGFTD audit')
    parser.add_argument('--rgftd_light_audit_lambda', type=float, default=0.01,
                        help='Distillation lambda for WANN-risk-triggered light RGFTD audit')
    parser.add_argument('--rgftd_light_audit_core_ratio_thresh', type=float, default=0.05,
                        help='Core ratio threshold that marks WANN hard supervision deficiency')
    parser.add_argument('--rgftd_light_audit_low_maxp_thresh', type=float, default=0.95,
                        help='Low-reliability confidence threshold that triggers early RGFTD audit')
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
    parser.add_argument('--rgftd_spatial_support_enabled', type=int, default=1,
                        help='Softly weight RGFTD foreground release by target-side WANN candidate or seed-near support')
    parser.add_argument('--rgftd_spatial_support_radius', type=int, default=2,
                        help='Dilation radius around target foreground seed support for RGFTD spatial weighting')
    parser.add_argument('--rgftd_spatial_candidate_weight', type=float, default=1.0,
                        help='Soft spatial weight for teacher foreground pixels in target foreground candidate regions')
    parser.add_argument('--rgftd_spatial_near_seed_weight', type=float, default=0.75,
                        help='Soft spatial weight for teacher foreground pixels near target foreground seeds')
    parser.add_argument('--rgftd_spatial_far_weight', type=float, default=0.15,
                        help='Soft spatial weight for teacher foreground pixels far from target support but inside RGFTD region')
    parser.add_argument('--rgftd_max_bg_fg_ratio', type=float, default=1.0,
                        help='Maximum allowed active background-to-foreground pixel ratio for RGFTD')
    parser.add_argument('--rgftd_allow_bg_without_fg', type=int, default=0,
                        help='Whether RGFTD may keep background pixels when no foreground pixels survive')
    parser.add_argument('--rgftd_lambda_eff_cap', type=float, default=0.02,
                        help='Upper bound on effective RGFTD lambda after release and safety scaling')
    parser.add_argument('--rgftd_refine_enabled', type=int, default=0,
                        help='Enable target-side soft refinement before RGFTD KL distillation')
    parser.add_argument('--rgftd_refine_iters', type=int, default=3,
                        help='Local affinity propagation iterations for RGFTD refined soft target')
    parser.add_argument('--rgftd_refine_affinity_sigma', type=float, default=0.75,
                        help='Intensity-affinity bandwidth for RGFTD target-side refinement')
    parser.add_argument('--rgftd_refine_affinity_mix', type=float, default=0.35,
                        help='Per-iteration mixing strength for RGFTD target-side refinement')
    parser.add_argument('--rgftd_refine_seed_strength', type=float, default=0.95,
                        help='Strength used to clamp target weak seeds in RGFTD refined target')
    parser.add_argument('--rgftd_refine_core_anchor_radius', type=int, default=1,
                        help='Radius used to include nearby WANN core pixels as fixed anchors during RGFTD refinement')
    parser.add_argument('--rgftd_refine_unsupported_fg_scale', type=float, default=0.25,
                        help='Foreground probability scale for active teacher islands unsupported by target seed/candidate structure')
    parser.add_argument('--rgftd_refine_fg_floor', type=float, default=0.02,
                        help='Minimum foreground probability kept in active foreground refinement regions')
    parser.add_argument('--rgftd_refine_bg_ceiling', type=float, default=0.98,
                        help='Maximum background probability allowed inside RGFTD refinement ROI')
    parser.add_argument('--rgftd_refine_min_fg_mass', type=float, default=1.0,
                        help='Minimum refined foreground mass required to keep RGFTD active')
    parser.add_argument('--rgftd_refine_min_roi_pixels', type=float, default=1.0,
                        help='Minimum refinement ROI pixels required to keep RGFTD active')
    parser.add_argument('--rgftd_v3_enabled', type=int, default=0,
                        help='Enable RGFTD-v3 target-aware teacher routing')
    parser.add_argument('--rdsi_enabled', type=int, default=1,
                        help='Enable RDSI region-wise domain-specialist teacher selection inside RGFTD-v3')
    parser.add_argument('--rdsi_class_aware', type=int, default=-1,
                        help='Use class-aware RDSI scoring; -1 enables it automatically when num_classes > 2')
    parser.add_argument('--rdsi_teacher_sup_types', type=str, default='',
                        help='Comma-separated per-client weak-label types used only for RDSI teacher-type diagnostics')
    parser.add_argument('--rdsi_residual_alpha', type=float, default=0.35,
                        help='Maximum residual intervention strength for benefit-validated RDSI teacher targets')
    parser.add_argument('--rdsi_hard_core_conf_thresh', type=float, default=0.90,
                        help='Student confidence required for WANN core pixels to be treated as hard preserve core')
    parser.add_argument('--rdsi_hard_core_entropy_thresh', type=float, default=0.25,
                        help='Maximum student entropy for WANN core pixels to be treated as hard preserve core')
    parser.add_argument('--rdsi_hard_core_reliability_thresh', type=float, default=0.65,
                        help='Minimum WANN reliability for WANN core pixels to be treated as hard preserve core')
    parser.add_argument('--rdsi_benefit_topk_ratio', type=float, default=0.002,
                        help='Per-image top-ratio of benefit-validated RDSI regions allowed to receive teacher intervention')
    parser.add_argument('--rdsi_benefit_topk_min_pixels', type=int, default=8,
                        help='Minimum pixels selected by benefit-validated RDSI per image when candidates exist')
    parser.add_argument('--rdsi_benefit_topk_max_pixels', type=int, default=4096,
                        help='Maximum pixels selected by benefit-validated RDSI per image; <=0 means no cap')
    parser.add_argument('--rdsi_benefit_score_floor', type=float, default=1e-6,
                        help='Minimum benefit-validated RDSI score required before top-k selection')
    parser.add_argument('--rdsi_budget_fraction', type=float, default=0.20,
                        help='Fraction of the current RDSI risk region assigned to budgeted residual transfer')
    parser.add_argument('--rdsi_budget_min_ratio', type=float, default=0.003,
                        help='Minimum image-level active ratio for budgeted RDSI residual transfer when candidates exist')
    parser.add_argument('--rdsi_budget_max_ratio', type=float, default=0.015,
                        help='Maximum image-level active ratio for budgeted RDSI residual transfer')
    parser.add_argument('--rdsi_entropy_increase_margin', type=float, default=0.05,
                        help='Allowed teacher foreground entropy increase before RDSI benefit is suppressed')
    parser.add_argument('--rdsi_entropy_increase_scale', type=float, default=0.35,
                        help='Scale used to suppress high-entropy teacher foreground intervention')
    parser.add_argument('--rdsi_fg_excess_margin', type=float, default=0.05,
                        help='Allowed regional foreground increase before RDSI foreground-excess suppression')
    parser.add_argument('--rdsi_fg_excess_scale', type=float, default=0.20,
                        help='Scale used to suppress foreground-excess RDSI teacher intervention')
    parser.add_argument('--rdsi_boundary_radius', type=int, default=-1,
                        help='Local radius used to score boundary/core-damage-aware RDSI regions; <0 follows rgftd_teacher_foreground_radius')
    parser.add_argument('--rdsi_boundary_uncertainty_width', type=float, default=0.25,
                        help='Foreground-mass band width used for RDSI boundary uncertainty scoring')
    parser.add_argument('--rdsi_core_damage_veto', type=float, default=0.30,
                        help='Hard veto threshold for RDSI teacher core-damage proxy')
    parser.add_argument('--rdsi_unsafe_gap_scale', type=float, default=0.40,
                        help='Scale used to normalize RDSI unsafe teacher-student gap')
    parser.add_argument('--rdsi_boundary_support_weight', type=float, default=0.35,
                        help='Positive weight of boundary support in benefit-validated RDSI scoring')
    parser.add_argument('--rdsi_core_preserving_fg_weight', type=float, default=0.25,
                        help='Positive weight of core-preserving foreground support in RDSI scoring')
    parser.add_argument('--rdsi_teacher_reliability_weight', type=float, default=0.20,
                        help='Positive weight of local teacher reliability in RDSI scoring')
    parser.add_argument('--rdsi_student_risk_weight', type=float, default=0.20,
                        help='Positive weight of student risk evidence in RDSI scoring')
    parser.add_argument('--rdsi_core_damage_weight', type=float, default=0.45,
                        help='Negative weight of teacher core-damage proxy in RDSI scoring')
    parser.add_argument('--rdsi_unsafe_gap_weight', type=float, default=0.30,
                        help='Negative weight of unsafe teacher-student gap in RDSI scoring')
    parser.add_argument('--rdsi_foreground_excess_weight', type=float, default=0.25,
                        help='Negative weight of foreground-excess proxy in RDSI scoring')
    parser.add_argument('--rdsi_knowledge_tiebreak_weight', type=float, default=0.0,
                        help='Deprecated logging-compatible weight; RDSI no longer uses knowledge gap as a positive score term')
    parser.add_argument('--rdsi_safe_budget_gain', type=float, default=1.50,
                        help='Budget multiplier gain when selected RDSI regions have strong safe signal')
    parser.add_argument('--rdsi_unsafe_budget_decay', type=float, default=1.00,
                        help='Budget multiplier decay when selected RDSI candidates have unsafe signal')
    parser.add_argument('--rdsi_max_budget_factor', type=float, default=4.00,
                        help='Maximum adaptive RDSI active-region budget multiplier')
    parser.add_argument('--rdsi_core_reopen_boundary_floor', type=float, default=0.20,
                        help='Student boundary-gradient floor used to reopen uncertain WANN hard core for RDSI')
    parser.add_argument('--rgftd_v3_stable_teacher_enabled', type=int, default=0,
                        help='Use audit-selected stable client snapshots as the RGFTD-v3 teacher bank')
    parser.add_argument('--rgftd_v3_server_ema_fallback', type=int, default=-1,
                        help='Allow RGFTD-v3 server EMA fallback; -1 keeps legacy fallback except stable-teacher mode')
    parser.add_argument('--rgftd_v3_teacher_pool_topk', type=int, default=1,
                        help='Top-k teachers kept by RGFTD-v3 routing; first full version uses top_k=1')
    parser.add_argument('--rgftd_v3_audit_start_iters', type=int, default=800,
                        help='Iteration after which RGFTD-v3 starts target-side routing audit')
    parser.add_argument('--rgftd_v3_audit_interval_iters', type=int, default=1000,
                        help='Low-frequency routing audit interval in iterations; <=0 means one-shot')
    parser.add_argument('--rgftd_v3_empty_pool_retry_iters', type=int, default=10,
                        help='Retry interval after an empty teacher-pool audit before a full routing interval elapses')
    parser.add_argument('--rgftd_v3_audit_batches', type=int, default=4,
                        help='Number of local batches used by each RGFTD-v3 routing audit')
    parser.add_argument('--rgftd_v3_audit_score_thresh', type=float, default=0.05,
                        help='Minimum routing score required for a teacher to enter the RGFTD-v3 pool')
    parser.add_argument('--rgftd_v3_audit_teacher_reliability_min', type=float, default=0.30,
                        help='Minimum teacher reliability required by RGFTD-v3 routing audit')
    parser.add_argument('--rgftd_v3_fallback_lambda_eff_cap', type=float, default=0.01,
                        help='Effective lambda cap used when RGFTD-v3 falls back to the server EMA teacher')
    parser.add_argument('--rgftd_v3_benefit_enabled', type=int, default=1,
                        help='Enable RGFTD-v3.2 online benefit-aware teacher scoring')
    parser.add_argument('--rgftd_v3_lease_iters', type=int, default=0,
                        help='Deprecated compatibility flag; region-conditional RGFTD does not grant teacher leases')
    parser.add_argument('--rgftd_v3_benefit_momentum', type=float, default=0.80,
                        help='EMA momentum for RGFTD-v3.2 online benefit score')
    parser.add_argument('--rgftd_v3_benefit_good_thresh', type=float, default=0.70,
                        help='Benefit score threshold for renewing a teacher at full cap')
    parser.add_argument('--rgftd_v3_benefit_decay_thresh', type=float, default=0.50,
                        help='Benefit score threshold below which the teacher cap is decayed')
    parser.add_argument('--rgftd_v3_benefit_revoke_thresh', type=float, default=0.35,
                        help='Benefit score threshold below which the selected teacher is revoked')
    parser.add_argument('--rgftd_v3_routing_benefit_floor', type=float, default=0.25,
                        help='Minimum benefit multiplier used by routing audit so teachers can recover after new audits')
    parser.add_argument('--rgftd_v3_recover_audit_score_thresh', type=float, default=0.60,
                        help='Audit score threshold that can restore a previously decayed teacher to the routing floor')
    parser.add_argument('--rgftd_v3_decay_cap_scale', type=float, default=0.50,
                        help='Lambda cap scale for a decayed RGFTD-v3.2 teacher')
    parser.add_argument('--rgftd_v3_min_cap_scale', type=float, default=0.25,
                        help='Minimum lambda cap scale before a teacher is fully revoked')
    parser.add_argument('--rgftd_v3_bgfg_warn_thresh', type=float, default=0.75,
                        help='Background/foreground ratio threshold used by RGFTD-v3.2 stability scoring')
    parser.add_argument('--rgftd_v3_cap_hit_penalty', type=float, default=0.20,
                        help='Stability penalty when effective lambda repeatedly hits the cap')
    parser.add_argument('--rgftd_v3_lowmaxp_delta_thresh', type=float, default=0.02,
                        help='Allowed positive delta of WANN low-R max probability before stability penalty')
    parser.add_argument('--rgftd_v3_wann_mass_drop_thresh', type=float, default=0.05,
                        help='Allowed WANN effective-mass drop before stability penalty')
    parser.add_argument('--rgftd_stable_min_score', type=float, default=0.05,
                        help='Minimum local weak-label proxy score required to mark a stable RGFTD teacher valid')
    parser.add_argument('--rgftd_stable_update_margin', type=float, default=0.01,
                        help='Minimum score improvement required to refresh the stable teacher snapshot')
    parser.add_argument('--rgftd_stable_seed_prob_floor', type=float, default=0.35,
                        help='Foreground seed probability floor used by local stable teacher selection')
    parser.add_argument('--rgftd_stable_seed_margin_floor', type=float, default=0.05,
                        help='Foreground-vs-background seed margin floor used by local stable teacher selection')
    parser.add_argument('--rgftd_stable_max_core_conflict', type=float, default=0.20,
                        help='Maximum core conflict allowed by local stable teacher selection')
    parser.add_argument('--rgftd_stable_lowmaxp_delta_thresh', type=float, default=0.02,
                        help='Allowed positive WANN low-R max-probability delta for stable teacher selection')
    parser.add_argument('--rgftd_stable_wann_mass_drop_thresh', type=float, default=0.05,
                        help='Allowed WANN effective-mass drop for stable teacher selection')
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
    assert args.img_class in ['odoc', 'odoc_binary', 'faz', 'polyp', 'prostate', 'isic', 'busi', 'tn3k', 'duts', 'glas', 'ebhiseg']
    valid_sup_types = ['mask', 'scribble', 'scribble_noisy', 'block', 'box', 'keypoint']
    assert args.sup_type in valid_sup_types or str(args.sup_type).startswith('sparse_scribble_')
    assert args.wann_enabled in [0, 1]
    assert args.ala_max_epochs >= 0
    assert args.disable_ala in [0, 1]
    assert args.learnable_nwr_enabled in [0, 1]
    assert args.learnable_nwr_hidden_dim > 0
    assert args.learnable_nwr_lr > 0.0
    assert args.learnable_nwr_tau > 0.0
    assert args.learnable_nwr_meta_temp > 0.0
    assert 0.0 < args.learnable_nwr_meta_fraction < 1.0
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
    assert args.wann_sparse_adaptive_core in [0, 1]
    assert 0.0 <= args.wann_sparse_min_core_ratio <= 1.0
    assert 0.0 <= args.wann_sparse_max_core_ratio <= 1.0
    assert 0.0 <= args.wann_sparse_core_min_reliability <= args.wann_r_max
    assert 0.0 <= args.wann_low_confident_thresh <= 1.0
    assert 0.0 <= args.wann_target_core_ratio <= 1.0
    assert args.wann_core_deficit_soft_boost >= 0.0
    assert args.acg_enabled in [0, 1]
    assert args.acg_lambda >= 0.0
    assert args.acg_warmup_iters >= 0
    assert args.acg_context_soft_foreground in [0, 1]
    assert 0.0 <= args.acg_context_margin_floor <= 1.0
    assert 0.0 <= args.acg_context_margin_ceiling <= 1.0
    assert args.acg_context_margin_floor <= args.acg_context_margin_ceiling
    assert args.acg_context_band_width >= 0.0
    assert args.acg_context_band_over_weight >= 0.0
    assert args.agc_enabled in [0, 1]
    assert args.agc_mode in ['conservative', 'moderate']
    assert args.agc_spatial_rule in [
        'mode',
        'confidence_only',
        'candidate_only',
        'connected_only',
        'candidate_or_connected',
        'candidate_and_connected',
    ]
    assert args.agc_disable_cap in [0, 1]
    assert args.agc_disable_conf_gate in [0, 1]
    assert args.agc_complete_as_seed in [0, 1]
    assert args.agc_start_iter >= 0
    assert args.agc_ramp_iters >= 0
    assert args.agc_connect_steps >= 0
    assert 0.0 <= args.agc_conservative_tau <= 1.0
    assert args.agc_conservative_max_target_mult >= 0.0
    assert 0.0 <= args.agc_conservative_max_image_ratio <= 1.0
    assert args.agc_conservative_weight >= 0.0
    assert 0.0 <= args.agc_conservative_reliability <= 1.0
    assert 0.0 <= args.agc_moderate_tau <= 1.0
    assert args.agc_moderate_max_target_mult >= 0.0
    assert 0.0 <= args.agc_moderate_max_image_ratio <= 1.0
    assert args.agc_moderate_weight >= 0.0
    assert 0.0 <= args.agc_moderate_reliability <= 1.0
    if args.acg_enabled == 1:
        assert args.wann_enabled == 1
    if args.agc_enabled == 1:
        assert args.wann_enabled == 1
    assert args.rgftd_enabled in [0, 1]
    if args.rgftd_enabled == 1:
        assert args.wann_enabled == 1
        assert args.strategy in ['FedUniV2', 'FedUniV2.1']
    assert args.rgftd_lambda >= 0.0
    assert args.rgftd_warmup_iters >= 0
    assert args.rgftd_rampup_iters >= 0
    assert args.rgftd_light_audit_enabled in [0, 1]
    assert args.rgftd_light_audit_start_iters >= 0
    assert args.rgftd_light_audit_lambda >= 0.0
    assert 0.0 <= args.rgftd_light_audit_core_ratio_thresh <= 1.0
    assert 0.0 <= args.rgftd_light_audit_low_maxp_thresh <= 1.0
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
    assert args.rgftd_spatial_support_enabled in [0, 1]
    assert args.rgftd_spatial_support_radius >= 0
    assert 0.0 <= args.rgftd_spatial_candidate_weight <= 1.0
    assert 0.0 <= args.rgftd_spatial_near_seed_weight <= 1.0
    assert 0.0 <= args.rgftd_spatial_far_weight <= 1.0
    assert args.rgftd_max_bg_fg_ratio >= 0.0
    assert args.rgftd_allow_bg_without_fg in [0, 1]
    assert args.rgftd_lambda_eff_cap >= 0.0
    assert args.rgftd_refine_enabled in [0, 1]
    assert args.rgftd_refine_iters >= 0
    assert args.rgftd_refine_affinity_sigma > 0.0
    assert 0.0 <= args.rgftd_refine_affinity_mix <= 1.0
    assert 0.0 <= args.rgftd_refine_seed_strength <= 1.0
    assert args.rgftd_refine_core_anchor_radius >= 0
    assert 0.0 <= args.rgftd_refine_unsupported_fg_scale <= 1.0
    assert 0.0 <= args.rgftd_refine_fg_floor <= 1.0
    assert 0.0 <= args.rgftd_refine_bg_ceiling <= 1.0
    assert args.rgftd_refine_min_fg_mass >= 0.0
    assert args.rgftd_refine_min_roi_pixels >= 0.0
    assert args.rgftd_v3_enabled in [0, 1]
    assert args.rdsi_enabled in [0, 1]
    assert args.rdsi_class_aware in [-1, 0, 1]
    if args.rdsi_class_aware == -1:
        args.rdsi_class_aware = 1 if args.num_classes > 2 else 0
    assert 0.0 <= args.rdsi_residual_alpha <= 1.0
    assert 0.0 <= args.rdsi_hard_core_conf_thresh <= 1.0
    assert 0.0 <= args.rdsi_hard_core_entropy_thresh <= 1.0
    assert 0.0 <= args.rdsi_hard_core_reliability_thresh <= 1.0
    assert 0.0 <= args.rdsi_benefit_topk_ratio <= 1.0
    assert args.rdsi_benefit_topk_min_pixels >= 0
    assert args.rdsi_benefit_topk_max_pixels >= 0
    assert args.rdsi_benefit_topk_max_pixels == 0 or args.rdsi_benefit_topk_max_pixels >= args.rdsi_benefit_topk_min_pixels
    assert args.rdsi_benefit_score_floor >= 0.0
    assert 0.0 <= args.rdsi_budget_fraction <= 1.0
    assert 0.0 <= args.rdsi_budget_min_ratio <= args.rdsi_budget_max_ratio <= 1.0
    assert args.rdsi_entropy_increase_margin >= 0.0
    assert args.rdsi_entropy_increase_scale > 0.0
    assert args.rdsi_fg_excess_margin >= 0.0
    assert args.rdsi_fg_excess_scale > 0.0
    assert args.rdsi_boundary_radius >= -1
    assert args.rdsi_boundary_uncertainty_width > 0.0
    assert 0.0 <= args.rdsi_core_damage_veto <= 1.0
    assert args.rdsi_unsafe_gap_scale > 0.0
    assert args.rdsi_boundary_support_weight >= 0.0
    assert args.rdsi_core_preserving_fg_weight >= 0.0
    assert args.rdsi_teacher_reliability_weight >= 0.0
    assert args.rdsi_student_risk_weight >= 0.0
    assert args.rdsi_core_damage_weight >= 0.0
    assert args.rdsi_unsafe_gap_weight >= 0.0
    assert args.rdsi_foreground_excess_weight >= 0.0
    assert args.rdsi_knowledge_tiebreak_weight >= 0.0
    assert args.rdsi_safe_budget_gain >= 0.0
    assert args.rdsi_unsafe_budget_decay >= 0.0
    assert args.rdsi_max_budget_factor >= 1.0
    assert args.rdsi_core_reopen_boundary_floor >= 0.0
    assert args.rgftd_v3_stable_teacher_enabled in [0, 1]
    assert args.rgftd_v3_server_ema_fallback in [-1, 0, 1]
    assert args.rgftd_v3_teacher_pool_topk >= 1
    assert args.rgftd_v3_audit_start_iters >= 0
    assert args.rgftd_v3_audit_interval_iters >= 0 or args.rgftd_v3_audit_interval_iters == -1
    assert args.rgftd_v3_empty_pool_retry_iters >= 0
    assert args.rgftd_v3_audit_batches >= 1
    assert 0.0 <= args.rgftd_v3_audit_score_thresh <= 1.0
    assert 0.0 <= args.rgftd_v3_audit_teacher_reliability_min <= 1.0
    assert args.rgftd_v3_fallback_lambda_eff_cap >= 0.0
    assert args.rgftd_v3_benefit_enabled in [0, 1]
    assert args.rgftd_v3_lease_iters >= 0
    assert 0.0 <= args.rgftd_v3_benefit_momentum < 1.0
    assert 0.0 <= args.rgftd_v3_benefit_revoke_thresh <= args.rgftd_v3_benefit_decay_thresh <= args.rgftd_v3_benefit_good_thresh <= 1.0
    assert 0.0 <= args.rgftd_v3_routing_benefit_floor <= 1.0
    assert 0.0 <= args.rgftd_v3_recover_audit_score_thresh <= 1.0
    assert 0.0 <= args.rgftd_v3_decay_cap_scale <= 1.0
    assert 0.0 <= args.rgftd_v3_min_cap_scale <= 1.0
    assert args.rgftd_v3_min_cap_scale <= args.rgftd_v3_decay_cap_scale
    assert args.rgftd_v3_bgfg_warn_thresh >= 0.0
    assert 0.0 <= args.rgftd_v3_cap_hit_penalty <= 1.0
    assert args.rgftd_v3_lowmaxp_delta_thresh >= 0.0
    assert args.rgftd_v3_wann_mass_drop_thresh >= 0.0
    assert args.rgftd_stable_min_score >= 0.0
    assert args.rgftd_stable_update_margin >= 0.0
    assert 0.0 <= args.rgftd_stable_seed_prob_floor <= 1.0
    assert -1.0 <= args.rgftd_stable_seed_margin_floor <= 1.0
    assert args.rgftd_stable_max_core_conflict >= 0.0
    assert args.rgftd_stable_lowmaxp_delta_thresh >= 0.0
    assert args.rgftd_stable_wann_mass_drop_thresh >= 0.0
    assert args.rgftd_teacher_score_core_weight >= 0.0
    assert args.rgftd_teacher_score_support_weight >= 0.0
    assert args.rgftd_teacher_score_class_weight >= 0.0
    assert args.rgftd_teacher_score_conf_weight >= 0.0
    if args.rgftd_v3_enabled == 1:
        assert args.rgftd_enabled == 1
        assert args.rgftd_v3_teacher_pool_topk == 1

    # Configure logger
    if args.role == 'server':
        if int(getattr(args, 'save_code_snapshot', 1)) == 1:
            if os.path.exists(snapshot_path + '/code'):
                shutil.rmtree(snapshot_path + '/code')
            shutil.copytree('.', snapshot_path + '/code',
                            shutil.ignore_patterns(['.git', '__pycache__']))
        fl.common.logger.configure('server', filename=os.path.join(snapshot_path, 'server.log'))
        writer = build_summary_writer(args, snapshot_path + '/log')
    else:
        fl.common.logger.configure('client_{}'.format(args.cid), filename=os.path.join(snapshot_path, 'client_{}.log'.format(args.cid)))

    log(INFO, 'Arguments: {}'.format(args))

    # Load model and data
    db_train_full = BaseDataSets(base_dir=args.root_path, split='train', transform=transforms.Compose([
        RandomGenerator(args.patch_size, img_class=args.img_class)
    ]), client=args.client, sup_type=args.sup_type, img_class=args.img_class)
    db_meta_full = BaseDataSets(
        base_dir=args.root_path,
        split='train',
        transform=None,
        client=args.client,
        sup_type=args.sup_type,
        img_class=args.img_class,
    )
    db_val = BaseDataSets(base_dir=args.root_path,
                          client=args.client, split='val', img_class=args.img_class)

    if args.learnable_nwr_enabled == 1:
        indices = list(range(len(db_train_full)))
        split_rng = random.Random(args.seed + int(args.cid) * 1009)
        split_rng.shuffle(indices)
        meta_count = int(round(len(indices) * args.learnable_nwr_meta_fraction))
        assert 0 < meta_count < len(indices)
        meta_indices = sorted(indices[:meta_count])
        train_indices = sorted(indices[meta_count:])
        db_train = Subset(db_train_full, train_indices)
        db_meta = Subset(db_meta_full, meta_indices)
        print('learnable NWR meta split: train {} meta {}'.format(len(db_train), len(db_meta)))
    else:
        db_train = db_train_full
        db_meta = None

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    drop_last = True if args.strategy in ['FedUni', 'FedUniV2', 'FedUniV2.1'] else False
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=drop_last)
    meta_valloader = None
    if db_meta is not None:
        meta_valloader = DataLoader(db_meta, batch_size=1, shuffle=False, num_workers=0)
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
                writer = build_summary_writer(args, snapshot_path + '/pretrain_log')
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
                'wann_seed_support_ratio',
                'wann_core_candidate_ratio',
                'wann_sparse_seed_protocol',
                'wann_block_like_protocol',
                'wann_mean_reliability',
                'wann_entropy_low_r',
                'wann_max_prob_low_r',
                'wann_foreground_ratio_low_r',
                'wann_update_norm',
                'wann_update_cos_loo',
                'wann_update_conflict',
            ]
        if args.acg_enabled == 1:
            train_scalar_metrics += [
                'acg_loss',
                'acg_lambda',
                'acg_enabled',
                'acg_core_miss_loss',
                'acg_support_miss_loss',
                'acg_range_loss',
                'acg_range_under_loss',
                'acg_range_over_loss',
                'acg_unsupported_leak_loss',
                'acg_boundary_loss',
                'acg_boundary_cons_loss',
                'acg_boundary_smooth_loss',
                'acg_shape_contrast_loss',
                'acg_core_ratio',
                'acg_support_ratio',
                'acg_unsupported_ratio',
                'acg_boundary_ratio',
                'acg_shape_ring_ratio',
                'acg_shape_anchor_ratio',
                'acg_core_mass',
                'acg_envelope_mass',
                'acg_lower_mass',
                'acg_upper_mass',
                'acg_uncertain_mass',
                'acg_reliability_mean',
                'acg_pred_fg_mass',
                'acg_pred_fg_core_mean',
                'acg_pred_fg_support_mean',
                'acg_pred_fg_unsupported_mean',
                'acg_pred_fg_shape_ring_mean',
                'acg_pred_fg_shape_anchor_mean',
                'acg_core_weight_scale',
                'acg_support_weight_scale',
                'acg_range_weight_scale',
                'acg_leak_weight_scale',
                'acg_boundary_weight_scale',
                'acg_shape_weight_scale',
                'acg_fg_violation_share',
                'acg_range_violation_share',
                'acg_leak_violation_share',
                'acg_boundary_violation_share',
                'acg_shape_violation_share',
                'acg_nwr_fg_loss',
                'acg_nwr_seed_fg_loss',
                'acg_nwr_context_fg_loss',
                'acg_nwr_bg_loss',
                'acg_nwr_context_fg_target',
                'acg_nwr_context_fg_upper_target',
                'acg_nwr_context_fg_margin_gap',
                'acg_nwr_context_fg_over_gap',
                'acg_nwr_fg_weight_mass',
                'acg_nwr_seed_fg_weight_mass',
                'acg_nwr_context_fg_weight_mass',
                'acg_nwr_bg_weight_mass',
                'acg_nwr_fg_region_ratio',
                'acg_nwr_seed_fg_region_ratio',
                'acg_nwr_context_fg_region_ratio',
                'acg_nwr_bg_region_ratio',
                'acg_nwr_region_count',
                'acg_nwr_fg_prior',
                'acg_nwr_seed_fg_prior',
                'acg_nwr_context_fg_prior',
                'acg_nwr_bg_prior',
                'acg_nwr_pred_fg_on_fg',
                'acg_nwr_pred_fg_on_seed_fg',
                'acg_nwr_pred_fg_on_context_fg',
                'acg_nwr_pred_fg_on_bg',
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
                'rgftd_risk_region_ratio',
                'rgftd_preserve_region_ratio',
                'rgftd_teacher_accept_ratio',
                'rgftd_teacher_reject_ratio',
                'rgftd_reject_by_support',
                'rgftd_reject_by_core_conflict',
                'rgftd_reject_by_fg_ratio',
                'rgftd_release_score_mean',
                'rgftd_release_score_top',
                'rgftd_rdsi_enabled',
                'rgftd_rdsi_teacher_compete_count',
                'rgftd_rdsi_candidate_teacher_count',
                'rgftd_rdsi_multi_teacher_active',
                'rgftd_rdsi_best_vs_second_gap',
                'rgftd_rdsi_selected_score_mean',
                'rgftd_rdsi_selected_score_top',
                'rgftd_rdsi_score_benefit_mean',
                'rgftd_rdsi_selected_teacher_mean',
                'rgftd_rdsi_selected_teacher_switch_ratio',
                'rgftd_rdsi_teacher_scribble_ratio',
                'rgftd_rdsi_teacher_keypoint_ratio',
                'rgftd_rdsi_teacher_block_ratio',
                'rgftd_rdsi_teacher_unknown_ratio',
                'rgftd_rdsi_teacher_reliable_score',
                'rgftd_rdsi_selected_teacher_reliable',
                'rgftd_rdsi_selected_teacher_benefit',
                'rgftd_rdsi_selected_teacher_gap',
                'rgftd_rdsi_selected_student_risk',
                'rgftd_rdsi_selected_spatial_support',
                'rgftd_rdsi_transfer_compatibility_mean',
                'rgftd_rdsi_transfer_compatibility_top',
                'rgftd_rdsi_knowledge_gap_score',
                'rgftd_rdsi_risk_region_ratio',
                'rgftd_rdsi_hard_core_ratio',
                'rgftd_rdsi_soft_core_ratio',
                'rgftd_rdsi_fg_deficient_ratio',
                'rgftd_rdsi_fg_excessive_ratio',
                'rgftd_rdsi_candidate_ratio',
                'rgftd_rdsi_accept_ratio',
                'rgftd_rdsi_reject_ratio',
                'rgftd_rdsi_reject_by_core',
                'rgftd_rdsi_reject_by_seed',
                'rgftd_rdsi_reject_by_prior',
                'rgftd_rdsi_reject_by_entropy',
                'rgftd_rdsi_reject_by_fg_excess',
                'rgftd_rdsi_reject_by_bg_only',
                'rgftd_rdsi_reject_by_no_fg_lift',
                'rgftd_rdsi_foreground_active_ratio',
                'rgftd_rdsi_background_paired_ratio',
                'rgftd_rdsi_fg_repair_active_ratio',
                'rgftd_rdsi_bg_suppress_active_ratio',
                'rgftd_rdsi_boundary_active_ratio',
                'rgftd_rdsi_background_pair_ratio',
                'rgftd_rdsi_fg_repair_score',
                'rgftd_rdsi_bg_suppress_score',
                'rgftd_rdsi_boundary_score',
                'rgftd_rdsi_bg_only_ratio',
                'rgftd_rdsi_fg_lift_mean',
                'rgftd_rdsi_fg_lift_top',
                'rgftd_rdsi_bg_suppression_mean',
                'rgftd_rdsi_bg_suppression_top',
                'rgftd_rdsi_benefit_mean',
                'rgftd_rdsi_benefit_top',
                'rgftd_rdsi_boundary_support_mean',
                'rgftd_rdsi_boundary_support_top',
                'rgftd_rdsi_core_preserving_fg_mean',
                'rgftd_rdsi_core_preserving_fg_top',
                'rgftd_rdsi_core_damage_mean',
                'rgftd_rdsi_core_damage_top',
                'rgftd_rdsi_seed_conflict_mean',
                'rgftd_rdsi_unsafe_gap_mean',
                'rgftd_rdsi_unsafe_gap_top',
                'rgftd_rdsi_foreground_excess_proxy',
                'rgftd_rdsi_safe_signal',
                'rgftd_rdsi_unsafe_signal',
                'rgftd_rdsi_safe_budget_factor',
                'rgftd_rdsi_budget_target_ratio',
                'rgftd_rdsi_fg_budget_demand',
                'rgftd_rdsi_bg_budget_demand',
                'rgftd_rdsi_boundary_budget_demand',
                'rgftd_rdsi_fg_budget_alloc',
                'rgftd_rdsi_bg_budget_alloc',
                'rgftd_rdsi_boundary_budget_alloc',
                'rgftd_rdsi_fg_budget_unused',
                'rgftd_rdsi_bg_budget_unused',
                'rgftd_rdsi_boundary_budget_unused',
                'rgftd_rdsi_budget_reflow_ratio',
                'rgftd_rdsi_quality_mean',
                'rgftd_rdsi_effective_topk_ratio',
                'rgftd_rdsi_effective_min_pixels',
                'rgftd_rdsi_core_reopen_ratio',
                'rgftd_rdsi_core_reopen_active_ratio',
                'rgftd_rdsi_veto_by_core_damage',
                'rgftd_rdsi_alpha_mean',
                'rgftd_rdsi_alpha_top',
                'rgftd_rdsi_raw_teacher_fg_delta',
                'rgftd_rdsi_raw_teacher_conf_mean',
                'rgftd_rdsi_raw_teacher_fg_ratio',
                'rgftd_rdsi_q_fg_delta',
                'rgftd_rdsi_fg_repair_q_delta',
                'rgftd_rdsi_bg_suppress_q_delta',
                'rgftd_rdsi_boundary_q_delta',
                'rgftd_rdsi_target_conf_mean',
                'rgftd_rdsi_target_entropy_mean',
                'rgftd_rdsi_loss_raw',
                'rgftd_rdsi_loss_weighted',
                'rgftd_rdsi_proto_loss',
                'rgftd_rdsi_proto_weight_mean',
                'rgftd_rdsi_proto_weight_top',
                'rgftd_rdsi_proto_cosine',
                'rgftd_rdsi_proto_region_ratio',
                'rgftd_rdsi_proto_teacher_entropy',
                'rgftd_rdsi_proto_teacher_weight_max',
                'rgftd_rdsi_proto_valid_batches',
                'rgftd_rdsi_fg_repair_loss',
                'rgftd_rdsi_bg_suppress_loss',
                'rgftd_rdsi_boundary_loss',
                'rgftd_teacher_reliable_score',
                'rgftd_student_risk_score',
                'rgftd_knowledge_gap_score',
                'rgftd_selected_gap_mean',
                'rgftd_rejected_gap_mean',
                'rgftd_teacher_active_loss',
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
                'rgftd_spatial_support_ratio',
                'rgftd_foreground_candidate_ratio',
                'rgftd_spatial_weight_mean',
                'rgftd_spatial_weight_candidate_mean',
                'rgftd_spatial_weight_near_seed_mean',
                'rgftd_spatial_weight_far_mean',
                'rgftd_spatial_loss_scale',
                'rgftd_active_foreground_pixels_pre_spatial',
                'rgftd_active_foreground_spatial_keep_ratio',
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
                'rgftd_active_fg_seed_precision',
                'rgftd_active_fg_seed_recall',
                'rgftd_active_fg_support_precision',
                'rgftd_active_fg_support_recall',
                'rgftd_active_fg_candidate_ratio',
                'rgftd_active_fg_fg_candidate_ratio',
                'rgftd_active_fg_near_seed_ratio',
                'rgftd_refine_enabled',
                'rgftd_refine_silent',
                'rgftd_refine_roi_ratio',
                'rgftd_refine_affinity_mean',
                'rgftd_refine_teacher_q_kl',
                'rgftd_refine_q_entropy_mean',
                'rgftd_refine_q_fg_mass',
                'rgftd_refine_q_fg_ratio',
                'rgftd_refine_q_fg_delta',
                'rgftd_refine_q_seed_precision',
                'rgftd_refine_q_seed_recall',
                'rgftd_refine_q_candidate_ratio',
                'rgftd_refine_q_near_seed_ratio',
                'rgftd_refine_unsupported_fg_ratio',
                'rgftd_refine_unsupported_fg_scale',
                'rgftd_refine_q_core_conflict',
                'rgftd_background_suppression_mean',
                'rgftd_return_reason',
                'rgftd_v3_pool_size',
                'rgftd_v3_pool_stable',
                'rgftd_v3_no_teacher',
                'rgftd_v3_fallback_teacher',
                'rgftd_v3_teacher_source',
                'rgftd_v3_selected_teacher',
                'rgftd_v3_selected_score',
                'rgftd_v3_selected_stable_score',
                'rgftd_v3_best_failed_teacher',
                'rgftd_v3_best_failed_score',
                'rgftd_v3_routing_score',
                'rgftd_v3_audit_seed_fg_prob_mean',
                'rgftd_v3_audit_seed_fg_margin_mean',
                'rgftd_v3_audit_seed_fg_recall',
                'rgftd_v3_audit_core_conflict',
                'rgftd_v3_audit_core_valid',
                'rgftd_v3_audit_teacher_reliability',
                'rgftd_v3_audit_release_factor',
                'rgftd_v3_lease_active',
                'rgftd_v3_lease_age',
                'rgftd_v3_benefit_score',
                'rgftd_v3_window_score',
                'rgftd_v3_stability_score',
                'rgftd_v3_teacher_renew',
                'rgftd_v3_teacher_revoke',
                'rgftd_v3_teacher_decay',
                'rgftd_v3_cap_current',
                'rgftd_v3_cap_scale',
                'rgftd_v3_ret0_ratio',
                'rgftd_v3_cap_hit_ratio',
                'rgftd_v3_bgfg_window',
                'rgftd_v3_lowmaxp_delta',
                'rgftd_v3_wann_mass_delta',
                'rgftd_stable_valid',
                'rgftd_stable_score',
                'rgftd_stable_best_score',
                'rgftd_stable_updated',
            ]
            for class_id in range(1, args.num_classes):
                train_scalar_metrics += [
                    'rgftd_teacher_class{}_agreement'.format(class_id),
                    'rgftd_teacher_class{}_core_agreement'.format(class_id),
                    'rgftd_teacher_class{}_reliability'.format(class_id),
                    'rgftd_teacher_class{}_release_ratio'.format(class_id),
                ]
            for teacher_id in range(min(int(getattr(args, 'min_num_clients', 0)), 10)):
                train_scalar_metrics += [
                    'rgftd_rdsi_teacher{}_ratio'.format(teacher_id),
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
        client = MyClient(args, model, trainloader, valloader, meta_valloader=meta_valloader, amp=(args.amp == 1))
        fl.client.start_client(server_address=args.server_address, client=client)



if __name__ == '__main__':
    main()
