# -*- coding:utf-8 -*-
"""Offline oracle diagnostics for FedRAP-RDSI.

The script does not enter training.  It uses full GT only for offline
measurement and uses the current WANN/RDSI observable evidence for scoring.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
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


PROSTATE_CLIENTS = ["client1", "client2", "client3", "client4", "client5", "client6"]
POLYP_CLIENTS = ["client1", "client2", "client3", "client4"]
FAZ_CLIENTS = ["client1", "client2", "client3", "client4", "client5"]
UNIFORM_SPARSE5 = "sparse_scribble_5"


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


def default_setup(img_class):
    if img_class == "prostate":
        return PROSTATE_CLIENTS, 2, 1, 6
    if img_class == "polyp":
        return POLYP_CLIENTS, 2, 3, 4
    if img_class == "faz":
        return FAZ_CLIENTS, 2, 1, 5
    raise ValueError("Unsupported img_class: {}".format(img_class))


def primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def auxiliary_logits(model_out):
    if isinstance(model_out, (tuple, list)) and len(model_out) > 8:
        return model_out[8]
    return None


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
    obj.sup_type = UNIFORM_SPARSE5
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
    for p in model.parameters():
        p.requires_grad = False
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


def dice_region(pred_fg, gt_fg, region):
    pred = pred_fg & region
    gt = gt_fg & region
    denom = float(pred.sum().item() + gt.sum().item())
    if denom <= 0.0:
        return 1.0
    return float(2.0 * (pred & gt).sum().item() / denom)


def error_rate(pred, gt, region):
    denom = float(region.sum().item())
    if denom <= 0.0:
        return 0.0
    return float(((pred != gt) & region).sum().item() / denom)


def masked_mean(value, mask):
    denom = mask.float().sum()
    if float(denom.item()) <= 0.0:
        return 0.0
    return float((value * mask.float()).sum().item() / denom.item())


def normalized_entropy(prob):
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    return (entropy / max(math.log(float(prob.shape[1])), 1e-6)).clamp(0.0, 1.0)


def dilate(mask, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return mask
    y = F.max_pool2d(mask.float().unsqueeze(1), kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[:, 0] > 0.5


def region_types(student_pred, gt, risk_region):
    gt_fg = gt > 0
    student_fg = student_pred > 0
    missing = risk_region & gt_fg & (~student_fg)
    excess = risk_region & (~gt_fg) & student_fg
    correct_fg = risk_region & gt_fg & student_fg
    boundary = risk_region & dilate(gt_fg, 2) & dilate(~gt_fg, 2)
    return {
        "fg_missing": missing,
        "fg_excess": excess,
        "correct_fg": correct_fg,
        "boundary": boundary,
    }


def rdsi_observable_scores(args, student_logits, teacher_logits, label, maps, risk_region):
    student_prob = torch.softmax(student_logits.detach(), dim=1)
    teacher_prob = torch.softmax(teacher_logits.detach(), dim=1)
    student_conf = student_prob.max(dim=1)[0]
    teacher_conf = teacher_prob.max(dim=1)[0]
    student_entropy = normalized_entropy(student_prob)
    teacher_entropy = normalized_entropy(teacher_prob)
    student_pred = student_prob.argmax(dim=1)
    teacher_pred = teacher_prob.argmax(dim=1)
    log_s = torch.log(student_prob.clamp_min(1e-6))
    log_t = torch.log(teacher_prob.clamp_min(1e-6))
    kl_ts = (teacher_prob * (log_t - log_s)).sum(dim=1)
    kl_st = (student_prob * (log_s - log_t)).sum(dim=1)
    knowledge_gap = (0.5 * (kl_ts + kl_st) / max(math.log(float(teacher_prob.shape[1])), 1e-6)).clamp(0.0, 1.0)
    disagreement = (1.0 - (teacher_prob * student_prob).sum(dim=1)).clamp(0.0, 1.0)

    weak_anchor = maps.seed_support_mask & risk_region
    weak_agree = torch.where(
        weak_anchor,
        ((teacher_pred == maps.target_label) & weak_anchor).float(),
        torch.ones_like(teacher_conf),
    )
    core = maps.core_mask
    if bool(core.any().item()):
        core_agree = (((teacher_pred == maps.target_label) & core).float().sum() / core.float().sum().clamp_min(1.0)).clamp(0.0, 1.0)
    else:
        core_agree = torch.tensor(1.0, device=teacher_conf.device)
    if teacher_prob.shape[1] > 1:
        teacher_fg_prob = teacher_prob[:, 1:].max(dim=1)[0]
        student_fg_prob = student_prob[:, 1:].max(dim=1)[0]
    else:
        teacher_fg_prob = torch.zeros_like(teacher_conf)
        student_fg_prob = torch.zeros_like(student_conf)
    teacher_bg_prob = teacher_prob[:, 0]
    teacher_fg_ready = ((teacher_pred > 0) & (teacher_conf >= args.rgftd_teacher_conf_thresh)) | (
        (teacher_fg_prob >= args.rgftd_teacher_fg_prob_thresh) & risk_region
    )
    teacher_bg_safe = teacher_fg_prob <= args.rgftd_bg_max_fg_prob
    prior_safe = torch.where(teacher_fg_ready, torch.ones_like(teacher_conf), teacher_bg_safe.float())
    teacher_reliable = (
        teacher_conf
        * (1.0 - teacher_entropy).clamp(0.0, 1.0)
        * weak_agree
        * core_agree.detach().clamp(0.0, 1.0)
        * prior_safe
    ).clamp(0.0, 1.0)
    student_risk = (
        (1.0 - maps.reliability).clamp(0.0, 1.0)
        * torch.maximum(student_entropy, torch.maximum(1.0 - student_conf, disagreement))
    ).clamp(0.0, 1.0)
    entropy_margin = float(getattr(args, "rdsi_entropy_increase_margin", 0.05))
    entropy_scale = max(float(getattr(args, "rdsi_entropy_increase_scale", 0.35)), 1e-6)
    entropy_increase = (teacher_entropy - student_entropy - entropy_margin).clamp_min(0.0)
    entropy_safe = torch.where(
        teacher_fg_ready,
        (1.0 - entropy_increase / entropy_scale).clamp(0.0, 1.0),
        torch.ones_like(teacher_conf),
    )
    fg_excess_margin = float(getattr(args, "rdsi_fg_excess_margin", 0.05))
    fg_excess_scale = max(float(getattr(args, "rdsi_fg_excess_scale", 0.20)), 1e-6)
    denom = risk_region.float().flatten(1).sum(dim=1).clamp_min(1.0).view(-1, 1, 1)
    teacher_fg_region_ratio = (teacher_fg_prob * risk_region.float()).flatten(1).sum(dim=1).view(-1, 1, 1) / denom
    student_fg_region_ratio = (student_fg_prob * risk_region.float()).flatten(1).sum(dim=1).view(-1, 1, 1) / denom
    fg_excess = (teacher_fg_region_ratio - (student_fg_region_ratio + fg_excess_margin)).clamp_min(0.0)
    fg_excess_safe = (1.0 - fg_excess / fg_excess_scale).clamp(0.0, 1.0).expand_as(teacher_conf)
    benefit_score = (weak_agree * core_agree.detach() * entropy_safe * torch.where(teacher_fg_ready, fg_excess_safe, teacher_bg_safe.float())).clamp(0.0, 1.0)
    rdsi_score = (teacher_reliable * student_risk * knowledge_gap * benefit_score * risk_region.float()).clamp(0.0, 1.0)
    return {
        "rdsi_score_mean": masked_mean(rdsi_score, risk_region),
        "teacher_reliable_score": masked_mean(teacher_reliable, risk_region),
        "student_risk_score": masked_mean(student_risk, risk_region),
        "knowledge_gap_score": masked_mean(knowledge_gap, risk_region),
        "benefit_score": masked_mean(benefit_score, risk_region),
        "teacher_conf": masked_mean(teacher_conf, risk_region),
        "teacher_entropy": masked_mean(teacher_entropy, risk_region),
        "teacher_fg_ratio": masked_mean((teacher_pred > 0).float(), risk_region),
        "teacher_core_conflict": 1.0 - masked_mean((teacher_pred == maps.target_label).float(), core) if bool(core.any().item()) else 0.0,
    }


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
    parser.add_argument("--max_cases_per_client", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Run model inference on CPU or CUDA.")
    parser.add_argument("--analysis_iter", type=int, default=1600)
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
    parser.add_argument("--rgftd_teacher_conf_thresh", type=float, default=0.90)
    parser.add_argument("--rgftd_teacher_fg_prob_thresh", type=float, default=0.35)
    parser.add_argument("--rgftd_bg_max_fg_prob", type=float, default=0.15)
    parser.add_argument("--rdsi_entropy_increase_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_entropy_increase_scale", type=float, default=0.35)
    parser.add_argument("--rdsi_fg_excess_margin", type=float, default=0.05)
    parser.add_argument("--rdsi_fg_excess_scale", type=float, default=0.20)
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
    out_dir = args.output_dir or os.path.join(args.snapshot_path, "rdsi_oracle_diagnostics_cpu")
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_paths = resolve_checkpoints(args, len(clients))
    models = [load_model(args, cid, path) for cid, path in enumerate(checkpoint_paths)]
    sample_records = []
    oracle_records = []
    wann_records = []
    core_damage_records = []
    specialization_counter = Counter()

    with torch.no_grad():
        for target_cid, client in enumerate(clients):
            dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=client,
                sup_type=UNIFORM_SPARSE5,
                img_class=args.img_class,
            )
            if args.max_cases_per_client > 0:
                dataset = Subset(dataset, list(range(min(args.max_cases_per_client, len(dataset)))))
                sample_list = dataset.dataset.sample_list
            else:
                sample_list = dataset.sample_list
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            sample_offset = 0
            for batch in loader:
                images = to_image_tensor(batch["image"], args.img_class).to(args.device)
                weak_label = batch["label"].long().to(args.device)
                idxs = batch["idx"].tolist()
                gt_list = []
                rel_paths = []
                for idx in idxs:
                    rel_path = sample_list[int(idx)]
                    rel_paths.append(rel_path)
                    gt_list.append(load_full_gt(args.root_path, rel_path))
                gt = torch.stack(gt_list, dim=0).long().to(args.device)

                student_out = models[target_cid](images)
                student_logits = primary_logits(student_out)
                student_aux = auxiliary_logits(student_out)
                student_prob = torch.softmax(student_logits, dim=1)
                student_pred = student_prob.argmax(dim=1)
                maps = build_wann_maps(
                    image=images,
                    label=weak_label,
                    logits=student_logits,
                    aux_logits=student_aux,
                    sup_type=UNIFORM_SPARSE5,
                    img_class=args.img_class,
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )
                risk_region = (maps.ignore_mask | maps.soft_band) & (~maps.core_mask)
                core_region = maps.core_mask
                gt_fg = gt > 0
                student_fg = student_pred > 0
                sample_teacher_rows = []

                teacher_logits_list = [primary_logits(model(images)) for model in models]
                teacher_pred_list = [torch.softmax(logits, dim=1).argmax(dim=1) for logits in teacher_logits_list]

                for b, rel_path in enumerate(rel_paths):
                    risk_b = risk_region[b:b + 1]
                    core_b = core_region[b:b + 1]
                    gt_b = gt[b:b + 1]
                    stu_pred_b = student_pred[b:b + 1]
                    stu_fg_b = student_fg[b:b + 1]
                    stu_risk_dice = dice_region(stu_fg_b, gt_b > 0, risk_b)
                    stu_core_dice = dice_region(stu_fg_b, gt_b > 0, core_b)
                    risk_err = error_rate(stu_pred_b, gt_b, risk_b)
                    core_err = error_rate(stu_pred_b, gt_b, core_b)
                    wann_records.append({
                        "img_class": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "sample_idx": int(idxs[b]),
                        "case": rel_path,
                        "risk_pixels": int(risk_b.sum().item()),
                        "core_pixels": int(core_b.sum().item()),
                        "risk_error_rate": risk_err,
                        "core_error_rate": core_err,
                        "risk_core_error_ratio": risk_err / max(core_err, 1e-6),
                        "student_risk_dice": stu_risk_dice,
                        "student_core_dice": stu_core_dice,
                    })
                    rtypes = region_types(stu_pred_b, gt_b, risk_b)
                    best_row = None
                    for teacher_cid, teacher_logits in enumerate(teacher_logits_list):
                        teacher_pred = teacher_pred_list[teacher_cid][b:b + 1]
                        teacher_fg = teacher_pred > 0
                        t_risk_dice = dice_region(teacher_fg, gt_b > 0, risk_b)
                        t_core_dice = dice_region(teacher_fg, gt_b > 0, core_b)
                        benefit = t_risk_dice - stu_risk_dice
                        score_info = rdsi_observable_scores(
                            args,
                            student_logits[b:b + 1],
                            teacher_logits[b:b + 1],
                            weak_label[b:b + 1],
                            type("MapObj", (), {k: getattr(maps, k)[b:b + 1] if torch.is_tensor(getattr(maps, k)) else getattr(maps, k) for k in [
                                "core_mask", "soft_band", "ignore_mask", "reliability", "target_label", "valid_mask",
                                "support_mask", "seed_support_mask", "candidate_mask"
                            ]})(),
                            risk_b,
                        )
                        type_gains = {}
                        type_sizes = {}
                        for type_name, type_mask in rtypes.items():
                            type_sizes[type_name] = int(type_mask.sum().item())
                            type_gains[type_name] = dice_region(teacher_fg, gt_b > 0, type_mask) - dice_region(stu_fg_b, gt_b > 0, type_mask)
                        row = {
                            "img_class": args.img_class,
                            "target_cid": target_cid,
                            "target_client": client,
                            "teacher_cid": teacher_cid,
                            "teacher_client": clients[teacher_cid],
                            "same_client": int(teacher_cid == target_cid),
                            "sample_idx": int(idxs[b]),
                            "case": rel_path,
                            "risk_pixels": int(risk_b.sum().item()),
                            "core_pixels": int(core_b.sum().item()),
                            "student_risk_dice": stu_risk_dice,
                            "teacher_risk_dice": t_risk_dice,
                            "oracle_benefit": benefit,
                            "teacher_core_dice": t_core_dice,
                            "student_core_dice": stu_core_dice,
                            "teacher_core_conflict": error_rate(teacher_pred, gt_b, core_b),
                            "teacher_core_damage": error_rate(teacher_pred, gt_b, core_b) - error_rate(stu_pred_b, gt_b, core_b),
                        }
                        row.update(score_info)
                        for type_name in type_gains:
                            row["{}_gain".format(type_name)] = type_gains[type_name]
                            row["{}_pixels".format(type_name)] = type_sizes[type_name]
                        sample_records.append(row)
                        sample_teacher_rows.append(row)
                        core_damage_records.append({
                            "img_class": args.img_class,
                            "target_cid": target_cid,
                            "teacher_cid": teacher_cid,
                            "case": rel_path,
                            "teacher_core_conflict": row["teacher_core_conflict"],
                            "teacher_core_damage": row["teacher_core_damage"],
                        })
                        if best_row is None or row["oracle_benefit"] > best_row["oracle_benefit"]:
                            best_row = row
                    if best_row is not None:
                        best_type = "mixed"
                        typed = [(name, best_row.get("{}_gain".format(name), 0.0), best_row.get("{}_pixels".format(name), 0)) for name in rtypes]
                        typed = [x for x in typed if x[2] > 0]
                        if typed:
                            best_type = max(typed, key=lambda x: x[1])[0]
                        specialization_counter[(best_row["teacher_cid"], best_type, target_cid)] += 1
                        oracle_records.append({
                            "img_class": args.img_class,
                            "target_cid": target_cid,
                            "target_client": client,
                            "sample_idx": int(idxs[b]),
                            "case": rel_path,
                            "oracle_teacher_cid": int(best_row["teacher_cid"]),
                            "oracle_teacher_client": best_row["teacher_client"],
                            "oracle_benefit": float(best_row["oracle_benefit"]),
                            "oracle_teacher_risk_dice": float(best_row["teacher_risk_dice"]),
                            "student_risk_dice": float(best_row["student_risk_dice"]),
                            "positive_oracle": int(best_row["oracle_benefit"] > 1e-6),
                            "oracle_region_type": best_type,
                        })
                sample_offset += len(idxs)
                print("processed {} target_client={} samples={}".format(args.img_class, client, sample_offset), flush=True)

    sample_df = pd.DataFrame(sample_records)
    oracle_df = pd.DataFrame(oracle_records)
    wann_df = pd.DataFrame(wann_records)
    core_df = pd.DataFrame(core_damage_records)
    spec_rows = [
        {
            "teacher_cid": teacher_cid,
            "teacher_client": clients[teacher_cid],
            "region_type": region_type,
            "target_cid": target_cid,
            "target_client": clients[target_cid],
            "oracle_selected_count": count,
        }
        for (teacher_cid, region_type, target_cid), count in specialization_counter.items()
    ]
    spec_df = pd.DataFrame(spec_rows)

    sample_csv = os.path.join(out_dir, "sample_teacher_oracle.csv")
    oracle_csv = os.path.join(out_dir, "oracle_selected_regions.csv")
    wann_csv = os.path.join(out_dir, "wann_risk_core_error.csv")
    core_csv = os.path.join(out_dir, "teacher_core_damage.csv")
    spec_csv = os.path.join(out_dir, "teacher_specialization_map.csv")
    sample_df.to_csv(sample_csv, index=False)
    oracle_df.to_csv(oracle_csv, index=False)
    wann_df.to_csv(wann_csv, index=False)
    core_df.to_csv(core_csv, index=False)
    spec_df.to_csv(spec_csv, index=False)

    summary = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "analysis_iter": args.analysis_iter,
        "num_sample_teacher_rows": int(len(sample_df)),
        "num_oracle_regions": int(len(oracle_df)),
        "oracle_positive_ratio": float(oracle_df["positive_oracle"].mean()) if len(oracle_df) else 0.0,
        "oracle_mean_gain": float(oracle_df["oracle_benefit"].mean()) if len(oracle_df) else 0.0,
        "oracle_top_gain": float(oracle_df["oracle_benefit"].max()) if len(oracle_df) else 0.0,
        "oracle_no_better_teacher_ratio": float((oracle_df["positive_oracle"] == 0).mean()) if len(oracle_df) else 0.0,
        "wann_risk_error_rate": float(wann_df["risk_error_rate"].mean()) if len(wann_df) else 0.0,
        "wann_core_error_rate": float(wann_df["core_error_rate"].mean()) if len(wann_df) else 0.0,
        "wann_risk_core_error_ratio": float(wann_df["risk_core_error_ratio"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(wann_df) else 0.0,
        "mean_teacher_core_conflict": float(core_df["teacher_core_conflict"].mean()) if len(core_df) else 0.0,
        "mean_teacher_core_damage": float(core_df["teacher_core_damage"].mean()) if len(core_df) else 0.0,
        "correlations": {
            "rdsi_score_vs_oracle_benefit": corr_summary(sample_df, "rdsi_score_mean", "oracle_benefit"),
            "teacher_reliable_vs_oracle_benefit": corr_summary(sample_df, "teacher_reliable_score", "oracle_benefit"),
            "knowledge_gap_vs_oracle_benefit": corr_summary(sample_df, "knowledge_gap_score", "oracle_benefit"),
            "benefit_score_vs_oracle_benefit": corr_summary(sample_df, "benefit_score", "oracle_benefit"),
        },
        "files": {
            "sample_teacher_oracle": sample_csv,
            "oracle_selected_regions": oracle_csv,
            "wann_risk_core_error": wann_csv,
            "teacher_core_damage": core_csv,
            "teacher_specialization_map": spec_csv,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    pd.DataFrame([summary]).to_json(os.path.join(out_dir, "summary_flat.json"), orient="records", force_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
