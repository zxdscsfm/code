# -*- coding: utf-8 -*-
"""Offline RGFTD teacher reliability audit.

This script is intentionally read-only for training artifacts.  It audits a
proxy federated teacher reconstructed by averaging client checkpoints, because
the live server EMA teacher is not saved as a standalone checkpoint.
"""

import argparse
import json
import math
import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def _disable_cuda_for_cpu_audit():
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self


def _load_state(path):
    return torch.load(path, map_location="cpu")


def _average_states(paths):
    states = [_load_state(path) for path in paths]
    avg = {}
    for key in states[0].keys():
        first = states[0][key]
        if torch.is_tensor(first) and first.dtype.is_floating_point:
            avg[key] = torch.stack([state[key].float() for state in states], dim=0).mean(dim=0).to(first.dtype)
        else:
            avg[key] = first
    return avg


def _primary_logits(model_out):
    if torch.is_tensor(model_out):
        return model_out
    return model_out[0]


def _aux_logits(model_out):
    if isinstance(model_out, (tuple, list)) and len(model_out) >= 9:
        return model_out[8]
    return _primary_logits(model_out)


def _dilate(mask, radius):
    radius = int(max(radius, 0))
    if radius <= 0:
        return mask
    x = mask.float().unsqueeze(1)
    y = F.max_pool2d(x, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return y[:, 0] > 0.5


def _topk_anchor(score, valid_mask, topk_ratio, min_pixels):
    anchors = torch.zeros_like(valid_mask, dtype=torch.bool)
    flat_score = score.flatten(1)
    flat_valid = valid_mask.flatten(1)
    budgets = []
    for b in range(score.shape[0]):
        valid_count = int(flat_valid[b].sum().item())
        if valid_count <= 0:
            budgets.append(0)
            continue
        k = max(int(min_pixels), int(math.ceil(valid_count * float(topk_ratio))))
        k = min(k, valid_count)
        masked = flat_score[b].masked_fill(~flat_valid[b], -1.0)
        idx = torch.topk(masked, k=k, largest=True).indices
        anchors.flatten(1)[b, idx] = True
        budgets.append(k)
    return anchors & valid_mask, budgets


def _seed_or_fallback(seed_mask, fallback_mask):
    valid = fallback_mask.clone()
    fallback_used = []
    for b in range(seed_mask.shape[0]):
        has_seed = bool(seed_mask[b].any().item())
        fallback_used.append(not has_seed)
        if has_seed:
            valid[b] = seed_mask[b]
    return valid & fallback_mask, fallback_used


def _safe_mean(values):
    values = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    return float(np.mean(values)) if values else 0.0


def _ratio(num, den):
    den = float(den)
    return float(num) / den if den > 0 else 0.0


def _make_args(cid, sup_type):
    return SimpleNamespace(
        model="unet_univ5",
        in_chns=1,
        num_classes=2,
        min_num_clients=6,
        cid=int(cid),
        prompt="universal",
        attention="dual",
        sup_type=sup_type,
        label_prompt=1,
        img_size=384,
        img_class="prostate",
        wann_keypoint_soft_radius=2,
        wann_scribble_soft_radius=4,
        wann_box_soft_radius=2,
        wann_mask_soft_radius=1,
        wann_dilated_support_score=0.55,
        wann_appearance_temp=1.5,
        wann_texture_kernel_size=5,
        wann_texture_temp=1.0,
        wann_texture_weight=0.25,
        wann_pred_start_iter=800,
        wann_entropy_weight=0.5,
        wann_agreement_weight=0.5,
        wann_global_agreement_weight=0.5,
        wann_r_max=1.2,
        wann_core_thresh=0.65,
        wann_soft_thresh=0.25,
        wann_core_min_weight=0.8,
        rgftd_low_r_thresh=0.25,
        rgftd_use_soft_band=0,
        rgftd_teacher_foreground_radius=2,
        rgftd_min_foreground_pixels=8,
        rgftd_min_foreground_ratio=0.05,
        rgftd_teacher_fg_prob_thresh=0.35,
        rgftd_teacher_fg_topk_ratio=0.002,
        rgftd_teacher_fg_topk_min_pixels=8,
        rgftd_teacher_student_fg_margin=0.05,
        rgftd_teacher_bg_conf_thresh=0.98,
        rgftd_teacher_conf_thresh=0.90,
        rgftd_student_conf_thresh=0.80,
        rgftd_student_entropy_thresh=0.35,
    )


def audit_client(client_name, cid, sup_type, root_path, model_dir, iter_id, max_samples):
    from dataloaders.dataset import BaseDataSets
    from networks.net_factory import net_factory
    from weak_annotation_reliability import build_wann_maps

    args = _make_args(cid, sup_type)
    client_paths = [
        os.path.join(model_dir, f"client_{i}_iter_{iter_id}_dice_")
        for i in range(6)
    ]
    resolved = []
    for prefix in client_paths:
        matches = [os.path.join(model_dir, name) for name in os.listdir(model_dir)
                   if name.startswith(os.path.basename(prefix)) and name.endswith(".pth")]
        if not matches:
            raise FileNotFoundError(prefix)
        resolved.append(sorted(matches)[-1])

    teacher_state = _average_states(resolved)
    student_path = resolved[int(cid)]

    teacher = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    student = net_factory(args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    teacher.load_state_dict(teacher_state, strict=False)
    student.load_state_dict(_load_state(student_path), strict=False)
    teacher.eval()
    student.eval()

    ds = BaseDataSets(base_dir=root_path, split="train", transform=None, client=client_name,
                      sup_type=sup_type, img_class="prostate")

    out = []
    with torch.no_grad():
        for idx in range(min(max_samples, len(ds))):
            sample = ds[idx]
            image_np = sample["image"].astype(np.float32)
            label_np = sample["label"].astype(np.int64)
            image = torch.from_numpy(image_np).float()
            if image.dim() == 2:
                image = image.unsqueeze(0)
            image = image.unsqueeze(0)
            label = torch.from_numpy(label_np).long().unsqueeze(0)

            s_out = student(image)
            t_out = teacher(image)
            s_logits = _primary_logits(s_out)
            s_aux = _aux_logits(s_out)
            t_logits = _primary_logits(t_out)
            maps = build_wann_maps(
                image=image,
                label=label,
                logits=s_logits,
                aux_logits=s_aux,
                sup_type=sup_type,
                img_class="prostate",
                num_classes=2,
                iter_num=iter_id,
                args=args,
                ref_logits=s_logits,
            )

            t_prob = torch.softmax(t_logits, dim=1)
            t_pred = t_prob.argmax(dim=1)
            t_conf = t_prob.max(dim=1)[0]
            t_fg_prob = t_prob[:, 1:].max(dim=1)[0]
            t_fg = t_pred > 0
            t_bg = ~t_fg

            candidate = maps.candidate_mask & (~maps.core_mask)
            region = (maps.ignore_mask | maps.soft_band) & (~maps.core_mask)
            t_fg_conf = t_fg & (t_conf >= args.rgftd_teacher_conf_thresh)
            t_fg_prob_anchor = (t_fg_prob >= args.rgftd_teacher_fg_prob_thresh) & region
            seed = (t_fg_conf | t_fg_prob_anchor) & region
            topk_valid, fallback_used = _seed_or_fallback(seed, region)
            anchor, budgets = _topk_anchor(
                t_fg_prob,
                topk_valid,
                args.rgftd_teacher_fg_topk_ratio,
                args.rgftd_teacher_fg_topk_min_pixels,
            )
            support = maps.support_mask
            core = maps.core_mask
            label_fg = label > 0
            support_fg = support & label_fg
            support_bg = support & (label == 0)
            near_support = _dilate(support, 8)

            gt_precision = None
            gt_recall = None
            h5_path = os.path.join(root_path, ds.sample_list[idx])
            if os.path.exists(h5_path):
                with h5py.File(h5_path, "r") as h5f:
                    if "mask" in h5f:
                        gt = torch.from_numpy(h5f["mask"][:].astype(np.int64)).unsqueeze(0)
                        gt_fg = gt > 0
                        gt_precision = _ratio((anchor & gt_fg).sum().item(), anchor.sum().item())
                        gt_recall = _ratio((anchor & gt_fg).sum().item(), gt_fg.sum().item())

            item = {
                "idx": int(idx),
                "support_px": int(support.sum().item()),
                "core_px": int(core.sum().item()),
                "candidate_px": int(candidate.sum().item()),
                "region_px": int(region.sum().item()),
                "seed_px": int(seed.sum().item()),
                "topk_valid_px": int(topk_valid.sum().item()),
                "topk_budget": int(sum(budgets)),
                "topk_selected_px": int(anchor.sum().item()),
                "topk_saturation": _ratio(anchor.sum().item(), sum(budgets)),
                "topk_fallback_used": float(any(fallback_used)),
                "teacher_support_agreement": _ratio(((t_pred == label) & support).sum().item(), support.sum().item()),
                "teacher_core_agreement": _ratio(((t_pred == label) & core).sum().item(), core.sum().item()),
                "teacher_support_fg_recall": _ratio((t_fg & support_fg).sum().item(), support_fg.sum().item()),
                "teacher_support_bg_agreement": _ratio(((~t_fg) & support_bg).sum().item(), support_bg.sum().item()),
                "anchor_in_candidate_ratio": _ratio((anchor & candidate).sum().item(), anchor.sum().item()),
                "anchor_near_support_r8_ratio": _ratio((anchor & near_support).sum().item(), anchor.sum().item()),
                "anchor_gt_fg_precision": gt_precision,
                "anchor_gt_fg_recall": gt_recall,
            }
            out.append(item)

    summary = {}
    for key in out[0].keys():
        if key == "idx":
            continue
        summary[key] = _safe_mean([item[key] for item in out])
    return {
        "client": client_name,
        "cid": cid,
        "sup_type": sup_type,
        "iter": iter_id,
        "teacher_proxy": "mean(client_0..5 checkpoint states at same iter)",
        "student_checkpoint": os.path.basename(student_path),
        "num_samples": len(out),
        "summary": summary,
        "samples": out,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--iter", type=int, default=970)
    parser.add_argument("--max_samples", type=int, default=12)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    _disable_cuda_for_cpu_audit()
    results = [
        audit_client("client1", 0, "block", args.root_path, args.model_dir, args.iter, args.max_samples),
        audit_client("client2", 1, "keypoint", args.root_path, args.model_dir, args.iter, args.max_samples),
        audit_client("client3", 2, "scribble", args.root_path, args.model_dir, args.iter, args.max_samples),
        audit_client("client4", 3, "keypoint", args.root_path, args.model_dir, args.iter, args.max_samples),
        audit_client("client5", 4, "scribble", args.root_path, args.model_dir, args.iter, args.max_samples),
        audit_client("client6", 5, "box", args.root_path, args.model_dir, args.iter, args.max_samples),
    ]
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
