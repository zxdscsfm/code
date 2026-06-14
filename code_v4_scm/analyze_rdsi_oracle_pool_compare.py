# -*- coding:utf-8 -*-
"""Compare RDSI runs by offline teacher-pool oracle diagnostics.

This script never participates in training. It uses full GT only to answer
diagnostic questions:

1. Are WANN risk pixels actually error-prone?
2. In those risk pixels, does any external teacher outperform the student?
3. Which typed errors are correctable by the teacher pool?
4. Which teacher/client contributes the correctable regions?

The measured quantities are oracle upper bounds, not train-time signals.
"""

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analyze_rdsi_tcr_diagnostics import (
    auxiliary_logits,
    default_setup,
    dilate,
    load_model,
    parse_client_sup_types,
    primary_logits,
    to_image_tensor,
)
from dataloaders.dataset import BaseDataSets
from weak_annotation_reliability import build_wann_maps


def load_full_gt(root_path, rel_path):
    h5_path = os.path.join(root_path, rel_path)
    with h5py.File(h5_path, "r") as h5f:
        if "mask" not in h5f:
            raise KeyError("Full GT mask not found in {}".format(h5_path))
        return torch.from_numpy(h5f["mask"][:].astype(np.int64))


def ratio(numer_mask, denom_mask):
    denom = float(denom_mask.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 0.0
    return float((numer_mask.float() * denom_mask.float()).sum().detach().cpu().item() / denom)


def masked_mean(value, mask):
    denom = float(mask.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 0.0
    return float((value.float() * mask.float()).sum().detach().cpu().item() / denom)


def binary_dice(pred, gt):
    pred_fg = pred > 0
    gt_fg = gt > 0
    inter = float((pred_fg & gt_fg).float().sum().detach().cpu().item())
    denom = float(pred_fg.float().sum().detach().cpu().item() + gt_fg.float().sum().detach().cpu().item())
    if denom <= 0.0:
        return 1.0
    return 2.0 * inter / denom


def parse_run_specs(raw_specs):
    specs = []
    for raw in raw_specs:
        if "=" not in raw:
            raise ValueError("--run must be formatted as name=/path/to/snapshot")
        name, path = raw.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError("--run must be formatted as name=/path/to/snapshot")
        specs.append((name, path))
    return specs


def resolve_checkpoints(snapshot_path, pattern, model, num_clients):
    paths = []
    for cid in range(num_clients):
        path = os.path.join(snapshot_path, pattern.format(cid=cid, model=model))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


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
    obj.sup_type = args.client_sup_type_list[int(cid)]
    obj.label_prompt = args.label_prompt
    obj.img_size = args.img_size
    obj.img_class = args.img_class
    return obj


def load_run_models(args, snapshot_path, num_clients):
    checkpoints = resolve_checkpoints(snapshot_path, args.checkpoint_pattern, args.model, num_clients)
    models = []
    for cid, ckpt in enumerate(checkpoints):
        model_args = make_model_args(args, cid)
        old_values = (
            args.num_classes,
            args.in_chns,
            args.min_num_clients,
            getattr(args, "cid", None),
            getattr(args, "sup_type", None),
        )
        args.cid = cid
        args.sup_type = args.client_sup_type_list[cid]
        model = load_model(args, cid, ckpt)
        args.num_classes, args.in_chns, args.min_num_clients, old_cid, old_sup = old_values
        if old_cid is not None:
            args.cid = old_cid
        if old_sup is not None:
            args.sup_type = old_sup
        del model_args
        models.append(model)
    return models


def teacher_specialization_counts(teacher_ids, teacher_correct_stack, mask):
    counts = defaultdict(float)
    if not bool(mask.any().detach().cpu().item()):
        return counts
    masked_correct = teacher_correct_stack & mask.unsqueeze(0)
    any_correct = masked_correct.any(dim=0)
    if not bool(any_correct.any().detach().cpu().item()):
        return counts
    first_correct = masked_correct.float().argmax(dim=0)
    for stack_idx, teacher_cid in enumerate(teacher_ids):
        selected = any_correct & (first_correct == stack_idx)
        counts[int(teacher_cid)] += float(selected.float().sum().detach().cpu().item())
    return counts


def add_count_dict(prefix, out, counts, teacher_ids):
    total = sum(counts.values())
    out[prefix + "_pixels"] = float(total)
    for teacher_cid in teacher_ids:
        value = float(counts.get(int(teacher_cid), 0.0))
        out["{}_teacher{}_ratio".format(prefix, teacher_cid)] = value / total if total > 0.0 else 0.0


def run_one(args, run_name, snapshot_path, clients):
    models = load_run_models(args, snapshot_path, len(clients))
    rows = []

    with torch.no_grad():
        for target_cid, client in enumerate(clients):
            dataset = BaseDataSets(
                base_dir=args.root_path,
                split="train",
                transform=None,
                client=client,
                sup_type=args.client_sup_type_list[target_cid],
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
                student_prob = F.softmax(student_logits.detach(), dim=1)
                student_pred = student_prob.argmax(dim=1)
                maps = build_wann_maps(
                    image=images,
                    label=weak_label,
                    logits=student_logits,
                    aux_logits=auxiliary_logits(student_out),
                    sup_type=args.client_sup_type_list[target_cid],
                    img_class=args.img_class,
                    num_classes=args.num_classes,
                    iter_num=args.analysis_iter,
                    args=args,
                    ref_logits=None,
                )

                teacher_preds = []
                teacher_ids = []
                for teacher_cid, model in enumerate(models):
                    if int(args.exclude_same_client) == 1 and teacher_cid == target_cid:
                        continue
                    teacher_logits = primary_logits(model(images)).detach()
                    teacher_preds.append(F.softmax(teacher_logits, dim=1).argmax(dim=1))
                    teacher_ids.append(teacher_cid)
                if not teacher_preds:
                    continue
                teacher_pred_stack = torch.stack(teacher_preds, dim=0)
                teacher_correct_stack = teacher_pred_stack == gt.unsqueeze(0)
                teacher_wrong_stack = teacher_pred_stack != gt.unsqueeze(0)
                any_teacher_correct = teacher_correct_stack.any(dim=0)
                any_teacher_wrong = teacher_wrong_stack.any(dim=0)
                all_teacher_correct_frac = teacher_correct_stack.float().mean(dim=0)

                valid_mask = getattr(maps, "valid_mask", torch.ones_like(student_pred, dtype=torch.bool))
                hard_core = maps.core_mask & valid_mask
                risk_region = ((maps.ignore_mask | maps.soft_band) & (~hard_core) & valid_mask)
                gt_fg = gt > 0
                student_fg = student_pred > 0
                student_error = student_pred != gt
                risk_error = risk_region & student_error
                risk_correct = risk_region & (~student_error)
                fg_missing = risk_region & gt_fg & (~student_fg)
                fg_excess = risk_region & (~gt_fg) & student_fg
                boundary_truth = risk_region & dilate(gt_fg, 2) & dilate(~gt_fg, 2) & (~fg_missing) & (~fg_excess)
                boundary_error = boundary_truth & student_error

                oracle_pred = student_pred.clone()
                oracle_fix = risk_error & any_teacher_correct
                oracle_pred[oracle_fix] = gt[oracle_fix]

                for b, rel_path in enumerate(rel_paths):
                    row = {
                        "run": run_name,
                        "snapshot_path": snapshot_path,
                        "img_class": args.img_class,
                        "target_cid": target_cid,
                        "target_client": client,
                        "case": rel_path,
                        "sample_idx": int(idxs[b]),
                        "student_dice": binary_dice(student_pred[b], gt[b]),
                        "oracle_dice": binary_dice(oracle_pred[b], gt[b]),
                    }
                    row["oracle_dice_gain"] = row["oracle_dice"] - row["student_dice"]
                    masks = {
                        "risk": risk_region[b],
                        "core": hard_core[b],
                        "risk_error": risk_error[b],
                        "risk_correct": risk_correct[b],
                        "fg_missing": fg_missing[b],
                        "fg_excess": fg_excess[b],
                        "boundary": boundary_truth[b],
                        "boundary_error": boundary_error[b],
                    }
                    row["risk_ratio"] = float(masks["risk"].float().mean().detach().cpu().item())
                    row["core_ratio"] = float(masks["core"].float().mean().detach().cpu().item())
                    row["risk_error_rate"] = ratio(masks["risk_error"], masks["risk"])
                    row["core_error_rate"] = ratio(student_error[b], masks["core"])
                    row["risk_error_vs_core_error"] = row["risk_error_rate"] / max(row["core_error_rate"], 1e-6)
                    row["oracle_correctable_risk_error"] = ratio(any_teacher_correct[b] & masks["risk_error"], masks["risk_error"])
                    row["teacher_noise_on_risk_correct"] = ratio(any_teacher_wrong[b] & masks["risk_correct"], masks["risk_correct"])
                    row["teacher_correct_frac_on_risk_error"] = masked_mean(all_teacher_correct_frac[b], masks["risk_error"])
                    row["fg_missing_ratio_in_risk"] = ratio(masks["fg_missing"], masks["risk"])
                    row["fg_excess_ratio_in_risk"] = ratio(masks["fg_excess"], masks["risk"])
                    row["boundary_ratio_in_risk"] = ratio(masks["boundary"], masks["risk"])
                    row["oracle_fg_missing_recall"] = ratio(any_teacher_correct[b] & masks["fg_missing"], masks["fg_missing"])
                    row["oracle_fg_excess_recall"] = ratio(any_teacher_correct[b] & masks["fg_excess"], masks["fg_excess"])
                    row["oracle_boundary_recall"] = ratio(any_teacher_correct[b] & masks["boundary_error"], masks["boundary_error"])

                    for prefix, mask in [
                        ("spec_risk_error", masks["risk_error"]),
                        ("spec_fg_missing", masks["fg_missing"]),
                        ("spec_fg_excess", masks["fg_excess"]),
                        ("spec_boundary_error", masks["boundary_error"]),
                    ]:
                        counts = teacher_specialization_counts(teacher_ids, teacher_correct_stack[:, b], mask)
                        add_count_dict(prefix, row, counts, teacher_ids)
                    rows.append(row)

                processed += len(idxs)
                print("processed run={} client={} samples={}".format(run_name, client, processed), flush=True)

    for model in models:
        del model
    if args.device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize(df):
    metrics = [
        "student_dice",
        "oracle_dice",
        "oracle_dice_gain",
        "risk_ratio",
        "core_ratio",
        "risk_error_rate",
        "core_error_rate",
        "risk_error_vs_core_error",
        "oracle_correctable_risk_error",
        "teacher_noise_on_risk_correct",
        "teacher_correct_frac_on_risk_error",
        "fg_missing_ratio_in_risk",
        "fg_excess_ratio_in_risk",
        "boundary_ratio_in_risk",
        "oracle_fg_missing_recall",
        "oracle_fg_excess_recall",
        "oracle_boundary_recall",
    ]
    run_summary = df.groupby("run")[metrics].mean().reset_index()
    client_summary = df.groupby(["run", "target_client"])[metrics].mean().reset_index()
    return run_summary, client_summary


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="name=/snapshot/path. Can be repeated.")
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--img_class", choices=["prostate", "polyp", "faz"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="client_{cid}_async_{model}_best_model.pth")
    parser.add_argument("--client_sup_types", default="")
    parser.add_argument("--max_cases_per_client", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--analysis_iter", type=int, default=5000)
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
    os.makedirs(args.output_dir, exist_ok=True)

    frames = []
    for run_name, snapshot_path in parse_run_specs(args.run):
        frames.append(run_one(args, run_name, snapshot_path, clients))

    df = pd.concat(frames, ignore_index=True)
    sample_csv = os.path.join(args.output_dir, "oracle_pool_samples.csv")
    run_csv = os.path.join(args.output_dir, "oracle_pool_run_summary.csv")
    client_csv = os.path.join(args.output_dir, "oracle_pool_client_summary.csv")
    summary_json = os.path.join(args.output_dir, "oracle_pool_summary.json")
    df.to_csv(sample_csv, index=False)
    run_summary, client_summary = summarize(df)
    run_summary.to_csv(run_csv, index=False)
    client_summary.to_csv(client_csv, index=False)
    payload = {
        "img_class": args.img_class,
        "root_path": args.root_path,
        "analysis_iter": args.analysis_iter,
        "num_samples": int(len(df)),
        "runs": [{"name": name, "snapshot_path": path} for name, path in parse_run_specs(args.run)],
        "files": {
            "samples": sample_csv,
            "run_summary": run_csv,
            "client_summary": client_csv,
        },
        "run_summary": run_summary.to_dict(orient="records"),
        "client_summary": client_summary.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
