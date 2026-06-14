# -*- coding:utf-8 -*-
"""Offline teacher-pool consensus safety diagnostics for TED checkpoints.

This script uses GT only for diagnosis. It does not train or write model
weights. It measures whether external teacher-pool agreement can identify
student false positive, false negative, and boundary-error regions.
"""

import argparse
import json
import os

import h5py
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
    ratio,
    to_image_tensor,
)
from dataloaders.dataset import BaseDataSets
from weak_annotation_reliability import build_wann_maps


def load_full_gt(root_path, rel_path):
    h5_path = os.path.join(root_path, rel_path)
    with h5py.File(h5_path, "r") as h5f:
        if "mask" not in h5f:
            raise KeyError("Full GT mask not found in {}".format(h5_path))
        return torch.from_numpy(h5f["mask"][:].astype("int64"))


def resolve_checkpoints(snapshot_path, pattern, model, num_clients):
    paths = []
    for cid in range(num_clients):
        path = os.path.join(snapshot_path, pattern.format(cid=cid, model=model))
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def load_models(args, clients):
    checkpoints = resolve_checkpoints(args.snapshot_path, args.checkpoint_pattern, args.model, len(clients))
    models = []
    for cid, ckpt in enumerate(checkpoints):
        args.cid = cid
        args.sup_type = args.client_sup_type_list[cid]
        models.append(load_model(args, cid, ckpt))
    return models


def count(mask):
    return float(mask.float().sum().detach().cpu().item())


def run(args):
    clients, num_classes, in_chns, min_num_clients = default_setup(args.img_class)
    args.num_classes = num_classes
    args.in_chns = in_chns
    args.min_num_clients = min_num_clients
    args.client_sup_type_list = parse_client_sup_types(args, len(clients))

    models = load_models(args, clients)
    case_rows = []
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
                rel_paths = [base_sample_list[int(i)] for i in idxs]
                gt = torch.stack([load_full_gt(args.root_path, p) for p in rel_paths], dim=0).long().to(args.device)
                gt_fg = gt > 0

                student_out = models[target_cid](images)
                student_logits = primary_logits(student_out)
                student_prob = F.softmax(student_logits.detach(), dim=1)
                student_pred = student_prob.argmax(dim=1)
                student_fg = student_pred > 0

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
                valid = getattr(maps, "valid_mask", torch.ones_like(student_pred, dtype=torch.bool))
                hard_core = maps.core_mask & valid
                risk = ((maps.ignore_mask | maps.soft_band) & (~hard_core) & valid)

                teacher_preds = []
                for teacher_cid, model in enumerate(models):
                    if int(args.exclude_same_client) == 1 and teacher_cid == target_cid:
                        continue
                    out = model(images)
                    teacher_preds.append(F.softmax(primary_logits(out).detach(), dim=1).argmax(dim=1))
                if not teacher_preds:
                    continue

                stack = torch.stack(teacher_preds, dim=0)
                teacher_fg_vote = (stack > 0).float().mean(dim=0)
                teacher_bg_vote = (stack == 0).float().mean(dim=0)
                majority_fg = teacher_fg_vote >= args.majority_thresh
                majority_bg = teacher_bg_vote >= args.majority_thresh
                strong_fg = teacher_fg_vote >= args.strong_thresh
                strong_bg = teacher_bg_vote >= args.strong_thresh
                disagree = (teacher_fg_vote > (1.0 - args.strong_thresh)) & (teacher_fg_vote < args.strong_thresh)

                student_error = student_pred != gt
                fp = risk & student_fg & (~gt_fg)
                fn = risk & (~student_fg) & gt_fg
                risk_correct = risk & (~student_error)
                gt_boundary = risk & dilate(gt_fg, args.boundary_radius) & dilate(~gt_fg, args.boundary_radius)
                boundary_error = gt_boundary & student_error
                boundary_correct = gt_boundary & (~student_error)

                bg_suppression = risk & student_fg & majority_bg
                fg_expansion = risk & (~student_fg) & majority_fg
                boundary_consensus = gt_boundary & (majority_fg | majority_bg) & (~disagree)

                for b, rel_path in enumerate(rel_paths):
                    row = {
                        "img_class": args.img_class,
                        "target_cid": int(target_cid),
                        "target_client": client,
                        "case": rel_path,
                        "sample_idx": int(idxs[b]),
                        "risk_ratio": count(risk[b]) / risk[b].numel(),
                        "core_ratio": count(hard_core[b]) / hard_core[b].numel(),
                        "risk_error_rate": ratio(student_error[b] & risk[b], risk[b]),
                        "fp_ratio_in_risk": ratio(fp[b], risk[b]),
                        "fn_ratio_in_risk": ratio(fn[b], risk[b]),
                        "boundary_error_ratio_in_boundary": ratio(boundary_error[b], gt_boundary[b]),
                    }
                    for name, mask in [
                        ("majority_bg", majority_bg[b]),
                        ("strong_bg", strong_bg[b]),
                        ("majority_fg", majority_fg[b]),
                        ("strong_fg", strong_fg[b]),
                        ("disagree", disagree[b]),
                    ]:
                        row[name + "_risk_ratio"] = ratio(mask & risk[b], risk[b])
                        row[name + "_on_fp"] = ratio(mask & fp[b], fp[b])
                        row[name + "_on_fn"] = ratio(mask & fn[b], fn[b])
                        row[name + "_on_risk_correct"] = ratio(mask & risk_correct[b], risk_correct[b])
                        row[name + "_on_boundary_error"] = ratio(mask & boundary_error[b], boundary_error[b])
                        row[name + "_on_boundary_correct"] = ratio(mask & boundary_correct[b], boundary_correct[b])

                    row["bg_suppression_precision"] = ratio(fp[b] & bg_suppression[b], bg_suppression[b])
                    row["bg_suppression_fp_recall"] = ratio(fp[b] & bg_suppression[b], fp[b])
                    row["fg_expansion_precision"] = ratio(fn[b] & fg_expansion[b], fg_expansion[b])
                    row["fg_expansion_fn_recall"] = ratio(fn[b] & fg_expansion[b], fn[b])
                    row["boundary_consensus_error_precision"] = ratio(boundary_error[b] & boundary_consensus[b], boundary_consensus[b])
                    row["boundary_consensus_error_recall"] = ratio(boundary_error[b] & boundary_consensus[b], boundary_error[b])
                    case_rows.append(row)

                processed += len(idxs)
                print("processed {} client={} samples={}".format(args.img_class, client, processed), flush=True)

    df = pd.DataFrame(case_rows)
    metrics = [c for c in df.columns if c not in ["img_class", "target_cid", "target_client", "case", "sample_idx"]]
    run_summary = df[metrics].mean().to_frame().T
    run_summary.insert(0, "img_class", args.img_class)
    client_summary = df.groupby(["img_class", "target_client"])[metrics].mean().reset_index()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_csv = os.path.join(args.output_dir, "pool_consensus_samples.csv")
    run_csv = os.path.join(args.output_dir, "pool_consensus_run_summary.csv")
    client_csv = os.path.join(args.output_dir, "pool_consensus_client_summary.csv")
    summary_json = os.path.join(args.output_dir, "pool_consensus_summary.json")
    df.to_csv(sample_csv, index=False)
    run_summary.to_csv(run_csv, index=False)
    client_summary.to_csv(client_csv, index=False)
    payload = {
        "img_class": args.img_class,
        "snapshot_path": args.snapshot_path,
        "root_path": args.root_path,
        "num_samples": int(len(df)),
        "files": {"samples": sample_csv, "run_summary": run_csv, "client_summary": client_csv},
        "run_summary": run_summary.to_dict(orient="records"),
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot_path", required=True)
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--img_class", choices=["prostate", "polyp", "faz"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_pattern", default="client_{cid}_async_{model}_best_model.pth")
    parser.add_argument("--client_sup_types", default="")
    parser.add_argument("--max_cases_per_client", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--analysis_iter", type=int, default=5000)
    parser.add_argument("--exclude_same_client", type=int, default=1)
    parser.add_argument("--model", default="unet_univ5")
    parser.add_argument("--prompt", default="universal")
    parser.add_argument("--attention", default="dual")
    parser.add_argument("--label_prompt", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--majority_thresh", type=float, default=0.5)
    parser.add_argument("--strong_thresh", type=float, default=0.75)
    parser.add_argument("--boundary_radius", type=int, default=2)

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
            raise RuntimeError("CUDA requested but not available")
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    run(args)


if __name__ == "__main__":
    main()
