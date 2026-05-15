"""Quantify weak-label density and spatial distribution for FedLPPA datasets.

This script produces:
1. Per-sample metrics for the assigned weak label used by each client.
2. Per-client summaries of foreground density and spatial support.
3. Per-key pooled summaries (e.g. scribble/block/box) across all clients.
4. Client heatmaps and density histograms for quick visual inspection.

The main metric requested by the user is `ann_fg_ratio`, the fraction of
pixels marked as foreground by the weak annotation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


ASSIGNED_SUP_TYPES: Dict[str, List[str]] = {
    "faz": ["scribble_noisy", "keypoint", "block", "box", "scribble"],
    "odoc": ["scribble", "scribble_noisy", "scribble_noisy", "keypoint", "block"],
    "prostate": ["block", "keypoint", "scribble", "keypoint", "scribble", "box"],
    "polyp": ["keypoint", "scribble", "box", "block"],
}

CANONICAL_ANALYSIS_KEYS = [
    "scribble",
    "scribble_noisy",
    "keypoint",
    "block",
    "box",
]


@dataclass
class DatasetSpec:
    name: str
    root: str
    assigned_sup_types: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(ASSIGNED_SUP_TYPES.keys()))
    parser.add_argument("--root", required=True, help="Dataset root that contains Domain*/train and Domain*/test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-width", type=int, default=5)
    parser.add_argument(
        "--pooled-keys",
        default=",".join(CANONICAL_ANALYSIS_KEYS),
        help="Comma-separated annotation keys to pool across clients",
    )
    parser.add_argument(
        "--limit-per-client",
        type=int,
        default=0,
        help="Optional cap on the number of train samples analyzed per client",
    )
    return parser.parse_args()


def list_client_train_files(root: str) -> List[Tuple[str, List[str]]]:
    domain_dirs = sorted(
        d for d in os.listdir(root) if d.startswith("Domain") and os.path.isdir(os.path.join(root, d))
    )
    client_files: List[Tuple[str, List[str]]] = []
    for domain in domain_dirs:
        train_dir = os.path.join(root, domain, "train")
        files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
        client_files.append((domain, files))
    return client_files


def infer_unlabeled_value(label: np.ndarray, dataset: str) -> int | None:
    values = np.unique(label)
    if dataset == "odoc":
        return 3 if 3 in values else None
    if dataset in {"faz", "polyp", "prostate"}:
        return 2 if 2 in values else None
    return None


def foreground_mask(label: np.ndarray, dataset: str) -> np.ndarray:
    unlabeled = infer_unlabeled_value(label, dataset)
    if unlabeled is None:
        return label > 0
    return np.logical_and(label > 0, label != unlabeled)


def labeled_mask(label: np.ndarray, dataset: str) -> np.ndarray:
    unlabeled = infer_unlabeled_value(label, dataset)
    if unlabeled is None:
        return np.ones_like(label, dtype=bool)
    return label != unlabeled


def binary_boundary_and_core(mask: np.ndarray, width: int) -> Tuple[np.ndarray, np.ndarray]:
    if width <= 0:
        return mask.copy(), mask.copy()
    if not np.any(mask):
        empty = np.zeros_like(mask, dtype=bool)
        return empty, empty
    eroded = ndimage.binary_erosion(mask, iterations=width, border_value=0)
    boundary = np.logical_and(mask, np.logical_not(eroded))
    core = eroded
    return boundary, core


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(arr.size),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "min": float(np.nanmin(arr)),
        "median": float(np.nanmedian(arr)),
        "max": float(np.nanmax(arr)),
    }


def save_heatmap_pair(
    ann_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(ann_heatmap, cmap="magma", vmin=0.0, vmax=max(1e-6, float(ann_heatmap.max())))
    axes[0].set_title("Weak FG Frequency")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(gt_heatmap, cmap="viridis", vmin=0.0, vmax=max(1e-6, float(gt_heatmap.max())))
    axes[1].set_title("GT FG Frequency")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_density_histogram(
    series_by_name: Dict[str, List[float]],
    out_path: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in series_by_name.items():
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        ax.hist(arr, bins=30, alpha=0.45, label=name)
    ax.set_xlabel("Weak FG ratio")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def analyze_annotation(
    ann: np.ndarray,
    gt: np.ndarray,
    dataset: str,
    boundary_width: int,
) -> Dict[str, float]:
    ann_fg = foreground_mask(ann, dataset)
    ann_labeled = labeled_mask(ann, dataset)
    gt_fg = foreground_mask(gt, dataset)

    total_px = int(ann.size)
    ann_fg_px = int(np.count_nonzero(ann_fg))
    ann_labeled_px = int(np.count_nonzero(ann_labeled))
    gt_fg_px = int(np.count_nonzero(gt_fg))

    boundary, core = binary_boundary_and_core(gt_fg, boundary_width)
    ann_fg_on_gt = int(np.count_nonzero(np.logical_and(ann_fg, gt_fg)))
    ann_fg_on_boundary = int(np.count_nonzero(np.logical_and(ann_fg, boundary)))
    ann_fg_in_core = int(np.count_nonzero(np.logical_and(ann_fg, core)))

    return {
        "total_px": total_px,
        "ann_fg_px": ann_fg_px,
        "ann_labeled_px": ann_labeled_px,
        "gt_fg_px": gt_fg_px,
        "ann_fg_ratio": safe_div(ann_fg_px, total_px),
        "ann_labeled_ratio": safe_div(ann_labeled_px, total_px),
        "gt_fg_ratio": safe_div(gt_fg_px, total_px),
        "ann_fg_inside_gt_ratio": safe_div(ann_fg_on_gt, ann_fg_px),
        "gt_fg_covered_by_ann_ratio": safe_div(ann_fg_on_gt, gt_fg_px),
        "ann_fg_boundary_ratio": safe_div(ann_fg_on_boundary, ann_fg_px),
        "ann_fg_core_ratio": safe_div(ann_fg_in_core, ann_fg_px),
    }


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spec = DatasetSpec(
        name=args.dataset,
        root=args.root,
        assigned_sup_types=ASSIGNED_SUP_TYPES[args.dataset],
    )
    pooled_keys = [k.strip() for k in args.pooled_keys.split(",") if k.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    client_train_files = list_client_train_files(spec.root)
    if len(client_train_files) != len(spec.assigned_sup_types):
        raise ValueError(
            f"Dataset {spec.name} expects {len(spec.assigned_sup_types)} clients, "
            f"but found {len(client_train_files)} domains under {spec.root}"
        )

    per_sample_rows: List[Dict[str, object]] = []
    per_client_metric_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_key_metric_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_client_hist_values: Dict[str, List[float]] = defaultdict(list)
    per_key_hist_values: Dict[str, List[float]] = defaultdict(list)
    per_client_heatmaps: Dict[str, np.ndarray] = {}
    per_client_gt_heatmaps: Dict[str, np.ndarray] = {}
    per_client_heatmap_counts: Dict[str, int] = defaultdict(int)

    for client_idx, (domain, files) in enumerate(client_train_files):
        assigned_key = spec.assigned_sup_types[client_idx]
        if args.limit_per_client > 0:
            files = files[: args.limit_per_client]
        client_name = f"client{client_idx + 1}"

        for file_path in files:
            with h5py.File(file_path, "r") as h5f:
                gt = h5f["mask"][:]
                ann = h5f[assigned_key][:]
                metrics = analyze_annotation(ann, gt, spec.name, args.boundary_width)
                row = {
                    "dataset": spec.name,
                    "client": client_name,
                    "domain": domain,
                    "assigned_key": assigned_key,
                    "file": os.path.basename(file_path),
                }
                row.update(metrics)
                per_sample_rows.append(row)

                for metric_name, value in metrics.items():
                    if metric_name.endswith("_px") or metric_name == "total_px":
                        continue
                    per_client_metric_buckets[client_name][metric_name].append(value)
                per_client_hist_values[client_name].append(metrics["ann_fg_ratio"])

                ann_fg = foreground_mask(ann, spec.name).astype(np.float32)
                gt_fg = foreground_mask(gt, spec.name).astype(np.float32)
                if client_name not in per_client_heatmaps:
                    per_client_heatmaps[client_name] = np.zeros_like(ann_fg, dtype=np.float64)
                    per_client_gt_heatmaps[client_name] = np.zeros_like(gt_fg, dtype=np.float64)
                per_client_heatmaps[client_name] += ann_fg
                per_client_gt_heatmaps[client_name] += gt_fg
                per_client_heatmap_counts[client_name] += 1

                available_keys = [k for k in pooled_keys if k in h5f]
                for pooled_key in available_keys:
                    pooled_metrics = analyze_annotation(h5f[pooled_key][:], gt, spec.name, args.boundary_width)
                    for metric_name, value in pooled_metrics.items():
                        if metric_name.endswith("_px") or metric_name == "total_px":
                            continue
                        per_key_metric_buckets[pooled_key][metric_name].append(value)
                    per_key_hist_values[pooled_key].append(pooled_metrics["ann_fg_ratio"])

    per_sample_fields = [
        "dataset",
        "client",
        "domain",
        "assigned_key",
        "file",
        "total_px",
        "ann_fg_px",
        "ann_labeled_px",
        "gt_fg_px",
        "ann_fg_ratio",
        "ann_labeled_ratio",
        "gt_fg_ratio",
        "ann_fg_inside_gt_ratio",
        "gt_fg_covered_by_ann_ratio",
        "ann_fg_boundary_ratio",
        "ann_fg_core_ratio",
    ]
    write_csv(os.path.join(args.output_dir, "per_sample_assigned.csv"), per_sample_rows, per_sample_fields)

    client_summary_rows: List[Dict[str, object]] = []
    for client_name, metric_buckets in sorted(per_client_metric_buckets.items()):
        row: Dict[str, object] = {"dataset": spec.name, "client": client_name}
        for metric_name, values in sorted(metric_buckets.items()):
            metric_summary = summarize(values)
            for stat_name, stat_value in metric_summary.items():
                row[f"{metric_name}_{stat_name}"] = stat_value
        client_summary_rows.append(row)

        heatmap = per_client_heatmaps[client_name] / max(1, per_client_heatmap_counts[client_name])
        gt_heatmap = per_client_gt_heatmaps[client_name] / max(1, per_client_heatmap_counts[client_name])
        save_heatmap_pair(
            heatmap,
            gt_heatmap,
            os.path.join(args.output_dir, f"{client_name}_heatmap.png"),
            f"{spec.name.upper()} {client_name} ({spec.assigned_sup_types[int(client_name.replace('client', '')) - 1]})",
        )

    client_summary_fields = sorted({key for row in client_summary_rows for key in row.keys()})
    write_csv(os.path.join(args.output_dir, "per_client_assigned_summary.csv"), client_summary_rows, client_summary_fields)

    pooled_summary_rows: List[Dict[str, object]] = []
    for pooled_key, metric_buckets in sorted(per_key_metric_buckets.items()):
        row = {"dataset": spec.name, "pooled_key": pooled_key}
        for metric_name, values in sorted(metric_buckets.items()):
            metric_summary = summarize(values)
            for stat_name, stat_value in metric_summary.items():
                row[f"{metric_name}_{stat_name}"] = stat_value
        pooled_summary_rows.append(row)
    pooled_summary_fields = sorted({key for row in pooled_summary_rows for key in row.keys()})
    write_csv(os.path.join(args.output_dir, "per_key_pooled_summary.csv"), pooled_summary_rows, pooled_summary_fields)

    save_density_histogram(
        per_client_hist_values,
        os.path.join(args.output_dir, "client_assigned_density_hist.png"),
        f"{spec.name.upper()} assigned weak-label density",
    )
    save_density_histogram(
        per_key_hist_values,
        os.path.join(args.output_dir, "pooled_key_density_hist.png"),
        f"{spec.name.upper()} pooled weak-label density by key",
    )

    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
