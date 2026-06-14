import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from datetime import datetime

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


SUP_TYPES = ["keypoint", "scribble", "block"]


def safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num) / float(den)


def signed_distance_to_boundary(gt_fg: np.ndarray) -> np.ndarray:
    gt_fg = gt_fg.astype(bool)
    inside = ndimage.distance_transform_edt(gt_fg)
    outside = ndimage.distance_transform_edt(~gt_fg)
    return inside - outside


def display_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr


def weak_overlay(image: np.ndarray, weak: np.ndarray) -> np.ndarray:
    base = display_image(image)
    rgb = np.stack([base, base, base], axis=-1)
    weak_bg = weak == 0
    weak_fg = weak == 1
    rgb[weak_bg] = 0.70 * rgb[weak_bg] + np.array([1.0, 0.9, 0.0]) * 0.30
    rgb[weak_fg] = 0.15 * rgb[weak_fg] + np.array([0.0, 1.0, 1.0]) * 0.85
    return np.clip(rgb, 0, 1)


def metric_record(weak: np.ndarray, gt_fg: np.ndarray) -> dict:
    weak_fg = weak == 1
    weak_bg = weak == 0
    inter = np.logical_and(weak_fg, gt_fg).sum()
    fg_pixels = gt_fg.sum()
    bg_pixels = (~gt_fg).sum()
    return {
        "fg_recall": safe_div(inter, fg_pixels),
        "fg_precision": safe_div(inter, weak_fg.sum()),
        "weak_fg_density": safe_div(weak_fg.sum(), weak.size),
        "labeled_density": safe_div((weak != 2).sum(), weak.size),
        "weak_bg_density": safe_div(weak_bg.sum(), weak.size),
        "weak_fg_outside_gt_rate": safe_div(np.logical_and(weak_fg, ~gt_fg).sum(), bg_pixels),
    }


def choose_example(records):
    candidates = []
    for rec in records:
        vals = rec["metrics"]
        kr = vals["keypoint"]["fg_recall"]
        sr = vals["scribble"]["fg_recall"]
        br = vals["block"]["fg_recall"]
        kp = vals["keypoint"]["fg_precision"]
        sp = vals["scribble"]["fg_precision"]
        bp = vals["block"]["fg_precision"]
        area = rec["gt_fg_density"]
        if not np.isfinite([kr, sr, br, kp, sp, bp, area]).all():
            continue
        # Prefer a clear hierarchy. If no partial scribble exists, this still
        # picks a sample that makes point sparsity and block over-inclusion visible.
        partial_bonus = 0.35 - abs(sr - 0.45)
        score = (br - kr) + 0.25 * partial_bonus + 0.15 * bp + min(area, 0.05)
        candidates.append((score, rec))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def plot_example(example, root_path: str, out_path: str) -> None:
    sample = example["sample"]
    with h5py.File(os.path.join(root_path, sample), "r") as f:
        image = f["image"][:]
        gt = (f["mask"][:] == 1).astype(np.uint8)
        weak = {sup: f[sup][:] for sup in SUP_TYPES}

    ys, xs = np.where(gt.astype(bool))
    margin = 45
    y0 = max(int(ys.min()) - margin, 0)
    y1 = min(int(ys.max()) + margin + 1, image.shape[-2])
    x0 = max(int(xs.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, image.shape[-1])
    sl = (slice(y0, y1), slice(x0, x1))

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    axes[0].imshow(display_image(image[sl]), cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(gt[sl], cmap="gray")
    axes[1].set_title(f"GT\nfg {example['gt_fg_density'] * 100:.2f}%")
    titles = {
        "keypoint": "Point label",
        "scribble": "Scribble label",
        "block": "Block label",
    }
    for ax, sup in zip(axes[2:], SUP_TYPES):
        ax.imshow(weak_overlay(image[sl], weak[sup][sl]))
        m = example["metrics"][sup]
        ax.set_title(
            f"{titles[sup]}\nrec {m['fg_recall']:.3f}, den {m['weak_fg_density'] * 100:.3f}%",
            fontsize=10,
        )
    for ax in axes:
        ax.axis("off")
    fig.text(
        0.5,
        0.02,
        "Cyan = weak foreground pixels, yellow = weak background pixels, unlabeled pixels are hidden",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_summary(summary, curve_rows, bins, out_path: str) -> None:
    colors = {"keypoint": "#1677ff", "scribble": "#19a974", "block": "#d9480f"}
    x = np.array([(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = ["fg_recall", "fg_precision", "labeled_density"]
    labels = ["FG recall", "FG precision", "Labeled density"]
    width = 0.25
    pos = np.arange(len(metrics))
    for offset, sup in zip([-width, 0, width], SUP_TYPES):
        vals = [summary[sup][m] for m in metrics]
        axes[0].bar(pos + offset, vals, width=width, label=sup, color=colors[sup])
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Dataset-level weak-label geometry")
    axes[0].legend(frameon=False)

    for sup in SUP_TYPES:
        fg_prob = np.array([curve_rows[sup][i]["weak_fg_prob"] for i in range(len(x))])
        axes[1].plot(x, fg_prob, label=sup, color=colors[sup], linewidth=2)
    axes[1].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Signed distance to GT boundary (pixels)")
    axes[1].set_ylabel("P(weak foreground)")
    axes[1].set_title("Foreground-label probability by geometry")
    axes[1].legend(frameon=False)

    for sup in SUP_TYPES:
        labeled_prob = np.array([curve_rows[sup][i]["labeled_prob"] for i in range(len(x))])
        axes[2].plot(x, labeled_prob, label=sup, color=colors[sup], linewidth=2)
    axes[2].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[2].set_xlabel("Signed distance to GT boundary (pixels)")
    axes[2].set_ylabel("P(labeled pixel)")
    axes[2].set_title("Supervision density by geometry")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_abs_dist", type=int, default=60)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        "/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs",
        "prostate_geometry_visual_proof_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    bins = np.arange(-args.max_abs_dist, args.max_abs_dist + 2, 2, dtype=np.float64)
    n_bins = len(bins) - 1
    curve = {
        sup: {
            "total": np.zeros(n_bins, dtype=np.float64),
            "weak_fg": np.zeros(n_bins, dtype=np.float64),
            "weak_bg": np.zeros(n_bins, dtype=np.float64),
            "labeled": np.zeros(n_bins, dtype=np.float64),
        }
        for sup in SUP_TYPES
    }
    metric_sums = {sup: defaultdict(float) for sup in SUP_TYPES}
    metric_counts = {sup: defaultdict(int) for sup in SUP_TYPES}
    sample_records = []

    paths = sorted(glob.glob(os.path.join(args.root_path, "Domain*", "train", "*.h5")))
    for path in paths:
        with h5py.File(path, "r") as f:
            if not all(k in f for k in ["image", "mask", *SUP_TYPES]):
                continue
            gt_fg = f["mask"][:] == 1
            if gt_fg.sum() == 0:
                continue
            dist = signed_distance_to_boundary(gt_fg)
            clipped = np.clip(dist, bins[0] + 1e-6, bins[-1] - 1e-6)
            bin_idx = np.digitize(clipped.ravel(), bins) - 1
            valid = (bin_idx >= 0) & (bin_idx < n_bins)
            rec = {
                "sample": os.path.relpath(path, args.root_path),
                "gt_fg_density": float(gt_fg.mean()),
                "metrics": {},
            }
            for sup in SUP_TYPES:
                weak = f[sup][:]
                sup_metrics = metric_record(weak, gt_fg)
                rec["metrics"][sup] = sup_metrics
                for key, value in sup_metrics.items():
                    if np.isfinite(value):
                        metric_sums[sup][key] += float(value)
                        metric_counts[sup][key] += 1

                flat_weak = weak.ravel()
                for b in range(n_bins):
                    idx = valid & (bin_idx == b)
                    if not np.any(idx):
                        continue
                    vals = flat_weak[idx]
                    curve[sup]["total"][b] += float(vals.size)
                    curve[sup]["weak_fg"][b] += float((vals == 1).sum())
                    curve[sup]["weak_bg"][b] += float((vals == 0).sum())
                    curve[sup]["labeled"][b] += float((vals != 2).sum())
            sample_records.append(rec)

    summary = {
        sup: {
            key: safe_div(metric_sums[sup][key], metric_counts[sup][key])
            for key in sorted(metric_sums[sup])
        }
        for sup in SUP_TYPES
    }
    curve_rows = {sup: [] for sup in SUP_TYPES}
    for sup in SUP_TYPES:
        total = curve[sup]["total"]
        for i in range(n_bins):
            curve_rows[sup].append(
                {
                    "bin_left": float(bins[i]),
                    "bin_right": float(bins[i + 1]),
                    "weak_fg_prob": safe_div(curve[sup]["weak_fg"][i], total[i]),
                    "weak_bg_prob": safe_div(curve[sup]["weak_bg"][i], total[i]),
                    "labeled_prob": safe_div(curve[sup]["labeled"][i], total[i]),
                    "n_pixels": int(total[i]),
                }
            )

    summary_png = os.path.join(out_dir, "prostate_geometry_bias_statistical_visual.png")
    plot_summary(summary, curve_rows, bins, summary_png)

    example = choose_example(sample_records)
    example_png = os.path.join(out_dir, "prostate_geometry_bias_example.png")
    plot_example(example, args.root_path, example_png)

    summary_json = os.path.join(out_dir, "prostate_geometry_bias_summary.json")
    with open(summary_json, "w") as f:
        json.dump(
            {
                "root_path": args.root_path,
                "n_samples": len(sample_records),
                "summary": summary,
                "example": example,
                "summary_png": summary_png,
                "example_png": example_png,
            },
            f,
            indent=2,
        )

    curve_csv = os.path.join(out_dir, "prostate_geometry_bias_distance_curves.csv")
    with open(curve_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sup_type", "bin_left", "bin_right", "weak_fg_prob", "weak_bg_prob", "labeled_prob", "n_pixels"],
        )
        writer.writeheader()
        for sup in SUP_TYPES:
            for row in curve_rows[sup]:
                writer.writerow({"sup_type": sup, **row})

    print(summary_png)
    print(example_png)
    print(summary_json)
    print(curve_csv)
    print(json.dumps({"n_samples": len(sample_records), "summary": summary, "example": example}, indent=2))


if __name__ == "__main__":
    main()
