import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.nn.Module.cuda = lambda self, *args, **kwargs: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self

from dataloaders.dataset import BaseDataSets  # noqa: E402
from networks.net_factory import net_factory  # noqa: E402


@dataclass
class RunConfig:
    name: str
    model_dir: str


ROOT_PATH = "/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd"
CLIENTS = ["client1", "client2", "client3", "client4", "client5", "client6"]
SUP_TYPES = ["scribble", "keypoint", "scribble", "block", "scribble", "scribble"]
RUNS = [
    RunConfig(
        name="FedLPPA",
        model_dir="/data/jianbingshen/yanghongji/fedlppaSOTA/model/prostate/FedLPPA_sota_diag_fedlppa_rdsi3_sd_prostate_r500_l10_20260608_144436_seed2022",
    ),
    RunConfig(
        name="WANN_ACG",
        model_dir="/data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_wannacggeocal_prostate_r500_l10_20260609_041125_seed2022",
    ),
]


class Args:
    pass


def model_args(cid: int, sup_type: str) -> Args:
    args = Args()
    args.cid = cid
    args.min_num_clients = len(CLIENTS)
    args.prompt = "universal"
    args.attention = "dual"
    args.dual_init = "aggregated"
    args.label_prompt = 1
    args.img_size = 384
    args.sup_type = sup_type
    args.device = "cpu"
    args.use_cuda = 0
    return args


def load_model(run: RunConfig, cid: int, sup_type: str):
    ckpt = os.path.join(run.model_dir, f"client_{cid}_unet_univ5_best_model.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    net = net_factory(model_args(cid, sup_type), net_type="unet_univ5", in_chns=1, class_num=2)
    net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    net.eval()
    return net, ckpt


def to_tensor(image: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(image).float()
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        if x.shape[0] != 1:
            x = x[:1]
        x = x.unsqueeze(0)
    return x


def predict(net, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        out = net(to_tensor(image))[0]
        pred = torch.argmax(torch.softmax(out, dim=1), dim=1)
    return pred.squeeze(0).detach().cpu().numpy().astype(np.uint8)


def safe_div(num: float, den: float) -> float:
    return float("nan") if den == 0 else float(num) / float(den)


def binary_stats(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred_fg = pred.astype(bool)
    gt_fg = gt.astype(bool)
    total = float(gt_fg.size)
    pred_sum = float(pred_fg.sum())
    gt_sum = float(gt_fg.sum())
    gt_bg_sum = total - gt_sum
    tp = float(np.logical_and(pred_fg, gt_fg).sum())
    fp = float(np.logical_and(pred_fg, ~gt_fg).sum())
    fn = float(np.logical_and(~pred_fg, gt_fg).sum())
    tn = float(np.logical_and(~pred_fg, ~gt_fg).sum())
    return {
        "dice": safe_div(2.0 * tp, 2.0 * tp + fp + fn),
        "precision": safe_div(tp, tp + fp),
        "recall": safe_div(tp, tp + fn),
        "specificity": safe_div(tn, tn + fp),
        "fg_ratio": safe_div(pred_sum, gt_sum),
        "fp_bg_rate": safe_div(fp, gt_bg_sum),
        "fn_fg_rate": safe_div(fn, gt_sum),
    }


def display_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    lo, hi = np.percentile(arr, [1, 99])
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return arr


def error_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = display_image(image)
    rgb = np.stack([base, base, base], axis=-1)
    gt_fg = gt == 1
    pred_fg = pred == 1
    tp = pred_fg & gt_fg
    fp = pred_fg & ~gt_fg
    fn = ~pred_fg & gt_fg
    rgb[tp] = 0.55 * rgb[tp] + np.array([0.0, 0.80, 0.0]) * 0.45
    rgb[fp] = 0.45 * rgb[fp] + np.array([1.0, 0.0, 0.0]) * 0.55
    rgb[fn] = 0.45 * rgb[fn] + np.array([0.0, 0.25, 1.0]) * 0.55
    return np.clip(rgb, 0, 1)


def weak_overlay(image: np.ndarray, weak: np.ndarray) -> np.ndarray:
    base = display_image(image)
    rgb = np.stack([base, base, base], axis=-1)
    weak_fg = weak == 1
    weak_bg = weak == 0
    rgb[weak_bg] = 0.65 * rgb[weak_bg] + np.array([1.0, 1.0, 0.0]) * 0.35
    rgb[weak_fg] = 0.30 * rgb[weak_fg] + np.array([0.0, 1.0, 1.0]) * 0.70
    return np.clip(rgb, 0, 1)


def save_figure(row: Dict[str, object], out_path: str) -> None:
    image = row["image"]
    gt = row["gt"]
    weak = row["weak"]
    base_pred = row["FedLPPA_pred"]
    method_pred = row["WANN_ACG_pred"]
    base = row["FedLPPA"]
    method = row["WANN_ACG"]

    fig, axes = plt.subplots(1, 5, figsize=(17, 4))
    axes[0].imshow(display_image(image), cmap="gray")
    axes[0].set_title("Image")
    axes[1].imshow(weak_overlay(image, weak))
    axes[1].set_title(f"Weak label: {row['sup_type']}")
    axes[2].imshow(gt, cmap="gray")
    axes[2].set_title("GT mask")
    axes[3].imshow(error_overlay(image, gt, base_pred))
    axes[3].set_title(
        f"FedLPPA\nD={base['dice']:.3f}, R={base['fg_ratio']:.2f}",
        fontsize=10,
    )
    axes[4].imshow(error_overlay(image, gt, method_pred))
    axes[4].set_title(
        f"Ours\nD={method['dice']:.3f}, R={method['fg_ratio']:.2f}",
        fontsize=10,
    )
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Prostate {row['client']} {row['sample_id']} | green=TP, red=FP, blue=FN",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--top_n", type=int, default=12)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        "/data/jianbingshen/yanghongji/FedLPPA_Original/code_v4/logs",
        "prostate_baseline_vs_wacggeocal_vis_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for cid, (client, sup_type) in enumerate(zip(CLIENTS, SUP_TYPES)):
        db = BaseDataSets(ROOT_PATH, split="val", transform=None, client=client, sup_type=sup_type, img_class="prostate")
        models = {}
        for run in RUNS:
            models[run.name], _ = load_model(run, cid, sup_type)
        for sample_id in db.sample_list:
            h5_path = os.path.join(ROOT_PATH, sample_id)
            with h5py.File(h5_path, "r") as f:
                image = f["image"][:]
                gt = (f["mask"][:] == 1).astype(np.uint8)
                weak = f[sup_type][:] if sup_type in f else np.full_like(gt, 2, dtype=np.uint8)
            record: Dict[str, object] = {
                "client": client,
                "cid": cid,
                "sup_type": sup_type,
                "sample_id": sample_id,
                "image": image,
                "gt": gt,
                "weak": weak,
            }
            for run in RUNS:
                pred = predict(models[run.name], image)
                record[f"{run.name}_pred"] = pred
                record[run.name] = binary_stats(pred == 1, gt == 1)
            base = record["FedLPPA"]
            method = record["WANN_ACG"]
            dice_gain = method["dice"] - base["dice"]
            base_fg_error = abs(base["fg_ratio"] - 1.0)
            method_fg_error = abs(method["fg_ratio"] - 1.0)
            record["dice_gain"] = dice_gain
            record["fg_error_reduction"] = base_fg_error - method_fg_error
            record["rank_score"] = dice_gain + 0.15 * max(record["fg_error_reduction"], 0.0)
            rows.append(record)
        del models

    rows = sorted(rows, key=lambda r: r["rank_score"], reverse=True)
    manifest_rows = []
    for idx, row in enumerate(rows[: args.top_n]):
        safe_sample = os.path.basename(str(row["sample_id"])).replace(".h5", "")
        fig_path = os.path.join(out_dir, f"{idx:02d}_{row['client']}_{row['sup_type']}_{safe_sample}.png")
        save_figure(row, fig_path)
        manifest_rows.append({
            "rank": idx,
            "client": row["client"],
            "cid": row["cid"],
            "sup_type": row["sup_type"],
            "sample_id": row["sample_id"],
            "figure": fig_path,
            "dice_gain": row["dice_gain"],
            "fg_error_reduction": row["fg_error_reduction"],
            "FedLPPA_dice": row["FedLPPA"]["dice"],
            "FedLPPA_fg_ratio": row["FedLPPA"]["fg_ratio"],
            "FedLPPA_precision": row["FedLPPA"]["precision"],
            "FedLPPA_recall": row["FedLPPA"]["recall"],
            "Ours_dice": row["WANN_ACG"]["dice"],
            "Ours_fg_ratio": row["WANN_ACG"]["fg_ratio"],
            "Ours_precision": row["WANN_ACG"]["precision"],
            "Ours_recall": row["WANN_ACG"]["recall"],
        })

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(out_dir)
    print(manifest_path)
    print("top")
    for row in manifest_rows[:5]:
        print(row)


if __name__ == "__main__":
    main()
