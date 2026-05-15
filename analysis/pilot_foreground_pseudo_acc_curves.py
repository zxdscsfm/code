import argparse
import glob
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")


REPO_CODE_DIR = "/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/code_v4"


@dataclass(frozen=True)
class ClientSpec:
    cid: int
    client_name: str
    domain: str
    sup_type: str
    label: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    title: str
    model_root: str
    data_root: str
    in_chns: int
    num_classes: int
    img_size: int
    client_specs: Dict[int, ClientSpec]


DATASETS: Dict[str, DatasetConfig] = {
    "odoc": DatasetConfig(
        name="odoc",
        title="ODOC clean-baseline",
        model_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/model/odoc/FedLPPA_official_clean_odoc_paper_r500_l10_20260418_041021_seed2022",
        data_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/data/ODOC_h5",
        in_chns=3,
        num_classes=3,
        img_size=384,
        client_specs={
            0: ClientSpec(0, "client1", "Domain1", "scribble", "client_0 / Domain1 / scribble"),
            1: ClientSpec(1, "client2", "Domain2", "scribble_noisy", "client_1 / Domain2 / scribble_noisy"),
            2: ClientSpec(2, "client3", "Domain3", "scribble_noisy", "client_2 / Domain3 / scribble_noisy"),
            3: ClientSpec(3, "client4", "Domain4", "keypoint", "client_3 / Domain4 / keypoint"),
            4: ClientSpec(4, "client5", "Domain5", "block", "client_4 / Domain5 / block"),
        },
    ),
    "faz": DatasetConfig(
        name="faz",
        title="FAZ clean-baseline",
        model_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/model/faz/FedLPPA_official_clean_faz_paper_r500_l10_20260425_142513_seed2022",
        data_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/data/FAZ_h5",
        in_chns=1,
        num_classes=2,
        img_size=256,
        client_specs={
            0: ClientSpec(0, "client1", "Domain1", "scribble_noisy", "client_0 / Domain1 / scribble_noisy"),
            1: ClientSpec(1, "client2", "Domain2", "keypoint", "client_1 / Domain2 / keypoint"),
            2: ClientSpec(2, "client3", "Domain3", "block", "client_2 / Domain3 / block"),
            3: ClientSpec(3, "client4", "Domain4", "box", "client_3 / Domain4 / box"),
            4: ClientSpec(4, "client5", "Domain5", "scribble", "client_4 / Domain5 / scribble"),
        },
    ),
    "prostate": DatasetConfig(
        name="prostate",
        title="Prostate clean-baseline",
        model_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/model/prostate/FedLPPA_official_clean_prostate_paper_r500_l10_20260429_054819_seed2022",
        data_root="/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/PROSTATE_h5",
        in_chns=1,
        num_classes=2,
        img_size=384,
        client_specs={
            0: ClientSpec(0, "client1", "Domain1", "block", "client_0 / Domain1 / block"),
            1: ClientSpec(1, "client2", "Domain2", "keypoint", "client_1 / Domain2 / keypoint"),
            2: ClientSpec(2, "client3", "Domain3", "scribble", "client_2 / Domain3 / scribble"),
            3: ClientSpec(3, "client4", "Domain4", "keypoint", "client_3 / Domain4 / keypoint"),
            4: ClientSpec(4, "client5", "Domain5", "scribble", "client_4 / Domain5 / scribble"),
            5: ClientSpec(5, "client6", "Domain6", "box", "client_5 / Domain6 / box"),
        },
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["odoc", "faz", "prostate"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--pseudo-alpha", type=float, default=0.75)
    parser.add_argument("--max-cases-per-client", type=int, default=0)
    parser.add_argument("--checkpoint-step-interval", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--metric",
        choices=["fg_acc", "bg_expansion"],
        default="bg_expansion",
    )
    parser.add_argument(
        "--pseudo-modes",
        nargs="+",
        default=["backbone", "blended"],
        choices=["backbone", "blended"],
    )
    return parser.parse_args()


def load_case_arrays(case_path: str, sup_type: str):
    with h5py.File(case_path, "r") as f:
        image = f["image"][:].astype(np.float32)
        mask = f["mask"][:].astype(np.int64)
        weak = f[sup_type][:].astype(np.int64)
    return image, mask, weak


def load_train_cases(data_root: str, domain: str, max_cases: int):
    train_dir = os.path.join(data_root, domain, "train")
    cases = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
    if max_cases > 0 and len(cases) > max_cases:
        idx = np.linspace(0, len(cases) - 1, num=max_cases, dtype=int)
        cases = [cases[i] for i in idx]
    return cases


def event_path_for_model_root(model_root: str):
    event_paths = sorted(glob.glob(os.path.join(model_root, "log", "events.out.tfevents.*")))
    if not event_paths:
        raise FileNotFoundError(f"No TensorBoard event file found under {model_root}/log")
    return event_paths[0]


def collect_checkpoint_paths(model_root: str, cid: int, step_interval: int):
    pattern = os.path.join(model_root, f"client_{cid}_iter_*_dice_*.pth")
    paths = sorted(glob.glob(pattern))
    records = []
    for path in paths:
        match = re.search(rf"client_{cid}_iter_(\d+)_dice_", os.path.basename(path))
        if match:
            records.append((int(match.group(1)), path))

    dedup = {}
    for step, path in records:
        dedup[step] = path
    items = sorted(dedup.items())
    if not items:
        return items

    selected = {}
    for step, path in items:
        bucket = int(round(step / step_interval) * step_interval)
        prev = selected.get(bucket)
        if prev is None or abs(step - bucket) < abs(prev[0] - bucket):
            selected[bucket] = (step, path)
    selected[items[-1][0]] = items[-1]
    return sorted(set(selected.values()))


def build_model(cfg: DatasetConfig, spec: ClientSpec, device: str):
    if REPO_CODE_DIR not in sys.path:
        sys.path.insert(0, REPO_CODE_DIR)
    from networks.unet import UNet_UniV5

    model = UNet_UniV5(
        in_chns=cfg.in_chns,
        class_num=cfg.num_classes,
        prompt_type="universal",
        attention_type="dual",
        sup_type=spec.sup_type,
        use_label_prompt=1,
        client_num=len(cfg.client_specs),
        client_id=spec.cid,
        img_size=cfg.img_size,
    )
    model.eval()
    model.to(device)
    return model


def load_val_dice_summary(cfg: DatasetConfig):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(event_path_for_model_root(cfg.model_root), size_guidance={"scalars": 0})
    ea.Reload()
    rows = []
    for cid in sorted(cfg.client_specs):
        vals = ea.Scalars(f"info_client_{cid}/val_mean_dice")
        best = max(vals, key=lambda x: x.value)
        rows.append({"cid": cid, "best_val_dice": best.value, "best_val_step": best.step})
    return pd.DataFrame(rows)


def image_to_tensor(cfg: DatasetConfig, image: np.ndarray, device: str):
    if cfg.in_chns == 1:
        return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    return torch.from_numpy(image).unsqueeze(0).to(device)


def reconstruct_pseudo(probs_main: np.ndarray, probs_aux: np.ndarray, pseudo_mode: str, pseudo_alpha: float):
    if pseudo_mode == "backbone":
        return np.argmax(probs_main, axis=0)
    blend = pseudo_alpha * probs_main + (1.0 - pseudo_alpha) * probs_aux
    return np.argmax(blend, axis=0)


def compute_curve_for_client(
    cfg: DatasetConfig,
    spec: ClientSpec,
    alpha: float,
    pseudo_alpha: float,
    max_cases: int,
    step_interval: int,
    device: str,
    metric: str,
    pseudo_mode: str,
):
    cases = load_train_cases(cfg.data_root, spec.domain, max_cases=max_cases)
    model = build_model(cfg, spec, device=device)
    checkpoint_steps = collect_checkpoint_paths(cfg.model_root, spec.cid, step_interval=step_interval)
    rows = []

    for step, ckpt_path in checkpoint_steps:
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

        total_metric_pixels = 0
        total_metric_hits = 0

        with torch.no_grad():
            for case_path in cases:
                image, mask, weak = load_case_arrays(case_path, spec.sup_type)
                unlabeled = weak == cfg.num_classes
                gt_fg = mask > 0
                gt_bg = mask == 0
                if metric == "fg_acc":
                    metric_region = unlabeled & gt_fg
                else:
                    metric_region = unlabeled & gt_bg
                metric_region_count = int(metric_region.sum())
                if metric_region_count == 0:
                    continue

                x = image_to_tensor(cfg, image, device)
                outputs, _, _, _, _, _, _, _, outputs_aux, _, _ = model(x)
                probs_main = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                probs_aux = torch.softmax(outputs_aux, dim=1)[0].cpu().numpy()
                pseudo = reconstruct_pseudo(probs_main, probs_aux, pseudo_mode=pseudo_mode, pseudo_alpha=pseudo_alpha)

                if metric == "fg_acc":
                    total_metric_hits += int((pseudo[metric_region] == mask[metric_region]).sum())
                else:
                    total_metric_hits += int((pseudo[metric_region] > 0).sum())
                total_metric_pixels += metric_region_count

        metric_value = np.nan if total_metric_pixels == 0 else total_metric_hits / total_metric_pixels
        rows.append(
            {
                "dataset": cfg.name,
                "cid": spec.cid,
                "client_name": spec.client_name,
                "domain": spec.domain,
                "sup_type": spec.sup_type,
                "step": step,
                "metric": metric,
                "pseudo_mode": pseudo_mode,
                "metric_pixels": total_metric_pixels,
                "metric_value": metric_value,
                "checkpoint": ckpt_path,
            }
        )
        print(f"{cfg.name} | {spec.label} | {pseudo_mode}: step={step}, {metric}={metric_value:.6f}", flush=True)

    return pd.DataFrame(rows)


def metric_label(metric: str):
    return {
        "fg_acc": "Foreground-only pseudo-label accuracy",
        "bg_expansion": "Background expansion error rate",
    }[metric]


def metric_title_suffix(metric: str):
    return {
        "fg_acc": "foreground-only pseudo-label accuracy",
        "bg_expansion": "background expansion error",
    }[metric]


def pseudo_mode_label(pseudo_mode: str):
    return {
        "backbone": "Backbone-only",
        "blended": "Blended",
    }[pseudo_mode]


def aggregate_dataset_curves(curve_df: pd.DataFrame):
    return (
        curve_df.groupby(["dataset", "step", "pseudo_mode"], as_index=False)
        .agg(metric_value=("metric_value", "mean"))
        .sort_values(["pseudo_mode", "step"])
    )


def plot_dataset_curves(curve_df: pd.DataFrame, cfg: DatasetConfig, output_png: str, metric: str):
    agg_df = aggregate_dataset_curves(curve_df)
    plt.figure(figsize=(6.1, 4.5))
    for pseudo_mode, sub_df in agg_df.groupby("pseudo_mode"):
        plt.plot(
            sub_df["step"],
            sub_df["metric_value"],
            marker="o",
            markersize=3,
            linewidth=2,
            label=pseudo_mode_label(pseudo_mode),
        )
    plt.xlabel("Training iteration")
    plt.ylabel(metric_label(metric))
    plt.title(f"{cfg.title}: backbone-only vs blended")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_combined(dataset_frames: Dict[str, pd.DataFrame], output_png: str, metric: str):
    ordered = [name for name in ["odoc", "faz", "prostate"] if name in dataset_frames]
    if not ordered:
        return
    fig, axes = plt.subplots(1, len(ordered), figsize=(5.5 * len(ordered), 4.6), sharey=True)
    if len(ordered) == 1:
        axes = [axes]
    for ax, dataset_name in zip(axes, ordered):
        df = aggregate_dataset_curves(dataset_frames[dataset_name])
        cfg = DATASETS[dataset_name]
        for pseudo_mode, sub_df in df.groupby("pseudo_mode"):
            ax.plot(
                sub_df["step"],
                sub_df["metric_value"],
                marker="o",
                markersize=2.8,
                linewidth=2,
                label=pseudo_mode_label(pseudo_mode),
            )
        ax.set_title(cfg.title)
        ax.set_xlabel("Training iteration")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(metric_label(metric))
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def build_client_delta_table(curve_df: pd.DataFrame):
    pivot = (
        curve_df.groupby(["dataset", "cid", "client_name", "domain", "sup_type", "pseudo_mode"], as_index=False)
        .agg(mean_metric_value=("metric_value", "mean"))
        .pivot_table(
            index=["dataset", "cid", "client_name", "domain", "sup_type"],
            columns="pseudo_mode",
            values="mean_metric_value",
        )
        .reset_index()
    )
    pivot.columns.name = None
    if "backbone" in pivot.columns and "blended" in pivot.columns:
        pivot["delta_blended_minus_backbone"] = pivot["blended"] - pivot["backbone"]
    return pivot


def plot_client_delta_bars(delta_df: pd.DataFrame, cfg: DatasetConfig, output_png: str):
    plt.figure(figsize=(7.2, 4.5))
    labels = [f"c{int(cid)}-{sup}" for cid, sup in zip(delta_df["cid"], delta_df["sup_type"])]
    values = delta_df["delta_blended_minus_backbone"]
    colors = ["#d62728" if x > 0 else "#2ca02c" for x in values]
    plt.bar(labels, values, color=colors)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.ylabel("Expansion error delta\n(blended - backbone-only)")
    plt.title(f"{cfg.title}: per-client expansion delta")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    summary_rows: List[dict] = []
    dataset_frames: Dict[str, pd.DataFrame] = {}

    for dataset_name in args.datasets:
        cfg = DATASETS[dataset_name]
        print(f"[start] dataset={dataset_name}", flush=True)
        curve_df = pd.concat(
            [
                compute_curve_for_client(
                    cfg,
                    spec,
                    alpha=args.alpha,
                    pseudo_alpha=args.pseudo_alpha,
                    max_cases=args.max_cases_per_client,
                    step_interval=args.checkpoint_step_interval,
                    device=device,
                    metric=args.metric,
                    pseudo_mode=pseudo_mode,
                )
                for spec in cfg.client_specs.values()
                for pseudo_mode in args.pseudo_modes
            ],
            ignore_index=True,
        )
        dataset_frames[dataset_name] = curve_df

        metric_stem = "fg_pseudo_acc" if args.metric == "fg_acc" else "bg_expansion_error"
        csv_path = os.path.join(args.output_dir, f"{dataset_name}_clean_backbone_vs_blended_{metric_stem}_curves.csv")
        png_path = os.path.join(args.output_dir, f"{dataset_name}_clean_backbone_vs_blended_{metric_stem}_curves.png")
        meta_path = os.path.join(args.output_dir, f"{dataset_name}_clean_backbone_vs_blended_{metric_stem}_meta.txt")
        delta_csv_path = os.path.join(args.output_dir, f"{dataset_name}_clean_backbone_vs_blended_{metric_stem}_client_delta.csv")
        delta_png_path = os.path.join(args.output_dir, f"{dataset_name}_clean_backbone_vs_blended_{metric_stem}_client_delta.png")

        curve_df.to_csv(csv_path, index=False)
        plot_dataset_curves(curve_df, cfg, png_path, args.metric)
        delta_df = build_client_delta_table(curve_df)
        delta_df.to_csv(delta_csv_path, index=False)
        plot_client_delta_bars(delta_df, cfg, delta_png_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write("All clients are included.\n")
            f.write("Comparison is within the same client and same checkpoint.\n")
            f.write(f"\nMetric: {metric_label(args.metric)}.\n")
            if args.metric == "fg_acc":
                f.write("Foreground-only = only GT foreground pixels inside the unlabeled weak region are counted.\n")
            else:
                f.write("Background expansion = within GT background pixels inside the unlabeled weak region, count the fraction predicted as foreground.\n")
            f.write("Backbone-only = argmax(main decoder softmax).\n")
            f.write(
                f"Blended = argmax({args.pseudo_alpha:.2f} * main + {1.0 - args.pseudo_alpha:.2f} * auxiliary), "
                "matching the clean-baseline dual-branch pseudo-label rule.\n"
            )
            f.write(
                f"Max sampled train cases per client={args.max_cases_per_client}, "
                f"checkpoint interval={args.checkpoint_step_interval}.\n"
            )

        summary_rows.append(
            {
                "dataset": dataset_name,
                "metric": args.metric,
                "pseudo_modes": ",".join(args.pseudo_modes),
                "csv_path": csv_path,
                "png_path": png_path,
                "delta_csv_path": delta_csv_path,
                "delta_png_path": delta_png_path,
            }
        )
        print(f"[done] dataset={dataset_name} csv={csv_path} png={png_path} delta_png={delta_png_path}", flush=True)
        summary_df_partial = pd.DataFrame(summary_rows)
        summary_stem_partial = (
            "pilot_fg_pseudo_acc_backbone_vs_blended"
            if args.metric == "fg_acc"
            else "pilot_bg_expansion_error_backbone_vs_blended"
        )
        summary_df_partial.to_csv(
            os.path.join(args.output_dir, f"{summary_stem_partial}_summary.csv"),
            index=False,
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_stem = (
        "pilot_fg_pseudo_acc_backbone_vs_blended"
        if args.metric == "fg_acc"
        else "pilot_bg_expansion_error_backbone_vs_blended"
    )
    summary_df.to_csv(os.path.join(args.output_dir, f"{summary_stem}_summary.csv"), index=False)
    plot_combined(dataset_frames, os.path.join(args.output_dir, f"{summary_stem}_triptych.png"), args.metric)

    print(summary_df.to_string(index=False))
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
