import argparse
import copy
import inspect
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataloaders.dataset import BaseDataSets
from flower_common_v4 import VAL_METRICS, evaluate
from networks.net_factory import net_factory


DEFAULT_SITE_LABELS = ["Site A", "Site B", "Site C", "Site D", "Site E"]
DEFAULT_CLIENTS = ["client1", "client2", "client3", "client4", "client5"]
DEFAULT_SUP_TYPES = ["scribble_noisy", "keypoint", "block", "box", "scribble"]
TABLE_METRICS = ["dice", "hd95"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate personalized FedLPPA checkpoints on site-specific test sets."
    )
    parser.add_argument("--root_path", type=str, required=True, help="Dataset root, e.g. ../data/FAZ_h5")
    parser.add_argument("--exp", type=str, default="faz/FedLPPA", help="Experiment name under ../model")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory containing saved checkpoints")
    parser.add_argument(
        "--checkpoint_kind",
        type=str,
        default="final",
        choices=["final", "best", "async_best"],
        help="Checkpoint naming convention to use when --checkpoint_paths is not provided",
    )
    parser.add_argument(
        "--checkpoint_paths",
        nargs="+",
        default=None,
        help="Optional explicit checkpoint paths, one per site, in site order",
    )
    parser.add_argument("--output_dir", type=str, default="", help="Directory to save CSV/Markdown summaries")
    parser.add_argument("--method_name", type=str, default="FedLPPA", help="Method label for the final table")
    parser.add_argument("--model", type=str, default="unet_univ5", help="Model name")
    parser.add_argument("--img_class", type=str, default="faz", help="Dataset type")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--in_chns", type=int, default=1, help="Input image channels")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--min_num_clients", type=int, default=5, help="Number of clients")
    parser.add_argument("--prompt", type=str, default="universal", help="Prompt type")
    parser.add_argument("--attention", type=str, default="dual", help="Attention type")
    parser.add_argument("--dual_init", type=str, default="aggregated", help="Dual branch init")
    parser.add_argument("--label_prompt", type=int, default=1, help="Use label prompt or not")
    parser.add_argument("--asp_mode", type=str, default="hybrid", help="ASP mode for stage-2 quality model")
    parser.add_argument("--annotation_agnostic", type=int, default=0, help="Enable metadata-free weak supervision setting")
    parser.add_argument("--quality_static_dim", type=int, default=7, help="Static quality feature dimension")
    parser.add_argument("--quality_dynamic_dim", type=int, default=5, help="Dynamic quality feature dimension")
    parser.add_argument("--quality_dim", type=int, default=16, help="Quality token dimension")
    parser.add_argument("--quality_hidden_dim", type=int, default=32, help="Hidden dimension for quality modules")
    parser.add_argument("--quality_prompt_channels", type=int, default=2, help="Quality prompt channels")
    parser.add_argument("--reliability_gate", type=int, default=0, help="Enable reliability gate")
    parser.add_argument("--reliability_gate_hidden", type=int, default=32, help="Reliability gate hidden channels")
    parser.add_argument("--geometry_guided", type=int, default=0, help="Enable rule-bin annotation geometry guidance")
    parser.add_argument("--geometry_num_bins", type=int, default=4, help="Number of geometry bins")
    parser.add_argument(
        "--geometry_pseudo_weights",
        type=str,
        default="1.0,0.8,0.5,0.2",
        help="Pseudo-label weights per geometry bin; kept for CLI compatibility with training runs",
    )
    parser.add_argument("--geometry_distance_clip", type=float, default=64.0, help="Clip radius for normalized geometry distances")
    parser.add_argument("--geometry_density_kernel", type=int, default=15, help="Kernel size for local labeled density")
    parser.add_argument("--geometry_near_radius", type=float, default=8.0, help="Near-radius threshold for rule-based geometry bins")
    parser.add_argument("--geometry_mid_radius", type=float, default=24.0, help="Mid-radius threshold for rule-based geometry bins")
    parser.add_argument("--geometry_route_lambda", type=float, default=0.5, help="Residual strength of routed geometry prototypes")
    parser.add_argument("--amp", type=int, default=0, help="Enable AMP evaluation")
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed")
    parser.add_argument("--site_labels", nargs="+", default=DEFAULT_SITE_LABELS, help="Human-readable site labels")
    parser.add_argument("--clients", nargs="+", default=DEFAULT_CLIENTS, help="Client names")
    parser.add_argument("--sup_type_list", nargs="+", default=DEFAULT_SUP_TYPES, help="Supervision type per site")
    args = parser.parse_args()

    expected_len = args.min_num_clients
    for field_name in ["site_labels", "clients", "sup_type_list"]:
        values = getattr(args, field_name)
        if len(values) != expected_len:
            raise ValueError(f"{field_name} must have {expected_len} entries, got {len(values)}")

    if args.checkpoint_paths is not None and len(args.checkpoint_paths) != expected_len:
        raise ValueError(
            f"checkpoint_paths must have {expected_len} entries, got {len(args.checkpoint_paths)}"
        )

    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.normpath(os.path.join("..", "model", args.exp))

    if not args.output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.checkpoint_dir, f"test_summary_{timestamp}")

    if args.annotation_agnostic == 1 and args.model != "unet_univ6":
        raise ValueError("annotation_agnostic=1 is only implemented for model=unet_univ6")
    if args.geometry_guided == 1 and args.model not in ["unet_univ5", "unet_univ6"]:
        raise ValueError("geometry_guided=1 is only implemented for model in ['unet_univ5', 'unet_univ6']")

    args.patch_size = [args.img_size, args.img_size]
    return args


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def resolve_checkpoint_path(args, site_index):
    if args.checkpoint_paths is not None:
        return args.checkpoint_paths[site_index]

    if args.checkpoint_kind == "final":
        filename = f"client_{site_index}_{args.model}_final_model.pth"
    elif args.checkpoint_kind == "best":
        filename = f"client_{site_index}_{args.model}_best_model.pth"
    else:
        filename = f"client_{site_index}_async_{args.model}_best_model.pth"

    return os.path.join(args.checkpoint_dir, filename)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def build_site_args(args, site_index):
    site_args = copy.deepcopy(args)
    site_args.cid = site_index
    site_args.client = args.clients[site_index]
    site_args.sup_type = args.sup_type_list[site_index]
    site_args.role = "client"
    site_args.strategy = "FedUniV2.1"
    site_args.snapshot_path = args.checkpoint_dir
    return site_args


def evaluate_site(args, site_index):
    site_args = build_site_args(args, site_index)
    checkpoint_path = resolve_checkpoint_path(args, site_index)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_kwargs = dict(
        base_dir=args.root_path,
        split="val",
        transform=None,
        client=site_args.client,
        sup_type=site_args.sup_type,
        img_class=args.img_class,
        annotation_agnostic=bool(getattr(args, "annotation_agnostic", 0)),
        geometry_guided=bool(getattr(args, "geometry_guided", 0)),
        geometry_distance_clip=getattr(args, "geometry_distance_clip", 64.0),
        geometry_density_kernel=getattr(args, "geometry_density_kernel", 15),
        geometry_near_radius=getattr(args, "geometry_near_radius", 8.0),
        geometry_mid_radius=getattr(args, "geometry_mid_radius", 24.0),
    )
    dataset_signature = inspect.signature(BaseDataSets.__init__)
    filtered_dataset_kwargs = {
        key: value for key, value in dataset_kwargs.items() if key in dataset_signature.parameters
    }
    dataset = BaseDataSets(**filtered_dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = net_factory(site_args, net_type=args.model, in_chns=args.in_chns, class_num=args.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    metrics = evaluate(site_args, model, dataloader, amp=bool(args.amp))
    result = {
        "site": args.site_labels[site_index],
        "client": site_args.client,
        "cid": site_index,
        "sup_type": site_args.sup_type,
        "num_cases": len(dataset),
        "checkpoint_path": checkpoint_path,
    }
    for class_idx in range(1, args.num_classes):
        for metric_name in VAL_METRICS:
            result[f"class_{class_idx}_{metric_name}"] = float(
                metrics[f"val_{class_idx}_{metric_name}"]
            )
    for metric_name in VAL_METRICS:
        result[metric_name] = float(metrics[f"val_mean_{metric_name}"])
    return result


def build_site_dataframe(site_results):
    columns = [
        "site",
        "client",
        "cid",
        "sup_type",
        "num_cases",
        "dice",
        "hd95",
        "recall",
        "precision",
        "jc",
        "specificity",
        "ravd",
        "checkpoint_path",
    ]
    classwise_columns = []
    for key in site_results[0].keys():
        if key.startswith("class_"):
            classwise_columns.append(key)
    columns.extend(sorted(classwise_columns))
    return pd.DataFrame(site_results, columns=columns)


def get_class_aliases(args):
    if args.img_class == "odoc" and args.num_classes == 3:
        return {
            1: ("OC", "cup"),
            2: ("OD", "disc"),
        }
    return {}


def build_aggregate_rows(df):
    metric_columns = list(VAL_METRICS)
    classwise_columns = [col for col in df.columns if col.startswith("class_")]
    macro_row = {
        "site": "Macro Avg",
        "client": "-",
        "cid": -1,
        "sup_type": "-",
        "num_cases": int(df["num_cases"].sum()),
        "checkpoint_path": "-",
    }
    weighted_row = {
        "site": "Weighted Avg",
        "client": "-",
        "cid": -2,
        "sup_type": "-",
        "num_cases": int(df["num_cases"].sum()),
        "checkpoint_path": "-",
    }

    for metric_name in metric_columns:
        macro_row[metric_name] = float(df[metric_name].mean())
        weighted_row[metric_name] = float(np.average(df[metric_name], weights=df["num_cases"]))

    for metric_name in classwise_columns:
        macro_row[metric_name] = float(df[metric_name].mean())
        weighted_row[metric_name] = float(np.average(df[metric_name], weights=df["num_cases"]))

    return pd.DataFrame([macro_row, weighted_row])


def build_table_iii_like(df, method_name):
    macro_row = df[df["site"] == "Macro Avg"].iloc[0]
    site_rows = df[df["cid"] >= 0]
    row = {"Method": method_name}

    for metric_name in TABLE_METRICS:
        metric_label = metric_name.upper() if metric_name != "dice" else "DSC"
        for _, site_row in site_rows.iterrows():
            row[f"{site_row['site']} {metric_label}"] = float(site_row[metric_name])
        row[f"Avg {metric_label}"] = float(macro_row[metric_name])

    return pd.DataFrame([row])


def build_odoc_paper_table(df, method_name, class_aliases):
    macro_row = df[df["site"] == "Macro Avg"].iloc[0]
    site_rows = df[df["cid"] >= 0]
    row = {"Method": method_name}

    ordered_classes = [class_idx for class_idx in [2, 1] if class_idx in class_aliases]
    for class_idx in ordered_classes:
        short_name, _ = class_aliases[class_idx]
        for metric_name in TABLE_METRICS:
            metric_label = "DSC" if metric_name == "dice" else "HD95"
            for _, site_row in site_rows.iterrows():
                row[f"{site_row['site']} {metric_label}({short_name})"] = float(
                    site_row[f"class_{class_idx}_{metric_name}"]
                )
            row[f"Avg {metric_label}({short_name})"] = float(macro_row[f"class_{class_idx}_{metric_name}"])

    return pd.DataFrame([row])


def format_metric(value, precision=4):
    return f"{value:.{precision}f}"


def write_markdown(site_df, table_df, output_dir):
    summary_md = os.path.join(output_dir, "summary.md")
    metric_headers = ["Site", "Cases", "Dice", "HD95", "Recall", "Precision", "JC", "Specificity", "RAVD"]
    lines = ["# Personalized FedLPPA Test Summary", ""]
    lines.append("| " + " | ".join(metric_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(metric_headers)) + "|")

    for _, row in site_df.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                row["site"],
                int(row["num_cases"]),
                format_metric(row["dice"]),
                format_metric(row["hd95"]),
                format_metric(row["recall"]),
                format_metric(row["precision"]),
                format_metric(row["jc"]),
                format_metric(row["specificity"]),
                format_metric(row["ravd"]),
            )
        )

    lines.extend(["", "## Table III-like Summary", ""])
    table_headers = list(table_df.columns)
    lines.append("| " + " | ".join(table_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(table_headers)) + "|")
    for _, row in table_df.iterrows():
        rendered = []
        for header in table_headers:
            value = row[header]
            rendered.append(
                format_metric(value) if isinstance(value, (float, np.floating)) else str(value)
            )
        lines.append("| " + " | ".join(rendered) + " |")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_odoc_markdown(classwise_df, odoc_table_df, output_dir):
    summary_md = os.path.join(output_dir, "summary_odoc.md")
    metric_headers = ["Site", "Cases", "DSC(OC)", "HD95(OC)", "DSC(OD)", "HD95(OD)"]
    lines = ["# Personalized FedLPPA ODOC Summary", ""]
    lines.append("| " + " | ".join(metric_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(metric_headers)) + "|")

    for _, row in classwise_df.iterrows():
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                row["site"],
                int(row["num_cases"]),
                format_metric(row["oc_dice"]),
                format_metric(row["oc_hd95"]),
                format_metric(row["od_dice"]),
                format_metric(row["od_hd95"]),
            )
        )

    lines.extend(["", "## Table II-like Summary", ""])
    table_headers = list(odoc_table_df.columns)
    lines.append("| " + " | ".join(table_headers) + " |")
    lines.append("|" + "|".join(["---"] * len(table_headers)) + "|")
    for _, row in odoc_table_df.iterrows():
        rendered = []
        for header in table_headers:
            value = row[header]
            rendered.append(
                format_metric(value) if isinstance(value, (float, np.floating)) else str(value)
            )
        lines.append("| " + " | ".join(rendered) + " |")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_classwise_dataframe(df, num_classes):
    rows = []
    site_rows = df[df["cid"] >= 0]
    aggregate_rows = df[df["cid"] < 0]
    for _, row in pd.concat([site_rows, aggregate_rows], ignore_index=True).iterrows():
        entry = {
            "site": row["site"],
            "num_cases": row["num_cases"],
        }
        for class_idx in range(1, num_classes):
            entry[f"class_{class_idx}_dice"] = row[f"class_{class_idx}_dice"]
            entry[f"class_{class_idx}_hd95"] = row[f"class_{class_idx}_hd95"]
        rows.append(entry)
    return pd.DataFrame(rows)


def add_class_alias_columns(classwise_df, class_aliases):
    aliased_df = classwise_df.copy()
    for class_idx, (short_name, long_name) in class_aliases.items():
        for metric_name in TABLE_METRICS:
            source_col = f"class_{class_idx}_{metric_name}"
            aliased_df[f"{short_name.lower()}_{metric_name}"] = aliased_df[source_col]
            aliased_df[f"{long_name.lower()}_{metric_name}"] = aliased_df[source_col]
    return aliased_df


def print_console_summary(site_df, table_df):
    print("\nPer-site metrics:")
    print(
        site_df[
            ["site", "num_cases", "dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )
    print("\nTable III-like summary:")
    print(table_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    set_deterministic(args.seed)

    site_results = []
    for site_index in range(args.min_num_clients):
        print(
            f"Evaluating {args.site_labels[site_index]} "
            f"({args.clients[site_index]}, sup_type={args.sup_type_list[site_index]})"
        )
        site_results.append(evaluate_site(args, site_index))

    site_df = build_site_dataframe(site_results)
    aggregate_df = build_aggregate_rows(site_df)
    final_site_df = pd.concat([site_df, aggregate_df], ignore_index=True)
    table_df = build_table_iii_like(final_site_df, args.method_name)
    classwise_df = build_classwise_dataframe(final_site_df, args.num_classes)
    class_aliases = get_class_aliases(args)
    odoc_table_df = None
    if class_aliases:
        classwise_df = add_class_alias_columns(classwise_df, class_aliases)
        odoc_table_df = build_odoc_paper_table(final_site_df, args.method_name, class_aliases)

    final_site_df.to_csv(os.path.join(args.output_dir, "site_metrics.csv"), index=False)
    table_df.to_csv(os.path.join(args.output_dir, "table_iii_like.csv"), index=False)
    classwise_df.to_csv(os.path.join(args.output_dir, "classwise_metrics.csv"), index=False)
    write_markdown(final_site_df, table_df, args.output_dir)
    if odoc_table_df is not None:
        odoc_table_df.to_csv(os.path.join(args.output_dir, "table_ii_odoc_like.csv"), index=False)
        write_odoc_markdown(classwise_df, odoc_table_df, args.output_dir)
    print_console_summary(final_site_df, table_df)

    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
