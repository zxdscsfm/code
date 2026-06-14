import argparse
import os

import pandas as pd


METRICS = ["dice", "hd95", "precision", "jc", "specificity", "ravd"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    frames = []
    for name in ["ted", "sota"]:
        path = os.path.join(args.out_dir, f"{name}_per_client_metrics.csv")
        if not os.path.exists(path):
            raise RuntimeError(f"Missing input CSV: {path}")
        frames.append(pd.read_csv(path))
    per_client = pd.concat(frames, ignore_index=True)

    dataset_mean = (
        per_client.groupby(["method", "dataset"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "method"])
    )

    wide = per_client.pivot_table(index=["dataset", "client"], columns="method", values=METRICS)
    wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
    wide = wide.reset_index()
    if "dice_RDSI-TED" in wide.columns and "dice_FedLPPA-SOTA" in wide.columns:
        wide["delta_dice_TED_minus_SOTA"] = wide["dice_RDSI-TED"] - wide["dice_FedLPPA-SOTA"]
    if "hd95_RDSI-TED" in wide.columns and "hd95_FedLPPA-SOTA" in wide.columns:
        wide["delta_hd95_TED_minus_SOTA"] = wide["hd95_RDSI-TED"] - wide["hd95_FedLPPA-SOTA"]
    if "ravd_RDSI-TED" in wide.columns and "ravd_FedLPPA-SOTA" in wide.columns:
        wide["delta_abs_ravd_TED_minus_SOTA"] = wide["ravd_RDSI-TED"].abs() - wide["ravd_FedLPPA-SOTA"].abs()

    out_csv = os.path.join(args.out_dir, "final_per_client_metrics.csv")
    out_mean = os.path.join(args.out_dir, "final_dataset_mean_metrics.csv")
    out_wide = os.path.join(args.out_dir, "final_per_client_comparison_wide.csv")
    out_xlsx = os.path.join(args.out_dir, "final_per_client_metrics.xlsx")
    per_client.to_csv(out_csv, index=False)
    dataset_mean.to_csv(out_mean, index=False)
    wide.to_csv(out_wide, index=False)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        per_client.to_excel(writer, index=False, sheet_name="per_client")
        dataset_mean.to_excel(writer, index=False, sheet_name="dataset_mean")
        wide.to_excel(writer, index=False, sheet_name="wide_compare")

    print(f"final_csv={out_csv}")
    print(f"mean_csv={out_mean}")
    print(f"wide_csv={out_wide}")
    print(f"excel={out_xlsx}")
    print("DATASET MEAN")
    print(dataset_mean.to_string(index=False))
    print("WIDE CLIENT COMPARISON")
    cols = ["dataset", "client"]
    for col in [
        "dice_RDSI-TED",
        "dice_FedLPPA-SOTA",
        "delta_dice_TED_minus_SOTA",
        "hd95_RDSI-TED",
        "hd95_FedLPPA-SOTA",
        "ravd_RDSI-TED",
        "ravd_FedLPPA-SOTA",
    ]:
        if col in wide.columns:
            cols.append(col)
    print(wide[cols].to_string(index=False))


if __name__ == "__main__":
    main()
