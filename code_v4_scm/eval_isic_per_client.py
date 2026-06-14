import argparse
import os
import random
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import torch
from medpy import metric

from networks.net_factory import net_factory


ROOT_PATH = "/data/jianbingshen/yanghongji/FedLPPA_Original/data/ISIC_h5_rdsi3_sd"
NUM_CLIENTS = 4
SUP_TYPES = ["scribble", "keypoint", "scribble", "block"]

RUNS = {
    "ted": {
        "method": "RDSI-TED-gradfix",
        "exp_dir": "/data/jianbingshen/yanghongji/FedLPPA_Original/model/isic/FedLPPA_isic_weak_ted_gradfix_full_r500_l10_20260531_104926_seed2022",
    },
    "sota": {
        "method": "FedLPPA-SOTA-clean-head",
        "exp_dir": "/data/jianbingshen/yanghongji/fedlppaSOTA_clean_head_isic_20260531_001/model/isic/FedLPPA_sota_clean_head_isic_weak_full_r500_l10_20260530_233748_seed2022",
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def client_cases(cid):
    test_dir = os.path.join(ROOT_PATH, f"Domain{cid + 1}", "test")
    return [os.path.join(f"Domain{cid + 1}", "test", name) for name in sorted(os.listdir(test_dir))]


def calculate_metric_percase(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    if pred.sum() > 0:
        if gt.sum() > 0:
            return {
                "dice": float(metric.binary.dc(pred, gt)),
                "hd95": float(metric.binary.hd95(pred, gt)),
                "recall": float(metric.binary.recall(pred, gt)),
                "precision": float(metric.binary.precision(pred, gt)),
                "jc": float(metric.binary.jc(pred, gt)),
                "specificity": float(metric.binary.specificity(pred, gt)),
                "ravd": float(metric.binary.ravd(pred, gt)),
            }
        return {
            "dice": 0.0,
            "hd95": 192.0,
            "recall": 0.0,
            "precision": 0.0,
            "jc": 0.0,
            "specificity": 0.0,
            "ravd": 1.0,
        }
    return {
        "dice": 0.0,
        "hd95": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "jc": 0.0,
        "specificity": 0.0,
        "ravd": 0.0,
    }


def infer_case(case_path, model, device):
    with h5py.File(case_path, "r") as handle:
        image = handle["image"][:]
        label = handle["mask"][:]

    if len(image.shape) == 3:
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(device)
    elif len(image.shape) == 2:
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device)
    else:
        raise RuntimeError(f"Unsupported image shape {image.shape} for {case_path}")

    model.eval()
    with torch.no_grad():
        output = model(tensor)[0]
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()
    return pred, label


def build_model(cid, device):
    args = SimpleNamespace(
        min_num_clients=NUM_CLIENTS,
        cid=cid,
        prompt="universal",
        attention="dual",
        dual_init="aggregated",
        label_prompt=1,
        sup_type=SUP_TYPES[cid],
        img_size=384,
        use_cuda=1 if device.type == "cuda" else 0,
        device=str(device),
    )
    return net_factory(args, net_type="unet_univ5", in_chns=3, class_num=2)


def checkpoint_for(exp_dir, cid):
    candidates = [
        os.path.join(exp_dir, f"client_{cid}_async_unet_univ5_best_model.pth"),
        os.path.join(exp_dir, f"client_{cid}_unet_univ5_best_model.pth"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Missing checkpoint for client{cid}: {candidates}")


def evaluate_client(method_key, cid, device):
    run = RUNS[method_key]
    checkpoint_path = checkpoint_for(run["exp_dir"], cid)
    model = build_model(cid, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    rows = []
    for rel_case in client_cases(cid):
        pred, label = infer_case(os.path.join(ROOT_PATH, rel_case), model, device)
        rows.append(
            {
                "method": run["method"],
                "dataset": "isic",
                "client": f"client{cid}",
                "case": rel_case,
                "sup_type": SUP_TYPES[cid],
                "checkpoint": checkpoint_path,
                **calculate_metric_percase(pred == 1, label == 1),
            }
        )
    return rows


def summarize_client(rows):
    df = pd.DataFrame(rows)
    metric_names = ["dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]
    means = df[metric_names].mean()
    stds = df[metric_names].std(ddof=0)
    first = rows[0]
    return {
        "method": first["method"],
        "dataset": "isic",
        "client": first["client"],
        "num_cases": len(rows),
        "sup_type": first["sup_type"],
        "checkpoint": first["checkpoint"],
        **{name: float(means[name]) for name in metric_names},
        **{f"{name}_std": float(stds[name]) for name in metric_names},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(RUNS), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--use-cuda", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda == 1 and torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    all_case_rows = []
    client_rows = []
    for cid in range(NUM_CLIENTS):
        print(f"Evaluating {RUNS[args.method]['method']} client{cid}", flush=True)
        rows = evaluate_client(args.method, cid, device)
        all_case_rows.extend(rows)
        client_rows.append(summarize_client(rows))

    case_df = pd.DataFrame(all_case_rows)
    client_df = pd.DataFrame(client_rows)
    metric_names = ["dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]
    avg_df = (
        client_df.groupby(["method", "dataset"], as_index=False)[metric_names]
        .mean()
        .assign(client="mean", num_cases="")
    )

    out_case = os.path.join(args.out_dir, f"{args.method}_isic_per_case_metrics.csv")
    out_client = os.path.join(args.out_dir, f"{args.method}_isic_per_client_metrics.csv")
    out_avg = os.path.join(args.out_dir, f"{args.method}_isic_dataset_mean_metrics.csv")
    case_df.to_csv(out_case, index=False)
    client_df.to_csv(out_client, index=False)
    avg_df.to_csv(out_avg, index=False)

    print(f"case_csv={out_case}")
    print(f"client_csv={out_client}")
    print(f"avg_csv={out_avg}")
    print(client_df[["method", "client", "num_cases", "sup_type", "dice", "hd95", "precision", "jc", "specificity", "ravd"]].to_string(index=False))
    print(avg_df[["method", "client", "dice", "hd95", "precision", "jc", "specificity", "ravd"]].to_string(index=False))


if __name__ == "__main__":
    main()
