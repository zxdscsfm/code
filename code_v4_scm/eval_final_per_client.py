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


DATASETS = {
    "prostate": {
        "root_path": "/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd",
        "num_clients": 6,
        "num_classes": 2,
        "in_chns": 1,
        "img_size": 384,
        "sup_types": ["scribble", "keypoint", "scribble", "block", "scribble", "scribble"],
    },
    "polyp": {
        "root_path": "/data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd",
        "num_clients": 4,
        "num_classes": 2,
        "in_chns": 3,
        "img_size": 384,
        "sup_types": ["scribble", "keypoint", "scribble", "block"],
    },
    "faz": {
        "root_path": "/data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd",
        "num_clients": 5,
        "num_classes": 2,
        "in_chns": 1,
        "img_size": 256,
        "sup_types": ["scribble", "keypoint", "scribble", "block", "scribble"],
    },
}


RUNS = {
    "ted": {
        "method": "RDSI-TED",
        "model_root": "/data/jianbingshen/yanghongji/FedLPPA_Original/model",
        "experiments": {
            "prostate": "prostate/FedLPPA_rdsited_prostate_r500_l10_20260527_134611_seed2022",
            "polyp": "polyp/FedLPPA_rdsited_polyp_r500_l10_20260527_134611_seed2022",
            "faz": "faz/FedLPPA_rdsited_faz_r500_l10_20260527_190324_seed2022",
        },
    },
    "sota": {
        "method": "FedLPPA-SOTA",
        "model_root": "/data/jianbingshen/yanghongji/fedlppaSOTA/model",
        "experiments": {
            "prostate": "prostate/FedLPPA_sota_fedlppa_rdsi3_sd_prostate_r500_l10_20260528_232843_seed2022",
            "polyp": "polyp/FedLPPA_sota_fedlppa_rdsi3_sd_polyp_r500_l10_20260522_230837_seed2022",
            "faz": "faz/FedLPPA_sota_fedlppa_rdsi3_sd_faz_r500_l10_20260523_232614_seed2022",
        },
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


def client_cases(root_path, cid):
    test_dir = os.path.join(root_path, f"Domain{cid + 1}", "test")
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


def build_model(dataset_name, cid, sup_type, device):
    cfg = DATASETS[dataset_name]
    args = SimpleNamespace(
        min_num_clients=cfg["num_clients"],
        cid=cid,
        prompt="universal",
        attention="dual",
        dual_init="aggregated",
        label_prompt=1,
        sup_type=sup_type,
        img_size=cfg["img_size"],
        use_cuda=1 if device.type == "cuda" else 0,
        device=str(device),
    )
    return net_factory(args, net_type="unet_univ5", in_chns=cfg["in_chns"], class_num=cfg["num_classes"])


def evaluate_client(method_key, dataset_name, cid, checkpoint_path, device):
    cfg = DATASETS[dataset_name]
    sup_type = cfg["sup_types"][cid]
    model = build_model(dataset_name, cid, sup_type, device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    case_rows = []
    for rel_case in client_cases(cfg["root_path"], cid):
        pred, label = infer_case(os.path.join(cfg["root_path"], rel_case), model, device)
        metrics = calculate_metric_percase(pred == 1, label == 1)
        case_rows.append(
            {
                "method": RUNS[method_key]["method"],
                "dataset": dataset_name,
                "client": f"client{cid}",
                "case": rel_case,
                "sup_type": sup_type,
                "checkpoint": checkpoint_path,
                **metrics,
            }
        )
    return case_rows


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

    run = RUNS[args.method]
    all_case_rows = []
    all_client_rows = []
    for dataset_name, exp in run["experiments"].items():
        cfg = DATASETS[dataset_name]
        exp_dir = os.path.join(run["model_root"], exp)
        if not os.path.isdir(exp_dir):
            raise RuntimeError(f"Missing experiment directory: {exp_dir}")
        for cid in range(cfg["num_clients"]):
            checkpoint_path = os.path.join(exp_dir, f"client_{cid}_async_unet_univ5_best_model.pth")
            if not os.path.exists(checkpoint_path):
                checkpoint_path = os.path.join(exp_dir, f"client_{cid}_unet_univ5_best_model.pth")
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Missing checkpoint for {run['method']} {dataset_name} client{cid}: {checkpoint_path}")

            print(f"Evaluating {run['method']} {dataset_name} client{cid}: {checkpoint_path}", flush=True)
            case_rows = evaluate_client(args.method, dataset_name, cid, checkpoint_path, device)
            all_case_rows.extend(case_rows)
            df = pd.DataFrame(case_rows)
            means = df[["dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]].mean()
            stds = df[["dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]].std(ddof=0)
            all_client_rows.append(
                {
                    "method": run["method"],
                    "dataset": dataset_name,
                    "client": f"client{cid}",
                    "num_cases": len(case_rows),
                    "sup_type": cfg["sup_types"][cid],
                    "checkpoint": checkpoint_path,
                    **{name: float(means[name]) for name in means.index},
                    **{f"{name}_std": float(stds[name]) for name in stds.index},
                }
            )

    case_df = pd.DataFrame(all_case_rows)
    client_df = pd.DataFrame(all_client_rows)
    avg_df = (
        client_df.groupby(["method", "dataset"], as_index=False)[
            ["dice", "hd95", "recall", "precision", "jc", "specificity", "ravd"]
        ]
        .mean()
        .assign(client="mean", num_cases="")
    )
    out_case = os.path.join(args.out_dir, f"{args.method}_per_case_metrics.csv")
    out_client = os.path.join(args.out_dir, f"{args.method}_per_client_metrics.csv")
    out_avg = os.path.join(args.out_dir, f"{args.method}_dataset_mean_metrics.csv")
    case_df.to_csv(out_case, index=False)
    client_df.to_csv(out_client, index=False)
    avg_df.to_csv(out_avg, index=False)
    print(f"case_csv={out_case}")
    print(f"client_csv={out_client}")
    print(f"avg_csv={out_avg}")
    print(client_df[["method", "dataset", "client", "dice", "hd95", "precision", "jc", "specificity", "ravd"]].to_string(index=False))


if __name__ == "__main__":
    main()
