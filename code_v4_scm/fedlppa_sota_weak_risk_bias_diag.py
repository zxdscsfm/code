import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch


torch.nn.Module.cuda = lambda self, *args, **kwargs: self
torch.Tensor.cuda = lambda self, *args, **kwargs: self


from dataloaders.dataset import BaseDataSets  # noqa: E402
from networks.net_factory import net_factory  # noqa: E402


@dataclass
class DatasetConfig:
    dataset: str
    img_class: str
    root_path: str
    model_dir: str
    clients: List[str]
    sup_types: List[str]
    in_chns: int
    img_size: int
    min_num_clients: int


DATASETS = {
    "prostate": DatasetConfig(
        dataset="prostate",
        img_class="prostate",
        root_path="/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd",
        model_dir="/data/jianbingshen/yanghongji/fedlppaSOTA/model/prostate/FedLPPA_sota_diag_fedlppa_rdsi3_sd_prostate_r500_l10_20260608_144436_seed2022",
        clients=["client1", "client2", "client3", "client4", "client5", "client6"],
        sup_types=["scribble", "keypoint", "scribble", "block", "scribble", "scribble"],
        in_chns=1,
        img_size=384,
        min_num_clients=6,
    ),
    "prostate_wacggeocal": DatasetConfig(
        dataset="prostate_wacggeocal",
        img_class="prostate",
        root_path="/data/jianbingshen/yanghongji/FedLPPA_Original/data/PROSTATE_h5_rdsi3_sd",
        model_dir="/data/jianbingshen/yanghongji/FedLPPA_Original/model/prostate/FedLPPA_wannacggeocal_prostate_r500_l10_20260609_041125_seed2022",
        clients=["client1", "client2", "client3", "client4", "client5", "client6"],
        sup_types=["scribble", "keypoint", "scribble", "block", "scribble", "scribble"],
        in_chns=1,
        img_size=384,
        min_num_clients=6,
    ),
    "polyp": DatasetConfig(
        dataset="polyp",
        img_class="polyp",
        root_path="/data/jianbingshen/yanghongji/FedLPPA_Original/data/POLYP_h5_rdsi3_sd",
        model_dir="/data/jianbingshen/yanghongji/fedlppaSOTA/model/polyp/FedLPPA_sota_diag_fedlppa_rdsi3_sd_polyp_r500_l10_20260608_144436_seed2022",
        clients=["client1", "client2", "client3", "client4"],
        sup_types=["scribble", "keypoint", "scribble", "block"],
        in_chns=3,
        img_size=384,
        min_num_clients=4,
    ),
    "faz": DatasetConfig(
        dataset="faz",
        img_class="faz",
        root_path="/data/jianbingshen/yanghongji/FedLPPA_Original/data/FAZ_h5_rdsi3_sd",
        model_dir="/data/jianbingshen/yanghongji/fedlppaSOTA/model/faz/FedLPPA_sota_diag_fedlppa_rdsi3_sd_faz_r500_l10_20260608_185327_seed2022",
        clients=["client1", "client2", "client3", "client4", "client5"],
        sup_types=["scribble", "keypoint", "scribble", "block", "scribble"],
        in_chns=1,
        img_size=256,
        min_num_clients=5,
    ),
    "isic": DatasetConfig(
        dataset="isic",
        # The clean-head SOTA diagnostic run used the unchanged polyp RGB-binary path.
        img_class="polyp",
        root_path="/data/jianbingshen/yanghongji/FedLPPA_Original/data/ISIC_h5_rdsi3_sd",
        model_dir="/data/jianbingshen/yanghongji/fedlppaSOTA_clean_head_isic_20260531_001/model/isic/FedLPPA_sota_diag_clean_head_isic_polyp_path_weak_r500_l10_alacap500_20260609_025800_seed2022",
        clients=["client1", "client2", "client3", "client4"],
        sup_types=["scribble", "keypoint", "scribble", "block"],
        in_chns=3,
        img_size=384,
        min_num_clients=4,
    ),
}


class Args:
    pass


def build_model_args(cfg: DatasetConfig, cid: int, sup_type: str) -> Args:
    args = Args()
    args.cid = cid
    args.min_num_clients = cfg.min_num_clients
    args.prompt = "universal"
    args.attention = "dual"
    args.dual_init = "aggregated"
    args.label_prompt = 1
    args.img_size = cfg.img_size
    args.sup_type = sup_type
    args.device = "cpu"
    args.use_cuda = 0
    return args


def to_image_tensor(image: np.ndarray, in_chns: int) -> torch.Tensor:
    x = torch.from_numpy(image).float()
    if in_chns == 1:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[0] != 1:
                x = x[:1]
            x = x.unsqueeze(0)
    else:
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.unsqueeze(0)
    return x


def update_acc(acc: Dict[str, float], weak: np.ndarray, gt: np.ndarray, p_fg: np.ndarray) -> None:
    y = (gt > 0).astype(np.float64)
    p = p_fg.astype(np.float64)
    g = p - y
    abs_g = np.abs(g)
    full_mean = float(g.mean())

    masks = {
        "all_labeled": weak != 2,
        "weak_fg": weak == 1,
        "weak_bg": weak == 0,
    }

    acc["n_pixels"] += int(g.size)
    acc["full_g_sum"] += float(g.sum())
    acc["full_abs_g_sum"] += float(abs_g.sum())
    acc["gt_fg_pixels"] += int((gt > 0).sum())
    acc["pred_fg_prob_sum"] += float(p.sum())

    for name, m in masks.items():
        m = m.astype(bool)
        m_count = int(m.sum())
        acc[f"{name}_pixels"] += m_count
        if m_count > 0:
            acc[f"{name}_g_sum"] += float(g[m].sum())
            acc[f"{name}_abs_g_sum"] += float(abs_g[m].sum())
        acc[f"{name}_cov_num_sum"] += float(((m.astype(np.float64) - m.mean()) * (g - full_mean)).sum())


def finalize(acc: Dict[str, float]) -> Dict[str, object]:
    n = max(acc["n_pixels"], 1)
    full_mean = acc["full_g_sum"] / n
    full_abs_mean = acc["full_abs_g_sum"] / n
    out = {
        "n_pixels": int(acc["n_pixels"]),
        "gt_fg_mass": acc["gt_fg_pixels"] / n,
        "pred_fg_prob_mass": acc["pred_fg_prob_sum"] / n,
        "full_mean_grad_fg_logit": full_mean,
        "full_mean_abs_grad_fg_logit": full_abs_mean,
    }
    for name in ["all_labeled", "weak_fg", "weak_bg"]:
        m_count = max(acc[f"{name}_pixels"], 1)
        weak_mean = acc[f"{name}_g_sum"] / m_count
        weak_abs_mean = acc[f"{name}_abs_g_sum"] / m_count
        bias = weak_mean - full_mean
        out[name] = {
            "mask_density_E_m": acc[f"{name}_pixels"] / n,
            "weak_mean_grad_fg_logit": weak_mean,
            "weak_mean_abs_grad_fg_logit": weak_abs_mean,
            "bias_weak_minus_full": bias,
            "cov_over_E_m_identity": bias,
            "gbi_scalar_abs_bias_over_full_abs_grad": abs(bias) / full_abs_mean if full_abs_mean else None,
            "abs_grad_ratio_weak_over_full": weak_abs_mean / full_abs_mean if full_abs_mean else None,
            "sign_flip_full_vs_weak": bool(np.sign(full_mean) != np.sign(weak_mean) and abs(full_mean) > 1e-12 and abs(weak_mean) > 1e-12),
        }
    return out


def summarize_geometry(payload: Dict[str, object]) -> Dict[str, object]:
    by_client = payload["by_client"]
    by_sup = payload["by_sup_type"]
    summary = {}
    for mask_name in ["all_labeled", "weak_fg", "weak_bg"]:
        sup_bias = {
            sup: rec[mask_name]["bias_weak_minus_full"]
            for sup, rec in by_sup.items()
            if mask_name in rec
        }
        pairwise = []
        for a, b in combinations(sorted(sup_bias), 2):
            pairwise.append({
                "sup_pair": [a, b],
                "abs_bias_gap": abs(sup_bias[a] - sup_bias[b]),
                "signed_bias_gap": sup_bias[a] - sup_bias[b],
            })
        max_gap = max((r["abs_bias_gap"] for r in pairwise), default=0.0)

        total_pixels = sum(rec["n_pixels"] for rec in by_client.values())
        weighted_bias = sum(
            rec["n_pixels"] * rec[mask_name]["bias_weak_minus_full"]
            for rec in by_client.values()
        ) / max(total_pixels, 1)
        weighted_full_abs = sum(
            rec["n_pixels"] * rec["full_mean_abs_grad_fg_logit"]
            for rec in by_client.values()
        ) / max(total_pixels, 1)

        densities = [rec[mask_name]["mask_density_E_m"] for rec in by_sup.values() if rec[mask_name]["mask_density_E_m"] > 0]
        summary[mask_name] = {
            "annotation_geometry_divergence_max_pairwise_bias": max_gap,
            "annotation_geometry_divergence_pairwise": pairwise,
            "aggregation_bias_client_pixel_weighted": weighted_bias,
            "aggregation_bias_abs_over_full_abs_grad": abs(weighted_bias) / weighted_full_abs if weighted_full_abs else None,
            "mask_density_max_over_min_by_sup_type": max(densities) / min(densities) if densities else None,
        }
    return summary


def run_dataset(cfg: DatasetConfig, out_dir: str) -> str:
    results = {}
    by_sup = defaultdict(lambda: defaultdict(float))
    for cid, (client, sup_type) in enumerate(zip(cfg.clients, cfg.sup_types)):
        ckpt = os.path.join(cfg.model_dir, f"client_{cid}_unet_univ5_best_model.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)

        args = build_model_args(cfg, cid, sup_type)
        net = net_factory(args, net_type="unet_univ5", in_chns=cfg.in_chns, class_num=2)
        state = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(state)
        net.eval()

        db = BaseDataSets(
            base_dir=cfg.root_path,
            split="train",
            transform=None,
            client=client,
            sup_type=sup_type,
            img_class=cfg.img_class,
        )
        acc = defaultdict(float)
        with torch.no_grad():
            for sample_id in db.sample_list:
                path = os.path.join(cfg.root_path, sample_id)
                with h5py.File(path, "r") as f:
                    image = f["image"][:]
                    gt = f["mask"][:]
                    weak = f[sup_type][:]
                out = net(to_image_tensor(image, cfg.in_chns))[0]
                prob = torch.softmax(out, dim=1)[0, 1].detach().cpu().numpy()
                update_acc(acc, weak, gt, prob)

        record = finalize(acc)
        record["client"] = client
        record["cid"] = cid
        record["sup_type"] = sup_type
        record["checkpoint"] = ckpt
        results[client] = record
        for k, v in acc.items():
            by_sup[sup_type][k] += v
        del net

    payload = {
        "dataset": cfg.dataset,
        "definition": {
            "scalar_gradient": "g_p = p_fg(theta, x)_p - 1[y_p=foreground], the CE gradient for the foreground logit",
            "full_mean": "E_p[g_p] over all pixels with GT mask",
            "weak_mean": "E_p[m_i(p) g_p] / E_p[m_i(p)]",
            "bias": "weak_mean - full_mean = Cov(m_i(p), g_p) / E_p[m_i(p)]",
            "gbi_proxy": "|bias| / E_p[|g_p|]",
            "agd_proxy": "max pairwise |bias_s - bias_t| across annotation types",
            "ab_proxy": "client-pixel-weighted average bias normalized by client-pixel-weighted E[|g_p|]",
        },
        "by_client": results,
        "by_sup_type": {sup: finalize(acc) for sup, acc in by_sup.items()},
    }
    payload["geometry_summary"] = summarize_geometry(payload)

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"{cfg.dataset}_weak_risk_bias_diag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({
        "dataset": cfg.dataset,
        "out_json": out_json,
        "geometry_summary": payload["geometry_summary"],
    }, indent=2))
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["prostate", "polyp", "faz"], choices=sorted(DATASETS))
    parser.add_argument("--out_dir", default="/data/jianbingshen/yanghongji/fedlppaSOTA/code_v4/logs/weak_risk_bias_diag")
    args = parser.parse_args()

    outputs = []
    for name in args.datasets:
        outputs.append(run_dataset(DATASETS[name], args.out_dir))
    print("saved_outputs")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
