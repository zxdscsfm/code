import argparse
import hashlib
import os
from glob import glob

import cv2
import h5py
import numpy as np
from scipy import ndimage


def _stable_seed(path, seed):
    h = hashlib.md5((path + str(seed)).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _thin_mask(mask):
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.sum() == 0:
        return mask_u8.astype(bool)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return mask_u8.astype(bool)
    local_max = dist == ndimage.maximum_filter(dist, size=3)
    skeleton = local_max & mask
    if skeleton.sum() == 0:
        skeleton = dist >= np.percentile(dist[mask], 75.0)
    return skeleton


def _sample_from(mask, n, rng):
    coords = np.argwhere(mask)
    if coords.size == 0 or n <= 0:
        return coords[:0]
    n = min(int(n), len(coords))
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx]


def make_sparse_scribble(mask, density, seed, min_fg=8, min_bg=8):
    mask = mask.astype(np.int64)
    classes = [int(v) for v in np.unique(mask) if int(v) >= 0]
    num_classes = max(classes) + 1 if classes else 1
    unknown = num_classes
    out = np.full(mask.shape, unknown, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    target_total = max(1, int(round(mask.size * density)))
    fg_classes = [c for c in classes if c > 0]
    fg_area = sum(int((mask == c).sum()) for c in fg_classes)
    bg_area = int((mask == 0).sum())
    if fg_area > 0:
        fg_budget = max(min_fg * len(fg_classes), int(round(target_total * 0.45)))
    else:
        fg_budget = 0
    bg_budget = max(min_bg, target_total - fg_budget)

    for cls in fg_classes:
        cls_mask = mask == cls
        cls_area = int(cls_mask.sum())
        if cls_area <= 0:
            continue
        cls_budget = max(min_fg, int(round(fg_budget * cls_area / max(fg_area, 1))))
        core = _thin_mask(cls_mask)
        coords = _sample_from(core, cls_budget, rng)
        if len(coords) < cls_budget:
            rest = cls_mask.copy()
            if len(coords):
                rest[coords[:, 0], coords[:, 1]] = False
            more = _sample_from(rest, cls_budget - len(coords), rng)
            coords = np.concatenate([coords, more], axis=0) if len(more) else coords
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = cls

    if bg_area > 0:
        fg_union = mask > 0
        if fg_union.any():
            dist_to_fg = ndimage.distance_transform_edt(~fg_union)
            bg_far = (mask == 0) & (dist_to_fg >= np.percentile(dist_to_fg[mask == 0], 65.0))
            bg_near = (mask == 0) & (dist_to_fg > 0) & (dist_to_fg <= np.percentile(dist_to_fg[mask == 0], 35.0))
            near_budget = int(round(bg_budget * 0.35))
            far_budget = bg_budget - near_budget
            coords = [_sample_from(bg_far, far_budget, rng), _sample_from(bg_near, near_budget, rng)]
            coords = np.concatenate([c for c in coords if len(c)], axis=0) if any(len(c) for c in coords) else np.empty((0, 2), dtype=int)
        else:
            coords = _sample_from(mask == 0, bg_budget, rng)
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = 0
    return out


def process(root, datasets, key, density, seed, overwrite):
    totals = {}
    for dataset in datasets:
        ds_root = os.path.join(root, dataset)
        paths = sorted(glob(os.path.join(ds_root, "Domain*", "train", "*.h5")))
        if not paths:
            print("skip", dataset, "no train h5")
            continue
        stats = []
        for path in paths:
            with h5py.File(path, "a") as f:
                if key in f and not overwrite:
                    arr = f[key][()]
                else:
                    arr = make_sparse_scribble(f["mask"][()], density, _stable_seed(path, seed))
                    if key in f:
                        del f[key]
                    f.create_dataset(key, data=arr, compression="gzip")
                mask = f["mask"][()]
                unknown = int(mask.max()) + 1
                labeled = arr != unknown
                fg = (arr > 0) & (arr != unknown)
                stats.append((float(labeled.mean()), float(fg.mean())))
        lab = np.array([s[0] for s in stats])
        fg = np.array([s[1] for s in stats])
        totals[dataset] = (float(lab.mean()), float(fg.mean()), float(lab.min()), float(lab.max()), len(paths))
        print(
            dataset,
            "files", len(paths),
            "labeled_mean %.5f fg_mean %.5f labeled_min %.5f labeled_max %.5f" %
            (lab.mean(), fg.mean(), lab.min(), lab.max()),
        )
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/jianbingshen/yanghongji/FedLPPA_Original/data")
    parser.add_argument("--datasets", nargs="+", default=["POLYP_h5", "PROSTATE_h5"])
    parser.add_argument("--key", default="sparse_scribble_5")
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    process(args.root, args.datasets, args.key, args.density, args.seed, args.overwrite)


if __name__ == "__main__":
    main()
