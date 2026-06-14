import argparse
import hashlib
import json
import os
import shutil
from glob import glob

import cv2
import h5py
import numpy as np
from scipy import ndimage


IGNORE_LABEL = 3
SUP_TYPES = ("scribble", "keypoint", "block")
SUP_CYCLE = ("scribble", "keypoint", "scribble", "block", "scribble")


def stable_seed(path, seed, salt):
    digest = hashlib.md5((path + str(seed) + salt).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def thin_mask(mask):
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


def sample_from(mask, n, rng):
    coords = np.argwhere(mask)
    if coords.size == 0 or n <= 0:
        return coords[:0]
    n = min(int(n), len(coords))
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx]


def make_sparse_scribble(mask, density, seed, min_fg=8, min_bg=8):
    mask = mask.astype(np.int64)
    classes = [int(v) for v in np.unique(mask) if int(v) >= 0]
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    target_total = max(1, int(round(mask.size * density)))
    fg_classes = [c for c in classes if c > 0]
    fg_area = sum(int((mask == c).sum()) for c in fg_classes)
    bg_area = int((mask == 0).sum())
    fg_budget = max(min_fg * len(fg_classes), int(round(target_total * 0.45))) if fg_area > 0 else 0
    bg_budget = max(min_bg, target_total - fg_budget)

    for cls in fg_classes:
        cls_mask = mask == cls
        cls_area = int(cls_mask.sum())
        if cls_area <= 0:
            continue
        cls_budget = max(min_fg, int(round(fg_budget * cls_area / max(fg_area, 1))))
        core = thin_mask(cls_mask)
        coords = sample_from(core, cls_budget, rng)
        if len(coords) < cls_budget:
            rest = cls_mask.copy()
            if len(coords):
                rest[coords[:, 0], coords[:, 1]] = False
            more = sample_from(rest, cls_budget - len(coords), rng)
            if len(more):
                coords = np.concatenate([coords, more], axis=0)
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = cls

    if bg_area > 0:
        fg_union = mask > 0
        if fg_union.any():
            dist_to_fg = ndimage.distance_transform_edt(~fg_union)
            bg_values = dist_to_fg[mask == 0]
            far_thresh = np.percentile(bg_values, 65.0)
            near_thresh = np.percentile(bg_values, 35.0)
            bg_far = (mask == 0) & (dist_to_fg >= far_thresh)
            bg_near = (mask == 0) & (dist_to_fg > 0) & (dist_to_fg <= near_thresh)
            near_budget = int(round(bg_budget * 0.35))
            far_budget = bg_budget - near_budget
            parts = [sample_from(bg_far, far_budget, rng), sample_from(bg_near, near_budget, rng)]
            coords = np.concatenate([p for p in parts if len(p)], axis=0) if any(len(p) for p in parts) else np.empty((0, 2), dtype=int)
        else:
            coords = sample_from(mask == 0, bg_budget, rng)
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = 0
    return out


def draw_disk(out, center, radius, value):
    yy, xx = np.ogrid[: out.shape[0], : out.shape[1]]
    cy, cx = int(center[0]), int(center[1])
    disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    out[disk] = value


def center_from_distance(mask, rng):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    dist = ndimage.distance_transform_edt(mask)
    max_val = dist[mask].max()
    candidates = np.argwhere(mask & (dist >= max_val))
    if len(candidates) == 0:
        return coords[rng.integers(0, len(coords))]
    return candidates[rng.integers(0, len(candidates))]


def make_keypoint(mask, radius, bg_radius, seed):
    mask = mask.astype(np.int64)
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    for cls in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
        center = center_from_distance(mask == cls, rng)
        if center is not None:
            draw_disk(out, center, radius, cls)

    bg = mask == 0
    if bg.any():
        fg_union = mask > 0
        if fg_union.any():
            dist = ndimage.distance_transform_edt(~fg_union)
            bg_dist = dist * bg
            max_val = bg_dist.max()
            candidates = np.argwhere(bg & (bg_dist >= max_val))
            center = candidates[rng.integers(0, len(candidates))] if len(candidates) else center_from_distance(bg, rng)
        else:
            center = center_from_distance(bg, rng)
        if center is not None:
            draw_disk(out, center, bg_radius, 0)
    return out


def make_block(mask, block_size, fg_thresh, bg_thresh):
    mask = mask.astype(np.int64)
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    h, w = mask.shape
    classes = sorted(int(v) for v in np.unique(mask) if int(v) >= 0)
    fg_classes = [c for c in classes if c > 0]
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            y1 = min(y0 + block_size, h)
            x1 = min(x0 + block_size, w)
            patch = mask[y0:y1, x0:x1]
            area = float(patch.size)
            fg_frac = float((patch > 0).sum()) / area
            label = None
            if fg_frac <= bg_thresh:
                label = 0
            else:
                best_cls = None
                best_frac = 0.0
                for cls in fg_classes:
                    frac = float((patch == cls).sum()) / area
                    if frac > best_frac:
                        best_frac = frac
                        best_cls = cls
                if best_cls is not None and best_frac >= fg_thresh:
                    label = best_cls
            if label is not None:
                out[y0:y1, x0:x1] = label
    return out


def dataset_stats(arr):
    arr = np.asarray(arr)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
    }


def process_file(src, dst, split, args):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with h5py.File(src, "r") as hsrc, h5py.File(dst, "w") as hdst:
        image = hsrc["image"][:]
        mask = hsrc["mask"][:].astype(np.uint8)
        hdst.create_dataset("image", data=image, compression="gzip")
        hdst.create_dataset("mask", data=mask, compression="gzip")
        if split == "train":
            rel = os.path.relpath(src, args.source)
            scribble = make_sparse_scribble(mask, args.scribble_density, stable_seed(rel, args.seed, "scribble"))
            keypoint = make_keypoint(mask, args.keypoint_radius, args.keypoint_bg_radius, stable_seed(rel, args.seed, "keypoint"))
            block = make_block(mask, args.block_size, args.block_fg_thresh, args.block_bg_thresh)
            hdst.create_dataset("scribble", data=scribble, compression="gzip")
            hdst.create_dataset("keypoint", data=keypoint, compression="gzip")
            hdst.create_dataset("block", data=block, compression="gzip")


def prepare(args):
    if os.path.abspath(args.source) == os.path.abspath(args.target):
        raise ValueError("source and target must be different")
    if os.path.exists(args.target):
        if not args.overwrite:
            raise FileExistsError(f"target exists: {args.target}. Use --overwrite to rebuild it.")
        shutil.rmtree(args.target)
    os.makedirs(args.target, exist_ok=True)

    manifest = {
        "source": args.source,
        "target": args.target,
        "cycle": list(SUP_CYCLE),
        "seed": args.seed,
        "params": {
            "scribble_density": args.scribble_density,
            "keypoint_radius": args.keypoint_radius,
            "keypoint_bg_radius": args.keypoint_bg_radius,
            "block_size": args.block_size,
            "block_fg_thresh": args.block_fg_thresh,
            "block_bg_thresh": args.block_bg_thresh,
        },
        "note": "ODOC rdsi3_sd protocol regenerated from full masks; only scribble/keypoint/block are kept for train weak labels; box and scribble_noisy are not used.",
        "domains": {},
    }

    for domain_idx in range(1, 6):
        domain = f"Domain{domain_idx}"
        assigned = SUP_CYCLE[domain_idx - 1]
        counts = {}
        label_density = {k: [] for k in SUP_TYPES}
        fg_density = {k: [] for k in SUP_TYPES}
        mask_values = set()
        weak_values = {k: set() for k in SUP_TYPES}

        for split in ("train", "test"):
            src_files = sorted(glob(os.path.join(args.source, domain, split, "*.h5")))
            counts[split] = len(src_files)
            for src in src_files:
                rel = os.path.relpath(src, args.source)
                dst = os.path.join(args.target, rel)
                process_file(src, dst, split, args)
                with h5py.File(dst, "r") as h:
                    mask = h["mask"][:]
                    mask_values.update(int(x) for x in np.unique(mask))
                    if split == "train":
                        for key in SUP_TYPES:
                            arr = h[key][:]
                            weak_values[key].update(int(x) for x in np.unique(arr))
                            labeled = arr != IGNORE_LABEL
                            fg = (arr > 0) & (arr != IGNORE_LABEL)
                            label_density[key].append(float(labeled.mean()))
                            fg_density[key].append(float(fg.mean()))

        manifest["domains"][domain] = {
            "assigned_sup_type": assigned,
            "train_files": counts.get("train", 0),
            "test_files": counts.get("test", 0),
            "mask_values": sorted(mask_values),
            "weak_values": {k: sorted(v) for k, v in weak_values.items()},
            "label_density": {k: dataset_stats(v) for k, v in label_density.items()},
            "fg_density": {k: dataset_stats(v) for k, v in fg_density.items()},
        }

    manifest_path = os.path.join(args.target, "rdsi3_sparse_hetero_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/data/jianbingshen/yanghongji/FedLPPA_Original/data/ODOC_h5")
    parser.add_argument("--target", default="/data/jianbingshen/yanghongji/FedLPPA_Original/data/ODOC_h5_rdsi3_sd")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--scribble_density", type=float, default=0.03)
    parser.add_argument("--keypoint_radius", type=int, default=4)
    parser.add_argument("--keypoint_bg_radius", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--block_fg_thresh", type=float, default=0.65)
    parser.add_argument("--block_bg_thresh", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
