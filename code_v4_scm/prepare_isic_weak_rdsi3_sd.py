import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image
from scipy import ndimage


IGNORE_LABEL = 2
SUP_TYPES = ("scribble", "keypoint", "block")
SUP_CYCLE = ("scribble", "keypoint", "scribble", "block")


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
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    target_total = max(1, int(round(mask.size * density)))
    fg_area = int((mask == 1).sum())
    bg_area = int((mask == 0).sum())
    fg_budget = max(min_fg, int(round(target_total * 0.45))) if fg_area > 0 else 0
    bg_budget = max(min_bg, target_total - fg_budget)

    if fg_area > 0:
        core = thin_mask(mask == 1)
        coords = sample_from(core, fg_budget, rng)
        if len(coords) < fg_budget:
            rest = mask == 1
            if len(coords):
                rest = rest.copy()
                rest[coords[:, 0], coords[:, 1]] = False
            more = sample_from(rest, fg_budget - len(coords), rng)
            if len(more):
                coords = np.concatenate([coords, more], axis=0)
        if len(coords):
            out[coords[:, 0], coords[:, 1]] = 1

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
            coords = (
                np.concatenate([p for p in parts if len(p)], axis=0)
                if any(len(p) for p in parts)
                else np.empty((0, 2), dtype=int)
            )
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
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    rng = np.random.default_rng(seed)

    center = center_from_distance(mask == 1, rng)
    if center is not None:
        draw_disk(out, center, radius, 1)

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
    out = np.full(mask.shape, IGNORE_LABEL, dtype=np.uint8)
    h, w = mask.shape
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            y1 = min(y0 + block_size, h)
            x1 = min(x0 + block_size, w)
            patch = mask[y0:y1, x0:x1]
            fg_frac = float((patch == 1).sum()) / float(patch.size)
            if fg_frac <= bg_thresh:
                out[y0:y1, x0:x1] = 0
            elif fg_frac >= fg_thresh:
                out[y0:y1, x0:x1] = 1
    return out


def load_pair(root, domain, name, size):
    image_path = root / domain / "image" / name
    mask_path = root / domain / "mask" / name
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    image_arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mask_arr = (np.asarray(mask) > 127).astype(np.uint8)
    return image_arr, mask_arr


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def write_case(src_root, dst_path, domain, split, name, args):
    os.makedirs(dst_path.parent, exist_ok=True)
    image, mask = load_pair(src_root, domain, name, args.size)
    with h5py.File(dst_path, "w") as h:
        h.create_dataset("image", data=image, compression="gzip")
        h.create_dataset("mask", data=mask, compression="gzip")
        if split == "train":
            rel = f"{domain}/{split}/{name}"
            h.create_dataset(
                "scribble",
                data=make_sparse_scribble(mask, args.scribble_density, stable_seed(rel, args.seed, "scribble")),
                compression="gzip",
            )
            h.create_dataset(
                "keypoint",
                data=make_keypoint(mask, args.keypoint_radius, args.keypoint_bg_radius, stable_seed(rel, args.seed, "keypoint")),
                compression="gzip",
            )
            h.create_dataset(
                "block",
                data=make_block(mask, args.block_size, args.block_fg_thresh, args.block_bg_thresh),
                compression="gzip",
            )


def prepare(args):
    src_root = Path(args.source)
    target = Path(args.target)
    split = json.loads(Path(args.split_json).read_text())
    if target.exists():
        if not args.overwrite:
            raise FileExistsError(f"target exists: {target}. Use --overwrite to rebuild it.")
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(src_root),
        "target": str(target),
        "split_json": args.split_json,
        "protocol": "FedLPPA weak-supervised h5; HSSF domain unlabeled_train is used as weak-label train, test_list as test; public source is not used.",
        "size": args.size,
        "seed": args.seed,
        "cycle": list(SUP_CYCLE),
        "params": {
            "scribble_density": args.scribble_density,
            "keypoint_radius": args.keypoint_radius,
            "keypoint_bg_radius": args.keypoint_bg_radius,
            "block_size": args.block_size,
            "block_fg_thresh": args.block_fg_thresh,
            "block_bg_thresh": args.block_bg_thresh,
        },
        "domains": {},
    }

    for idx, domain in enumerate(("domain1", "domain2", "domain3", "domain4"), start=1):
        out_domain = f"Domain{idx}"
        parts = split[domain]
        train_names = parts.get("unlabeled_train", [])
        test_names = parts.get("test_list", [])
        domain_stats = {"train_files": len(train_names), "test_files": len(test_names), "assigned_sup_type": SUP_CYCLE[idx - 1]}
        label_density = {k: [] for k in SUP_TYPES}
        fg_density = {k: [] for k in SUP_TYPES}

        for split_name, names in (("train", train_names), ("test", test_names)):
            for name in names:
                dst = target / out_domain / split_name / (Path(name).stem + ".h5")
                write_case(src_root, dst, domain, split_name, name, args)
                if split_name == "train":
                    with h5py.File(dst, "r") as h:
                        for key in SUP_TYPES:
                            arr = h[key][:]
                            labeled = arr != IGNORE_LABEL
                            fg = (arr == 1)
                            label_density[key].append(float(labeled.mean()))
                            fg_density[key].append(float(fg.mean()))

        domain_stats["label_density"] = {k: stats(v) for k, v in label_density.items()}
        domain_stats["fg_density"] = {k: stats(v) for k, v in fg_density.items()}
        manifest["domains"][out_domain] = domain_stats

    with open(target / "rdsi3_sparse_hetero_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/data/jianbingshen/yanghongji/HSSF_data/data/isic")
    parser.add_argument("--split_json", default="/data/jianbingshen/yanghongji/HSSF/data_split/isic-[1, 0.0, 0.0, 0.0, 0.0]-['public', 'domain1', 'domain2', 'domain3', 'domain4'].json")
    parser.add_argument("--target", default="/data/jianbingshen/yanghongji/FedLPPA_Original/data/ISIC_h5_rdsi3_sd")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--scribble_density", type=float, default=0.03)
    parser.add_argument("--keypoint_radius", type=int, default=4)
    parser.add_argument("--keypoint_bg_radius", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--block_fg_thresh", type=float, default=0.65)
    parser.add_argument("--block_bg_thresh", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
