import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


IGNORE = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a Prostate h5 variant with paper-aligned, high-quality weak labels: "
            "Domain5 gets mask-derived Scribble2 and Domain6 gets mask-eroded Block."
        )
    )
    parser.add_argument("--src-root", required=True, help="Source PROSTATE_h5 root.")
    parser.add_argument("--dst-root", required=True, help="Destination PROSTATE_h5 root.")
    parser.add_argument("--domain5", default="Domain5")
    parser.add_argument("--domain6", default="Domain6")
    parser.add_argument("--scribble2-key", default="scribble2")
    parser.add_argument("--block-key", default="block")
    parser.add_argument("--domain5-overwrite-scribble", action="store_true")
    parser.add_argument("--domain6-overwrite-block", action="store_true")
    parser.add_argument("--block-erode", type=int, default=8)
    parser.add_argument("--scribble-fg-erode", type=int, default=3)
    parser.add_argument("--scribble-bg-dilate", type=int, default=6)
    parser.add_argument("--scribble-bg-stride", type=int, default=12)
    parser.add_argument("--scribble-fg-dilate", type=int, default=1)
    parser.add_argument("--scribble-elastic-alpha", type=float, default=6.0)
    parser.add_argument("--scribble-elastic-sigma", type=float, default=10.0)
    parser.add_argument("--scribble-erasure-prob", type=float, default=0.35)
    parser.add_argument("--scribble-erasure-block", type=int, default=12)
    parser.add_argument("--scribble-seed", type=int, default=2022)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--copy-mode", choices=["copy", "hardlink"], default="copy")
    return parser.parse_args()


def ensure_safe_fresh_delete(dst_root: Path):
    name = dst_root.name.lower()
    allowed = ["paper_oracle", "scribble2", "maskblock", "processed"]
    if not any(token in name for token in allowed):
        raise ValueError(f"Refusing to delete destination with unsafe name: {dst_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)


def clone_tree(src_root: Path, dst_root: Path, modified_domains: set[str], copy_mode: str):
    dst_root.mkdir(parents=True, exist_ok=True)
    for src_path in src_root.rglob("*"):
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            continue
        must_copy = rel.parts and rel.parts[0] in modified_domains
        if copy_mode == "hardlink" and not must_copy:
            os.link(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num <= 1:
        return mask
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    return labeled == int(counts.argmax())


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    # Morphological skeleton: deterministic, dependency-light, and stable for 2D prostate masks.
    mask = mask.astype(bool)
    skel = np.zeros(mask.shape, dtype=bool)
    current = mask.copy()
    structure = ndimage.generate_binary_structure(2, 1)
    while current.any():
        opened = ndimage.binary_opening(current, structure=structure)
        skel |= current & ~opened
        current = ndimage.binary_erosion(current, structure=structure)
    return skel


def _sample_strokes(mask: np.ndarray, stride: int, phase: int = 0) -> np.ndarray:
    if stride <= 1:
        return mask
    coords = np.indices(mask.shape)
    keep = ((coords[0] + 2 * coords[1] + phase) % stride) == 0
    return mask & keep


def _elastic_transform_binary(mask: np.ndarray, alpha: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if alpha <= 0:
        return mask
    shape = mask.shape
    dx = ndimage.gaussian_filter((rng.random(shape) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
    dy = ndimage.gaussian_filter((rng.random(shape) * 2.0 - 1.0), sigma=sigma, mode="reflect") * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    warped = ndimage.map_coordinates(mask.astype(np.float32), [y + dy, x + dx], order=0, mode="nearest")
    return warped > 0.5


def _random_erase(mask: np.ndarray, erase_prob: float, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if erase_prob <= 0 or block_size <= 0 or not mask.any():
        return mask
    erased = mask.copy()
    h, w = mask.shape
    for y0 in range(0, h, block_size):
        for x0 in range(0, w, block_size):
            patch = mask[y0 : y0 + block_size, x0 : x0 + block_size]
            if patch.any() and rng.random() < erase_prob:
                erased[y0 : y0 + block_size, x0 : x0 + block_size] = False
    if not erased.any():
        ys, xs = np.where(mask)
        keep_idx = rng.integers(0, len(ys))
        erased[ys[keep_idx], xs[keep_idx]] = True
    return erased


def build_scribble2(
    mask: np.ndarray,
    fg_erode: int,
    bg_dilate: int,
    bg_stride: int,
    fg_dilate: int,
    elastic_alpha: float,
    elastic_sigma: float,
    erasure_prob: float,
    erasure_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    fg = mask > 0
    fg = _largest_component(fg)
    if fg_erode > 0:
        fg_core = ndimage.binary_erosion(fg, iterations=fg_erode, border_value=0)
        if not fg_core.any():
            fg_core = fg
    else:
        fg_core = fg

    fg_seed = _skeletonize(fg_core)
    fg_seed = _elastic_transform_binary(fg_seed, elastic_alpha, elastic_sigma, rng)
    fg_seed &= fg
    fg_seed = _random_erase(fg_seed, erasure_prob, erasure_block, rng)
    if fg_dilate > 0:
        fg_seed = ndimage.binary_dilation(fg_seed, iterations=fg_dilate) & fg_core
    if not fg_seed.any() and fg_core.any():
        dist = ndimage.distance_transform_edt(fg_core)
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        fg_seed[y, x] = True

    bg_safe = ~fg
    if bg_dilate > 0:
        bg_safe &= ~ndimage.binary_dilation(fg, iterations=bg_dilate)
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    bg_seed = ndimage.binary_dilation(border, iterations=max(1, bg_dilate // 2)) & bg_safe
    bg_seed |= _sample_strokes(bg_safe, bg_stride, phase=2)

    scribble = np.full(mask.shape, IGNORE, dtype=np.uint8)
    scribble[bg_seed] = 0
    scribble[fg_seed] = 1
    return scribble


def build_mask_block(mask: np.ndarray, erode: int) -> np.ndarray:
    fg = _largest_component(mask > 0)
    fg_core = ndimage.binary_erosion(fg, iterations=erode, border_value=0) if erode > 0 else fg
    if not fg_core.any() and fg.any():
        fg_core = ndimage.binary_erosion(fg, iterations=max(erode // 2, 1), border_value=0)
    if not fg_core.any():
        fg_core = fg

    block = np.full(mask.shape, IGNORE, dtype=np.uint8)
    block[~fg] = 0
    block[fg_core] = 1
    return block


def replace_dataset(h5f: h5py.File, key: str, data: np.ndarray, attrs: dict):
    if key in h5f:
        del h5f[key]
    ds = h5f.create_dataset(key, data=data.astype(np.uint8), compression="gzip")
    for attr_key, attr_val in attrs.items():
        ds.attrs[attr_key] = attr_val


def process_domain5(path: Path, args, rng: np.random.Generator) -> dict:
    with h5py.File(path, "a") as h5f:
        mask = h5f["mask"][()]
        scribble2 = build_scribble2(
            mask,
            fg_erode=args.scribble_fg_erode,
            bg_dilate=args.scribble_bg_dilate,
            bg_stride=args.scribble_bg_stride,
            fg_dilate=args.scribble_fg_dilate,
            elastic_alpha=args.scribble_elastic_alpha,
            elastic_sigma=args.scribble_elastic_sigma,
            erasure_prob=args.scribble_erasure_prob,
            erasure_block=args.scribble_erasure_block,
            rng=rng,
        )
        attrs = {
            "source_key": "mask",
            "conversion": "mask_derived_scribble2_erosion_skeleton_elastic_erasure",
            "fg_erode": args.scribble_fg_erode,
            "bg_dilate": args.scribble_bg_dilate,
            "bg_stride": args.scribble_bg_stride,
            "fg_dilate": args.scribble_fg_dilate,
            "elastic_alpha": args.scribble_elastic_alpha,
            "elastic_sigma": args.scribble_elastic_sigma,
            "erasure_prob": args.scribble_erasure_prob,
            "erasure_block": args.scribble_erasure_block,
        }
        replace_dataset(h5f, args.scribble2_key, scribble2, attrs)
        if args.domain5_overwrite_scribble:
            if "scribble_original_public" not in h5f and "scribble" in h5f:
                replace_dataset(h5f, "scribble_original_public", h5f["scribble"][()], {"source_key": "scribble"})
            replace_dataset(h5f, "scribble", scribble2, attrs)
    return summarize_label(scribble2, mask)


def process_domain6(path: Path, args) -> dict:
    with h5py.File(path, "a") as h5f:
        mask = h5f["mask"][()]
        block = build_mask_block(mask, args.block_erode)
        attrs = {
            "source_key": "mask",
            "conversion": "mask_derived_eroded_block_upper_bound",
            "erode": args.block_erode,
        }
        replace_dataset(h5f, "block_oracle", block, attrs)
        if args.domain6_overwrite_block:
            replace_dataset(h5f, args.block_key, block, attrs)
    return summarize_label(block, mask)


def summarize_label(label: np.ndarray, mask: np.ndarray) -> dict:
    fg = label == 1
    bg = label == 0
    valid = label != IGNORE
    gt = mask > 0
    return {
        "fg": int(fg.sum()),
        "bg": int(bg.sum()),
        "ignore": int((label == IGNORE).sum()),
        "valid_ratio": float(valid.mean()),
        "fg_precision": float((fg & gt).sum() / max(fg.sum(), 1)),
        "fg_recall": float((fg & gt).sum() / max(gt.sum(), 1)),
        "bg_precision": float((bg & ~gt).sum() / max(bg.sum(), 1)),
    }


def print_summary(name: str, rows: list[dict]):
    print(f"{name}: n={len(rows)}")
    for key in ["valid_ratio", "fg", "bg", "ignore", "fg_precision", "fg_recall", "bg_precision"]:
        vals = [r[key] for r in rows]
        print(f"  {key}: mean={float(np.mean(vals)):.6f} p50={float(np.percentile(vals, 50)):.6f}")


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if args.fresh:
        ensure_safe_fresh_delete(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(src_root)

    clone_tree(src_root, dst_root, {args.domain5, args.domain6}, args.copy_mode)

    d5_rows = []
    d6_rows = []
    rng = np.random.default_rng(args.scribble_seed)
    for split in ["train", "test"]:
        for path in sorted((dst_root / args.domain5 / split).glob("*.h5")):
            d5_rows.append(process_domain5(path, args, rng))
        for path in sorted((dst_root / args.domain6 / split).glob("*.h5")):
            d6_rows.append(process_domain6(path, args))

    print(f"source={src_root}")
    print(f"destination={dst_root}")
    print(f"domain5={args.domain5} scribble2_key={args.scribble2_key} overwrite_scribble={args.domain5_overwrite_scribble}")
    print(f"domain6={args.domain6} block_key={args.block_key} overwrite_block={args.domain6_overwrite_block}")
    print_summary("Domain5 scribble2", d5_rows)
    print_summary("Domain6 mask-block", d6_rows)


if __name__ == "__main__":
    main()
