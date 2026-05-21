import argparse
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a paper-style PROSTATE_h5 variant where Domain6 box labels are converted to block labels."
    )
    parser.add_argument("--src-root", required=True, help="Source PROSTATE_h5 root with Domain*/train|test.")
    parser.add_argument("--dst-root", required=True, help="Destination PROSTATE_h5 root to create or update.")
    parser.add_argument("--domain", default="Domain6", help="Domain containing bounding-box labels.")
    parser.add_argument("--box-key", default="box")
    parser.add_argument("--block-key", default="block")
    parser.add_argument("--erode-radius", type=int, default=20)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Remove dst-root before creating it. Refuses to remove paths that are not named like a box2block variant.",
    )
    parser.add_argument(
        "--overwrite-block",
        action="store_true",
        help="Overwrite an existing block key in the target h5 files.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to duplicate unchanged h5 files. Domain files being modified are always copied.",
    )
    return parser.parse_args()


def ensure_safe_fresh_delete(dst_root: Path):
    name = dst_root.name.lower()
    if "box2block" not in name:
        raise ValueError(f"Refusing to delete destination without 'box2block' in name: {dst_root}")
    if dst_root.exists():
        shutil.rmtree(dst_root)


def clone_tree(src_root: Path, dst_root: Path, modified_domain: str, copy_mode: str):
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
        must_copy = rel.parts and rel.parts[0] == modified_domain
        if copy_mode == "hardlink" and not must_copy:
            os.link(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)


def convert_one_file(path: Path, box_key: str, block_key: str, erode_radius: int, overwrite: bool):
    with h5py.File(path, "a") as h5f:
        if box_key not in h5f:
            raise KeyError(f"{path} has no key '{box_key}'")
        if block_key in h5f:
            if not overwrite:
                return False
            del h5f[block_key]

        box = h5f[box_key][()]
        support = box != 0
        foreground = ndimage.binary_erosion(support, iterations=erode_radius, border_value=0)

        block = np.full(box.shape, 2, dtype=np.uint8)
        block[~support] = 0
        block[foreground] = 1

        ds = h5f.create_dataset(block_key, data=block, compression="gzip")
        ds.attrs["source_key"] = box_key
        ds.attrs["conversion"] = "paper_style_box_inside_as_prostate_outside_as_bg_then_erode"
        ds.attrs["erode_radius"] = erode_radius
    return True


def summarize_domain(dst_root: Path, domain: str, block_key: str):
    rows = []
    for split in ["train", "test"]:
        for path in sorted((dst_root / domain / split).glob("*.h5")):
            with h5py.File(path, "r") as h5f:
                block = h5f[block_key][()]
                mask = h5f["mask"][()] > 0 if "mask" in h5f else None
            fg = block == 1
            bg = block == 0
            ign = block == 2
            row = {
                "split": split,
                "fg": int(fg.sum()),
                "bg": int(bg.sum()),
                "ignore": int(ign.sum()),
                "labeled_ratio": float((fg.sum() + bg.sum()) / block.size),
            }
            if mask is not None and fg.sum() > 0 and mask.sum() > 0:
                row["fg_precision_vs_mask"] = float(np.logical_and(fg, mask).sum() / fg.sum())
                row["fg_recall_vs_mask"] = float(np.logical_and(fg, mask).sum() / mask.sum())
            rows.append(row)
    for split in ["train", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        if not split_rows:
            continue
        print(
            "{} {}: n={} fg_mean={:.2f} ignore_mean={:.2f} labeled_ratio_mean={:.4f}".format(
                domain,
                split,
                len(split_rows),
                float(np.mean([r["fg"] for r in split_rows])),
                float(np.mean([r["ignore"] for r in split_rows])),
                float(np.mean([r["labeled_ratio"] for r in split_rows])),
            )
        )
        if "fg_precision_vs_mask" in split_rows[0]:
            print(
                "{} {}: fg_precision_vs_mask_mean={:.4f} fg_recall_vs_mask_mean={:.4f}".format(
                    domain,
                    split,
                    float(np.mean([r["fg_precision_vs_mask"] for r in split_rows])),
                    float(np.mean([r["fg_recall_vs_mask"] for r in split_rows])),
                )
            )


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()

    if not src_root.exists():
        raise FileNotFoundError(src_root)
    if args.erode_radius < 0:
        raise ValueError("--erode-radius must be non-negative")
    if args.fresh:
        ensure_safe_fresh_delete(dst_root)

    clone_tree(src_root, dst_root, args.domain, args.copy_mode)

    changed = 0
    for split in ["train", "test"]:
        for path in sorted((dst_root / args.domain / split).glob("*.h5")):
            changed += int(
                convert_one_file(
                    path,
                    box_key=args.box_key,
                    block_key=args.block_key,
                    erode_radius=args.erode_radius,
                    overwrite=args.overwrite_block,
                )
            )

    print(f"source={src_root}")
    print(f"destination={dst_root}")
    print(f"domain={args.domain} block_key={args.block_key} erode_radius={args.erode_radius}")
    print(f"converted_files={changed}")
    summarize_domain(dst_root, args.domain, args.block_key)


if __name__ == "__main__":
    main()
