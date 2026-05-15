import argparse
import glob
from pathlib import Path

import h5py
import numpy as np

from dataloaders.dataset import pseudo_label_generator_acdc


def parse_args():
    parser = argparse.ArgumentParser(description="Add offline random_walker pseudo labels to ACDC h5 files.")
    parser.add_argument(
        "--root_path",
        type=Path,
        default=Path("/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/ACDC_h5"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def iter_h5_paths(root_path: Path):
    pattern = str(root_path / "Domain*" / "*" / "*.h5")
    return sorted(glob.glob(pattern))


def main():
    args = parse_args()
    h5_paths = iter_h5_paths(args.root_path)
    if args.limit > 0:
        h5_paths = h5_paths[:args.limit]

    added = 0
    skipped = 0
    for h5_path in h5_paths:
        with h5py.File(h5_path, "a") as h5f:
            if "scribble" not in h5f:
                skipped += 1
                continue
            if "random_walker" in h5f and not args.overwrite:
                skipped += 1
                continue
            image = h5f["image"][:]
            scribble = h5f["scribble"][:]
            pseudo = pseudo_label_generator_acdc(image, scribble, img_class="acdc").astype(np.uint8)
            if "random_walker" in h5f:
                del h5f["random_walker"]
            h5f.create_dataset("random_walker", data=pseudo, compression="gzip")
            added += 1

    print(f"root_path={args.root_path}")
    print(f"processed={len(h5_paths)}")
    print(f"added_or_overwritten={added}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
