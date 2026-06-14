import glob
import sys

import h5py
import numpy as np

from dataloaders.dataset import random_rotate


ROOTS = [
    "/data/jianbingshen/yanghongji/FedLPPA_Original/data/ODOC_OD_h5_rdsi3_sd",
    "/data/jianbingshen/yanghongji/FedLPPA_Original/data/ODOC_OC_h5_rdsi3_sd",
]


def check_dataset(root):
    files = sorted(glob.glob(root + "/Domain*/*/*.h5"))
    mask_vals = set()
    weak_vals = {}
    for path in files:
        with h5py.File(path, "r") as handle:
            mask_vals.update(np.unique(handle["mask"][:]).astype(int).tolist())
            for key in ("scribble", "keypoint", "block"):
                if key in handle:
                    weak_vals.setdefault(key, set()).update(np.unique(handle[key][:]).astype(int).tolist())

    if not mask_vals <= {0, 1}:
        raise RuntimeError(f"{root} mask values {sorted(mask_vals)} exceed binary labels")
    for key, vals in weak_vals.items():
        if not vals <= {0, 1, 2}:
            raise RuntimeError(f"{root} {key} values {sorted(vals)} exceed binary ignore protocol")

    weak_msg = " ".join(f"{key}={sorted(vals)}" for key, vals in sorted(weak_vals.items()))
    print(root.split("/")[-1], "files", len(files), "mask", sorted(mask_vals), weak_msg)


def check_rotate():
    image = np.zeros((3, 384, 384), dtype=np.float32)
    label = np.zeros((384, 384), dtype=np.uint8)
    label[96:288, 96:288] = 1
    vals = set()
    for _ in range(32):
        _, rotated = random_rotate(image, label, img_class="odoc_binary")
        vals.update(np.unique(rotated).astype(int).tolist())
    if not vals <= {0, 1, 2}:
        raise RuntimeError(f"odoc_binary rotate produced invalid values {sorted(vals)}")
    print("odoc_binary rotate", sorted(vals))


def main():
    for root in ROOTS:
        check_dataset(root)
    check_rotate()


if __name__ == "__main__":
    sys.exit(main())
