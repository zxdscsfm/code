import argparse
import json
import os
from glob import glob

import h5py
import numpy as np


UNLABELED_VALUE = 3


def sorted_h5_files(split_dir):
    return sorted(glob(os.path.join(split_dir, "*.h5")))


def summarize_array(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if arr.size else 0.0,
        "std": float(arr.std()) if arr.size else 0.0,
        "min": float(arr.min()) if arr.size else 0.0,
        "max": float(arr.max()) if arr.size else 0.0,
    }


def analyze_split(files, weak_key=None):
    image_means = []
    image_stds = []
    mask_fg_ratio = []
    mask_class1_ratio = []
    mask_class2_ratio = []
    weak_labeled_ratio = []
    weak_conflict_ratio = []
    weak_class0_ratio = []
    weak_class1_ratio = []
    weak_class2_ratio = []
    keys_seen = set()

    for path in files:
        with h5py.File(path, "r") as handle:
            keys_seen.update(handle.keys())
            image = np.asarray(handle["image"][:], dtype=np.float32)
            mask = np.asarray(handle["mask"][:], dtype=np.int32)

            image_means.append(float(image.mean()))
            image_stds.append(float(image.std()))
            mask_fg_ratio.append(float((mask > 0).mean()))
            mask_class1_ratio.append(float((mask == 1).mean()))
            mask_class2_ratio.append(float((mask == 2).mean()))

            if weak_key and weak_key in handle:
                weak = np.asarray(handle[weak_key][:], dtype=np.int32)
                labeled = weak != UNLABELED_VALUE
                weak_labeled_ratio.append(float(labeled.mean()))
                weak_class0_ratio.append(float((weak == 0).mean()))
                weak_class1_ratio.append(float((weak == 1).mean()))
                weak_class2_ratio.append(float((weak == 2).mean()))
                if labeled.any():
                    weak_conflict_ratio.append(float((weak[labeled] != mask[labeled]).mean()))
                else:
                    weak_conflict_ratio.append(0.0)

    summary = {
        "num_files": len(files),
        "keys_seen": sorted(keys_seen),
        "image_mean": summarize_array(image_means),
        "image_std": summarize_array(image_stds),
        "mask_fg_ratio": summarize_array(mask_fg_ratio),
        "mask_class1_ratio": summarize_array(mask_class1_ratio),
        "mask_class2_ratio": summarize_array(mask_class2_ratio),
    }
    if weak_key:
        summary.update(
            {
                "weak_key": weak_key,
                "weak_labeled_ratio": summarize_array(weak_labeled_ratio),
                "weak_conflict_ratio": summarize_array(weak_conflict_ratio),
                "weak_class0_ratio": summarize_array(weak_class0_ratio),
                "weak_class1_ratio": summarize_array(weak_class1_ratio),
                "weak_class2_ratio": summarize_array(weak_class2_ratio),
            }
        )
    return summary


def analyze_domain(root_path, domain_name, weak_key):
    domain_dir = os.path.join(root_path, domain_name)
    train_files = sorted_h5_files(os.path.join(domain_dir, "train"))
    test_files = sorted_h5_files(os.path.join(domain_dir, "test"))
    return {
        "domain": domain_name,
        "train": analyze_split(train_files, weak_key=weak_key),
        "test": analyze_split(test_files, weak_key=None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--domains", nargs="+", required=True)
    parser.add_argument("--weak_key", default="scribble_noisy")
    args = parser.parse_args()

    output = {
        "root_path": args.root_path,
        "domains": [analyze_domain(args.root_path, domain_name, args.weak_key) for domain_name in args.domains],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
