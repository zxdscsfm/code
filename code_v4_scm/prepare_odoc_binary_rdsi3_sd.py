import argparse
import json
import os
import shutil
from glob import glob

import h5py
import numpy as np


IGNORE_IN = 3
IGNORE_OUT = 2
LABEL_KEYS = ("mask", "scribble", "keypoint", "block")


def convert_label(arr, target):
    arr = np.asarray(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    ignore = arr == IGNORE_IN
    if target == "od":
        out[(arr == 1) | (arr == 2)] = 1
    elif target == "oc":
        out[arr == 2] = 1
    else:
        raise ValueError(f"unknown target: {target}")
    out[ignore] = IGNORE_OUT
    return out


def copy_file(src, dst, target):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with h5py.File(src, "r") as hsrc, h5py.File(dst, "w") as hdst:
        for key in hsrc.keys():
            data = hsrc[key][:]
            if key in LABEL_KEYS:
                data = convert_label(data, target)
            hdst.create_dataset(key, data=data, compression="gzip")


def unique_values(path):
    values = {key: set() for key in LABEL_KEYS}
    for file_path in glob(os.path.join(path, "**", "*.h5"), recursive=True):
        with h5py.File(file_path, "r") as h:
            for key in LABEL_KEYS:
                if key in h:
                    values[key].update(int(x) for x in np.unique(h[key][:]))
    return {key: sorted(vals) for key, vals in values.items() if vals}


def convert_tree(source, target_root, target_name, overwrite):
    if os.path.exists(target_root):
        if not overwrite:
            raise FileExistsError(f"target exists: {target_root}")
        shutil.rmtree(target_root)
    os.makedirs(target_root, exist_ok=True)

    files = sorted(glob(os.path.join(source, "**", "*.h5"), recursive=True))
    for src in files:
        rel = os.path.relpath(src, source)
        dst = os.path.join(target_root, rel)
        copy_file(src, dst, target_name)

    manifest = {
        "source": source,
        "target": target_root,
        "target_name": target_name,
        "num_classes": 2,
        "ignore_label": IGNORE_OUT,
        "mapping": {
            "od": "0->0, 1/2->1, 3->2(ignore)",
            "oc": "0/1->0, 2->1, 3->2(ignore)",
        }[target_name],
        "file_count": len(files),
        "unique_values": unique_values(target_root),
    }
    with open(os.path.join(target_root, "binary_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target_name", choices=("od", "oc"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert_tree(args.source, args.target, args.target_name, args.overwrite)


if __name__ == "__main__":
    main()
