import argparse
import glob
import os
from typing import List, Tuple

import h5py


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset/launcher compatibility before submitting jobs.")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--img_class", type=str, required=True, choices=["odoc", "faz", "polyp", "prostate"])
    parser.add_argument("--in_chns", type=int, required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--min_num_clients", type=int, required=True)
    parser.add_argument(
        "--client_spec",
        action="append",
        default=[],
        help="client spec in the form client_name:sup_type, e.g. client1:scribble_noisy",
    )
    return parser.parse_args()


def parse_client_specs(specs: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid client_spec: {spec}")
        client_name, sup_type = spec.split(":", 1)
        parsed.append((client_name.strip(), sup_type.strip()))
    return parsed


def expected_domain_name(index: int) -> str:
    return f"Domain{index + 1}"


def check_image_shape(image_shape, expected_in_chns: int, expected_img_size: int):
    if len(image_shape) == 2:
        if expected_in_chns != 1:
            raise AssertionError(f"Image shape {image_shape} implies 1 channel, but in_chns={expected_in_chns}")
        h, w = image_shape
    elif len(image_shape) == 3:
        c, h, w = image_shape
        if c != expected_in_chns:
            raise AssertionError(f"Image shape {image_shape} mismatches in_chns={expected_in_chns}")
    else:
        raise AssertionError(f"Unsupported image shape: {image_shape}")

    if h != expected_img_size or w != expected_img_size:
        raise AssertionError(
            f"Image spatial size {(h, w)} mismatches img_size={expected_img_size}"
        )


def main():
    args = parse_args()
    client_specs = parse_client_specs(args.client_spec)

    if len(client_specs) != args.min_num_clients:
        raise AssertionError(
            f"client_spec count {len(client_specs)} mismatches min_num_clients={args.min_num_clients}"
        )

    if not os.path.isdir(args.root_path):
        raise AssertionError(f"root_path does not exist: {args.root_path}")

    print(f"Checking dataset preflight for {args.img_class}")
    print(f"root_path={args.root_path}")
    print(f"in_chns={args.in_chns}, img_size={args.img_size}, min_num_clients={args.min_num_clients}")

    for idx, (client_name, sup_type) in enumerate(client_specs):
        domain = expected_domain_name(idx)
        train_dir = os.path.join(args.root_path, domain, "train")
        test_dir = os.path.join(args.root_path, domain, "test")
        train_files = sorted(glob.glob(os.path.join(train_dir, "*.h5")))
        test_files = sorted(glob.glob(os.path.join(test_dir, "*.h5")))

        if not train_files:
            raise AssertionError(f"{domain} has no train h5 files: {train_dir}")
        if not test_files:
            raise AssertionError(f"{domain} has no test h5 files: {test_dir}")

        sample_path = train_files[0]
        with h5py.File(sample_path, "r") as f:
            keys = list(f.keys())
            if "image" not in f or "mask" not in f:
                raise AssertionError(f"{sample_path} is missing image/mask keys: {keys}")
            if sup_type != "mask" and sup_type not in f:
                raise AssertionError(f"{sample_path} is missing weak label key '{sup_type}': {keys}")

            image_shape = tuple(f["image"].shape)
            mask_shape = tuple(f["mask"].shape)
            check_image_shape(image_shape, args.in_chns, args.img_size)
            if len(mask_shape) != 2 or mask_shape[0] != args.img_size or mask_shape[1] != args.img_size:
                raise AssertionError(f"Mask shape {mask_shape} mismatches img_size={args.img_size}")

            print(
                f"[OK] {domain} {client_name} sup={sup_type} "
                f"train={len(train_files)} test={len(test_files)} image={image_shape} mask={mask_shape}"
            )

    print("Preflight passed.")


if __name__ == "__main__":
    main()
