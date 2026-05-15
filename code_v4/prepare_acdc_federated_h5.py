import argparse
import zipfile
from pathlib import Path

import cv2
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


IGNORE_INDEX = 4
CLASS_VALUES = (1, 2, 3)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare federated ACDC h5 slices for FedLPPA.")
    parser.add_argument(
        "--raw_root",
        type=Path,
        default=Path("/data/jianbingshen/yanghongji/FedLPPA_github_official_clean/ACDC_raw"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/jianbingshen/yanghongji/FedLPPA_github_official_full/data/ACDC_h5"),
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_domains", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_extracted(raw_root: Path):
    for split in ("training", "testing"):
        split_dir = raw_root / split
        zip_path = raw_root / f"{split}.zip"
        if split_dir.is_dir():
            continue
        if not zip_path.is_file():
            raise FileNotFoundError(f"Missing both {split_dir} and {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_root)


def chunk_evenly(items, num_chunks):
    items = list(sorted(items))
    total = len(items)
    if total == 0:
        return [[] for _ in range(num_chunks)]
    base = total // num_chunks
    remainder = total % num_chunks
    chunks = []
    start = 0
    for idx in range(num_chunks):
        size = base + (1 if idx < remainder else 0)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def load_nifti_array(path: Path):
    try:
        image = nib.load(str(path))
        array = np.asarray(image.dataobj)
        if array.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI volume, got shape {array.shape} for {path}")
        return np.transpose(array, (2, 0, 1))
    except Exception:
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image)


def normalize_slice(image_slice):
    image_slice = image_slice.astype(np.float32)
    min_value = float(image_slice.min())
    max_value = float(image_slice.max())
    if max_value <= min_value:
        return np.zeros_like(image_slice, dtype=np.float32)
    return (image_slice - min_value) / (max_value - min_value)


def resize_image_and_mask(image_slice, mask_slice, img_size):
    resized_image = cv2.resize(image_slice, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask_slice.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return resized_image.astype(np.float32), resized_mask.astype(np.uint8)


def choose_core_mask(binary_mask):
    if not np.any(binary_mask):
        return binary_mask
    core = binary_mask.astype(bool)
    last_valid = core
    for _ in range(8):
        eroded = ndimage.binary_erosion(core)
        if not np.any(eroded):
            break
        last_valid = eroded
        core = eroded
    return last_valid


def draw_class_scribble(mask, class_value, scribble):
    class_mask = mask == class_value
    if not np.any(class_mask):
        return
    core_mask = choose_core_mask(class_mask)
    distance = ndimage.distance_transform_edt(core_mask)
    center = np.unravel_index(int(np.argmax(distance)), distance.shape)
    cy, cx = center
    scribble[cy, cx] = class_value

    row_positions = np.where(core_mask[cy])[0]
    col_positions = np.where(core_mask[:, cx])[0]
    if row_positions.size > 0:
        scribble[cy, row_positions] = class_value
    if col_positions.size > 0:
        scribble[col_positions, cx] = class_value


def build_scribble(mask):
    scribble = np.full(mask.shape, IGNORE_INDEX, dtype=np.uint8)
    foreground_mask = mask > 0
    if np.any(foreground_mask):
        distance_to_fg = ndimage.distance_transform_edt(~foreground_mask)
        bg_anchor_mask = distance_to_fg >= max(3.0, 0.03 * min(mask.shape))
        scribble[bg_anchor_mask] = 0
    scribble[0, :] = 0
    scribble[-1, :] = 0
    scribble[:, 0] = 0
    scribble[:, -1] = 0

    for class_value in CLASS_VALUES:
        draw_class_scribble(mask, class_value, scribble)
    return scribble


def iter_labeled_frames(patient_dir: Path):
    for image_path in sorted(patient_dir.glob("*_frame*.nii.gz")):
        if image_path.name.endswith("_gt.nii.gz") or image_path.name.endswith("_4d.nii.gz"):
            continue
        gt_path = image_path.with_name(image_path.name.replace(".nii.gz", "_gt.nii.gz"))
        if gt_path.is_file():
            yield image_path, gt_path


def valid_slice_indices(mask_volume):
    foreground_slices = [idx for idx in range(mask_volume.shape[0]) if np.any(mask_volume[idx] > 0)]
    if not foreground_slices:
        return []
    return list(range(min(foreground_slices), max(foreground_slices) + 1))


def write_case_slices(patient_dir, target_split_dir: Path, img_size: int):
    patient_name = patient_dir.name
    written = 0
    for image_path, gt_path in iter_labeled_frames(patient_dir):
        image_volume = load_nifti_array(image_path)
        mask_volume = load_nifti_array(gt_path).astype(np.uint8)
        for slice_idx in valid_slice_indices(mask_volume):
            image_slice = normalize_slice(image_volume[slice_idx])
            mask_slice = mask_volume[slice_idx]
            image_slice, mask_slice = resize_image_and_mask(image_slice, mask_slice, img_size)
            scribble = build_scribble(mask_slice)
            frame_id = image_path.stem.replace(".nii", "")
            output_path = target_split_dir / f"{patient_name}_{frame_id}_slice{slice_idx:02d}.h5"
            with h5py.File(output_path, "w") as h5f:
                h5f.create_dataset("image", data=image_slice, compression="gzip")
                h5f.create_dataset("mask", data=mask_slice, compression="gzip")
                h5f.create_dataset("scribble", data=scribble, compression="gzip")
            written += 1
    return written


def prepare_split(raw_root: Path, output_root: Path, split_name: str, num_domains: int, img_size: int):
    patient_dirs = sorted([path for path in (raw_root / split_name).iterdir() if path.is_dir() and path.name.startswith("patient")])
    domain_patient_groups = chunk_evenly(patient_dirs, num_domains)
    total_written = 0
    for domain_idx, patient_group in enumerate(domain_patient_groups, start=1):
        split_dir_name = "train" if split_name == "training" else "test"
        target_split_dir = output_root / f"Domain{domain_idx}" / split_dir_name
        target_split_dir.mkdir(parents=True, exist_ok=True)
        for patient_dir in patient_group:
            total_written += write_case_slices(patient_dir, target_split_dir, img_size)
    return total_written


def main():
    args = parse_args()
    if args.output_root.exists():
        existing_h5 = list(args.output_root.glob("Domain*/train/*.h5"))
        if existing_h5 and not args.force:
            raise FileExistsError(
                f"{args.output_root} already contains processed h5 files. Pass --force only after clearing it manually."
            )

    ensure_extracted(args.raw_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_count = prepare_split(args.raw_root, args.output_root, "training", args.num_domains, args.img_size)
    test_count = prepare_split(args.raw_root, args.output_root, "testing", args.num_domains, args.img_size)
    print(f"Prepared ACDC federated h5 dataset at {args.output_root}")
    print(f"train_slices={train_count}")
    print(f"test_slices={test_count}")


if __name__ == "__main__":
    main()
