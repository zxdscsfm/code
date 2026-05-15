#!/usr/bin/env python
"""
Offline diagnostic: compute annotation geometry bin distributions for each client.
Validates whether geometry bins produce structural cross-client differences.

Usage (on cluster):
    python analyze_geometry_bins.py --root_path ../data/FAZ_h5 --img_class faz
    python analyze_geometry_bins.py --root_path ../data/Polyp_h5 --img_class polyp
    python analyze_geometry_bins.py --root_path ../data/Polyp_h5 --img_class polyp --binning_mode rule
"""
import os
import argparse
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
# FAZ: 5 clients, each with different weak annotation type
CLIENT_SUP_TYPES = {
    'faz': {
        'client1': 'scribble_noisy',
        'client2': 'keypoint',
        'client3': 'block',
        'client4': 'box',
        'client5': 'scribble',
    },
    'polyp': {
        'client1': 'keypoint',
        'client2': 'scribble',
        'client3': 'box',
        'client4': 'block',
    },
}

DOMAIN_MAP = {
    'client1': 'Domain1',
    'client2': 'Domain2',
    'client3': 'Domain3',
    'client4': 'Domain4',
    'client5': 'Domain5',
}


def infer_unlabeled_value(img_class):
    """Return the integer value used for unlabeled pixels."""
    if img_class == 'odoc':
        return 3
    return 2  # faz, polyp


def compute_geometry_for_sample(label, img_class):
    """
    Compute annotation geometry features for a single sample.

    Returns:
        d_fg:    distance to nearest foreground-labeled pixel (H, W)
        d_bg:    distance to nearest background-labeled pixel (H, W)
        is_labeled: binary mask of labeled pixels (H, W)
        local_density: fraction of labeled pixels in 15x15 neighborhood (H, W)
        geometry_support: combined support score (H, W)
    """
    unlabeled_val = infer_unlabeled_value(img_class)

    fg_mask = (label == 1).astype(np.float64)
    bg_mask = (label == 0).astype(np.float64)
    is_labeled = (label != unlabeled_val).astype(np.float64)

    # Distance transforms (distance to nearest labeled pixel of each type)
    if fg_mask.any():
        d_fg = distance_transform_edt(1 - fg_mask)
    else:
        d_fg = np.full_like(label, fill_value=999.0, dtype=np.float64)

    if bg_mask.any():
        d_bg = distance_transform_edt(1 - bg_mask)
    else:
        d_bg = np.full_like(label, fill_value=999.0, dtype=np.float64)

    # Local label density: average of is_labeled in 15x15 window
    from scipy.ndimage import uniform_filter
    local_density = uniform_filter(is_labeled, size=15, mode='constant', cval=0.0)

    # Geometry support: high when close to fg annotation AND local density is high
    tau = 10.0  # distance scale
    geo_support = np.exp(-d_fg / tau) * local_density
    geo_support = np.clip(geo_support, 0.0, 1.0)

    return d_fg, d_bg, is_labeled, local_density, geo_support


def bin_geometry_support(geo_support, n_bins=4):
    """
    Assign each pixel to a geometry bin based on geo_support value.
    Returns bin edges and pixel fraction in each bin.
    """
    edges = np.linspace(0, 1, n_bins + 1)  # [0, 0.25, 0.5, 0.75, 1.0]
    total_pixels = geo_support.size
    bin_fractions = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            count = np.sum((geo_support >= lo) & (geo_support <= hi))
        else:
            count = np.sum((geo_support >= lo) & (geo_support < hi))
        bin_fractions.append(count / total_pixels)
    return bin_fractions, edges


def bin_geometry_rule(d_fg, d_bg, is_labeled, near_radius=8.0, mid_radius=24.0):
    """
    Rule-based geometry bins that avoid early scalar collapse.

    Bins:
        0: directly labeled pixels
        1: unlabeled but near any labeled pixel
        2: unlabeled and mid-range from any labeled pixel
        3: unlabeled and far from all labeled pixels
    """
    min_d = np.minimum(d_fg, d_bg)
    labeled_mask = is_labeled > 0.5
    near_mask = (~labeled_mask) & (min_d <= near_radius)
    mid_mask = (~labeled_mask) & (min_d > near_radius) & (min_d <= mid_radius)
    far_mask = (~labeled_mask) & (min_d > mid_radius)

    masks = [labeled_mask, near_mask, mid_mask, far_mask]
    total_pixels = float(is_labeled.size)
    bin_fractions = [float(mask.sum() / total_pixels) for mask in masks]
    bin_names = [
        'direct_labeled',
        f'near_any_label(d<={near_radius:g})',
        f'mid_range({near_radius:g}<d<={mid_radius:g})',
        f'far_from_label(d>{mid_radius:g})',
    ]
    return bin_fractions, bin_names


def get_bin_fractions(d_fg, d_bg, is_labeled, geo_support, args):
    if args.binning_mode == 'scalar':
        bin_fracs, edges = bin_geometry_support(geo_support, args.n_bins)
        bin_names = []
        for i in range(args.n_bins):
            left = edges[i]
            right = edges[i + 1]
            right_bracket = ']' if i == args.n_bins - 1 else ')'
            bin_names.append(f'[{left:.2f},{right:.2f}{right_bracket}')
        return bin_fracs, bin_names

    if args.n_bins != 4:
        raise ValueError('rule binning mode currently requires --n_bins 4')

    return bin_geometry_rule(
        d_fg,
        d_bg,
        is_labeled,
        near_radius=args.near_radius,
        mid_radius=args.mid_radius,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--img_class', type=str, required=True, choices=['faz', 'polyp'])
    parser.add_argument('--n_bins', type=int, default=4)
    parser.add_argument('--binning_mode', type=str, default='rule', choices=['rule', 'scalar'])
    parser.add_argument('--near_radius', type=float, default=8.0)
    parser.add_argument('--mid_radius', type=float, default=24.0)
    parser.add_argument('--output_dir', type=str, default='geometry_analysis')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    clients = CLIENT_SUP_TYPES[args.img_class]
    all_results = []

    for client_name, sup_type in clients.items():
        domain = DOMAIN_MAP[client_name]
        train_dir = os.path.join(args.root_path, domain, 'train')

        if not os.path.isdir(train_dir):
            print(f"WARNING: {train_dir} not found, skipping {client_name}")
            continue

        h5_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.h5')])
        print(f"\n{client_name} ({sup_type}): {len(h5_files)} samples")

        client_bin_fractions = []
        client_d_fg_means = []
        client_d_bg_means = []
        client_labeled_ratios = []

        for h5_name in h5_files:
            h5_path = os.path.join(train_dir, h5_name)
            with h5py.File(h5_path, 'r') as h5f:
                label = h5f[sup_type][:]

            d_fg, d_bg, is_labeled, local_density, geo_support = compute_geometry_for_sample(
                label, args.img_class
            )
            bin_fracs, bin_names = get_bin_fractions(d_fg, d_bg, is_labeled, geo_support, args)

            client_bin_fractions.append(bin_fracs)
            client_d_fg_means.append(d_fg.mean())
            client_d_bg_means.append(d_bg.mean())
            client_labeled_ratios.append(is_labeled.mean())

        # Aggregate across samples
        mean_bin_fracs = np.mean(client_bin_fractions, axis=0)
        std_bin_fracs = np.std(client_bin_fractions, axis=0)

        print(f"  labeled_ratio: {np.mean(client_labeled_ratios):.4f}")
        print(f"  mean d_fg: {np.mean(client_d_fg_means):.2f}")
        print(f"  mean d_bg: {np.mean(client_d_bg_means):.2f}")
        print(f"  binning_mode: {args.binning_mode}")
        if args.binning_mode == 'rule':
            print(f"  near_radius: {args.near_radius:.1f}, mid_radius: {args.mid_radius:.1f}")
        print(f"  bin fractions (mean): {[f'{x:.4f}' for x in mean_bin_fracs]}")
        print(f"  bin fractions (std):  {[f'{x:.4f}' for x in std_bin_fracs]}")

        for i in range(args.n_bins):
            all_results.append({
                'client': client_name,
                'sup_type': sup_type,
                'binning_mode': args.binning_mode,
                'n_samples': len(h5_files),
                'labeled_ratio': np.mean(client_labeled_ratios),
                'mean_d_fg': np.mean(client_d_fg_means),
                'mean_d_bg': np.mean(client_d_bg_means),
                'bin': i,
                'bin_name': bin_names[i],
                'bin_frac_mean': mean_bin_fracs[i],
                'bin_frac_std': std_bin_fracs[i],
            })

    df = pd.DataFrame(all_results)
    out_path = os.path.join(args.output_dir, f'{args.img_class}_geometry_bins_{args.binning_mode}.csv')
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary table: bin fractions per client (the key diagnostic)
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTIC: Bin fraction distribution per client")
    print("=" * 70)
    pivot = df.pivot_table(
        index=['client', 'sup_type'],
        columns='bin',
        values='bin_frac_mean',
    )
    bin_name_lookup = df.drop_duplicates('bin').sort_values('bin')[['bin', 'bin_name']]
    pivot.columns = [
        f"bin_{int(row['bin'])} {row['bin_name']}"
        for _, row in bin_name_lookup.iterrows()
    ]
    print(pivot.to_string())

    # Cross-client std per bin
    print("\n" + "=" * 70)
    print("Cross-client std per bin (higher = more differentiation)")
    print("=" * 70)
    for i in range(args.n_bins):
        bin_vals = df[df['bin'] == i]['bin_frac_mean'].values
        print(f"  Bin {i}: std = {bin_vals.std():.4f}  (values: {[f'{x:.4f}' for x in bin_vals]})")


if __name__ == '__main__':
    main()
