"""Compute static annotation-quality oracle scores for each client.

For each client, compute the average full-image Dice between a weak
annotation mask and the ground-truth full mask across training samples.
This is intentionally stricter than the previous "labeled-pixels-only"
definition: unlabeled regions are treated as non-foreground, so sparse
scribbles/keypoints receive lower scores than dense annotations.

Output can be directly pasted into --oracle_global_agg_static_scores.

Usage:
    python compute_static_oracle_scores.py [root_path]
    # default root_path: ../data/FAZ_h5
"""
import glob
import sys

import h5py
import numpy as np

CLIENT_SUP_TYPES = {
    0: 'scribble_noisy',  # Domain1 / client1
    1: 'keypoint',        # Domain2 / client2
    2: 'block',           # Domain3 / client3
    3: 'box',             # Domain4 / client4
    4: 'scribble',        # Domain5 / client5
}


UNLABELED_VALUE = 2


def dice_binary(pred, gt):
    intersection = np.sum(pred * gt)
    denom = np.sum(pred) + np.sum(gt)
    if denom == 0:
        return 1.0 if np.sum(gt) == 0 else 0.0
    return 2.0 * intersection / denom


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '../data/FAZ_h5'
    scores = []

    for cid in range(5):
        domain = 'Domain{}'.format(cid + 1)
        label_key = CLIENT_SUP_TYPES[cid]
        pattern = '{}/{}/train/*.h5'.format(root, domain)
        files = sorted(glob.glob(pattern))
        if not files:
            print('WARNING: no files found for pattern {}'.format(pattern))
            scores.append(0.0)
            continue

        dices = []
        for f in files:
            with h5py.File(f, 'r') as h:
                gt = (np.array(h['mask']) > 0).astype(np.float32)
                ann = np.array(h[label_key])
                # FAZ weak labels use {0: bg, 1: fg, 2: unlabeled}. For the
                # static oracle we want a full-image quality score, not a
                # labeled-only agreement score. Therefore, only pixels
                # explicitly marked as foreground count as prediction.
                ann_binary = (ann == 1).astype(np.float32)
                if np.all(ann == UNLABELED_VALUE):
                    continue
                dices.append(dice_binary(ann_binary, gt))

        score = float(np.mean(dices)) if dices else 0.0
        scores.append(score)
        print('Client {} ({:>15s}): {:.4f}  ({} samples)'.format(
            cid, label_key, score, len(dices)))

    print()
    print('--oracle_global_agg_static_scores "{}"'.format(
        ','.join('{:.4f}'.format(s) for s in scores)))


if __name__ == '__main__':
    main()
