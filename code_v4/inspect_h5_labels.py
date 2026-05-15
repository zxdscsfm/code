"""Quick inspection of h5 label value distributions for each annotation type."""
import glob
import sys

import h5py
import numpy as np

CLIENT_SUP_TYPES = {
    0: 'scribble_noisy',
    1: 'keypoint',
    2: 'block',
    3: 'box',
    4: 'scribble',
}

root = sys.argv[1] if len(sys.argv) > 1 else '../data/FAZ_h5'

for cid in range(5):
    domain = 'Domain{}'.format(cid + 1)
    label_key = CLIENT_SUP_TYPES[cid]
    files = sorted(glob.glob('{}/{}/train/*.h5'.format(root, domain)))
    if not files:
        print('No files for {}'.format(domain))
        continue

    # Check first 3 files
    print('=== Client {} ({}) - {} files ==='.format(cid, label_key, len(files)))
    for f in files[:3]:
        with h5py.File(f, 'r') as h:
            print('  Keys:', list(h.keys()))
            mask = np.array(h['mask'])
            ann = np.array(h[label_key])
            print('  mask  shape={} dtype={} unique={}'.format(mask.shape, mask.dtype, np.unique(mask).tolist()))
            print('  {}  shape={} dtype={} unique={}'.format(label_key, ann.shape, ann.dtype, np.unique(ann).tolist()))
            # Compare
            total = mask.size
            agree = np.sum(mask == ann)
            print('  agree={}/{} ({:.1f}%)  mask_fg={} ann_fg={}'.format(
                agree, total, 100.0 * agree / total,
                np.sum(mask > 0), np.sum(ann > 0)))
    print()
