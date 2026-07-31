# AnnoEvent Code Package

This branch contains the reproducibility code package for AnnoEvent.

The package is intentionally scoped to the paper implementation: training entry points, the federated U-Net backbone path, weak-label data loading, WAED evidence construction, AGC context admission, NWR event-risk losses, and evaluation utilities. It does not include datasets, checkpoints, private server paths, historical sweeps, or unrelated exploratory scripts.

## Contents

- `code_v4/flower_pCE_2D_v4_FedLPPA.py`: main Flower training entry.
- `code_v4/weak_annotation_reliability.py`: WAED evidence-map and weak-support construction.
- `code_v4/annotation_geometry_calibration.py`: AGC/NWR event-risk implementation.
- `code_v4/dataloaders/`: HDF5 dataset loader and weak-label transforms.
- `code_v4/networks/`, `code_v4/utils/`: model and loss/metric utilities used by the training entry.
- `code_v4/run_annocal_agc_mech_r500_l10.sh`: configurable example launcher for the paper protocol.
- `fedlppa.yaml`: conda environment specification.

## Data

Datasets are not redistributed here. Obtain the public datasets from their original providers and organize the processed HDF5 folders according to the paper protocol. The launcher expects `ROOT_PATH` to point to the selected dataset root.

## Environment

The reported runs used Python 3.9, PyTorch 1.10.2, CUDA 11.3, Flower 1.0.0, and MedPy 0.4.0. A compatible conda environment can be created from:

```bash
conda env create -f fedlppa.yaml
conda activate fed39v2
```

If the optional tree-filter CUDA extension is unavailable, install or build it following the `utils/TreeEnergyLoss` instructions used by the inherited backbone code.

## Example Run

Set local paths with environment variables instead of editing the script:

```bash
cd code_v4
export CONDA_PATH="$HOME/anaconda3"
export CONDA_ENV="fed39v2"
export REPO_ROOT="/path/to/this/repository"
export DATASET="prostate"
export ROOT_PATH="/path/to/PROSTATE_h5_protocol"
export SERVER_ADDRESS="127.0.0.1:8851"
bash run_annocal_agc_mech_r500_l10.sh
```

Launch one server process and the corresponding client processes by setting the role/client arguments as needed, or adapt the generated command block in the launcher for the local cluster environment.

## Notes

The code keeps the paper setting fixed: 5000 local updates, validation every 10 updates, point/scribble/box weak protocols, and no dense masks during local training. Validation masks are used for checkpoint selection, and test masks are used only for final evaluation.

Some legacy HDF5 protocol fields use `keypoint` for the point protocol and `block` for the box protocol. The implementation keeps these aliases for compatibility with existing prepared weak-label files.
