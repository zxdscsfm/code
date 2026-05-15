# External Data and Model Artifacts

This `final` branch is a runnable code snapshot of `/data/jianbingshen/yanghongji/FedLPPA`.

Large runtime artifacts are intentionally not committed to GitHub:

- `model/`: training outputs and checkpoints, about 101G on the server.
- `code_v4/logs/`: training logs, about 420M on the server.
- `data/`: datasets or dataset links, expected to be provided externally.

On the server where this snapshot was prepared, the original dataset link was:

```text
/data/jianbingshen/yanghongji/FedLPPA/data/ODOC_h5 -> /data/jianbingshen/yanghongji/FedLPPA_github_official_clean/data/ODOC_h5
```

To run after cloning, recreate the required dataset directories under `data/` or update the `--root_path` arguments in the run scripts under `code_v4/`.
