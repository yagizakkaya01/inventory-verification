# data/

Gitignored (images are large). See **[docs/dataset.md](../docs/dataset.md)** for
the download link and details.

```
raw/         original captures, by session (raw/2026-09-01/<scenario>/)
interim/     scratch space, pre-labeling
datasets/    YOLO-format dataset, ready for training
  inventory/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml
```

## Workflow

1. Capture into `raw/<date>/<scenario>/` with `scripts/capture.py`.
2. Label in Label Studio (bootstrap ~50 by hand → `scripts/ls_prelabel.py` /
   `ls_autoaccept.py` for the rest → review).
3. `python -m scripts.build_dataset` → writes `datasets/inventory/`.
4. `configs/data.yaml` already points at it.

## Split discipline

`build_dataset.py` splits **per scenario, in capture-timestamp order** — never
random. Frames from one continuous burst are near-duplicates; a random split
leaks them between train and val/test and inflates mAP.
