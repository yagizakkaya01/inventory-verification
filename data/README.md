# data/

Everything here except this file is gitignored — images/labels are managed in
Roboflow and pulled down, not committed.

```
raw/         original captures, untouched (organize by session: raw/2026-09-01/)
interim/     cleaned / deduplicated / renamed, pre-labeling
datasets/    YOLO-format export, ready for training
  inventory/
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml            # (or use the repo's configs/data.yaml)
```

## Workflow

1. Capture into `raw/<date>/`.
2. Upload to Roboflow, label (model-assisted once the 50-image model exists).
3. Export as **YOLOv8** format, 70/20/10 train/val/test, into
   `datasets/inventory/`.
4. Keep `configs/data.yaml` pointing at it.

## Split discipline

Split by **capture session / arrangement**, not by random frame — frames from
one continuous recording are near-duplicates and will leak between train and
val, inflating mAP.
