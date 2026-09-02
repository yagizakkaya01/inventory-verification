# Dataset

YOLO-format object-detection dataset for the detector. Not committed to git
(images are ~75 MB) — download it from the **[dataset-v1 release][rel]** and
unpack into `data/datasets/inventory/`:

```
data/datasets/inventory/
  images/{train,val,test}/*.jpg
  labels/{train,val,test}/*.txt
  data.yaml
```

[rel]: https://github.com/yagizakkaya01/inventory-verification/releases/tag/dataset-v1

## Contents

| | |
|---|---|
| Classes | `0 tufek` (rifle) · `1 tabanca` (pistol) · `2 sarjor` (standalone magazine) |
| Images | 452 — 317 train / 90 val / 45 test |
| Split | per scenario, in capture-timestamp order, so continuous-capture bursts don't leak across splits (`scripts/build_dataset.py`) |
| Negatives | 14 empty frames with empty label files |

Capture: fixed top-down OAK-D over a white sheet, replica objects, via
`scripts/capture.py`. Labelling: **[Label Studio][ls]** — ~50 frames by hand,
the rest model-assisted (`scripts/ls_prelabel.py` / `ls_autoaccept.py`) then
reviewed. Full workflow in [labeling.md](labeling.md).

[ls]: https://github.com/HumanSignal/label-studio

The rifle's **attached** magazine is intentionally not labelled `sarjor`;
`sarjor` is only the standalone magazine the system tracks as its own item.

## Rebuilding

If the Label Studio annotations change, regenerate with:

```bash
python -m scripts.build_dataset
```

## Trained model

`inventory-yolo11s` (200 epochs) — test-set mAP@50 **0.94**
(tabanca 0.99, sarjor 0.92, tufek 0.90). Weights are in the same release.
