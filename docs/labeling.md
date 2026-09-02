# Labeling — bootstrap loop

Annotation was done with **[Label Studio][ls]** (HumanSignal, Apache-2.0) running
locally, in its own venv (`.venv-ls`), with a Local Storage connection to
`data/raw/` so nothing is uploaded.

[ls]: https://github.com/HumanSignal/label-studio

The 452 frames were **not** all boxed by hand. The loop:

```
capture (scripts/capture.py)
   │
   ▼
label ~50 frames by hand in Label Studio        (all 3 classes, a few of each scenario)
   │
   ▼
scripts/ls_export.py   → read the finished annotations straight from Label
                          Studio's SQLite, write a YOLO dataset (no API token)
   │
   ▼
train a small yolo11n on those ~50
   │
   ▼
scripts/ls_prelabel.py → run that model over every un-annotated task, write the
                          boxes back into Label Studio as *predictions*
   │
   ▼
Label Studio "Prelabeling" setting → tasks now open with the boxes pre-filled;
you review / fix / submit instead of drawing from scratch
   │
   ▼
scripts/ls_autoaccept.py → for scenarios with a known object set (ok, eksik_*,
                            bos, …) auto-submit predictions whose class multiset
                            matches and whose boxes clear a confidence floor;
                            the ambiguous ones stay for manual review
   │
   ▼
scripts/ls_dedup.py    → drop accidental duplicate boxes (same class, high
                          overlap) that slipped through review
   │
   ▼
retrain on the larger set, repeat until done
   │
   ▼
scripts/build_dataset.py → final YOLO dataset, 70/20/10 split per scenario in
                            capture-timestamp order
```

Roughly: **~50 boxed by hand, ~280 auto-accepted, ~120 reviewed with boxes
pre-filled.** Hand-labeling time was cut by more than half.

## Scripts

| script | what it does |
|--------|--------------|
| `ls_export.py`    | Label Studio SQLite → YOLO dataset (bootstrap; no token) |
| `ls_prelabel.py`  | write model predictions into Label Studio as pre-annotations |
| `ls_autoaccept.py`| auto-submit scenario-consistent, high-confidence predictions |
| `ls_dedup.py`     | remove duplicate same-class boxes from finished annotations |
| `build_dataset.py`| final leak-free train/val/test split |

All of them read/write the local DB at
`.venv-ls/ls-data/label_studio.sqlite3` — Label Studio Community edition, so no
API server or token is involved.

## Running Label Studio

```bash
python -m venv .venv-ls
.venv-ls/Scripts/pip install label-studio
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/data/raw" \
  .venv-ls/Scripts/label-studio start --port 8080
```

Project labeling config (`RectangleLabels` with `tufek` / `tabanca` / `sarjor`),
then add a **Local files** source storage pointed at `data/raw/2026-09-01` and
sync.
