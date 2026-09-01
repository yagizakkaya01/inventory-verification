"""Read finished annotations straight from the Label Studio SQLite DB and write
a YOLO-format dataset. No API token needed — reads the local .sqlite3.

    python -m scripts.ls_export --project 1 --val-frac 0.2

Output:
    data/datasets/inventory/
        images/{train,val}/*.jpg   (copied from data/raw)
        labels/{train,val}/*.txt
        data.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sqlite3
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / ".venv-ls" / "ls-data" / "label_studio.sqlite3"
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "datasets" / "inventory"
CLASSES = ["tufek", "tabanca", "sarjor"]


def resolve_image(data_json: str) -> Path | None:
    """LS task.data -> local file path under data/raw."""
    d = json.loads(data_json)
    url = next(iter(d.values()))                      # key is '$undefined$'
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    rel = urllib.parse.unquote(q["d"][0]).replace("\\", "/")
    p = RAW / rel
    return p if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """select tc.result, t.data
           from task_completion tc join task t on t.id = tc.task_id
           where tc.was_cancelled = 0 and tc.project_id = ?""",
        (args.project,),
    ).fetchall()

    samples = []          # (image_path, [ (cls_id, xc, yc, w, h) normalized ])
    skipped = 0
    for r in rows:
        img = resolve_image(r["data"])
        if img is None:
            skipped += 1
            continue
        labels = []
        for item in json.loads(r["result"]):
            v = item.get("value", {})
            names = v.get("rectanglelabels") or []
            if not names:
                continue
            cid = CLASSES.index(names[0])
            # LS gives x,y,w,h as % of image, top-left origin
            xc = (v["x"] + v["width"] / 2) / 100
            yc = (v["y"] + v["height"] / 2) / 100
            labels.append((cid, xc, yc, v["width"] / 100, v["height"] / 100))
        samples.append((img, labels))          # keep empties (negative images)

    if skipped:
        print(f"uyarı: {skipped} annotation'ın görseli bulunamadı, atlandı")

    random.Random(args.seed).shuffle(samples)
    n_val = max(1, round(len(samples) * args.val_frac))
    split = {"val": samples[:n_val], "train": samples[n_val:]}

    if OUT.exists():
        shutil.rmtree(OUT)
    for s in ("train", "val"):
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

    for s, items in split.items():
        for img, labels in items:
            stem = f"{img.parent.name}__{img.stem}"       # keep it unique
            shutil.copy2(img, OUT / "images" / s / f"{stem}.jpg")
            txt = OUT / "labels" / s / f"{stem}.txt"
            txt.write_text("".join(
                f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in labels
            ))

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    print(f"train {len(split['train'])}  val {len(split['val'])}  -> {OUT}")


if __name__ == "__main__":
    main()
