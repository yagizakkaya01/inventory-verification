"""Build the final YOLO dataset from all finished Label Studio annotations.

Split is 70/20/10 train/val/test, done *per scenario folder and in timestamp
order* — consecutive burst frames from continuous capture stay in the same
split, so near-duplicates don't leak between train and val/test.

    python -m scripts.build_dataset
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import urllib.parse
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / ".venv-ls" / "ls-data" / "label_studio.sqlite3"
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "datasets" / "inventory"
CLASSES = ["tufek", "tabanca", "sarjor"]
SPLITS = (("train", 0.70), ("val", 0.20), ("test", 0.10))


def rel_path(data_json: str) -> str:
    url = next(iter(json.loads(data_json).values()))
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    return urllib.parse.unquote(q["d"][0]).replace("\\", "/")


def main() -> None:
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """select tc.result, t.data
           from task_completion tc join task t on t.id = tc.task_id
           where tc.was_cancelled = 0 and tc.project_id = 1"""
    ).fetchall()

    # group by scenario folder
    by_folder: dict[str, list[tuple[Path, list]]] = defaultdict(list)
    for r in rows:
        rel = rel_path(r["data"])
        img = RAW / rel
        if not img.exists():
            continue
        labels = []
        for it in json.loads(r["result"]):
            v = it["value"]
            names = v.get("rectanglelabels") or []
            if not names:
                continue
            cid = CLASSES.index(names[0])
            xc = (v["x"] + v["width"] / 2) / 100
            yc = (v["y"] + v["height"] / 2) / 100
            labels.append((cid, xc, yc, v["width"] / 100, v["height"] / 100))
        by_folder[img.parent.name].append((img, labels))

    if OUT.exists():
        shutil.rmtree(OUT)
    for s, _ in SPLITS:
        (OUT / "images" / s).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / s).mkdir(parents=True, exist_ok=True)

    counts = {s: 0 for s, _ in SPLITS}
    for folder, items in by_folder.items():
        items.sort(key=lambda t: t[0].name)          # timestamp order
        n = len(items)
        i_tr = round(n * SPLITS[0][1])
        i_va = i_tr + round(n * SPLITS[1][1])
        chunks = {"train": items[:i_tr], "val": items[i_tr:i_va], "test": items[i_va:]}
        for s, chunk in chunks.items():
            for img, labels in chunk:
                stem = f"{folder}__{img.stem}"
                shutil.copy2(img, OUT / "images" / s / f"{stem}.jpg")
                (OUT / "labels" / s / f"{stem}.txt").write_text(
                    "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n"
                            for c, a, b, w, h in labels)
                )
                counts[s] += 1

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    print("split:", counts, " toplam:", sum(counts.values()))


if __name__ == "__main__":
    main()
