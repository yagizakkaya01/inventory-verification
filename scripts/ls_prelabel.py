"""Run the bootstrap detector over every not-yet-annotated Label Studio task and
write the boxes back as *predictions* (pre-annotations) straight into the LS
SQLite DB. The labeler then just reviews / corrects them.

    # stop Label Studio first, then:
    python -m scripts.ls_prelabel --weights models/checkpoints/bootstrap/weights/best.pt

Re-running replaces predictions from the same model_version.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / ".venv-ls" / "ls-data" / "label_studio.sqlite3"
RAW = REPO / "data" / "raw"
CLASSES = ["tufek", "tabanca", "sarjor"]


def resolve_image(data_json: str) -> Path | None:
    d = json.loads(data_json)
    url = next(iter(d.values()))
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    rel = urllib.parse.unquote(q["d"][0]).replace("\\", "/")
    p = RAW / rel
    return p if p.exists() else None


def to_ls_result(res) -> tuple[list, float]:
    """Ultralytics Result -> Label Studio rectanglelabels regions (percent coords)."""
    regions, scores = [], []
    h, w = res.orig_shape
    for box in res.boxes:
        cls = CLASSES[int(box.cls[0])]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        regions.append({
            "type": "rectanglelabels",
            "from_name": "label", "to_name": "image",
            "original_width": w, "original_height": h,
            "image_rotation": 0,
            "value": {
                "x": x1 / w * 100, "y": y1 / h * 100,
                "width": (x2 - x1) / w * 100, "height": (y2 - y1) / h * 100,
                "rotation": 0, "rectanglelabels": [cls],
            },
            "score": conf,
        })
        scores.append(conf)
    return regions, (sum(scores) / len(scores) if scores else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/checkpoints/bootstrap/weights/best.pt")
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--model-version", default="bootstrap-yolo11n")
    ap.add_argument("--include-annotated", action="store_true",
                    help="also predict on tasks that already have an annotation")
    args = ap.parse_args()

    from ultralytics import YOLO

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row

    cond = "" if args.include_annotated else "and t.total_annotations = 0"
    tasks = db.execute(
        f"select t.id, t.data from task t where t.project_id = ? {cond}",
        (args.project,),
    ).fetchall()
    print(f"{len(tasks)} task için tahmin üretilecek")

    model = YOLO(args.weights)
    now = datetime.now(timezone.utc).isoformat()

    # eski aynı-sürüm tahminleri temizle
    db.execute("delete from prediction where project_id = ? and model_version = ?",
               (args.project, args.model_version))

    written = 0
    empty = 0
    for i, t in enumerate(tasks, 1):
        img = resolve_image(t["data"])
        if img is None:
            continue
        res = model.predict(str(img), conf=args.conf, verbose=False)[0]
        regions, score = to_ls_result(res)
        if not regions:
            empty += 1
        db.execute(
            """insert into prediction
               (result, score, model_version, created_at, updated_at,
                task_id, project_id, mislabeling)
               values (?, ?, ?, ?, ?, ?, ?, 0.0)""",
            (json.dumps(regions), score, args.model_version, now, now,
             t["id"], args.project),
        )
        written += 1
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}")

    # LS'in sayaçlarını güncelle
    db.execute(
        """update task set total_predictions = (
               select count(*) from prediction p where p.task_id = task.id
           ) where project_id = ?""",
        (args.project,),
    )
    db.commit()
    print(f"{written} tahmin yazıldı ({empty} tanesi boş — model nesne bulamadı)")
    print("Label Studio'yu yeniden başlat, task'larda ön-etiketler görünecek.")


if __name__ == "__main__":
    main()
