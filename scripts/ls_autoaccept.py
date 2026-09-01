"""Turn confident, scenario-consistent detector predictions into *submitted*
annotations, so only the ambiguous tasks are left for a human.

A prediction is accepted only when, for that task's scenario folder, the set of
predicted classes exactly matches what the scenario must contain AND every box
clears --min-conf. Everything else is left untouched for manual review.

    # stop Label Studio first
    python -m scripts.ls_autoaccept --min-conf 0.55 --dry-run
    python -m scripts.ls_autoaccept --min-conf 0.55

Writes straight to the LS SQLite (Community edition doesn't use the FSM tables).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / ".venv-ls" / "ls-data" / "label_studio.sqlite3"
RAW = REPO / "data" / "raw"
CLASSES = ["tufek", "tabanca", "sarjor"]
USER_ID = 1

# exact class multiset each scenario folder must contain
EXPECT: dict[str, Counter] = {
    "ok":            Counter({"tufek": 1, "tabanca": 1, "sarjor": 1}),
    "yanlis_sira":   Counter({"tufek": 1, "tabanca": 1, "sarjor": 1}),
    "eksik_tufek":   Counter({"tabanca": 1, "sarjor": 1}),
    "eksik_tabanca": Counter({"tufek": 1, "sarjor": 1}),
    "eksik_sarjor":  Counter({"tufek": 1, "tabanca": 1}),
    "bos":           Counter(),
}
SINGLE_ANY = {"tekil_sarjor", "unlabeled"}     # exactly one box, any class
NEVER = {"occlusion", "yanlis_kombinasyon", "eksik_malzeme"}   # always manual


def resolve(data_json: str) -> Path | None:
    url = next(iter(json.loads(data_json).values()))
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    p = RAW / urllib.parse.unquote(q["d"][0]).replace("\\", "/")
    return p if p.exists() else None


def regions_from(res, min_conf: float):
    """Ultralytics result -> (LS regions, class Counter, min conf seen)."""
    regions, cnt, confs = [], Counter(), []
    h, w = res.orig_shape
    for b in res.boxes:
        conf = float(b.conf[0])
        if conf < min_conf:
            continue
        cls = CLASSES[int(b.cls[0])]
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        regions.append({
            "original_width": w, "original_height": h, "image_rotation": 0,
            "value": {"x": x1 / w * 100, "y": y1 / h * 100,
                      "width": (x2 - x1) / w * 100, "height": (y2 - y1) / h * 100,
                      "rotation": 0, "rectanglelabels": [cls]},
            "id": uuid.uuid4().hex[:10],
            "from_name": "label", "to_name": "image",
            "type": "rectanglelabels", "origin": "prediction",
        })
        cnt[cls] += 1
        confs.append(conf)
    return regions, cnt, (min(confs) if confs else 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="models/pretrained/bootstrap-yolo11n.pt")
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    tasks = db.execute(
        "select id, data from task where project_id = ? and total_annotations = 0",
        (args.project,),
    ).fetchall()
    model = YOLO(args.weights)
    now = datetime.now(timezone.utc).isoformat(sep=" ")

    accepted = skipped = 0
    reasons: Counter = Counter()
    for t in tasks:
        img = resolve(t["data"])
        if img is None:
            skipped += 1; reasons["görsel yok"] += 1; continue
        folder = img.parent.name
        if folder in NEVER:
            skipped += 1; reasons[f"manuel: {folder}"] += 1; continue

        res = model.predict(str(img), conf=0.4, verbose=False)[0]
        regions, cnt, _ = regions_from(res, args.min_conf)

        ok = False
        if folder in EXPECT:
            ok = cnt == EXPECT[folder]
        elif folder in SINGLE_ANY:
            ok = sum(cnt.values()) == 1
        if not ok:
            skipped += 1; reasons[f"eşleşmedi: {folder}"] += 1; continue

        accepted += 1
        if args.dry_run:
            continue
        db.execute(
            """insert into task_completion
               (result, was_cancelled, created_at, updated_at, task_id, prediction,
                lead_time, result_count, completed_by_id, ground_truth, project_id,
                updated_by_id, unique_id, bulk_created)
               values (?, 0, ?, ?, ?, '{}', 0, ?, ?, 0, ?, ?, ?, 1)""",
            (json.dumps(regions), now, now, t["id"], len(regions),
             USER_ID, args.project, USER_ID, uuid.uuid4().hex),
        )
        db.execute(
            "update task set total_annotations = 1, is_labeled = 1, updated_by_id = ?, "
            "updated_at = ? where id = ?",
            (USER_ID, now, t["id"]),
        )

    if not args.dry_run:
        db.commit()

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}otomatik onaylanan: {accepted}")
    print(f"manuel review'a kalan: {skipped}")
    for r, n in reasons.most_common():
        print(f"  {n:4}  {r}")


if __name__ == "__main__":
    main()
