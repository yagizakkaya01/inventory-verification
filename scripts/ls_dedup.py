"""Remove accidental duplicate boxes from finished Label Studio annotations:
when two boxes of the *same* class overlap (IoU > --iou), keep the larger one
and drop the rest. Writes back to the LS SQLite.

    # stop Label Studio first
    python -m scripts.ls_dedup --dry-run
    python -m scripts.ls_dedup
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / ".venv-ls" / "ls-data" / "label_studio.sqlite3"


def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def dedup(regions: list, thr: float) -> list:
    keep, dropped = [], set()
    for i, r in enumerate(regions):
        if i in dropped:
            continue
        for j in range(i + 1, len(regions)):
            if j in dropped:
                continue
            a, b = r["value"], regions[j]["value"]
            if a["rectanglelabels"] == b["rectanglelabels"] and iou(a, b) > thr:
                area_i = a["width"] * a["height"]
                area_j = b["width"] * b["height"]
                dropped.add(j if area_j <= area_i else i)
        if i not in dropped:
            keep.append(r)
    return [r for k, r in enumerate(regions) if k not in dropped]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=int, default=1)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        "select id, result from task_completion where was_cancelled = 0 and project_id = ?",
        (args.project,),
    ).fetchall()

    changed = removed = 0
    for r in rows:
        regions = json.loads(r["result"])
        cleaned = dedup(regions, args.iou)
        if len(cleaned) != len(regions):
            changed += 1
            removed += len(regions) - len(cleaned)
            if not args.dry_run:
                db.execute(
                    "update task_completion set result = ?, result_count = ? where id = ?",
                    (json.dumps(cleaned), len(cleaned), r["id"]),
                )
    if not args.dry_run:
        db.commit()

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}{changed} annotation düzeltildi, "
          f"{removed} fazla kutu silindi")


if __name__ == "__main__":
    main()
