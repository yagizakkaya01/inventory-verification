"""Render the per-scenario verdict grid used in the README
(`docs/images/verdicts.jpg`): detector boxes + the state layer's verdict banner,
one representative frame per scenario.

    python -m scripts.render_examples
"""

from __future__ import annotations

import glob
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import Detection
from src.state.temporal_smoothing import SceneReading
from src.state.verifier import Verifier
from src.utils.config import PipelineConfig

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "images" / "verdicts.jpg"
COLORS = {"tufek": (80, 80, 255), "tabanca": (255, 160, 60), "sarjor": (90, 220, 120)}

# one frame per scenario — index picks a mid-run frame, not the first
PICKS = [
    ("ok", 10),
    ("eksik_sarjor", 8),
    ("yanlis_sira", 20),
    ("bos", 4),
]


def render(model, verifier, path: str):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    res = model.predict(img, conf=0.30, verbose=False)[0]
    dets = [
        Detection(res.names[int(b.cls[0])], float(b.conf[0]),
                  tuple(float(v) for v in b.xyxy[0].tolist()))
        for b in res.boxes
    ]
    verdict = verifier.check(SceneReading.from_detections(dets))

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        c = COLORS[d.cls_name]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        cv2.rectangle(img, (x1, y1 - 26), (x1 + len(d.cls_name) * 14 + 64, y1), c, -1)
        cv2.putText(img, f"{d.cls_name} {d.confidence:.2f}", (x1 + 4, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)

    ok = verdict.verdict.value == "OK"
    cv2.rectangle(img, (0, 0), (w, 64), (40, 160, 40) if ok else (30, 30, 200), -1)
    cv2.putText(img, verdict.verdict.value, (16, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    if verdict.detail:
        cv2.putText(img, verdict.detail[:58], (16, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main() -> None:
    from ultralytics import YOLO

    cfg = PipelineConfig.load("configs/pipeline.yaml")
    model = YOLO(cfg.detection.weights)
    verifier = Verifier(cfg.expected.sequence)

    tiles = []
    for folder, idx in PICKS:
        files = sorted(glob.glob(f"data/raw/2026-09-01/{folder}/*.jpg"))
        tiles.append(cv2.resize(render(model, verifier, files[idx]), (640, 360)))

    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
