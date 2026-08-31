"""End-to-end smoke test: load pretrained YOLOv8n and run one inference.

Confirms ultralytics + torch + opencv are wired up before we touch real data.
Downloads yolov8n.pt (~6 MB) on first run.
"""

from __future__ import annotations

import numpy as np


def main() -> int:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy, verbose=False)
    print(f"OK — model loaded, ran inference, {len(results[0].boxes)} boxes on a blank frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
