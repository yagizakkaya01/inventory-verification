"""Measure detection throughput/latency for a given weights file on this machine.

    python -m benchmarks.benchmark --weights best.pt --runs 200
    python -m benchmarks.benchmark --weights best.engine --runs 200 --device 0

Records to benchmarks/results/<name>.json so desktop vs Jetson vs OAK-D and
FP32 vs FP16 vs INT8 can be compared side by side. Accuracy (mAP) is measured
separately with `YOLO(weights).val(data=...)`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--device", default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    frame = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)

    lat_ms: list[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model.predict(frame, imgsz=args.imgsz, device=args.device, verbose=False)
        lat_ms.append((time.perf_counter() - t0) * 1000)

    lat_ms.sort()
    report = {
        "weights": args.weights,
        "imgsz": args.imgsz,
        "runs": args.runs,
        "device": str(args.device),
        "fps_mean": 1000 / statistics.fmean(lat_ms),
        "latency_ms_mean": statistics.fmean(lat_ms),
        "latency_ms_p50": lat_ms[len(lat_ms) // 2],
        "latency_ms_p95": lat_ms[int(len(lat_ms) * 0.95)],
    }
    print(json.dumps(report, indent=2))

    RESULTS.mkdir(exist_ok=True)
    name = args.name or Path(args.weights).stem
    (RESULTS / f"{name}.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
