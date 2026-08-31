"""Export a trained YOLO checkpoint to ONNX (shared starting point for TensorRT
and OAK-D blob conversion).

    python -m src.edge.export_onnx --weights models/checkpoints/exp/weights/best.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument("--simplify", action="store_true")
    args = p.parse_args()

    from ultralytics import YOLO

    YOLO(args.weights).export(
        format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=args.simplify
    )


if __name__ == "__main__":
    main()
