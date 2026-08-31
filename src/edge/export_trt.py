"""Export / build a TensorRT engine for Jetson or desktop NVIDIA GPUs.

The simplest path is Ultralytics' built-in exporter (needs TensorRT installed;
on Jetson it ships with JetPack):

    python -m src.edge.export_trt --weights best.pt --half
    python -m src.edge.export_trt --weights best.pt --int8 --data configs/data.yaml

INT8 needs a calibration set — point --data at the dataset config so Ultralytics
can calibrate on the val split. Record the accuracy delta in benchmarks/.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--half", action="store_true", help="FP16")
    p.add_argument("--int8", action="store_true", help="INT8 (needs --data)")
    p.add_argument("--data", default=None, help="dataset yaml for INT8 calibration")
    p.add_argument("--workspace", type=int, default=4, help="GiB")
    args = p.parse_args()

    from ultralytics import YOLO

    kwargs = dict(format="engine", imgsz=args.imgsz, workspace=args.workspace)
    if args.int8:
        kwargs.update(int8=True, data=args.data)
    elif args.half:
        kwargs.update(half=True)

    YOLO(args.weights).export(**kwargs)


if __name__ == "__main__":
    main()
