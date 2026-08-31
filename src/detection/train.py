"""Fine-tune a YOLO detector via transfer learning.

Usage:
    python -m src.detection.train --config configs/train.yaml
"""

from __future__ import annotations

import argparse

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    from ultralytics import YOLO

    model = YOLO(cfg.pop("model"))
    results = model.train(**cfg)
    print(f"Done. Best weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
