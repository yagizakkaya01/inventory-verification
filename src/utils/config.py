"""Typed loading of configs/pipeline.yaml."""

from __future__ import annotations

import yaml
from pydantic import BaseModel


class SourceCfg(BaseModel):
    type: str = "webcam"
    index: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class DetectionCfg(BaseModel):
    weights: str
    conf: float = 0.35
    iou: float = 0.5
    imgsz: int = 640
    device: int | str = 0


class SmoothingCfg(BaseModel):
    window: int = 8
    min_agree: int = 6


class ExpectedCfg(BaseModel):
    sequence: list[str]
    row_tolerance: float = 0.15


class OutputCfg(BaseModel):
    draw: bool = True
    log_path: str = "runs/pipeline.log"


class PipelineConfig(BaseModel):
    source: SourceCfg
    detection: DetectionCfg
    smoothing: SmoothingCfg
    expected: ExpectedCfg
    output: OutputCfg

    @classmethod
    def load(cls, path: str = "configs/pipeline.yaml") -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**yaml.safe_load(fh))
