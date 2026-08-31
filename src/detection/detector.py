"""Thin wrapper around an Ultralytics YOLO model.

Keeps the rest of the pipeline independent of the Ultralytics API surface so we
can later swap in an ONNX / TensorRT / OAK-D backend behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Detection:
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]  # pixel coords

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2


class Detector:
    def __init__(
        self,
        weights: str,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 640,
        device: int | str = 0,
    ) -> None:
        from ultralytics import YOLO  # imported lazily so tests don't need torch

        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def infer(self, frame) -> list[Detection]:
        """Run detection on a single BGR frame (numpy array)."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        names = results.names
        out: list[Detection] = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            out.append(
                Detection(
                    cls_name=names[cls_id],
                    confidence=float(box.conf[0]),
                    xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                )
            )
        return out
