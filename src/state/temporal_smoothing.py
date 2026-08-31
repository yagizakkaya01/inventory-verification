"""Temporal smoothing: require a scene reading to hold across several frames
before the state logic acts on it. Filters detector flicker and brief occlusion.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

from src.detection.detector import Detection


@dataclass(frozen=True)
class SceneReading:
    """Order-aware snapshot of one frame, reduced to what the verifier needs."""

    sequence: tuple[str, ...]  # class names, left-to-right by centroid x

    @property
    def multiset(self) -> frozenset[tuple[str, int]]:
        return frozenset(Counter(self.sequence).items())

    @classmethod
    def from_detections(
        cls, dets: list[Detection], row_tolerance_px: float | None = None
    ) -> "SceneReading":
        # For a single-row scene, sort purely by x. row_tolerance is reserved
        # for future multi-row layouts.
        ordered = sorted(dets, key=lambda d: d.cx)
        return cls(sequence=tuple(d.cls_name for d in ordered))


class TemporalSmoother:
    def __init__(self, window: int = 8, min_agree: int = 6) -> None:
        if min_agree > window:
            raise ValueError("min_agree cannot exceed window")
        self.window = window
        self.min_agree = min_agree
        self._buf: deque[SceneReading] = deque(maxlen=window)
        self._stable: SceneReading | None = None

    def update(self, reading: SceneReading) -> SceneReading | None:
        """Add a frame reading. Returns the current stable reading, or None if
        no reading currently has enough agreement in the window."""
        self._buf.append(reading)
        if len(self._buf) < self.window:
            return self._stable
        top, count = Counter(self._buf).most_common(1)[0]
        if count >= self.min_agree:
            self._stable = top
        return self._stable

    @property
    def stable(self) -> SceneReading | None:
        return self._stable
