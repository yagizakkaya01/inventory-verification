"""Real-time verification loop: source -> detector -> smoother -> verifier -> FSM.

Usage:
    python -m src.pipeline.run --config configs/pipeline.yaml
"""

from __future__ import annotations

import argparse
import time

from src.detection.detector import Detector
from src.pipeline.sources import open_source
from src.state.state_machine import StateMachine
from src.state.temporal_smoothing import SceneReading, TemporalSmoother
from src.state.verifier import Verifier
from src.utils.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--no-window", action="store_true", help="disable cv2 display")
    args = parser.parse_args()

    cfg = PipelineConfig.load(args.config)

    detector = Detector(
        weights=cfg.detection.weights,
        conf=cfg.detection.conf,
        iou=cfg.detection.iou,
        imgsz=cfg.detection.imgsz,
        device=cfg.detection.device,
    )
    smoother = TemporalSmoother(cfg.smoothing.window, cfg.smoothing.min_agree)
    verifier = Verifier(cfg.expected.sequence)
    fsm = StateMachine()

    draw = cfg.output.draw and not args.no_window
    fps_ema = 0.0

    for frame in open_source(cfg.source):
        t0 = time.perf_counter()

        dets = detector.infer(frame)
        reading = SceneReading.from_detections(dets)
        stable = smoother.update(reading)
        result = verifier.check(stable)
        transition = fsm.update(result)

        if transition:
            print(f"[{transition.frm} -> {transition.to}] "
                  f"{transition.verdict.value}: {transition.detail}")

        dt = time.perf_counter() - t0
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if dt > 0 else fps_ema

        if draw:
            _render(frame, dets, result, fsm, fps_ema)
            import cv2
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if draw:
        import cv2
        cv2.destroyAllWindows()


def _render(frame, dets, result, fsm, fps) -> None:
    import cv2

    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, f"{d.cls_name} {d.confidence:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    color = (0, 200, 0) if fsm.state.value == "OK" else (0, 0, 255)
    cv2.putText(frame, f"{fsm.state.value}  {result.verdict.value}", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{fps:4.1f} FPS", (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("verification", frame)


if __name__ == "__main__":
    main()
