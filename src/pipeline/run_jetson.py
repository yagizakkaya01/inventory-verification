"""Jetson-compatible pipeline: source -> TRT detector -> smoother -> verifier -> FSM.

This is a standalone entrypoint for the Jetson Nano that uses the native
TensorRT detector (no Ultralytics dependency) and is compatible with
Python 3.6 (JetPack 4.6.4).

Usage:
    python -m src.pipeline.run_jetson --config configs/jetson.yaml
"""

import argparse
import time
import yaml
import cv2


def _load_config(path):
    """Load pipeline config from YAML as a plain dict."""
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _open_source(cfg):
    """Open a frame source based on config. Yields BGR numpy frames.

    Supports 'webcam', 'video', and 'oakd'.
    """
    src_type = cfg.get("type", "webcam")
    width = cfg.get("width", 1280)
    height = cfg.get("height", 720)
    fps = cfg.get("fps", 30)

    if src_type in ("webcam", "video"):
        index = cfg.get("index", 0)
        cap = cv2.VideoCapture(index)
        if src_type == "webcam":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    elif src_type == "oakd":
        import depthai as dai

        pipeline = dai.Pipeline()

        cam = pipeline.createColorCamera()
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(width, height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(fps)

        xout = pipeline.createXLinkOut()
        xout.setStreamName("rgb")
        cam.preview.link(xout.input)

        with dai.Device(pipeline) as device:
            q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            while True:
                frame = q.get().getCvFrame()
                yield frame
    else:
        raise ValueError("Unknown source type: {}".format(src_type))


# -- State logic (inlined, Python 3.6 compatible) -------------------------

from collections import Counter, deque
from enum import Enum


class Verdict(str, Enum):
    OK = "OK"
    MISSING = "MISSING"
    WRONG_ORDER = "WRONG_ORDER"
    WRONG_COMBINATION = "WRONG_COMBINATION"
    EMPTY = "EMPTY"


class VerificationResult(object):
    __slots__ = ("verdict", "expected", "observed", "detail")

    def __init__(self, verdict, expected, observed, detail=""):
        self.verdict = verdict
        self.expected = expected
        self.observed = observed
        self.detail = detail

    @property
    def ok(self):
        return self.verdict is Verdict.OK


class SceneReading(object):
    __slots__ = ("sequence",)

    def __init__(self, sequence):
        # type: (tuple) -> None
        self.sequence = sequence

    def __eq__(self, other):
        if not isinstance(other, SceneReading):
            return False
        return self.sequence == other.sequence

    def __hash__(self):
        return hash(self.sequence)

    @classmethod
    def from_detections(cls, dets):
        """Build a reading from a list of Detection objects, sorted left-to-right."""
        # Dedupe same-class overlapping boxes
        kept = []
        sorted_dets = sorted(dets, key=lambda d: -d.confidence)
        for d in sorted_dets:
            dominated = False
            for k in kept:
                if k.cls_name == d.cls_name and _overlap(k, d) > 0.55:
                    dominated = True
                    break
            if not dominated:
                kept.append(d)
        ordered = sorted(kept, key=lambda d: d.cx)
        return cls(sequence=tuple(d.cls_name for d in ordered))


def _overlap(a, b):
    """Intersection over smaller box area."""
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


class TemporalSmoother(object):
    def __init__(self, window=8, min_agree=6):
        self.window = window
        self.min_agree = min_agree
        self._buf = deque(maxlen=window)
        self._stable = None  # type: SceneReading

    def update(self, reading):
        self._buf.append(reading)
        if len(self._buf) < self.window:
            return self._stable
        top, count = Counter(self._buf).most_common(1)[0]
        if count >= self.min_agree:
            self._stable = top
        return self._stable


class Verifier(object):
    def __init__(self, expected_sequence):
        self.expected = tuple(expected_sequence)
        self._expected_counts = Counter(self.expected)

    def check(self, reading):
        observed = tuple(reading.sequence) if reading else ()
        if not observed:
            return VerificationResult(Verdict.EMPTY, self.expected, observed,
                                      "no objects detected")
        obs_counts = Counter(observed)
        if obs_counts == self._expected_counts:
            if observed == self.expected:
                return VerificationResult(Verdict.OK, self.expected, observed)
            return VerificationResult(
                Verdict.WRONG_ORDER, self.expected, observed,
                "expected order {}, saw {}".format(self.expected, observed))
        missing = list((self._expected_counts - obs_counts).elements())
        extra = list((obs_counts - self._expected_counts).elements())
        if extra:
            return VerificationResult(
                Verdict.WRONG_COMBINATION, self.expected, observed,
                "unexpected: {}".format(extra) +
                (", missing: {}".format(missing) if missing else ""))
        return VerificationResult(
            Verdict.MISSING, self.expected, observed,
            "missing: {}".format(missing))


class State(str, Enum):
    INIT = "INIT"
    OK = "OK"
    ERROR = "ERROR"


class Transition(object):
    __slots__ = ("frm", "to", "verdict", "detail")

    def __init__(self, frm, to, verdict, detail):
        self.frm = frm
        self.to = to
        self.verdict = verdict
        self.detail = detail


class StateMachine(object):
    def __init__(self, confirm_frames=3):
        self.confirm_frames = confirm_frames
        self.state = State.INIT
        self._pending = None
        self._pending_count = 0

    def update(self, result):
        target = State.OK if result.ok else State.ERROR
        if target == self.state:
            self._pending = None
            self._pending_count = 0
            return None
        if target == self._pending:
            self._pending_count += 1
        else:
            self._pending = target
            self._pending_count = 1
        if self._pending_count >= self.confirm_frames:
            frm = self.state
            self.state = target
            self._pending = None
            self._pending_count = 0
            return Transition(frm, self.state, result.verdict, result.detail)
        return None


# -- Render ----------------------------------------------------------------

def _render(frame, dets, result, fsm, fps):
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(frame, "{} {:.2f}".format(d.cls_name, d.confidence),
                    (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    color = (0, 200, 0) if fsm.state.value == "OK" else (0, 0, 255)
    cv2.putText(frame, "{}  {}".format(fsm.state.value, result.verdict.value),
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "{:4.1f} FPS".format(fps), (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("verification", frame)


# -- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/jetson.yaml")
    parser.add_argument("--no-window", action="store_true", help="disable cv2 display")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    # Import the TRT detector
    from src.detection.trt_detector import TRTDetector

    det_cfg = cfg["detection"]
    detector = TRTDetector(
        engine_path=det_cfg["weights"],
        conf=det_cfg.get("conf", 0.35),
        iou=det_cfg.get("iou", 0.5),
        imgsz=det_cfg.get("imgsz", 640),
    )

    sm_cfg = cfg.get("smoothing", {})
    smoother = TemporalSmoother(sm_cfg.get("window", 8), sm_cfg.get("min_agree", 6))

    exp_cfg = cfg.get("expected", {})
    verifier = Verifier(exp_cfg["sequence"])

    fsm = StateMachine()

    draw = cfg.get("output", {}).get("draw", True) and not args.no_window
    fps_ema = 0.0

    for frame in _open_source(cfg["source"]):
        t0 = time.time()

        dets = detector.infer(frame)
        reading = SceneReading.from_detections(dets)
        stable = smoother.update(reading)
        result = verifier.check(stable)
        transition = fsm.update(result)

        if transition:
            print("[{} -> {}] {}: {}".format(
                transition.frm, transition.to,
                transition.verdict.value, transition.detail))

        dt = time.time() - t0
        fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if dt > 0 else fps_ema

        if draw:
            _render(frame, dets, result, fsm, fps_ema)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if draw:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
