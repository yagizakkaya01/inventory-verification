"""Frame sources. Each yields BGR numpy frames.

Only the webcam/video path (OpenCV) is implemented now. OAK-D and Astra are
stubbed with the SDK entry points noted so they can be filled in on the office
hardware.
"""

from __future__ import annotations

from collections.abc import Iterator

from src.utils.config import SourceCfg


def open_source(cfg: SourceCfg) -> Iterator:
    if cfg.type in ("webcam", "video"):
        yield from _opencv_source(cfg)
    elif cfg.type == "oakd":
        yield from _oakd_source(cfg)
    elif cfg.type == "astra":
        yield from _astra_source(cfg)
    else:
        raise ValueError(f"unknown source type: {cfg.type}")


def _opencv_source(cfg: SourceCfg) -> Iterator:
    import cv2

    cap = cv2.VideoCapture(cfg.index)
    if cfg.type == "webcam":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _oakd_source(cfg: SourceCfg) -> Iterator:
    # TODO: build a depthai.Pipeline with a ColorCamera -> XLinkOut("rgb"),
    # then loop on device.getOutputQueue("rgb").get().getCvFrame()
    raise NotImplementedError("OAK-D source: implement with `depthai` on the device")


def _astra_source(cfg: SourceCfg) -> Iterator:
    # TODO: openni2.initialize() -> Device.open_any() -> create_color_stream()
    raise NotImplementedError("Astra source: implement with OpenNI / pyorbbecsdk")
