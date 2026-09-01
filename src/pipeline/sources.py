"""Frame sources. Each yields BGR numpy frames.

webcam/video (OpenCV) and oakd (DepthAI v3) are implemented. Astra is stubbed
with the SDK entry point noted.
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
    """RGB stream from an OAK-D via the DepthAI v3 API (CAM_A / color sensor).

    Runs detection on the *host* (our YOLO model on the RTX 4070). To move
    detection onto the camera's VPU instead, add a `dai.node.DetectionNetwork`
    fed by a compiled `.blob` (see src/edge/export_blob.py) and yield its
    results rather than raw frames.
    """
    import depthai as dai

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        # Request NV12 (YUV420) not BGR888i: ~1/2 the USB bandwidth, so 720p@30
        # holds even on a USB2 link. getCvFrame() converts to BGR host-side.
        out = cam.requestOutput(
            (cfg.width, cfg.height), dai.ImgFrame.Type.NV12, fps=cfg.fps
        )
        # Non-blocking, shallow queue: always process the newest frame and drop
        # any the consumer was too slow to take (real-time > completeness).
        queue = out.createOutputQueue(maxSize=4, blocking=False)
        pipeline.start()
        # First frame can take a few seconds on a USB2 link (sensor init +
        # autoexposure); steady state is ~30 FPS at 720p NV12.
        #
        # Two things matter for throughput on a USB2 link:
        #  - don't call pipeline.isRunning() in the loop: it does an XLink
        #    round-trip that competes with the video stream and tanks the FPS;
        #  - drain the queue every iteration so we stay on the live frame,
        #    otherwise work between get() calls lets frames back up and choke.
        while True:
            frame = queue.get()
            while (newer := queue.tryGet()) is not None:
                frame = newer
            yield frame.getCvFrame()


def _astra_source(cfg: SourceCfg) -> Iterator:
    # TODO: openni2.initialize() -> Device.open_any() -> create_color_stream()
    raise NotImplementedError("Astra source: implement with OpenNI / pyorbbecsdk")
