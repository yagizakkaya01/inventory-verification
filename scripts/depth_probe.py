"""OAK-D stereo depth probe: capture RGB + depth (aligned to RGB), run the
detector on RGB, print the median depth inside each box, and save an annotated
frame + a colourised depth map.

    python -m scripts.depth_probe

First step toward using depth in the pipeline (object distance / lift detection /
overlap separation).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import Detector
from src.utils.config import PipelineConfig

OUT = Path("C:/Users/Acer/AppData/Local/Temp")


def main() -> None:
    import depthai as dai

    cfg = PipelineConfig.load("configs/pipeline.yaml")
    det = Detector(cfg.detection.weights, cfg.detection.conf, cfg.detection.iou,
                   cfg.detection.imgsz, cfg.detection.device)

    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)
        stereo.setOutputSize(640, 400)          # multiple of 16; host resizes to RGB

        # keep host bandwidth low (USB2): small-ish RGB, mono stays on-device
        rgb_out = cam.requestOutput((960, 540), dai.ImgFrame.Type.NV12, fps=12)
        left.requestOutput((640, 400), dai.ImgFrame.Type.NV12, fps=12).link(stereo.left)
        right.requestOutput((640, 400), dai.ImgFrame.Type.NV12, fps=12).link(stereo.right)

        q_rgb = rgb_out.createOutputQueue(maxSize=4, blocking=False)
        q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()
        print("warmup...", flush=True)
        rgb = depth = None
        t0 = time.time()
        try:
            while time.time() - t0 < 8:          # let AE/AWB + stereo settle
                r = q_rgb.tryGet()
                if r is not None:
                    rgb = r.getCvFrame()
                d = q_depth.tryGet()
                if d is not None:
                    depth = d.getFrame()          # uint16, mm, aligned to RGB
                if rgb is not None and depth is not None and time.time() - t0 > 5:
                    break
                time.sleep(0.03)
        finally:
            pipeline.stop()
        print(f"rgb={'var' if rgb is not None else 'YOK'} "
              f"depth={'var' if depth is not None else 'YOK'}", flush=True)

    if rgb is None or depth is None:
        print("kare alınamadı")
        return

    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    valid = depth[depth > 0]
    print(f"RGB {rgb.shape}  depth {depth.shape}  "
          f"geçerli piksel %{100 * valid.size / depth.size:.0f}  "
          f"aralık {valid.min() if valid.size else 0}-{np.percentile(valid, 95) if valid.size else 0:.0f} mm")

    dets = det.infer(rgb)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        patch = depth[y1:y2, x1:x2]
        patch = patch[patch > 0]
        mm = float(np.median(patch)) if patch.size else 0.0
        print(f"  {d.cls_name:8} conf {d.confidence:.2f}  median depth {mm:6.0f} mm")
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(rgb, f"{d.cls_name} {mm:.0f}mm", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    # colourised depth
    dvis = np.clip(depth, 0, 2000).astype(np.float32) / 2000 * 255
    dvis = cv2.applyColorMap(dvis.astype(np.uint8), cv2.COLORMAP_TURBO)
    dvis[depth == 0] = 0

    cv2.imwrite(str(OUT / "depth_rgb.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(OUT / "depth_map.jpg"), dvis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"kaydedildi: {OUT/'depth_rgb.jpg'}  {OUT/'depth_map.jpg'}")


if __name__ == "__main__":
    main()
