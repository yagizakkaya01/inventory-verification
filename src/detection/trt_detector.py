"""TensorRT-based detector for Jetson Nano (Python 3.6, no Ultralytics dependency).

Loads a TensorRT .engine file built from the YOLOv11 ONNX export and runs
inference on the Jetson GPU. Post-processing (box decode + NMS) is done in
NumPy so there is zero dependency on ultralytics at runtime.

Usage (from the pipeline):
    detector = TRTDetector("best.engine", conf=0.30, iou=0.6, imgsz=640)
    dets = detector.infer(bgr_frame)
"""

import numpy as np
import cv2


class Detection(object):
    """Mirror of src.detection.detector.Detection, Python 3.6 compatible."""

    __slots__ = ("cls_name", "confidence", "xyxy")

    def __init__(self, cls_name, confidence, xyxy):
        # type: (str, float, tuple) -> None
        self.cls_name = cls_name
        self.confidence = confidence
        self.xyxy = xyxy  # (x1, y1, x2, y2) pixel coords

    @property
    def cx(self):
        # type: () -> float
        return (self.xyxy[0] + self.xyxy[2]) / 2.0

    @property
    def cy(self):
        # type: () -> float
        return (self.xyxy[1] + self.xyxy[3]) / 2.0


class TRTDetector(object):
    """YOLOv11 detector backed by a native TensorRT engine."""

    # YOLOv11 class names — must match the training order
    CLASS_NAMES = {0: "tufek", 1: "tabanca", 2: "sarjor"}

    def __init__(self, engine_path, conf=0.35, iou=0.5, imgsz=640, class_names=None):
        # type: (str, float, float, int, dict) -> None
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — initialises the CUDA context

        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        if class_names is not None:
            self.CLASS_NAMES = class_names

        # Load the serialised engine
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate host + device buffers for every binding
        self._inputs = []
        self._outputs = []
        self._d_inputs = []
        self._d_outputs = []
        self._bindings = []
        self._stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            host_buf = cuda.pagelocked_empty(size, dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            self._bindings.append(int(dev_buf))

            if self.engine.binding_is_input(i):
                self._inputs.append(host_buf)
                self._d_inputs.append(dev_buf)
            else:
                self._outputs.append(host_buf)
                self._d_outputs.append(dev_buf)

        # Cache output shape for reshaping later
        out_idx = 1  # first output binding index
        self._out_shape = tuple(self.engine.get_binding_shape(out_idx))

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _preprocess(self, frame):
        """Resize + pad (letterbox) + normalise to [0,1] CHW float32."""
        ih, iw = frame.shape[:2]
        scale = min(self.imgsz / iw, self.imgsz / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        dx, dy = (self.imgsz - nw) // 2, (self.imgsz - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, 0-1
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.ascontiguousarray(blob[np.newaxis, ...])  # add batch dim
        return blob, scale, dx, dy

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _xywh_to_xyxy(boxes):
        """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        """Pure-numpy NMS."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]
        return np.array(keep, dtype=np.intp)

    def _postprocess(self, raw, scale, dx, dy, orig_h, orig_w):
        """Decode YOLOv11 output tensor into a list of Detection objects.

        YOLOv11 output shape: (1, 4+num_classes, num_predictions)
        Transposed to (num_predictions, 4+num_classes).
        """
        # raw shape: (1, 4+C, N) -> (N, 4+C)
        preds = raw.reshape(self._out_shape)
        if preds.ndim == 3:
            preds = preds[0]  # remove batch
        preds = preds.T  # (4+C, N) -> (N, 4+C)

        num_classes = preds.shape[1] - 4
        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # Per-row max class
        class_ids = class_scores.argmax(axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = confidences >= self.conf
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        boxes_xyxy = self._xywh_to_xyxy(boxes_xywh)

        # NMS per class
        keep_all = []
        for cls_id in np.unique(class_ids):
            cls_mask = class_ids == cls_id
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = confidences[cls_mask]
            indices = np.where(cls_mask)[0]
            k = self._nms(cls_boxes, cls_scores, self.iou)
            keep_all.extend(indices[k].tolist())

        detections = []
        for idx in keep_all:
            x1, y1, x2, y2 = boxes_xyxy[idx]
            # Undo letterbox: subtract padding, then unscale
            x1 = (x1 - dx) / scale
            y1 = (y1 - dy) / scale
            x2 = (x2 - dx) / scale
            y2 = (y2 - dy) / scale
            # Clip to frame
            x1 = float(max(0, min(x1, orig_w)))
            y1 = float(max(0, min(y1, orig_h)))
            x2 = float(max(0, min(x2, orig_w)))
            y2 = float(max(0, min(y2, orig_h)))

            cls_id = int(class_ids[idx])
            cls_name = self.CLASS_NAMES.get(cls_id, "class_{}".format(cls_id))
            detections.append(Detection(cls_name, float(confidences[idx]),
                                        (x1, y1, x2, y2)))
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, frame):
        """Run detection on a single BGR frame (numpy array)."""
        import pycuda.driver as cuda

        orig_h, orig_w = frame.shape[:2]
        blob, scale, dx, dy = self._preprocess(frame)

        # Copy input to device
        np.copyto(self._inputs[0], blob.ravel())
        cuda.memcpy_htod_async(self._d_inputs[0], self._inputs[0], self._stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )

        # Copy output back to host
        cuda.memcpy_dtoh_async(self._outputs[0], self._d_outputs[0], self._stream)
        self._stream.synchronize()

        return self._postprocess(self._outputs[0], scale, dx, dy, orig_h, orig_w)
