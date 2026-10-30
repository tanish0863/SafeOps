"""
detection.py — YOLOv8 person detection module.

Responsibilities:
  - Load and manage YOLOv8 model lifecycle
  - Run inference on frames
  - Filter results to person class only
  - Return normalized detection data ready for DeepSORT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO
import cv2

import config

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PersonDetection:
    """Immutable container for a single person detection."""
    bbox_xyxy: np.ndarray    # [x1, y1, x2, y2] in pixel coords
    confidence: float
    frame_h: int
    frame_w: int
    detection_state: str = "SAFE"

    @property
    def bbox_xywh(self) -> np.ndarray:
        """Returns [cx, cy, w, h] — format required by DeepSORT."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ], dtype=np.float32)

    @property
    def foot_point(self) -> Tuple[int, int]:
        """Bottom-center of bounding box — used for zone containment check."""
        x1, y1, x2, y2 = self.bbox_xyxy.astype(int)
        return ((x1 + x2) // 2, y2)

    @property
    def height_px(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    def crop_frame(self, frame: np.ndarray, padding: float = config.PPE_CROP_PADDING) -> np.ndarray:
        """
        Return a padded crop of the person from the given frame.
        Padding expands the box by `padding` fraction on each side.
        """
        x1, y1, x2, y2 = self.bbox_xyxy.astype(float)
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(self.frame_w - 1, int(x2 + pw))
        y2 = min(self.frame_h - 1, int(y2 + ph))
        return frame[y1:y2, x1:x2].copy()


class PersonDetector:
    """
    Wraps a YOLOv8 model configured exclusively for person detection.

    Optimizations applied:
      - Half-precision (FP16) on CUDA
      - Fixed inference resolution to avoid dynamic shape recompilation
      - Persistent model on device (no per-frame model reload)
      - Vectorised numpy post-processing — no Python loops over raw tensors
    """

    def __init__(
        self,
        model_path: str = config.PERSON_MODEL_PATH,
        conf_threshold: float = config.PERSON_CONF_THRESHOLD,
        iou_threshold: float  = config.PERSON_IOU_THRESHOLD,
        input_size: int       = config.INFERENCE_INPUT_SIZE,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size

        # Auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[PersonDetector] Loading model '{model_path}' on device='{self.device}'")
        self.model = YOLO(model_path)
        self.model.to(self.device)

        # Warm-up pass to avoid JIT latency on first real frame.
        # If the environment is memory-constrained, keep the model usable
        # instead of failing the whole website startup.
        try:
            self._warmup()
        except Exception as exc:
            logger.warning(f"[PersonDetector] Warm-up skipped: {exc}")

    def _warmup(self) -> None:
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[config.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )
        logger.info("[PersonDetector] Warm-up complete.")

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """
        Mild contrast normalization to help recovery after close occlusions.
        Keeps the image shape intact and stays cheap enough for realtime use.
        """
        try:
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            normalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            return cv2.GaussianBlur(normalized, (3, 3), 0)
        except Exception:
            return frame

    def _predict_raw(self, source_frame: np.ndarray, conf_threshold: float):
        return self.model.predict(
            source=source_frame,
            imgsz=self.input_size,
            conf=conf_threshold,
            iou=self.iou_threshold,
            classes=[config.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
            agnostic_nms=True,
            half=(self.device == "cuda"),
        )

    @classmethod
    def _filter_detections(
        cls,
        detections: List[PersonDetection],
        conf_threshold: float,
        relaxed: bool = False,
    ) -> List[PersonDetection]:
        if not detections:
            return []

        kept: List[PersonDetection] = []
        if relaxed:
            min_height = 36.0
            min_width = 16.0
            min_area = 576.0
        else:
            min_height = 40.0
            min_width = 20.0
            min_area = 800.0

        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if det.confidence < conf_threshold:
                continue
            x1, y1, x2, y2 = det.bbox_xyxy.astype(float)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area = width * height
            if width >= min_width and height >= min_height and area >= min_area:
                det.detection_state = "SAFE"
                kept.append(det)
            elif relaxed and width >= 14.0 and height >= 28.0 and area >= 392.0:
                det.detection_state = "UNCERTAIN"
                kept.append(det)
        return kept

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Run YOLOv8 inference on a BGR frame.

        Returns:
            List of PersonDetection objects, filtered by minimum height.
        """
        frame_h, frame_w = frame.shape[:2]

        def predict_candidates(conf_threshold: float, source_frame: np.ndarray):
            results = self._predict_raw(source_frame, conf_threshold)
            detections: List[PersonDetection] = []
            if not results or results[0].boxes is None:
                return detections
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                detections.append(
                    PersonDetection(
                        bbox_xyxy=xyxy[i],
                        confidence=float(confs[i]),
                        frame_h=frame_h,
                        frame_w=frame_w,
                    )
                )
            return detections

        detections = predict_candidates(self.conf_threshold, frame)
        detections = self._filter_detections(detections, conf_threshold=self.conf_threshold, relaxed=False)
        if detections:
            return detections

        normalized_frame = self._normalize_frame(frame)
        recovery_conf = max(0.50, self.conf_threshold - 0.02)
        recovery_detections = predict_candidates(recovery_conf, normalized_frame)
        relaxed_detections = self._filter_detections(
            recovery_detections,
            conf_threshold=recovery_conf,
            relaxed=True,
        )
        return relaxed_detections

    def batch_detect(self, frames: List[np.ndarray]) -> List[List[PersonDetection]]:
        """
        Run inference on a batch of frames in one forward pass.
        Useful for multi-camera setups.
        """
        if not frames:
            return []

        frame_dims = [(f.shape[0], f.shape[1]) for f in frames]
        results = self.model.predict(
            source=frames,
            imgsz=self.input_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=[config.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
            half=(self.device == "cuda"),
        )

        all_detections: List[List[PersonDetection]] = []
        for result, (fh, fw) in zip(results, frame_dims):
            detections = []
            if result.boxes is not None:
                xyxy  = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                for i in range(len(xyxy)):
                    det = PersonDetection(
                        bbox_xyxy=xyxy[i],
                        confidence=float(confs[i]),
                        frame_h=fh,
                        frame_w=fw,
                    )
                    detections.append(det)
            filtered = self._filter_detections(detections, conf_threshold=self.conf_threshold, relaxed=False)
            if not filtered:
                filtered = self._filter_detections(
                    detections,
                    conf_threshold=max(0.35, self.conf_threshold - 0.05),
                    relaxed=True,
                )
            all_detections.append(filtered)

        return all_detections
