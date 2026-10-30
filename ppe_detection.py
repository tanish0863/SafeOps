"""
ppe_detection.py - Helmet/PPE detection on per-frame head regions.

This version keeps the PPE stage intentionally simple:
  - one input type only: the top part of the detected person bbox
  - one inference pass per frame
  - no tracking history
  - no temporal smoothing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import config
from tracking import TrackedPerson

logger = logging.getLogger(__name__)


def _normalize_class_name(name: object) -> str:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def _infer_helmet_class_sets(names: object) -> tuple[set[int], set[int]]:
    helmet_ids: set[int] = set()
    no_helmet_ids: set[int] = set()

    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, (list, tuple)):
        items = enumerate(names)
    else:
        items = []

    for class_id, raw_name in items:
        name = _normalize_class_name(raw_name)
        if "no_helmet" in name or name.startswith("without_helmet") or name.startswith("nohelmet"):
            no_helmet_ids.add(int(class_id))
        elif "helmet" in name:
            helmet_ids.add(int(class_id))

    return helmet_ids, no_helmet_ids


MIN_ROI_W = 20
MIN_ROI_H = 20


@dataclass
class PPEResult:
    track_id: int
    has_helmet: Optional[bool]
    confidence: float
    raw_detections: list


class PPEDetector:
    """
    Detects helmet/no-helmet from the top region of each detected person bbox.

    Key rules:
      - only use the detected person bbox as source of truth
      - only use the head region for inference
      - make an immediate frame-local decision
    """

    PPE_INFERENCE_SIZE = 320
    HEAD_ROI_TOP_RATIO = 0.75
    HEAD_ROI_SIDE_RATIO = 0.0

    def __init__(
        self,
        model_path: str = config.PPE_MODEL_PATH,
        conf: float = config.PPE_CONF_THRESHOLD,
        iou: float = config.PPE_IOU_THRESHOLD,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.is_stub = False

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"[PPEDetector] Loading model '{model_path}' on device='{self.device}'")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self._warmup()
            self.helmet_class_ids, self.no_helmet_class_ids = _infer_helmet_class_sets(getattr(self.model, "names", {}))
            if not self.helmet_class_ids and not self.no_helmet_class_ids:
                self.helmet_class_ids, self.no_helmet_class_ids = {0}, {1}
        except Exception as e:
            logger.error(f"[PPEDetector] Failed to load PPE model: {e}")
            logger.warning("[PPEDetector] Running in STUB mode - all persons flagged as UNKNOWN")
            self.model = None
            self.is_stub = True
            self.helmet_class_ids, self.no_helmet_class_ids = {0}, {1}

    def _warmup(self) -> None:
        dummy = np.zeros((self.PPE_INFERENCE_SIZE, self.PPE_INFERENCE_SIZE, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy,
            imgsz=self.PPE_INFERENCE_SIZE,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        logger.info("[PPEDetector] Warm-up complete.")

    @staticmethod
    def _extract_head_roi(track: TrackedPerson, frame: np.ndarray) -> np.ndarray:
        """
        Extract the top slice of the detected person bbox.

        This keeps the crop stable because it is derived from the current frame
        person box and it only uses the head region.
        """
        x1, y1, x2, y2 = track.bbox_xyxy.astype(float)
        fw = track.frame_w
        fh = track.frame_h
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        roi_x1 = max(0, int(x1 + box_w * PPEDetector.HEAD_ROI_SIDE_RATIO))
        roi_x2 = min(fw, int(x2 - box_w * PPEDetector.HEAD_ROI_SIDE_RATIO))
        roi_y1 = max(0, int(y1))
        roi_y2 = min(fh, int(y1 + box_h * PPEDetector.HEAD_ROI_TOP_RATIO))

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi.copy() if roi.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

    def process_tracks(self, tracks: List[TrackedPerson], frame: np.ndarray) -> None:
        """
        Run PPE detection on all tracks IN-PLACE.
        """
        if not tracks:
            return

        crops: List[np.ndarray] = []
        track_indices: List[int] = []

        for idx, track in enumerate(tracks):
            head_roi = self._extract_head_roi(track, frame)
            h, w = head_roi.shape[:2]
            if h < MIN_ROI_H or w < MIN_ROI_W:
                track.has_helmet = None
                track.ppe_confidence = 0.0
                continue
            crops.append(head_roi)
            track_indices.append(idx)

        if not crops:
            return

        if self.model is None:
            for idx in track_indices:
                tracks[idx].has_helmet = None
                tracks[idx].ppe_confidence = 0.0
            return

        results = self.model.predict(
            source=crops,
            imgsz=self.PPE_INFERENCE_SIZE,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
            half=(self.device == "cuda"),
        )

        for result, track_idx in zip(results, track_indices):
            current_value, current_conf = self._parse_single_result(result)
            tracks[track_idx].has_helmet = current_value
            tracks[track_idx].ppe_confidence = current_conf

    def _parse_single_result(self, result) -> Tuple[Optional[bool], float]:
        """
        Return helmet/no-helmet/unknown from one crop.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return None, 0.0

        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        no_helmet_mask = np.isin(classes, list(self.no_helmet_class_ids))
        helmet_mask = np.isin(classes, list(self.helmet_class_ids))

        best_no_helmet = float(confs[no_helmet_mask].max()) if no_helmet_mask.any() else 0.0
        best_helmet = float(confs[helmet_mask].max()) if helmet_mask.any() else 0.0

        if best_helmet > best_no_helmet and best_helmet > 0.0:
            return True, best_helmet
        elif best_no_helmet > 0.0:
            return False, best_no_helmet

        return None, 0.0

    @staticmethod
    def visualize_crop_debug(crop: np.ndarray, result, window_name: str = "PPE Debug") -> None:
        annotated = result.plot()
        cv2.imshow(window_name, annotated)
