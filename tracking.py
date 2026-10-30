"""
tracking.py — DeepSORT multi-object tracking module.

Responsibilities:
  - Initialize and manage the DeepSORT tracker
  - Convert PersonDetection list → DeepSORT input format
  - Return confirmed tracks with stable IDs
  - Expose a clean TrackedPerson dataclass consumed by downstream stages
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

import config
from detection import PersonDetection

logger = logging.getLogger(__name__)


@dataclass
class TrackedPerson:
    """
    Full state of a confirmed, tracked person for one frame.
    Downstream modules (PPE, zone) consume this object.
    """
    track_id:    int
    bbox_xyxy:   np.ndarray   # [x1, y1, x2, y2] pixels
    confidence:  float
    frame_h:     int
    frame_w:     int
    is_confirmed: bool = True
    detection_state: str = "SAFE"

    # Populated by PPE detector
    has_helmet:       Optional[bool]  = None
    ppe_confidence:   Optional[float] = None

    # Populated by zone checker
    in_restricted_zone: bool = False
    zone_ids:           List[int] = field(default_factory=list)

    @property
    def foot_point(self):
        x1, y1, x2, y2 = self.bbox_xyxy.astype(int)
        return ((x1 + x2) // 2, y2)

    def crop_frame(self, frame: np.ndarray, padding: float = config.PPE_CROP_PADDING) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox_xyxy.astype(float)
        pw = (x2 - x1) * padding
        ph = (y2 - y1) * padding
        x1 = max(0, int(x1 - pw))
        y1 = max(0, int(y1 - ph))
        x2 = min(self.frame_w - 1, int(x2 + pw))
        y2 = min(self.frame_h - 1, int(y2 + ph))
        crop = frame[y1:y2, x1:x2]
        return crop.copy() if crop.size > 0 else np.zeros((1, 1, 3), dtype=np.uint8)

    @property
    def has_violation(self) -> bool:
        return self.in_restricted_zone or (self.has_helmet is False)

    @property
    def violation_label(self) -> str:
        parts = []
        if self.has_helmet is False:
            parts.append("NO HELMET")
        if self.in_restricted_zone:
            zone_str = ",".join(str(z) for z in self.zone_ids)
            parts.append(f"ZONE VIOLATION[{zone_str}]")
        return " | ".join(parts) if parts else "OK"


class PersonTracker:
    """
    Wraps deep-sort-realtime's DeepSort tracker.

    Key design decisions:
      - `embedder=None` lets DeepSORT use its built-in MobileNet Re-ID.
        For industrial environments, swap to a fine-tuned Re-ID model.
      - We keep an ID→previous_bbox mapping to interpolate lost tracks
        during the DEEPSORT_MAX_AGE window.
      - `update()` returns ONLY confirmed tracks (n_init frames seen).
    """

    def __init__(
        self,
        max_age:         int   = config.DEEPSORT_MAX_AGE,
        n_init:          int   = config.DEEPSORT_N_INIT,
        max_cosine_dist: float = config.DEEPSORT_MAX_COSINE_DIST,
        nn_budget:       int   = config.DEEPSORT_NN_BUDGET,
        max_iou_dist:    float = config.DEEPSORT_MAX_IOU_DIST,
    ) -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_dist = max_cosine_dist
        self.nn_budget = nn_budget
        self.max_iou_dist = max_iou_dist
        use_gpu = torch.cuda.is_available()

        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_dist,
            nn_budget=self.nn_budget,
            max_iou_distance=self.max_iou_dist,
            embedder="mobilenet",      # Built-in Re-ID; replace with custom if needed
            half=use_gpu,              # FP16 only when CUDA is available
            bgr=True,                  # Our frames are BGR (OpenCV default)
            embedder_gpu=use_gpu,      # Keep CPU environments stable
            polygon=False,
            today=None,
        )
        self._track_registry: Dict[int, np.ndarray] = {}  # id → last known xyxy
        logger.info("[PersonTracker] DeepSORT initialized.")

    def update(
        self,
        detections: List[PersonDetection],
        frame: np.ndarray,
    ) -> List[TrackedPerson]:
        """
        Feed new detections into DeepSORT and return confirmed tracks.

        Args:
            detections: Output of PersonDetector.detect()
            frame:      Current BGR frame (used by Re-ID embedder)

        Returns:
            List of TrackedPerson — one per confirmed, active track.
        """
        frame_h, frame_w = frame.shape[:2]

        # Build DeepSORT input: list of ([x1,y1,w,h], confidence, class_id)
        raw_detections = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            w = x2 - x1
            h = y2 - y1
            raw_detections.append(([x1, y1, w, h], det.confidence, "person"))

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        tracked_persons: List[TrackedPerson] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            ltrb = track.to_ltrb()   # [x1, y1, x2, y2]

            # Clamp to frame boundaries
            x1 = max(0, min(int(ltrb[0]), frame_w - 1))
            y1 = max(0, min(int(ltrb[1]), frame_h - 1))
            x2 = max(0, min(int(ltrb[2]), frame_w - 1))
            y2 = max(0, min(int(ltrb[3]), frame_h - 1))
            bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Ignore degenerate boxes
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            det_conf = track.det_conf if track.det_conf is not None else 0.0

            tp = TrackedPerson(
                track_id=track_id,
                bbox_xyxy=bbox,
                confidence=float(det_conf),
                frame_h=frame_h,
                frame_w=frame_w,
            )
            self._track_registry[track_id] = bbox
            tracked_persons.append(tp)

        return tracked_persons

    def reset(self) -> None:
        """Re-initialize tracker state (e.g., on camera switch)."""
        use_gpu = torch.cuda.is_available()
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            max_cosine_distance=self.max_cosine_dist,
            nn_budget=self.nn_budget,
            max_iou_distance=self.max_iou_dist,
            embedder="mobilenet",
            half=use_gpu,
            bgr=True,
            embedder_gpu=use_gpu,
        )
        self._track_registry.clear()
        logger.info("[PersonTracker] Tracker reset.")
