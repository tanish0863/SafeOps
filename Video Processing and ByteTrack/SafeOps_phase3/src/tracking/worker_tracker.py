"""Worker detection + ByteTrack tracking using YOLOv8 Pose (Ultralytics).

Features:
- YOLOv8-Pose model running on COCO class 0 (Person only). Never runs on PPE.
- ByteTrack with custom parameters (custom_bytetrack.yaml).
- Geometric continuity validation: detects overlapping trajectories and flags
  uncertain continuity during crossings or rapid scale changes.
- Gap-recovery tracking: detects when a track was resumed from the track buffer.
- Strict temporary identity semantics: temporary video tracking IDs only.
- GPU FP16 support on CUDA, with clean CPU fallback.
- Explicit .reset() between independent video streams.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from src.contracts.schemas import TrackingContinuityState


def _box_iou(b1: List[float], b2: List[float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


class WorkerTracker:
    def __init__(
        self,
        model_path: str = "models/yolov8s-pose.pt",
        tracker: str = "src/tracking/custom_bytetrack.yaml",
        conf: float = 0.10,
        iou: float = 0.7,
        imgsz: int = 640,
        overlap_uncertainty_iou: float = 0.35,
        device: str | None = None,
        half: bool | None = None,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_cuda = str(device).startswith("cuda")
        self.half = self.use_cuda if half is None else half
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.overlap_uncertainty_iou = overlap_uncertainty_iou

        path = Path(model_path)
        if not path.exists():
            print(f"{path} not found; letting Ultralytics download '{path.name}'")
            path = Path(path.name)
        self.model = YOLO(str(path))

        # Track history for geometric continuity
        self._last_seen_frame: Dict[int, int] = {}
        self._prev_boxes: Dict[int, List[float]] = {}
        self._current_frame_number: int = 0

    def reset(self) -> None:
        """Clear tracker state between independent videos."""
        self.model.predictor = None
        self._last_seen_frame.clear()
        self._prev_boxes.clear()
        self._current_frame_number = 0

    def update(self, frame: np.ndarray, frame_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """Process a video frame and return tracked worker records.

        Args:
            frame: BGR image numpy array
            frame_number: Optional zero-based frame index

        Returns:
            List of worker dicts matching WorkerTrackingRecord schema.
        """
        if frame_number is not None:
            self._current_frame_number = frame_number
        else:
            self._current_frame_number += 1

        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],  # COCO class 0 = person only
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        result = results[0]

        # Early exit if no boxes or IDs are present
        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        kp_xy = kp_conf = None
        if result.keypoints is not None:
            kp_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                kp_conf = result.keypoints.conf.cpu().numpy()

        n_workers = len(track_ids)
        workers: List[Dict[str, Any]] = []

        # 1. Identify overlapping pairs to flag geometric ambiguity
        overlapping_ids = set()
        for i in range(n_workers):
            for j in range(i + 1, n_workers):
                iou_val = _box_iou(boxes[i].tolist(), boxes[j].tolist())
                if iou_val >= self.overlap_uncertainty_iou:
                    overlapping_ids.add(int(track_ids[i]))
                    overlapping_ids.add(int(track_ids[j]))

        # 2. Build structured worker records with continuity states
        for i in range(n_workers):
            tid = int(track_ids[i])
            box = [float(v) for v in boxes[i]]
            conf_val = float(confs[i])

            # Assess continuity state
            continuity = TrackingContinuityState.CONFIRMED
            reason = "active_continuous"

            last_seen = self._last_seen_frame.get(tid)
            if last_seen is None:
                continuity = TrackingContinuityState.CONFIRMED
                reason = "new_track"
            elif self._current_frame_number - last_seen > 1:
                # Track was resumed after a temporary disappearance
                continuity = TrackingContinuityState.RECOVERED
                reason = f"recovered_after_{self._current_frame_number - last_seen}_frames"

            if tid in overlapping_ids:
                continuity = TrackingContinuityState.UNCERTAIN
                reason = "spatial_trajectory_overlap"

            # Size jump check (e.g. sudden scale distortion)
            prev_box = self._prev_boxes.get(tid)
            if prev_box is not None:
                prev_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                curr_area = (box[2] - box[0]) * (box[3] - box[1])
                if prev_area > 0 and (curr_area / prev_area > 3.0 or curr_area / prev_area < 0.33):
                    continuity = TrackingContinuityState.UNCERTAIN
                    reason = "abrupt_box_scale_change"

            self._last_seen_frame[tid] = self._current_frame_number
            self._prev_boxes[tid] = box

            workers.append({
                "track_id": tid,
                "bbox_xyxy": box,
                "confidence": conf_val,
                "keypoints_xy": kp_xy[i].tolist() if kp_xy is not None else None,
                "keypoints_conf": kp_conf[i].tolist() if kp_conf is not None else None,
                "continuity_state": continuity.value,
                "continuity_reason": reason,
            })

        return workers
