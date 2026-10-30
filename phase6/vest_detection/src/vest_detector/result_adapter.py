"""
result_adapter.py
-----------------
Translates raw model outputs into canonical SafeOps schemas.
Handles label normalization, spatial coordinate mapping from sub-region
crops back to full-image pixel space, and geometric overlap calculations.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np

from .config import VestDetectorConfig
from .schemas import Detection, VestDetectionResult


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def compute_intersection_ratio(target_box: Sequence[float], reference_box: Sequence[float]) -> float:
    """
    Calculate fraction of target_box area that falls inside reference_box.
    Formula: Area(target_box ∩ reference_box) / Area(target_box)
    """
    tx1, ty1, tx2, ty2 = target_box
    rx1, ry1, rx2, ry2 = reference_box

    inter_x1 = max(tx1, rx1)
    inter_y1 = max(ty1, ry1)
    inter_x2 = min(tx2, rx2)
    inter_y2 = min(ty2, ry2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    if target_area <= 0.0:
        return 0.0

    return inter_area / target_area


class ResultAdapter:
    """
    Converts and standardizes raw model detections into SafeOps schemas.
    """

    def __init__(self, config: Optional[VestDetectorConfig] = None):
        self.config = config or VestDetectorConfig()

    def adapt_detections(
        self,
        raw_boxes: Sequence[Sequence[float]],
        raw_scores: Sequence[float],
        raw_classes: Sequence[Any],
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        """
        Convert raw model predictions to Detection objects with original-image coordinates.

        Parameters
        ----------
        raw_boxes : Sequence of [x1, y1, x2, y2]
        raw_scores : Confidence scores
        raw_classes : Class labels (names or IDs mapped to names)
        offset_x : X-offset to add (for crop mapping back to full image)
        offset_y : Y-offset to add (for crop mapping back to full image)
        image_shape : Optional (height, width) of the original image for clipping

        Returns
        -------
        List of Detection objects strictly in original-image pixel space.
        """
        detections: List[Detection] = []

        for box, score, cls_name in zip(raw_boxes, raw_scores, raw_classes):
            score_val = float(score)

            normalized_label = self.config.normalize_label(str(cls_name))
            if normalized_label is None:
                # Class is not a vest or no-vest (e.g. Helmet, Person, Machinery)
                continue

            # Class-specific confidence filtering
            min_conf = (
                self.config.vest_confidence_threshold
                if normalized_label == "WITH_VEST"
                else self.config.no_vest_confidence_threshold
            )
            if score_val < min_conf:
                continue

            # Map coordinates by offset (original image space)
            orig_x1 = float(box[0]) + offset_x
            orig_y1 = float(box[1]) + offset_y
            orig_x2 = float(box[2]) + offset_x
            orig_y2 = float(box[3]) + offset_y

            # Clamp to full image bounds if provided
            if image_shape is not None:
                h, w = image_shape[:2]
                orig_x1 = max(0.0, min(float(w), orig_x1))
                orig_y1 = max(0.0, min(float(h), orig_y1))
                orig_x2 = max(0.0, min(float(w), orig_x2))
                orig_y2 = max(0.0, min(float(h), orig_y2))

            if orig_x2 <= orig_x1 or orig_y2 <= orig_y1:
                continue

            detections.append(
                Detection(
                    class_name=normalized_label,
                    confidence=round(score_val, 4),
                    bbox=[
                        round(orig_x1, 2),
                        round(orig_y1, 2),
                        round(orig_x2, 2),
                        round(orig_y2, 2),
                    ],
                )
            )

        return detections

    def build_result(
        self,
        detections: List[Detection],
        metadata: Optional[dict] = None,
    ) -> VestDetectionResult:
        """Construct top-level VestDetectionResult."""
        return VestDetectionResult(
            schema_version="1.0",
            detections=detections,
            metadata=metadata,
        )
