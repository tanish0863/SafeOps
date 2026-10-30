"""
torso_inference.py
------------------
Local worker torso cropping, quality validation, and three-state compliance assessment.
Enforces the strict requirements:
  - PRESENT: Confident vest detection overlapping visible torso.
  - ABSENT: Torso is sufficiently visible/large, and no vest is detected (or explicit NO_VEST).
  - UNKNOWN: Heavily occluded, too distant/small, poor lighting, or out of frame.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import cv2
import numpy as np

from .config import VestDetectorConfig
from .result_adapter import compute_intersection_ratio, compute_iou
from .schemas import (
    Detection,
    TorsoAssessmentEvidence,
    TorsoVestAssessment,
    VestAssessmentState,
)


def validate_torso_geometry(
    torso_bbox: Sequence[float],
    image_shape: Tuple[int, int],
    config: VestDetectorConfig,
) -> Tuple[bool, List[str], dict]:
    """
    Validate that the torso bounding box is geometrically valid, sufficiently large,
    and positioned within the image frame.
    """
    reasons: List[str] = []
    img_h, img_w = image_shape[:2]

    if len(torso_bbox) < 4:
        return False, ["Invalid bbox specification: requires 4 coordinates."], {}

    x1, y1, x2, y2 = [float(c) for c in torso_bbox[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    stats = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": width,
        "height": height,
        "area": area,
    }

    # Check 1: Non-zero positive dimensions
    if width <= 0.0 or height <= 0.0:
        reasons.append("Torso bounding box has non-positive width or height.")
        return False, reasons, stats

    # Check 2: Within frame bounds
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        clipped_x1 = max(0.0, x1)
        clipped_y1 = max(0.0, y1)
        clipped_x2 = min(float(img_w), x2)
        clipped_y2 = min(float(img_h), y2)
        clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)

        if clipped_area <= 0.0:
            reasons.append("Torso is completely outside the camera frame.")
            return False, reasons, stats

        clipped_fraction = (area - clipped_area) / area
        if clipped_fraction > 0.35:
            reasons.append(
                f"Torso is heavily clipped by image border ({clipped_fraction * 100:.1f}% out of frame)."
            )
            return False, reasons, stats

    # Check 3: Minimum resolution and size requirements
    if width < config.min_torso_width or height < config.min_torso_height:
        reasons.append(
            f"Torso dimensions ({width:.1f}x{height:.1f} px) are below minimum resolution "
            f"threshold ({config.min_torso_width}x{config.min_torso_height} px) for reliable detection."
        )
        return False, reasons, stats

    if area < config.min_torso_area:
        reasons.append(
            f"Torso area ({area:.0f} px²) is too small for reliable feature extraction "
            f"(minimum {config.min_torso_area:.0f} px²)."
        )
        return False, reasons, stats

    return True, reasons, stats


def evaluate_torso_image_quality(
    crop_bgr: np.ndarray,
    config: VestDetectorConfig,
) -> Tuple[bool, List[str]]:
    """
    Evaluate basic optical quality of the torso crop (lighting, blur).
    Prevents false ABSENT judgments on dark, overexposed, or severely blurred crops.
    """
    reasons: List[str] = []

    if crop_bgr.size == 0:
        return False, ["Torso image crop is empty."]

    # Lighting: Mean intensity
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
    mean_intensity = float(np.mean(gray))

    if mean_intensity < config.min_intensity_mean:
        reasons.append(
            f"Torso region is severely underexposed (mean intensity {mean_intensity:.1f} < {config.min_intensity_mean})."
        )
        return False, reasons

    if mean_intensity > config.max_intensity_mean:
        reasons.append(
            f"Torso region is severely overexposed/glared (mean intensity {mean_intensity:.1f} > {config.max_intensity_mean})."
        )
        return False, reasons

    # Blur: Laplacian variance and standard deviation
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std_var = float(np.std(gray))

    if lap_var < config.min_laplacian_variance and std_var < 8.0:
        reasons.append(
            f"Torso region is severely blurred / lacking optical contrast (Laplacian variance {lap_var:.1f} < {config.min_laplacian_variance})."
        )
        return False, reasons

    return True, reasons


def crop_torso_region(
    image_bgr: np.ndarray,
    torso_bbox: Sequence[float],
    padding_fraction: float = 0.08,
) -> Tuple[np.ndarray, Tuple[float, float, float, float], Tuple[float, float]]:
    """
    Crop the torso region from the image with optional contextual padding.

    Returns
    -------
    crop : np.ndarray
    padded_box : [x1, y1, x2, y2]
    offset : (offset_x, offset_y) = (x1, y1)
    """
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [float(c) for c in torso_bbox[:4]]

    w = x2 - x1
    h = y2 - y1

    pad_w = w * padding_fraction
    pad_h = h * padding_fraction

    px1 = max(0.0, x1 - pad_w)
    py1 = max(0.0, y1 - pad_h)
    px2 = min(float(img_w), x2 + pad_w)
    py2 = min(float(img_h), y2 + pad_h)

    ix1, iy1, ix2, iy2 = int(round(px1)), int(round(py1)), int(round(px2)), int(round(py2))
    crop = image_bgr[iy1:iy2, ix1:ix2].copy()

    return crop, (px1, py1, px2, py2), (float(ix1), float(iy1))


class TorsoInferenceEngine:
    """
    Orchestrates torso extraction, spatial correlation, and three-state compliance assessment.
    """

    def __init__(self, config: Optional[VestDetectorConfig] = None):
        self.config = config or VestDetectorConfig()

    def assess(
        self,
        image_bgr: np.ndarray,
        torso_bbox: Sequence[float],
        detections: List[Detection],
    ) -> TorsoVestAssessment:
        """
        Execute three-state vest assessment on a specified torso region.

        Parameters
        ----------
        image_bgr : Complete input image in BGR format
        torso_bbox : [x1, y1, x2, y2] in original image coordinates
        detections : Detections from full image or torso crop, mapped to original image coordinates

        Returns
        -------
        TorsoVestAssessment with status in {PRESENT, ABSENT, UNKNOWN}
        """
        img_shape = image_bgr.shape[:2]
        geom_valid, geom_reasons, stats = validate_torso_geometry(torso_bbox, img_shape, self.config)

        # 1. If geometry is invalid or torso is too small/clipped -> UNKNOWN
        if not geom_valid:
            return TorsoVestAssessment(
                status=VestAssessmentState.UNKNOWN,
                confidence=0.50,
                vest_detected=False,
                detections=[],
                evidence=TorsoAssessmentEvidence(
                    torso_bbox=[round(c, 2) for c in torso_bbox[:4]],
                    torso_width=round(stats.get("width", 0.0), 2),
                    torso_height=round(stats.get("height", 0.0), 2),
                    torso_area=round(stats.get("area", 0.0), 2),
                    overlap_iou=0.0,
                    intersection_ratio=0.0,
                    decision_reasons=geom_reasons,
                    quality_checks_passed=False,
                ),
            )

        # 2. Extract torso crop and check optical quality
        crop, _, _ = crop_torso_region(image_bgr, torso_bbox, padding_fraction=0.0)
        quality_ok, quality_reasons = evaluate_torso_image_quality(crop, self.config)

        # 3. Analyze spatial overlap of candidate detections with the torso box
        tx1, ty1, tx2, ty2 = [float(c) for c in torso_bbox[:4]]
        tbox = [tx1, ty1, tx2, ty2]

        vest_matches: List[Tuple[Detection, float, float]] = []
        no_vest_matches: List[Tuple[Detection, float, float]] = []

        min_vest_conf = self.config.vest_confidence_threshold
        min_no_vest_conf = self.config.no_vest_confidence_threshold

        for det in detections:
            iou = compute_iou(det.bbox, tbox)
            inter_ratio = compute_intersection_ratio(det.bbox, tbox)
            overlaps = (iou >= self.config.min_overlap_iou or inter_ratio >= self.config.min_intersection_ratio)

            if overlaps:
                if det.class_name == "WITH_VEST" and det.confidence >= min_vest_conf:
                    vest_matches.append((det, iou, inter_ratio))
                elif det.class_name == "NO_VEST" and det.confidence >= min_no_vest_conf:
                    no_vest_matches.append((det, iou, inter_ratio))

        # Decision Logic:
        # Rule 1. PRESENT:
        #    - A valid WITH_VEST prediction exists.
        #    - Its confidence meets the configured threshold.
        #    - Its bounding box sufficiently overlaps the supplied torso.
        if vest_matches:
            vest_matches.sort(key=lambda x: x[0].confidence, reverse=True)
            best_det, best_iou, best_ratio = vest_matches[0]

            return TorsoVestAssessment(
                status=VestAssessmentState.PRESENT,
                confidence=best_det.confidence,
                vest_detected=True,
                detections=[d[0] for d in vest_matches],
                evidence=TorsoAssessmentEvidence(
                    torso_bbox=[round(c, 2) for c in torso_bbox[:4]],
                    torso_width=round(stats["width"], 2),
                    torso_height=round(stats["height"], 2),
                    torso_area=round(stats["area"], 2),
                    overlap_iou=round(best_iou, 4),
                    intersection_ratio=round(best_ratio, 4),
                    decision_reasons=[
                        f"Confirmed worn safety vest (confidence={best_det.confidence:.2f}, "
                        f"overlap IoU={best_iou:.2f}, intersection={best_ratio:.2f})."
                    ],
                    quality_checks_passed=quality_ok,
                ),
            )

        # Rule 2. ABSENT:
        #    - A valid NO_VEST prediction exists.
        #    - Its confidence meets a separately configurable threshold.
        #    - Its bounding box sufficiently overlaps the supplied torso.
        #    - Torso visibility, image quality and minimum size checks pass.
        if no_vest_matches:
            if not quality_ok:
                return TorsoVestAssessment(
                    status=VestAssessmentState.UNKNOWN,
                    confidence=0.50,
                    vest_detected=False,
                    detections=[d[0] for d in no_vest_matches],
                    evidence=TorsoAssessmentEvidence(
                        torso_bbox=[round(c, 2) for c in torso_bbox[:4]],
                        torso_width=round(stats["width"], 2),
                        torso_height=round(stats["height"], 2),
                        torso_area=round(stats["area"], 2),
                        overlap_iou=round(no_vest_matches[0][1], 4),
                        intersection_ratio=round(no_vest_matches[0][2], 4),
                        decision_reasons=quality_reasons + [
                            "Negative vest detection present but optical quality checks failed; cannot confirm ABSENT."
                        ],
                        quality_checks_passed=False,
                    ),
                )

            no_vest_matches.sort(key=lambda x: x[0].confidence, reverse=True)
            best_no, best_iou, best_ratio = no_vest_matches[0]

            return TorsoVestAssessment(
                status=VestAssessmentState.ABSENT,
                confidence=best_no.confidence,
                vest_detected=False,
                detections=[d[0] for d in no_vest_matches],
                evidence=TorsoAssessmentEvidence(
                    torso_bbox=[round(c, 2) for c in torso_bbox[:4]],
                    torso_width=round(stats["width"], 2),
                    torso_height=round(stats["height"], 2),
                    torso_area=round(stats["area"], 2),
                    overlap_iou=round(best_iou, 4),
                    intersection_ratio=round(best_ratio, 4),
                    decision_reasons=[
                        f"Validated NO_VEST prediction on torso (confidence={best_no.confidence:.2f}, "
                        f"overlap IoU={best_iou:.2f})."
                    ],
                    quality_checks_passed=True,
                ),
            )

        # Rule 3. UNKNOWN:
        #    - No valid positive or negative evidence exists.
        #    - Image is dark, blurry, too small or heavily occluded.
        #    - Detection confidence is insufficient.
        #    - Bounding boxes do not provide sufficient torso evidence.
        # Note: The fallback returning ABSENT with 0.88 is completely removed.
        # A missing detection returns UNKNOWN unless validated negative evidence supports ABSENT.
        unknown_reasons = list(quality_reasons)
        if not unknown_reasons:
            if not detections:
                unknown_reasons.append("No safety vest or negative vest evidence detected in the torso region.")
            else:
                candidate_labels = [f"{d.class_name} ({d.confidence:.2f})" for d in detections]
                unknown_reasons.append(
                    f"Candidate detection(s) found ({', '.join(candidate_labels)}) but none met the required "
                    f"confidence threshold or spatial overlap criteria for the supplied torso."
                )

        return TorsoVestAssessment(
            status=VestAssessmentState.UNKNOWN,
            confidence=0.50,
            vest_detected=False,
            detections=[],
            evidence=TorsoAssessmentEvidence(
                torso_bbox=[round(c, 2) for c in torso_bbox[:4]],
                torso_width=round(stats["width"], 2),
                torso_height=round(stats["height"], 2),
                torso_area=round(stats["area"], 2),
                overlap_iou=0.0,
                intersection_ratio=0.0,
                decision_reasons=unknown_reasons,
                quality_checks_passed=quality_ok,
            ),
        )
