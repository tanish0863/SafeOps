"""Geometric head-region estimation from COCO face keypoints.

This is a heuristic, NOT a trained head detector, and NOT helmet detection.
Missing face landmarks never imply "no head" or "no helmet". When the evidence
is insufficient the head region is None rather than an invented rectangle.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

FACE_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

# Tunable constants (validate against real helmet images).
EAR_TO_HEAD_WIDTH = 1.15       # head width ~ ear distance
EYE_TO_HEAD_WIDTH = 2.4        # head width ~ inter-eye distance
SHOULDER_TO_HEAD_WIDTH = 0.45  # fallback only; unreliable for bent/rotated people
EXTENT_TO_HEAD_WIDTH = 1.8     # fallback when only e.g. nose + one ear
HEAD_HEIGHT_RATIO = 1.25       # taller than wide; includes helmet crown
ANCHOR_FROM_TOP = 0.60         # mean face-point height sits ~60% down the box
MAX_HEAD_FRACTION = 0.60       # head width <= this * min(bbox w, bbox h)
MIN_HEAD_PX = 8.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _pt(kp: Optional[dict]) -> Optional[Tuple[float, float]]:
    return None if kp is None else (float(kp["x"]), float(kp["y"]))


def classify_visibility(n_reliable: int) -> str:
    """VISIBLE only when all five face landmarks (nose, both eyes, both ears) are reliable."""
    if n_reliable >= 5:
        return "VISIBLE"
    if n_reliable >= 2:
        return "PARTIAL"
    return "UNCERTAIN"


def _head_width(keypoints: Dict[str, Optional[dict]], face: Dict[str, Tuple[float, float]],
                extent_x: float) -> Optional[float]:
    if "left_ear" in face and "right_ear" in face:
        return _dist(face["left_ear"], face["right_ear"]) * EAR_TO_HEAD_WIDTH
    if "left_eye" in face and "right_eye" in face:
        return _dist(face["left_eye"], face["right_eye"]) * EYE_TO_HEAD_WIDTH
    ls, rs = _pt(keypoints.get("left_shoulder")), _pt(keypoints.get("right_shoulder"))
    if ls and rs:
        return _dist(ls, rs) * SHOULDER_TO_HEAD_WIDTH
    if extent_x >= 4:
        return extent_x * EXTENT_TO_HEAD_WIDTH
    return None


def estimate_head_region(
    keypoints: Dict[str, Optional[dict]],
    frame_width: int,
    frame_height: int,
    person_bbox: Optional[List[float]] = None,
) -> dict:
    """keypoints: name -> {"x","y","confidence"} or None (already confidence-filtered)."""
    face = {}
    for n in FACE_NAMES:
        p = _pt(keypoints.get(n))
        if p is not None:
            face[n] = p
    visibility = classify_visibility(len(face))
    if len(face) < 2:
        return {"head_region": None, "head_visibility": "UNCERTAIN"}

    xs = [p[0] for p in face.values()]
    ys = [p[1] for p in face.values()]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)

    width = _head_width(keypoints, face, max(xs) - min(xs))
    if width is None:
        return {"head_region": None, "head_visibility": "UNCERTAIN"}

    if person_bbox is not None:
        bw, bh = person_bbox[2] - person_bbox[0], person_bbox[3] - person_bbox[1]
        width = min(width, MAX_HEAD_FRACTION * min(bw, bh))
    width = max(width, MIN_HEAD_PX)
    height = width * HEAD_HEIGHT_RATIO

    x1, x2 = cx - width / 2, cx + width / 2
    y1, y2 = cy - ANCHOR_FROM_TOP * height, cy + (1 - ANCHOR_FROM_TOP) * height
    x1, x2 = max(0.0, x1), min(float(frame_width), x2)
    y1, y2 = max(0.0, y1), min(float(frame_height), y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return {"head_region": None, "head_visibility": "UNCERTAIN"}

    return {"head_region": [round(v, 1) for v in (x1, y1, x2, y2)],
            "head_visibility": visibility}
