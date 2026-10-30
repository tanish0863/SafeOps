"""Pydantic models that define the Phase 2 -> Phase 3 JSON contract."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

HeadVisibility = Literal["VISIBLE", "PARTIAL", "UNCERTAIN"]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float


class Detection(BaseModel):
    """Component A output. detection_id is frame-local (tracking IDs come later)."""
    detection_id: str
    bbox: List[float] = Field(min_length=4, max_length=4)  # x1, y1, x2, y2 (pixels)
    confidence: float
    frame_width: int
    frame_height: int


class Pose(BaseModel):
    """Component B output. All 17 names present; unreliable points are None."""
    detection_id: str
    keypoints: Dict[str, Optional[Keypoint]]


class Person(BaseModel):
    detection_id: str
    bbox: List[float] = Field(min_length=4, max_length=4)
    confidence: float
    head_region: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    head_visibility: HeadVisibility
    keypoints: Dict[str, Optional[Keypoint]]


class FrameRecord(BaseModel):
    """One JSONL line per processed frame."""
    frame_number: int
    timestamp_s: Optional[float] = None
    frame_width: int
    frame_height: int
    persons: List[Person]
