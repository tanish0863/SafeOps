"""
schemas.py
----------
Pydantic data models and schemas for Safety Vest Detection.
Defines strict detection contracts, bounding box representation,
and three-state vest compliance assessments.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VestClassName(str, Enum):
    """Canonical class identifiers for vest detection outputs."""
    WITH_VEST = "WITH_VEST"
    NO_VEST = "NO_VEST"


class VestAssessmentState(str, Enum):
    """
    Three-state operational assessment for safety vest compliance.

    PRESENT: A confident vest detection exists with credible torso overlap.
    ABSENT:  Torso is reliably visible and clear, with validated absence of vest.
    UNKNOWN: Ambiguous, occluded, too distant, poorly lit, or out of frame.
    """
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class Detection(StrictBase):
    """
    Single bounding box detection in original-image pixel space.

    Coordinates must be [x1, y1, x2, y2] in absolute pixels of the full frame.
    """
    class_name: str = Field(
        ...,
        description="Detection label, typically 'WITH_VEST' or 'NO_VEST'."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in range [0.0, 1.0]."
    )
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="[x1, y1, x2, y2] in absolute original-image pixel coordinates."
    )

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: List[float]) -> List[float]:
        x1, y1, x2, y2 = v
        if x2 < x1:
            raise ValueError(f"Invalid bbox: x2 ({x2}) must be >= x1 ({x1})")
        if y2 < y1:
            raise ValueError(f"Invalid bbox: y2 ({y2}) must be >= y1 ({y1})")
        return [float(x) for x in v]


class VestDetectionResult(StrictBase):
    """
    Top-level output schema matching SafeOps specification.
    """
    schema_version: str = Field(default="1.0", description="Contract version.")
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detections in original-image pixel coordinates."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic metadata (inference time, input shape, model info)."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to standard Python dictionary matching the required JSON format."""
        return self.model_dump(exclude_none=True)


class TorsoAssessmentEvidence(StrictBase):
    """Diagnostic details supporting the three-state vest assessment."""
    torso_bbox: List[float]
    torso_width: float
    torso_height: float
    torso_area: float
    overlap_iou: Optional[float] = None
    intersection_ratio: Optional[float] = None
    decision_reasons: List[str] = Field(default_factory=list)
    quality_checks_passed: bool = True


class TorsoVestAssessment(StrictBase):
    """
    Comprehensive three-state vest evaluation for a specific worker torso region.
    """
    status: VestAssessmentState = Field(
        ...,
        description="Assessment verdict: PRESENT, ABSENT, or UNKNOWN."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of the assessment verdict."
    )
    vest_detected: bool = Field(
        default=False,
        description="True if a worn safety vest was positively confirmed."
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="Detections mapped to original-image coordinates."
    )
    evidence: TorsoAssessmentEvidence = Field(
        ...,
        description="Audit trail explaining the decision."
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
