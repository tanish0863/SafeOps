"""Versioned Data Contracts for SafeOps Production PPE Compliance System.

Version: 1.0.0
Defines schemas for:
- Video metadata
- Worker detections & tracking records
- PPE detections & worker-level association
- Temporal consistency & compliance evaluations
- Violation events
- Final aggregated analysis reports
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PPEItemType(str, Enum):
    HELMET = "helmet"
    VEST = "vest"


class PPEStatus(str, Enum):
    DETECTED = "detected"
    NOT_DETECTED = "not_detected"
    UNKNOWN = "unknown"


class ComplianceStatus(str, Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNKNOWN = "unknown"


class TrackingContinuityState(str, Enum):
    CONFIRMED = "confirmed"
    UNCERTAIN = "uncertain"
    RECOVERED = "recovered"


class ViolationStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


class HeadROIQuality(str, Enum):
    CONFIRMED = "confirmed"
    APPROXIMATE = "approximate"
    UNKNOWN = "unknown"


# ============================================================================
# Video & Streaming Schemas
# ============================================================================

class VideoMetadata(BaseModel):
    video_id: str = Field(default="video", description="Unique video identifier")
    filename: str = Field(default="video.mp4", description="Original video file name")
    fps: float = Field(..., gt=0, description="Frames per second")
    width: int = Field(..., gt=0, description="Video width in pixels")
    height: int = Field(..., gt=0, description="Video height in pixels")
    total_frames: int = Field(..., ge=0, description="Total frame count in video")
    duration_seconds: float = Field(default=0.0, ge=0, description="Video duration in seconds")

    @property
    def duration(self) -> float:
        if self.duration_seconds > 0.0:
            return self.duration_seconds
        return (self.total_frames / self.fps) if self.fps > 0 else 0.0


# ============================================================================
# PPE Detection & Association Schemas
# ============================================================================

class PPEDetectionItem(BaseModel):
    item_type: PPEItemType
    bbox_xyxy: List[float] = Field(..., min_length=4, max_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    raw_class: Optional[str] = None


class WorkerPPEState(BaseModel):
    status: PPEStatus
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    bbox_xyxy: Optional[List[float]] = Field(None, min_length=4, max_length=4)
    reason: Optional[str] = None


class WorkerPPEAssociation(BaseModel):
    track_id: int
    helmet: WorkerPPEState
    vest: WorkerPPEState
    unassociated_ppe_count: int = 0
    head_roi_quality: Optional[HeadROIQuality] = None


# ============================================================================
# Tracking & Frame Records
# ============================================================================

class WorkerTrackingRecord(BaseModel):
    frame_number: int
    timestamp_seconds: float
    track_id: int
    bbox_xyxy: List[float]
    confidence: float
    keypoints_xy: Optional[List[List[float]]] = None
    keypoints_conf: Optional[List[float]] = None
    continuity_state: TrackingContinuityState = TrackingContinuityState.CONFIRMED
    ppe: WorkerPPEAssociation
    compliance: Optional[Dict[str, Any]] = None


# ============================================================================
# Compliance & Violation Schemas
# ============================================================================

class SafetyPolicyConfig(BaseModel):
    policy_name: str = "Standard Industrial Safety Policy"
    require_helmet: bool = True
    require_vest: bool = False  # Set False when vest model is uninstalled/unknown
    min_observations_for_decision: int = 15  # Minimum frames (~0.5s at 30fps)
    violation_confirmation_seconds: float = 1.0  # Duration non-compliance must persist
    recovery_confirmation_seconds: float = 1.0   # Duration compliance must persist to resolve


class WorkerComplianceState(BaseModel):
    track_id: int
    overall_status: ComplianceStatus
    helmet_status: ComplianceStatus
    vest_status: ComplianceStatus
    observation_count: int
    consecutive_non_compliant_seconds: float = 0.0
    active_violation: bool = False
    details: Optional[str] = None


class ViolationEvent(BaseModel):
    event_id: str
    track_id: int
    violation_type: str  # e.g., "missing_helmet", "missing_vest"
    start_timestamp: float
    end_timestamp: Optional[float] = None
    status: ViolationStatus
    evidence_frames: List[int] = Field(default_factory=list)
    confidence: float = 1.0
    summary: str


# ============================================================================
# Final Analysis & API Reports
# ============================================================================

class ProcessingSummary(BaseModel):
    total_frames_processed: int
    duration_seconds: float
    average_fps: float
    unique_workers_tracked: int
    total_violations_detected: int
    compliance_rate_percent: float


class WorkerSummary(BaseModel):
    track_id: int
    first_seen_timestamp: float
    last_seen_timestamp: float
    total_frames_present: int
    compliance_status: ComplianceStatus
    violations_count: int


class VideoAnalysisReport(BaseModel):
    version: str = "1.0.0"
    video_id: str
    metadata: VideoMetadata
    policy: SafetyPolicyConfig
    summary: ProcessingSummary
    workers: List[WorkerSummary]
    violations: List[ViolationEvent]
    created_at: str
