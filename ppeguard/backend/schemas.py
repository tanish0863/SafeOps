"""
Pydantic v2 schemas for the SafeOps REST API.
Strictly matches frontend contracts in src/data/demoData.ts and src/services/detectionService.ts.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


ComplianceStatus = Literal["compliant", "violation", "review"]
Severity = Literal["critical", "high", "medium", "low"]
ReviewStatus = Literal["confirmed", "pending", "resolved"]
PpeStatus = Literal["detected", "missing", "uncertain"]
ShiftType = Literal["A", "B", "C"]


# ---------------------------------------------------------------------------
# Worker Schemas
# ---------------------------------------------------------------------------

class WorkerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    zone: str = Field(..., min_length=1, max_length=100)
    shift: ShiftType = "A"


class WorkerCreate(WorkerBase):
    id: Optional[str] = None
    helmet: bool = False
    vest: bool = False
    status: ComplianceStatus = "compliant"
    needsReview: bool = False


class WorkerUpdate(BaseModel):
    name: Optional[str] = None
    zone: Optional[str] = None
    shift: Optional[ShiftType] = None
    helmet: Optional[bool] = None
    vest: Optional[bool] = None
    status: Optional[ComplianceStatus] = None
    needsReview: Optional[bool] = None
    lastScan: Optional[str] = None
    scans: Optional[int] = None


class WorkerResponse(WorkerBase):
    id: str
    lastScan: str
    scans: int
    helmet: bool
    vest: bool
    status: ComplianceStatus
    needsReview: bool

    model_config = ConfigDict(from_attributes=True)


class WorkerListResponse(BaseModel):
    total: int
    data: List[WorkerResponse]


# ---------------------------------------------------------------------------
# Violation Schemas
# ---------------------------------------------------------------------------

class ViolationBase(BaseModel):
    workerId: str
    workerName: str
    type: str
    zone: str
    severity: Severity = "medium"
    reviewStatus: ReviewStatus = "pending"
    note: str = ""


class ViolationCreate(ViolationBase):
    id: Optional[str] = None
    timestamp: Optional[str] = None
    evidenceCropUrl: Optional[str] = None


class ViolationUpdate(BaseModel):
    reviewStatus: Optional[ReviewStatus] = None
    severity: Optional[Severity] = None
    note: Optional[str] = None


class ViolationResponse(ViolationBase):
    id: str
    timestamp: str
    evidenceCropUrl: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ViolationListResponse(BaseModel):
    total: int
    data: List[ViolationResponse]


# ---------------------------------------------------------------------------
# Media & Detection Schemas
# ---------------------------------------------------------------------------

class UploadedMediaResponse(BaseModel):
    mediaId: str
    fileName: str
    mediaType: Literal["image", "video"]
    fileSizeBytes: int
    storageUrl: str
    uploadedAt: str


class PpeRules(BaseModel):
    helmetRequired: bool = True
    vestRequired: bool = True


class DetectionJobCreate(BaseModel):
    mediaId: Optional[str] = None
    fileName: Optional[str] = "inspection_frame.jpg"
    rules: PpeRules = Field(default_factory=PpeRules)


class DetectionJobResponse(BaseModel):
    jobId: str
    mediaId: Optional[str] = None
    status: Literal["queued", "processing", "complete", "failed"]
    createdAt: str


class NormalizedBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class PpeDetection(BaseModel):
    kind: Literal["helmet", "vest"]
    status: PpeStatus
    box: Optional[NormalizedBox] = None
    confidence: float = 0.0
    evaluated: bool = True


class WorkspaceWorker(BaseModel):
    workerId: str
    box: NormalizedBox
    personConfidence: float
    helmet: PpeDetection
    vest: PpeDetection
    compliance: ComplianceStatus
    associationConfidence: float
    reviewReason: Optional[str] = None


class WorkspaceAnalysis(BaseModel):
    jobId: str
    fileName: str
    analyzedAt: str
    model: str
    isDemo: bool = True
    workers: List[WorkspaceWorker]


# ---------------------------------------------------------------------------
# Dashboard & Analytics Schemas
# ---------------------------------------------------------------------------

class MetricDelta(BaseModel):
    value: str
    delta: str


class OverviewStats(BaseModel):
    complianceRate: float = 81.8
    totalWorkers: int = 148
    compliantCount: int = 121
    violationCount: int = 19
    reviewCount: int = 8
    criticalCount: int = 3
    workersDetected: MetricDelta = Field(default_factory=lambda: MetricDelta(value="148", delta="+6 this week"))
    compliant: MetricDelta = Field(default_factory=lambda: MetricDelta(value="121", delta="81.8% compliance"))
    violations: MetricDelta = Field(default_factory=lambda: MetricDelta(value="19", delta="3 critical"))
    needsReview: MetricDelta = Field(default_factory=lambda: MetricDelta(value="8", delta="4 flagged today"))


class ActivityItem(BaseModel):
    id: str
    time: str
    message: str
    kind: Literal["scan", "violation", "review"]


class TelemetryResponse(BaseModel):
    activeSensors: int = 8
    totalSensors: int = 8
    fps: float = 30.4
    latencyMs: float = 38.2
    trackedPersonnel: int = 148
    cudaActive: bool = True
    pipelineVersion: str = "v2.4 (Edge-Optimized)"
    activePlant: str = "Facility Unit 01"


class TrendPoint(BaseModel):
    label: str
    rate: float


class ZoneStat(BaseModel):
    zone: str
    violations: int
    compliance: float


class PpeBreakdownItem(BaseModel):
    ppe: str
    violations: int


class AnalyticsResponse(BaseModel):
    complianceTrend: List[TrendPoint]
    zoneStats: List[ZoneStat]
    ppeBreakdown: List[PpeBreakdownItem]


# ---------------------------------------------------------------------------
# Settings & Camera Schemas
# ---------------------------------------------------------------------------

class SystemSettingsDTO(BaseModel):
    personConfidenceThreshold: float = 0.50
    ppeConfidenceThreshold: float = 0.40
    torsoOverlapThreshold: float = 0.30
    alertCriticalViolations: bool = True
    alertReviewRequired: bool = False
    enableAudioAlerts: bool = True
    retentionDays: int = 30
    webhookUrl: Optional[str] = "https://hooks.safeops.local/alerts"
    notificationEmail: Optional[str] = "safety-ops@safeops.local"


class CameraDeviceCreate(BaseModel):
    name: str
    zone: str
    rtspUrl: str
    targetFps: int = 30
    enabled: bool = True


class CameraDeviceResponse(CameraDeviceCreate):
    id: str
    status: Literal["online", "offline", "degraded"] = "online"
    lastSeen: str

    model_config = ConfigDict(from_attributes=True)
