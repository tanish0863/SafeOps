"""
vest_detector
=============
Independent Safety Vest Detection Module for SafeOps Phase 6.

Provides full-frame inference, localized torso crop inference,
and three-state compliance assessment (PRESENT, ABSENT, UNKNOWN).
"""

from .config import VestDetectorConfig
from .schemas import (
    Detection,
    TorsoAssessmentEvidence,
    TorsoVestAssessment,
    VestAssessmentState,
    VestClassName,
    VestDetectionResult,
)
from .vest_detector import VestDetector

__version__ = "1.0.0"

__all__ = [
    "VestDetector",
    "VestDetectorConfig",
    "Detection",
    "VestDetectionResult",
    "VestAssessmentState",
    "VestClassName",
    "TorsoVestAssessment",
    "TorsoAssessmentEvidence",
]
