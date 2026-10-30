"""Safety Policy Compliance Engine for SafeOps.

Decouples compliance evaluation rules from model inference:
- Applies configurable safety policies (e.g., helmet mandatory, vest optional/uninstalled).
- Integrates temporal smoothing from TrackPPEHistory.
- Distinguishes compliant, non-compliant, insufficient_evidence, and unknown.
- Tracks non-compliance duration across consecutive frames.
"""
from __future__ import annotations

from typing import Dict, Optional

from src.compliance.temporal_state import TemporalConsistencyTracker, TrackPPEHistory
from src.contracts.schemas import (
    ComplianceStatus,
    PPEItemType,
    SafetyPolicyConfig,
    WorkerComplianceState,
    WorkerPPEAssociation,
)


class ComplianceEngine:
    """Evaluates individual worker compliance against safety policies."""

    def __init__(
        self,
        policy: Optional[SafetyPolicyConfig] = None,
        temporal_window_seconds: float = 2.0,
        fps: float = 30.0,
    ):
        self.policy = policy or SafetyPolicyConfig()
        self.temporal_tracker = TemporalConsistencyTracker(
            window_seconds=temporal_window_seconds,
            default_fps=fps,
        )
        # Track non-compliance duration per track ID: tid -> consecutive non-compliant seconds
        self._non_compliant_seconds: Dict[int, float] = {}
        self._prev_timestamps: Dict[int, float] = {}

    def reset(self) -> None:
        """Reset temporal state between independent video streams."""
        self.temporal_tracker.reset()
        self._non_compliant_seconds.clear()
        self._prev_timestamps.clear()

    def evaluate_worker(
        self,
        association: WorkerPPEAssociation,
        timestamp_seconds: float,
    ) -> WorkerComplianceState:
        """Evaluate smoothed compliance for a worker in the current frame."""
        tid = association.track_id

        # 1. Update temporal history
        history = self.temporal_tracker.update(
            track_id=tid,
            helmet=association.helmet,
            vest=association.vest,
            timestamp_seconds=timestamp_seconds,
        )

        # 2. Evaluate requirements independently
        helmet_compliance = ComplianceStatus.UNKNOWN
        if self.policy.require_helmet:
            helmet_compliance = history.evaluate_item_status(
                PPEItemType.HELMET,
                min_valid_observations=self.policy.min_observations_for_decision,
                current_timestamp=timestamp_seconds,
            )

        vest_compliance = ComplianceStatus.UNKNOWN
        if self.policy.require_vest:
            vest_compliance = history.evaluate_item_status(
                PPEItemType.VEST,
                min_valid_observations=self.policy.min_observations_for_decision,
                current_timestamp=timestamp_seconds,
            )

        # 3. Derive Overall Compliance State
        # If any mandatory requirement is NON_COMPLIANT -> NON_COMPLIANT
        # Else if any mandatory requirement is INSUFFICIENT_EVIDENCE -> INSUFFICIENT_EVIDENCE
        # Else if any mandatory requirement is UNKNOWN -> UNKNOWN
        # Else -> COMPLIANT
        mandatory_statuses = []
        if self.policy.require_helmet:
            mandatory_statuses.append(helmet_compliance)
        if self.policy.require_vest:
            mandatory_statuses.append(vest_compliance)

        if not mandatory_statuses:
            overall = ComplianceStatus.COMPLIANT
        elif any(s == ComplianceStatus.NON_COMPLIANT for s in mandatory_statuses):
            overall = ComplianceStatus.NON_COMPLIANT
        elif any(s == ComplianceStatus.INSUFFICIENT_EVIDENCE for s in mandatory_statuses):
            overall = ComplianceStatus.INSUFFICIENT_EVIDENCE
        elif any(s == ComplianceStatus.UNKNOWN for s in mandatory_statuses):
            overall = ComplianceStatus.UNKNOWN
        else:
            overall = ComplianceStatus.COMPLIANT

        # 4. Track consecutive non-compliance duration
        prev_t = self._prev_timestamps.get(tid, timestamp_seconds)
        dt = max(0.0, timestamp_seconds - prev_t)
        self._prev_timestamps[tid] = timestamp_seconds

        if overall == ComplianceStatus.NON_COMPLIANT:
            current_duration = self._non_compliant_seconds.get(tid, 0.0) + dt
            self._non_compliant_seconds[tid] = current_duration
        elif overall == ComplianceStatus.COMPLIANT:
            self._non_compliant_seconds[tid] = 0.0
        else:
            # UNKNOWN or INSUFFICIENT_EVIDENCE: pause timer, do NOT increment duration
            pass

        consec_sec = round(self._non_compliant_seconds.get(tid, 0.0), 3)
        is_sustained_violation = (
            overall == ComplianceStatus.NON_COMPLIANT
            and consec_sec >= self.policy.violation_confirmation_seconds
        )

        details = None
        if overall == ComplianceStatus.NON_COMPLIANT:
            reasons = []
            if helmet_compliance == ComplianceStatus.NON_COMPLIANT:
                reasons.append("helmet_missing")
            if vest_compliance == ComplianceStatus.NON_COMPLIANT:
                reasons.append("vest_missing")
            details = f"Violations: {', '.join(reasons)} (persisting {consec_sec:.1f}s)"
        elif overall == ComplianceStatus.INSUFFICIENT_EVIDENCE:
            details = f"Gathering evidence ({history.frames_present}/{self.policy.min_observations_for_decision} frames)"
        elif overall == ComplianceStatus.UNKNOWN:
            details = "Mandatory PPE model unavailable or uninstalled"

        return WorkerComplianceState(
            track_id=tid,
            overall_status=overall,
            helmet_status=helmet_compliance,
            vest_status=vest_compliance,
            observation_count=history.frames_present,
            consecutive_non_compliant_seconds=consec_sec,
            active_violation=is_sustained_violation,
            details=details,
        )

