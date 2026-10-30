"""Comprehensive Unit Tests for SafeOps Temporal Consistency & Compliance Engine.

Validates:
1. Temporal smoothing over sliding observation window
2. Distinguishing COMPLIANT, NON_COMPLIANT, INSUFFICIENT_EVIDENCE, and UNKNOWN
3. Prevention of false-positive violations from brief 1-2 frame detection drops
4. Violation confirmation duration requirements
5. Violation recovery and event resolution
6. Event deduplication: single event per violation episode
"""
import pytest

from src.compliance.compliance_engine import ComplianceEngine
from src.compliance.temporal_state import TemporalConsistencyTracker, TrackPPEHistory
from src.compliance.violation_manager import ViolationManager
from src.contracts.schemas import (
    ComplianceStatus,
    PPEItemType,
    PPEStatus,
    SafetyPolicyConfig,
    ViolationStatus,
    WorkerPPEAssociation,
    WorkerPPEState,
)


def make_assoc(tid: int, helmet_status: PPEStatus, vest_status: PPEStatus = PPEStatus.UNKNOWN) -> WorkerPPEAssociation:
    return WorkerPPEAssociation(
        track_id=tid,
        helmet=WorkerPPEState(status=helmet_status, confidence=0.9 if helmet_status == PPEStatus.DETECTED else None),
        vest=WorkerPPEState(status=vest_status),
    )


def test_insufficient_evidence_initially():
    """Worker in first few frames must report INSUFFICIENT_EVIDENCE, not non-compliant."""
    policy = SafetyPolicyConfig(require_helmet=True, min_observations_for_decision=10)
    engine = ComplianceEngine(policy=policy)

    for f in range(5):
        assoc = make_assoc(1, PPEStatus.NOT_DETECTED)
        c = engine.evaluate_worker(assoc, timestamp_seconds=f * 0.033)
        assert c.overall_status == ComplianceStatus.INSUFFICIENT_EVIDENCE
        assert c.active_violation is False


def test_sustained_non_compliance_triggers_violation():
    """Continuous missing helmet beyond confirmation window triggers active violation."""
    policy = SafetyPolicyConfig(
        require_helmet=True,
        min_observations_for_decision=10,
        violation_confirmation_seconds=1.0,
    )
    engine = ComplianceEngine(policy=policy)
    vm = ViolationManager(recovery_confirmation_seconds=1.0)

    # 45 frames (~1.5s at 30fps) with missing helmet
    all_events = []
    for f in range(45):
        t = f * 0.033
        assoc = make_assoc(1, PPEStatus.NOT_DETECTED)
        c = engine.evaluate_worker(assoc, timestamp_seconds=t)
        events = vm.process_worker_compliance(c, frame_number=f, timestamp_seconds=t)
        all_events.extend(events)

    # After 1.0s of continuous non-compliance (from frame 30 onwards), active_violation should be True
    last_c = engine.evaluate_worker(make_assoc(1, PPEStatus.NOT_DETECTED), timestamp_seconds=1.5)
    assert last_c.active_violation is True

    # Violation manager must record confirmed event
    confirmed_events = [e for e in vm.get_all_events() if e.status == ViolationStatus.CONFIRMED]
    assert len(confirmed_events) == 1
    assert confirmed_events[0].track_id == 1
    assert confirmed_events[0].violation_type == "missing_helmet"


def test_temporary_dropout_does_not_break_compliance():
    """A compliant worker with 2 transient missing helmet frames stays COMPLIANT."""
    policy = SafetyPolicyConfig(
        require_helmet=True,
        min_observations_for_decision=10,
    )
    engine = ComplianceEngine(policy=policy)

    # 20 frames with helmet detected
    for f in range(20):
        c = engine.evaluate_worker(make_assoc(1, PPEStatus.DETECTED), timestamp_seconds=f * 0.033)
    assert c.overall_status == ComplianceStatus.COMPLIANT

    # 2 transient frames of missing detection (e.g. head turned, occlusion)
    c1 = engine.evaluate_worker(make_assoc(1, PPEStatus.NOT_DETECTED), timestamp_seconds=20 * 0.033)
    c2 = engine.evaluate_worker(make_assoc(1, PPEStatus.NOT_DETECTED), timestamp_seconds=21 * 0.033)

    # Because 20/22 observations are DETECTED (> 50%), worker remains COMPLIANT
    assert c1.overall_status == ComplianceStatus.COMPLIANT
    assert c2.overall_status == ComplianceStatus.COMPLIANT
    assert c2.active_violation is False


def test_violation_recovery_and_resolution():
    """When a non-compliant worker puts on a helmet, the violation resolves."""
    policy = SafetyPolicyConfig(
        require_helmet=True,
        min_observations_for_decision=5,
        violation_confirmation_seconds=0.5,
        recovery_confirmation_seconds=0.5,
    )
    engine = ComplianceEngine(policy=policy)
    vm = ViolationManager(recovery_confirmation_seconds=0.5)

    # 1. Cause confirmed violation (30 frames no helmet)
    for f in range(30):
        t = f * 0.033
        c = engine.evaluate_worker(make_assoc(1, PPEStatus.NOT_DETECTED), timestamp_seconds=t)
        vm.process_worker_compliance(c, frame_number=f, timestamp_seconds=t)

    assert any(e.status == ViolationStatus.CONFIRMED for e in vm.get_all_events())

    # 2. Worker puts on helmet for 60 frames (~2.0s) to clear history window and confirm recovery
    for f in range(30, 90):
        t = f * 0.033
        c = engine.evaluate_worker(make_assoc(1, PPEStatus.DETECTED), timestamp_seconds=t)
        vm.process_worker_compliance(c, frame_number=f, timestamp_seconds=t)

    # Event should be resolved
    all_events = vm.finalize_stream(final_timestamp=90 * 0.033)
    assert len(all_events) == 1
    assert all_events[0].status == ViolationStatus.RESOLVED


def test_missing_vest_model_reports_unknown():
    """When vest model is uninstalled/unknown, it never produces false non-compliance."""
    policy = SafetyPolicyConfig(require_helmet=True, require_vest=True, min_observations_for_decision=5)
    engine = ComplianceEngine(policy=policy)

    # Worker wears helmet, but vest is UNKNOWN
    for f in range(10):
        assoc = make_assoc(1, helmet_status=PPEStatus.DETECTED, vest_status=PPEStatus.UNKNOWN)
        c = engine.evaluate_worker(assoc, timestamp_seconds=f * 0.033)

    assert c.helmet_status == ComplianceStatus.COMPLIANT
    assert c.vest_status == ComplianceStatus.UNKNOWN
    # Overall status is UNKNOWN because mandatory vest cannot be verified
    assert c.overall_status == ComplianceStatus.UNKNOWN
    assert c.active_violation is False