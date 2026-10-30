"""Regression tests for Phase 5 PPE association and compliance identity integrity.

Verifies:
- Test A: Worker Crossing (no status stealing, ambiguous produces UNKNOWN, no false violations)
- Test B: Temporary Occlusion (status UNKNOWN, compliance timers pause, no status transfer)
- Test C: Detection Blackout (recovered track resumes, no history transfer, violation resolves on compliance)
- Test D: Exit and Re-entry (new track starts fresh evidence, no history transfer)
"""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("."))
from src.contracts.schemas import (
    ComplianceStatus,
    HeadROIQuality,
    PPEDetectionItem,
    PPEItemType,
    PPEStatus,
    SafetyPolicyConfig,
    ViolationStatus,
    WorkerPPEAssociation,
    WorkerPPEState,
)
from src.compliance.compliance_engine import ComplianceEngine
from src.compliance.violation_manager import ViolationManager
from src.association.ppe_association import associate_ppe_to_workers


def test_audit_situation_a_worker_crossing():
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Pre-crossing: Worker 1 (helmet), Worker 2 (no helmet)
    w1_pre = {"track_id": 1, "bbox_xyxy": [100.0, 100.0, 200.0, 400.0], "keypoints_xy": [[150.0, 130.0]] * 17, "keypoints_conf": [0.95] * 17}
    w2_pre = {"track_id": 2, "bbox_xyxy": [400.0, 100.0, 500.0, 400.0], "keypoints_xy": [[450.0, 130.0]] * 17, "keypoints_conf": [0.95] * 17}
    helmet_det = PPEDetectionItem(item_type=PPEItemType.HELMET, bbox_xyxy=[130.0, 95.0, 170.0, 135.0], confidence=0.94)

    for fn in range(30):
        t = fn * 0.033
        assocs, _ = associate_ppe_to_workers([w1_pre, w2_pre], [helmet_det], has_vest_model=False)
        c1 = engine.evaluate_worker(assocs[1], timestamp_seconds=t)
        c2 = engine.evaluate_worker(assocs[2], timestamp_seconds=t)
        mgr.process_worker_compliance(c1, fn, t)
        mgr.process_worker_compliance(c2, fn, t)

    assert assocs[1].helmet.status == PPEStatus.DETECTED
    assert assocs[2].helmet.status == PPEStatus.NOT_DETECTED

    # Crossing point: ambiguous proximity
    w1_cross = {"track_id": 1, "bbox_xyxy": [250.0, 100.0, 350.0, 400.0], "keypoints_xy": [[300.0, 130.0]] * 17, "keypoints_conf": [0.95] * 17}
    w2_cross = {"track_id": 2, "bbox_xyxy": [260.0, 100.0, 360.0, 400.0], "keypoints_xy": [[310.0, 130.0]] * 17, "keypoints_conf": [0.95] * 17}
    helmet_cross = PPEDetectionItem(item_type=PPEItemType.HELMET, bbox_xyxy=[285.0, 95.0, 325.0, 135.0], confidence=0.91)

    assocs_cross, _ = associate_ppe_to_workers([w1_cross, w2_cross], [helmet_cross], ambiguity_margin=0.15, has_vest_model=False)

    # Both must be marked UNKNOWN, no helmet theft
    assert assocs_cross[1].helmet.status == PPEStatus.UNKNOWN
    assert assocs_cross[2].helmet.status == PPEStatus.UNKNOWN

    # Compliance does not trigger false violation on worker 1
    c1_cross = engine.evaluate_worker(assocs_cross[1], timestamp_seconds=1.0)
    c2_cross = engine.evaluate_worker(assocs_cross[2], timestamp_seconds=1.0)
    mgr.process_worker_compliance(c1_cross, 31, 1.0)
    mgr.process_worker_compliance(c2_cross, 31, 1.0)

    w1_violations = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    assert len(w1_violations) == 0


def test_audit_situation_b_temporary_occlusion():
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Compliant worker
    assoc_compliant = WorkerPPEAssociation(
        track_id=1,
        helmet=WorkerPPEState(status=PPEStatus.DETECTED, confidence=0.92),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    for fn in range(30):
        t = fn * 0.033
        c = engine.evaluate_worker(assoc_compliant, timestamp_seconds=t)
        mgr.process_worker_compliance(c, fn, t)

    assert c.overall_status == ComplianceStatus.COMPLIANT

    # Occluded worker: head ROI UNKNOWN
    assoc_occluded = WorkerPPEAssociation(
        track_id=1,
        helmet=WorkerPPEState(status=PPEStatus.UNKNOWN),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.UNKNOWN,
    )
    for fn in range(30, 60):
        t = fn * 0.033
        c_occ = engine.evaluate_worker(assoc_occluded, timestamp_seconds=t)
        mgr.process_worker_compliance(c_occ, fn, t)

    assert c_occ.overall_status == ComplianceStatus.UNKNOWN
    # Timer paused: consecutive non compliant seconds remains 0.0
    assert c_occ.consecutive_non_compliant_seconds == 0.0
    # No false violation
    w1_violations = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    assert len(w1_violations) == 0


def test_audit_situation_c_detection_blackout():
    policy = SafetyPolicyConfig(require_helmet=True, min_observations_for_decision=5, violation_confirmation_seconds=0.5)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=0.5)
    engine.reset()
    mgr.reset()

    assoc_violator = WorkerPPEAssociation(
        track_id=1,
        helmet=WorkerPPEState(status=PPEStatus.NOT_DETECTED),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    for fn in range(35):
        t = fn * 0.033
        c = engine.evaluate_worker(assoc_violator, timestamp_seconds=t)
        mgr.process_worker_compliance(c, fn, t)

    active_v = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    assert len(active_v) == 1

    # Blackout & return compliant
    assoc_recovered = WorkerPPEAssociation(
        track_id=1,
        helmet=WorkerPPEState(status=PPEStatus.DETECTED, confidence=0.93),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    for fn in range(45, 100):
        t = fn * 0.033
        c_rec = engine.evaluate_worker(assoc_recovered, timestamp_seconds=t)
        mgr.process_worker_compliance(c_rec, fn, t)

    resolved = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.RESOLVED]
    assert len(resolved) == 1


def test_audit_situation_d_exit_and_reentry():
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Track 1 compliant exits
    assoc_t1 = WorkerPPEAssociation(
        track_id=1,
        helmet=WorkerPPEState(status=PPEStatus.DETECTED, confidence=0.91),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    for fn in range(30):
        t = fn * 0.033
        c1 = engine.evaluate_worker(assoc_t1, timestamp_seconds=t)
        mgr.process_worker_compliance(c1, fn, t)

    # Re-entry as Track 7 without helmet: must start fresh gathering evidence
    assoc_t7 = WorkerPPEAssociation(
        track_id=7,
        helmet=WorkerPPEState(status=PPEStatus.NOT_DETECTED),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    c7 = engine.evaluate_worker(assoc_t7, timestamp_seconds=2.5)

    assert c7.overall_status == ComplianceStatus.INSUFFICIENT_EVIDENCE
    assert c7.observation_count == 1
    assert c7.active_violation is False

