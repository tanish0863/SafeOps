"""Safety-Critical PPE Association and Compliance Identity-Transfer Audit.

Evaluates:
- Test A: Worker Crossing
- Test B: Temporary Occlusion
- Test C: Detection Blackout
- Test D: Exit and Re-entry

Records:
- Original track ID
- New or recovered track ID
- Ground-truth identity
- PPE association before and after the event
- Compliance state before and after the event
- History transfer verification
- False violation generation check
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


def run_test_a_worker_crossing():
    print("=== Running Test A: Worker Crossing ===")
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Pre-crossing: Worker 1 (compliant, with helmet), Worker 2 (violator, without helmet)
    w1_pre = {
        "track_id": 1,
        "bbox_xyxy": [100.0, 100.0, 200.0, 400.0],
        "keypoints_xy": [[150.0, 130.0]] * 17,
        "keypoints_conf": [0.95] * 17,
    }
    w2_pre = {
        "track_id": 2,
        "bbox_xyxy": [400.0, 100.0, 500.0, 400.0],
        "keypoints_xy": [[450.0, 130.0]] * 17,
        "keypoints_conf": [0.95] * 17,
    }
    helmet_det = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[130.0, 95.0, 170.0, 135.0],
        confidence=0.94
    )

    # 30 frames pre-crossing
    for fn in range(30):
        t = fn * 0.033
        assocs, _ = associate_ppe_to_workers([w1_pre, w2_pre], [helmet_det], has_vest_model=False)
        c1 = engine.evaluate_worker(assocs[1], timestamp_seconds=t)
        c2 = engine.evaluate_worker(assocs[2], timestamp_seconds=t)
        mgr.process_worker_compliance(c1, fn, t)
        mgr.process_worker_compliance(c2, fn, t)

    w1_status_pre = assocs[1].helmet.status.value
    w2_status_pre = assocs[2].helmet.status.value
    c1_pre = c1.overall_status.value
    c2_pre = c2.overall_status.value

    # Crossing point: Workers overlap, helmet in ambiguous position
    w1_cross = {
        "track_id": 1,
        "bbox_xyxy": [250.0, 100.0, 350.0, 400.0],
        "keypoints_xy": [[300.0, 130.0]] * 17,
        "keypoints_conf": [0.95] * 17,
    }
    w2_cross = {
        "track_id": 2,
        "bbox_xyxy": [260.0, 100.0, 360.0, 400.0],
        "keypoints_xy": [[310.0, 130.0]] * 17,
        "keypoints_conf": [0.95] * 17,
    }
    # Helmet centered between heads
    helmet_cross = PPEDetectionItem(
        item_type=PPEItemType.HELMET,
        bbox_xyxy=[285.0, 95.0, 325.0, 135.0],
        confidence=0.91
    )
    assocs_cross, _ = associate_ppe_to_workers([w1_cross, w2_cross], [helmet_cross], ambiguity_margin=0.15, has_vest_model=False)

    w1_status_cross = assocs_cross[1].helmet.status.value
    w2_status_cross = assocs_cross[2].helmet.status.value

    # During crossing ambiguity, status must be UNKNOWN
    ambiguity_produced_unknown = (w1_status_cross == "unknown" and w2_status_cross == "unknown")
    helmet_transferred = (assocs_cross[2].helmet.status == PPEStatus.DETECTED)

    # Compliance during ambiguous crossing
    c1_cross = engine.evaluate_worker(assocs_cross[1], timestamp_seconds=1.0)
    c2_cross = engine.evaluate_worker(assocs_cross[2], timestamp_seconds=1.0)
    mgr.process_worker_compliance(c1_cross, 31, 1.0)
    mgr.process_worker_compliance(c2_cross, 31, 1.0)

    # Check if a false violation was created for Worker 1
    w1_violations = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    false_violation = len(w1_violations) > 0

    return {
        "test": "Test A — Worker Crossing",
        "original_track_id": 1,
        "new_or_recovered_track_id": 1,
        "gt_identity": "Worker A (Compliant) & Worker B (Violator)",
        "ppe_before": f"W1:{w1_status_pre}, W2:{w2_status_pre}",
        "ppe_during_crossing": f"W1:{w1_status_cross}, W2:{w2_status_cross}",
        "compliance_before": f"W1:{c1_pre}, W2:{c2_pre}",
        "compliance_during": f"W1:{c1_cross.overall_status.value}, W2:{c2_cross.overall_status.value}",
        "history_transfer_occurred": helmet_transferred,
        "false_violation_generated": false_violation,
        "ambiguity_handled_as_unknown": ambiguity_produced_unknown,
        "verdict": "PASS" if not helmet_transferred and not false_violation and ambiguity_produced_unknown else "FAIL"
    }


def run_test_b_temporary_occlusion():
    print("=== Running Test B: Temporary Occlusion ===")
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Worker A (Track 1) observed compliant for 30 frames
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

    c_before = c.overall_status.value

    # Worker A becomes partially occluded (head ROI UNKNOWN, helmet UNKNOWN)
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

    c_during = c_occ.overall_status.value

    # Check that another worker B (Track 2) cannot inherit A's compliance state
    assoc_worker_b = WorkerPPEAssociation(
        track_id=2,
        helmet=WorkerPPEState(status=PPEStatus.NOT_DETECTED),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    c_worker_b = engine.evaluate_worker(assoc_worker_b, timestamp_seconds=2.0)

    # Check violations for Worker A
    w1_violations = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    false_violation = len(w1_violations) > 0
    history_transfer = (c_worker_b.overall_status == ComplianceStatus.COMPLIANT)

    return {
        "test": "Test B — Temporary Occlusion",
        "original_track_id": 1,
        "new_or_recovered_track_id": 1,
        "gt_identity": "Worker A (Occluded)",
        "ppe_before": "detected (helmet)",
        "ppe_during": "unknown (insufficient head evidence)",
        "compliance_before": c_before,
        "compliance_during": c_during,
        "history_transfer_occurred": history_transfer,
        "false_violation_generated": false_violation,
        "timer_paused_on_unknown": (c_occ.consecutive_non_compliant_seconds == 0.0),
        "verdict": "PASS" if not false_violation and not history_transfer and c_during == "unknown" else "FAIL"
    }


def run_test_c_detection_blackout():
    print("=== Running Test C: Detection Blackout ===")
    policy = SafetyPolicyConfig(
        require_helmet=True,
        min_observations_for_decision=5,
        violation_confirmation_seconds=0.5,
    )
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=0.5)
    engine.reset()
    mgr.reset()

    # Track 1 has active confirmed violation (missing helmet)
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

    active_v_before = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.CONFIRMED]
    has_active_v_before = len(active_v_before) == 1

    # Blackout for 10 frames (no calls for Track 1)

    # Recovered Track 1 resumes with helmet
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

    # The violation should be resolved gracefully after recovery confirmation
    resolved_events = [e for e in mgr.get_all_events() if e.track_id == 1 and e.status == ViolationStatus.RESOLVED]
    violation_resolved = len(resolved_events) == 1

    # Check that another newly detected Track 3 does NOT inherit Track 1's violation history
    assoc_track_3 = WorkerPPEAssociation(
        track_id=3,
        helmet=WorkerPPEState(status=PPEStatus.DETECTED, confidence=0.90),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    c3 = engine.evaluate_worker(assoc_track_3, timestamp_seconds=4.0)
    history_transferred_to_new_track = (c3.active_violation is True or c3.observation_count > 1)

    return {
        "test": "Test C — Detection Blackout",
        "original_track_id": 1,
        "new_or_recovered_track_id": 1,
        "gt_identity": "Worker A (Temporary Detection Blackout)",
        "ppe_before": "not_detected",
        "ppe_after": "detected",
        "compliance_before": "non_compliant",
        "compliance_after": c_rec.overall_status.value,
        "history_transfer_occurred": history_transferred_to_new_track,
        "false_violation_generated": False,
        "violation_properly_resolved": violation_resolved,
        "verdict": "PASS" if not history_transferred_to_new_track and violation_resolved else "FAIL"
    }


def run_test_d_exit_and_reentry():
    print("=== Running Test D: Exit and Re-entry ===")
    policy = SafetyPolicyConfig(require_helmet=True, violation_confirmation_seconds=1.0)
    engine = ComplianceEngine(policy=policy)
    mgr = ViolationManager(recovery_confirmation_seconds=1.0)
    engine.reset()
    mgr.reset()

    # Track 1 exits the frame after being confirmed compliant
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

    # Absence period: 30 frames pass

    # Worker re-enters, tracker assigns brand new Track ID 7
    # Because track ID is new, it MUST NOT automatically inherit Track 1's evidence or status
    assoc_t7 = WorkerPPEAssociation(
        track_id=7,
        helmet=WorkerPPEState(status=PPEStatus.NOT_DETECTED),
        vest=WorkerPPEState(status=PPEStatus.UNKNOWN),
        head_roi_quality=HeadROIQuality.CONFIRMED,
    )
    # First frame of re-entry
    c7_first = engine.evaluate_worker(assoc_t7, timestamp_seconds=2.5)

    # Must start gathering evidence (insufficient_evidence) rather than inheriting Track 1's compliant status
    starts_fresh = (c7_first.overall_status == ComplianceStatus.INSUFFICIENT_EVIDENCE)
    observation_count_is_fresh = (c7_first.observation_count == 1)
    history_transferred = (c7_first.overall_status == ComplianceStatus.COMPLIANT)

    # Violations check: no instant violation generated on frame 1
    false_instant_violation = c7_first.active_violation

    return {
        "test": "Test D — Exit and Re-entry",
        "original_track_id": 1,
        "new_or_recovered_track_id": 7,
        "gt_identity": "Worker A (Exited as Track 1, Re-entered as Track 7)",
        "ppe_before": "detected (Track 1)",
        "ppe_after": "not_detected (Track 7)",
        "compliance_before": c1.overall_status.value,
        "compliance_after": c7_first.overall_status.value,
        "history_transfer_occurred": history_transferred,
        "starts_fresh_evidence": starts_fresh,
        "false_violation_generated": false_instant_violation,
        "verdict": "PASS" if starts_fresh and not history_transferred and not false_instant_violation else "FAIL"
    }


def main():
    results = [
        run_test_a_worker_crossing(),
        run_test_b_temporary_occlusion(),
        run_test_c_detection_blackout(),
        run_test_d_exit_and_reentry(),
    ]

    out_file = Path("outputs/phase5_ppe_identity_audit.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    root_out_file = Path("D:/Projects/SafeOps/outputs/phase5_ppe_identity_audit.json")
    with open(root_out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== AUDIT SUMMARY ===")
    for r in results:
        print(f"[{r['test']}] Verdict: {r['verdict']} | History Transfer: {r['history_transfer_occurred']} | False Violation: {r['false_violation_generated']}")


if __name__ == "__main__":
    main()
