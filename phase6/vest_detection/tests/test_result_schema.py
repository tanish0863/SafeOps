"""
test_result_schema.py
---------------------
Tests for Detection, VestDetectionResult, and TorsoVestAssessment schemas.
Verifies contract adherence, data types, coordinate ordering, and JSON serialization.
"""

import pytest
from pydantic import ValidationError

from vest_detector.schemas import (
    Detection,
    TorsoAssessmentEvidence,
    TorsoVestAssessment,
    VestAssessmentState,
    VestDetectionResult,
)


def test_valid_detection_schema():
    det = Detection(
        class_name="WITH_VEST",
        confidence=0.93,
        bbox=[100.0, 160.0, 290.0, 380.0],
    )
    assert det.class_name == "WITH_VEST"
    assert det.confidence == 0.93
    assert det.bbox == [100.0, 160.0, 290.0, 380.0]


def test_detection_invalid_coordinates():
    # x2 < x1 should raise ValueError
    with pytest.raises(ValidationError):
        Detection(
            class_name="WITH_VEST",
            confidence=0.85,
            bbox=[300.0, 100.0, 200.0, 200.0],
        )

    # y2 < y1 should raise ValueError
    with pytest.raises(ValidationError):
        Detection(
            class_name="WITH_VEST",
            confidence=0.85,
            bbox=[100.0, 400.0, 200.0, 200.0],
        )


def test_detection_invalid_confidence():
    # Confidence > 1.0
    with pytest.raises(ValidationError):
        Detection(
            class_name="WITH_VEST",
            confidence=1.45,
            bbox=[10.0, 10.0, 50.0, 50.0],
        )

    # Confidence < 0.0
    with pytest.raises(ValidationError):
        Detection(
            class_name="WITH_VEST",
            confidence=-0.1,
            bbox=[10.0, 10.0, 50.0, 50.0],
        )


def test_detection_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        Detection(
            class_name="WITH_VEST",
            confidence=0.9,
            bbox=[10.0, 10.0, 50.0, 50.0],
            extra_field="disallowed",
        )


def test_vest_detection_result_serialization():
    det = Detection(
        class_name="WITH_VEST",
        confidence=0.93,
        bbox=[100.0, 160.0, 290.0, 380.0],
    )
    res = VestDetectionResult(
        schema_version="1.0",
        detections=[det],
    )
    data = res.to_dict()

    assert data["schema_version"] == "1.0"
    assert len(data["detections"]) == 1
    assert data["detections"][0]["class_name"] == "WITH_VEST"
    assert data["detections"][0]["confidence"] == 0.93
    assert data["detections"][0]["bbox"] == [100.0, 160.0, 290.0, 380.0]


def test_torso_vest_assessment_schema():
    evidence = TorsoAssessmentEvidence(
        torso_bbox=[200.0, 120.0, 400.0, 360.0],
        torso_width=200.0,
        torso_height=240.0,
        torso_area=48000.0,
        overlap_iou=0.82,
        intersection_ratio=0.95,
        decision_reasons=["Confirmed worn safety vest with high confidence."],
        quality_checks_passed=True,
    )

    assessment = TorsoVestAssessment(
        status=VestAssessmentState.PRESENT,
        confidence=0.94,
        vest_detected=True,
        detections=[
            Detection(
                class_name="WITH_VEST",
                confidence=0.94,
                bbox=[205.0, 125.0, 395.0, 355.0],
            )
        ],
        evidence=evidence,
    )

    d = assessment.to_dict()
    assert d["status"] == "PRESENT"
    assert d["confidence"] == 0.94
    assert d["vest_detected"] is True
    assert len(d["detections"]) == 1
    assert d["evidence"]["overlap_iou"] == 0.82
