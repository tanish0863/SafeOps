"""
test_torso_inference.py
-----------------------
Unit and integration tests for localized torso crop inference,
coordinate mapping back to original frame, and Three-State Compliance Assessment
(PRESENT, ABSENT, UNKNOWN) following the strict rules:
  1. PRESENT: Confident WITH_VEST detection overlapping torso.
  2. ABSENT: Confident NO_VEST detection overlapping torso with valid quality checks.
  3. UNKNOWN: Missing detections, degraded optical conditions, small/distant torsos,
              insufficient confidence, or non-overlapping bounding boxes.

NOTE: Tests in this file use MockModelBackend with controlled detection outputs
to rigorously verify decision logic boundaries independently from real-world weights.
"""

from pathlib import Path
import numpy as np
import pytest

from vest_detector.config import VestDetectorConfig
from vest_detector.model_loader import MockModelBackend
from vest_detector.schemas import VestAssessmentState
from vest_detector.vest_detector import VestDetector

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


# ---------------------------------------------------------------------------
# Coordinate Mapping Tests
# ---------------------------------------------------------------------------

def test_detect_torso_coordinate_mapping():
    """
    CRITICAL REQUIREMENT:
    Predictions inside a cropped torso region MUST be mapped back to original image space.
    Crop-relative coordinates must never be returned.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [200.0, 150.0, 400.0, 350.0]  # width=200, height=200

    # Model detects a vest inside the crop at sub-region coordinates [16, 16, 180, 180]
    # In padded crop (padding=0.08 -> pad_w=16, pad_h=16 -> offset_x=184, offset_y=134)
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.94, "bbox": [16.0, 16.0, 180.0, 180.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    full_image = np.ones((600, 800, 3), dtype=np.uint8) * 120

    res = detector.detect_torso(full_image, torso_bbox)
    detections = res["detections"]

    assert len(detections) == 1
    det = detections[0]
    assert det["class_name"] == "WITH_VEST"

    # Mapped coordinates must be in original full-image range, NOT crop-relative [16, 16, 180, 180]
    x1, y1, x2, y2 = det["bbox"]
    assert x1 >= 184.0, f"Expected x1 mapped to full image, got {x1}"
    assert y1 >= 134.0, f"Expected y1 mapped to full image, got {y1}"
    assert x2 <= 400.0 + 16.0
    assert y2 <= 350.0 + 16.0


# ---------------------------------------------------------------------------
# Three-State Assessment Rule Tests
# ---------------------------------------------------------------------------

def test_assess_torso_state_present():
    """
    Rule 1: WITH_VEST prediction with confidence meeting threshold and overlapping torso -> PRESENT.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]

    # Valid WITH_VEST detection overlapping torso
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.95, "bbox": [20.0, 20.0, 180.0, 220.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    img_sample = SAMPLES_DIR / "worker_with_vest.jpg"
    if not img_sample.exists():
        pytest.skip("worker_with_vest.jpg sample not found.")

    assessment = detector.assess_torso(img_sample, torso_bbox)

    assert assessment.status == VestAssessmentState.PRESENT
    assert assessment.vest_detected is True
    assert assessment.confidence >= 0.90
    assert len(assessment.detections) == 1
    assert assessment.evidence.overlap_iou > 0.0


def test_assess_torso_state_absent():
    """
    Rule 2: Valid NO_VEST prediction meeting threshold, overlapping torso, with quality checks passed -> ABSENT.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]

    # Model explicitly detects NO_VEST on torso
    mock_backend.set_next_detections([
        {"class_name": "NO-Safety Vest", "confidence": 0.89, "bbox": [20.0, 20.0, 180.0, 220.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    img_sample = SAMPLES_DIR / "worker_no_vest.jpg"
    if not img_sample.exists():
        pytest.skip("worker_no_vest.jpg sample not found.")

    assessment = detector.assess_torso(img_sample, torso_bbox)

    assert assessment.status == VestAssessmentState.ABSENT
    assert assessment.vest_detected is False
    assert assessment.confidence == 0.89
    assert assessment.evidence.quality_checks_passed is True
    assert any("no_vest" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_no_predictions_returns_unknown():
    """
    Rule 3: Missing predictions must NOT return ABSENT.
    Without verified negative evidence, a lack of detection yields UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]
    # Zero detections returned by model
    mock_backend.set_next_detections([])

    detector = VestDetector(backend=mock_backend)
    img_sample = SAMPLES_DIR / "worker_no_vest.jpg"
    if not img_sample.exists():
        pytest.skip("worker_no_vest.jpg sample not found.")

    assessment = detector.assess_torso(img_sample, torso_bbox)

    # MUST be UNKNOWN, not ABSENT
    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False
    assert any("no safety vest or negative vest evidence" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_low_confidence_with_vest_returns_unknown():
    """
    Rule 3: WITH_VEST prediction below configured threshold (e.g. 0.20 < 0.35) -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]

    # Model returns low-confidence vest detection
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.20, "bbox": [20.0, 20.0, 180.0, 220.0]}
    ])

    cfg = VestDetectorConfig(vest_confidence_threshold=0.35)
    detector = VestDetector(backend=mock_backend, config=cfg)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    assessment = detector.assess_torso(dummy_img, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False


def test_assess_torso_low_confidence_no_vest_returns_unknown():
    """
    Rule 3: NO_VEST prediction below configured threshold (e.g. 0.22 < 0.35) -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]

    # Model returns low-confidence NO_VEST detection
    mock_backend.set_next_detections([
        {"class_name": "NO-Safety Vest", "confidence": 0.22, "bbox": [20.0, 20.0, 180.0, 220.0]}
    ])

    cfg = VestDetectorConfig(no_vest_confidence_threshold=0.35)
    detector = VestDetector(backend=mock_backend, config=cfg)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    assessment = detector.assess_torso(dummy_img, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False


def test_assess_torso_bright_shirt_without_negative_evidence_returns_unknown():
    """
    Rule: A worker wearing a bright shirt where model detects neither WITH_VEST nor NO_VEST -> UNKNOWN.
    Must never default to ABSENT.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [200.0, 100.0, 400.0, 350.0]
    # No vest or negative vest predictions
    mock_backend.set_next_detections([])

    detector = VestDetector(backend=mock_backend)
    # Create bright shirt image with natural texture (so quality checks pass)
    np.random.seed(42)
    bright_canvas = np.random.randint(80, 160, (480, 640, 3), dtype=np.uint8)
    # Textured bright yellow shirt inside torso
    bright_canvas[100:350, 200:400, 0] = np.random.randint(0, 40, (250, 200), dtype=np.uint8)
    bright_canvas[100:350, 200:400, 1] = np.random.randint(180, 245, (250, 200), dtype=np.uint8)
    bright_canvas[100:350, 200:400, 2] = np.random.randint(180, 245, (250, 200), dtype=np.uint8)

    assessment = detector.assess_torso(bright_canvas, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False
    assert assessment.evidence.quality_checks_passed is True
    assert any("no safety vest or negative vest evidence" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_dark_image_returns_unknown():
    """
    Rule: Severely underexposed / dark torso crop -> UNKNOWN.
    Even if NO_VEST was predicted, optical quality failure forces UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]
    mock_backend.set_next_detections([
        {"class_name": "NO-Safety Vest", "confidence": 0.90, "bbox": [20.0, 20.0, 180.0, 220.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    img_sample = SAMPLES_DIR / "worker_dark.jpg"
    if not img_sample.exists():
        pytest.skip("worker_dark.jpg sample not found.")

    assessment = detector.assess_torso(img_sample, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.evidence.quality_checks_passed is False
    assert any("underexposed" in r.lower() or "optical quality" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_blurry_image_returns_unknown():
    """
    Rule: Severely blurred / low-contrast torso crop -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [220.0, 140.0, 420.0, 380.0]
    mock_backend.set_next_detections([])

    # Extremely blurred / flat image
    blurry_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    cfg = VestDetectorConfig(min_laplacian_variance=20.0)
    detector = VestDetector(backend=mock_backend, config=cfg)

    assessment = detector.assess_torso(blurry_img, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.evidence.quality_checks_passed is False
    assert any("blurred" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_small_torso_returns_unknown():
    """
    Rule: Distant worker where torso bounding box is below minimum size threshold -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    tiny_torso_bbox = [445.0, 270.0, 465.0, 295.0]  # width=20, height=25

    detector = VestDetector(backend=mock_backend)
    img_sample = SAMPLES_DIR / "worker_distant.jpg"
    if not img_sample.exists():
        pytest.skip("worker_distant.jpg sample not found.")

    assessment = detector.assess_torso(img_sample, tiny_torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False
    assert any("too small" in r.lower() or "below minimum" in r.lower() for r in assessment.evidence.decision_reasons)


def test_assess_torso_negative_prediction_outside_torso_returns_unknown():
    """
    Rule: A NO_VEST prediction located elsewhere in the frame (not overlapping torso) -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [200.0, 150.0, 350.0, 350.0]

    # Candidate NO_VEST is far away at [500, 400, 600, 450]
    mock_backend.set_next_detections([
        {"class_name": "NO-Safety Vest", "confidence": 0.92, "bbox": [500.0, 400.0, 600.0, 450.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    assessment = detector.assess_torso(dummy_img, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False
    assert assessment.evidence.overlap_iou == 0.0


def test_assess_torso_positive_prediction_outside_torso_returns_unknown():
    """
    Rule: A WITH_VEST prediction located elsewhere in the frame (e.g. nearby discarded vest) -> UNKNOWN.
    Must not mark this worker's torso as PRESENT.
    """
    mock_backend = MockModelBackend()
    torso_bbox = [100.0, 100.0, 250.0, 300.0]

    # Vest is nearby on a rack/ground at [400, 300, 500, 420]
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.95, "bbox": [400.0, 300.0, 500.0, 420.0]}
    ])

    detector = VestDetector(backend=mock_backend)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    assessment = detector.assess_torso(dummy_img, torso_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert assessment.vest_detected is False
    assert assessment.evidence.overlap_iou == 0.0


def test_assess_torso_out_of_frame_returns_unknown():
    """
    Rule: Torso bbox completely outside frame -> UNKNOWN.
    """
    mock_backend = MockModelBackend()
    outside_bbox = [-200.0, -100.0, -50.0, -20.0]

    detector = VestDetector(backend=mock_backend)
    dummy_img = np.ones((480, 640, 3), dtype=np.uint8) * 128

    assessment = detector.assess_torso(dummy_img, outside_bbox)

    assert assessment.status == VestAssessmentState.UNKNOWN
    assert any("outside" in r.lower() for r in assessment.evidence.decision_reasons)
