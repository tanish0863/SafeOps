"""
test_vest_detection.py
----------------------
Tests for full-frame vest detection, input formats (path vs array),
class filtering, and confidence thresholds.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

from vest_detector.config import VestDetectorConfig
from vest_detector.model_loader import MockModelBackend
from vest_detector.vest_detector import VestDetector

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_PATH = MODELS_DIR / "yolov8n_vest_best.pt"


def test_detect_full_image_with_mock():
    mock_backend = MockModelBackend()
    # Mock output includes:
    # 1. Valid Safety Vest
    # 2. Hardhat (should be filtered out by adapter)
    # 3. Low confidence vest (should be filtered by threshold)
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.92, "bbox": [150.0, 100.0, 350.0, 300.0]},
        {"class_name": "Hardhat", "confidence": 0.95, "bbox": [200.0, 40.0, 300.0, 95.0]},
        {"class_name": "Safety Vest", "confidence": 0.20, "bbox": [400.0, 100.0, 450.0, 200.0]},
    ])

    detector = VestDetector(
        config=VestDetectorConfig(confidence_threshold=0.35),
        backend=mock_backend,
    )

    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(dummy_image)

    assert result["schema_version"] == "1.0"
    assert "detections" in result
    detections = result["detections"]

    # Only the confident safety vest should be returned
    assert len(detections) == 1
    assert detections[0]["class_name"] == "WITH_VEST"
    assert detections[0]["confidence"] == 0.92
    assert detections[0]["bbox"] == [150.0, 100.0, 350.0, 300.0]


def test_detect_with_file_path_and_array():
    sample_file = SAMPLES_DIR / "worker_with_vest.jpg"
    if not sample_file.exists():
        pytest.skip("Sample worker_with_vest.jpg not found.")

    mock_backend = MockModelBackend()
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.88, "bbox": [220.0, 140.0, 420.0, 380.0]}
    ])

    detector = VestDetector(backend=mock_backend)

    # 1. Path input
    res_path = detector.detect(sample_file)
    assert len(res_path["detections"]) == 1

    # 2. NumPy array input
    img_array = cv2.imread(str(sample_file))
    res_array = detector.detect(img_array)
    assert len(res_array["detections"]) == 1
    assert res_array["detections"][0]["bbox"] == [220.0, 140.0, 420.0, 380.0]


def test_detect_frame_video_contract():
    mock_backend = MockModelBackend()
    mock_backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.90, "bbox": [100.0, 150.0, 250.0, 350.0]}
    ])
    detector = VestDetector(backend=mock_backend)

    video_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    res = detector.detect_frame(video_frame)

    assert res["schema_version"] == "1.0"
    assert len(res["detections"]) == 1
    assert res["detections"][0]["class_name"] == "WITH_VEST"


def test_detect_real_weights_if_available():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Pretrained checkpoint not present at {CHECKPOINT_PATH}")

    sample_file = SAMPLES_DIR / "worker_with_vest.jpg"
    if not sample_file.exists():
        pytest.skip("Sample worker_with_vest.jpg not found.")

    detector = VestDetector(
        model_path=CHECKPOINT_PATH,
        device="cuda",
    )

    result = detector.detect(sample_file)
    assert result["schema_version"] == "1.0"
    assert "detections" in result
    assert len(result["detections"]) > 0, "Expected at least 1 detection on worker_with_vest.jpg"

    # Detections should have correct coordinates and valid classes
    vest_found = False
    for det in result["detections"]:
        assert det["class_name"] in ["WITH_VEST", "NO_VEST"]
        assert 0.0 <= det["confidence"] <= 1.0
        assert len(det["bbox"]) == 4
        if det["class_name"] == "WITH_VEST":
            vest_found = True
    assert vest_found, "Expected at least one WITH_VEST detection on worker_with_vest.jpg"
