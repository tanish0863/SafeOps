"""
test_model_loading.py
---------------------
Tests for model loading, weight file resolution, device placement,
and mock backend execution.
"""

from pathlib import Path
import pytest
import torch

from vest_detector.config import VestDetectorConfig
from vest_detector.model_loader import (
    MockModelBackend,
    UltralyticsYOLOBackend,
    load_model,
    resolve_device,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECKPOINT_PATH = MODELS_DIR / "yolov8n_vest_best.pt"


def test_resolve_device():
    # CPU should always resolve to cpu
    assert resolve_device("cpu") == "cpu"

    # CUDA should resolve to cuda if available, else cpu fallback
    resolved = resolve_device("cuda", fallback_to_cpu=True)
    if torch.cuda.is_available():
        assert "cuda" in resolved
    else:
        assert resolved == "cpu"


def test_mock_backend():
    backend = MockModelBackend()
    backend.set_next_detections([
        {"class_name": "Safety Vest", "confidence": 0.91, "bbox": [50.0, 50.0, 150.0, 200.0]}
    ])

    dummy_img = torch.zeros((100, 100, 3)).numpy().astype("uint8")
    boxes, scores, classes = backend.predict(dummy_img, conf_threshold=0.3, iou_threshold=0.45)

    assert len(boxes) == 1
    assert scores[0] == 0.91
    assert classes[0] == "Safety Vest"
    assert boxes[0] == [50.0, 50.0, 150.0, 200.0]


def test_load_real_yolo_model():
    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"Model checkpoint not present at {CHECKPOINT_PATH}")

    config = VestDetectorConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    backend = load_model(model_path=CHECKPOINT_PATH, config=config, use_mock=False)

    assert isinstance(backend, UltralyticsYOLOBackend)
    assert hasattr(backend, "names")
    names_dict = backend.names

    # Verify key target classes exist in model
    class_values = list(names_dict.values())
    assert "Safety Vest" in class_values
    assert "NO-Safety Vest" in class_values


def test_missing_model_raises():
    fake_path = MODELS_DIR / "non_existent_weights_xyz.pt"
    config = VestDetectorConfig()
    with pytest.raises(FileNotFoundError):
        load_model(model_path=fake_path, config=config, use_mock=False)
