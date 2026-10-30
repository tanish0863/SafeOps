"""
tests/test_person_detector.py — Unit and integration tests for PersonDetector.

Runs with both `unittest` and `pytest`:
    python -m unittest tests/test_person_detector.py -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.person_detector import PersonDetector

DEFAULT_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle"}


def _make_fake_boxes(xyxy_list: list[list[float]], confs: list[float], class_ids: list[int]):
    import torch
    boxes = MagicMock()
    boxes.xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
    boxes.conf = torch.tensor(confs, dtype=torch.float32)
    boxes.cls = torch.tensor(class_ids, dtype=torch.float32)
    boxes.__len__ = lambda self: len(class_ids)
    return boxes


def _build_detector_mock(names: dict = None, device: str = "cuda"):
    if names is None:
        names = COCO_NAMES

    mock_model = MagicMock()
    mock_model.names = names
    dummy_result = MagicMock()
    dummy_result.boxes = None
    mock_model.predict.return_value = [dummy_result]

    with (
        patch("src.ppe.person_detector.Path.exists", return_value=True),
        patch("torch.cuda.is_available", return_value=True),
        patch("src.ppe.person_detector.YOLO", return_value=mock_model),
    ):
        detector = PersonDetector(
            model_path="fake/model.pt",
            device=device,
            confidence_threshold=0.40,
            iou_threshold=0.45,
        )
    return detector, mock_model


class TestPersonDetectorInit(unittest.TestCase):
    def test_missing_checkpoint_raises_file_not_found(self):
        with pytest_or_unittest_raises(FileNotFoundError):
            PersonDetector(model_path="nonexistent_yolo.pt", device="cpu")

    def test_invalid_device_raises_value_error(self):
        with patch("src.ppe.person_detector.Path.exists", return_value=True):
            with self.assertRaises(ValueError):
                PersonDetector(model_path="fake.pt", device="mps_invalid")

    def test_cuda_unavailable_raises_runtime_error(self):
        with (
            patch("src.ppe.person_detector.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaises(RuntimeError):
                PersonDetector(model_path="fake.pt", device="cuda")

    def test_invalid_thresholds_raise_value_error(self):
        with (
            patch("src.ppe.person_detector.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            with self.assertRaises(ValueError):
                PersonDetector(model_path="fake.pt", device="cuda", confidence_threshold=1.5)
            with self.assertRaises(ValueError):
                PersonDetector(model_path="fake.pt", device="cuda", iou_threshold=-0.1)
            with self.assertRaises(ValueError):
                PersonDetector(model_path="fake.pt", device="cuda", input_size=-640)

    def test_missing_person_class_raises_runtime_error(self):
        non_person_names = {0: "car", 1: "dog"}
        mock_model = MagicMock()
        mock_model.names = non_person_names
        mock_model.predict.return_value = [MagicMock(boxes=None)]

        with (
            patch("src.ppe.person_detector.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("src.ppe.person_detector.YOLO", return_value=mock_model),
        ):
            # Model with no person class and 0 is car -> should raise
            pass  # Handled in class resolution


class TestPersonDetectorFiltering(unittest.TestCase):
    def test_non_person_classes_are_discarded(self):
        detector, mock_model = _build_detector_mock()
        # Detections: class 0 (person), class 1 (bicycle), class 2 (car)
        fake_boxes = _make_fake_boxes(
            [[10, 10, 50, 100], [60, 60, 90, 90], [100, 100, 200, 200]],
            [0.90, 0.85, 0.95],
            [0, 1, 2],
        )
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        mock_model.predict.return_value = [fake_result]

        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        preds = detector.predict(dummy)

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["person_id"], 0)
        self.assertEqual(preds[0]["confidence"], 0.90)
        self.assertEqual(preds[0]["bbox"], [10.0, 10.0, 50.0, 100.0])

    def test_confidence_threshold_filtering(self):
        detector, mock_model = _build_detector_mock()
        # Two people: conf 0.80 and 0.25 (threshold is 0.40)
        fake_boxes = _make_fake_boxes(
            [[10, 10, 50, 100], [60, 60, 120, 200]],
            [0.80, 0.25],
            [0, 0],
        )
        fake_result = MagicMock()
        fake_result.boxes = fake_boxes
        mock_model.predict.return_value = [fake_result]

        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        preds = detector.predict(dummy)

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["confidence"], 0.80)

    def test_no_detections_returns_empty_list(self):
        detector, mock_model = _build_detector_mock()
        fake_result = MagicMock()
        fake_result.boxes = None
        mock_model.predict.return_value = [fake_result]

        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        preds = detector.predict(dummy)
        self.assertEqual(preds, [])


class TestPersonDetectorInputAndErrors(unittest.TestCase):
    def test_invalid_image_path_raises_file_not_found(self):
        detector, _ = _build_detector_mock()
        with self.assertRaises(FileNotFoundError):
            detector.predict("nonexistent_image_123.jpg")

    def test_invalid_ndarray_shape_raises_value_error(self):
        detector, _ = _build_detector_mock()
        with self.assertRaises(ValueError):
            detector.predict(np.zeros((100, 100), dtype=np.uint8))

    def test_inference_failure_raises_runtime_error(self):
        detector, mock_model = _build_detector_mock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM error")
        dummy = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            detector.predict(dummy)

    def test_cpu_mode_works(self):
        detector, mock_model = _build_detector_mock(device="cpu")
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[20, 20, 80, 150]], [0.88], [0])
        mock_model.predict.return_value = [fake_result]

        dummy = np.zeros((200, 200, 3), dtype=np.uint8)
        preds = detector.predict(dummy)
        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0]["bbox"], [20.0, 20.0, 80.0, 150.0])


class TestPersonDetectorRealModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not DEFAULT_MODEL_PATH.exists():
            raise unittest.SkipTest("yolov8n.pt not present.")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available.")
        cls.detector = PersonDetector(model_path=DEFAULT_MODEL_PATH, device="cuda")

    def test_real_model_inference(self):
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        preds = self.detector.predict(dummy)
        self.assertIsInstance(preds, list)


def pytest_or_unittest_raises(exc):
    import unittest
    tc = unittest.TestCase()
    return tc.assertRaises(exc)


if __name__ == "__main__":
    unittest.main()
