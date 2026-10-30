"""
tests/test_helmet_adapter.py — Automated unit and integration tests for PPEHelmetAdapter.

Compatible with both standard library `unittest` and `pytest`:
    python -m unittest tests/test_helmet_adapter.py -v
    pytest tests/test_helmet_adapter.py -v

Coverage:
  - Valid checkpoint load
  - Missing checkpoint -> FileNotFoundError
  - CUDA unavailable -> RuntimeError (mocked)
  - Invalid device -> ValueError
  - Required class missing -> RuntimeError (mocked)
  - Bike detection excluded
  - with_helmet mapping -> WITH_HELMET
  - without_helmet mapping -> WITHOUT_HELMET
  - Unknown class excluded
  - Shared confidence threshold filtering
  - Per-class confidence threshold filtering
  - Bounding boxes in original-image XYXY coordinates
  - Confidence between 0 and 1
  - No detections -> empty list
  - Invalid image path -> FileNotFoundError
  - Invalid ndarray shape -> ValueError
  - Non-image input -> ValueError
  - Inference exception -> RuntimeError
  - Repeated inference reuses same model instance
  - CPU inference support
  - Real checkpoint integration tests (CUDA & CPU)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.helmet_adapter import PPEHelmetAdapter, CLASS_MAPPING

MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
SAMPLE_HELMET_DIR = PROJECT_ROOT / "samples" / "helmet"
SAMPLE_NO_HELMET_DIR = PROJECT_ROOT / "samples" / "no_helmet"

FAKE_NAMES = {0: "bike", 1: "with_helmet", 2: "without_helmet"}


def _make_fake_boxes(xyxy_list: list[list[float]], confs: list[float], class_ids: list[int]):
    import torch
    boxes = MagicMock()
    boxes.xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
    boxes.conf = torch.tensor(confs, dtype=torch.float32)
    boxes.cls = torch.tensor(class_ids, dtype=torch.float32)
    boxes.__len__ = lambda self: len(class_ids)
    return boxes


def _build_adapter_with_mock_model(
    names: dict = None,
    device: str = "cuda",
    confidence_threshold=0.25,
    iou_threshold: float = 0.45,
) -> tuple[PPEHelmetAdapter, MagicMock]:
    if names is None:
        names = FAKE_NAMES

    mock_model = MagicMock()
    mock_model.names = names
    dummy_result = MagicMock()
    dummy_result.boxes = None
    mock_model.predict.return_value = [dummy_result]

    with (
        patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
        patch("torch.cuda.is_available", return_value=True),
        patch("src.ppe.helmet_adapter.YOLO", return_value=mock_model),
    ):
        adapter = PPEHelmetAdapter(
            model_path="fake/best.pt",
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    return adapter, mock_model


class TestAdapterInitialization(unittest.TestCase):
    def test_valid_checkpoint_loads_successfully(self):
        if not MODEL_PATH.exists():
            self.skipTest("Checkpoint not present.")
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available.")
        adapter = PPEHelmetAdapter(model_path=MODEL_PATH, device="cuda")
        self.assertIsNotNone(adapter)

    def test_missing_checkpoint_raises_file_not_found(self):
        nonexistent = PROJECT_ROOT / "nonexistent_model_checkpoint.pt"
        with self.assertRaises(FileNotFoundError):
            PPEHelmetAdapter(model_path=nonexistent, device="cpu")

    def test_cuda_unavailable_raises_runtime_error(self):
        with (
            patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=False),
        ):
            with self.assertRaises(RuntimeError):
                PPEHelmetAdapter(model_path="fake/best.pt", device="cuda")

    def test_invalid_device_raises_value_error(self):
        with patch("src.ppe.helmet_adapter.Path.exists", return_value=True):
            with self.assertRaises(ValueError):
                PPEHelmetAdapter(model_path="fake/best.pt", device="tpu")

    def test_missing_required_class_raises_runtime_error(self):
        broken_names = {0: "bike", 1: "person"}
        mock_model = MagicMock()
        mock_model.names = broken_names
        mock_model.predict.return_value = [MagicMock(boxes=None)]

        with (
            patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
            patch("src.ppe.helmet_adapter.YOLO", return_value=mock_model),
        ):
            with self.assertRaises(RuntimeError):
                PPEHelmetAdapter(model_path="fake/best.pt", device="cuda")


class TestConfidenceThreshold(unittest.TestCase):
    def test_invalid_shared_threshold_raises(self):
        with (
            patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            with self.assertRaises(ValueError):
                PPEHelmetAdapter(model_path="fake/best.pt", device="cuda", confidence_threshold=1.5)

    def test_invalid_per_class_threshold_raises(self):
        with (
            patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
            patch("torch.cuda.is_available", return_value=True),
        ):
            with self.assertRaises(ValueError):
                PPEHelmetAdapter(model_path="fake/best.pt", device="cuda", confidence_threshold={"with_helmet": -0.1})

    def test_shared_threshold_filters_low_confidence(self):
        adapter, mock_model = _build_adapter_with_mock_model(confidence_threshold=0.80)
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[10, 20, 50, 60]], [0.50], [1])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(result, [])

    def test_per_class_threshold_applied_correctly(self):
        adapter, mock_model = _build_adapter_with_mock_model(
            confidence_threshold={"with_helmet": 0.90, "without_helmet": 0.20}
        )
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[10, 20, 50, 60]], [0.50], [2])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "WITHOUT_HELMET")


class TestDetectionFiltering(unittest.TestCase):
    def test_bike_detection_excluded(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[5, 5, 100, 200]], [0.99], [0])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(result, [])

    def test_with_helmet_mapped_to_uppercase(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[10, 20, 50, 60]], [0.91], [1])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "WITH_HELMET")

    def test_without_helmet_mapped_to_uppercase(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[30, 40, 80, 90]], [0.87], [2])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((200, 200, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "WITHOUT_HELMET")

    def test_unknown_class_excluded(self):
        adapter, mock_model = _build_adapter_with_mock_model(
            names={0: "bike", 1: "with_helmet", 2: "without_helmet", 99: "hardhat"}
        )
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[1, 1, 10, 10]], [0.95], [99])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(result, [])

    def test_no_detections_returns_empty_list(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        fake_result = MagicMock()
        fake_result.boxes = None
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(result, [])


class TestBoundingBoxes(unittest.TestCase):
    def test_bounding_boxes_are_xyxy_pixel_coordinates(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        expected_bbox = [120.0, 50.0, 210.0, 130.0]
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([expected_bbox], [0.91], [1])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((300, 300, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], expected_bbox)

    def test_confidence_is_float_between_0_and_1(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[10, 20, 50, 60]], [0.75], [1])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertTrue(0.0 <= result[0]["confidence"] <= 1.0)


class TestImageInputValidation(unittest.TestCase):
    def test_invalid_image_path_raises_file_not_found(self):
        adapter, _ = _build_adapter_with_mock_model()
        with self.assertRaises(FileNotFoundError):
            adapter.predict("/nonexistent/path/image.jpg")

    def test_invalid_ndarray_shape_raises_value_error(self):
        adapter, _ = _build_adapter_with_mock_model()
        bad_image = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaises(ValueError):
            adapter.predict(bad_image)

    def test_non_image_type_raises_value_error(self):
        adapter, _ = _build_adapter_with_mock_model()
        with self.assertRaises(ValueError):
            adapter.predict(12345)


class TestInferenceErrors(unittest.TestCase):
    def test_inference_exception_raises_runtime_error(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        mock_model.predict.side_effect = RuntimeError("GPU OOM")

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            adapter.predict(dummy_image)

    def test_inference_error_is_not_empty_list(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        mock_model.predict.side_effect = Exception("unexpected")

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            adapter.predict(dummy_image)


class TestModelReuse(unittest.TestCase):
    def test_repeated_inference_reuses_same_model_instance(self):
        adapter, mock_model = _build_adapter_with_mock_model()
        dummy_result = MagicMock()
        dummy_result.boxes = None
        mock_model.predict.return_value = [dummy_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for _ in range(5):
            adapter.predict(dummy_image)

        self.assertIs(adapter._model, mock_model)


class TestCPUInference(unittest.TestCase):
    def test_cpu_inference_works(self):
        adapter, mock_model = _build_adapter_with_mock_model(device="cpu")
        fake_result = MagicMock()
        fake_result.boxes = _make_fake_boxes([[10, 20, 50, 60]], [0.88], [1])
        mock_model.predict.return_value = [fake_result]

        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = adapter.predict(dummy_image)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["class_name"], "WITH_HELMET")


class TestRealCheckpointIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("Checkpoint not present.")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available.")
        cls.real_adapter = PPEHelmetAdapter(
            model_path=MODEL_PATH,
            device="cuda",
            confidence_threshold=0.25,
        )

    def test_real_checkpoint_classes(self):
        self.assertEqual(
            self.real_adapter._checkpoint_names,
            {0: "bike", 1: "with_helmet", 2: "without_helmet"},
        )

    def test_real_cuda_inference_on_dummy_image(self):
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        result = self.real_adapter.predict(dummy)
        self.assertIsInstance(result, list)

    def test_real_inference_helmet_sample(self):
        samples = list(SAMPLE_HELMET_DIR.glob("*.jpg"))[:3] if SAMPLE_HELMET_DIR.exists() else []
        if not samples:
            self.skipTest("No helmet sample images found.")
        for img_path in samples:
            predictions = self.real_adapter.predict(img_path)
            self.assertIsInstance(predictions, list)
            for p in predictions:
                self.assertIn(p["class_name"], ("WITH_HELMET", "WITHOUT_HELMET"))
                self.assertTrue(0.0 <= p["confidence"] <= 1.0)
                self.assertEqual(len(p["bbox"]), 4)

    def test_real_inference_no_helmet_sample(self):
        samples = list(SAMPLE_NO_HELMET_DIR.glob("*.jpg"))[:3] if SAMPLE_NO_HELMET_DIR.exists() else []
        if not samples:
            self.skipTest("No no-helmet sample images found.")
        for img_path in samples:
            predictions = self.real_adapter.predict(img_path)
            self.assertIsInstance(predictions, list)

    def test_real_predict_with_result_returns_tuple(self):
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        predictions, raw_result = self.real_adapter.predict_with_result(dummy)
        self.assertIsInstance(predictions, list)
        self.assertIsNotNone(raw_result)

    def test_real_cpu_inference(self):
        if not MODEL_PATH.exists():
            self.skipTest("Checkpoint not present.")
        adapter_cpu = PPEHelmetAdapter(
            model_path=MODEL_PATH,
            device="cpu",
            confidence_threshold=0.25,
        )
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        result = adapter_cpu.predict(dummy)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
