"""
tests/test_ppe_pipeline.py — Unit and integration tests for PPEPipeline.

Runs with both `unittest` and `pytest`:
    python -m unittest tests/test_ppe_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ppe.ppe_pipeline import PPEPipeline

PERSON_MODEL_PATH = PROJECT_ROOT / "yolov8n.pt"
HELMET_MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"
SAMPLE_DIR = PROJECT_ROOT / "samples"


def _build_mocked_pipeline():
    mock_person_model = MagicMock()
    mock_person_model.names = {0: "person", 1: "car"}
    mock_helmet_model = MagicMock()
    mock_helmet_model.names = {0: "bike", 1: "with_helmet", 2: "without_helmet"}

    dummy_res = MagicMock()
    dummy_res.boxes = None
    mock_person_model.predict.return_value = [dummy_res]
    mock_helmet_model.predict.return_value = [dummy_res]

    with (
        patch("src.ppe.person_detector.Path.exists", return_value=True),
        patch("src.ppe.helmet_adapter.Path.exists", return_value=True),
        patch("torch.cuda.is_available", return_value=True),
        patch("src.ppe.person_detector.YOLO", return_value=mock_person_model),
        patch("src.ppe.helmet_adapter.YOLO", return_value=mock_helmet_model),
    ):
        pipeline = PPEPipeline(
            person_model_path="fake_p.pt",
            helmet_model_path="fake_h.pt",
            device="cuda",
        )
    return pipeline, mock_person_model, mock_helmet_model


class TestPPEPipelineMocked(unittest.TestCase):
    def test_pipeline_predict_structure(self):
        pipeline, _, _ = _build_mocked_pipeline()

        # Mock detector outputs
        pipeline.person_detector.predict = MagicMock(
            return_value=[{"person_id": 0, "bbox": [100.0, 50.0, 200.0, 300.0], "confidence": 0.95}]
        )
        pipeline.helmet_detector.predict = MagicMock(
            return_value=[{"class_name": "WITH_HELMET", "confidence": 0.91, "bbox": [130.0, 55.0, 170.0, 95.0]}]
        )

        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        result = pipeline.predict(dummy)

        self.assertIn("image", result)
        self.assertIn("persons", result)
        self.assertIn("unmatched_helmets", result)
        self.assertIn("summary", result)
        self.assertEqual(result["summary"]["persons_detected"], 1)
        self.assertEqual(result["summary"]["helmets_associated"], 1)
        self.assertEqual(result["persons"][0]["association_status"], "associated")

    def test_predict_and_annotate(self):
        pipeline, _, _ = _build_mocked_pipeline()
        pipeline.person_detector.predict = MagicMock(
            return_value=[{"person_id": 0, "bbox": [50.0, 50.0, 150.0, 250.0], "confidence": 0.88}]
        )
        pipeline.helmet_detector.predict = MagicMock(return_value=[])

        dummy = np.zeros((300, 300, 3), dtype=np.uint8)
        result, annotated = pipeline.predict_and_annotate(dummy)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(annotated, np.ndarray)
        self.assertEqual(annotated.shape, dummy.shape)
        # Person has no helmet detection
        self.assertEqual(result["persons"][0]["association_status"], "no_helmet_detection")

    def test_invalid_image_raises_error(self):
        pipeline, _, _ = _build_mocked_pipeline()
        with self.assertRaises(FileNotFoundError):
            pipeline.predict("nonexistent_image_foo.png")


class TestPPEPipelineRealModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not PERSON_MODEL_PATH.exists():
            raise unittest.SkipTest("yolov8n.pt not present.")
        if not HELMET_MODEL_PATH.exists():
            raise unittest.SkipTest("best.pt helmet model not present.")
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available.")

        cls.pipeline = PPEPipeline(
            person_model_path=PERSON_MODEL_PATH,
            helmet_model_path=HELMET_MODEL_PATH,
            device="cuda",
        )

    def test_real_pipeline_cuda_inference(self):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.pipeline.predict(dummy)
        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)

    def test_real_pipeline_on_sample_image(self):
        sample_images = list(SAMPLE_DIR.rglob("*.jpg"))
        if not sample_images:
            self.skipTest("No sample images found under samples/")

        test_img = sample_images[0]
        result, annotated = self.pipeline.predict_and_annotate(test_img)

        self.assertIsInstance(result, dict)
        self.assertIsInstance(annotated, np.ndarray)
        # Check JSON serializable
        json_str = json.dumps(result)
        self.assertTrue(len(json_str) > 0)

    def test_real_pipeline_cpu_inference(self):
        pipeline_cpu = PPEPipeline(
            person_model_path=PERSON_MODEL_PATH,
            helmet_model_path=HELMET_MODEL_PATH,
            device="cpu",
        )
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        result = pipeline_cpu.predict(dummy)
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
