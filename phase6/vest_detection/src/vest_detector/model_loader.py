"""
model_loader.py
---------------
Model loading, weight management, device placement, and inference execution.
Supports Ultralytics YOLOv8 checkpoints as well as deterministic mock backends
for test environments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .config import VestDetectorConfig

logger = logging.getLogger("vest_detector.model_loader")


class ModelBackend:
    """Abstract interface for model execution."""

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Run inference on a single image.

        Returns
        -------
        boxes : List of [x1, y1, x2, y2]
        scores : List of confidence scores
        classes : List of raw class names
        """
        raise NotImplementedError


class UltralyticsYOLOBackend(ModelBackend):
    """Execution backend using Ultralytics YOLO runtime."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        half: bool = True,
        image_size: int = 640,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.device = device
        self.half = half and ("cuda" in device)
        self.image_size = image_size

        logger.info(f"Loading YOLO model from: {self.model_path} on device: {self.device}")
        self.yolo = YOLO(self.model_path)
        self.names = getattr(self.yolo, "names", {})  # dict or list of class names

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Run YOLO inference on BGR image array.
        """
        results = self.yolo.predict(
            source=image_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            imgsz=self.image_size,
            verbose=False,
        )

        boxes: List[List[float]] = []
        scores: List[float] = []
        classes: List[str] = []

        if not results:
            return boxes, scores, classes

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return boxes, scores, classes

        boxes_data = res.boxes.xyxy.cpu().numpy()
        confs_data = res.boxes.conf.cpu().numpy()
        cls_data = res.boxes.cls.cpu().numpy().astype(int)

        for b, s, c_idx in zip(boxes_data, confs_data, cls_data):
            if isinstance(self.names, dict):
                c_name = self.names.get(c_idx, str(c_idx))
            elif isinstance(self.names, (list, tuple)) and 0 <= c_idx < len(self.names):
                c_name = self.names[c_idx]
            else:
                c_name = str(c_idx)

            boxes.append([float(coord) for coord in b[:4]])
            scores.append(float(s))
            classes.append(c_name)

        return boxes, scores, classes


class MockModelBackend(ModelBackend):
    """
    Deterministic mock backend for unit tests, offline validation,
    and synthetic scenario generation without requiring model weights.
    """

    def __init__(self, predefined_detections: Optional[List[dict]] = None):
        self.predefined_detections = predefined_detections or []

    def set_next_detections(self, detections: List[dict]):
        """Set detections to be returned on next predict() call."""
        self.predefined_detections = detections

    def predict(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        boxes: List[List[float]] = []
        scores: List[float] = []
        classes: List[str] = []

        for d in self.predefined_detections:
            s = float(d.get("confidence", 0.90))
            if s >= conf_threshold:
                boxes.append([float(c) for c in d["bbox"]])
                scores.append(s)
                classes.append(str(d["class_name"]))

        return boxes, scores, classes


def resolve_device(requested_device: str, fallback_to_cpu: bool = True) -> str:
    """
    Verify and return the optimal available computing device (CUDA vs. CPU).
    """
    if "cuda" in requested_device.lower():
        if torch.cuda.is_available():
            return requested_device
        if fallback_to_cpu:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        raise RuntimeError("CUDA is not available on this system.")
    return "cpu"


def load_model(
    model_path: Optional[Union[str, Path]] = None,
    config: Optional[VestDetectorConfig] = None,
    use_mock: bool = False,
) -> ModelBackend:
    """
    Factory function to instantiate the model execution backend.
    """
    cfg = config or VestDetectorConfig()
    device = resolve_device(cfg.device, cfg.fallback_to_cpu)

    if use_mock:
        logger.info("Instantiating MockModelBackend for testing.")
        return MockModelBackend()

    # If no path specified, check standard models directory
    if model_path is None:
        standard_path = cfg.models_dir / cfg.default_model_filename
        if standard_path.exists():
            model_path = standard_path
        else:
            logger.warning(
                f"Model weights not found at {standard_path}. "
                "Defaulting to MockModelBackend for offline operation."
            )
            return MockModelBackend()

    path_obj = Path(model_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path_obj}")

    return UltralyticsYOLOBackend(
        model_path=path_obj,
        device=device,
        half=cfg.half_precision,
        image_size=cfg.image_size,
    )
