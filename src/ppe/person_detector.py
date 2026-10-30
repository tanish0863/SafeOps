"""
src/ppe/person_detector.py — Reusable YOLOv8 Person Detection Module.

Wraps a pretrained YOLOv8 model (defaulting to yolov8n.pt) to detect
people in images and video frames. Exposes a clean, consistent API that
integrates directly with the SafeOps PPE compliance pipeline.

Key Responsibilities:
  - Load the person detection checkpoint exactly once at initialization.
  - Explicit device selection (CUDA or CPU) with strict validation.
  - Filter detections strictly to the 'person' class.
  - Return bounding boxes in original-image pixel coordinates [x1, y1, x2, y2].
  - Provide stable per-frame identifiers (person_id).
  - Support single-pass inference returning both standardized records and raw
    Ultralytics Results objects (predict_with_result).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_VALID_DEVICES: frozenset[str] = frozenset({"cuda", "cpu"})
_DEFAULT_MODEL_NAME = "yolov8n.pt"


class PersonDetector:
    """
    Single-instance YOLOv8 person detection adapter.

    Parameters
    ----------
    model_path : Union[str, Path]
        Path to the pretrained YOLO model (e.g., 'yolov8n.pt' or 'yolov8s.pt').
    device : str
        Inference device — ``"cuda"`` or ``"cpu"``.
        A ``RuntimeError`` is raised if ``"cuda"`` is requested but unavailable.
    confidence_threshold : float
        Minimum detection confidence threshold in [0.0, 1.0].
    iou_threshold : float
        IoU threshold for Non-Maximum Suppression (NMS) in [0.0, 1.0].
    input_size : int
        Image inference size (default 640).
    """

    def __init__(
        self,
        model_path: Union[str, Path] = _DEFAULT_MODEL_NAME,
        device: str = "cuda",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        input_size: int = 640,
    ) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._input_size = input_size

        # ── Parameter validation ──────────────────────────────────────────
        self._validate_model_path()
        self._validate_device()
        self._validate_thresholds()

        # ── Load model ────────────────────────────────────────────────────
        logger.info(
            "[PersonDetector] Loading checkpoint '%s' on device='%s'",
            self._model_path,
            self._device,
        )
        self._model = YOLO(str(self._model_path))
        self._model.to(self._device)

        # ── Resolve person class ID ───────────────────────────────────────
        self._checkpoint_names: dict[int, str] = getattr(self._model, "names", {})
        self._person_class_id: int = self._resolve_person_class_id()

        logger.info(
            "[PersonDetector] Person class resolved to ID %d ('%s')",
            self._person_class_id,
            self._checkpoint_names.get(self._person_class_id, "unknown"),
        )

        # ── Warm-up pass ──────────────────────────────────────────────────
        self._warmup()
        logger.info("[PersonDetector] Ready.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self, image: Union[str, Path, np.ndarray]) -> list[dict]:
        """
        Detect persons in *image* and return a list of detection dicts.

        Parameters
        ----------
        image : Union[str, Path, np.ndarray]
            An image file path or an OpenCV BGR ``np.ndarray`` (HxWx3).

        Returns
        -------
        list[dict]
            Each element has:
              - ``person_id``: int (index starting from 0)
              - ``bbox``: [x1, y1, x2, y2] (float pixel coordinates)
              - ``confidence``: float [0.0, 1.0]
        """
        predictions, _ = self.predict_with_result(image)
        return predictions

    def predict_with_result(
        self, image: Union[str, Path, np.ndarray]
    ) -> tuple[list[dict], object]:
        """
        Run person detection and return standardized predictions along with
        the raw Ultralytics Results object for single-pass processing.

        Returns
        -------
        predictions : list[dict]
            Standardized detection list.
        result : ultralytics.engine.results.Results
            The raw Ultralytics result for the image.
        """
        prepared = self._prepare_image(image)

        try:
            results = self._model.predict(
                source=prepared,
                imgsz=self._input_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=[self._person_class_id],
                device=self._device,
                verbose=False,
                half=(self._device == "cuda"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"[PersonDetector] YOLO inference failed: {exc}"
            ) from exc

        result = results[0]
        predictions = self._parse_result(result)
        return predictions, result

    # -----------------------------------------------------------------------
    # Validation helpers
    # -----------------------------------------------------------------------

    def _validate_model_path(self) -> None:
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"[PersonDetector] Checkpoint not found: {self._model_path}"
            )

    def _validate_device(self) -> None:
        if self._device not in _VALID_DEVICES:
            raise ValueError(
                f"[PersonDetector] Invalid device '{self._device}'. "
                f"Supported values: {sorted(_VALID_DEVICES)}"
            )
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "[PersonDetector] CUDA was requested but is not available "
                "on this system. Use device='cpu' or ensure a CUDA-capable "
                "GPU is configured."
            )

    def _validate_thresholds(self) -> None:
        if not isinstance(self._confidence_threshold, (int, float)) or not (
            0.0 <= float(self._confidence_threshold) <= 1.0
        ):
            raise ValueError(
                f"[PersonDetector] confidence_threshold must be in [0.0, 1.0]; "
                f"got {self._confidence_threshold!r}"
            )
        if not isinstance(self._iou_threshold, (int, float)) or not (
            0.0 <= float(self._iou_threshold) <= 1.0
        ):
            raise ValueError(
                f"[PersonDetector] iou_threshold must be in [0.0, 1.0]; "
                f"got {self._iou_threshold!r}"
            )
        if not isinstance(self._input_size, int) or self._input_size <= 0:
            raise ValueError(
                f"[PersonDetector] input_size must be a positive int; "
                f"got {self._input_size!r}"
            )

    def _resolve_person_class_id(self) -> int:
        for cid, name in self._checkpoint_names.items():
            if str(name).strip().lower() == "person":
                return int(cid)
        # Fallback to class 0 if names not explicitly labeled
        if 0 in self._checkpoint_names:
            logger.warning(
                "[PersonDetector] 'person' class name not explicitly found in %s; defaulting to class ID 0",
                self._checkpoint_names,
            )
            return 0
        raise RuntimeError(
            f"[PersonDetector] Model does not contain a 'person' class. "
            f"Available classes: {self._checkpoint_names}"
        )

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def _warmup(self) -> None:
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy,
            imgsz=320,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            classes=[self._person_class_id],
            device=self._device,
            verbose=False,
            half=(self._device == "cuda"),
        )
        logger.info("[PersonDetector] Warm-up complete.")

    def _prepare_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(
                    f"[PersonDetector] Image not found: {path}"
                )
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(
                    f"[PersonDetector] cv2.imread returned None for: {path}. "
                    "The file may be corrupt or in an unsupported format."
                )
            return img

        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"[PersonDetector] image must be a file path or np.ndarray; "
                f"got {type(image).__name__!r}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"[PersonDetector] NumPy image must be HxWx3 (BGR); "
                f"got shape {image.shape}"
            )
        return image

    def _parse_result(self, result) -> list[dict]:
        predictions: list[dict] = []
        if result.boxes is None or len(result.boxes) == 0:
            return predictions

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        person_idx = 0
        for xyxy, conf, cid in zip(boxes_xyxy, confidences, class_ids):
            if cid != self._person_class_id:
                continue
            if float(conf) < self._confidence_threshold:
                continue

            predictions.append(
                {
                    "person_id": person_idx,
                    "bbox": [round(float(v), 2) for v in xyxy],
                    "confidence": round(float(conf), 4),
                }
            )
            person_idx += 1

        return predictions
