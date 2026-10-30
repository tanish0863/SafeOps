"""
src/ppe/helmet_adapter.py — Reusable YOLOv8 Helmet Detection Adapter.

Wraps the trained checkpoint at runs/detect/train/weights/best.pt and
exposes a clean, stable inference API for downstream consumers such as
the SafeOps PPE compliance pipeline.

Responsibilities:
  - Load the checkpoint exactly once at initialization.
  - Detect ``with_helmet`` and ``without_helmet`` classes.
  - Discard ``bike`` class detections entirely.
  - Map original class names to uppercase standardized labels.
  - Return bounding boxes as original-image XYXY pixel coordinates.
  - Support per-class configurable confidence thresholds.
  - Provide single-pass inference with optional Ultralytics result object.

Out of scope (by design):
  - Person detection or person-crop association.
  - Worker tracking or track-ID management.
  - Safety-vest or other PPE categories.
  - Final compliance evaluation.
  - Thread-safety (see Known Limitations in docs/helmet_integration.md).
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

# ---------------------------------------------------------------------------
# Class mapping — source-of-truth for all label transformations.
# Keys are the raw names stored in the checkpoint's metadata.
# Values are the standardized uppercase labels returned to callers.
# ---------------------------------------------------------------------------
CLASS_MAPPING: dict[str, str] = {
    "with_helmet":    "WITH_HELMET",
    "without_helmet": "WITHOUT_HELMET",
}

# Classes to silently discard; never returned to callers.
_EXCLUDED_CLASSES: frozenset[str] = frozenset({"bike"})

# Allowed device strings.
_VALID_DEVICES: frozenset[str] = frozenset({"cuda", "cpu"})


class PPEHelmetAdapter:
    """
    Single-instance YOLOv8 helmet-detection adapter.

    Parameters
    ----------
    model_path:
        Path to the trained YOLOv8 checkpoint (.pt file).
    device:
        Inference device — ``"cuda"`` or ``"cpu"``.
        CUDA is **not** silently downgraded to CPU; a ``RuntimeError`` is
        raised if CUDA is requested but unavailable.
    confidence_threshold:
        Either a single float applied to all classes, or a dict mapping
        raw class names (``"with_helmet"``, ``"without_helmet"``) to
        per-class float thresholds.
    iou_threshold:
        IoU threshold for NMS. Must be in [0.0, 1.0].

    Examples
    --------
    Shared threshold::

        adapter = PPEHelmetAdapter(
            model_path="runs/detect/train/weights/best.pt",
            device="cuda",
            confidence_threshold=0.25,
        )

    Per-class thresholds::

        adapter = PPEHelmetAdapter(
            model_path="runs/detect/train/weights/best.pt",
            device="cuda",
            confidence_threshold={
                "with_helmet":    0.30,
                "without_helmet": 0.40,
            },
        )
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cuda",
        confidence_threshold: Union[float, dict[str, float]] = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self._model_path = Path(model_path)
        self._device = device
        self._iou_threshold = iou_threshold

        # ── Validate inputs before any expensive operations ────────────────
        self._validate_model_path()
        self._validate_device()
        self._validate_iou_threshold(iou_threshold)

        # Parse and validate confidence thresholds
        self._class_conf: dict[str, float] = self._parse_confidence_threshold(
            confidence_threshold
        )
        # NMS uses the minimum configured threshold so that class-level
        # filtering (applied post-NMS) can see all borderline detections.
        self._nms_conf: float = min(self._class_conf.values())

        # ── Load model ────────────────────────────────────────────────────
        logger.info(
            "[PPEHelmetAdapter] Loading checkpoint '%s' on device='%s'",
            self._model_path,
            self._device,
        )
        self._model = YOLO(str(self._model_path))
        self._model.to(self._device)

        # ── Resolve class IDs from checkpoint metadata ────────────────────
        self._checkpoint_names: dict[int, str] = getattr(self._model, "names", {})
        self._validate_required_classes()

        # Map raw class name → class ID (e.g. "with_helmet" → 1)
        self._name_to_id: dict[str, int] = {
            v: k for k, v in self._checkpoint_names.items()
        }
        # Map class ID → standardized label (excludes bike / unknown)
        self._id_to_label: dict[int, str] = {}
        for raw_name, label in CLASS_MAPPING.items():
            cid = self._name_to_id.get(raw_name)
            if cid is not None:
                self._id_to_label[cid] = label

        logger.info(
            "[PPEHelmetAdapter] Checkpoint classes: %s",
            self._checkpoint_names,
        )
        logger.info(
            "[PPEHelmetAdapter] Active class→label map: %s",
            self._id_to_label,
        )

        # ── Warm-up pass to pre-JIT CUDA kernels ─────────────────────────
        self._warmup()
        logger.info("[PPEHelmetAdapter] Ready.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def predict(self, image: Union[str, Path, np.ndarray]) -> list[dict]:
        """
        Run helmet detection on *image* and return a list of detections.

        Parameters
        ----------
        image:
            An image file path **or** an OpenCV BGR ``np.ndarray``.

        Returns
        -------
        list[dict]
            Each element has the keys ``class_name``, ``confidence``,
            and ``bbox`` (``[x1, y1, x2, y2]`` in original-image pixels).
            Returns ``[]`` when no helmet-class objects are detected above
            the configured thresholds.

        Raises
        ------
        FileNotFoundError
            Image path does not exist.
        ValueError
            Provided NumPy array is not a valid 3-channel image.
        RuntimeError
            YOLO inference raised an unexpected exception.
        """
        predictions, _ = self.predict_with_result(image)
        return predictions

    def predict_with_result(
        self, image: Union[str, Path, np.ndarray]
    ) -> tuple[list[dict], object]:
        """
        Run helmet detection and return both the standardized predictions
        and the raw Ultralytics ``Results`` object.

        Use this method when you need to generate an annotated image via
        ``result.plot()`` without running inference a second time.

        Parameters
        ----------
        image:
            An image file path **or** an OpenCV BGR ``np.ndarray``.

        Returns
        -------
        predictions : list[dict]
            Standardized detection list (see :meth:`predict`).
        result : ultralytics.engine.results.Results
            The raw Ultralytics result for the first (and only) image.
        """
        prepared = self._prepare_image(image)

        try:
            results = self._model.predict(
                source=prepared,
                imgsz=640,
                conf=self._nms_conf,
                iou=self._iou_threshold,
                device=self._device,
                verbose=False,
                half=(self._device == "cuda"),
            )
        except Exception as exc:
            raise RuntimeError(
                f"[PPEHelmetAdapter] YOLO inference failed: {exc}"
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
                f"[PPEHelmetAdapter] Checkpoint not found: {self._model_path}"
            )

    def _validate_device(self) -> None:
        if self._device not in _VALID_DEVICES:
            raise ValueError(
                f"[PPEHelmetAdapter] Invalid device '{self._device}'. "
                f"Supported values: {sorted(_VALID_DEVICES)}"
            )
        if self._device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "[PPEHelmetAdapter] CUDA was requested but is not available "
                "on this system. Use device='cpu' or ensure a CUDA-capable "
                "GPU with the correct drivers is present."
            )

    @staticmethod
    def _validate_iou_threshold(value: float) -> None:
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"[PPEHelmetAdapter] iou_threshold must be in [0.0, 1.0]; got {value!r}"
            )

    @staticmethod
    def _validate_single_conf(name: str, value: float) -> None:
        if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
            raise ValueError(
                f"[PPEHelmetAdapter] confidence_threshold['{name}'] must be a float "
                f"in [0.0, 1.0]; got {value!r}"
            )

    def _parse_confidence_threshold(
        self, threshold: Union[float, dict]
    ) -> dict[str, float]:
        """
        Normalise the caller-supplied threshold into a per-class dict.
        """
        if isinstance(threshold, dict):
            result: dict[str, float] = {}
            for key, val in threshold.items():
                self._validate_single_conf(key, val)
                result[key] = float(val)
            # Fill missing classes with a permissive default so every
            # CLASS_MAPPING key always has an entry.
            for cls_name in CLASS_MAPPING:
                result.setdefault(cls_name, 0.0)
            return result

        # Single scalar threshold
        if not isinstance(threshold, (int, float)):
            raise ValueError(
                f"[PPEHelmetAdapter] confidence_threshold must be a float or dict; "
                f"got {type(threshold).__name__!r}"
            )
        self._validate_single_conf("confidence_threshold", threshold)
        return {cls_name: float(threshold) for cls_name in CLASS_MAPPING}

    def _validate_required_classes(self) -> None:
        """
        Ensure every CLASS_MAPPING key is present in the checkpoint.
        """
        checkpoint_names = set(self._checkpoint_names.values())
        required = set(CLASS_MAPPING.keys())
        missing = required - checkpoint_names
        if missing:
            raise RuntimeError(
                f"[PPEHelmetAdapter] Checkpoint is missing required classes: {missing}. "
                f"Classes found in checkpoint: {checkpoint_names}"
            )

    # -----------------------------------------------------------------------
    # Inference helpers
    # -----------------------------------------------------------------------

    def _warmup(self) -> None:
        """Single dummy forward pass to pre-compile CUDA kernels."""
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy,
            imgsz=320,
            conf=self._nms_conf,
            iou=self._iou_threshold,
            device=self._device,
            verbose=False,
            half=(self._device == "cuda"),
        )
        logger.info("[PPEHelmetAdapter] Warm-up complete.")

    def _prepare_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Accept a file path or an ndarray; always return a BGR ndarray.

        Raises
        ------
        FileNotFoundError : path does not exist.
        ValueError        : ndarray is not a valid 3-channel image.
        RuntimeError      : cv2.imread returned None for an existing file.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(
                    f"[PPEHelmetAdapter] Image not found: {path}"
                )
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(
                    f"[PPEHelmetAdapter] cv2.imread returned None for: {path}. "
                    "The file may be corrupt or in an unsupported format."
                )
            return img

        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"[PPEHelmetAdapter] image must be a file path or np.ndarray; "
                f"got {type(image).__name__!r}"
            )
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"[PPEHelmetAdapter] NumPy image must be HxWx3 (BGR); "
                f"got shape {image.shape}"
            )
        return image

    def _parse_result(self, result) -> list[dict]:
        """
        Convert a single Ultralytics ``Results`` object into the
        standardized prediction list.

        - Discards ``bike`` class detections.
        - Applies per-class confidence thresholds.
        - Returns original-image XYXY pixel coordinates.
        """
        predictions: list[dict] = []

        if result.boxes is None or len(result.boxes) == 0:
            return predictions

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()       # shape (N, 4)
        confidences = result.boxes.conf.cpu().numpy()       # shape (N,)
        class_ids   = result.boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        for xyxy, conf, cid in zip(boxes_xyxy, confidences, class_ids):
            raw_name = self._checkpoint_names.get(cid, "")

            # Discard excluded classes (bike) and unknown classes
            if raw_name in _EXCLUDED_CLASSES or raw_name not in CLASS_MAPPING:
                continue

            label = CLASS_MAPPING[raw_name]

            # Apply per-class confidence threshold
            class_conf_threshold = self._class_conf.get(raw_name, self._nms_conf)
            if float(conf) < class_conf_threshold:
                continue

            predictions.append(
                {
                    "class_name": label,
                    "confidence": round(float(conf), 4),
                    "bbox": [round(float(v), 2) for v in xyxy],
                }
            )

        return predictions
