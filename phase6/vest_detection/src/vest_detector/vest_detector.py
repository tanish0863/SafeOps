"""
vest_detector.py
----------------
Core VestDetector class providing unified API for full-frame detection,
localized torso crop detection, and three-state compliance assessment.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import cv2
import numpy as np

from .config import VestDetectorConfig
from .model_loader import ModelBackend, load_model
from .result_adapter import ResultAdapter
from .schemas import Detection, TorsoVestAssessment, VestDetectionResult
from .torso_inference import TorsoInferenceEngine, crop_torso_region

logger = logging.getLogger("vest_detector")


class VestDetector:
    """
    Main Safety Vest Detector interface for SafeOps Phase 6.

    Provides:
      - Full-frame inference: `detect(image)`
      - Localized torso crop inference: `detect_torso(image, torso_bbox)`
      - Video frame inference: `detect_frame(frame)`
      - Three-state vest assessment: `assess_torso(image, torso_bbox)`
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        config: Optional[VestDetectorConfig] = None,
        backend: Optional[ModelBackend] = None,
    ):
        """
        Initialize the VestDetector.

        Parameters
        ----------
        model_path : Path to pre-trained YOLO checkpoint (e.g., .pt file)
        device : "cuda" (default) or "cpu"
        config : Optional configuration overrides
        backend : Optional custom ModelBackend instance (e.g. MockModelBackend)
        """
        self.config = config or VestDetectorConfig(device=device)
        self.config.device = device

        if backend is not None:
            self.backend = backend
        else:
            self.backend = load_model(
                model_path=model_path,
                config=self.config,
                use_mock=False,
            )

        self.adapter = ResultAdapter(config=self.config)
        self.torso_engine = TorsoInferenceEngine(config=self.config)

    def _load_image_array(self, image: Union[np.ndarray, str, Path]) -> np.ndarray:
        """
        Normalize input into a 3-channel BGR NumPy array.
        """
        if isinstance(image, (str, Path)):
            path_str = str(image)
            img = cv2.imread(path_str)
            if img is None:
                raise ValueError(f"Could not load image from file path: {path_str}")
        elif isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError("Input image array is empty (size 0).")
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}. Expected np.ndarray, str, or Path.")

        # Ensure 3-channel BGR format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img

    def _get_raw_predict_threshold(self) -> float:
        """
        Return the minimum configured threshold so that raw model inference
        does not prematurely discard candidates needed by class-specific filters.
        """
        return min(
            self.config.confidence_threshold,
            self.config.vest_confidence_threshold,
            self.config.no_vest_confidence_threshold,
        )

    def detect(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """
        Detect safety vests across a complete image.

        Parameters
        ----------
        image : BGR NumPy array or file path

        Returns
        -------
        Dictionary conforming to SafeOps schema:
        {
          "schema_version": "1.0",
          "detections": [
            {
              "class_name": "WITH_VEST",
              "confidence": 0.93,
              "bbox": [x1, y1, x2, y2]
            }
          ]
        }
        """
        img_bgr = self._load_image_array(image)
        h, w = img_bgr.shape[:2]

        raw_conf = self._get_raw_predict_threshold()
        raw_boxes, raw_scores, raw_classes = self.backend.predict(
            image_bgr=img_bgr,
            conf_threshold=raw_conf,
            iou_threshold=self.config.iou_threshold,
        )

        detections = self.adapter.adapt_detections(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_classes=raw_classes,
            offset_x=0.0,
            offset_y=0.0,
            image_shape=(h, w),
        )

        result = self.adapter.build_result(
            detections=detections,
            metadata={"image_width": w, "image_height": h, "mode": "full_image"},
        )

        return result.to_dict()

    def detect_torso(
        self,
        image: Union[np.ndarray, str, Path],
        torso_bbox: Sequence[Union[int, float]],
    ) -> Dict[str, Any]:
        """
        Detect a vest within a supplied torso region.

        CRITICAL: All returned bounding box coordinates are mapped back to
        the ORIGINAL full-frame image pixel space. Crop-relative coordinates
        are never returned as original-image coordinates.

        Parameters
        ----------
        image : Complete full-frame image (BGR array or path)
        torso_bbox : [x1, y1, x2, y2] in original-image pixel space

        Returns
        -------
        Dictionary conforming to SafeOps schema with coordinates in original-image space.
        """
        img_bgr = self._load_image_array(image)
        h, w = img_bgr.shape[:2]

        # Crop the torso region with padding
        crop, padded_box, (offset_x, offset_y) = crop_torso_region(
            image_bgr=img_bgr,
            torso_bbox=torso_bbox,
            padding_fraction=self.config.padding_fraction,
        )

        if crop.size == 0:
            return {"schema_version": "1.0", "detections": []}

        raw_conf = self._get_raw_predict_threshold()
        raw_boxes, raw_scores, raw_classes = self.backend.predict(
            image_bgr=crop,
            conf_threshold=raw_conf,
            iou_threshold=self.config.iou_threshold,
        )

        # Map predictions back to global full-image pixel coordinates
        detections = self.adapter.adapt_detections(
            raw_boxes=raw_boxes,
            raw_scores=raw_scores,
            raw_classes=raw_classes,
            offset_x=offset_x,
            offset_y=offset_y,
            image_shape=(h, w),
        )

        result = self.adapter.build_result(
            detections=detections,
            metadata={
                "torso_bbox": [round(float(c), 2) for c in torso_bbox[:4]],
                "crop_offset": [round(offset_x, 2), round(offset_y, 2)],
                "mode": "torso_crop",
            },
        )

        return result.to_dict()

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect vests in an individual video frame (does not perform tracking).
        """
        return self.detect(frame)

    def assess_torso(
        self,
        image: Union[np.ndarray, str, Path],
        torso_bbox: Sequence[Union[int, float]],
        detections: Optional[Sequence[Union[Detection, Dict[str, Any]]]] = None,
    ) -> TorsoVestAssessment:
        """
        Perform a comprehensive three-state vest assessment (PRESENT, ABSENT, UNKNOWN)
        on the designated torso region.
        """
        img_bgr = self._load_image_array(image)

        if detections is not None:
            cand_detections = [
                d if isinstance(d, Detection) else Detection(
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                )
                for d in detections
            ]
        else:
            # 1. Run localized torso detection
            torso_res = self.detect_torso(img_bgr, torso_bbox)
            cand_detections = [
                Detection(
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=d["bbox"],
                )
                for d in torso_res.get("detections", [])
            ]

            # 2. If localized crop yields no vest detections, check full-frame detections
            if not cand_detections:
                full_res = self.detect(img_bgr)
                cand_detections = [
                    Detection(
                        class_name=d["class_name"],
                        confidence=d["confidence"],
                        bbox=d["bbox"],
                    )
                    for d in full_res.get("detections", [])
                ]

        # Evaluate three-state decision logic
        return self.torso_engine.assess(
            image_bgr=img_bgr,
            torso_bbox=torso_bbox,
            detections=cand_detections,
        )
