"""Modular PPE Detection Subsystem for SafeOps.

Architecture:
- BasePPEDetector: Abstract interface for PPE detectors.
- YOLOv8HelmetDetector: Implementation wrapping models/ppe_helmet.pt.
  * Maps 'with_helmet' -> PPEItemType.HELMET.
  * Safely ignores non-worker classes like 'bike'.
  * Distinguishes candidate absence from definitive proof.
- ModularPPEEngine: Composite detector aggregating helmet and vest models.
  * Gracefully handles missing vest model by reporting status=UNKNOWN with
    reason='vest_model_not_installed', never fabricating vest detections.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from src.contracts.schemas import PPEDetectionItem, PPEItemType, PPEStatus, WorkerPPEState


class BasePPEDetector(ABC):
    """Abstract base detector for personal protective equipment."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[PPEDetectionItem]:
        """Detect PPE in a frame and return structured detection items."""
        pass


class YOLOv8HelmetDetector(BasePPEDetector):
    """Helmet detector wrapping custom weights (e.g., models/ppe_helmet.pt)."""

    def __init__(
        self,
        model_path: str = "models/ppe_helmet.pt",
        conf: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
        half: Optional[bool] = None,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Helmet model weights not found at: {model_path}")

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_cuda = str(self.device).startswith("cuda")
        self.half = self.use_cuda if half is None else half
        self.conf = conf
        self.imgsz = imgsz

        self.model = YOLO(str(path))
        # Inspect model class mapping
        # Expected: {0: 'bike', 1: 'with_helmet', 2: 'without_helmet'}
        self.class_names: Dict[int, str] = self.model.names
        self.helmet_class_ids = [
            cid for cid, name in self.class_names.items()
            if "helmet" in name.lower() and not name.lower().startswith("without") and not name.lower().startswith("no")
        ]
        print(f"Loaded Helmet Detector: {len(self.helmet_class_ids)} helmet class ID(s) identified in {self.class_names}")

    def detect(self, frame: np.ndarray) -> List[PPEDetectionItem]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        detections: List[PPEDetectionItem] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.int().cpu().numpy()

        for box, conf, cls_id in zip(boxes, confs, classes):
            if int(cls_id) in self.helmet_class_ids:
                detections.append(
                    PPEDetectionItem(
                        item_type=PPEItemType.HELMET,
                        bbox_xyxy=[float(v) for v in box],
                        confidence=float(conf),
                        raw_class=self.class_names.get(int(cls_id)),
                    )
                )

        return detections


class ModularPPEEngine:
    """Composite PPE detection engine managing both helmet and vest detectors."""

    def __init__(
        self,
        helmet_model_path: Optional[str] = "models/ppe_helmet.pt",
        vest_model_path: Optional[str] = None,
        conf: float = 0.35,
        device: Optional[str] = None,
    ):
        self.helmet_detector: Optional[YOLOv8HelmetDetector] = None
        if helmet_model_path and os.path.exists(helmet_model_path):
            self.helmet_detector = YOLOv8HelmetDetector(
                model_path=helmet_model_path,
                conf=conf,
                device=device,
            )
        else:
            print("WARNING: Helmet model path not specified or file not found.")

        # Vest detector is modular and pluggable
        self.vest_detector: Optional[BasePPEDetector] = None
        if vest_model_path and os.path.exists(vest_model_path):
            # Future vest model loading
            pass
        else:
            # Documented: no vest model installed in the current phase
            self._vest_status_reason = "vest_model_not_installed"

    @property
    def has_vest_model(self) -> bool:
        return self.vest_detector is not None

    def detect_all(self, frame: np.ndarray) -> Dict[PPEItemType, List[PPEDetectionItem]]:
        """Run all active PPE detectors and group detections by PPEItemType."""
        results: Dict[PPEItemType, List[PPEDetectionItem]] = {
            PPEItemType.HELMET: [],
            PPEItemType.VEST: [],
        }

        if self.helmet_detector is not None:
            results[PPEItemType.HELMET] = self.helmet_detector.detect(frame)

        if self.vest_detector is not None:
            results[PPEItemType.VEST] = self.vest_detector.detect(frame)

        return results

    def default_vest_state(self) -> WorkerPPEState:
        """Return the explicit state for vests when model is unavailable."""
        return WorkerPPEState(
            status=PPEStatus.UNKNOWN,
            confidence=None,
            bbox_xyxy=None,
            reason="vest_model_not_installed",
        )

