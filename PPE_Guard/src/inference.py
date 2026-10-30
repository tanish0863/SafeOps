"""Single shared YOLO Pose inference. Loaded once, run once per frame."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawInference:
    """Index i in every array refers to the SAME person. Never reorder independently."""
    boxes: np.ndarray       # (N, 4) xyxy pixels
    scores: np.ndarray      # (N,)
    keypoints: np.ndarray   # (N, 17, 3) x, y, conf


class PoseEngine:
    def __init__(self, model_path: str = "models/yolo26n-pose.pt", conf: float = 0.35,
                 imgsz: int = 640, device=None):
        import torch
        from ultralytics import YOLO

        if device is None:
            device = 0 if torch.cuda.is_available() else "cpu"
        self.device = device
        self.half = device != "cpu"
        self.conf, self.imgsz = conf, imgsz
        p = Path(model_path)
        # Offline deployment: a path into models/ must exist locally. A bare name
        # (e.g. "yolo11n-pose.pt") lets Ultralytics auto-download on first use.
        if p.parent != Path(".") and not p.exists():
            raise FileNotFoundError(
                f"{model_path} not found. While online, run "
                f"`python scripts/download_model.py` once; weights then live in models/.")
        self.model = YOLO(model_path)

    def infer(self, frame: np.ndarray) -> RawInference:
        result = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                    device=self.device, half=self.half, verbose=False)[0]
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            return RawInference(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0, 17, 3)))
        return RawInference(
            boxes=result.boxes.xyxy.cpu().numpy(),
            scores=result.boxes.conf.cpu().numpy(),
            keypoints=result.keypoints.data.cpu().numpy(),
        )
