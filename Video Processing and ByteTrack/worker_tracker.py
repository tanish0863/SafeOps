"""Worker detection + ByteTrack tracking using YOLOv8 Pose (Ultralytics).

Tracking runs on the *person* detections only. Never run ByteTrack on PPE
detections to create worker IDs.
"""
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO


class WorkerTracker:
    def __init__(
        self,
        model_path: str = "models/yolov8s-pose.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.10,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
        half: bool | None = None,
    ):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.use_cuda = str(device).startswith("cuda")
        # FP16 saves VRAM/time on the 3050; meaningless on CPU.
        self.half = self.use_cuda if half is None else half
        self.tracker = tracker
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        path = Path(model_path)
        if not path.exists():
            print(f"{path} not found; letting Ultralytics download '{path.name}'")
            path = Path(path.name)
        self.model = YOLO(str(path))

        print("Worker detector loaded")
        if self.use_cuda:
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("WARNING: running on CPU (CUDA not available)")

    def reset(self) -> None:
        """Clear tracker state. Call between videos so IDs restart cleanly."""
        self.model.predictor = None

    def update(self, frame) -> list[dict]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            classes=[0],  # COCO class 0 = person
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        result = results[0]
        # Boxes may be empty, and IDs are None until a track is confirmed.
        if result.boxes is None or result.boxes.id is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        kp_xy = kp_conf = None
        if result.keypoints is not None:
            kp_xy = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                kp_conf = result.keypoints.conf.cpu().numpy()

        workers = []
        for i in range(len(track_ids)):
            workers.append(
                {
                    "track_id": int(track_ids[i]),
                    "bbox_xyxy": [float(v) for v in boxes[i]],
                    "confidence": float(confs[i]),
                    "keypoints_xy": kp_xy[i].tolist() if kp_xy is not None else None,
                    "keypoints_conf": kp_conf[i].tolist() if kp_conf is not None else None,
                }
            )
        return workers
