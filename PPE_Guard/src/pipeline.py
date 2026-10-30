"""Phase 2 pipeline: ONE inference pass -> boxes + keypoints (index-matched) -> head regions."""
from __future__ import annotations

from typing import Iterator, Optional

import cv2

from .head_region import estimate_head_region
from .inference import PoseEngine
from .person_detector import PersonDetector
from .pose_estimator import PoseEstimator
from .schemas import FrameRecord, Person


class Phase2Pipeline:
    def __init__(self, model_path: str = "models/yolo26n-pose.pt", conf: float = 0.35,
                 keypoint_conf: float = 0.5, imgsz: int = 640, device=None, engine=None):
        self.engine = engine or PoseEngine(model_path, conf=conf, imgsz=imgsz, device=device)
        self.detector = PersonDetector(engine=self.engine)
        self.estimator = PoseEstimator(engine=self.engine, keypoint_conf=keypoint_conf)

    def process_frame(self, frame, frame_number: int = 0,
                      timestamp_s: Optional[float] = None) -> FrameRecord:
        h, w = frame.shape[:2]
        raw = self.engine.infer(frame)                       # single inference
        detections = self.detector.from_raw(raw, w, h)       # same index i
        poses = self.estimator.from_raw(raw, w, h)           # same index i
        persons = []
        for det, pose in zip(detections, poses):
            assert det.detection_id == pose.detection_id
            kp_dicts = {k: (v.model_dump() if v else None) for k, v in pose.keypoints.items()}
            head = estimate_head_region(kp_dicts, w, h, person_bbox=det.bbox)
            persons.append(Person(
                detection_id=det.detection_id, bbox=det.bbox, confidence=det.confidence,
                head_region=head["head_region"], head_visibility=head["head_visibility"],
                keypoints=pose.keypoints))
        return FrameRecord(frame_number=frame_number, timestamp_s=timestamp_s,
                           frame_width=w, frame_height=h, persons=persons)

    def process_video(self, path: str, stride: int = 1) -> Iterator[tuple]:
        """Yield (frame, FrameRecord) for every `stride`-th frame."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if n % stride == 0:
                    ts = round(n / fps, 3) if fps > 0 else None
                    yield frame, self.process_frame(frame, n, ts)
                n += 1
        finally:
            cap.release()


def write_jsonl_line(fh, record: FrameRecord) -> None:
    fh.write(record.model_dump_json() + "\n")
