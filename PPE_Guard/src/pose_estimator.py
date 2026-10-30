"""Component B - 17-keypoint COCO pose extraction with low-confidence rejection."""
from __future__ import annotations

import math
from typing import List, Optional

from .inference import PoseEngine, RawInference
from .schemas import KEYPOINT_NAMES, Keypoint, Pose


class PoseEstimator:
    def __init__(self, engine: Optional[PoseEngine] = None, keypoint_conf: float = 0.5,
                 **engine_kwargs):
        self.engine = engine or PoseEngine(**engine_kwargs)
        self.keypoint_conf = keypoint_conf

    def estimate(self, frame) -> List[Pose]:
        h, w = frame.shape[:2]
        return self.from_raw(self.engine.infer(frame), w, h)

    def from_raw(self, raw: RawInference, frame_width: int, frame_height: int) -> List[Pose]:
        poses = []
        for i in range(len(raw.keypoints)):
            kps = {}
            for name, (x, y, c) in zip(KEYPOINT_NAMES, raw.keypoints[i]):
                reliable = (
                    all(math.isfinite(float(v)) for v in (x, y, c))
                    and c >= self.keypoint_conf
                    and 0 <= x <= frame_width and 0 <= y <= frame_height
                )
                kps[name] = (Keypoint(x=round(float(x), 1), y=round(float(y), 1),
                                      confidence=round(float(c), 3)) if reliable else None)
            poses.append(Pose(detection_id=f"det_{i:04d}", keypoints=kps))
        return poses
