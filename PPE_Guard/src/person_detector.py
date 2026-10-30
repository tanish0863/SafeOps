"""Component A - person detection (bounding boxes + confidence)."""
from __future__ import annotations

from typing import List, Optional

from .inference import PoseEngine, RawInference
from .schemas import Detection


class PersonDetector:
    """Interface over the shared pose inference. Pass an existing engine to avoid a second model load."""

    def __init__(self, engine: Optional[PoseEngine] = None, **engine_kwargs):
        self.engine = engine or PoseEngine(**engine_kwargs)

    def detect(self, frame) -> List[Detection]:
        h, w = frame.shape[:2]
        return self.from_raw(self.engine.infer(frame), w, h)

    @staticmethod
    def from_raw(raw: RawInference, frame_width: int, frame_height: int) -> List[Detection]:
        return [
            Detection(
                detection_id=f"det_{i:04d}",
                bbox=[round(float(v), 1) for v in raw.boxes[i]],
                confidence=round(float(raw.scores[i]), 4),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            for i in range(len(raw.boxes))
        ]
