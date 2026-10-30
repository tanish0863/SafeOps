"""OpenCV overlay. Green = person box, blue = head region, yellow = face keypoints, red = UNCERTAIN."""
from __future__ import annotations

import cv2

from .schemas import FrameRecord

GREEN, BLUE, YELLOW, RED = (0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)  # BGR
FACE = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def draw(frame, record: FrameRecord):
    out = frame.copy()
    for p in record.persons:
        x1, y1, x2, y2 = map(int, p.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)
        if p.head_region:
            hx1, hy1, hx2, hy2 = map(int, p.head_region)
            cv2.rectangle(out, (hx1, hy1), (hx2, hy2), BLUE, 2)
        for name in FACE:
            kp = p.keypoints.get(name)
            if kp:
                cv2.circle(out, (int(kp.x), int(kp.y)), 3, YELLOW, -1)
        color = RED if p.head_visibility == "UNCERTAIN" else GREEN
        label = f"{p.detection_id} {p.confidence:.2f} {p.head_visibility}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out
