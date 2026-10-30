#!/usr/bin/env python3
"""
calibrate_live_profile.py - quick live-profile probe on bundled videos.

This script is intentionally small: it compares a few person-detection presets
against local videos so we can choose accuracy-first defaults without guessing.
"""

from __future__ import annotations

import cv2
from pathlib import Path

from detection import PersonDetector


ROOT = Path(__file__).resolve().parent
VIDEOS = [
    ROOT / "videos" / "v4.mp4",
    ROOT / "videos" / "v5.mp4",
    ROOT / "videos" / "vi3.mp4",
]

PRESETS = [
    ("n_0.20_640", ROOT / "yolov8n.pt", 0.20, 640),
    ("n_0.15_768", ROOT / "yolov8n.pt", 0.15, 768),
    ("n_0.12_768", ROOT / "yolov8n.pt", 0.12, 768),
    ("n_0.10_896", ROOT / "yolov8n.pt", 0.10, 896),
    ("s_0.15_768", ROOT / "yolov8s.pt", 0.15, 768),
    ("s_0.12_768", ROOT / "yolov8s.pt", 0.12, 768),
    ("s_0.10_768", ROOT / "yolov8s.pt", 0.10, 768),
    ("s_0.12_896", ROOT / "yolov8s.pt", 0.12, 896),
    ("s_0.10_896", ROOT / "yolov8s.pt", 0.10, 896),
]


def sample_average_people(detector: PersonDetector, video_path: Path, sample_limit: int = 20) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return 0, 0.0

    seen = 0
    total_people = 0
    frame_index = 0

    try:
        while frame_index < sample_limit:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % 2 != 0:
                continue
            seen += 1
            total_people += len(detector.detect(frame))
    finally:
        capture.release()

    average = round(total_people / seen, 2) if seen else 0.0
    return seen, average


def main() -> int:
    for preset_name, model_path, conf, input_size in PRESETS:
        print(f"\nPRESET {preset_name}")
        detector = PersonDetector(
            model_path=str(model_path),
            conf_threshold=conf,
            input_size=input_size,
        )
        for video_path in VIDEOS:
            if not video_path.exists():
                continue
            seen, average = sample_average_people(detector, video_path)
            print(f"{video_path.name:<14} samples={seen:<2d} avg_people={average}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
