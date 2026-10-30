#!/usr/bin/env python3
"""
scan_accuracy_frames.py - find frames that produce the most stable live tracks.
"""

from __future__ import annotations

import cv2
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "videos" / "v5.mp4"


def load_pipeline():
    spec = importlib.util.spec_from_file_location("_root_safety_main", ROOT / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load pipeline")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafetyPipeline(source="0", display=False, headless=True, profile="accuracy")


def main() -> int:
    pipe = load_pipeline()
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        print(f"Failed to open {VIDEO}")
        return 1

    best = (-1, -1, -1, -1)
    try:
        for idx in range(0, 180, 10):
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            _, tracks, alerts = pipe.process_single_frame(frame, idx)
            helmet = sum(1 for t in tracks if t.has_helmet is True)
            no_helmet = sum(1 for t in tracks if t.has_helmet is False)
            print(f"frame={idx:<3d} tracks={len(tracks):<2d} helmet={helmet:<2d} no_helmet={no_helmet:<2d} alerts={len(alerts)}")
            score = len(tracks) * 10 + helmet + no_helmet
            if score > best[0]:
                best = (score, idx, len(tracks), helmet + no_helmet)
    finally:
        capture.release()

    print(f"best_frame={best[1]} tracks={best[2]} known_ppe={best[3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
