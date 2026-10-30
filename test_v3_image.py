#!/usr/bin/env python3
"""
test_v3_image.py — Live test runner for image v3.png.

Usage:
  python test_v3_image.py
  python test_v3_image.py --save-only
"""

from __future__ import annotations

import argparse
import sys
import cv2
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from main import SafetyPipeline
import visualization as vis


def run_live_test(display: bool = True) -> int:
    img_path = ROOT / "videos" / "v3.png"
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        return 1

    print("=" * 60)
    print("LIVE TEST: v3.png (Image)")
    print("=" * 60)

    # Initialize safety pipeline in accuracy mode
    pipeline = SafetyPipeline(source=str(img_path), display=False, headless=True, profile="accuracy")
    frame = cv2.imread(str(img_path))
    if frame is None:
        print("[ERROR] Failed to load image.")
        return 1

    if len(frame.shape) == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    print(f"Loaded image: {frame.shape[1]}x{frame.shape[0]}")
    output, tracks, alerts = pipeline.process_single_frame(frame, frame_num=1)

    print("\n--- DETECTION & COMPLIANCE SUMMARY ---")
    print(f"Total Persons Detected: {len(tracks)}")
    for track in tracks:
        status_label = "WITH HELMET (SAFE)" if track.has_helmet else ("NO HELMET (VIOLATION)" if track.has_helmet is False else "UNKNOWN")
        print(f"  • Track {track.track_id}: {status_label}")
        print(f"    - BBox: {track.bbox_xyxy.astype(int).tolist()}")
        print(f"    - Person Conf: {track.confidence:.2f}")
        print(f"    - Helmet/PPE Conf: {track.ppe_confidence:.2f}")

    print(f"\nActive Alerts Triggered: {len(alerts)}")
    for alert in alerts:
        print(f"  ⚠ {alert}")

    out_path = ROOT / "v3_live_test_result.png"
    cv2.imwrite(str(out_path), output)
    print(f"\n[OK] Annotated test result saved to: {out_path}")

    if display:
        print("\nOpening live OpenCV window (press any key or 'q' to close)...")
        win_name = "Live Test - v3.png"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        display_frame = vis.resize_for_display(output)
        cv2.imshow(win_name, display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run live safety test on v3.png")
    parser.add_argument("--save-only", action="store_true", help="Do not display GUI window")
    args = parser.parse_args()
    sys.exit(run_live_test(display=not args.save_only))

