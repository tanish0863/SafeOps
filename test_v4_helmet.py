#!/usr/bin/env python3
"""
test_v4_helmet.py - Run helmet detection on v4.mp4
Mirrors the exact testing structure of test_v5_helmet.py and test_vi3_helmet.py.
Execute: python test_v4_helmet.py
"""

import cv2
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from detection import PersonDetector
from ppe_detection import PPEDetector
from tracking import PersonTracker
import config


def test_v4_helmet() -> int:
    logger.info("=" * 80)
    logger.info("VIDEO TEST: v4.mp4")
    logger.info("=" * 80)

    vid_path = root / "videos" / "v4.mp4"
    if not vid_path.exists():
        logger.error(f"Video not found: {vid_path}")
        return 1

    logger.info("Loading models...")
    person_detector = PersonDetector(
        model_path=str(root / "yolov8n.pt"),
        conf_threshold=config.PERSON_CONF_THRESHOLD,
        iou_threshold=config.PERSON_IOU_THRESHOLD,
    )
    ppe_detector = PPEDetector()
    tracker = PersonTracker()

    if ppe_detector.is_stub:
        logger.warning("PPE detector is in STUB mode; results are not authoritative.")
    else:
        logger.info(f"Using PPE model: {ppe_detector.model_path}")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        logger.error("Failed to open video")
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} fps")

    sample_every = 10
    frame_idx = 0
    samples_processed = 0
    helmet_stats = {}
    frame_results = []

    logger.info(f"Processing (sampling every {sample_every} frames)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        samples_processed += 1

        detections = person_detector.detect(frame)
        tracks = tracker.update(detections, frame)
        ppe_detector.process_tracks(tracks, frame)

        frame_result = {
            "frame": frame_idx,
            "persons": len(tracks),
            "with_helmet": 0,
            "without_helmet": 0,
            "unknown": 0,
        }

        for track in tracks:
            if track.has_helmet is True:
                frame_result["with_helmet"] += 1
            elif track.has_helmet is False:
                frame_result["without_helmet"] += 1
            else:
                frame_result["unknown"] += 1

            if track.track_id not in helmet_stats:
                helmet_stats[track.track_id] = {"with": 0, "without": 0}
            if track.has_helmet is True:
                helmet_stats[track.track_id]["with"] += 1
            elif track.has_helmet is False:
                helmet_stats[track.track_id]["without"] += 1

        frame_results.append(frame_result)

        if samples_processed % 20 == 0:
            logger.info(f"  {samples_processed} samples processed (frame {frame_idx})...")

    cap.release()

    total_persons = sum(r["persons"] for r in frame_results)
    total_with_helmet = sum(r["with_helmet"] for r in frame_results)
    total_without_helmet = sum(r["without_helmet"] for r in frame_results)
    total_unknown = sum(r["unknown"] for r in frame_results)

    logger.info("-" * 80)
    logger.info("SUMMARY")
    logger.info("-" * 80)
    logger.info(f"Samples processed: {samples_processed}")
    logger.info(f"Total person observations: {total_persons}")
    logger.info(f"With helmet: {total_with_helmet}")
    logger.info(f"Without helmet: {total_without_helmet}")
    logger.info(f"Unknown: {total_unknown}")

    known = total_with_helmet + total_without_helmet
    if known > 0:
        compliance = 100.0 * total_with_helmet / known
        logger.info(f"Compliance over known labels: {compliance:.2f}%")

    if total_persons > 0:
        unknown_rate = 100.0 * total_unknown / total_persons
        logger.info(f"Unknown rate: {unknown_rate:.2f}%")

    logger.info("\nPer-Track Summary:")
    logger.info("-" * 80)
    for tid in sorted(helmet_stats.keys()):
        stats = helmet_stats[tid]
        with_h = stats["with"]
        without_h = stats["without"]
        total = with_h + without_h
        compliance = 100 * with_h / total if total > 0 else 0
        status = "✓ COMPLIANT" if without_h == 0 else "✗ VIOLATION"
        logger.info(f"  Track {tid:3d}: {with_h:2d} with helmet, {without_h:2d} WITHOUT → {status} ({compliance:.0f}%)")

    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(test_v4_helmet())
