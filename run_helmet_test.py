#!/usr/bin/env python3
"""
run_helmet_test.py - Run helmet detection on v2.mp4 and v3.png
Execute: python run_helmet_test.py
"""

import cv2
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add workspace to path
root = Path("d:/Projects/Hackive (angadi)/vision")
sys.path.insert(0, str(root))

from detection import PersonDetector
from ppe_detection import PPEDetector
from tracking import PersonTracker
import config


def _format_ppe_status(track) -> str:
    if track.has_helmet is True:
        return "✓ WITH HELMET"
    if track.has_helmet is False:
        return "✗ NO HELMET"
    return "? UNKNOWN"

def test_image_helmet():
    """Test helmet detection on v3.png"""
    logger.info("="*80)
    logger.info("IMAGE TEST: v3.png")
    logger.info("="*80)
    
    img_path = root / "videos" / "v3.png"
    if not img_path.exists():
        logger.error(f"Image not found: {img_path}")
        return
    
    # Load models
    logger.info("Loading models...")
    person_detector = PersonDetector(
        model_path=config.PERSON_MODEL_PATH,
        conf_threshold=config.PERSON_CONF_THRESHOLD,
        iou_threshold=config.PERSON_IOU_THRESHOLD,
    )
    ppe_detector = PPEDetector()
    tracker = PersonTracker()

    if ppe_detector.is_stub:
        logger.warning("PPE detector is in STUB mode; results are not authoritative.")
    else:
        logger.info(f"Using PPE model: {ppe_detector.model_path}")
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        logger.error("Failed to read image")
        return
    
    h, w = img.shape[:2]
    logger.info(f"Image shape: {w}x{h}")
    
    # Run pipeline
    logger.info("Detecting persons...")
    detections = person_detector.detect(img)
    logger.info(f"  → Detected {len(detections)} persons")
    
    logger.info("Tracking...")
    tracks = tracker.update(detections, img)
    logger.info(f"  → Tracked {len(tracks)} persons")
    
    logger.info("Running helmet detection...")
    ppe_detector.process_tracks(tracks, img)
    
    # Results
    logger.info("\nRESULTS:")
    logger.info("-" * 80)
    if len(tracks) == 0:
        logger.info("No persons detected")
    else:
        with_helmet = 0
        without_helmet = 0
        unknown = 0
        for track in tracks:
            status = _format_ppe_status(track)
            conf = track.ppe_confidence
            logger.info(f"  Track {track.track_id}: {status} (conf={conf:.3f})")
            if track.has_helmet is True:
                with_helmet += 1
            elif track.has_helmet is False:
                without_helmet += 1
            else:
                unknown += 1
        logger.info("-" * 80)
        logger.info(f"Summary: {with_helmet} with helmet, {without_helmet} WITHOUT helmet, {unknown} UNKNOWN")
    logger.info("")


def test_video_helmet():
    """Test helmet detection on v2.mp4"""
    logger.info("="*80)
    logger.info("VIDEO TEST: v2.mp4")
    logger.info("="*80)
    
    vid_path = root / "videos" / "v2.mp4"
    if not vid_path.exists():
        logger.error(f"Video not found: {vid_path}")
        return
    
    # Load models
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
    
    # Open video
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        logger.error("Failed to open video")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video: {total_frames} frames @ {fps:.2f} fps")
    
    # Sample every Nth frame for speed
    sample_every = 10
    frame_idx = 0
    samples_processed = 0
    
    helmet_stats = {}  # track_id -> {"with": count, "without": count}
    unknown_stats = {}
    frame_results = []
    
    logger.info(f"Processing (sampling every {sample_every} frames)...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Sample
        if frame_idx % sample_every != 0:
            continue
        
        samples_processed += 1
        
        # Pipeline
        detections = person_detector.detect(frame)
        tracks = tracker.update(detections, frame)
        ppe_detector.process_tracks(tracks, frame)
        
        # Collect stats
        frame_result = {
            'frame': frame_idx,
            'persons': len(tracks),
            'with_helmet': 0,
            'without_helmet': 0,
            'unknown': 0,
            'tracks': []
        }
        
        for track in tracks:
            if track.has_helmet is True:
                frame_result['with_helmet'] += 1
            elif track.has_helmet is False:
                frame_result['without_helmet'] += 1
            else:
                frame_result['unknown'] += 1
            
            frame_result['tracks'].append({
                'id': track.track_id,
                'has_helmet': track.has_helmet,
                'confidence': track.ppe_confidence
            })
            
            # Track running stats
            if track.track_id not in helmet_stats:
                helmet_stats[track.track_id] = {"with": 0, "without": 0}
                unknown_stats[track.track_id] = 0
            if track.has_helmet is True:
                helmet_stats[track.track_id]["with"] += 1
            elif track.has_helmet is False:
                helmet_stats[track.track_id]["without"] += 1
            else:
                unknown_stats[track.track_id] += 1
        
        frame_results.append(frame_result)
        
        if samples_processed % 10 == 0:
            logger.info(f"  {samples_processed} samples processed (frame {frame_idx})...")
    
    cap.release()
    
    # Print results
    logger.info("\nRESULTS:")
    logger.info("-" * 80)
    logger.info("Frame  | Persons | With Helmet | Without Helmet | Unknown | Compliance")
    logger.info("-" * 80)
    
    total_persons = 0
    total_with_helmet = 0
    total_without_helmet = 0
    
    for r in frame_results:
        frame = r['frame']
        persons = r['persons']
        with_h = r['with_helmet']
        without_h = r['without_helmet']
        unknown = r['unknown']
        
        total_persons += persons
        total_with_helmet += with_h
        total_without_helmet += without_h
        
        compliance = 100 * with_h / persons if persons > 0 else 0
        logger.info(f" {frame:5d} | {persons:7d} | {with_h:11d} | {without_h:14d} | {unknown:7d} | {compliance:6.1f}%")
    
    logger.info("-" * 80)
    logger.info(f"TOTAL: {total_persons} persons, {total_with_helmet} with helmet, {total_without_helmet} WITHOUT")
    
    if total_persons > 0:
        overall = 100 * total_with_helmet / total_persons
        logger.info(f"Overall compliance: {overall:.1f}%")
    
    logger.info("\nPer-Track Summary:")
    logger.info("-" * 80)
    for tid in sorted(helmet_stats.keys()):
        stats = helmet_stats[tid]
        with_h = stats["with"]
        without_h = stats["without"]
        unknown = unknown_stats.get(tid, 0)
        total = with_h + without_h
        compliance = 100 * with_h / total if total > 0 else 0
        status = "✓ COMPLIANT" if without_h == 0 and unknown == 0 else ("? UNKNOWN" if unknown > 0 and without_h == 0 else "✗ VIOLATION")
        logger.info(f"  Track {tid:3d}: {with_h:2d} with helmet, {without_h:2d} WITHOUT, {unknown:2d} UNKNOWN → {status} ({compliance:.0f}%)")
    
    logger.info("-" * 80)
    logger.info("")


if __name__ == "__main__":
    try:
        test_image_helmet()
        test_video_helmet()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
