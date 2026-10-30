"""
test_helmet_video.py — Test helmet detection on video file

Processes the video and outputs detailed helmet detection metrics.
"""

import cv2
import sys
from pathlib import Path
from detection import PersonDetector
from ppe_detection import PPEDetector
from tracking import PersonTracker
from zone_logic import ZoneChecker
from alert_manager import AlertManager
from visualization import compose_frame
from config import (
    FRAME_SKIP, ALERT_COOLDOWN_FRAMES, VIOLATION_CONFIRM_FRAMES,
    PERSON_CONF_THRESHOLD, PPE_CONF_THRESHOLD
)


def test_helmet_detection(video_path: str):
    """Process video and test helmet detection."""
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return
    
    print(f"[INFO] Processing video: {video_path.name}")
        print(f"[INFO] Person confidence: {PERSON_CONF_THRESHOLD}, PPE threshold: {PPE_CONF_THRESHOLD}")
    print("=" * 80)
    
    # Initialize models
    print("[INFO] Initializing models...")
        person_detector = PersonDetector(min_confidence=PERSON_CONF_THRESHOLD)
        ppe_detector = PPEDetector(confidence_threshold=PPE_CONF_THRESHOLD)
    tracker = PersonTracker()
    zone_checker = ZoneChecker()
    alert_manager = AlertManager(
        confirmation_frames=VIOLATION_CONFIRM_FRAMES,
        cooldown_frames=ALERT_COOLDOWN_FRAMES
    )
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video properties:")
    print(f"       FPS: {fps}, Total frames: {total_frames}")
    print(f"       Resolution: {frame_width}x{frame_height}")
    print("=" * 80)
    print()
    
    # Statistics
    frame_count = 0
    person_frames = 0
    helmet_detections = []
    no_helmet_frames = []
    helmet_violations = []
    
    print("HELMET DETECTION RESULTS")
    print("-" * 80)
    print("Frame | Persons | Helmets | No Helmet | Violation | Details")
    print("-" * 80)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames
            if frame_count % FRAME_SKIP != 0:
                continue
            
            # Detect persons
            detections = person_detector.detect(frame)
            
            # Track persons
            tracked_persons = tracker.update(frame, detections)
            
            # Check zones and PPE
            for person in tracked_persons:
                zone_checker.check_violations(person, frame.shape)
                ppe_detector.detect_ppe(person, frame)
            
                # Count helmets
                if person.has_no_helmet:
                    if frame_count not in no_helmet_frames:
                        no_helmet_frames.append(frame_count)
                        helmet_violations.append({
                            "frame": frame_count,
                            "person_id": person.id,
                            "status": "NO HELMET",
                            "confidence": person.ppe_confidence
                        })
                else:
                    if frame_count not in helmet_detections:
                        helmet_detections.append({
                            "frame": frame_count,
                            "person_id": person.id,
                            "status": "HELMET ON",
                            "confidence": person.ppe_confidence
                        })
            
            # Alert check
            violations = alert_manager.update(tracked_persons, frame_count)
            
            # Print this frame
            if len(tracked_persons) > 0:
                person_frames += 1
                helmet_on = sum(1 for p in tracked_persons if not p.has_no_helmet)
                no_helmet = sum(1 for p in tracked_persons if p.has_no_helmet)
                has_violation = len(violations) > 0
                
                violation_str = "ALERT" if has_violation else "OK"
                details = f"People: {len(tracked_persons)}"
                
                print(f"{frame_count:5d} | {len(tracked_persons):7d} | {helmet_on:7d} | {no_helmet:9d} | {violation_str:9s} | {details}")
            
            # Progress
            if frame_count % 100 == 0:
                print(f"  [Progress] Processed {frame_count}/{total_frames} frames...")
    
    finally:
        cap.release()
    
    print("-" * 80)
    print()
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total frames processed:     {frame_count}")
    print(f"Frames with people:         {person_frames}")
    print(f"Frames with helmet on:      {len(helmet_detections)}")
    print(f"Frames with NO helmet:      {len(no_helmet_frames)}")
    print(f"PPE violations detected:    {len(helmet_violations)}")
    
    if person_frames > 0:
        helmet_on_rate = (len(helmet_detections) / person_frames) * 100
        print(f"\nHelmet On Rate:             {helmet_on_rate:.1f}%")
        print(f"Without Helmet Rate:        {100 - helmet_on_rate:.1f}%")
    
    print()
    print("=" * 80)
    print("HELMET VIOLATION DETAILS")
    print("=" * 80)
    
    if helmet_violations:
        print(f"\nNo Helmet Detected on {len(helmet_violations)} occasions:\n")
        for i, v in enumerate(helmet_violations[:20], 1):  # Show first 20
            print(f"  {i:2d}. Frame {v['frame']:4d} | Person ID: {v['person_id']} | Confidence: {v['confidence']:.3f}")
        
        if len(helmet_violations) > 20:
            print(f"  ... and {len(helmet_violations) - 20} more")
    else:
        print("\n✓ No helmet violations detected!")
    
    print()
    print("=" * 80)
    print("HELMET DETECTION SUCCESS RATE")
    print("=" * 80)
    
    positive_frames = len(helmet_detections)
    negative_frames = len(no_helmet_frames)
    total_detections = positive_frames + negative_frames
    
    if total_detections > 0:
        success_rate = (positive_frames / total_detections) * 100
        print(f"\nPersons WITH helmet:        {positive_frames} frames ({success_rate:.1f}%)")
        print(f"Persons WITHOUT helmet:     {negative_frames} frames ({100-success_rate:.1f}%)")
        print(f"Total helmet detections:    {total_detections}")
    
    # Save predictions to CSV for evaluation
    save_predictions_csv(helmet_detections, no_helmet_frames, total_frames, fps)
    
    print()
    print("[INFO] Test complete! Predictions saved to test_predictions.csv")


def save_predictions_csv(helmet_frames, no_helmet_frames, total_frames, fps):
    """Save predictions in evaluation format."""
    
    with open("test_predictions.csv", "w") as f:
        f.write("frame_id,timestamp_seconds,person_present,helmet_visible,in_restricted_zone\n")
        
        for frame_id in range(1, total_frames + 1):
            timestamp = frame_id / fps if fps > 0 else 0
            
            if frame_id in [h['frame'] for h in helmet_frames]:
                person = "Yes"
                helmet = "Yes"
                zone = "No"  # Not tracking zone for this test
            elif frame_id in no_helmet_frames:
                person = "Yes"
                helmet = "No"
                zone = "No"
            else:
                person = "No"
                helmet = "N/A"
                zone = "N/A"
            
            f.write(f"{frame_id},{timestamp:.2f},{person},{helmet},{zone}\n")
    
    print(f"[INFO] Saved test_predictions.csv with {total_frames} frames")


if __name__ == "__main__":
    video_file = "d:\\Projects\\Hackive (angadi)\\vision\\videos\\for helmet v1.mp4"
    test_helmet_detection(video_file)
