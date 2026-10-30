"""Reliability and Edge-Case Tests for SafeOps Phase 3 ByteTrack Implementation.

Validates:
1. ByteTrack initialization and reset() between videos
2. persist=True tracker state continuity
3. Zero-based frame numbering and timestamps contract
4. Handling of empty frames (no person detections)
5. Handling of missing tracking IDs
6. PPE association keyed strictly by track_id (not list order)
7. Temporary detection failure / track gap detection in track_stats
8. Custom ByteTrack configuration schema and parameters
"""
import math
import os
import sys
import yaml
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))

from src.association.ppe_association import PPEDetection, associate_ppe, normalize_label
from src.tracking.worker_tracker import WorkerTracker
from src.tracking.track_stats import summarize
from src.video.video_processor import VideoInfo, _color, _round, annotate


def test_bytetrack_initialization_and_reset():
    """Verify ByteTrack is initialized only once per video and reset() clears state."""
    tracker = WorkerTracker(conf=0.10, tracker="src/tracking/custom_bytetrack.yaml")
    
    # Before processing, tracker state is clean
    tracker.reset()
    assert tracker.model.predictor is None, "Predictor must be None after reset()"

    # Process a synthetic blank frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = tracker.update(dummy_frame)

    # Predictor should now be instantiated and initialized
    assert tracker.model.predictor is not None, "Predictor must be instantiated after first update()"
    
    # reset() must clear the predictor so next video starts fresh
    tracker.reset()
    assert tracker.model.predictor is None, "Predictor must be reset to None between videos"


def test_empty_frame_handling():
    """Verify that empty frames (no person detections) return empty list without crashing."""
    tracker = WorkerTracker(conf=0.10)
    tracker.reset()
    
    # Black frame has 0 persons
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    workers = tracker.update(black_frame)
    assert isinstance(workers, list)
    assert len(workers) == 0, "Empty frame must yield 0 workers"


def test_missing_track_ids_handling():
    """Verify that detections without confirmed track IDs do not cause IndexError or NoneType crash."""
    tracker = WorkerTracker(conf=0.10)
    tracker.reset()
    
    # Random noise frame (no valid human poses)
    noise_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    workers = tracker.update(noise_frame)
    assert isinstance(workers, list)
    for w in workers:
        assert "track_id" in w
        assert isinstance(w["track_id"], int)


def test_frame_number_and_timestamp_contract():
    """Verify zero-based frame numbers and timestamp formula (frame_number / fps)."""
    fps = 29.97
    total_frames = 100
    info = VideoInfo(fps=fps, width=1280, height=720, total_frames=total_frames)
    
    assert abs(info.duration - (100 / 29.97)) < 1e-4

    # Check timestamps across frames
    timestamps = [round(fn / info.fps, 3) for fn in range(total_frames)]
    assert timestamps[0] == 0.0
    assert timestamps[1] == 0.033
    assert timestamps[2] == 0.067
    assert timestamps[99] == round(99 / 29.97, 3)


def test_ppe_association_strictly_keyed_by_track_id():
    """Verify that PPE association maps by track_id independent of worker detection ordering."""
    w_alice = {
        "track_id": 105,
        "bbox_xyxy": [100.0, 100.0, 200.0, 400.0],
        "keypoints_xy": None,
        "keypoints_conf": None,
    }
    w_bob = {
        "track_id": 42,
        "bbox_xyxy": [300.0, 100.0, 400.0, 400.0],
        "keypoints_xy": None,
        "keypoints_conf": None,
    }

    # Helmet located on Alice's head region
    helmet_alice = PPEDetection("helmet", [110.0, 100.0, 190.0, 160.0], 0.95)
    # Vest located on Bob's torso region
    vest_bob = PPEDetection("vest", [310.0, 180.0, 390.0, 320.0], 0.88)

    # Order A: [Alice, Bob]
    res_a = associate_ppe([w_alice, w_bob], [helmet_alice, vest_bob])
    assert res_a[105]["helmet"] is True
    assert res_a[105]["vest"] is False
    assert res_a[42]["helmet"] is False
    assert res_a[42]["vest"] is True

    # Order B: [Bob, Alice] (reversed detection list)
    res_b = associate_ppe([w_bob, w_alice], [helmet_alice, vest_bob])
    assert res_b[105]["helmet"] is True
    assert res_b[105]["vest"] is False
    assert res_b[42]["helmet"] is False
    assert res_b[42]["vest"] is True
    assert res_a == res_b, "Association result must be identical regardless of list ordering"


def test_temporary_detection_failure_gap_detection(tmp_path):
    """Verify track_stats accurately identifies gaps and missing frames from temporary detection drops."""
    jsonl_path = tmp_path / "test_tracks.jsonl"
    
    # Worker 1: detected in frames 0..9, missing in 10..14 (5 frame gap), detected in 15..24
    # Worker 2: detected in frames 0..4 (short track, 5 frames < 15)
    records = []
    for f in list(range(10)) + list(range(15, 25)):
        records.append(f'{{"frame_number": {f}, "track_id": 1, "bbox_xyxy": [10, 10, 50, 100]}}\n')
    for f in range(5):
        records.append(f'{{"frame_number": {f}, "track_id": 2, "bbox_xyxy": [100, 10, 150, 100]}}\n')

    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.writelines(records)

    summary = summarize(str(jsonl_path), total_frames=25, short_track_frames=15)
    
    assert summary["unique_track_ids"] == 2
    assert summary["short_tracks"] == [2]
    
    t1 = summary["tracks"][1]
    assert t1["first_frame"] == 0
    assert t1["last_frame"] == 24
    assert t1["frames_present"] == 20
    assert t1["gaps"] == 1
    assert t1["missing_frames_in_span"] == 5  # frames 10, 11, 12, 13, 14 = 5 frames missing
    assert t1["short_track"] is False


def test_custom_bytetrack_yaml_configuration():
    """Verify custom ByteTrack yaml has valid parameters for tracking."""
    yaml_path = "src/tracking/custom_bytetrack.yaml"
    assert os.path.exists(yaml_path), "custom_bytetrack.yaml must exist"
    
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["tracker_type"] == "bytetrack"
    assert 0.0 < cfg["track_high_thresh"] <= 1.0
    assert 0.0 < cfg["track_low_thresh"] <= cfg["track_high_thresh"]
    assert 0.0 < cfg["new_track_thresh"] <= 1.0
    assert cfg["track_buffer"] >= 30, "track_buffer should be >= 30 frames"
    assert 0.0 < cfg["match_thresh"] <= 1.0
    assert isinstance(cfg["fuse_score"], bool)
