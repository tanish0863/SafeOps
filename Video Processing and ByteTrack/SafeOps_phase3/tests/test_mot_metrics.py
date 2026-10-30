"""Unit tests for the MOT metrics calculator."""
import pytest
from src.evaluation.mot_metrics import compute_iou, evaluate_tracking


def test_compute_iou():
    b1 = [0, 0, 10, 10]
    b2 = [0, 0, 10, 10]
    assert compute_iou(b1, b2) == 1.0

    b3 = [5, 0, 15, 10]  # overlap 5x10 = 50, area1=100, area2=100, union=150 -> 50/150 = 1/3
    assert abs(compute_iou(b1, b3) - (1.0 / 3.0)) < 1e-4

    b4 = [20, 20, 30, 30]  # disjoint
    assert compute_iou(b1, b4) == 0.0


def test_compute_iou_identical_boxes():
    box = [50.0, 50.0, 150.0, 250.0]
    assert compute_iou(box, box) == 1.0


def test_compute_iou_normal_overlapping_boxes():
    box1 = [0.0, 0.0, 100.0, 100.0]  # area = 10000
    box2 = [50.0, 0.0, 150.0, 100.0]  # area = 10000, inter = 50*100=5000, union = 15000
    assert abs(compute_iou(box1, box2) - (1.0 / 3.0)) < 1e-5


def test_compute_iou_zero_area_boxes():
    # Zero width
    box_zero_w = [10.0, 10.0, 10.0, 50.0]
    box_normal = [0.0, 0.0, 50.0, 50.0]
    assert compute_iou(box_zero_w, box_normal) == 0.0
    assert compute_iou(box_normal, box_zero_w) == 0.0

    # Zero height
    box_zero_h = [10.0, 10.0, 50.0, 10.0]
    assert compute_iou(box_zero_h, box_normal) == 0.0
    assert compute_iou(box_zero_h, box_zero_h) == 0.0


def test_compute_iou_inverted_coordinates():
    # x2 < x1 or y2 < y1
    box_inverted_x = [100.0, 10.0, 10.0, 50.0]
    box_normal = [0.0, 0.0, 50.0, 50.0]
    assert compute_iou(box_inverted_x, box_normal) == 0.0
    assert compute_iou(box_normal, box_inverted_x) == 0.0

    box_inverted_y = [10.0, 100.0, 50.0, 10.0]
    assert compute_iou(box_inverted_y, box_normal) == 0.0


def test_compute_iou_non_overlapping_boxes():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [100.0, 100.0, 110.0, 110.0]
    assert compute_iou(box1, box2) == 0.0


def test_perfect_tracking():
    # 2 GT trajectories across 10 frames, tracker perfectly matches both with constant IDs
    gt_frames = {
        f: [
            {"id": 1, "bbox": [10.0 + f, 20.0, 50.0 + f, 100.0]},
            {"id": 2, "bbox": [100.0, 20.0, 150.0, 100.0]},
        ]
        for f in range(10)
    }
    tracker_frames = {
        f: [
            {"track_id": 101, "bbox_xyxy": [10.0 + f, 20.0, 50.0 + f, 100.0]},
            {"track_id": 102, "bbox_xyxy": [100.0, 20.0, 150.0, 100.0]},
        ]
        for f in range(10)
    }

    metrics = evaluate_tracking(gt_frames, tracker_frames)
    assert metrics["id_switches"] == 0
    assert metrics["fragmentation"] == 0
    assert metrics["idf1"] == 1.0
    assert metrics["hota"] == 1.0
    assert metrics["mostly_tracked"] == 2
    assert metrics["mostly_lost"] == 0


def test_id_switch_detection():
    # GT 1 is matched to tracker 101 in frames 0..4, and tracker 102 in frames 5..9
    gt_frames = {
        f: [{"id": 1, "bbox": [10.0, 20.0, 50.0, 100.0]}]
        for f in range(10)
    }
    tracker_frames = {
        f: [{"track_id": 101 if f < 5 else 102, "bbox_xyxy": [10.0, 20.0, 50.0, 100.0]}]
        for f in range(10)
    }

    metrics = evaluate_tracking(gt_frames, tracker_frames)
    assert metrics["id_switches"] == 1
    assert metrics["unique_track_ids"] == 2
    assert metrics["hota"] < 1.0
    assert metrics["idf1"] < 1.0

