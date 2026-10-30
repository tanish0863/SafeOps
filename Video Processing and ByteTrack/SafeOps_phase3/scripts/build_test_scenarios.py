"""Build the 6 Required Validation Test Scenarios with Ground-Truth Annotations.

Scenarios:
1. One worker moving
2. Two workers crossing paths
3. Two workers standing close together
4. Worker leaving and re-entering (within track buffer)
5. Temporary person detection failure (7 frame blackout/drop)
6. Partial occlusion (vertical industrial girder/pipe occluding 40% torso)

Uses real high-resolution video frames as the base so YOLOv8-pose detects realistic
human bodies and keypoints, and records exact ground-truth trajectories per frame.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from ultralytics import YOLO

OUT_DIR = Path("videos/scenarios")
OUT_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR = OUT_DIR / "ground_truth"
GT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_V2 = "D:/Projects/new/videos/v2.mp4"
SOURCE_V4 = "D:/Projects/new/videos/v4.mp4"


def get_clean_frame_and_worker(source_video: str, frame_idx: int, conf_thresh: float = 0.8):
    """Extract a frame and high-confidence worker patch and bounding box."""
    cap = cv2.VideoCapture(source_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {source_video}")

    model = YOLO("models/yolov8s-pose.pt")
    res = model.predict(frame, classes=[0], conf=conf_thresh, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        raise RuntimeError("No worker detected in source frame")

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    box = [float(v) for v in boxes[best_idx]]
    return frame, box


def save_scenario(name: str, frames: list[np.ndarray], gt_records: dict[int, list[dict]], fps: float = 30.0):
    video_path = OUT_DIR / f"{name}.mp4"
    gt_path = GT_DIR / f"{name}_gt.json"

    h, w, _ = frames[0].shape
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, indent=2)

    print(f"Created {name}: {len(frames)} frames -> {video_path} & {gt_path}")


def build_scenario_1_single_worker_moving():
    """Scenario 1: One worker moving smoothly across the camera view."""
    # From v4.mp4 frames 110 to 170 (Track 8 moving)
    cap = cv2.VideoCapture(SOURCE_V4)
    model = YOLO("models/yolov8s-pose.pt")
    frames = []
    gt = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 110)
    for f_idx in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[0], conf=0.4, verbose=False)[0]
        # In this segment, the primary worker is detected with high confidence
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            # Select the main walking worker
            best_box = [float(v) for v in boxes[0]]
            gt[f_idx] = [{"id": 1, "bbox": best_box}]
        else:
            gt[f_idx] = []
        frames.append(frame)
    cap.release()
    save_scenario("scenario1_single_worker_moving", frames, gt)


def build_scenario_2_two_workers_crossing():
    """Scenario 2: Two workers walking towards and crossing paths."""
    cap = cv2.VideoCapture(SOURCE_V2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    ok, base_frame = cap.read()
    cap.release()

    model = YOLO("models/yolov8s-pose.pt")
    res = model.predict(base_frame, classes=[0], conf=0.7, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    assert len(boxes) >= 2, "Need 2 workers in base frame"

    b1 = [int(v) for v in boxes[0]]
    b2 = [int(v) for v in boxes[1]]

    # Scale down crops slightly (0.7x) so they fit comfortably on 854x480 during crossing
    crop1_orig = base_frame[b1[1]:b1[3], b1[0]:b1[2]]
    crop2_orig = base_frame[b2[1]:b2[3], b2[0]:b2[2]]

    w1_w, w1_h = int((b1[2] - b1[0]) * 0.7), int((b1[3] - b1[1]) * 0.7)
    w2_w, w2_h = int((b2[2] - b2[0]) * 0.7), int((b2[3] - b2[1]) * 0.7)

    crop1 = cv2.resize(crop1_orig, (w1_w, w1_h))
    crop2 = cv2.resize(crop2_orig, (w2_w, w2_h))

    bg = cv2.GaussianBlur(base_frame, (21, 21), 0)

    frames = []
    gt = {}
    total_frames = 60
    fh, fw, _ = base_frame.shape

    # Worker 1 moves left-to-right from x=10 to x=fw - w1_w - 10
    # Worker 2 moves right-to-left from x=fw - w2_w - 10 to x=10
    # They cross at frame 30 (center)
    x1_start, x1_end = 10, fw - w1_w - 10
    x2_start, x2_end = fw - w2_w - 10, 10
    y_pos = 100

    for f_idx in range(total_frames):
        f = bg.copy()
        alpha = f_idx / (total_frames - 1)
        x1 = int(x1_start + alpha * (x1_end - x1_start))
        x2 = int(x2_start + alpha * (x2_end - x2_start))

        # Clamp strictly
        x1 = max(0, min(fw - w1_w, x1))
        x2 = max(0, min(fw - w2_w, x2))

        # Worker 1
        f[y_pos:y_pos + w1_h, x1:x1 + w1_w] = crop1
        # Worker 2
        f[y_pos:y_pos + w2_h, x2:x2 + w2_w] = crop2

        gt[f_idx] = [
            {"id": 1, "bbox": [float(x1), float(y_pos), float(x1 + w1_w), float(y_pos + w1_h)]},
            {"id": 2, "bbox": [float(x2), float(y_pos), float(x2 + w2_w), float(y_pos + w2_h)]},
        ]
        frames.append(f)

    save_scenario("scenario2_two_workers_crossing", frames, gt)


def build_scenario_3_two_workers_close_together():
    """Scenario 3: Two workers standing close together."""
    cap = cv2.VideoCapture(SOURCE_V2)
    model = YOLO("models/yolov8s-pose.pt")
    frames = []
    gt = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[0], conf=0.5, verbose=False)[0]
        gt_list = []
        if res.boxes is not None and len(res.boxes) >= 2:
            boxes = res.boxes.xyxy.cpu().numpy()
            # Sort by x1: left worker = ID 2, right worker = ID 1
            sorted_boxes = sorted(boxes[:2], key=lambda b: b[0])
            gt_list = [
                {"id": 2, "bbox": [float(v) for v in sorted_boxes[0]]},
                {"id": 1, "bbox": [float(v) for v in sorted_boxes[1]]},
            ]
        gt[f_idx] = gt_list
        frames.append(frame)
    cap.release()
    save_scenario("scenario3_two_workers_close_together", frames, gt)


def build_scenario_4_worker_leaving_and_reentering():
    """Scenario 4: Worker leaving scene and re-entering within track buffer (30 frames gap)."""
    cap = cv2.VideoCapture(SOURCE_V2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, base_frame = cap.read()
    cap.release()

    model = YOLO("models/yolov8s-pose.pt")
    res = model.predict(base_frame, classes=[0], conf=0.7, verbose=False)[0]
    b1 = [int(v) for v in res.boxes.xyxy.cpu().numpy()[0]]

    # Scale 0.7x for comfortable boundaries
    crop_orig = base_frame[b1[1]:b1[3], b1[0]:b1[2]]
    cw, ch = int((b1[2] - b1[0]) * 0.7), int((b1[3] - b1[1]) * 0.7)
    crop = cv2.resize(crop_orig, (cw, ch))
    bg = cv2.GaussianBlur(base_frame, (21, 21), 0)

    frames = []
    gt = {}
    fh, fw, _ = base_frame.shape
    y_pos = 100

    # Phase 1: frames 0..24 (visible, moves right toward exit)
    # Phase 2: frames 25..54 (absent, 30 frames gap < 60 buffer)
    # Phase 3: frames 55..79 (re-enters from right side)
    for f_idx in range(80):
        f = bg.copy()
        if f_idx < 25:
            x = int(100 + f_idx * 18)
            x = max(0, min(fw - cw, x))
            f[y_pos:y_pos + ch, x:x + cw] = crop
            gt[f_idx] = [{"id": 1, "bbox": [float(x), float(y_pos), float(x + cw), float(y_pos + ch)]}]
        elif 25 <= f_idx < 55:
            gt[f_idx] = []
        else:
            re_idx = f_idx - 55
            x = int((fw - cw - 10) - re_idx * 14)
            x = max(0, min(fw - cw, x))
            f[y_pos:y_pos + ch, x:x + cw] = crop
            gt[f_idx] = [{"id": 1, "bbox": [float(x), float(y_pos), float(x + cw), float(y_pos + ch)]}]
        frames.append(f)

    save_scenario("scenario4_worker_leaving_reentering", frames, gt)


def build_scenario_5_temp_detection_failure():
    """Scenario 5: Temporary person detection failure (7 frames drop due to sudden darkness/blur)."""
    cap = cv2.VideoCapture(SOURCE_V2)
    model = YOLO("models/yolov8s-pose.pt")
    frames = []
    gt = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[0], conf=0.7, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        b1 = [float(v) for v in boxes[0]]

        if 25 <= f_idx < 32:
            # 7 frames of temporary detection drop: blackout / extreme blur on the worker
            x1, y1, x2, y2 = map(int, b1)
            frame[y1:y2, x1:x2] = 0  # Blackout causes detector to fail for these 7 frames
            # In GT, the worker is present at this location
            gt[f_idx] = [{"id": 1, "bbox": b1}]
        else:
            gt[f_idx] = [{"id": 1, "bbox": b1}]
        frames.append(frame)
    cap.release()
    save_scenario("scenario5_temp_detection_failure", frames, gt)


def build_scenario_6_partial_occlusion():
    """Scenario 6: Partial occlusion (vertical beam/pillar occluding 40% of torso/legs)."""
    cap = cv2.VideoCapture(SOURCE_V2)
    model = YOLO("models/yolov8s-pose.pt")
    frames = []
    gt = {}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for f_idx in range(60):
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, classes=[0], conf=0.7, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        b1 = [float(v) for v in boxes[0]]

        # Vertical occluder pillar over worker
        if 15 <= f_idx < 45:
            x1, y1, x2, y2 = map(int, b1)
            # Occlude central 40% horizontally
            occ_x1 = int(x1 + 0.3 * (x2 - x1))
            occ_x2 = int(x1 + 0.7 * (x2 - x1))
            cv2.rectangle(frame, (occ_x1, y1 + int(0.2 * (y2 - y1))), (occ_x2, y2), (70, 70, 70), -1)

        gt[f_idx] = [{"id": 1, "bbox": b1}]
        frames.append(frame)
    cap.release()
    save_scenario("scenario6_partial_occlusion", frames, gt)


if __name__ == "__main__":
    print("Building Scenario 1...")
    build_scenario_1_single_worker_moving()
    print("Building Scenario 2...")
    build_scenario_2_two_workers_crossing()
    print("Building Scenario 3...")
    build_scenario_3_two_workers_close_together()
    print("Building Scenario 4...")
    build_scenario_4_worker_leaving_and_reentering()
    print("Building Scenario 5...")
    build_scenario_5_temp_detection_failure()
    print("Building Scenario 6...")
    build_scenario_6_partial_occlusion()
    print("All 6 validation scenarios created successfully.")
