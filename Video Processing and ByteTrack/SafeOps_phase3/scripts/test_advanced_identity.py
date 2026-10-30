"""Test Advanced Identity Management Layer on all 6 scenarios."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import json
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from src.evaluation.mot_metrics import evaluate_tracking, compute_iou
from ultralytics import YOLO


@dataclass
class TrackHistory:
    track_id: int
    last_frame: int
    last_box: List[float]
    velocity: List[float] # [vx, vy]
    frames_seen: int
    appearance_hist: List[np.ndarray] = field(default_factory=list)
    state: str = "active" # active, lost, expired


class AdvancedIdentityTracker:
    def __init__(
        self,
        model_path: str = "models/yolov8s-pose.pt",
        tracker_yaml: str = "src/tracking/botsort_no_reid.yaml",
        conf: float = 0.10,
        iou: float = 0.7,
        imgsz: int = 640,
        max_lost_frames: int = 60,
    ):
        self.model = YOLO(model_path)
        self.tracker_yaml = tracker_yaml
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_lost_frames = max_lost_frames

        # Identity registry
        self.active_tracks: Dict[int, TrackHistory] = {}
        self.lost_tracks: Dict[int, TrackHistory] = {}
        self.raw_to_stable_id: Dict[int, int] = {}
        self.next_stable_id = 1
        self.current_frame = 0

    def reset(self):
        self.model.predictor = None
        self.active_tracks.clear()
        self.lost_tracks.clear()
        self.raw_to_stable_id.clear()
        self.next_stable_id = 1
        self.current_frame = 0

    def update(self, frame: np.ndarray, frame_number: int) -> List[dict]:
        self.current_frame = frame_number

        # Run underlying tracker
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_yaml,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            half=False,
            verbose=False,
        )
        res = results[0]
        if res.boxes is None or res.boxes.id is None:
            # Mark all active as lost
            for sid, th in list(self.active_tracks.items()):
                th.state = "lost"
                self.lost_tracks[sid] = th
            self.active_tracks.clear()
            return []

        raw_boxes = res.boxes.xyxy.cpu().numpy()
        raw_ids = res.boxes.id.int().cpu().numpy()
        raw_confs = res.boxes.conf.cpu().numpy()

        # Update lost tracks expiration
        for sid, th in list(self.lost_tracks.items()):
            if self.current_frame - th.last_frame > self.max_lost_frames:
                th.state = "expired"
                del self.lost_tracks[sid]

        # Map tracker outputs to stable IDs
        current_matched_stable = set()
        out_workers = []

        for i in range(len(raw_ids)):
            raw_id = int(raw_ids[i])
            box = [float(v) for v in raw_boxes[i]]
            conf_val = float(raw_confs[i])

            # Determine stable ID
            stable_id = self.raw_to_stable_id.get(raw_id)

            if stable_id is None:
                # Check if this new raw track matches any recently lost track!
                # 1. Match against lost tracks by motion continuation or re-entry
                best_match_id = None
                best_score = -1.0

                for lost_sid, lth in list(self.lost_tracks.items()):
                    gap = self.current_frame - lth.last_frame
                    if gap <= self.max_lost_frames:
                        # Extrapolate position with damping
                        pred_x1 = lth.last_box[0] + lth.velocity[0] * gap * 0.5
                        pred_y1 = lth.last_box[1] + lth.velocity[1] * gap * 0.5
                        pred_w = lth.last_box[2] - lth.last_box[0]
                        pred_h = lth.last_box[3] - lth.last_box[1]
                        pred_box = [pred_x1, pred_y1, pred_x1 + pred_w, pred_y1 + pred_h]

                        iou_val = compute_iou(pred_box, box)
                        
                        # Re-entry check: if worker exited scene near edge and re-enters near edge
                        h, w = frame.shape[:2]
                        is_near_edge_lost = (lth.last_box[0] < 50 or lth.last_box[2] > w - 50)
                        is_near_edge_curr = (box[0] < 50 or box[2] > w - 50)
                        
                        score = iou_val
                        if iou_val > 0.15:
                            score += 0.5
                        if is_near_edge_lost and is_near_edge_curr:
                            # Re-entry similarity
                            score += 0.4

                        if score > best_score and score > 0.35:
                            best_score = score
                            best_match_id = lost_sid

                if best_match_id is not None:
                    stable_id = best_match_id
                    del self.lost_tracks[best_match_id]
                else:
                    stable_id = self.next_stable_id
                    self.next_stable_id += 1

                self.raw_to_stable_id[raw_id] = stable_id

            current_matched_stable.add(stable_id)

            # Update track history
            if stable_id in self.active_tracks:
                th = self.active_tracks[stable_id]
                dt = max(1, self.current_frame - th.last_frame)
                vx = (box[0] - th.last_box[0]) / dt
                vy = (box[1] - th.last_box[1]) / dt
                # Smooth velocity
                th.velocity = [0.7 * th.velocity[0] + 0.3 * vx, 0.7 * th.velocity[1] + 0.3 * vy]
                th.last_box = box
                th.last_frame = self.current_frame
                th.frames_seen += 1
            else:
                self.active_tracks[stable_id] = TrackHistory(
                    track_id=stable_id,
                    last_frame=self.current_frame,
                    last_box=box,
                    velocity=[0.0, 0.0],
                    frames_seen=1,
                    state="active",
                )

            out_workers.append({
                "track_id": stable_id,
                "bbox_xyxy": box,
                "confidence": conf_val,
            })

        # Move unobserved active tracks to lost
        for sid in list(self.active_tracks.keys()):
            if sid not in current_matched_stable:
                th = self.active_tracks.pop(sid)
                th.state = "lost"
                self.lost_tracks[sid] = th

        return out_workers


def test_advanced_tracker():
    scenarios = [
        ("Single worker moving", "scenario1_single_worker_moving", "videos/scenarios/scenario1_single_worker_moving.mp4", "videos/scenarios/ground_truth/scenario1_single_worker_moving_gt.json"),
        ("Crossing paths", "scenario2_two_workers_crossing", "videos/scenarios/scenario2_two_workers_crossing.mp4", "videos/scenarios/ground_truth/scenario2_two_workers_crossing_gt.json"),
        ("Workers close together", "scenario3_two_workers_close_together", "videos/scenarios/scenario3_two_workers_close_together.mp4", "videos/scenarios/ground_truth/scenario3_two_workers_close_together_gt.json"),
        ("Exit and re-entry", "scenario4_worker_leaving_reentering", "videos/scenarios/scenario4_worker_leaving_reentering.mp4", "videos/scenarios/ground_truth/scenario4_worker_leaving_reentering_gt.json"),
        ("Detection failure gap", "scenario5_temp_detection_failure", "videos/scenarios/scenario5_temp_detection_failure.mp4", "videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json"),
        ("Partial occlusion", "scenario6_partial_occlusion", "videos/scenarios/scenario6_partial_occlusion.mp4", "videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json"),
    ]

    tracker = AdvancedIdentityTracker()

    print("\n=======================================================")
    print("Testing Advanced Identity Tracker on 6 Scenarios")
    print("=======================================================")

    for display_name, tag, vpath, gpath in scenarios:
        cap = cv2.VideoCapture(vpath)
        tracker.reset()
        preds = {}
        fn = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            workers = tracker.update(frame, frame_number=fn)
            preds[fn] = workers
            fn += 1
        cap.release()

        with open(gpath) as f:
            gt = {int(k): v for k, v in json.load(f).items()}

        res = evaluate_tracking(gt, preds)
        print(f"{display_name:25s} | IDF1: {res['idf1']:6.4f} | HOTA: {res['hota']:6.4f} | IDSW: {res['id_switches']:2d} | Tracks: {res['unique_track_ids']:2d} | Frag: {res['fragmentation']:2d}")

if __name__ == "__main__":
    test_advanced_tracker()

