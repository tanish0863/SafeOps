"""Comprehensive Evaluation Sweep of Tracking Configurations.

Compares:
1. Baseline ByteTrack (Control)
2. Tuned ByteTrack (higher new_track_thresh, calibrated match_thresh)
3. BoT-SORT without ReID (gmc_method: none)
4. BoT-SORT with GMC (sparseOptFlow)
5. BoT-SORT with ReID (auto)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath("."))

from src.evaluation.mot_metrics import evaluate_tracking
from src.tracking.worker_tracker import WorkerTracker

SCENARIOS = [
    ("Single worker moving", "scenario1_single_worker_moving", "videos/scenarios/scenario1_single_worker_moving.mp4", "videos/scenarios/ground_truth/scenario1_single_worker_moving_gt.json"),
    ("Crossing paths", "scenario2_two_workers_crossing", "videos/scenarios/scenario2_two_workers_crossing.mp4", "videos/scenarios/ground_truth/scenario2_two_workers_crossing_gt.json"),
    ("Workers close together", "scenario3_two_workers_close_together", "videos/scenarios/scenario3_two_workers_close_together.mp4", "videos/scenarios/ground_truth/scenario3_two_workers_close_together_gt.json"),
    ("Exit and re-entry", "scenario4_worker_leaving_reentering", "videos/scenarios/scenario4_worker_leaving_reentering.mp4", "videos/scenarios/ground_truth/scenario4_worker_leaving_reentering_gt.json"),
    ("Detection failure gap", "scenario5_temp_detection_failure", "videos/scenarios/scenario5_temp_detection_failure.mp4", "videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json"),
    ("Partial occlusion", "scenario6_partial_occlusion", "videos/scenarios/scenario6_partial_occlusion.mp4", "videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json"),
]


def evaluate_config(config_name: str, yaml_path: str) -> dict:
    print(f"\n=======================================================")
    print(f"EVALUATING CONFIG: {config_name} ({yaml_path})")
    print(f"=======================================================")

    results = {}

    for display_name, tag, vpath, gpath in SCENARIOS:
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = WorkerTracker(
            model_path="models/yolov8s-pose.pt",
            tracker=yaml_path,
            conf=0.10,
            iou=0.7,
            half=False,
        )
        tracker.reset()

        dummy = np.zeros((height, width, 3), dtype=np.uint8)
        _ = tracker.update(dummy)
        tracker.reset()

        torch.cuda.reset_peak_memory_stats()
        frame_latencies_ms = []
        preds_by_frame = {}
        fn = 0

        t_start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            workers = tracker.update(frame, frame_number=fn)
            t1 = time.perf_counter()
            frame_latencies_ms.append((t1 - t0) * 1000.0)

            preds = []
            for w in workers:
                preds.append({
                    "track_id": w["track_id"],
                    "bbox_xyxy": w["bbox_xyxy"],
                    "confidence": w["confidence"],
                })
            preds_by_frame[fn] = preds
            fn += 1
        t_end = time.perf_counter()
        cap.release()

        dur = t_end - t_start
        actual_fps = fn / dur if dur > 0 else 0.0
        mean_lat = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0
        p95_lat = float(np.percentile(frame_latencies_ms, 95)) if frame_latencies_ms else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        with open(gpath, "r", encoding="utf-8") as gf:
            gt_data = {int(k): v for k, v in json.load(gf).items()}

        mot = evaluate_tracking(gt_data, preds_by_frame, iou_threshold=0.5)
        total_gt = mot["total_gt_detections"]
        mota = 1.0 - (mot["false_negatives"] + mot["false_positives"] + mot["id_switches"]) / total_gt if total_gt > 0 else 0.0

        results[tag] = {
            "scenario": display_name,
            "idf1": mot["idf1"],
            "hota": mot["hota"],
            "mota": round(mota, 4),
            "id_switches": mot["id_switches"],
            "false_positives": mot["false_positives"],
            "false_negatives": mot["false_negatives"],
            "unique_track_ids": mot["unique_track_ids"],
            "fragmentation": mot["fragmentation"],
            "fps": round(actual_fps, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "p95_latency_ms": round(p95_lat, 2),
            "peak_vram_mb": round(peak_vram, 2),
        }
        print(f"  {display_name:25s} | IDF1: {mot['idf1']:6.4f} | HOTA: {mot['hota']:6.4f} | IDSW: {mot['id_switches']:2d} | Tracks: {mot['unique_track_ids']:2d} | Frag: {mot['fragmentation']:2d} | FPS: {actual_fps:5.1f}")

    return results


def main():
    # 1. Config A: Tuned ByteTrack
    tuned_bt_path = "src/tracking/tuned_bytetrack.yaml"
    with open(tuned_bt_path, "w") as f:
        yaml.dump({
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.30,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.45,  # Suppress spurious fragments
            "track_buffer": 90,        # Extended window
            "match_thresh": 0.85,      # Tolerant spatial match
            "fuse_score": True,
        }, f)

    # 2. Config B: BoT-SORT without ReID (GMC none)
    botsort_no_reid_path = "src/tracking/botsort_no_reid.yaml"
    with open(botsort_no_reid_path, "w") as f:
        yaml.dump({
            "tracker_type": "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.35,
            "track_buffer": 90,
            "match_thresh": 0.80,
            "fuse_score": True,
            "gmc_method": "none",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": False,
        }, f)

    # 3. Config B2: BoT-SORT with GMC (sparseOptFlow)
    botsort_gmc_path = "src/tracking/botsort_gmc.yaml"
    with open(botsort_gmc_path, "w") as f:
        yaml.dump({
            "tracker_type": "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.35,
            "track_buffer": 90,
            "match_thresh": 0.80,
            "fuse_score": True,
            "gmc_method": "sparseOptFlow",
            "proximity_thresh": 0.5,
            "appearance_thresh": 0.8,
            "with_reid": False,
        }, f)

    # 4. Config C: BoT-SORT with ReID (auto)
    botsort_reid_path = "src/tracking/botsort_reid.yaml"
    with open(botsort_reid_path, "w") as f:
        yaml.dump({
            "tracker_type": "botsort",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.35,
            "track_buffer": 90,
            "match_thresh": 0.80,
            "fuse_score": True,
            "gmc_method": "none",
            "proximity_thresh": 0.3,   # Relaxed proximity to allow wider ReID match
            "appearance_thresh": 0.70, # Appearance similarity threshold
            "with_reid": True,
            "model": "auto",
        }, f)

    configs = [
        ("Current ByteTrack", "src/tracking/custom_bytetrack.yaml"),
        ("Tuned ByteTrack", tuned_bt_path),
        ("BoT-SORT (no ReID, no GMC)", botsort_no_reid_path),
        ("BoT-SORT (with GMC)", botsort_gmc_path),
        ("BoT-SORT (with ReID auto)", botsort_reid_path),
    ]

    all_results = {}
    for name, path in configs:
        all_results[name] = evaluate_config(name, path)

    out_file = Path("outputs/sweep_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll configurations swept. Saved to {out_file}")


if __name__ == "__main__":
    main()

