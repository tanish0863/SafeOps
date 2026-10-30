"""Phase 5: Benchmark All Candidate Tracker Configurations.

Produces:
- outputs/tracking_comparison.json
- outputs/tracking_comparison.csv
"""
from __future__ import annotations

import csv
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


def run_benchmark_for_config(config_name: str, yaml_path: str) -> list[dict]:
    print(f"\n=================================================================")
    print(f"BENCHMARKING CONFIGURATION: {config_name}")
    print(f"Config File: {yaml_path}")
    print(f"=================================================================")

    config_results = []

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

        record = {
            "Configuration": config_name,
            "Scenario": display_name,
            "IDF1": mot["idf1"],
            "HOTA": mot["hota"],
            "MOTA": round(mota, 4),
            "IDSW": mot["id_switches"],
            "FP": mot["false_positives"],
            "FN": mot["false_negatives"],
            "Track_IDs": mot["unique_track_ids"],
            "Fragments": mot["fragmentation"],
            "FPS": round(actual_fps, 2),
            "Mean_Latency_ms": round(mean_lat, 2),
            "P95_Latency_ms": round(p95_lat, 2),
            "Peak_VRAM_MB": round(peak_vram, 2),
        }
        config_results.append(record)
        print(f"  {display_name:25s} | IDF1: {mot['idf1']:6.4f} | HOTA: {mot['hota']:6.4f} | IDSW: {mot['id_switches']:2d} | Tracks: {mot['unique_track_ids']:2d} | Frag: {mot['fragmentation']:2d} | FPS: {actual_fps:5.1f}")

    return config_results


def main():
    # Ensure all YAMLs exist
    os.makedirs("src/tracking", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1. Optimized ByteTrack
    opt_bt_path = "src/tracking/optimized_bytetrack.yaml"
    with open(opt_bt_path, "w") as f:
        yaml.dump({
            "tracker_type": "bytetrack",
            "track_high_thresh": 0.25,
            "track_low_thresh": 0.10,
            "new_track_thresh": 0.45,  # Higher threshold suppresses fragment tracks in occlusion
            "track_buffer": 90,        # 3.0 seconds memory
            "match_thresh": 0.85,      # Tolerant association
            "fuse_score": True,
        }, f)

    # 2. BoT-SORT without ReID
    botsort_noreid_path = "src/tracking/botsort_no_reid.yaml"
    with open(botsort_noreid_path, "w") as f:
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

    # 3. BoT-SORT with ReID (auto)
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
            "proximity_thresh": 0.3,
            "appearance_thresh": 0.70,
            "with_reid": True,
            "model": "auto",
        }, f)

    # 4. BoT-SORT with GMC (camera motion compensation)
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

    configs = [
        ("Current ByteTrack", "src/tracking/custom_bytetrack.yaml"),
        ("Optimized ByteTrack", opt_bt_path),
        ("BoT-SORT without ReID", botsort_noreid_path),
        ("BoT-SORT with ReID", botsort_reid_path),
        ("BoT-SORT with GMC", botsort_gmc_path),
    ]

    all_records = []
    nested_json = {}

    for name, path in configs:
        records = run_benchmark_for_config(name, path)
        all_records.extend(records)
        nested_json[name] = {r["Scenario"]: r for r in records}

    # Save outputs
    out_json = Path("outputs/tracking_comparison.json")
    out_csv = Path("outputs/tracking_comparison.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(nested_json, f, indent=2)

    fieldnames = list(all_records[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)

    print(f"\nSaved tracking comparison to {out_json}")
    print(f"Saved tracking comparison to {out_csv}")


if __name__ == "__main__":
    main()

