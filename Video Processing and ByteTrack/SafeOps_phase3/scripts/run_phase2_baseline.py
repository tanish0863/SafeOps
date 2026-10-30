"""Phase 2: Establish Reproducible Tracking Baseline.

Executes all six scenarios using the existing ByteTrack configuration and produces:
- outputs/tracking_baseline.json
- outputs/tracking_baseline.csv
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

sys.path.insert(0, os.path.abspath("."))

from src.evaluation.mot_metrics import evaluate_tracking
from src.tracking.worker_tracker import WorkerTracker


def run_baseline_benchmark() -> dict:
    scenarios = [
        ("Single worker moving", "scenario1_single_worker_moving", "videos/scenarios/scenario1_single_worker_moving.mp4", "videos/scenarios/ground_truth/scenario1_single_worker_moving_gt.json"),
        ("Crossing paths", "scenario2_two_workers_crossing", "videos/scenarios/scenario2_two_workers_crossing.mp4", "videos/scenarios/ground_truth/scenario2_two_workers_crossing_gt.json"),
        ("Workers close together", "scenario3_two_workers_close_together", "videos/scenarios/scenario3_two_workers_close_together.mp4", "videos/scenarios/ground_truth/scenario3_two_workers_close_together_gt.json"),
        ("Exit and re-entry", "scenario4_worker_leaving_reentering", "videos/scenarios/scenario4_worker_leaving_reentering.mp4", "videos/scenarios/ground_truth/scenario4_worker_leaving_reentering_gt.json"),
        ("Detection failure gap", "scenario5_temp_detection_failure", "videos/scenarios/scenario5_temp_detection_failure.mp4", "videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json"),
        ("Partial occlusion", "scenario6_partial_occlusion", "videos/scenarios/scenario6_partial_occlusion.mp4", "videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json"),
    ]

    results = {}
    csv_rows = []

    print("================================================================")
    print("SafeOps Phase 2: Establishing Reproducible Tracking Baseline")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("================================================================\n")

    for display_name, tag, vpath, gpath in scenarios:
        print(f"Benchmarking: {display_name} ({vpath})...")
        cap = cv2.VideoCapture(vpath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open {vpath}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = WorkerTracker(
            model_path="models/yolov8s-pose.pt",
            tracker="src/tracking/custom_bytetrack.yaml",
            conf=0.10,
            iou=0.7,
            half=False,
        )
        tracker.reset()

        # Warm up
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

        total_dur = t_end - t_start
        actual_fps = fn / total_dur if total_dur > 0 else 0.0
        mean_latency = float(np.mean(frame_latencies_ms)) if frame_latencies_ms else 0.0
        p95_latency = float(np.percentile(frame_latencies_ms, 95)) if frame_latencies_ms else 0.0
        peak_vram_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        peak_vram_res = torch.cuda.max_memory_reserved() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        with open(gpath, "r", encoding="utf-8") as gf:
            gt_data = {int(k): v for k, v in json.load(gf).items()}

        mot = evaluate_tracking(gt_data, preds_by_frame, iou_threshold=0.5)

        # Standard CLEAR MOTA: 1 - (FN + FP + IDSW) / total_gt_detections
        total_gt = mot["total_gt_detections"]
        mota = 1.0 - (mot["false_negatives"] + mot["false_positives"] + mot["id_switches"]) / total_gt if total_gt > 0 else 0.0
        mot["mota"] = round(mota, 4)

        scenario_record = {
            "scenario_name": display_name,
            "scenario_tag": tag,
            "video_path": vpath,
            "frames_processed": fn,
            "metrics": {
                "idf1": mot["idf1"],
                "hota": mot["hota"],
                "mota": mot["mota"],
                "id_switches": mot["id_switches"],
                "false_positives": mot["false_positives"],
                "false_negatives": mot["false_negatives"],
                "unique_track_ids": mot["unique_track_ids"],
                "fragmentation": mot["fragmentation"],
                "mostly_tracked": mot["mostly_tracked"],
                "partially_tracked": mot["partially_tracked"],
                "mostly_lost": mot["mostly_lost"],
                "detection_coverage_recall": mot["detection_coverage_recall"],
                "precision": mot["precision"],
            },
            "performance": {
                "processing_fps": round(actual_fps, 2),
                "mean_latency_ms": round(mean_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "peak_vram_allocated_mb": round(peak_vram_alloc, 2),
                "peak_vram_reserved_mb": round(peak_vram_res, 2),
            }
        }
        results[tag] = scenario_record

        csv_rows.append({
            "Configuration": "Baseline ByteTrack",
            "Scenario": display_name,
            "IDF1": mot["idf1"],
            "HOTA": mot["hota"],
            "MOTA": mot["mota"],
            "IDSW": mot["id_switches"],
            "FP": mot["false_positives"],
            "FN": mot["false_negatives"],
            "Track_IDs": mot["unique_track_ids"],
            "Fragments": mot["fragmentation"],
            "Mostly_Tracked": mot["mostly_tracked"],
            "Partially_Tracked": mot["partially_tracked"],
            "Mostly_Lost": mot["mostly_lost"],
            "Recall": mot["detection_coverage_recall"],
            "Precision": mot["precision"],
            "FPS": round(actual_fps, 2),
            "Mean_Latency_ms": round(mean_latency, 2),
            "P95_Latency_ms": round(p95_latency, 2),
            "VRAM_Alloc_MB": round(peak_vram_alloc, 2),
        })

        print(f"  IDF1: {mot['idf1']} | HOTA: {mot['hota']} | MOTA: {mot['mota']} | IDSW: {mot['id_switches']} | Tracks: {mot['unique_track_ids']} | Frag: {mot['fragmentation']}")
        print(f"  FPS: {round(actual_fps, 2)} | Mean Latency: {round(mean_latency, 2)}ms | P95: {round(p95_latency, 2)}ms | VRAM: {round(peak_vram_alloc, 2)}MB\n")

    out_json = Path("outputs/tracking_baseline.json")
    out_csv = Path("outputs/tracking_baseline.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fieldnames = list(csv_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Saved baseline JSON to {out_json}")
    print(f"Saved baseline CSV to {out_csv}")
    return results


if __name__ == "__main__":
    run_baseline_benchmark()

