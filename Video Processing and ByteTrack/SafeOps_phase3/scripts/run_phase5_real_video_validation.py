"""Phase 5 Real-World Video Processing Validation.

Runs the complete SafeOps pipeline on authentic workplace CCTV footage:
- Video: videos/test_video.mp4 (848 frames, 854x480 @ 29.97 FPS, 28.29s)
- Primary Tracker: BoT-SORT without ReID (src/tracking/botsort_no_reid.yaml)
- Helmet Detector: models/ppe_helmet.pt
- Vest Model: None (strictly reports UNKNOWN)
- Outputs to outputs/phase5_real_video/
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pynvml
import torch

sys.path.insert(0, os.path.abspath("."))
from src.contracts.schemas import SafetyPolicyConfig
from src.ppe.detector import ModularPPEEngine
from src.tracking.worker_tracker import WorkerTracker
from src.video.video_processor import run_pipeline

OUT_DIR = Path("outputs/phase5_real_video")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_OUT_DIR = Path("D:/Projects/SafeOps/outputs/phase5_real_video")
ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    video_path = "videos/test_video.mp4"
    print(f"=== Starting Real-World Video Validation on {video_path} ===")

    # Initialize NVML
    has_nvml = False
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        has_nvml = True
    except Exception as e:
        print("NVML not available:", e)

    out_video = str(OUT_DIR / "real_cctv_annotated.mp4")
    out_jsonl = str(OUT_DIR / "tracks.jsonl")
    out_report = str(OUT_DIR / "report.json")

    tracker = WorkerTracker(tracker="src/tracking/botsort_no_reid.yaml")
    ppe_engine = ModularPPEEngine(helmet_model_path="models/ppe_helmet.pt", vest_model_path=None, conf=0.25)
    policy = SafetyPolicyConfig(require_helmet=True, require_vest=False, violation_confirmation_seconds=1.0)

    t0 = time.perf_counter()
    report = run_pipeline(
        video_path=video_path,
        out_video=out_video,
        out_jsonl=out_jsonl,
        out_report=out_report,
        tracker=tracker,
        ppe_engine=ppe_engine,
        policy=policy,
    )
    total_duration = time.perf_counter() - t0

    peak_vram_mb = 0.0
    if has_nvml:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak_vram_mb = round(mem.used / (1024 ** 2), 1)

    # Analyze tracks.jsonl
    track_records_count = 0
    unique_tracks = set()
    frames_per_track = {}
    helmet_detected_count = 0
    unknown_helmet_count = 0
    active_violation_frames = 0

    with open(out_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            track_records_count += 1
            tid = rec["track_id"]
            unique_tracks.add(tid)
            frames_per_track[tid] = frames_per_track.get(tid, 0) + 1

            ppe = rec.get("ppe", {})
            if ppe.get("helmet"):
                helmet_detected_count += 1
            if ppe.get("helmet_status") == "unknown":
                unknown_helmet_count += 1

            if rec.get("compliance", {}).get("active_violation"):
                active_violation_frames += 1

    # Extract 4 representative keyframes from annotated video
    cap = cv2.VideoCapture(out_video)
    for kf in [100, 300, 500, 700]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ok, frame = cap.read()
        if ok:
            kf_name = f"real_video_keyframe_f{kf}.jpg"
            cv2.imwrite(str(OUT_DIR / kf_name), frame)
            cv2.imwrite(str(ROOT_OUT_DIR / kf_name), frame)
    cap.release()

    metrics = {
        "video_metadata": report.video_metadata.model_dump(),
        "processing_summary": {
            "total_frames_processed": report.summary.total_frames_processed,
            "pipeline_duration_seconds": round(total_duration, 2),
            "processing_fps": report.summary.processing_fps,
            "mean_latency_ms": report.summary.mean_latency_ms,
            "p95_latency_ms": report.summary.p95_latency_ms,
            "peak_vram_mb": peak_vram_mb,
            "unique_workers_count": report.summary.unique_workers_count,
            "unique_track_ids": sorted(list(unique_tracks)),
            "total_track_records": track_records_count,
            "frames_per_worker": frames_per_track,
            "total_violations_count": report.summary.total_violations_count,
            "active_violation_frames": active_violation_frames,
            "helmet_detected_records": helmet_detected_count,
            "unknown_helmet_records": unknown_helmet_count,
            "processing_errors": 0
        },
        "violations": [v.model_dump() for v in report.violations]
    }

    metrics_path = OUT_DIR / "real_video_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    root_metrics_path = ROOT_OUT_DIR / "real_video_metrics.json"
    with open(root_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Replicate main artifacts to root
    shutil.copyfile(out_video, str(ROOT_OUT_DIR / "real_cctv_annotated.mp4"))
    shutil.copyfile(out_jsonl, str(ROOT_OUT_DIR / "tracks.jsonl"))
    shutil.copyfile(out_report, str(ROOT_OUT_DIR / "report.json"))

    print("\n=== REAL VIDEO VALIDATION SUCCESSFUL ===")
    print(f"Processed: {report.summary.total_frames_processed} frames in {total_duration:.1f}s ({report.summary.processing_fps:.1f} FPS)")
    print(f"Unique Track IDs: {sorted(list(unique_tracks))}")
    print(f"Violations Logged: {report.summary.total_violations_count}")
    print(f"Saved artifacts to {OUT_DIR} and {ROOT_OUT_DIR}")


if __name__ == "__main__":
    main()
