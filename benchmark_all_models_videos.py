#!/usr/bin/env python3
"""
benchmark_all_models_videos.py
Runs a comprehensive evaluation matrix:
5 Test Videos x 5 Distinct Custom PPE Models on CUDA GPU (RTX 4050).
"""

import cv2
import time
import json
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO

# Import local pipeline modules
from detection import PersonDetector
from ppe_detection import PPEDetector
import config

ROOT = Path(__file__).resolve().parent

VIDEOS = [
    ROOT / "videos" / "for helmet v1.mp4",
    ROOT / "videos" / "v2.mp4",
    ROOT / "videos" / "v4.mp4",
    ROOT / "videos" / "v5.mp4",
    ROOT / "videos" / "vi3.mp4",
]

MODELS = [
    {
        "id": "M1_prod_best",
        "name": "Production (train/best.pt)",
        "path": ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        "arch": "YOLOv8 Medium (85.3 MB)",
    },
    {
        "id": "M2_ppe_seg",
        "name": "PPE Seg (models/ppe_helmet.pt)",
        "path": ROOT / "models" / "ppe_helmet.pt",
        "arch": "YOLOv8s-seg (25.3 MB)",
    },
    {
        "id": "M3_train2",
        "name": "Train Run 2 (train-2/best.pt)",
        "path": ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt",
        "arch": "YOLOv8 Small (21.4 MB)",
    },
    {
        "id": "M4_train3",
        "name": "Train Run 3 (train-3/best.pt)",
        "path": ROOT / "runs" / "detect" / "train-3" / "weights" / "best.pt",
        "arch": "YOLOv8 Small (21.4 MB)",
    },
    {
        "id": "M5_train4",
        "name": "Train Run 4 (train-4/best.pt)",
        "path": ROOT / "runs" / "detect" / "train-4" / "weights" / "best.pt",
        "arch": "YOLOv8 Medium (85.3 MB)",
    },
]


def run_benchmark():
    print("=" * 80)
    print("COMPREHENSIVE MULTI-MODEL x MULTI-VIDEO BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    person_detector = PersonDetector(
        model_path=str(ROOT / "yolov8n.pt"),
        conf_threshold=config.PERSON_CONF_THRESHOLD,
        iou_threshold=config.PERSON_IOU_THRESHOLD,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results_matrix = []

    for m_info in MODELS:
        m_path = m_info["path"]
        if not m_path.exists():
            print(f"Skipping {m_info['name']} (file not found)")
            continue

        print(f"\n--> Loading Model: {m_info['name']} [{m_info['arch']}]")
        ppe_detector = PPEDetector(
            model_path=str(m_path),
            conf=config.PPE_CONF_THRESHOLD,
            iou=config.PPE_IOU_THRESHOLD,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        for v_path in VIDEOS:
            if not v_path.exists():
                continue

            cap = cv2.VideoCapture(str(v_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample 25 frames evenly across video
            sample_stride = max(1, total_frames // 25)

            total_persons = 0
            with_helmet = 0
            without_helmet = 0
            unknown = 0
            inference_times = []

            frame_idx = 0
            samples_tested = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % sample_stride != 0:
                    continue

                samples_tested += 1

                # Person detection
                detections = person_detector.detect(frame)
                if not detections:
                    continue

                # Temporary mock tracks for person crops
                from tracking import TrackedPerson
                tracks = []
                for idx, det in enumerate(detections):
                    tp = TrackedPerson(
                        track_id=idx + 1,
                        bbox_xyxy=det.bbox_xyxy,
                        confidence=det.confidence,
                        frame_h=det.frame_h,
                        frame_w=det.frame_w,
                    )
                    tracks.append(tp)

                t0 = time.perf_counter()
                ppe_detector.process_tracks(tracks, frame)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_times.append(time.perf_counter() - t0)

                for tp in tracks:
                    total_persons += 1
                    if tp.has_helmet is True:
                        with_helmet += 1
                    elif tp.has_helmet is False:
                        without_helmet += 1
                    else:
                        unknown += 1

            cap.release()

            avg_ms = (np.mean(inference_times) * 1000) if inference_times else 0.0
            known = with_helmet + without_helmet
            compliance = (100.0 * with_helmet / known) if known > 0 else 0.0

            record = {
                "model_id": m_info["id"],
                "model_name": m_info["name"],
                "architecture": m_info["arch"],
                "video": v_path.name,
                "samples_tested": samples_tested,
                "total_persons": total_persons,
                "with_helmet": with_helmet,
                "without_helmet": without_helmet,
                "unknown": unknown,
                "compliance_pct": round(compliance, 1),
                "avg_inference_ms": round(avg_ms, 2),
            }
            results_matrix.append(record)
            print(f"   [{v_path.name:18s}] Persons: {total_persons:3d} | Helmet: {with_helmet:2d} | No-Helmet: {without_helmet:2d} | Compliance: {compliance:4.1f}% | Latency: {avg_ms:5.1f}ms")

    # Save to JSON report
    report_file = ROOT / "all_models_benchmark_report.json"
    report_file.write_text(json.dumps(results_matrix, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved complete results matrix to {report_file.name}")


if __name__ == "__main__":
    run_benchmark()
