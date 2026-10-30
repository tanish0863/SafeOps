"""Real-World Industrial Workplace Image Validation for SafeOps Phase 5.

Evaluates:
- 10+ real-world workplace images from:
  1. Safety Helmet Wearing Dataset (SHWD, Apache 2.0 License, wujixiu/helmet-detection)
  2. Real CCTV industrial video keyframes (v4.mp4, v2.mp4)
- Tests:
  * Worker wearing helmet
  * Worker without helmet
  * Worker wearing safety vest / without vest (reports UNKNOWN if vest model uninstalled)
  * Multiple workers
  * Overlapping workers
  * Small/distant workers
  * Partially occluded workers
  * Different lighting conditions

Generates:
- Annotated images in outputs/phase5_real_images/
- Metrics in outputs/phase5_real_images_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath("."))
from src.association.ppe_association import associate_ppe_to_workers
from src.contracts.schemas import HeadROIQuality, PPEStatus
from src.ppe.detector import YOLOv8HelmetDetector

OUT_DIR = Path("outputs/phase5_real_images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_OUT_DIR = Path("D:/Projects/SafeOps/outputs/phase5_real_images")
ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 curated real images from Apache 2.0 SHWD dataset
DATASET_URLS = [
    ("real_img_01_single_helmet.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00163.jpg", "Single worker wearing yellow hard hat"),
    ("real_img_02_multiple_workers.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00197.jpg", "Multiple workers on industrial site"),
    ("real_img_03_no_helmet.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00250.jpg", "Worker without helmet (bare head)"),
    ("real_img_04_overlapping_workers.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00311.jpg", "Workers close together with partial overlap"),
    ("real_img_05_distant_workers.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00325.jpg", "Distant and small-scale workers"),
    ("real_img_06_partially_occluded.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00371.jpg", "Worker partially occluded by industrial railing"),
    ("real_img_07_outdoor_bright.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00380.jpg", "High contrast direct sunlight"),
    ("real_img_08_indoor_shadowed.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00413.jpg", "Indoor factory floor with shadowing"),
    ("real_img_09_mixed_compliance.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00766.jpg", "Mixed compliance: workers with and without helmets"),
    ("real_img_10_equipment_proximity.jpg", "https://raw.githubusercontent.com/wujixiu/helmet-detection/master/hardhat-wearing-detection/imgs/00807.jpg", "Workers near heavy machinery"),
]


def download_or_load_images():
    images = []
    print("=== Downloading / Loading Real-World Test Images ===")
    raw_dir = Path("outputs/phase5_real_images_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. External dataset images
    for fname, url, desc in DATASET_URLS:
        local_path = raw_dir / fname
        if not local_path.exists():
            try:
                print(f"Downloading {fname} from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
                    f.write(resp.read())
            except Exception as e:
                print(f"Failed to download {fname}: {e}")
                continue

        img = cv2.imread(str(local_path))
        if img is not None:
            images.append({
                "name": fname,
                "source": "SHWD (Apache 2.0 License, wujixiu/helmet-detection)",
                "description": desc,
                "image": img,
                "width": img.shape[1],
                "height": img.shape[0]
            })

    # 2. Extract 2 real CCTV video frames from project assets
    cctv_sources = [
        ("real_img_11_cctv_plant_v4.jpg", "D:/Projects/new/videos/v4.mp4", 120, "Real plant CCTV keyframe from v4.mp4"),
        ("real_img_12_cctv_indoor_v2.jpg", "D:/Projects/new/videos/v2.mp4", 50, "Real indoor industrial CCTV keyframe from v2.mp4"),
    ]
    for fname, vpath, fidx, desc in cctv_sources:
        if os.path.exists(vpath):
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            cap.release()
            if ok:
                images.append({
                    "name": fname,
                    "source": f"Local CCTV Video Asset ({Path(vpath).name})",
                    "description": desc,
                    "image": frame,
                    "width": frame.shape[1],
                    "height": frame.shape[0]
                })

    print(f"Total real images prepared: {len(images)}")
    return images


def run_pipeline_on_images(images):
    print("=== Loading Models for Image Evaluation ===")
    pose_model = YOLO("models/yolov8s-pose.pt")
    helmet_detector = YOLOv8HelmetDetector("models/ppe_helmet.pt", conf=0.15)
    print(f"Helmet detector loaded with classes: {helmet_detector.class_names} (Active helmet IDs: {helmet_detector.helmet_class_ids})")

    results_summary = []

    for item in images:
        name = item["name"]
        frame = item["image"]
        desc = item["description"]
        source = item["source"]
        h, w = frame.shape[:2]

        # 1. Pose detection (person class 0)
        res_pose = pose_model.predict(frame, classes=[0], conf=0.25, verbose=False)[0]
        workers = []
        if res_pose.boxes is not None and len(res_pose.boxes) > 0:
            boxes = res_pose.boxes.xyxy.cpu().numpy()
            confs = res_pose.boxes.conf.cpu().numpy()
            kps = res_pose.keypoints.xy.cpu().numpy() if res_pose.keypoints is not None else None
            kp_confs = res_pose.keypoints.conf.cpu().numpy() if res_pose.keypoints is not None else None

            for i in range(len(boxes)):
                workers.append({
                    "track_id": i + 1,
                    "bbox_xyxy": [float(v) for v in boxes[i]],
                    "confidence": float(confs[i]),
                    "keypoints_xy": kps[i].tolist() if kps is not None else None,
                    "keypoints_conf": kp_confs[i].tolist() if kp_confs is not None else None,
                })

        # 2. Helmet detection (filtering out 'bike' and non-helmet classes)
        ppe_dets = helmet_detector.detect(frame)

        # 3. PPE Association
        associations, unassociated = associate_ppe_to_workers(workers, ppe_dets, has_vest_model=False)

        # 4. Render Annotations
        vis = frame.copy()
        for ppe in ppe_dets:
            bx = [int(v) for v in ppe.bbox_xyxy]
            cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
            cv2.putText(vis, f"HELMET {ppe.confidence:.2f}", (bx[0], max(12, bx[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        worker_stats = []
        for w_dict in workers:
            tid = w_dict["track_id"]
            wb = [int(v) for v in w_dict["bbox_xyxy"]]
            assoc = associations[tid]
            h_stat = assoc.helmet.status.value
            roi_qual = assoc.head_roi_quality.value

            color = (0, 255, 0) if h_stat == "detected" else ((0, 165, 255) if h_stat == "unknown" else (0, 0, 255))
            cv2.rectangle(vis, (wb[0], wb[1]), (wb[2], wb[3]), color, 2)
            label = f"Worker {tid} [{h_stat.upper()}] head:{roi_qual}"
            cv2.putText(vis, label, (wb[0], max(15, wb[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Draw keypoints if available
            if w_dict["keypoints_xy"]:
                for pt in w_dict["keypoints_xy"][:5]:  # nose, eyes, ears
                    px, py = int(pt[0]), int(pt[1])
                    if px > 0 and py > 0:
                        cv2.circle(vis, (px, py), 3, (255, 255, 0), -1)

            worker_stats.append({
                "worker_id": tid,
                "confidence": round(w_dict["confidence"], 3),
                "bbox": [round(v, 1) for v in w_dict["bbox_xyxy"]],
                "helmet_status": h_stat,
                "helmet_confidence": round(assoc.helmet.confidence, 3) if assoc.helmet.confidence else None,
                "vest_status": assoc.vest.status.value,
                "head_roi_quality": roi_qual,
                "reason": assoc.helmet.reason,
            })

        out_img_path = OUT_DIR / f"{Path(name).stem}_annotated.jpg"
        root_img_path = ROOT_OUT_DIR / f"{Path(name).stem}_annotated.jpg"
        cv2.imwrite(str(out_img_path), vis)
        cv2.imwrite(str(root_img_path), vis)

        img_report = {
            "image_name": name,
            "source": source,
            "description": desc,
            "dimensions": f"{w}x{h}",
            "workers_detected": len(workers),
            "helmets_detected": len(ppe_dets),
            "unassociated_helmets": len(unassociated),
            "workers": worker_stats,
            "annotated_path": str(out_img_path)
        }
        results_summary.append(img_report)
        print(f"[{name}] {len(workers)} workers, {len(ppe_dets)} helmets -> {out_img_path.name}")

    with open("outputs/phase5_real_images_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    with open("D:/Projects/SafeOps/outputs/phase5_real_images_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("\nSaved real image validation metrics to outputs/phase5_real_images_metrics.json")
    return results_summary


if __name__ == "__main__":
    imgs = download_or_load_images()
    if imgs:
        run_pipeline_on_images(imgs)
    else:
        print("ERROR: No images found or downloaded.")
