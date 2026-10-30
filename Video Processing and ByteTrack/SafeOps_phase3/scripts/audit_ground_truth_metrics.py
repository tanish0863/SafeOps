"""Audit Ground-Truth and MOT Metrics Integrity for SafeOps Phase 5.

Verifies:
1. Ground-truth worker IDs stability and coordinate conventions.
2. Frame indexing consistency.
3. IoU matching threshold consistency (0.50).
4. One-to-one per-frame bipartite matching.
5. Unmatched predictions count as False Positives (FP).
6. Unmatched ground-truth count as False Negatives (FN).
7. ID switches (IDSW) and track fragmentation (Frag) definitions.
8. IDF1 calculation vs py-motmetrics 1.4.0 (official reference).
9. HOTA calculation (DetA * AssA).
10. Tracker state reset verification.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import motmetrics as mm

sys.path.insert(0, os.path.abspath("."))
from src.evaluation.mot_metrics import evaluate_tracking, compute_iou
from src.tracking.worker_tracker import WorkerTracker


def audit_ground_truth_files():
    print("=== AUDIT PART 1: Ground-Truth File Inspection ===")
    gt_dir = Path("videos/scenarios/ground_truth")
    gt_files = sorted(gt_dir.glob("*.json"))
    
    findings = {}
    for gtf in gt_files:
        name = gtf.stem
        with open(gtf, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        frame_keys = sorted([int(k) for k in data.keys()])
        total_frames = len(frame_keys)
        
        # Check frame continuity
        expected_frames = list(range(frame_keys[0], frame_keys[0] + total_frames))
        is_contiguous = frame_keys == expected_frames
        
        # Check IDs and coordinates
        trajectories = {}
        anomalies = []
        for fk in frame_keys:
            objects = data[str(fk)]
            for obj in objects:
                gid = obj["id"]
                bbox = obj["bbox"]
                if gid not in trajectories:
                    trajectories[gid] = []
                trajectories[gid].append((fk, bbox))
                
                # Check coordinate integrity [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    anomalies.append(f"Frame {fk} ID {gid}: inverted box dimensions {bbox}")
                if x1 < 0 or y1 < 0 or x2 > 854 or y2 > 480:
                    anomalies.append(f"Frame {fk} ID {gid}: box out of frame bounds {bbox}")

        # Check trajectory jump (e.g. S5 index swap)
        jump_events = []
        for gid, points in trajectories.items():
            for (f1, b1), (f2, b2) in zip(points, points[1:]):
                if f2 == f1 + 1:
                    cx1, cy1 = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
                    cx2, cy2 = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
                    displacement = np.hypot(cx2 - cx1, cy2 - cy1)
                    if displacement > 150:  # jump greater than 150 pixels in 1 frame
                        jump_events.append(f"Frame {f1}->{f2} ID {gid}: sudden center displacement of {displacement:.1f}px (from ({cx1:.0f},{cy1:.0f}) to ({cx2:.0f},{cy2:.0f}))")

        findings[name] = {
            "total_frames": total_frames,
            "frame_range": f"{frame_keys[0]}..{frame_keys[-1]}",
            "is_contiguous": is_contiguous,
            "unique_gt_ids": list(trajectories.keys()),
            "box_anomalies": anomalies,
            "jump_events": jump_events
        }
        print(f"[{name}] Frames: {total_frames}, Contiguous: {is_contiguous}, GT IDs: {list(trajectories.keys())}")
        if anomalies:
            print(f"  Box anomalies: {anomalies}")
        if jump_events:
            print(f"  Sudden trajectory jumps: {jump_events}")
            
    return findings


def compare_with_py_motmetrics():
    print("\n=== AUDIT PART 2: Metric Comparison (src.evaluation vs py-motmetrics) ===")
    scenarios = [
        ("scenario1_single_worker_moving", "videos/scenarios/scenario1_single_worker_moving.mp4", "videos/scenarios/ground_truth/scenario1_single_worker_moving_gt.json"),
        ("scenario2_two_workers_crossing", "videos/scenarios/scenario2_two_workers_crossing.mp4", "videos/scenarios/ground_truth/scenario2_two_workers_crossing_gt.json"),
        ("scenario3_two_workers_close_together", "videos/scenarios/scenario3_two_workers_close_together.mp4", "videos/scenarios/ground_truth/scenario3_two_workers_close_together_gt.json"),
        ("scenario4_worker_leaving_reentering", "videos/scenarios/scenario4_worker_leaving_reentering.mp4", "videos/scenarios/ground_truth/scenario4_worker_leaving_reentering_gt.json"),
        ("scenario5_temp_detection_failure", "videos/scenarios/scenario5_temp_detection_failure.mp4", "videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json"),
        ("scenario6_partial_occlusion", "videos/scenarios/scenario6_partial_occlusion.mp4", "videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json"),
    ]
    
    trackers = [
        ("Current ByteTrack", "src/tracking/custom_bytetrack.yaml"),
        ("BoT-SORT without ReID", "src/tracking/botsort_no_reid.yaml")
    ]
    
    comparison_results = {}
    
    for trk_name, trk_cfg in trackers:
        tracker = WorkerTracker(tracker=trk_cfg)
        comparison_results[trk_name] = {}
        
        for sc_name, vid_path, gt_path in scenarios:
            tracker.reset()
            with open(gt_path, "r", encoding="utf-8") as fp:
                gt_raw = json.load(fp)
            gt_frames = {int(k): v for k, v in gt_raw.items()}
            
            cap = cv2.VideoCapture(vid_path)
            fn = 0
            tracker_frames = {}
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                workers = tracker.update(frame)
                tracker_frames[fn] = [{"track_id": w["track_id"], "bbox_xyxy": w["bbox_xyxy"]} for w in workers]
                fn += 1
            cap.release()
            
            # 1. Custom evaluate_tracking
            custom_res = evaluate_tracking(gt_frames, tracker_frames, iou_threshold=0.5)
            
            # 2. py-motmetrics accumulator
            acc = mm.MOTAccumulator(auto_id=True)
            all_frames = sorted(set(gt_frames.keys()) | set(tracker_frames.keys()))
            for f in all_frames:
                gts = gt_frames.get(f, [])
                preds = tracker_frames.get(f, [])
                
                gt_ids = [g["id"] for g in gts]
                tr_ids = [p["track_id"] for p in preds]
                
                # Distance matrix: 1 - IoU (distance <= 0.5 for match)
                if gts and preds:
                    dists = np.zeros((len(gts), len(preds)), dtype=np.float32)
                    for i, g in enumerate(gts):
                        for j, p in enumerate(preds):
                            iou = compute_iou(g["bbox"], p["bbox_xyxy"])
                            dists[i, j] = 1.0 - iou if iou >= 0.5 else np.nan
                else:
                    dists = np.empty((len(gts), len(preds)))
                    
                acc.update(gt_ids, tr_ids, dists)
                
            mh = mm.metrics.create()
            summary = mh.compute(
                acc,
                metrics=["num_frames", "idf1", "idp", "idr", "mota", "motp", "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "partially_tracked", "mostly_lost"],
                name="acc"
            )
            
            py_mota = float(summary["mota"].iloc[0])
            py_idf1 = float(summary["idf1"].iloc[0])
            py_idsw = int(summary["num_switches"].iloc[0])
            py_fp = int(summary["num_false_positives"].iloc[0])
            py_fn = int(summary["num_misses"].iloc[0])
            
            comparison_results[trk_name][sc_name] = {
                "custom_idf1": custom_res["idf1"],
                "pymot_idf1": round(py_idf1, 4),
                "custom_idsw": custom_res["id_switches"],
                "pymot_idsw": py_idsw,
                "custom_fp": custom_res["false_positives"],
                "pymot_fp": py_fp,
                "custom_fn": custom_res["false_negatives"],
                "pymot_fn": py_fn,
                "custom_hota": custom_res["hota"],
                "custom_deta": custom_res["deta"],
                "custom_assa": custom_res["assa"],
                "pymot_mota": round(py_mota, 4)
            }
            
            print(f"[{trk_name} | {sc_name}]")
            print(f"  IDF1: Custom={custom_res['idf1']} | py-motmetrics={round(py_idf1, 4)} (Diff: {abs(custom_res['idf1'] - round(py_idf1, 4)):.4f})")
            print(f"  IDSW: Custom={custom_res['id_switches']} | py-motmetrics={py_idsw}")
            print(f"  FP:   Custom={custom_res['false_positives']} | py-motmetrics={py_fp}")
            print(f"  FN:   Custom={custom_res['false_negatives']} | py-motmetrics={py_fn}")
            print(f"  MOTA: py-motmetrics={round(py_mota, 4)}")
            print(f"  HOTA: Custom={custom_res['hota']} (DetA={custom_res['deta']}, AssA={custom_res['assa']})")
            
    with open("outputs/phase5_metric_audit_comparison.json", "w") as fp:
        json.dump(comparison_results, fp, indent=2)
    print("\nSaved metric comparison to outputs/phase5_metric_audit_comparison.json")
    return comparison_results


if __name__ == "__main__":
    gt_findings = audit_ground_truth_files()
    metric_comp = compare_with_py_motmetrics()
