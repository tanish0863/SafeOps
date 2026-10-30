"""Visual Failure Investigation and Frame Analysis for SafeOps Phase 5.

Focuses on:
- Scenario 2: Crossing paths
- Scenario 4: Exit and re-entry
- Scenario 5: Detection blackout
- Scenario 6: Partial occlusion

Compares:
- ByteTrack Baseline (`src/tracking/custom_bytetrack.yaml`)
- BoT-SORT without ReID (`src/tracking/botsort_no_reid.yaml`)

Generates:
- Annotated error frames in outputs/phase5_failure_cases/
- CSV log: outputs/phase5_failure_analysis.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("."))
from src.evaluation.mot_metrics import compute_iou
from src.tracking.worker_tracker import WorkerTracker

OUT_DIR = Path("outputs/phase5_failure_cases")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROOT_OUT_DIR = Path("D:/Projects/SafeOps/outputs/phase5_failure_cases")
ROOT_OUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_and_render_scenario(
    scenario_name: str,
    video_path: str,
    gt_path: str,
    tracker_name: str,
    tracker_cfg: str,
    csv_writer,
):
    tracker = WorkerTracker(tracker=tracker_cfg)
    tracker.reset()

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_raw = json.load(f)
    gt_frames = {int(k): v for k, v in gt_raw.items()}

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotated_video_path = OUT_DIR / f"{scenario_name}_{tracker_name.replace(' ', '_')}_annotated.mp4"
    writer = cv2.VideoWriter(str(annotated_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    gt_history: dict[int, list[tuple[int, int]]] = {}  # gt_id -> [(frame, trk_id)]
    gt_presence: dict[int, list[int]] = {}
    for f_idx, objs in gt_frames.items():
        for o in objs:
            gt_presence.setdefault(o["id"], []).append(f_idx)

    fn = 0
    saved_failures = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        workers = tracker.update(frame)
        gts = gt_frames.get(fn, [])

        # Build IoU cost matrix
        if gts and workers:
            cost = np.zeros((len(gts), len(workers)), dtype=np.float32)
            for i, g in enumerate(gts):
                for j, p in enumerate(workers):
                    cost[i, j] = compute_iou(g["bbox"], p["bbox_xyxy"])
            r_ind, c_ind = linear_sum_assignment(-cost)
            matched_gts = set()
            matched_trks = set()
            matches = []
            for r, c in zip(r_ind, c_ind):
                if cost[r, c] >= 0.5:
                    matches.append((r, c, cost[r, c]))
                    matched_gts.add(r)
                    matched_trks.add(c)
        else:
            matches = []
            matched_gts = set()
            matched_trks = set()

        # Check errors & log CSV
        # 1. Matched pairs
        for r, c, iou_val in matches:
            gid = gts[r]["id"]
            tid = workers[c]["track_id"]
            conf = workers[c]["confidence"]
            err_type = "NONE"

            # Check ID switch
            if gid in gt_history and len(gt_history[gid]) > 0:
                prev_f, prev_tid = gt_history[gid][-1]
                if tid != prev_tid:
                    err_type = "ID_SWITCH"
                elif fn > prev_f + 1:
                    # check if was present
                    gap_presence = [gf for gf in gt_presence.get(gid, []) if prev_f < gf < fn]
                    if gap_presence:
                        err_type = "FRAGMENTATION"
            gt_history.setdefault(gid, []).append((fn, tid))

            csv_writer.writerow([
                scenario_name, fn, gid, tid, round(iou_val, 3), round(conf, 3), "MATCHED", err_type
            ])

            # Draw
            gb = [int(v) for v in gts[r]["bbox"]]
            pb = [int(v) for v in workers[c]["bbox_xyxy"]]
            cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
            cv2.putText(vis, f"GT:{gid}", (gb[0], max(15, gb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            color = (0, 0, 255) if err_type != "NONE" else (255, 200, 0)
            cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), color, 2)
            label = f"TRK:{tid} IoU={iou_val:.2f}" + (f" [{err_type}]" if err_type != "NONE" else "")
            cv2.putText(vis, label, (pb[0], min(h - 10, pb[3] + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            if err_type in ("ID_SWITCH", "FRAGMENTATION") and saved_failures < 8:
                img_name = f"{scenario_name}_{tracker_name.replace(' ', '_')}_{err_type}_f{fn}.jpg"
                cv2.imwrite(str(OUT_DIR / img_name), vis)
                cv2.imwrite(str(ROOT_OUT_DIR / img_name), vis)
                saved_failures += 1

        # 2. Unmatched GTs -> False Negatives
        for r, g in enumerate(gts):
            if r not in matched_gts:
                gid = g["id"]
                csv_writer.writerow([
                    scenario_name, fn, gid, -1, 0.0, 0.0, "UNMATCHED_GT", "FALSE_NEGATIVE"
                ])
                gb = [int(v) for v in g["bbox"]]
                cv2.rectangle(vis, (gb[0], gb[1]), (gb[2], gb[3]), (0, 0, 255), 2)
                cv2.putText(vis, f"GT:{gid} [FN]", (gb[0], max(15, gb[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if saved_failures < 8 and fn % 10 == 0:
                    img_name = f"{scenario_name}_{tracker_name.replace(' ', '_')}_FN_f{fn}.jpg"
                    cv2.imwrite(str(OUT_DIR / img_name), vis)
                    cv2.imwrite(str(ROOT_OUT_DIR / img_name), vis)
                    saved_failures += 1

        # 3. Unmatched predictions -> False Positives or Duplicate Tracks
        for c, p in enumerate(workers):
            if c not in matched_trks:
                tid = p["track_id"]
                conf = p["confidence"]
                # Check duplicate track: overlaps with any matched prediction
                err_type = "FALSE_POSITIVE"
                for _, mc, _ in matches:
                    if compute_iou(p["bbox_xyxy"], workers[mc]["bbox_xyxy"]) > 0.4:
                        err_type = "DUPLICATE_TRACK"
                        break

                csv_writer.writerow([
                    scenario_name, fn, -1, tid, 0.0, round(conf, 3), "UNMATCHED_PRED", err_type
                ])
                pb = [int(v) for v in p["bbox_xyxy"]]
                color = (0, 165, 255) if err_type == "DUPLICATE_TRACK" else (128, 128, 128)
                cv2.rectangle(vis, (pb[0], pb[1]), (pb[2], pb[3]), color, 1)
                cv2.putText(vis, f"TRK:{tid} [{err_type}]", (pb[0], min(h - 5, pb[3] + 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Header HUD
        hud_text = f"{scenario_name} | {tracker_name} | Frame {fn}/{total_frames} | GT:{len(gts)} Pred:{len(workers)}"
        cv2.putText(vis, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        writer.write(vis)
        fn += 1

    cap.release()
    writer.release()
    print(f"[{tracker_name} | {scenario_name}] Finished. Video -> {annotated_video_path}")


def main():
    csv_path = Path("outputs/phase5_failure_analysis.csv")
    root_csv_path = Path("D:/Projects/SafeOps/outputs/phase5_failure_analysis.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "scenario", "frame_number", "gt_id", "predicted_track_id",
            "iou", "detection_confidence", "match_status", "error_type"
        ])

        scenarios = [
            ("scenario2_two_workers_crossing", "videos/scenarios/scenario2_two_workers_crossing.mp4", "videos/scenarios/ground_truth/scenario2_two_workers_crossing_gt.json"),
            ("scenario4_worker_leaving_reentering", "videos/scenarios/scenario4_worker_leaving_reentering.mp4", "videos/scenarios/ground_truth/scenario4_worker_leaving_reentering_gt.json"),
            ("scenario5_temp_detection_failure", "videos/scenarios/scenario5_temp_detection_failure.mp4", "videos/scenarios/ground_truth/scenario5_temp_detection_failure_gt.json"),
            ("scenario6_partial_occlusion", "videos/scenarios/scenario6_partial_occlusion.mp4", "videos/scenarios/ground_truth/scenario6_partial_occlusion_gt.json"),
        ]

        trackers = [
            ("ByteTrack", "src/tracking/custom_bytetrack.yaml"),
            ("BoT-SORT without ReID", "src/tracking/botsort_no_reid.yaml")
        ]

        for sc_name, vid, gt in scenarios:
            for trk_name, trk_cfg in trackers:
                analyze_and_render_scenario(sc_name, vid, gt, trk_name, trk_cfg, writer)

    # Copy to root
    import shutil
    shutil.copyfile(str(csv_path), str(root_csv_path))
    print(f"Saved failure analysis CSV to {csv_path} and {root_csv_path}")


if __name__ == "__main__":
    main()
