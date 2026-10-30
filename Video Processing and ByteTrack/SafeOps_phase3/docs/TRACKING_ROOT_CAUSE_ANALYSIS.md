# SafeOps Phase 4 — Tracking Root Cause Analysis

## Executive Summary

This diagnostic analysis evaluates the existing worker tracking system in SafeOps, which employs **YOLOv8s-Pose** for worker detection and keypoint extraction paired with **ByteTrack** for multi-object tracking. While the baseline system performs perfectly on simple single-worker or non-overlapping multi-worker scenes (IDF1 = 1.0000), it experiences severe identity fragmentation and tracking degradation in challenging CCTV conditions:
- **Crossing paths:** IDF1 drops to **0.3774** with **14 unique track IDs** (GT = 2) and 2 ID switches.
- **Exit and re-entry:** IDF1 drops to **0.1880** with **8 unique track IDs** (GT = 1) and 1 ID switch.
- **Temporary detection failure (blackout gap):** IDF1 drops to **0.4046** with **2 track IDs** (GT = 1) and 2 ID switches.
- **Partial occlusion:** IDF1 drops to **0.2294** with **6 unique track IDs** (GT = 1) and 3 ID switches.

---

## 1. System Architecture & Detection-to-Tracking Pipeline

### 1.1 Ingestion & Detection Flow
1. **Video Decoding:** `video_processor.py` ingests frames via OpenCV (`open_video_stream`) yielding `(frame_number, timestamp, frame)`.
2. **Inference Execution:** `WorkerTracker.update(frame, frame_number)` calls `self.model.track(frame, persist=True, tracker="src/tracking/custom_bytetrack.yaml", classes=[0], conf=0.10, iou=0.7, imgsz=640)`.
3. **Internal Ultralytics ByteTrack Pipeline:**
   - YOLOv8s-Pose generates bounding boxes, confidence scores, and 17 COCO keypoints for class 0 (`person`).
   - Detections with `conf < 0.10` are dropped before tracker ingestion.
   - ByteTrack categorizes remaining detections into:
     - High-confidence (`conf >= track_high_thresh` = 0.25): Stage 1 association with active tracks (`tracked_stracks`) using Kalman-predicted bounding boxes and IoU distance.
     - Low-confidence (`0.10 <= conf < track_low_thresh` = 0.25): Stage 2 association with remaining unassociated tracks (`lost_stracks`).
     - Unmatched high-confidence detections (`conf >= new_track_thresh` = 0.25) initialize new tracks (`activated_stracks`).

---

## 2. Answers to Diagnostic Questions

### Q1: How YOLO detections are passed into ByteTrack
- YOLOv8s-Pose produces boxes, scores, and keypoints.
- Ultralytics passes detected bounding boxes and scores into `BYTETracker.update(results)`.
- `conf=0.10` in `WorkerTracker` acts as a hard filter; detections below 0.10 never reach ByteTrack's second stage.
- Low-confidence detections between 0.10 and 0.25 participate in ByteTrack's second-stage recovery.

### Q2: Whether tracker IDs reset unexpectedly
- Within a continuous video stream, tracker IDs increment monotonically without unexpected zeroing.
- However, when a track is lost and exceeds `track_buffer` (or drifts beyond spatial IoU reach), it transitions to `removed_stracks`. When the worker is redetected, ByteTrack has no memory of the previous ID and assigns a brand-new ID.
- Explicit resets occur only via `WorkerTracker.reset()`, which sets `self.model.predictor = None`.

### Q3: Whether the tracker is recreated between frames
- No. Because `persist=True` is provided to `model.track()`, the `BYTETracker` instance on `model.predictor.trackers[0]` is retained across consecutive frames.

### Q4: Whether `persist=True` is used correctly
- Yes. `persist=True` is passed to every frame update call in `WorkerTracker.update()`.

### Q5: Whether frame IDs and timestamps remain consistent
- Yes. `frame_number` strictly increments monotonically (0 to $N-1$) and timestamps reflect $t = \text{frame} / \text{fps}$.

### Q6: Whether detection confidence thresholds discard valid partially occluded workers
- During partial occlusion, detector confidence drops from $\sim 0.94$ to $0.45 - 0.75$, remaining above `conf=0.10`.
- However, the occluded worker's geometry is frequently fractured into multiple sub-part proposals (torso vs. legs). Because `new_track_thresh=0.25` is relatively low, these fragments are promoted into independent new tracks instead of being suppressed or merged.

### Q7: Whether `track_buffer` is appropriate for the actual FPS
- `custom_bytetrack.yaml` specifies `track_buffer: 60`.
- In Ultralytics `BYTETracker`, `self.max_frames_lost = args.track_buffer` (60 frames $\approx 2.0$ seconds at 30 FPS).
- While 60 frames is numerically sufficient for short 7-frame to 30-frame gaps, pure ByteTrack relies **exclusively** on spatial IoU with the Kalman filter prediction. When a worker moves significantly or leaves the field of view for 30 frames, the Kalman filter extrapolates velocity linearly, moving the predicted box away. The spatial IoU between the re-entering worker and the extrapolated box is **0.0**, causing the track buffer to fail completely.

### Q8: Whether low-confidence detections are being used correctly for ByteTrack's second association stage
- Yes, detections with $0.10 \le \text{conf} < 0.25$ are evaluated in Stage 2.
- However, Stage 2 only resolves low-confidence detections against already active/lost tracks with overlapping spatial IoU ($> 0.20$). It does not prevent ID switches during crossings or track creation from background noise.

### Q9: Whether bounding-box coordinates and detection formats are correct
- Detections and ground-truth use standard pixel coordinates $[x_1, y_1, x_2, y_2]$ on the original frame canvas ($854 \times 480$).
- `mot_metrics.py` clamps width and height to non-negative values, preventing coordinate inversion artifacts.

### Q10: Whether the benchmark itself correctly calculates identity metrics
- `mot_metrics.py` implements standard MOT evaluation:
  - Frame-by-frame Hungarian bipartite matching ($\text{IoU} \ge 0.50$) for TP, FP, FN.
  - Global Hungarian matching across all frames for IDF1 (`idtp`).
  - HOTA calculation via $\sqrt{\text{DetA} \times \text{AssA}}$.
  - ID Switches (IDSW) and Fragmentation (Frag).
- **Benchmark Artifacts Identified:**
  1. *Background Worker Detections:* Scenarios 2, 4, 5, and 6 were synthesized by placing blurred video frames from `v2.mp4` / `v4.mp4` as the background. YOLO detects background workers present in the blurred plates with confidences $0.45 - 0.90$. Because the ground truth only annotated the primary moving workers, these background detections produce persistent False Positives (FPs: 147 in Scenario 2, 166 in Scenario 4, 127 in Scenario 6).
  2. *Scenario 5 GT Index Swap:* In Scenario 5 ground-truth generation (`build_test_scenarios.py`), `boxes[0]` from the detector was used to generate GT. At frame 33, YOLO's detection index order flipped from the right worker to the left worker, causing GT trajectory 1 to suddenly jump across the canvas, registering 2 artificial ID switches in the ground truth.
  3. Ground-truth annotations are strictly frozen as per requirements; the evaluation must operate against the official frozen annotations.

---

## 3. Scenario-by-Scenario Failure Breakdown

| Scenario | GT Traj | Tracker IDs | IDF1 | IDSW | Frag | FP | FN | Primary Failure Mechanism |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **1. Single worker moving** | 1 | 1 | 1.0000 | 0 | 0 | 0 | 0 | Perfect linear Kalman tracking. |
| **2. Crossing paths** | 2 | 14 | 0.3774 | 2 | 1 | 147 | 16 | Spatial overlap causes Kalman IoU ambiguity; NMS suppresses one box, causing track loss; emergence creates new IDs. |
| **3. Workers close together** | 2 | 2 | 1.0000 | 0 | 0 | 0 | 0 | Constant spatial offset preserved; no crossing. |
| **4. Exit and re-entry** | 1 | 8 | 0.1880 | 1 | 0 | 166 | 0 | Re-entry position has 0.0 IoU with extrapolated Kalman state; new track forced despite track buffer. |
| **5. Detection blackout gap** | 1 | 2 | 0.4046 | 2 | 1 | 60 | 7 | 7-frame blackout drops track; reappearance + background track interaction triggers ID swap. |
| **6. Partial occlusion** | 1 | 6 | 0.2294 | 3 | 2 | 127 | 29 | Vertical pillar splits detection into sub-parts; low new-track threshold creates spurious IDs; parent track drops. |

---

## 4. Root Causes Identified

1. **Absence of Visual Appearance / ReID:**
   ByteTrack relies 100% on spatial position and bounding-box geometry (motion + IoU). When workers cross, exit/re-enter, or experience temporary detection gaps, motion models cannot disambiguate identity. Re-identification requires visual feature embeddings.

2. **Kalman Filter Drift During Absence:**
   During a 30-frame exit or occlusion, linear velocity extrapolation moves the track's search window far from the actual reappearance point. Pure IoU association fails when displacement exceeds box dimensions.

3. **Overly Permissive Track Initialization (`new_track_thresh = 0.25`):**
   When occlusion fractures a worker or background noise appears, any detection $\ge 0.25$ immediately spawns a new track. Raising `new_track_thresh` or requiring multi-frame confirmation prevents track proliferation.

4. **Greedy First-Stage Matching in Ambiguous Crossings:**
   When two workers overlap ($\text{IoU} \ge 0.35$), pure IoU distance is uninformative. Without appearance distance and keypoint geometric cues, ID switches occur inevitably.

---

## 5. Strategic Roadmap for Phases 2–6

1. **Phase 2:** Execute and save the baseline results to `outputs/tracking_baseline.json` and `outputs/tracking_baseline.csv`.
2. **Phase 3:**
   - **Config A:** Parameter-tuned ByteTrack (higher `new_track_thresh`, tuned `match_thresh`, calibrated `track_buffer`).
   - **Config B:** BoT-SORT without ReID (evaluating motion compensation and improved Kalman state formulation).
   - **Config C:** BoT-SORT / Multi-feature tracker with Appearance ReID (using lightweight visual feature embeddings).
   - **Config D:** DeepSORT / StrongSORT or Spatial-Appearance Hybrid tracker.
3. **Phase 4 & 5:** Benchmark all configurations across all 6 scenarios, verify acceptance criteria, and integrate the optimal tracker maintaining full data contract compatibility.

