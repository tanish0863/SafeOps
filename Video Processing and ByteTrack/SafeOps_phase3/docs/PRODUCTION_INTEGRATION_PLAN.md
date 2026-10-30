# SafeOps — Production Integration Plan
## Phase A: Repository Audit & Architectural Roadmap

**Document Version**: 1.0.0  
**Repository**: `D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3`  
**Target Environment**: Windows 11, NVIDIA RTX 3050 6GB Laptop GPU, CUDA 12.4, PyTorch 2.6.0 (`.venv-gpu`)  
**Audit Date**: October 30, 2026  

---

## 1. Executive Overview

This plan defines the engineering blueprint to transition the existing SafeOps Phase 3 prototype into a robust, integrated, production-oriented PPE Compliance Detection system. The system processes CCTV surveillance video streams, detects workers and body keypoints, tracks temporary worker identities, associates PPE objects with workers via keypoint-guided geometry, filters transient noise via temporal consistency, evaluates safety policies, generates violation events, and produces structured reports and annotated video artifacts accessible via a FastAPI backend.

---

## 2. Existing Architecture & Reusable Components

### 2.1 Reusable Components
1. **Pose Detection & Worker Tracking (`src/tracking/worker_tracker.py`)**:
   - Uses YOLOv8s-Pose (`models/yolov8s-pose.pt`) in FP16 on CUDA.
   - Tracks person class `0` with ByteTrack using `persist=True`.
   - Has clean `.reset()` lifecycle management between videos.
   - Extracts 17 COCO keypoints and confidences alongside bounding boxes.
2. **Tracker Configuration (`src/tracking/custom_bytetrack.yaml`)**:
   - Proven baseline configuration (`track_high_thresh: 0.25`, `track_low_thresh: 0.10`, `track_buffer: 60`).
3. **Video Processing Pipeline (`src/video/video_processor.py`)**:
   - OpenCV sequential frame decoding and metadata reader (`VideoInfo`).
   - Zero-based frame indexing and timestamp generation.
   - Sequential model execution preserving VRAM on 6GB hardware.
4. **MOT Evaluation Library (`src/evaluation/mot_metrics.py`)**:
   - Standard academic MOT evaluation (HOTA, IDF1, ID Switches, Fragmentation, MT/PT/ML, Coverage).
5. **Existing Validation Benchmarks & Test Scenarios**:
   - 6 standardized scenarios in `videos/scenarios/` with frame-by-frame ground truth.

### 2.2 Reusable Model Assets
- `models/yolov8s-pose.pt` (22.4 MB): Verified COCO 17-keypoint human pose estimator.
- `models/ppe_helmet.pt` (26.5 MB): Verified YOLOv8-seg model with classes `{0: 'bike', 1: 'with_helmet', 2: 'without_helmet'}`.

---

## 3. Critical Architectural Deficiencies & Gaps

| Area | Current Limitation in Phase 3 | Required Production Implementation |
| :--- | :--- | :--- |
| **PPE Classification** | Booleans (`helmet: bool`, `vest: bool`) assume absence of detection equals non-compliance. | Explicit state enumeration: `detected`, `not_detected`, `unknown`. Distinguish absence of evidence from evidence of absence. |
| **Vest Detection** | `models/ppe_helmet.pt` has NO vest classes; `vest` returns `False` by default. | Modular PPE detector interface reporting vest as `unknown` with `insufficient_evidence` until dedicated model is provided. |
| **Temporal Consistency** | No temporal window. Every frame is judged independently, causing flickering and false alerts. | Per-track sliding temporal window (e.g., 2.0s observation, 1.0s confirmation) with confidence aggregation. |
| **Compliance Logic** | Hardcoded inside OpenCV drawing function; no policy separation. | Decoupled `ComplianceEngine` applying configurable policies (e.g. required vs optional PPE per zone). |
| **Violation Events** | No event abstraction. No persistent start/end timestamps or evidence frames. | `ViolationManager` creating unique violation events with lifecycle states (`candidate`, `confirmed`, `resolved`). |
| **Data Contracts** | Only `tracks.jsonl`. No structured summary, video metadata, or violation reports. | Versioned Pydantic schemas for Metadata, Tracking, PPE, Compliance, Violations, and Reports. |
| **Backend Integration**| CLI scripts only (`video_processor.py`). No REST API. | Asynchronous FastAPI service supporting video upload, background processing, progress polling, and data retrieval. |
| **Tracking Identity** | Crossing paths without ReID can switch IDs; re-entries create new IDs. | Introduce explicit `uncertain_continuity` flag and geometric trajectory validation to avoid invalid identity assumptions. |
| **Video Codec** | OpenCV writes `mp4v`, which fails in web browsers. | Support web-compatible H.264 encoding via OpenCV/ffmpeg fallback. |

---

## 4. Proposed System Architecture

```
                  +-----------------------------------------+
                  |         CCTV Video Input (.mp4, .avi)   |
                  +-----------------------------------------+
                                       |
                                       v
                  +-----------------------------------------+
                  |  Video Ingestion & Validation (OpenCV)  |
                  +-----------------------------------------+
                                       | Sequential Frames
                                       v
                  +-----------------------------------------+
                  |   Worker Detection & Pose Estimation    |
                  |          (YOLOv8s-Pose, Class 0)        |
                  +-----------------------------------------+
                                       | Person Boxes + 17 Keypoints
                                       v
                  +-----------------------------------------+
                  | ByteTrack (persist=True, Trajectory KF) |
                  +-----------------------------------------+
                                       | Temporary Track IDs + Geometric States
                                       v
                  +-----------------------------------------+
                  |   Modular PPE Detection (YOLOv8-Seg)    |
                  |       (Helmet Model + Vest Interface)   |
                  +-----------------------------------------+
                                       | PPE Detections
                                       v
                  +-----------------------------------------+
                  | Keypoint-Guided PPE-to-Worker Associator |
                  |  (Head/Torso Keypoints + IoA Matching)   |
                  +-----------------------------------------+
                                       | Worker-Level PPE States
                                       v
                  +-----------------------------------------+
                  |      Temporal Consistency Smoother      |
                  |     (Sliding Window, Evidence Gate)     |
                  +-----------------------------------------+
                                       | Smoothed Observations
                                       v
                  +-----------------------------------------+
                  |    SafeOps Policy Compliance Engine     |
                  |   (Compliant / Non-Compliant / Unknown) |
                  +-----------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+------------------------------------+  +------------------------------------+
|     Violation Event Manager        |  |     Annotated Video Generator      |
| (Deduplication, Evidence Tracking) |  | (Visual States, BBoxes, Skeletons) |
+------------------------------------+  +------------------------------------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
+----------------------------------------------------------------------------+
| Output Generation: Structured JSON Reports, tracks.jsonl, Annotated MP4    |
+----------------------------------------------------------------------------+
                                       |
                                       v
+----------------------------------------------------------------------------+
| FastAPI REST Backend: Uploads, Asynchronous Job Queue, Data & Media API    |
+----------------------------------------------------------------------------+
```

---

## 5. Phase-by-Phase Implementation Sequence

### Phase B — Video Processing Engine
- Enhance [`src/video/video_processor.py`](file:///D:/Projects/SafeOps/Video%20Processing%20and%20ByteTrack/SafeOps_phase3/src/video/video_processor.py) into a streaming video reader with robust error handling for corrupted frames, missing FPS, and variable frame rates.
- Maintain zero-based frame indexing and source-video timestamp derivation.
- Add resource cleanup guarantees via context managers.

### Phase C & D — Worker Detection & Tracking Reliability
- Preserve COCO person detection and 17 pose keypoints.
- Enforce strict temporary track identity semantics (no ReID, no facial recognition).
- Add track continuity assessment: mark tracks as `uncertain` during heavy overlaps or post-occlusion re-acquisition.
- Maintain backward compatibility with the existing JSONL tracking schema.

### Phase E & F — Modular PPE Detection & Geometric Association
- Create `src/ppe/detector.py`: Modular interface abstracting helmet and vest detection.
- Map `models/ppe_helmet.pt` `{1: 'with_helmet'}` to `helmet.detected`, `{2: 'without_helmet'}` to candidate `helmet.not_detected` (with confidence checks), and `{0: 'bike'}` ignored.
- Explicitly return `vest.status = "unknown"` with reason `"missing_vest_model"`.
- Enhance `src/association/ppe_association.py` to produce rich structured records:
  ```json
  {
    "track_id": 1,
    "helmet": {"status": "detected", "confidence": 0.94, "bbox_xyxy": [120, 45, 175, 102]},
    "vest": {"status": "unknown", "confidence": null, "bbox_xyxy": null, "reason": "model_unavailable"}
  }
  ```

### Phase G & H — Temporal Consistency & Policy Compliance Engine
- Implement `src/compliance/temporal_state.py`: Per-track sliding observation window (e.g. 2.0s window, configurable minimum observations) to eliminate single-frame flickering.
- Implement `src/compliance/compliance_engine.py`: Evaluates policy requirements (`helmet_required: true`, `vest_required: false/true`).
- Possible compliance outcomes: `compliant`, `non_compliant`, `insufficient_evidence`, `unknown`.

### Phase I — Violation Event Lifecycle Manager
- Implement `src/compliance/violation_manager.py`:
  - Lifecycle: `candidate` $\to$ `confirmed` $\to$ `resolved`.
  - Tracks `event_id`, `track_id`, `violation_type` (e.g. `missing_helmet`), `start_timestamp`, `end_timestamp`, `evidence_frames`, and `status`.
  - Deduplicates events to avoid alerting on every frame.

### Phase J — Versioned Pydantic Output Contracts
- Implement `src/contracts/schemas.py`: Versioned schemas (v1.0.0) for:
  - `VideoMetadata`
  - `TrackingRecord`
  - `WorkerPPEAssociation`
  - `WorkerComplianceRecord`
  - `ViolationEvent`
  - `VideoAnalysisReport`

### Phase K — Annotated Video Production
- Enhance visual overlays: Color-code by compliance state (Green = Compliant, Red = Non-Compliant, Orange = Insufficient Evidence / Unknown).
- Show worker ID, helmet state, vest state, and compliance status.
- Ensure `mp4v` and H.264 codec compatibility.

### Phase L & M — Backend REST API & Frontend Integration
- Implement `src/api/app.py` using FastAPI:
  - `POST /api/v1/videos/upload` (multipart upload with validation)
  - `POST /api/v1/videos/{video_id}/process` (trigger background task)
  - `GET /api/v1/videos/{video_id}/status` (processing progress & stats)
  - `GET /api/v1/videos/{video_id}/tracks` (filtered track records)
  - `GET /api/v1/videos/{video_id}/compliance` (per-worker compliance states)
  - `GET /api/v1/videos/{video_id}/violations` (confirmed violation events)
  - `GET /api/v1/videos/{video_id}/report` (full structured report)
  - `GET /api/v1/videos/{video_id}/annotated_video` (stream output video)

### Phase N & O — Benchmarking & Comprehensive Test Suite
- Benchmark the full end-to-end pipeline on RTX 3050 (decode + pose + track + ppe + association + temporal + compliance + encode).
- Build comprehensive unit, integration, and scenario tests covering all 20 required edge cases.

### Phase P — Complete Documentation Suite
- Produce all 10 required technical markdown documents in `docs/`.

---

## 6. Risk Matrix & Mitigations

| Risk | Impact | Likelihood | Mitigation Strategy |
| :--- | :---: | :---: | :--- |
| **VRAM Exhaustion on RTX 3050 (6GB)** | High | Low | Enforce strictly sequential GPU inference (Pose first, then PPE on crops/frame), use FP16, and release frame buffers promptly. |
| **Over-reporting Violations from Brief Occlusions** | High | Med | Temporal confirmation window (minimum 1.0s continuous non-compliance) before raising a confirmed violation event. |
| **Missing Vest Model Causing False Non-Compliance** | High | High | Explicit `unknown` status and policy decoupling; a missing vest model yields `insufficient_evidence`, never `non_compliant`. |
| **ID Switches during Overlapping Crossings** | Med | High | Mark tracks with `uncertain_continuity` flag during high IoU overlap; never describe track IDs as permanent employee identities. |
| **OpenCV Video Codec Browser Incompatibility** | Med | Med | Write standard MP4; provide transcoding utility / guidance for web browser playback. |

