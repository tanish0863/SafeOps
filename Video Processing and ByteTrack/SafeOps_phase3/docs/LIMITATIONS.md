# SafeOps Known Limitations and Operating Boundaries

This document provides a rigorous, transparent assessment of the technical boundaries, failure modes, and hardware constraints of the SafeOps PPE Compliance Detection System.

---

## 1. Multi-Object Tracking (ByteTrack) Boundaries

### 1.1 Strict Absence of Biometric Re-Identification
By architectural design and privacy compliance:
- **No Face Recognition**: The system does not capture, store, or process facial feature vectors.
- **No Appearance ReID**: The system does not compute deep appearance embeddings (such as OSNet or ResNet ReID features) to associate individuals across cameras or disparate video segments.
- **Temporary Scope**: Track IDs (e.g., `track_id=1`, `track_id=2`) are ephemeral, motion-based identifiers valid strictly within continuous video segments.

### 1.2 Crossing Paths and Spatial Overlap
- **Limitation**: When two workers cross paths and their 2D bounding boxes overlap with $\text{IoU} > 0.60$ for multiple consecutive frames, ByteTrack's motion Kalman filter can swap worker IDs upon separation.
- **Mitigation Implemented**: SafeOps actively monitors pairwise bounding box IoU. During and immediately following high-overlap states, tracks are assigned `continuity_state = TrackingContinuityState.UNCERTAIN`. This prevents the compliance engine from attributing non-compliant behavior to the wrong individual.

### 1.3 Exit and Re-Entry
- **Limitation**: When a worker completely exits the camera frame and re-enters after a time gap exceeding `track_buffer` ($\ge 60\text{ frames}$, or $\sim 2.0\text{ seconds}$ at 30 FPS), ByteTrack assigns a **new tracking ID**.
- **Impact**: The worker's previous compliance history is not linked to the new session ID. This prevents erroneous historical carryover.

### 1.4 Extended Static Occlusion
- **Limitation**: If a worker is completely obscured behind large industrial machinery or structural columns for $> 60$ frames, the track terminates. Upon emerging, a new track ID is spawned.

---

## 2. PPE Detection Model Limitations

### 2.1 Missing Safety Vest Model
- **Fact**: The provided model weights (`models/ppe_helmet.pt`) were trained exclusively on a helmet segmentation dataset (`with-or-without-helmet_prepared.yolov8`). Its class dictionary consists solely of:
  ```python
  {0: 'bike', 1: 'with_helmet', 2: 'without_helmet'}
  ```
- **Policy Enforcement**: The system strictly refuses to fabricate or mock safety vest detections. Safety vest status is systematically reported as:
  ```json
  {
    "vest_detected": "unknown",
    "reason": "vest_model_not_installed"
  }
  ```
- **Compliance Guard**: The compliance engine policy `require_vest=False` prevents false non-compliance violations from being triggered by the absent vest model. To enable vest tracking, a dedicated two-stage or multi-class model must be plugged into `ModularPPEEngine`.

### 2.2 Low Resolution and Worker Scale
- **Limitation**: Small workers located $> 25\text{ meters}$ from a 1080p camera occupy bounding boxes smaller than $40 \times 80$ pixels. At this scale, helmet and keypoint detection accuracy degrades significantly.
- **Recommended Operating Envelope**: Worker height $\ge 120\text{ pixels}$ in the camera frame.

### 2.3 Extreme Camera Angles & Adverse Lighting
- **Steep Overhead (Nadir) Views**: Steep bird's-eye cameras obscure body keypoints (shoulders and hips project into a single point), impeding keypoint-guided torso ROI derivation.
- **Low-Light / Lens Flare**: Extreme backlighting or heavy shadowing in poorly lit warehouse areas reduces YOLO detection confidence below the 0.25 threshold.

---

## 3. Geometric Association Constraints

### 3.1 Worker Clutter & Crowd Density
- **Limitation**: In dense worker groups (3+ workers in physical contact or carrying heavy objects together), head and torso regions can overlap heavily.
- **Mitigation Implemented**: The `associate_ppe_to_workers` algorithm enforces an `ambiguity_margin` check. If a helmet detection has comparable IoA overlap with two workers ($\Delta \text{IoA} \le 0.08$), the detection is rejected or marked ambiguous to prevent misattribution.

---

## 4. Hardware and Throughput Constraints

### 4.1 Real-Time Processing on RTX 3050 (6GB)
- **Measured Throughput**:
  - Worker Detection + Pose + ByteTrack: **56.3 FPS** (exceeds real-time 30 FPS requirement).
  - Full Integrated Pipeline (Pose + PPE Seg + Association + Video Render): **10.5 – 18.5 FPS**.
- **Inference Bottleneck**: Running two separate deep neural networks (YOLOv8s-Pose and YOLOv8-Seg PPE) sequentially on 1080p frames takes $\sim 50\text{ ms}$ on a mobile RTX 3050.
- **Production Recommendation**:
  1. For offline batch processing: Current 18.5 FPS throughput is fully functional.
  2. For 30+ FPS real-time CCTV streams: Convert models to TensorRT FP16 engine (`.engine`) or deploy on desktop/server GPUs (e.g., RTX 4070 / NVIDIA A10).

---

## 5. Summary Matrix of Failure Modes & Safeguards

| Failure Mode | Trigger Condition | System Safeguard |
| :--- | :--- | :--- |
| **Track ID Swap** | Workers cross with $\text{IoU} > 0.60$ | Flags `TrackingContinuityState.UNCERTAIN` on both tracks |
| **Re-entry New ID** | Worker absent $> 2.0\text{ s}$ | Ephemeral ID contract; never presumes biometric ReID |
| **Missing Vest Alerts** | Absence of vest model | Reports `unknown`; policy avoids false `non_compliant` |
| **Transient Detection Drop** | 1-2 frames missed detection | Temporal sliding window suppresses false violation alerts |
| **Web Browser Black Screen** | OpenCV `mp4v` FourCC | Explicit documentation & ffmpeg H.264 transcoding script |

