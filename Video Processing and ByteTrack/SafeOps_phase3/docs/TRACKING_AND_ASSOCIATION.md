# SafeOps Worker Tracking and PPE Association Guide

**Document Version**: 1.0.0  
**Repository**: `D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3`  

---

## 1. Worker Tracking Subsystem

### 1.1 Strict Identity Boundaries
- **Temporary Video Track IDs**: All worker IDs (`track_id: 1, 2, ...`) are strictly temporary spatial-temporal continuity markers within a single video file.
- **No Biometrics / No ReID**: Facial recognition, facial embeddings, appearance feature extractors, and clothing ReID models are **strictly excluded**.
- **No Employee Identity Claims**: A track ID must never be interpreted as an employee badge number or permanent identity.
- **Uncertainty Flagging**: If a worker leaves the scene or passes through a dense crowd crossing, the system marks the track with `continuity_state: "uncertain"` rather than fabricating continuity.

### 1.2 ByteTrack Configuration (`custom_bytetrack.yaml`)

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25   # First-pass association threshold
track_low_thresh: 0.10    # Second-pass recovery threshold for weak detections
new_track_thresh: 0.25    # Confidence required to initialize a new track
track_buffer: 60          # Base lost track buffer in frames (scaled by fps / 30)
match_thresh: 0.80        # IoU distance matching threshold
fuse_score: True          # Incorporates detection confidence into matching cost
```

### 1.3 Buffer Scaling Formula
Ultralytics ByteTrack scales `track_buffer` dynamically based on video FPS:
$$\text{effective\_buffer} = \text{track\_buffer} \times \frac{\text{FPS}}{30.0}$$
- At 30 FPS: $\text{effective\_buffer} = 60 \text{ frames} = 2.0\text{ seconds}$.
- At 15 FPS: $\text{effective\_buffer} = 30 \text{ frames} = 2.0\text{ seconds}$.

---

## 2. Keypoint-Guided PPE Association Methodology

### 2.1 Anatomical Region Derivation

```
+----------------------------------------------------+
| Worker Bounding Box (bbox_xyxy)                    |
|                                                    |
|    +-----------------------------+                 |
|    | Head Region                 |  <-- Keypoints  |
|    | [nose(0), eyes(1-2), ears]  |      0, 1, 2, 3,|
|    +-----------------------------+      padded up  |
|                                                    |
|    +-----------------------------+                 |
|    | Torso Region                |  <-- Shoulders  |
|    | [shoulders(5-6), hips(11-12)|      (5, 6) &   |
|    +-----------------------------+      hips(11-12)|
|                                                    |
+----------------------------------------------------+
```

1. **Head Region (Helmet Matching)**:
   - Evaluates face keypoints: nose (`0`), left/right eyes (`1, 2`), left/right ears (`3, 4`).
   - If at least 2 face landmarks have confidence $\ge 0.40$:
     Derives bounding box and pads upward: $\text{top} = \min(y) - 0.18 \times h$.
   - **Fallback**: When face keypoints are occluded or back is turned, uses top 25% of worker bbox:
     $[x_1, y_1, x_2, y_1 + 0.25 \times h]$.

2. **Torso Region (Vest Matching)**:
   - Evaluates shoulder landmarks (`5, 6`) and hip landmarks (`11, 12`).
   - If at least 2 landmarks are reliable:
     Derives torso box and pads outward by $0.06 \times h$.
   - **Fallback**: $[x_1, y_1 + 0.15 \times h, x_2, y_1 + 0.65 \times h]$.

### 2.2 Matching Score: Intersection over Area (IoA)
Rather than standard IoU, PPE association calculates Intersection over Area (IoA):
$$\text{IoA} = \frac{\text{Area}(\text{PPE Box} \cap \text{Worker Anatomical Region})}{\text{Area}(\text{PPE Box})}$$
- **Thresholds**: $\text{MIN\_IOA} = 0.40$ for helmets, $0.35$ for vests.
- **Score**: $\text{Score} = \text{IoA} \times \text{Confidence}(\text{PPE})$.

### 2.3 One-to-One Constraints & Ambiguity Handling
- Sorted in descending order of score.
- **Greedy Matching**: A PPE detection is awarded to the highest scoring worker and removed from candidate pools (`used_dets.add(di)`).
- **Slot Isolation**: Each worker receives at most 1 helmet and at most 1 vest (`used_worker_slots.add((tid, item_type))`).
- **Primary Key**: All assignments are indexed by `w["track_id"]`. Detection ordering in the input list has zero impact on output.

