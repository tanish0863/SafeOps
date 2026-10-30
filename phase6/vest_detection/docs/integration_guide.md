# Upstream Integration Guide — Safety Vest Detection Module

This document defines the integration specification, public API surface, and usage patterns for integrating the Safety Vest Detection module into the upstream **PPE Guard** pipeline.

---

## 1. Module Overview & Responsibilities

The module is an independent computer vision library located at:
```text
E:\PROJECTS\safeops\phase6\vest_detection
```

### Upstream Division of Responsibilities

| Pipeline Stage | Responsible Component | Implementation in Vest Module |
| :--- | :--- | :--- |
| **Video Decoding & Camera Quality** | Camera Quality Guardian / Video Streamer | **Excluded** |
| **Worker / Person Detection** | Upstream PPE Guard (YOLO Person / Pose) | **Excluded** |
| **Worker Tracking & IDs** | Upstream PPE Guard (ByteTrack / SORT) | **Excluded** |
| **Torso Region Estimation** | Upstream PPE Guard (Bounding box or pose keypoints) | **Excluded** |
| **Safety Vest Detection** | **VestDetector Module** (`detect`) | **Included** |
| **Torso Compliance Assessment** | **VestDetector Module** (`assess_torso`) | **Included** (`PRESENT`, `ABSENT`, `UNKNOWN`) |
| **Overall Worker Compliance Decision** | Upstream PPE Guard (Combining Helmet + Vest) | **Excluded** |

---

## 2. Installation & Environment Setup

### 2.1 Dependencies
Ensure the upstream environment has installed:
```text
torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.1.0
opencv-python>=4.8.0
pydantic>=2.5.0
numpy>=1.24.0
```

For GPU acceleration matching the RTX 3050 Laptop GPU driver (CUDA 12.4):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics opencv-python pydantic numpy
```

### 2.2 Model Weight Location
The genuine pretrained weights are located at:
```text
E:\PROJECTS\safeops\phase6\vest_detection\models\yolov8n_vest_best.pt
```
- **Architecture:** YOLOv8n (Nano), 3.2M parameters, 5.96 MB disk size.
- **License:** MIT License.

---

## 3. Initialization

Import `VestDetector` and `VestDetectorConfig` from `vest_detector`:

```python
import sys
from pathlib import Path

# Add vest_detection/src to PYTHONPATH if importing directly from repository
sys.path.insert(0, str(Path("E:/PROJECTS/safeops/phase6/vest_detection/src").resolve()))

from vest_detector import VestDetector, VestDetectorConfig
```

### 3.1 CUDA (GPU) Initialization (Recommended)
```python
# Auto-selects CUDA if available, falls back to CPU if unavailable
detector = VestDetector(
    model_path="E:/PROJECTS/safeops/phase6/vest_detection/models/yolov8n_vest_best.pt",
    device="cuda"
)
```

### 3.2 CPU-Only Initialization
```python
detector_cpu = VestDetector(
    model_path="E:/PROJECTS/safeops/phase6/vest_detection/models/yolov8n_vest_best.pt",
    device="cpu"
)
```

### 3.3 Custom Threshold Overrides
```python
config = VestDetectorConfig(
    vest_confidence_threshold=0.40,     # Confidence for WITH_VEST -> PRESENT
    no_vest_confidence_threshold=0.35,  # Confidence for NO_VEST -> ABSENT
    min_overlap_iou=0.18,               # Minimum IoU between torso and vest
    min_torso_width=35.0,               # Minimum torso width (px)
    min_torso_height=45.0               # Minimum torso height (px)
)

detector = VestDetector(
    model_path="E:/PROJECTS/safeops/phase6/vest_detection/models/yolov8n_vest_best.pt",
    device="cuda",
    config=config
)
```

---

## 4. Integration Usage Patterns

The module provides three distinct methods depending on the upstream workflow:

### Pattern A: Torso Compliance Assessment (Primary Integration Flow)
Use this when the upstream person detector / tracker provides the full video frame and the estimated torso bounding box of each worker.

```python
import cv2
from vest_detector import VestDetector, VestAssessmentState

detector = VestDetector("E:/PROJECTS/safeops/phase6/vest_detection/models/yolov8n_vest_best.pt")

# Frame: (H, W, 3) BGR numpy array
frame = cv2.imread("camera_frame.jpg")

# Worker torso in full-frame pixel coordinates [x1, y1, x2, y2]
worker_torso_bbox = [1357.0, 670.0, 1457.0, 810.0]

# Perform localized 3-state assessment
assessment = detector.assess_torso(frame, worker_torso_bbox)

# Interpretation:
print(f"Status: {assessment.status.value}")      # "PRESENT" | "ABSENT" | "UNKNOWN"
print(f"Confidence: {assessment.confidence:.2f}")# float (0.0 to 1.0)
print(f"Vest Detected: {assessment.vest_detected}")# True / False

# Detailed audit evidence:
evidence = assessment.evidence
print("Overlap IoU:", evidence.overlap_iou)
print("Quality Passed:", evidence.quality_checks_passed)
print("Reasons:", evidence.decision_reasons)

# Convert to dictionary or JSON for upstream logging:
result_dict = assessment.to_dict()
```

---

### Pattern B: Full-Frame Video Detection
Use this when you want to scan an entire frame for all worn safety vests across the site.

```python
# Returns dictionary conforming to SafeOps 1.0 schema:
full_result = detector.detect(frame)
# Or alias for individual video frames:
full_result = detector.detect_frame(frame)

for det in full_result["detections"]:
    cls_name = det["class_name"]   # Always "WITH_VEST"
    conf = det["confidence"]       # e.g. 0.89
    x1, y1, x2, y2 = det["bbox"]   # Original full-frame pixel coordinates
    print(f"Found {cls_name} ({conf:.2f}) at [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
```

---

### Pattern C: Standalone Torso Crop Inference
Use this if the upstream pipeline has already cropped the worker's torso as an isolated image patch and needs to detect vests inside that patch:

```python
# Crop already extracted by upstream caller
torso_crop = frame[670:810, 1357:1457]

# Run detection directly on the cropped array
crop_result = detector.detect(torso_crop)

# Note: In this pattern, returned bounding boxes are relative to torso_crop [0, 0]
for det in crop_result["detections"]:
    print("Vest inside crop:", det["bbox"], "Confidence:", det["confidence"])
```

---

## 5. Interpreting Three-State Assessment Output

The module strictly emits one of three canonical states:

| State | Semantic Meaning | Trigger Conditions | Recommended Upstream Action |
| :--- | :--- | :--- | :--- |
| **`PRESENT`** | Worker is compliant with safety vest requirements. | Confident `WITH_VEST` detection with verified torso overlap ($\text{IoU} \ge 0.18$ or $\ge 30\%$ area overlap). | Mark worker vest status as COMPLIANT. |
| **`ABSENT`** | Worker is confirmed to NOT be wearing a safety vest. | Confident `NO_VEST` detection on torso AND optical quality checks passed (good lighting and focus). | Flag worker vest violation. |
| **`UNKNOWN`** | Vest presence cannot be reliably determined. | Missing detections, worker too distant/small ($< 35\times 45$ px), dark ($< 20$ brightness), glare ($> 248$), or blurred. | Suppress violation alert; await clearer frames or worker approach. |

> [!IMPORTANT]
> **Absence Guarantee:** The module **never** defaults to `ABSENT` merely because no vest was detected. If evidence is ambiguous, it returns `UNKNOWN`.

---

## 6. Coordinate System Requirements

- **Reference Frame:** Absolute pixel coordinates $[x_1, y_1, x_2, y_2]$ where:
  - $x_1$ = left edge (pixels from image left)
  - $y_1$ = top edge (pixels from image top)
  - $x_2$ = right edge ($x_2 > x_1$)
  - $y_2$ = bottom edge ($y_2 > y_1$)
- **Resolution Origin:** $(0, 0)$ is the top-left corner of the full camera frame.
- **Crop Mapping Guarantee:** In `detect_torso()` and `assess_torso()`, all returned bounding boxes in `assessment.detections` are automatically re-projected back to full-frame pixel coordinates.

---

## 7. Error Handling & Edge Cases

The module handles edge cases safely without raising unhandled exceptions in production video loops:

```python
try:
    assessment = detector.assess_torso(frame, torso_bbox)
except ValueError as e:
    # Occurs if image is empty or invalid filepath
    logger.error(f"Image error: {e}")
except TypeError as e:
    # Occurs if image is not a numpy array, str, or Path
    logger.error(f"Type error: {e}")
```

### Handled Edge Cases (Returning `UNKNOWN` gracefully):
- **Torso outside frame:** If $x_1 < 0$, $y_1 < 0$, or coordinates exceed frame boundaries by $> 35\%$, returns `UNKNOWN` with `decision_reasons: ["Torso is completely outside the camera frame."]`.
- **Tiny / Distant worker:** If torso width $< 35$ px or height $< 45$ px, returns `UNKNOWN` with `decision_reasons: ["Torso dimensions below minimum resolution threshold."]`.
- **Night / Dark scene:** If mean pixel intensity $< 20.0$, returns `UNKNOWN` with `decision_reasons: ["Torso region is severely underexposed."]`.

---

## 8. Hardware & Performance Profile

Benchmarked on **NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)**:

- **Throughput:** **85.8 FPS** (11.66 ms / frame on CUDA FP16).
- **Torso Assessment Speed:** **80.4 FPS** (12.43 ms per worker torso assessment).
- **GPU VRAM Allocation:**
  - Active Tensor Memory: **5.80 MB** (14.96 MB peak activations).
  - PyTorch Caching Allocator: **34.00 MB**.
  - Total Python Process Footprint: **~500 MB** (including Windows WDDM & CUDA runtime context).
  - Headroom: **> 3.5 GB free VRAM** remaining on a 4.0 GB GPU, ensuring concurrent execution alongside person tracking and helmet models without OOM hazards.
