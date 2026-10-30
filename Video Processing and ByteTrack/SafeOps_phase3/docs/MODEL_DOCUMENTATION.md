# SafeOps Model Documentation & Asset Provenance

**Document Version**: 1.0.0  
**Repository**: `D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3`  

---

## 1. Worker Detection & Pose Estimation Model

- **File**: `models/yolov8s-pose.pt`
- **Architecture**: Ultralytics YOLOv8s Pose Estimator
- **Weight Size**: 22.4 MB (23,513,657 bytes)
- **Input Dimensions**: 640 x 640 x 3 (RGB/BGR)
- **Output Structure**:
  - Person bounding boxes (`boxes.xyxy`)
  - Class probabilities (filtered strictly on COCO class `0`: Person)
  - Detection confidences (`boxes.conf`)
  - 17 COCO body keypoints (`[x, y]` coordinates) and per-keypoint confidence scores:
    - 0: Nose
    - 1-2: Left / Right Eye
    - 3-4: Left / Right Ear
    - 5-6: Left / Right Shoulder
    - 7-8: Left / Right Elbow
    - 9-10: Left / Right Wrist
    - 11-12: Left / Right Hip
    - 13-14: Left / Right Knee
    - 15-16: Left / Right Ankle
- **Inference Mode**: FP16 (`half=True`) on NVIDIA GeForce RTX 3050 (6GB VRAM), FP32 on CPU.
- **Role in SafeOps**: Detects workers, estimates anatomical reference points (head, shoulders, torso), and feeds person boxes into ByteTrack.

---

## 2. Helmet PPE Detection Model

- **File**: `models/ppe_helmet.pt`
- **Architecture**: YOLOv8-seg (Segmentation Model)
- **Weight Size**: 26.5 MB (26,564,130 bytes)
- **Training Dataset**: `with-or-without-helmet_prepared.yolov8`
- **Input Dimensions**: 640 x 640 x 3
- **Trained Class Dictionary**:
  ```python
  {
      0: "bike",
      1: "with_helmet",
      2: "without_helmet"
  }
  ```
- **Class Mapping Strategy in SafeOps**:
  - Class `1` (`with_helmet`): Mapped directly to `PPEItemType.HELMET` (`status: "detected"`).
  - Class `2` (`without_helmet`): Candidate negative indicator. Note: Absence of helmet detection is NOT automatically treated as non-compliance unless confirmed by temporal observation of the worker's head region.
  - Class `0` (`bike`): Ignored (non-worker artifact).

---

## 3. Safety Vest Detection Model Status

- **Status**: **UNAVAILABLE / NOT INSTALLED**
- **Architecture Requirement**: A dedicated YOLOv8 object detection or segmentation model trained on high-visibility safety vests (`reflective_vest`, `hi_vis_vest`).
- **Engineering Handling in SafeOps**:
  - The PPE subsystem (`src/ppe/detector.py`) implements `ModularPPEEngine` with a modular vest slot (`self.vest_detector`).
  - When `vest_model_path` is `None` or file does not exist, `has_vest_model` returns `False`.
  - For every worker, vest status is explicitly emitted as:
    ```json
    {
      "status": "unknown",
      "confidence": null,
      "bbox_xyxy": null,
      "reason": "vest_model_not_installed"
    }
    ```
  - **Zero Fabrication**: The system never guesses, fakes, or invents vest compliance.
  - **Policy Compatibility**: The `SafetyPolicyConfig` defaults `require_vest: false` so that workers are not penalized for an uninstalled vest model.

