# Phase 1: Reusable YOLOv8 Helmet Detection Module Integration

## 1. Module Purpose

The `PPEHelmetAdapter` module (`src.ppe.helmet_adapter`) provides a production-grade, reusable wrapper around the pre-trained YOLOv8 Medium helmet detection checkpoint for the SafeOps Personal Protective Equipment (PPE) Compliance System.

It is designed to be imported and executed by downstream PPE pipeline stages (e.g., worker compliance evaluation, live camera monitoring, edge microservices) without exposing raw Ultralytics internals.

### Key Capabilities
- **Single Model Load**: Loads the model once during initialization and reuses the instance across all subsequent inference requests.
- **Hardware Acceleration**: Explicit support for NVIDIA CUDA GPU acceleration and CPU fallback.
- **Standardized Classes**: Converts internal model outputs into uppercase canonical labels (`WITH_HELMET`, `WITHOUT_HELMET`).
- **Strict Class Filtering**: Automatically discards non-helmet classes (such as `bike`) and any unexpected class IDs.
- **Original Pixel Coordinates**: Yields bounding boxes in absolute XYXY pixel coordinates relative to the input image.
- **Configurable Thresholds**: Supports both uniform float thresholds and granular per-class confidence thresholds.
- **Single-Pass Inference & Annotation**: Allows retrieving both standardized predictions and the raw Ultralytics `Results` object in a single forward pass via `predict_with_result()` to eliminate duplicate inference passes.

---

## 2. Checkpoint Location and Metadata

- **Checkpoint Path**: `runs/detect/train/weights/best.pt`
- **Architecture**: YOLOv8 Medium (`yolov8m` detection model)
- **Task**: Object Detection (`detect`)
- **Trained Model Class Mapping**:
  ```python
  {
      0: "bike",
      1: "with_helmet",
      2: "without_helmet"
  }
  ```

> **Note**: This checkpoint is frozen. It is strictly for inference integration; no fine-tuning or retraining was performed.

---

## 3. Supported Classes and Canonical Mapping

| Original Class Name | Model Class ID | Adapter Label | Action |
|---|---|---|---|
| `bike` | `0` | *None* | **Discarded** (never returned) |
| `with_helmet` | `1` | `WITH_HELMET` | **Included** |
| `without_helmet` | `2` | `WITHOUT_HELMET` | **Included** |
| *Other / Unknown* | Any | *None* | **Discarded** |

---

## 4. Installation & Environment Requirements

- **Python**: 3.10+ (Tested on Python 3.13)
- **PyTorch**: 2.0+ (Tested on PyTorch 2.6.0+cu124)
- **Torchvision**: 0.15+ (Tested on Torchvision 0.21.0+cu124)
- **Ultralytics**: 8.0+ (Tested on Ultralytics 8.4.193)
- **OpenCV**: 4.8+ (Tested on OpenCV 4.10 / 5.0)
- **NumPy**: 1.24+ (Tested on NumPy 2.2+)

---

## 5. CUDA and CPU Setup

The adapter supports two devices: `"cuda"` and `"cpu"`.

```python
# GPU execution (requires NVIDIA GPU with CUDA drivers)
detector = PPEHelmetAdapter(model_path="runs/detect/train/weights/best.pt", device="cuda")

# CPU execution
detector = PPEHelmetAdapter(model_path="runs/detect/train/weights/best.pt", device="cpu")
```

### Strict Device Guarantees
- If `device="cuda"` is specified but `torch.cuda.is_available()` returns `False`, the adapter **raises a `RuntimeError`**. It does **not** silently fall back to CPU.
- Any unrecognized device string raises a `ValueError`.

---

## 6. Adapter Initialization

```python
from src.ppe.helmet_adapter import PPEHelmetAdapter

adapter = PPEHelmetAdapter(
    model_path="runs/detect/train/weights/best.pt",
    device="cuda",              # "cuda" or "cpu"
    confidence_threshold=0.25,  # float or dict
    iou_threshold=0.45          # NMS IoU threshold (0.0 to 1.0)
)
```

Initialization performs:
1. File existence validation.
2. Device verification and CUDA readiness checks.
3. Checkpoint loading via `ultralytics.YOLO`.
4. Verification that required classes (`with_helmet`, `without_helmet`) exist in the checkpoint metadata.
5. Dummy forward pass warmup to pre-compile CUDA/cuDNN kernels.

---

## 7. Prediction Usage & Minimal Example

### Minimal Example
```python
from src.ppe.helmet_adapter import PPEHelmetAdapter

detector = PPEHelmetAdapter(
    model_path="runs/detect/train/weights/best.pt",
    device="cuda",
    confidence_threshold=0.25
)

predictions = detector.predict("samples/100_jpg.rf.UPi9uTv6J2hwVjamw6dj.jpg")

for detection in predictions:
    print(detection)
```

### Input Types Accepted
1. **File path**: `str` or `pathlib.Path` pointing to an existing image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
2. **NumPy Array**: `np.ndarray` of shape `(H, W, 3)` in standard OpenCV BGR format.

### Single-Pass Inference with Annotations
```python
predictions, raw_result = detector.predict_with_result(frame)
annotated_frame = raw_result.plot()  # Generated without running YOLO twice
```

---

## 8. Expected Output Schema

The `.predict()` method returns a `list[dict]` containing 0 or more detection objects:

```json
[
    {
        "class_name": "WITH_HELMET",
        "confidence": 0.9123,
        "bbox": [120.5, 50.25, 210.0, 130.75]
    },
    {
        "class_name": "WITHOUT_HELMET",
        "confidence": 0.8745,
        "bbox": [320.0, 80.0, 410.5, 190.25]
    }
]
```

### Contract Rules
- `class_name`: Either `"WITH_HELMET"` or `"WITHOUT_HELMET"`.
- `confidence`: `float` rounded to 4 decimal places, bounded within `[0.0, 1.0]`.
- `bbox`: Absolute pixel coordinates `[x1, y1, x2, y2]` in the original input image frame (not normalized, not scaled to 640x640).
- If no helmet objects meet the threshold, `[]` is returned.

---

## 9. Confidence Configuration

### Uniform Threshold
```python
adapter = PPEHelmetAdapter(
    model_path="runs/detect/train/weights/best.pt",
    confidence_threshold=0.30
)
```

### Granular Class-Specific Thresholds
Useful when tuning for high precision on `with_helmet` while maximizing recall on safety violations (`without_helmet`):

```python
adapter = PPEHelmetAdapter(
    model_path="runs/detect/train/weights/best.pt",
    confidence_threshold={
        "with_helmet": 0.35,
        "without_helmet": 0.20
    }
)
```

All thresholds are validated to be strictly between `0.0` and `1.0`.

---

## 10. Error Handling Matrix

| Scenario | Exception Raised | Error Message Summary |
|---|---|---|
| Checkpoint file does not exist | `FileNotFoundError` | `[PPEHelmetAdapter] Checkpoint not found: <path>` |
| CUDA requested but unavailable | `RuntimeError` | `[PPEHelmetAdapter] CUDA was requested but is not available on this system...` |
| Invalid device parameter | `ValueError` | `[PPEHelmetAdapter] Invalid device '<name>'. Supported values: ['cpu', 'cuda']` |
| Threshold outside `[0.0, 1.0]` | `ValueError` | `confidence_threshold must be a float in [0.0, 1.0]` |
| Missing required helmet classes | `RuntimeError` | `Checkpoint is missing required classes: ...` |
| Image file path not found | `FileNotFoundError` | `[PPEHelmetAdapter] Image not found: <path>` |
| Corrupt or unreadable image | `RuntimeError` | `cv2.imread returned None for: <path>` |
| Malformed NumPy input shape | `ValueError` | `NumPy image must be HxWx3 (BGR); got shape ...` |
| Underlying inference crash | `RuntimeError` | `[PPEHelmetAdapter] YOLO inference failed: <cause>` (chained with `from exc`) |

---

## 11. Testing Commands

### Automated Test Suite (29 unit & integration tests)
```powershell
.\.venv\Scripts\python.exe -m unittest tests/test_helmet_adapter.py -v
```
*(Also fully compatible with `pytest tests/test_helmet_adapter.py -v` if pytest is installed).*

### Sample Image Batch Runner
```powershell
.\.venv\Scripts\python.exe scripts/test_helmet.py
```

---

## 12. Output Locations

- **Source Code**: `src/ppe/helmet_adapter.py`
- **Module Init**: `src/ppe/__init__.py`
- **Test Script**: `scripts/test_helmet.py`
- **Unit Test Suite**: `tests/test_helmet_adapter.py`
- **Predictions JSON**: `artifacts/helmet_integration/predictions.json`
- **Annotated Images**: `artifacts/helmet_integration/annotated/*.jpg`
- **Documentation**: `docs/helmet_integration.md`

---

## 13. Known Limitations

1. **Non-Thread-Safe**: Ultralytics YOLO models maintain internal state across calls and are not thread-safe for simultaneous execution from multiple worker threads. For multi-threaded applications, instantiate one `PPEHelmetAdapter` per thread or use a process pool.
2. **Context Scope**: The adapter detects helmet objects present anywhere in the frame. It does **not** associate helmets with specific worker bodies or track identities. Worker association is handled in subsequent pipeline stages (Phase 2+).
3. **Hardware Precision**: On CUDA, half-precision (FP16) inference is enabled by default for maximum throughput. On CPU, single-precision (FP32) is enforced.
