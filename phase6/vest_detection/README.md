# SafeOps Phase 6 — Safety Vest Detection Module

An independent, high-performance Computer Vision module for detecting safety vests and assessing worker PPE compliance in real-time camera streams.

---

## Key Features

- **Strict Separation of Concerns:** Fully self-contained Python package located strictly within `vest_detection/`. No modifications to Camera Quality Guardian, existing helmet models, or tracking systems.
- **Genuine Pretrained Model:** Uses `Hansung-Cho/yolov8-ppe-detection` (MIT License) with dual labels (`Safety Vest` and `NO-Safety Vest`), eliminating naive absence heuristics.
- **Hardware Optimized:** Runs at **95.7 FPS** (10.45 ms/frame) on an NVIDIA RTX 3050 Laptop GPU with only **~40 MB** VRAM usage.
- **Original-Pixel Coordinate Guarantee:** All detections (full-frame or cropped torso inference) are strictly re-projected back to original image space `[x1, y1, x2, y2]`.
- **Three-State Assessment:** Accurately distinguishes `PRESENT`, `ABSENT`, and `UNKNOWN` based on optical visibility, illumination, edge sharpness, and IoU torso overlap.
- **100% Tested:** 27 automated unit and integration tests with 89% code coverage.

---

## Directory Layout

```text
vest_detection/
├── src/
│   └── vest_detector/
│       ├── __init__.py           # Package exports
│       ├── config.py             # Configuration thresholds and device settings
│       ├── model_loader.py       # YOLO backend and mock backend loaders
│       ├── vest_detector.py      # Main VestDetector interface
│       ├── torso_inference.py    # Torso optical quality & 3-state assessment
│       ├── result_adapter.py     # Label normalization & coordinate re-projection
│       └── schemas.py            # Pydantic v2 schemas and validation
├── models/
│   └── yolov8n_vest_best.pt      # Pretrained YOLOv8n weights (5.96 MB, MIT)
├── tests/
│   ├── test_model_loading.py     # Backend & device selection tests
│   ├── test_vest_detection.py    # Full-frame & video frame detection tests
│   ├── test_torso_inference.py   # Torso crop & 3-state evaluation tests
│   └── test_result_schema.py     # Schema validation and coordinate integrity tests
├── samples/                      # Realistic test fixtures (vest, no-vest, dark, etc.)
├── artifacts/
│   └── evaluation/
│       └── latency_benchmark.json # Empirical benchmarks (FPS, VRAM, latency)
├── docs/
│   ├── model_research.md         # Dataset & model candidate evaluation
│   ├── evaluation_report.md      # Performance benchmarks and decision metrics
│   ├── limitations.md            # Environmental and scale limitations
│   └── installation.md           # Setup, installation, and usage instructions
├── requirements.txt              # Pinned Python package dependencies
└── README.md                     # Project overview and documentation
```

---

## Required Interface Contract

```python
from vest_detector import VestDetector

detector = VestDetector(
    model_path="models/yolov8n_vest_best.pt",
    device="cuda"  # or "cpu"
)

# 1. Full-image detection
result = detector.detect(image)

# 2. Torso crop detection (coordinates returned in full image space)
torso_result = detector.detect_torso(image, torso_bbox=[100, 160, 290, 380])

# 3. Three-State Torso Assessment
assessment = detector.assess_torso(image, torso_bbox=[100, 160, 290, 380])
# assessment.state -> VestAssessmentState.PRESENT | ABSENT | UNKNOWN
```

### JSON Output Contract

```json
{
  "schema_version": "1.0",
  "detections": [
    {
      "class_name": "WITH_VEST",
      "confidence": 0.93,
      "bbox": [100, 160, 290, 380]
    }
  ]
}
```

---

## Three-State Assessment Rules

| State | Condition |
| :--- | :--- |
| **PRESENT** | A valid `WITH_VEST` prediction exists, its confidence meets `vest_confidence_threshold` (default 0.35), and its bounding box sufficiently overlaps the torso (`IoU >= 0.18` or `intersection_ratio >= 0.30`). |
| **ABSENT** | A valid `NO_VEST` prediction exists, its confidence meets `no_vest_confidence_threshold` (default 0.35), its bounding box sufficiently overlaps the torso, and torso visibility, lighting, blur, and size checks pass. |
| **UNKNOWN** | No valid positive or negative evidence exists, or the torso is dark, blurry, too small (< 35x45px), heavily occluded, or outside the camera frame. **A missing detection strictly returns UNKNOWN unless validated negative evidence supports ABSENT.** |

---

## Benchmark Summary

- **GPU (RTX 3050 Laptop):** 10.45 ms / frame (**95.7 FPS**), 40 MB PyTorch reserved cache (~500 MB total process footprint)
- **CPU (Host Fallback):** 37.32 ms / frame (**26.8 FPS**)
- **Test Suite:** 27 / 27 tests passing (100%), 89% code coverage
