# SafeOps Phase 2: Person Detection and Helmet-to-Person Association

## 1. Overview & Architecture

Phase 2 builds directly upon the Phase 1 `PPEHelmetAdapter` by adding:
1. **Person Detection** (`src.ppe.person_detector.PersonDetector`): Identifies human bodies and outputs bounding boxes in original image coordinates.
2. **Spatial Geometry Association** (`src.ppe.helmet_associator.HelmetAssociator`): Determines which helmet detections belong to which detected individuals based on upper-body containment, horizontal alignment, and relative geometry.
3. **End-to-End Orchestration** (`src.ppe.ppe_pipeline.PPEPipeline`): Combines person detection, helmet classification, and association into a single high-level interface that yields structured JSON outputs and annotated visualizations.

```
                  ┌───────────────────────┐
                  │      Input Image      │
                  └──────────┬────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│    PersonDetector     │         │   PPEHelmetAdapter    │
│      (yolov8n.pt)     │         │  (YOLOv8 Medium .pt)  │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
     [Person Boxes]                     [Helmet Boxes]
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │      HelmetAssociator       │
              │  - Headroom & upper-body    │
              │  - Intersection-over-Helmet │
              │  - 1-to-1 greedy matching   │
              │  - Ambiguity detection      │
              └──────────────┬──────────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │    Structured Result JSON   │
              │  + Optional Annotated Frame │
              └─────────────────────────────┘
```

---

## 2. Model Selection and Checkpoint Paths

| Component | Model Checkpoint | Source | Task | Format |
|---|---|---|---|---|
| **Person Detection** | `yolov8n.pt` (or `yolov8s.pt`) | Pretrained COCO (existing in repo) | Object Detection (Class 0: `person`) | PyTorch / Ultralytics |
| **Helmet Detection** | `runs/detect/train/weights/best.pt` | Custom SafeOps checkpoint | Object Detection (Classes 1: `with_helmet`, 2: `without_helmet`) | PyTorch / Ultralytics |

> **Note**: Both models are loaded once during initialization and retained in memory for repeated inference. No checkpoints were retrained, fine-tuned, or replaced.

---

## 3. Installation & Environment

Phase 2 introduces **zero new external dependencies**. It runs entirely within the verified SafeOps environment:
- **Python**: `3.13.13`
- **PyTorch**: `2.6.0+cu124`
- **Ultralytics**: `8.4.193`
- **OpenCV**: `4.10 / 5.0`
- **Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 12.4) or standard x86 CPU.

---

## 4. Configuration Parameters

### `PersonDetector`
- `model_path`: Path to YOLO model (default: `yolov8n.pt`).
- `device`: `"cuda"` or `"cpu"`.
- `confidence_threshold`: Minimum person confidence in `[0.0, 1.0]` (default: `0.40`).
- `iou_threshold`: NMS IoU threshold in `[0.0, 1.0]` (default: `0.45`).
- `input_size`: Inference resolution (default: `640`).

### `HelmetAssociator`
- `upper_body_ratio`: Maximum fraction of person height from top $y_1$ considered valid head zone (default: `0.45`).
- `headroom_ratio`: Allowed headroom above person box top $y_1$ as a fraction of person height to accommodate helmet crowns (default: `0.15`).
- `min_ioha`: Minimum Intersection-over-Helmet-Area (IoHA) required for matching (default: `0.25`).
- `min_match_score`: Minimum combined compatibility score in `[0.0, 1.0]` (default: `0.35`).
- `ambiguity_threshold`: Difference margin between top 2 candidate persons below which an association is flagged as `ambiguous` (default: `0.08`).

---

## 5. Association Algorithm & Scoring

Because hard hats and heads represent only ~5% to 15% of a full person bounding box, traditional Intersection-over-Union (IoU) is unsuited for matching (even a perfectly fitted helmet yields an ordinary IoU under 0.15).

Instead, the `HelmetAssociator` uses a multi-factor geometric scoring function:

1. **Vertical Feasibility**:
   A helmet center $(cx_h, cy_h)$ must lie within the head zone:
   $$y_1 - \text{headroom} \le cy_h \le y_1 + (\text{height} \times \text{upper\_body\_ratio})$$
   Helmets positioned below the upper body (e.g. at the waist or feet) receive a score of $0.0$.

2. **Intersection-over-Helmet-Area (IoHA)**:
   $$\text{IoHA} = \frac{\text{Area}(\text{Helmet} \cap \text{PersonHeadZone})}{\text{Area}(\text{Helmet})}$$
   Ensures the helmet is actually located within or overlapping the person's upper frame.

3. **Vertical Alignment**:
   Calculates proximity to the ideal head center ($\sim 8\%$ below person top $y_1$).

4. **Horizontal Alignment**:
   Measures horizontal centering relative to the person's mid-axis.

5. **Composite Score**:
   $$S = 0.40 \times \text{IoHA} + 0.35 \times \text{VerticalScore} + 0.25 \times \text{HorizontalScore}$$

6. **1-to-1 Greedy Matching & Ambiguity Handling**:
   - Matches are sorted descending by score.
   - Each helmet is assigned to at most one person; each person receives at most one helmet.
   - If two overlapping persons yield scores within `ambiguity_threshold` of each other, the association status is recorded as `"ambiguous"`.
   - Helmets that fail matching criteria are collected in `unmatched_helmets`.

---

## 6. Output JSON Schema

```json
{
  "image": "sample_01.jpg",
  "persons": [
    {
      "person_id": 0,
      "bbox": [162.78, 389.58, 546.95, 995.80],
      "confidence": 0.8027,
      "helmet": {
        "class_name": "WITH_HELMET",
        "confidence": 0.6104,
        "bbox": [262.08, 386.05, 414.01, 580.43]
      },
      "association_status": "associated"
    },
    {
      "person_id": 1,
      "bbox": [580.12, 410.20, 890.45, 950.10],
      "confidence": 0.7650,
      "helmet": null,
      "association_status": "no_helmet_detection"
    }
  ],
  "unmatched_helmets": [],
  "summary": {
    "persons_detected": 2,
    "helmets_associated": 1,
    "unmatched_helmets": 0,
    "ambiguous_associations": 0
  },
  "annotated_output": "artifacts/ppe_pipeline/annotated/sample_01_annotated.jpg"
}
```

### Classification vs. Association Status
- `WITH_HELMET` / `WITHOUT_HELMET`: Output classification from the helmet model.
- `associated`: The person successfully paired with a helmet detection.
- `no_helmet_detection`: No helmet detection matched this person (e.g. occlusion or detector miss).
- `ambiguous`: Multiple overlapping persons have competing claims on the helmet.

---

## 7. How to Run

### Automated Test Suite (58 Tests)
```powershell
.\.venv\Scripts\python.exe -m unittest discover tests -v
```

### Full Pipeline Batch Runner
```powershell
.\.venv\Scripts\python.exe scripts/test_ppe_pipeline.py --device cuda
```
For CPU execution:
```powershell
.\.venv\Scripts\python.exe scripts/test_ppe_pipeline.py --device cpu
```

Outputs are written to:
- `artifacts/ppe_pipeline/predictions.json`
- `artifacts/ppe_pipeline/annotated/`

---

## 8. Known Limitations & Next Steps

1. **Severe Occlusion & Crowds**: In dense crowds where multiple workers overlap directly in depth, bounding-box geometry alone can become ambiguous. Temporal tracking (Phase 3) will provide track trajectory history to disambiguate.
2. **Headless / Truncated Person Bounding Boxes**: If a worker's head is cut off by the camera boundary, the person detector may still bound the torso. The associator correctly flags this as `no_helmet_detection` rather than fabricating an association.
3. **Thread Safety**: YOLO models retain state per instance; concurrent inference across worker threads requires separate pipeline instances or process pooling.
