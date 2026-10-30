# PPE Guard - Phase 2: Person Detection + Pose + Head Region

One YOLO Pose model, one inference pass per frame. `PersonDetector` (boxes) and
`PoseEstimator` (17 COCO keypoints) are separate interfaces over the same result,
matched by index. `head_region.py` turns face keypoints into an approximate head box.

Head-region estimation is **not** helmet detection. Never declare PPE compliance from pose alone.

## Setup (Windows PowerShell)
```
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 1) CUDA PyTorch from https://pytorch.org/get-started/locally/
# 2) then:
pip install -r requirements.txt
python scripts/verify_gpu.py
```

## Get weights once (online), then work fully offline
```
python scripts/download_model.py yolo26n-pose.pt     # or yolo11n-pose.pt
```
If your Ultralytics version has no `yolo26n-pose.pt`, use `yolo11n-pose.pt` and pass `--model models/yolo11n-pose.pt`.

## Run
Put images/videos in `data/input/`, then:
```
python -m src.run --input data/input --output data/output
python -m src.run --input data/input/clip.mp4 --stride 2 --save-video
```
Outputs: `data/output/images.jsonl`, `<video>.jsonl`, `data/output/annotated/*.jpg`.
Overlay: green = person, blue = head region, yellow = face keypoints, red text = UNCERTAIN.

## Test
```
pytest -q
```
Tests use a fake engine, so they need no GPU or weights.

## Output contract (one JSON object per line)
`frame_number, timestamp_s, frame_width, frame_height, persons[]`; each person has
`detection_id` (frame-local), `bbox` [x1,y1,x2,y2], `confidence`, `head_region` (or null),
`head_visibility` (VISIBLE | PARTIAL | UNCERTAIN) and all 17 `keypoints` (null when
confidence < `--kp-conf`, default 0.5). Schema: `src/schemas.py`.

## Head visibility rules
- VISIBLE: nose + both eyes + both ears reliable
- PARTIAL: 2-4 reliable face landmarks
- UNCERTAIN: 0-1 reliable landmarks -> `head_region: null`

Size comes from ear distance, else eye distance, else shoulder width (fallback), capped by
person-box size. Constants at the top of `head_region.py` are heuristics: tune on real helmet images.

## Still to do on the GPU machine
Run the five scenarios (frontal, side, occluded, bent-over, multiple workers), inspect annotated
images, tune constants, and measure FPS on your own data. Nothing here claims accuracy or speed.

---

## Datasets (Phase 2 validation)

These datasets are used to validate the inference pipeline against ground-truth
COCO pose annotations. **No model is trained here.**

### Dataset overview

| Dataset | Purpose | Size (approx) | License |
|---|---|---|---|
| COCO8-Pose | Fast smoke-test; 8 COCO images with 17-kp YOLO labels | ~340 KB | CC BY 4.0 |
| COCO 2017 val images | Full 5 k-image validation set for inference quality checks | ~1 GB | CC BY 4.0 |
| COCO 2017 annotations | Ground-truth keypoint annotations for the val set | ~250 MB | CC BY 4.0 |

Attribution: COCO dataset — Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
Official site: <https://cocodataset.org>. Annotations license: Creative Commons Attribution 4.0.

### Download sources

```
COCO8-Pose   : https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip
val2017      : https://images.cocodataset.org/zips/val2017.zip
annotations  : https://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

> **Note:** `images.cocodataset.org` is served from Amazon CloudFront with a wildcard TLS
> certificate. On some Windows machines, `curl.exe` (Schannel) rejects it with
> `SEC_E_WRONG_PRINCIPAL`. The download script handles this automatically with `--insecure`
> for that specific host. No credentials are involved.

### Directory structure after download

```
datasets/
├── downloads/                          # original archives (kept for resume)
│   ├── coco8-pose.zip
│   ├── val2017.zip
│   └── annotations_trainval2017.zip
│
├── coco8-pose/                         # extracted COCO8-Pose
│   ├── images/train/  (4 images)
│   ├── images/val/    (4 images)
│   ├── labels/train/  (4 × .txt YOLO pose labels, 56 tokens each)
│   ├── labels/val/
│   └── coco8-pose.yaml
│
└── coco-pose/                          # extracted COCO 2017 val
    ├── images/val2017/   (~5,000 × .jpg)
    └── annotations/
        └── person_keypoints_val2017.json
```

### Preparing datasets

```powershell
# Download all three (COCO val images ~1 GB, allow time):
python scripts/download_datasets.py

# Skip the large image archive (annotations + COCO8 only):
python scripts/download_datasets.py --skip-coco-val
```

### Validating datasets

```powershell
# Full validation (includes per-image OpenCV readability sampling):
python scripts/validate_datasets.py

# Quick mode (skips image-by-image readability check):
python scripts/validate_datasets.py --quick
```

### Running existing unit tests

```powershell
pytest tests/test_phase2.py -v
```

Tests use a `FakeEngine` stub — no GPU, weights, or dataset files required.
