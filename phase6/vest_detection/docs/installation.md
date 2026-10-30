# Safety Vest Detection — Installation & Setup Guide

## 1. Prerequisites

- **Operating System:** Windows 10/11 or Ubuntu Linux (20.04/22.04)
- **Python:** Version 3.10, 3.11, or 3.12 (Python 3.12 tested)
- **GPU (Recommended):** NVIDIA GPU with CUDA 12.1+ support (RTX 3050 Laptop GPU tested)
- **VRAM:** Minimum 2 GB (4 GB recommended)

---

## 2. Environment Setup

### 2.1 Create Isolated Virtual Environment
```bash
cd E:\PROJECTS\safeops\phase6\vest_detection
python -m venv .venv
```

Activate the environment:
- **Windows PowerShell:**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux/macOS:**
  ```bash
  source .venv/bin/activate
  ```

### 2.2 Install PyTorch with CUDA Support
For CUDA 12.4 support (matches RTX 3050 driver):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
*(If running on CPU only, standard `pip install torch torchvision` is sufficient).*

### 2.3 Install Package Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Pretrained Model Weights

The pretrained YOLOv8n PPE/Vest model weights are stored at:
```text
models/yolov8n_vest_best.pt
```

To re-download the genuine checkpoint from Hugging Face:
```python
from huggingface_hub import hf_hub_download
import shutil

path = hf_hub_download(repo_id="Hansung-Cho/yolov8-ppe-detection", filename="best.pt")
shutil.copyfile(path, "models/yolov8n_vest_best.pt")
```

---

## 4. Running Tests

Run the full pytest suite with coverage:
```bash
pytest tests/ -v --cov=src/vest_detector --cov-report=term-missing
```

---

## 5. Quick Usage Example

```python
import cv2
from vest_detector import VestDetector

# 1. Initialize detector (auto-detects CUDA or falls back to CPU)
detector = VestDetector(
    model_path="models/yolov8n_vest_best.pt",
    device="cuda"
)

# 2. Detect vests in full frame
image = cv2.imread("samples/worker_with_vest.jpg")
result = detector.detect(image)
print(result)

# 3. Assess specific worker torso (Three-State: PRESENT, ABSENT, UNKNOWN)
torso_bbox = [325.0, 270.0, 425.0, 410.0]
assessment = detector.assess_torso(image, torso_bbox)
print(f"Vest State: {assessment.status.value} (Confidence: {assessment.confidence:.2f})")
```
