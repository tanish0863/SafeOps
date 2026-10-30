# SafeOps Production Deployment & Operations Guide

This guide details the deployment, configuration, and runtime operation of the SafeOps PPE Compliance Detection System on production infrastructure with NVIDIA GPU acceleration.

---

## 1. System Requirements & Hardware Specification

### Recommended Target Platform
- **Operating System**: Windows 11 (64-bit) or Ubuntu Linux 22.04 LTS
- **GPU**: NVIDIA RTX 3050 Laptop GPU (6GB VRAM) or desktop equivalent (e.g., RTX 3060, RTX 4060, T4, A10)
- **Driver Version**: NVIDIA Driver $\ge 550.54$ (CUDA 12.4 runtime compatible)
- **CPU**: Intel Core i5/i7 (11th Gen+) or AMD Ryzen 5/7 (6 physical cores minimum)
- **System RAM**: 16 GB DDR4/DDR5
- **Storage**: High-speed NVMe SSD (minimum 50 GB free for video caching and artifacts)

---

## 2. Environment Setup

### 2.1 Python Environment Configuration
The project is configured to run inside a dedicated virtual environment (`.venv-gpu`) with Python 3.12 and PyTorch with CUDA 12.4:

```powershell
# Navigate to the repository root
cd "D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3"

# Verify Python version
python --version  # Output: Python 3.12.x

# Verify PyTorch CUDA acceleration
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

### 2.2 Installing Dependencies
Install production dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

### 2.3 Required Model Weights & Paths
Ensure required models are present in the `models/` directory:

| Path | Model Type | Resolution | Size | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| `models/yolov8s-pose.pt` | YOLOv8s Pose | 640x640 | ~22 MB | Person detection & 17 body keypoints |
| `models/ppe_helmet.pt` | YOLOv8-Seg PPE | 640x640 | ~6 MB | Helmet detection (with/without helmet) |
| `src/tracking/custom_bytetrack.yaml` | Tracker Config | N/A | ~1 KB | ByteTrack hyperparameter configuration |

---

## 3. Execution Modes

### 3.1 Command-Line Interface (Batch Video Processing)
For offline batch analysis of recorded CCTV footage:

```powershell
# Basic run with Pose Tracking
python -m src.video.video_processor --video videos/test_video.mp4

# Full End-to-End Pipeline with PPE Evaluation & Custom Tracker Config
python -m src.video.video_processor `
    --video videos/test_video.mp4 `
    --ppe-model models/ppe_helmet.pt `
    --tracker src/tracking/custom_bytetrack.yaml `
    --output-dir outputs
```

**Artifacts Generated**:
- `outputs/tracked_video.mp4`: High-resolution annotated video with skeleton overlays and compliance HUD.
- `outputs/tracks.jsonl`: Frame-by-frame tracking records matching the SafeOps JSONL specification.
- `outputs/report.json`: Full structured analysis report with worker summaries, compliance scores, and confirmed violations.

---

### 3.2 FastAPI REST Service (Production Web API)
To start the production HTTP service for web dashboard and backend integration:

```powershell
# Launch Uvicorn server on port 8000
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
```

> [!IMPORTANT]
> **Single Worker Process per GPU**: Always run Uvicorn with `--workers 1` when sharing an RTX 3050 6GB GPU. Starting multiple worker processes causes redundant CUDA context allocation, duplicating the ~150 MB model weights and risking VRAM thrashing. For horizontal scaling across multiple GPUs, assign distinct `CUDA_VISIBLE_DEVICES` per process.

### Production Service Verification
Verify API health via HTTP GET:
```powershell
curl http://localhost:8000/api/v1/health
```
Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
}
```

---

## 4. Production Daemon Configuration (Windows Service)

For automated restarts and background operation on Windows Server or Windows 11 edge boxes, configure the service using [NSSM (Non-Sucking Service Manager)](https://nssm.cc/):

```powershell
# Install SafeOps service
nssm install SafeOpsAPI "D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3\.venv-gpu\Scripts\python.exe" "-m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1"

# Set working directory
nssm set SafeOpsAPI AppDirectory "D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3"

# Configure automatic startup and logging
nssm set SafeOpsAPI Start SERVICE_AUTO_START
nssm set SafeOpsAPI AppStdout "D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3\outputs\service_out.log"
nssm set SafeOpsAPI AppStderr "D:\Projects\SafeOps\Video Processing and ByteTrack\SafeOps_phase3\outputs\service_err.log"

# Start the service
nssm start SafeOpsAPI
```

---

## 5. Environment Variables & Runtime Overrides

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `SAFEOPS_DEVICE` | `cuda:0` | GPU device ID or `cpu` fallback |
| `SAFEOPS_DATA_DIR` | `./outputs` | Storage directory for video uploads, reports, and clips |
| `SAFEOPS_POSE_MODEL` | `models/yolov8s-pose.pt` | Path to YOLOv8 pose estimation weights |
| `SAFEOPS_PPE_MODEL` | `models/ppe_helmet.pt` | Path to PPE detection weights |
| `SAFEOPS_CONFIRMATION_SEC` | `1.0` | Sustained non-compliance duration to confirm a violation |
| `SAFEOPS_RESOLUTION_SEC` | `0.5` | Sustained compliance duration to resolve an active violation |

---

## 6. Video Codec & Browser Streaming Considerations

OpenCV writes default MP4 files using the `mp4v` FourCC codec. While native desktop media players (VLC, Windows Media Player) support `mp4v`, modern web browsers (Chrome, Edge, Safari) require H.264 (`avc1`) video with AAC audio for inline HTML5 playback.

### Web Compatibility Strategy
1. The pipeline automatically checks for native `avc1` or OpenH264 encoding support during initialization.
2. For web dashboard deployment, an optional post-processing `ffmpeg` transcode command can be executed automatically:
```powershell
ffmpeg -y -i outputs/tracked_video.mp4 -vcodec libx264 -pix_fmt yuv420p outputs/web_tracked_video.mp4
```
The FastAPI streaming endpoint (`/api/v1/videos/{video_id}/annotated_video`) serves with `media_type="video/mp4"` and supports standard HTTP range requests for dashboard video scrub bars.

