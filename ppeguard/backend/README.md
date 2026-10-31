# SafeOps Backend & Frontend Integration

This repository hosts the backend service and full-stack integration for the **SafeOps Security Suite**, an industrial computer-vision monitoring platform for PPE compliance tracking (helmets, high-visibility vests).

---

## 1. Architecture Overview

```
Frontend UI (React + TanStack Router + Tailwind CSS)
                       ↕ (HTTP / REST APIs over JSON & Multipart)
API Layer: FastAPI (prefix: /api/v1)
                       ↕
Business Logic & Routers (Dashboard, Detection, Workers, Violations, Analytics, Settings, Media)
                       ↕
Storage Layer: SQLAlchemy 2.0 ORM  +  Local Disk Storage
  ├── Database: SQLite (`safeops.db`) / PostgreSQL ready
  └── Uploads: `./uploads/` (images, videos, metadata)
                       ↕
Detection Integration Boundary (Deterministic PPE Rule Engine / No ML inference)
```

---

## 2. Technology Stack

- **Backend Framework**: Python 3.12 / 3.15, FastAPI, Starlette
- **Data Validation & Schemas**: Pydantic v2
- **ORM & Database**: SQLAlchemy 2.0, SQLite (`safeops.db`), extensible to PostgreSQL via `DATABASE_URL`
- **Testing**: Pytest, HTTPX, FastAPI TestClient
- **Frontend**: React 18, TanStack Router & TanStack Query, Vite / Bun runtime

---

## 3. Strict ML Integration Boundary (No ML Development)

As strictly stipulated, **no ML model development, training, weight loading, YOLO algorithms, or computer vision model modifications were performed**.
- The backend provides a decoupled ML interface for detection job submission, media storage, and result retrieval (`POST /api/v1/detection/jobs`, `POST /api/v1/detection/analyze`).
- Detection responses clearly identify demo/fallback evaluation with `isDemo: true` and `model: "SafeOps-Vision-v2.1 (Integration Boundary Demo)"`.
- The backend has zero dependencies on PyTorch, TensorFlow, Ultralytics, OpenCV, or GPU runtime to boot and serve all REST APIs.

---

## 4. API Endpoints (`/api/v1`)

| Method | Endpoint | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | Health & DB connectivity check | None | `{"status": "healthy", ...}` |
| `GET` | `/dashboard/stats` | Dashboard statistics & KPIs | None | `DashboardStatsResponse` |
| `GET` | `/dashboard/activity` | Recent audit events & activity log | None | `List[ActivityItemResponse]` |
| `GET` | `/dashboard/telemetry` | Worker count & compliance rate | None | `{"complianceRate": 85.5, ...}` |
| `POST` | `/dashboard/simulate-scan`| Trigger new simulated detection scan | None | `{"success": true, ...}` |
| `POST` | `/media/image` | Upload image for inspection | Multipart `file` | `MediaUploadResponse` |
| `POST` | `/media/video` | Upload video for inspection | Multipart `file` | `MediaUploadResponse` |
| `POST` | `/media/upload` | Generic media upload | Multipart `file` | `MediaUploadResponse` |
| `POST` | `/detection/jobs` | Create asynchronous detection job | `DetectionJobCreate` | `DetectionJobResponse` |
| `GET` | `/detection/jobs/{id}/result`| Retrieve detection job results | None | `WorkspaceAnalysisResponse` |
| `POST` | `/detection/analyze` | Synchronous frame/video analysis | `AnalysisRequest` | `WorkspaceAnalysisResponse` |
| `GET` | `/detection/history` | List past detection jobs | None | `List[DetectionJobResponse]` |
| `GET` | `/workers` | Filter & retrieve worker roster | Query: `search`, `status`, `shift` | `List[WorkerResponse]` |
| `POST` | `/workers` | Register new worker | `WorkerCreate` | `WorkerResponse` (201) |
| `GET` | `/workers/{id}` | Retrieve individual worker details | None | `WorkerResponse` |
| `PATCH`| `/workers/{id}` | Update worker information | `WorkerUpdate` | `WorkerResponse` |
| `DELETE`|`/workers/{id}` | Remove a worker | None | `204 No Content` |
| `GET` | `/violations` | Retrieve recorded violations | Query: `severity`, `reviewStatus` | `List[ViolationResponse]` |
| `GET` | `/violations/{id}` | Get specific violation record | None | `ViolationResponse` |
| `POST` | `/violations` | Record a new violation incident | `ViolationCreate` | `ViolationResponse` (201) |
| `PATCH`| `/violations/{id}` | Update violation status/notes | `ViolationUpdate` | `ViolationResponse` |
| `POST` | `/violations/{id}/resolve`| Mark violation as resolved | None | `ViolationResponse` |
| `GET` | `/analytics/compliance-trend`| 12-week compliance trend data | None | `List[TrendPoint]` |
| `GET` | `/analytics/zone-distribution`| Zone-wise violation & compliance stats | None | `List[ZoneStat]` |
| `GET` | `/analytics/ppe-breakdown` | Missing PPE category breakdown | None | `List[PpeBreakdownItem]` |
| `GET` | `/analytics/all` | Combined analytics payload | None | `AnalyticsAllResponse` |
| `GET` | `/settings` | Retrieve active system configuration | None | `SystemSettingsResponse` |
| `PUT` | `/settings` | Update system configuration | `SystemSettingsUpdate` | `SystemSettingsResponse` |
| `GET` | `/cameras` | List connected CCTV/camera streams | None | `List[CameraDeviceResponse]` |
| `POST` | `/cameras` | Register new camera stream | `CameraDeviceCreate` | `CameraDeviceResponse` |
| `DELETE`|`/cameras/{id}` | Deregister camera stream | None | `204 No Content` |

---

## 5. Database Schema & Persistence

SQLite database file: `safeops.db` located at workspace root. Automatically initialized and seeded on startup.

### Tables
1. **`workers`**:
   - `id` (VARCHAR PK, e.g. `W-101`), `name`, `zone`, `shift`, `status` (`compliant`, `violation`, `review`), `helmet` (BOOL), `vest` (BOOL), `needs_review` (BOOL), `scans` (INT), `last_scan` (VARCHAR), `created_at`, `updated_at`.
2. **`violations`**:
   - `id` (VARCHAR PK, e.g. `V-1018`), `worker_id` (FK -> `workers.id`), `worker_name`, `zone`, `timestamp`, `missing_ppe` (JSON list), `severity` (`critical`, `warning`, `info`), `review_status` (`open`, `resolved`, `investigating`), `notes`, `created_at`, `updated_at`.
3. **`media_files`**:
   - `id` (VARCHAR PK), `file_name`, `media_type` (`image`, `video`), `file_path`, `file_size_bytes`, `created_at`.
4. **`detection_jobs`**:
   - `id` (VARCHAR PK, e.g. `job_xxx`), `media_id` (FK -> `media_files.id`), `media_name`, `status` (`queued`, `processing`, `complete`, `failed`), `rules_config` (JSON), `result_payload` (JSON), `created_at`, `completed_at`.
5. **`system_settings`**:
   - `id` (INT PK), `person_confidence_threshold` (FLOAT), `ppe_confidence_threshold` (FLOAT), `torso_overlap_threshold` (FLOAT), `alert_critical_violations` (BOOL), `alert_review_required` (BOOL), `enable_audio_alerts` (BOOL), `retention_days` (INT), `webhook_url` (VARCHAR), `notification_email` (VARCHAR), `updated_at`.
6. **`activity_events`**:
   - `id` (INT PK), `action`, `actor`, `zone`, `detail`, `timestamp`.
7. **`camera_devices`**:
   - `id` (VARCHAR PK), `name`, `zone`, `stream_url`, `is_active`, `created_at`.

---

## 6. Execution Instructions (Windows PowerShell)

### Prerequisites
- Python 3.12+ installed
- Bun or Node.js 18+ installed

### Step 1: Start the Backend Server
```powershell
# Navigate to the project root
cd e:\PROJECTS\safeops\frontend\safeops-security-suite-main

# Start FastAPI using the virtual environment python
& ../../.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Backend API will be live at: `http://127.0.0.1:8000`
Swagger UI / OpenAPI docs: `http://127.0.0.1:8000/docs`

### Step 2: Start the Frontend Application
```powershell
# In a separate PowerShell terminal:
cd e:\PROJECTS\safeops\frontend\safeops-security-suite-main

# Run development server
bun run dev
```
Frontend web application will be live at: `http://localhost:8080` (or `http://localhost:5173`)

### Step 3: Run Backend Tests
```powershell
# Run the Pytest test suite
& ../../.venv/Scripts/python -m pytest backend/tests/test_api.py -v

# Run the live endpoint verification script
& ../../.venv/Scripts/python backend/verify_live.py
```

### Step 4: Run Frontend Typecheck & Build
```powershell
bun run build
```
