# SafeOps REST API Contracts & Schema Specification

**API Version**: `v1`  
**Schema Version**: `1.0.0`  
**Protocol**: HTTP/1.1 (JSON + Multipart)  
**Base URL**: `http://localhost:8000`  

---

## 1. Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System diagnostics & GPU telemetry |
| `POST` | `/api/v1/videos` | Upload video file (multipart form) |
| `GET` | `/api/v1/videos` | List all registered/processed video jobs |
| `GET` | `/api/v1/videos/{video_id}` | Retrieve video job metadata & status |
| `POST` | `/api/v1/videos/{video_id}/process` | Trigger asynchronous background processing |
| `GET` | `/api/v1/videos/{video_id}/status` | Poll real-time progress percentage |
| `GET` | `/api/v1/videos/{video_id}/tracks` | Query frame-by-frame worker tracking records |
| `GET` | `/api/v1/videos/{video_id}/compliance` | Get worker compliance summaries |
| `GET` | `/api/v1/videos/{video_id}/violations` | Get confirmed safety violation events |
| `GET` | `/api/v1/videos/{video_id}/report` | Retrieve final `VideoAnalysisReport` |
| `GET` | `/api/v1/videos/{video_id}/annotated_video` | Stream/download annotated MP4 video |

---

## 2. Endpoint Details

### 2.1 System Health
`GET /health`

**Response (`200 OK`)**:
```json
{
  "status": "healthy",
  "service": "SafeOps PPE Compliance API",
  "version": "1.0.0",
  "cuda_available": true,
  "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
}
```

---

### 2.2 Upload Video
`POST /api/v1/videos`

**Request**: `multipart/form-data` with `file: UploadFile` (`.mp4`, `.avi`, `.mov`).

**Response (`200 OK`)**:
```json
{
  "video_id": "vid_a1b2c3d4e5",
  "filename": "warehouse_feed.mp4",
  "status": "pending",
  "message": "Video uploaded successfully. Ready for processing."
}
```

---

### 2.3 Trigger Background Processing
`POST /api/v1/videos/{video_id}/process`

**Request Body** (optional):
```json
{
  "require_helmet": true,
  "require_vest": false,
  "min_observations": 15,
  "violation_confirmation_seconds": 1.0,
  "recovery_confirmation_seconds": 1.0,
  "max_frames": null
}
```

**Response (`200 OK`)**:
```json
{
  "video_id": "vid_a1b2c3d4e5",
  "status": "processing",
  "message": "Background processing initiated"
}
```

---

### 2.4 Poll Processing Status
`GET /api/v1/videos/{video_id}/status`

**Response (`200 OK`)**:
```json
{
  "video_id": "vid_a1b2c3d4e5",
  "status": "processing",
  "progress_pct": 68.4,
  "error_message": null
}
```

---

### 2.5 Query Worker Tracks
`GET /api/v1/videos/{video_id}/tracks?track_id=1&limit=100&offset=0`

**Response (`200 OK`)**:
```json
{
  "video_id": "vid_a1b2c3d4e5",
  "total_matched": 848,
  "offset": 0,
  "limit": 100,
  "records": [
    {
      "frame_number": 0,
      "timestamp_seconds": 0.0,
      "track_id": 1,
      "bbox_xyxy": [470.4, 135.7, 853.3, 474.7],
      "confidence": 0.9561,
      "keypoints_xy": [[616.1, 312.8], [649.8, 290.2]],
      "keypoints_conf": [0.99, 0.992],
      "continuity_state": "confirmed",
      "ppe": {
        "helmet": false,
        "vest": false,
        "helmet_status": "not_detected",
        "vest_status": "unknown",
        "helmet_conf": null,
        "vest_conf": null
      },
      "compliance": {
        "overall_status": "insufficient_evidence",
        "active_violation": false,
        "details": "Gathering evidence (1/15 frames)"
      }
    }
  ]
}
```

---

### 2.6 Full Video Analysis Report
`GET /api/v1/videos/{video_id}/report`

**Response (`200 OK`)**:
```json
{
  "version": "1.0.0",
  "video_id": "vid_a1b2c3d4e5",
  "metadata": {
    "video_id": "vid_a1b2c3d4e5",
    "filename": "warehouse_feed.mp4",
    "fps": 30.0,
    "width": 1280,
    "height": 720,
    "total_frames": 900,
    "duration_seconds": 30.0
  },
  "policy": {
    "policy_name": "Standard Industrial Safety Policy",
    "require_helmet": true,
    "require_vest": false,
    "min_observations_for_decision": 15,
    "violation_confirmation_seconds": 1.0,
    "recovery_confirmation_seconds": 1.0
  },
  "summary": {
    "total_frames_processed": 900,
    "duration_seconds": 30.0,
    "average_fps": 28.5,
    "unique_workers_tracked": 2,
    "total_violations_detected": 1,
    "compliance_rate_percent": 50.0
  },
  "workers": [
    {
      "track_id": 1,
      "first_seen_timestamp": 0.0,
      "last_seen_timestamp": 30.0,
      "total_frames_present": 900,
      "compliance_status": "compliant",
      "violations_count": 0
    },
    {
      "track_id": 2,
      "first_seen_timestamp": 0.0,
      "last_seen_timestamp": 30.0,
      "total_frames_present": 900,
      "compliance_status": "non_compliant",
      "violations_count": 1
    }
  ],
  "violations": [
    {
      "event_id": "VIOL-4E2F1A08",
      "track_id": 2,
      "violation_type": "missing_helmet",
      "start_timestamp": 1.5,
      "end_timestamp": 18.2,
      "status": "confirmed",
      "evidence_frames": [45, 46, 47, 50, 55],
      "confidence": 1.0,
      "summary": "Worker 2 confirmed missing helmet"
    }
  ],
  "created_at": "2026-10-30T18:00:00Z"
}
```

