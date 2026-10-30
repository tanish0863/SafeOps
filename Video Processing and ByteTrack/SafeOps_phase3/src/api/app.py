"""SafeOps FastAPI Backend Service.

Exposes RESTful endpoints for:
- Video upload & validation
- Asynchronous GPU-controlled background pipeline execution
- Live progress polling
- Worker tracking records & compliance data retrieval
- Confirmed safety violation event log
- VideoAnalysisReport generation
- Annotated video artifact streaming
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.api.services import JobStatus, global_job_manager
from src.contracts.schemas import (
    SafetyPolicyConfig,
    VideoAnalysisReport,
    ViolationEvent,
    WorkerSummary,
)

app = FastAPI(
    title="SafeOps PPE Compliance Detection API",
    version="1.0.0",
    description="Production-oriented AI surveillance backend for industrial worker PPE monitoring.",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    require_helmet: bool = True
    require_vest: bool = False
    min_observations: int = 15
    violation_confirmation_seconds: float = 1.0
    recovery_confirmation_seconds: float = 1.0
    max_frames: Optional[int] = None


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """System health check and GPU diagnostics."""
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    return {
        "status": "healthy",
        "service": "SafeOps PPE Compliance API",
        "version": "1.0.0",
        "cuda_available": cuda_avail,
        "device": device_name,
    }


@app.post("/api/v1/videos")
async def upload_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload a CCTV video file for compliance analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported video format: {suffix}")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes)")
        if len(content) > 500 * 1024 * 1024:  # 500 MB limit
            raise HTTPException(status_code=413, detail="File exceeds maximum size limit (500 MB)")

        job = global_job_manager.create_job(file.filename, content)
        return {
            "video_id": job.video_id,
            "filename": job.original_filename,
            "status": job.status,
            "message": "Video uploaded successfully. Ready for processing.",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/videos")
def list_videos() -> List[Dict[str, Any]]:
    """List all registered and processed videos."""
    return global_job_manager.list_jobs()


@app.get("/api/v1/videos/{video_id}")
def get_video_info(video_id: str) -> Dict[str, Any]:
    """Get metadata and job status for a video."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")

    return {
        "video_id": job.video_id,
        "filename": job.original_filename,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "error_message": job.error_message,
        "has_annotated_video": job.annotated_video_path is not None and job.annotated_video_path.exists(),
        "has_report": job.report is not None,
    }


@app.post("/api/v1/videos/{video_id}/process")
def process_video_endpoint(video_id: str, req: Optional[ProcessRequest] = None) -> Dict[str, Any]:
    """Trigger background pipeline processing on an uploaded video."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")

    if job.status == JobStatus.PROCESSING:
        return {"video_id": video_id, "status": job.status, "message": "Video is already processing"}

    policy = None
    max_frames = None
    if req:
        policy = SafetyPolicyConfig(
            require_helmet=req.require_helmet,
            require_vest=req.require_vest,
            min_observations_for_decision=req.min_observations,
            violation_confirmation_seconds=req.violation_confirmation_seconds,
            recovery_confirmation_seconds=req.recovery_confirmation_seconds,
        )
        max_frames = req.max_frames

    global_job_manager.start_processing(video_id, policy=policy, max_frames=max_frames)
    return {
        "video_id": video_id,
        "status": JobStatus.PROCESSING,
        "message": "Background processing initiated",
    }


@app.get("/api/v1/videos/{video_id}/status")
def get_processing_status(video_id: str) -> Dict[str, Any]:
    """Poll processing progress and status."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")

    return {
        "video_id": job.video_id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "error_message": job.error_message,
    }


@app.get("/api/v1/videos/{video_id}/tracks")
def get_video_tracks(
    video_id: str,
    track_id: Optional[int] = Query(None, description="Filter by specific worker track ID"),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """Retrieve frame-by-frame worker tracking records with optional filtering."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not job.tracks_jsonl_path or not job.tracks_jsonl_path.exists():
        raise HTTPException(status_code=400, detail="Tracks not available. Run processing first.")

    matched_records = []
    total_count = 0

    with open(job.tracks_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if track_id is not None and rec.get("track_id") != track_id:
                continue

            if total_count >= offset and len(matched_records) < limit:
                matched_records.append(rec)
            total_count += 1

    return {
        "video_id": video_id,
        "total_matched": total_count,
        "offset": offset,
        "limit": limit,
        "records": matched_records,
    }


@app.get("/api/v1/videos/{video_id}/compliance")
def get_worker_compliance(video_id: str) -> List[WorkerSummary]:
    """Retrieve per-worker compliance summaries."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not job.report:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    return job.report.workers


@app.get("/api/v1/videos/{video_id}/violations")
def get_video_violations(video_id: str) -> List[ViolationEvent]:
    """Retrieve confirmed safety violation events."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not job.report:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    return job.report.violations


@app.get("/api/v1/videos/{video_id}/report")
def get_full_report(video_id: str) -> VideoAnalysisReport:
    """Retrieve full structured analysis report."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not job.report:
        raise HTTPException(status_code=400, detail="Report not generated yet")

    return job.report


@app.get("/api/v1/videos/{video_id}/annotated_video")
def get_annotated_video(video_id: str):
    """Stream or download the annotated MP4 video."""
    job = global_job_manager.get_job(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not job.annotated_video_path or not job.annotated_video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")

    return FileResponse(
        str(job.annotated_video_path),
        media_type="video/mp4",
        filename=f"annotated_{job.original_filename}",
    )

