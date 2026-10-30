"""Asynchronous Processing Services and Job Manager for SafeOps API.

Handles:
- In-memory thread-safe video job registry
- Background video processing in a worker thread (non-blocking)
- File validation, storage, and cleanup
- Progress polling and state management
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.contracts.schemas import (
    SafetyPolicyConfig,
    VideoAnalysisReport,
    VideoMetadata,
    ViolationEvent,
    WorkerSummary,
)
from src.video.video_processor import run_pipeline
from src.video.video_stream import validate_video_file

UPLOAD_DIR = Path("outputs/uploads")
PROCESSED_DIR = Path("outputs/processed")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoJob:
    def __init__(self, video_id: str, original_filename: str, source_path: Path):
        self.video_id = video_id
        self.original_filename = original_filename
        self.source_path = source_path
        self.status = JobStatus.PENDING
        self.progress_pct: float = 0.0
        self.error_message: Optional[str] = None
        self.annotated_video_path: Optional[Path] = None
        self.tracks_jsonl_path: Optional[Path] = None
        self.report_path: Optional[Path] = None
        self.report: Optional[VideoAnalysisReport] = None


class JobManager:
    """Singleton-style manager for video processing jobs."""

    def __init__(self, max_workers: int = 1):
        # Limit GPU concurrency to 1 video at a time to prevent VRAM contention on RTX 3050 (6GB)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, VideoJob] = {}

    def create_job(self, filename: str, content_bytes: bytes) -> VideoJob:
        """Store uploaded file and register a new processing job."""
        video_id = f"vid_{uuid.uuid4().hex[:10]}"
        safe_filename = Path(filename).name
        target_path = UPLOAD_DIR / f"{video_id}_{safe_filename}"

        with open(target_path, "wb") as f:
            f.write(content_bytes)

        # Validate file
        validate_video_file(target_path)

        job = VideoJob(
            video_id=video_id,
            original_filename=safe_filename,
            source_path=target_path,
        )
        self.jobs[video_id] = job
        return job

    def register_existing_file(self, file_path: str | Path, video_id: Optional[str] = None) -> VideoJob:
        """Register a video already residing on disk."""
        path = validate_video_file(file_path)
        vid_id = video_id or f"vid_{uuid.uuid4().hex[:10]}"
        job = VideoJob(
            video_id=vid_id,
            original_filename=path.name,
            source_path=path,
        )
        self.jobs[vid_id] = job
        return job

    def get_job(self, video_id: str) -> Optional[VideoJob]:
        return self.jobs.get(video_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        return [
            {
                "video_id": j.video_id,
                "filename": j.original_filename,
                "status": j.status,
                "progress_pct": j.progress_pct,
                "has_report": j.report is not None,
            }
            for j in self.jobs.values()
        ]

    def start_processing(
        self,
        video_id: str,
        policy: Optional[SafetyPolicyConfig] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        """Queue video for background processing."""
        job = self.get_job(video_id)
        if not job:
            raise KeyError(f"Job {video_id} not found")
        if job.status == JobStatus.PROCESSING:
            return

        job.status = JobStatus.PROCESSING
        job.progress_pct = 0.0
        job.error_message = None

        job_dir = PROCESSED_DIR / video_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job.annotated_video_path = job_dir / "annotated_video.mp4"
        job.tracks_jsonl_path = job_dir / "tracks.jsonl"
        job.report_path = job_dir / "report.json"

        def _worker_task():
            try:
                def progress_cb(current, total, pct):
                    job.progress_pct = round(pct, 1)

                report = run_pipeline(
                    video_path=str(job.source_path),
                    out_video=str(job.annotated_video_path),
                    out_jsonl=str(job.tracks_jsonl_path),
                    out_report=str(job.report_path),
                    policy=policy,
                    max_frames=max_frames,
                    progress_callback=progress_cb,
                )
                job.report = report
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                print(f"ERROR processing job {video_id}: {e}")

        self.executor.submit(_worker_task)


# Global job manager instance
global_job_manager = JobManager(max_workers=1)

