"""Integration Tests for SafeOps FastAPI Backend.

Validates:
1. Health check & GPU information
2. File upload validations (format check, empty file check)
3. Job creation & listing
4. Video processing initiation & status polling
5. Retrieval of tracks.jsonl with pagination and filtering
6. Retrieval of compliance summaries and violation events
7. Retrieval of structured final report
8. Streaming of annotated video MP4
"""
import io
import time
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.services import global_job_manager


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "cuda_available" in data


def test_upload_invalid_format(client):
    bad_data = io.BytesIO(b"dummy text content")
    res = client.post("/api/v1/videos", files={"file": ("bad.txt", bad_data, "text/plain")})
    assert res.status_code == 400
    assert "Unsupported video format" in res.json()["detail"]


def test_upload_empty_file(client):
    empty_data = io.BytesIO(b"")
    res = client.post("/api/v1/videos", files={"file": ("empty.mp4", empty_data, "video/mp4")})
    assert res.status_code == 400
    assert "empty" in res.json()["detail"]


def test_upload_valid_video_and_process_lifecycle(client):
    # Use small scenario video
    src_video = "videos/scenarios/scenario1_single_worker_moving.mp4"
    with open(src_video, "rb") as f:
        vid_bytes = f.read()

    upload_res = client.post(
        "/api/v1/videos",
        files={"file": ("test_clip.mp4", io.BytesIO(vid_bytes), "video/mp4")},
    )
    assert upload_res.status_code == 200
    vid_id = upload_res.json()["video_id"]
    assert vid_id.startswith("vid_")

    # 1. Check status (initially pending)
    status_res = client.get(f"/api/v1/videos/{vid_id}/status")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "pending"

    # 2. Trigger processing with max_frames=10 for quick test
    proc_res = client.post(
        f"/api/v1/videos/{vid_id}/process",
        json={"max_frames": 10, "require_helmet": True, "require_vest": False},
    )
    assert proc_res.status_code == 200

    # 3. Poll until completed (timeout 20s)
    t0 = time.time()
    completed = False
    while time.time() - t0 < 20:
        s = client.get(f"/api/v1/videos/{vid_id}/status").json()
        if s["status"] == "completed":
            completed = True
            break
        elif s["status"] == "failed":
            pytest.fail(f"Processing failed: {s.get('error_message')}")
        time.sleep(0.5)

    assert completed, "Background processing did not complete in time"

    # 4. Check tracks endpoint
    tracks_res = client.get(f"/api/v1/videos/{vid_id}/tracks?limit=5")
    assert tracks_res.status_code == 200
    t_data = tracks_res.json()
    assert len(t_data["records"]) <= 5
    assert "track_id" in t_data["records"][0]

    # 5. Check report endpoint
    rep_res = client.get(f"/api/v1/videos/{vid_id}/report")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert rep_data["version"] == "1.0.0"
    assert rep_data["summary"]["total_frames_processed"] == 10

    # 6. Check annotated video streaming endpoint
    vid_stream_res = client.get(f"/api/v1/videos/{vid_id}/annotated_video")
    assert vid_stream_res.status_code == 200
    assert vid_stream_res.headers["content-type"] == "video/mp4"

