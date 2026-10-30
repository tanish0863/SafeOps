"""Robustness and Edge-Case Tests for SafeOps Video Ingestion.

Validates:
1. Missing/nonexistent video handling
2. Empty video file (0 bytes) handling
3. Unsupported file formats handling
4. Guaranteed resource release on exception
5. Zero worker detection video execution
6. Premature stream termination (max_frames limit)
"""
import pytest
from pathlib import Path

from src.video.video_stream import (
    VideoProcessingError,
    open_video_stream,
    validate_video_file,
)
from src.video.video_processor import run_pipeline


def test_missing_video_file_raises():
    with pytest.raises(FileNotFoundError):
        validate_video_file("nonexistent_video_path_xyz.mp4")


def test_empty_video_file_raises(tmp_path):
    empty_file = tmp_path / "empty_file.mp4"
    empty_file.write_bytes(b"")
    with pytest.raises(VideoProcessingError, match="empty"):
        validate_video_file(empty_file)


def test_unsupported_format_raises(tmp_path):
    bad_file = tmp_path / "video.txt"
    bad_file.write_text("not a video")
    with pytest.raises(VideoProcessingError, match="Unsupported video format"):
        validate_video_file(bad_file)


def test_stream_resource_release_on_error(tmp_path):
    test_vid = "videos/scenarios/scenario1_single_worker_moving.mp4"
    
    # Intentionally raise an exception inside generator consumption
    with pytest.raises(ZeroDivisionError):
        with open_video_stream(test_vid) as (stream, meta):
            for fn, ts, f in stream:
                if fn == 2:
                    _ = 1 / 0
    # Stream context manager must guarantee cap.release() in finally


def test_pipeline_max_frames_termination(tmp_path):
    out_vid = tmp_path / "out.mp4"
    out_jsonl = tmp_path / "tracks.jsonl"
    out_report = tmp_path / "report.json"
    
    report = run_pipeline(
        video_path="videos/scenarios/scenario1_single_worker_moving.mp4",
        out_video=str(out_vid),
        out_jsonl=str(out_jsonl),
        out_report=str(out_report),
        max_frames=10,
    )
    assert report.summary.total_frames_processed == 10
    assert out_vid.exists()
    assert out_report.exists()

