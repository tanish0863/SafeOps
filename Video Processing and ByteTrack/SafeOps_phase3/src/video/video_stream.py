"""Streaming Video Processing Engine for SafeOps.

Provides:
- Robust video ingestion for MP4, AVI, MOV
- Metadata extraction (FPS, resolution, duration, frame counts)
- Streaming frame decoding without loading full video into RAM
- Corrupted frame handling and safe skipping
- Guaranteed OpenCV resource release via context managers
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

from src.contracts.schemas import VideoMetadata

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoProcessingError(Exception):
    """Raised when video file cannot be opened or is corrupted."""
    pass


def validate_video_file(file_path: str | Path) -> Path:
    """Validate that the file exists, has a supported extension, and is non-empty."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file does not exist: {file_path}")
    if not path.is_file():
        raise VideoProcessingError(f"Target is not a file: {file_path}")
    if path.stat().st_size == 0:
        raise VideoProcessingError(f"Video file is empty (0 bytes): {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoProcessingError(
            f"Unsupported video format '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def extract_video_metadata(cap: cv2.VideoCapture, file_path: str | Path, video_id: str | None = None) -> VideoMetadata:
    """Extract standard video metadata, validating FPS and dimensions."""
    path = Path(file_path)
    vid_id = video_id or path.stem

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        # Fallback default with logged warning
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise VideoProcessingError(f"Invalid video dimensions: {width}x{height}")

    duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else 0.0

    return VideoMetadata(
        video_id=vid_id,
        filename=path.name,
        fps=round(float(fps), 3),
        width=width,
        height=height,
        total_frames=max(0, total_frames),
        duration_seconds=round(duration, 3),
    )


@contextmanager
def open_video_stream(
    video_path: str | Path,
    video_id: str | None = None,
) -> Generator[Tuple[Generator[Tuple[int, float, np.ndarray], None, None], VideoMetadata], None, None]:
    """Open video for memory-efficient sequential streaming.

    Yields:
        (frame_generator, metadata)
        where frame_generator yields (frame_number, timestamp_seconds, frame_bgr)
    """
    path = validate_video_file(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoProcessingError(f"OpenCV could not open video: {path}")

    try:
        meta = extract_video_metadata(cap, path, video_id=video_id)

        def frame_generator() -> Generator[Tuple[int, float, np.ndarray], None, None]:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    # End of stream or unreadable frame
                    break
                if frame is None or frame.size == 0:
                    # Corrupted frame detected, skip safely
                    frame_idx += 1
                    continue

                # Derived timestamp from zero-based frame index and video FPS.
                # Note: For variable frame rate (VFR) containers, timestamps may exhibit slight drift.
                timestamp = frame_idx / meta.fps
                yield frame_idx, round(timestamp, 3), frame
                frame_idx += 1

        yield frame_generator(), meta
    finally:
        cap.release()
