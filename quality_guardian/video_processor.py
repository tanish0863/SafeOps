"""
video_processor.py
------------------
Video and webcam quality analysis pipeline.

Supports:
    - Recorded video files (.mp4, .avi)
    - Live webcam input (device index 0 by default)

For video files:
    - Reads frames at a configurable sampling interval.
    - Analyses each sampled frame with the quality assessment engine.
    - Detects potentially frozen/blocked views by comparing consecutive frames.
    - Generates a JSON report with frame-wise results and a summary.
    - Records timestamps of problematic frames.

Resources:
    - The VideoCapture object is always released after processing,
      even when an error occurs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .assessment import AssessmentConfig, AssessmentResult, assess_frame, status_emoji
from .obstruction import check_frame_similarity, ObstructionStatus
from .report_generator import (
    generate_video_json_report,
    save_annotated_image,
    _safe_serialise,
)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi"}


# ---------------------------------------------------------------------------
# Low-level frame iterator
# ---------------------------------------------------------------------------

def _iter_frames(
    cap: cv2.VideoCapture,
    sample_interval: int = 30,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """
    Yield (frame_index, timestamp_seconds, frame_bgr) for sampled frames.

    Parameters
    ----------
    cap : cv2.VideoCapture
        An already-opened VideoCapture object.
    sample_interval : int
        Analyse every Nth frame (default 30).  Set to 1 to analyse every frame.
    """
    frame_index = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0  # fallback if FPS is unknown

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % sample_interval == 0:
            timestamp_sec = frame_index / fps
            yield frame_index, timestamp_sec, frame

        frame_index += 1


# ---------------------------------------------------------------------------
# Video file processing
# ---------------------------------------------------------------------------

def process_video(
    video_path: str | Path,
    config: Optional[AssessmentConfig] = None,
    sample_interval: int = 30,
    save_report: bool = True,
    save_annotated_frames: bool = False,   # annotated frames use disk space
    max_annotated_frames: int = 5,         # cap on saved annotated frames
) -> Tuple[list[dict], dict, Optional[Path]]:
    """
    Analyse a recorded video file for camera quality issues.

    Parameters
    ----------
    video_path : str | Path
        Path to the video file (.mp4 or .avi).
    config : AssessmentConfig, optional
        Custom thresholds and penalty configuration.
    sample_interval : int
        Analyse every Nth frame (default 30 ≈ every second at 30 fps).
    save_report : bool
        Save a JSON report to reports/ (default True).
    save_annotated_frames : bool
        Save annotated images for problematic frames (default False).
    max_annotated_frames : int
        Maximum number of annotated frames to save (default 5).

    Returns
    -------
    (frame_results, summary, json_report_path | None)
        frame_results : list of per-frame result dicts
        summary       : aggregated quality summary
        json_report_path : path to the saved report, or None
    """
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{path.suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(
            f"Could not open video file: {path}. "
            "The file may be corrupted or an unsupported codec is required."
        )

    try:
        return _analyse_capture(
            cap=cap,
            source_name=path.name,
            config=config,
            sample_interval=sample_interval,
            save_report=save_report,
            save_annotated_frames=save_annotated_frames,
            max_annotated_frames=max_annotated_frames,
            is_webcam=False,
        )
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Webcam (live) processing
# ---------------------------------------------------------------------------

def process_webcam(
    device_index: int = 0,
    config: Optional[AssessmentConfig] = None,
    max_frames: int = 150,         # number of frames to capture
    sample_interval: int = 15,
    save_report: bool = True,
) -> Tuple[list[dict], dict, Optional[Path]]:
    """
    Capture frames from a webcam and analyse them for quality issues.

    Parameters
    ----------
    device_index : int
        OpenCV device index (0 = first camera, default).
    config : AssessmentConfig, optional
        Custom threshold configuration.
    max_frames : int
        Total number of raw frames to read before stopping (default 150).
    sample_interval : int
        Analyse every Nth frame (default 15).
    save_report : bool
        Save a JSON report to reports/ (default True).

    Returns
    -------
    (frame_results, summary, json_report_path | None)
    """
    cap = cv2.VideoCapture(device_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at device index {device_index}. "
            "Ensure the webcam is connected and not in use by another application."
        )

    try:
        # Limit the number of frames read
        original_iter = _iter_frames
        results, summary, report_path = _analyse_capture(
            cap=cap,
            source_name=f"webcam_{device_index}",
            config=config,
            sample_interval=sample_interval,
            save_report=save_report,
            save_annotated_frames=False,
            max_annotated_frames=0,
            is_webcam=True,
            max_frames=max_frames,
        )
        return results, summary, report_path
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Shared analysis loop
# ---------------------------------------------------------------------------

def _analyse_capture(
    cap: cv2.VideoCapture,
    source_name: str,
    config: Optional[AssessmentConfig],
    sample_interval: int,
    save_report: bool,
    save_annotated_frames: bool,
    max_annotated_frames: int,
    is_webcam: bool,
    max_frames: Optional[int] = None,
) -> Tuple[list[dict], dict, Optional[Path]]:
    """Internal shared analysis loop for video files and webcam captures."""

    if config is None:
        config = AssessmentConfig()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_property = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_info = {
        "fps": round(fps, 2),
        "total_frames_in_source": total_frames_property if not is_webcam else "live",
        "width": vid_width,
        "height": vid_height,
        "sample_interval": sample_interval,
        "is_webcam": is_webcam,
    }

    ts = datetime.now()
    frame_results: list[dict] = []
    problematic_timestamps: list[float] = []

    scores: list[float] = []
    statuses: list[str] = []
    issue_counts: list[int] = []

    prev_frame: Optional[np.ndarray] = None
    annotated_saved = 0
    frames_read = 0

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_read += 1
        if max_frames and frames_read > max_frames:
            break

        timestamp_sec = frame_index / fps

        if frame_index % sample_interval == 0:
            # --- Quality assessment ---
            result = assess_frame(frame, config=config)

            # --- Frame similarity (obstruction via motion) ---
            similarity_info: Optional[dict] = None
            if prev_frame is not None:
                sim_result = check_frame_similarity(frame, prev_frame)
                similarity_info = {
                    "status": sim_result.status.value,
                    "confidence": sim_result.confidence,
                    "details": _safe_serialise(sim_result.details),
                    "issues": sim_result.issues,
                }

            prev_frame = frame.copy()

            is_problematic = result.overall_status in ("WARNING", "POOR")
            if is_problematic:
                problematic_timestamps.append(round(timestamp_sec, 2))

            # --- Optionally save annotated frame ---
            annotated_path_str: Optional[str] = None
            if save_annotated_frames and is_problematic and annotated_saved < max_annotated_frames:
                ap = save_annotated_image(
                    image_bgr=frame,
                    result=result,
                    source_name=f"{Path(source_name).stem}_f{frame_index}",
                    timestamp=ts,
                )
                annotated_path_str = str(ap)
                annotated_saved += 1

            # Serialise frame result
            frame_record: dict = {
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp_sec, 3),
                "overall_score": result.overall_score,
                "overall_status": result.overall_status,
                "issues": result.all_issues,
                "checks": {
                    name: {
                        "passed": cr.passed,
                        "value": cr.value,
                    }
                    for name, cr in result.checks.items()
                },
                "obstruction": {
                    "status": result.obstruction.status.value,
                    "confidence": result.obstruction.confidence,
                },
                "frame_similarity": similarity_info,
                "annotated_image": annotated_path_str,
            }

            frame_results.append(frame_record)
            scores.append(result.overall_score)
            statuses.append(result.overall_status)
            issue_counts.append(len(result.all_issues))

        frame_index += 1

    # --- Summary ---
    if scores:
        avg_score = round(float(np.mean(scores)), 1)
        min_score = round(float(np.min(scores)), 1)
        good_count = statuses.count("GOOD")
        warning_count = statuses.count("WARNING")
        poor_count = statuses.count("POOR")
        total_analysed = len(scores)

        # Overall video status = worst that appeared in > 20% of frames
        if poor_count / total_analysed >= 0.20:
            overall_video_status = "POOR"
        elif warning_count / total_analysed >= 0.20:
            overall_video_status = "WARNING"
        else:
            overall_video_status = "GOOD"
    else:
        avg_score = 0.0
        min_score = 0.0
        good_count = warning_count = poor_count = 0
        total_analysed = 0
        overall_video_status = "POOR"

    summary = {
        "frames_analysed": total_analysed,
        "average_score": avg_score,
        "minimum_score": min_score,
        "overall_video_status": overall_video_status,
        "frames_good": good_count,
        "frames_warning": warning_count,
        "frames_poor": poor_count,
        "problematic_timestamps_sec": problematic_timestamps,
        "disclaimer": (
            "Video quality status is determined by the fraction of frames "
            "in each category.  This is a heuristic estimate."
        ),
    }

    # --- Save report ---
    report_path: Optional[Path] = None
    if save_report:
        report_path = generate_video_json_report(
            frame_results=frame_results,
            summary=summary,
            source_name=source_name,
            video_info=video_info,
            timestamp=ts,
        )

    return frame_results, summary, report_path


def print_video_summary(summary: dict, source: str = "") -> None:
    """Print a human-readable video quality summary to stdout."""
    divider = "-" * 60
    status = summary.get("overall_video_status", "?")
    print(f"\n{divider}")
    print(f"  Video Quality Summary")
    if source:
        print(f"  Source : {source}")
    print(divider)
    print(f"  Overall Status    : {status}")
    print(f"  Average Score     : {summary.get('average_score', 0):.1f} / 100")
    print(f"  Minimum Score     : {summary.get('minimum_score', 0):.1f} / 100")
    print(f"  Frames Analysed   : {summary.get('frames_analysed', 0)}")
    print(f"  Frames GOOD       : {summary.get('frames_good', 0)}")
    print(f"  Frames WARNING    : {summary.get('frames_warning', 0)}")
    print(f"  Frames POOR       : {summary.get('frames_poor', 0)}")
    prob_ts = summary.get("problematic_timestamps_sec", [])
    if prob_ts:
        formatted = ", ".join(f"{t:.1f}s" for t in prob_ts[:10])
        suffix = f" ... (+{len(prob_ts)-10} more)" if len(prob_ts) > 10 else ""
        print(f"  Problematic Times : {formatted}{suffix}")
    print(divider + "\n")
