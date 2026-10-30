"""
report_generator.py
-------------------
Serialises AssessmentResult objects to JSON and saves annotated images.

All output goes into the ``reports/`` directory (relative to the project
root).  Original input files are never modified.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .assessment import AssessmentResult
from .obstruction import ObstructionStatus

# Root-relative reports directory (resolved at runtime so no hardcoded paths)
_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _ensure_reports_dir() -> Path:
    """Create the reports directory if it does not already exist."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return _REPORTS_DIR


def _safe_serialise(obj):
    """
    Recursively convert dataclass instances and non-serialisable types
    to plain Python dicts / primitives for JSON serialisation.

    NumPy scalar types handled explicitly:
        np.bool_    → bool    (numpy booleans are NOT Python bools)
        np.integer  → int
        np.floating → float
        np.ndarray  → list (via .tolist())
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _safe_serialise(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _safe_serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(i) for i in obj]
    # numpy.bool_ must be checked BEFORE numpy.integer because on some
    # NumPy versions bool_ is a subclass of integer.
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def generate_json_report(
    result: AssessmentResult,
    source_name: str,
    image_shape: tuple,                   # (height, width[, channels])
    timestamp: Optional[datetime] = None,
    extra_fields: Optional[dict] = None,
    output_filename: Optional[str] = None,
) -> Path:
    """
    Save an AssessmentResult as a JSON report file.

    Parameters
    ----------
    result : AssessmentResult
        The quality assessment to serialise.
    source_name : str
        Original filename or source description (e.g. "webcam_0").
    image_shape : tuple
        Shape of the analysed image/frame (height, width[, channels]).
    timestamp : datetime, optional
        Processing timestamp.  Defaults to the current local time.
    extra_fields : dict, optional
        Additional key-value pairs to include at the top level of the report.
    output_filename : str, optional
        Base filename for the JSON file.  Auto-generated when None.

    Returns
    -------
    Path
        Absolute path to the saved JSON file.
    """
    reports_dir = _ensure_reports_dir()
    ts = timestamp or datetime.now()

    height = image_shape[0]
    width = image_shape[1]

    report: dict = {
        "source": source_name,
        "processed_at": ts.isoformat(),
        "image_width": width,
        "image_height": height,
        "overall_score": result.overall_score,
        "overall_status": result.overall_status,
        "disclaimer": result.disclaimer,
        "issues": result.all_issues,
        "recommendations": result.all_recommendations,
        "score_breakdown": _safe_serialise(result.score_breakdown),
        "checks": {
            name: {
                "passed": cr.passed,
                "uncertain": getattr(cr, "uncertain", False),
                "value": cr.value,
                "threshold": cr.threshold,
                "details": _safe_serialise(cr.details),
                "issues": cr.issues,
                "recommendations": cr.recommendations,
            }
            for name, cr in result.checks.items()
        },
        "obstruction": {
            "status": result.obstruction.status.value,
            "confidence": result.obstruction.confidence,
            "details": _safe_serialise(result.obstruction.details),
            "issues": result.obstruction.issues,
            "recommendations": result.obstruction.recommendations,
            "note": result.obstruction.note,
        },
    }

    if extra_fields:
        report.update(extra_fields)

    if output_filename is None:
        safe_source = Path(source_name).stem.replace(" ", "_")
        output_filename = f"report_{safe_source}_{ts.strftime('%Y%m%d_%H%M%S')}.json"

    output_path = reports_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    return output_path


def generate_video_json_report(
    frame_results: list[dict],
    summary: dict,
    source_name: str,
    video_info: dict,
    timestamp: Optional[datetime] = None,
    output_filename: Optional[str] = None,
) -> Path:
    """
    Save a combined video quality report (frame-wise + summary) as JSON.

    Parameters
    ----------
    frame_results : list[dict]
        List of per-frame serialisable result dicts.
    summary : dict
        Aggregated video quality summary.
    source_name : str
        Original filename or source description.
    video_info : dict
        Video metadata (fps, total_frames, width, height, etc.).
    """
    reports_dir = _ensure_reports_dir()
    ts = timestamp or datetime.now()

    report = {
        "source": source_name,
        "processed_at": ts.isoformat(),
        "video_info": video_info,
        "summary": summary,
        "frames": frame_results,
    }

    if output_filename is None:
        safe_source = Path(source_name).stem.replace(" ", "_")
        output_filename = f"video_report_{safe_source}_{ts.strftime('%Y%m%d_%H%M%S')}.json"

    output_path = reports_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    return output_path


# ---------------------------------------------------------------------------
# Annotated image output
# ---------------------------------------------------------------------------

def create_annotated_image(
    image_bgr: np.ndarray,
    result: AssessmentResult,
) -> np.ndarray:
    """
    Draw quality assessment results onto a copy of the image.

    Highlights:
    - Coloured status banner at the top (green / orange / red).
    - Score and status text.
    - Glare region overlay (semi-transparent yellow mask).
    - Dark region overlay (semi-transparent blue mask).
    - Individual check badges along the bottom.

    The original image is never modified.

    Returns
    -------
    np.ndarray
        Annotated BGR image array.
    """
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    # --- Colour scheme ---
    STATUS_COLOURS = {
        "GOOD":    (34, 197, 94),    # green  (BGR)
        "WARNING": (34, 165, 255),   # orange (BGR)
        "POOR":    (60,  60, 220),   # red    (BGR)
    }
    colour = STATUS_COLOURS.get(result.overall_status, (128, 128, 128))

    # --- Glare overlay (yellowish semi-transparent mask) ---
    glare_check = result.checks.get("glare")
    if glare_check and not glare_check.passed:
        import cv2 as _cv2
        hsv = _cv2.cvtColor(image_bgr, _cv2.COLOR_BGR2HSV)
        _, sat, val = _cv2.split(hsv)
        bv = glare_check.details.get("bright_value_threshold", 230)
        bs = glare_check.details.get("bright_saturation_threshold", 30)
        glare_mask = ((val >= bv) & (sat <= bs)).astype(np.uint8) * 255
        overlay = annotated.copy()
        overlay[glare_mask > 0] = (0, 255, 255)  # yellow
        cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

    # --- Dark region overlay (blue-ish semi-transparent mask) ---
    darkness_check = result.checks.get("darkness")
    if darkness_check and not darkness_check.passed:
        gray_temp = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        dpv = darkness_check.details.get("dark_pixel_value_threshold", 50)
        dark_mask = (gray_temp < dpv).astype(np.uint8) * 255
        overlay = annotated.copy()
        overlay[dark_mask > 0] = (180, 60, 30)  # dark blue
        cv2.addWeighted(overlay, 0.30, annotated, 0.70, 0, annotated)

    # --- Status banner ---
    banner_height = max(50, h // 12)
    cv2.rectangle(annotated, (0, 0), (w, banner_height), colour, -1)

    score_text = f"Quality Score: {result.overall_score:.0f}/100  [{result.overall_status}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 800.0
    font_scale = max(0.5, min(font_scale, 1.2))
    thickness = max(1, int(font_scale * 2))

    text_size = cv2.getTextSize(score_text, font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2
    cv2.putText(annotated, score_text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # --- Check badges along the bottom ---
    badge_labels = []
    for check_name, check_result in result.checks.items():
        if check_result.passed:
            icon = "✓"
        elif getattr(check_result, "uncertain", False):
            icon = "?"
        else:
            icon = "✗"
        badge_labels.append(f"{icon} {check_name.title()}")

    # Obstruction
    obs_status = result.obstruction.status.value
    obs_icon = {"CLEAR": "✓", "UNCERTAIN": "?", "OBSTRUCTED": "✗"}.get(obs_status, "?")
    badge_labels.append(f"{obs_icon} Obstruction")

    badge_y_start = h - max(40, h // 12)
    cv2.rectangle(annotated, (0, badge_y_start), (w, h), (30, 30, 30), -1)

    badge_font_scale = font_scale * 0.65
    badge_thickness = max(1, thickness - 1)
    badge_x = 10
    for label in badge_labels:
        badge_colour = (100, 220, 100) if label.startswith("✓") else (
            (200, 200, 80) if label.startswith("?") else (80, 80, 220)
        )
        cv2.putText(annotated, label, (badge_x, badge_y_start + 28),
                    font, badge_font_scale, badge_colour, badge_thickness, cv2.LINE_AA)
        text_w = cv2.getTextSize(label, font, badge_font_scale, badge_thickness)[0][0]
        badge_x += text_w + 20

    return annotated


def save_annotated_image(
    image_bgr: np.ndarray,
    result: AssessmentResult,
    source_name: str,
    timestamp: Optional[datetime] = None,
    output_filename: Optional[str] = None,
) -> Path:
    """
    Draw quality assessment results onto a copy of the image and save it.

    Returns
    -------
    Path
        Absolute path to the saved annotated image.
    """
    reports_dir = _ensure_reports_dir()
    ts = timestamp or datetime.now()

    annotated = create_annotated_image(image_bgr, result)

    if output_filename is None:
        safe_source = Path(source_name).stem.replace(" ", "_")
        output_filename = f"annotated_{safe_source}_{ts.strftime('%Y%m%d_%H%M%S')}.jpg"

    output_path = reports_dir / output_filename
    cv2.imwrite(str(output_path), annotated)
    return output_path
