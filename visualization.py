"""
visualization.py — Rendering utilities for the safety monitor display.

All drawing is done on a pre-allocated output buffer to avoid per-frame
memory allocation. Supports both debug and production display modes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from config import VIS
from tracking import TrackedPerson


# Pre-load font face once
FONT = cv2.FONT_HERSHEY_DUPLEX


def _pick_bbox_color(track: TrackedPerson) -> Tuple[int, int, int]:
    """Select bounding box color based on violation state."""
    no_helmet   = track.has_helmet is False
    unknown_ppe  = track.has_helmet is None
    zone_breach = track.in_restricted_zone if getattr(config, "ENABLE_ZONE_DETECTION", False) else False
    uncertain = getattr(track, "detection_state", "SAFE") == "UNCERTAIN"

    if no_helmet and zone_breach:
        return VIS.both_violations
    if no_helmet:
        return VIS.violation_ppe
    if zone_breach:
        return VIS.violation_zone
    if uncertain:
        return (0, 215, 255)
    if unknown_ppe:
        return (160, 160, 160)
    return VIS.compliant


def draw_track(frame: np.ndarray, track: TrackedPerson) -> None:
    """
    Draw bounding box, track ID, PPE status, and zone status on frame.
    """
    x1, y1, x2, y2 = track.bbox_xyxy.astype(int)
    color = _pick_bbox_color(track)

    # ── Bounding box ──────────────────────────────────────────────────────────
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, VIS.bbox_thickness)

    # ── Header label (track ID + confidence) ─────────────────────────────────
    id_text = f"ID:{track.track_id}"
    if track.confidence > 0:
        id_text += f" {track.confidence:.2f}"
    if getattr(track, "detection_state", "SAFE") == "UNCERTAIN":
        id_text += " ?"

    (tw, th), baseline = cv2.getTextSize(id_text, FONT, VIS.font_scale, VIS.font_thickness)
    cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, id_text,
        (x1 + 2, y1 - baseline - 2),
        FONT, VIS.font_scale,
        VIS.track_id_text, VIS.font_thickness,
    )

    # ── Status labels below header ────────────────────────────────────────────
    status_lines: List[str] = []

    if track.has_helmet is True:
        ppe_label = f"HELMET ({track.ppe_confidence:.2f})" if track.ppe_confidence else "HELMET"
        status_lines.append(ppe_label)
    elif track.has_helmet is False:
        ppe_label = f"NO HELMET ({track.ppe_confidence:.2f})" if track.ppe_confidence else "NO HELMET"
        status_lines.append(ppe_label)
    else:
        status_lines.append("PPE: UNKNOWN")

    if getattr(config, "ENABLE_ZONE_DETECTION", False) and track.in_restricted_zone:
        zone_str = ",".join(str(z) for z in track.zone_ids)
        status_lines.append(f"ZONE BREACH [{zone_str}]")

    line_h = th + 6
    for i, line in enumerate(status_lines):
        text_y = y2 + line_h * (i + 1)
        # Background pill
        (lw, lh), lb = cv2.getTextSize(line, FONT, VIS.font_scale - 0.05, 1)
        if "UNKNOWN" in line:
            text_color = (180, 180, 180)
        elif "NO HELMET" in line or "ZONE" in line:
            text_color = (0, 0, 220)
        else:
            text_color = (50, 200, 50)
        bg_color   = (0, 0, 0)
        cv2.rectangle(frame, (x1, text_y - lh - 3), (x1 + lw + 4, text_y + lb), bg_color, -1)
        cv2.putText(
            frame, line,
            (x1 + 2, text_y),
            FONT, VIS.font_scale - 0.05,
            text_color, 1, cv2.LINE_AA,
        )

    # ── Foot-point indicator (only when zone detection is active) ─────────────
    if getattr(config, "ENABLE_ZONE_DETECTION", False):
        foot = track.foot_point
        foot_color = (0, 0, 220) if track.in_restricted_zone else (50, 205, 50)
        cv2.circle(frame, foot, 5, foot_color, -1)
        cv2.circle(frame, foot, 7, (255, 255, 255), 1)


def draw_fps_and_stats(
    frame: np.ndarray,
    fps: float,
    num_tracks: int,
    num_violations: int,
) -> None:
    """Draw FPS counter and summary stats in top-right corner."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {fps:.1f}",
        f"Tracks: {num_tracks}",
        f"Violations: {num_violations}",
    ]
    x_base = w - 200
    y_base = 30
    for i, line in enumerate(lines):
        color = (0, 220, 0) if fps >= 10 else (0, 100, 220)
        if "Violations" in line and num_violations > 0:
            color = (0, 0, 220)
        cv2.putText(
            frame, line,
            (x_base, y_base + i * 28),
            FONT, 0.65, (0, 0, 0), 4,   # Shadow
        )
        cv2.putText(
            frame, line,
            (x_base, y_base + i * 28),
            FONT, 0.65, color, 2,
        )


def draw_alert_banner(frame: np.ndarray, message: str) -> None:
    """Flash a red banner at the top of the frame on active violations."""
    h, w = frame.shape[:2]
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(
        frame, f"⚠ VIOLATION: {message}",
        (20, 35),
        FONT, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )


def compose_frame(
    frame: np.ndarray,
    tracks: List[TrackedPerson],
    fps: float,
    active_alerts: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Master compositor — applies all visual layers to the frame.

    Call this once per processed frame before cv2.imshow().
    """
    num_violations = sum(1 for t in tracks if t.has_violation)

    for track in tracks:
        draw_track(frame, track)

    if config.SHOW_FPS:
        draw_fps_and_stats(frame, fps, len(tracks), num_violations)

    if active_alerts:
        draw_alert_banner(frame, active_alerts[0])

    return frame


def resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Resize frame to display dimensions while preserving aspect ratio."""
    h, w = frame.shape[:2]
    target_w, target_h = config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT
    scale = min(target_w / w, target_h / h)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return frame
