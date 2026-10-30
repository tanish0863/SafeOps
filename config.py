"""
config.py — Centralized configuration for the Industrial Safety Vision System.
All hyperparameters, thresholds, model paths, and zone definitions live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


# ─────────────────────────────────────────────
# MODEL PATHS
# ─────────────────────────────────────────────
PERSON_MODEL_PATH = "yolov8s.pt"  # Prefer stronger model for better live detection accuracy
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_ppe_model_path() -> str:
    """
    Resolve the PPE model path in training-first order.

    Search order:
      1. PPE_MODEL_PATH environment variable
      2. Ultralytics training output: runs/detect/train/weights/best.pt
      3. Legacy fallback: models/ppe_helmet.pt

    Returns the first existing path when available, otherwise the preferred
    training output path so the failure mode is explicit.
    """
    env_path = os.getenv("PPE_MODEL_PATH")
    candidates = []

    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt",
        PROJECT_ROOT / "models" / "ppe_helmet.pt",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0] if env_path else candidates[0])


def resolve_person_model_path() -> str:
    """
    Resolve the person detector path with an accuracy-first preference.

    Search order:
      1. PERSON_MODEL_PATH environment variable
      2. yolov8s.pt for better recall on smaller or farther people
      3. yolov8n.pt as a lightweight fallback
    """
    env_path = os.getenv("PERSON_MODEL_PATH")
    candidates = []

    if env_path:
        candidates.append(Path(env_path))

    candidates.extend([
        PROJECT_ROOT / "yolov8s.pt",
        PROJECT_ROOT / "yolov8n.pt",
    ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0] if env_path else candidates[0])


# Optional runtime overrides:
#   Windows PowerShell: $env:PPE_MODEL_PATH = "D:\\path\\to\\best.pt"
#   Windows PowerShell: $env:PERSON_MODEL_PATH = "D:\\path\\to\\yolov8s.pt"
PPE_MODEL_PATH = resolve_ppe_model_path()
PERSON_MODEL_PATH = resolve_person_model_path()

# ─────────────────────────────────────────────
# DETECTION THRESHOLDS
# ─────────────────────────────────────────────
PERSON_CONF_THRESHOLD = 0.52   # Slightly lower confidence to keep real people visible
PERSON_IOU_THRESHOLD  = 0.60   # Higher NMS tolerance to keep nearby people separate
PPE_CONF_THRESHOLD    = 0.10   # Lower threshold so real helmet/no-helmet detections are not dropped
PPE_IOU_THRESHOLD     = 0.45

PERSON_CLASS_ID       = 0      # COCO class ID for "person"

# ─────────────────────────────────────────────
# TRACKING (DeepSORT)
# ─────────────────────────────────────────────
DEEPSORT_MAX_AGE        = 30   # Frames to keep a lost track alive
DEEPSORT_N_INIT         = 3    # Detections needed to confirm a new track
DEEPSORT_MAX_COSINE_DIST = 0.4 # ReID embedding cosine distance threshold
DEEPSORT_NN_BUDGET      = 100  # Max stored appearance descriptors per track
DEEPSORT_MAX_IOU_DIST   = 0.7  # IOU distance gate

# ─────────────────────────────────────────────
# FRAME PIPELINE
# ─────────────────────────────────────────────
INFERENCE_INPUT_SIZE  = 1024   # Better detail for realtime webcam detection
FRAME_SKIP            = 1      # Live mode already runs every frame; keep CLI/live consistent
PPE_CROP_PADDING      = 0.15   # Slightly larger context helps helmet classification
MIN_PERSON_HEIGHT_PX  = 44     # Slightly stricter size floor to keep noise down
PERSON_MIN_BOTTOM_FRAC = 0.30  # Kept for compatibility; not used by current filters
PERSON_MIN_ASPECT_RATIO = 0.80  # Kept for compatibility; not used by current filters
PERSON_MAX_ASPECT_RATIO = 10.0  # Kept for compatibility; not used by current filters
PERSON_MIN_WIDTH_PX = 20       # Kept for compatibility; not used by current filters
PERSON_MIN_AREA_FRAC = 0.0008  # Kept for compatibility; not used by current filters

# ─────────────────────────────────────────────
# ALERT / COOLDOWN SYSTEM
# ─────────────────────────────────────────────
ALERT_COOLDOWN_FRAMES = 90      # Min frames between repeated alerts for same track ID
VIOLATION_CONFIRM_FRAMES = 3    # Consecutive frames a violation must persist before alerting

# ─────────────────────────────────────────────
# ZONE DEFINITIONS
# Set ENABLE_ZONE_DETECTION = False to disable zone monitoring and overlay
# ─────────────────────────────────────────────
ENABLE_ZONE_DETECTION = False
RESTRICTED_ZONES: List[dict] = []

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
@dataclass
class VisColors:
    compliant:      Tuple[int,int,int] = (50, 205, 50)    # Green  — helmet OK
    violation_ppe:  Tuple[int,int,int] = (0, 0, 255)      # Red    — no helmet
    violation_zone: Tuple[int,int,int] = (0, 165, 255)    # Orange — zone breach
    both_violations:Tuple[int,int,int] = (0, 0, 180)      # Dark red — both
    track_id_text:  Tuple[int,int,int] = (255, 255, 255)  # White
    zone_fill_alpha: float = 0.25
    bbox_thickness: int = 2
    font_scale: float = 0.55
    font_thickness: int = 2

VIS = VisColors()

# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────
DISPLAY_WIDTH   = 1280
DISPLAY_HEIGHT  = 720
WINDOW_NAME     = "Industrial Safety Monitor"
SHOW_FPS        = True
SHOW_ZONE_OVERLAY = False
