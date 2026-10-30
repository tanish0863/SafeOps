"""
obstruction.py
--------------
Heuristic obstruction detection for camera frames.

IMPORTANT DISCLAIMER
--------------------
Obstruction detection is an *approximate heuristic*. It analyses image
properties that are common when a camera is physically blocked (e.g. a
lens cap, a hand, tape over the lens, or a completely stationary scene).
However, these same properties can appear in naturally plain, static
environments (e.g. a camera pointing at a blank white wall, or a scene
where nothing moves for an extended period).

For this reason:
  - Results are reported as UNCERTAIN when the signal is ambiguous.
  - Uncertain results do NOT reduce the overall quality score as a
    confirmed failure; they are surfaced as informational warnings.
  - Only results flagged OBSTRUCTED (high confidence) apply a score penalty.

The three heuristics used are:
  1. Edge density  – a fully blocked lens typically has very few edges.
  2. Intensity std  – a blocked/uniform frame has very low pixel variation.
  3. Frame similarity – for video, consecutive identical frames indicate
     the camera view may be frozen or physically blocked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class ObstructionStatus(str, Enum):
    CLEAR = "CLEAR"           # No evidence of obstruction
    UNCERTAIN = "UNCERTAIN"   # Signal is ambiguous; do not penalise score
    OBSTRUCTED = "OBSTRUCTED" # High-confidence obstruction detected


@dataclass
class ObstructionResult:
    """Result from an obstruction check."""

    status: ObstructionStatus
    confidence: float           # 0.0 = definitely clear, 1.0 = definitely blocked
    details: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    note: str = (
        "Obstruction detection is an approximate heuristic. "
        "Results marked UNCERTAIN should not be treated as confirmed failures."
    )


# ---------------------------------------------------------------------------
# Single-frame obstruction check
# ---------------------------------------------------------------------------

def check_obstruction_single_frame(
    image_bgr: np.ndarray,
    edge_density_low: float = 0.0008,    # fraction of edge pixels – very low = suspicious
    edge_density_uncertain: float = 0.004,
    std_low: float = 3.5,                # intensity std – very low = very uniform
    std_uncertain: float = 10.0,
    canny_low: int = 50,
    canny_high: int = 150,
    is_dark: Optional[bool] = None,
    is_bright: Optional[bool] = None,
    is_blurry: Optional[bool] = None,
) -> ObstructionResult:
    """
    Analyse a single frame for signs of physical obstruction.

    Heuristic logic & Conservative Classification
    ---------------------------------------------
    Physical obstruction cannot be reliably concluded from low edge density
    alone:
      1. Low brightness (darkness) compresses intensity gradients, causing low edge density.
      2. Excessive brightness (glare/overexposure) saturates sensors, causing low edge density.
      3. Optical blur suppresses high frequencies, causing low edge density.
      4. Plain scenes (walls, skies, clean graphics) naturally have low edge density but normal variance.

    Therefore:
      - If darkness, glare, or blur explains the lack of visual information,
        status is reported as UNCERTAIN (informational only; no score penalty).
      - If edge density is low but scene contrast/variation is present (std > std_low),
        low edge density alone does NOT trigger OBSTRUCTED; it is reported as UNCERTAIN or CLEAR.
      - An image is only classified as OBSTRUCTED when BOTH edge density AND intensity
        variation are near zero (std <= std_low AND edge_density <= edge_density_low)
        AND the condition is not explained by extreme lighting or blur.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    edge_density_low : float
        Edge density below this value is suspicious only when accompanied by near-zero std.
    edge_density_uncertain : float
        Edge density below this is considered low detail / ambiguous.
    std_low : float
        Intensity std below this value indicates near-total lack of contrast.
    std_uncertain : float
        Intensity std below this indicates low contrast.
    canny_low / canny_high : int
        Thresholds for the Canny edge detector.
    is_dark, is_bright, is_blurry : bool, optional
        Context flags from lighting/blur checks. If None, estimated from the frame.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # --- Signal 1: edge density ---
    edges = cv2.Canny(gray, canny_low, canny_high)
    edge_density = float(np.sum(edges > 0)) / edges.size

    # --- Signal 2: intensity standard deviation ---
    intensity_std = float(np.std(gray))
    mean_intensity = float(np.mean(gray))

    # Infer context if not explicitly provided
    if is_dark is None:
        is_dark = bool(mean_intensity < 40.0)
    if is_bright is None:
        is_bright = bool(mean_intensity > 225.0)
    if is_blurry is None:
        is_blurry = False

    issues: list[str] = []
    recommendations: list[str] = []

    # Conservative classification
    if is_dark:
        # Darkness explains low edges and low std
        status = ObstructionStatus.UNCERTAIN
        confidence = 0.5
        issues.append(
            f"Obstruction status is uncertain: low scene brightness (mean={mean_intensity:.1f}) "
            "prevents distinguishing physical obstruction from darkness."
        )
    elif is_bright:
        # Glare / overexposure explains low edges and low std
        status = ObstructionStatus.UNCERTAIN
        confidence = 0.5
        issues.append(
            f"Obstruction status is uncertain: excessive glare / overexposure (mean={mean_intensity:.1f}) "
            "prevents distinguishing physical obstruction from sensor saturation."
        )
    elif is_blurry:
        # Severe optical blur explains low edge density
        status = ObstructionStatus.UNCERTAIN
        confidence = 0.4
        issues.append(
            f"Obstruction status is uncertain: severe blur reduces edge density ({edge_density:.5f}), "
            "preventing reliable physical obstruction confirmation."
        )
    elif edge_density <= edge_density_low and intensity_std <= std_low:
        # Decisive evidence: normal lighting, yet zero edges and near-zero variation (e.g. lens cap in lighted room)
        status = ObstructionStatus.OBSTRUCTED
        confidence = 1.0
        issues.append(
            f"Camera may be physically obstructed "
            f"(edge density={edge_density:.5f}, intensity std={intensity_std:.2f})"
        )
        recommendations.append(
            "Inspect the camera lens for physical blockage (e.g. lens cap, tape, or debris)."
        )
        recommendations.append(
            "Check that the camera field of view has not been accidentally covered."
        )
    elif edge_density <= edge_density_low and intensity_std > std_low:
        # Low edge density alone must NOT prove physical obstruction
        status = ObstructionStatus.UNCERTAIN
        confidence = 0.4
        issues.append(
            f"Obstruction status is uncertain: low edge density ({edge_density:.5f}) detected, "
            f"but scene has tonal variation (std={intensity_std:.2f})."
        )
    elif edge_density <= edge_density_uncertain and intensity_std <= std_uncertain:
        # Ambiguous border zone
        status = ObstructionStatus.UNCERTAIN
        confidence = 0.3
        issues.append(
            f"Obstruction status is uncertain: scene has low edge density ({edge_density:.5f}) "
            f"and moderate variance (std={intensity_std:.2f})."
        )
    else:
        status = ObstructionStatus.CLEAR
        confidence = 0.0

    return ObstructionResult(
        status=status,
        confidence=round(confidence, 3),
        details={
            "edge_density": round(edge_density, 5),
            "intensity_std": round(intensity_std, 3),
            "mean_intensity": round(mean_intensity, 2),
            "edge_density_low_threshold": edge_density_low,
            "edge_density_uncertain_threshold": edge_density_uncertain,
            "std_low_threshold": std_low,
            "std_uncertain_threshold": std_uncertain,
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_blurry": is_blurry,
        },
        issues=issues,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# Video-sequence obstruction check (consecutive frame similarity)
# ---------------------------------------------------------------------------

def check_frame_similarity(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    similarity_high: float = 0.995,   # above this = almost certainly identical
    similarity_uncertain: float = 0.985,
) -> ObstructionResult:
    """
    Compare two consecutive video frames to detect a frozen / blocked camera.

    Approach
    --------
    Compute the normalised cross-correlation between the two grayscale frames.
    A value close to 1.0 means the frames are nearly identical, which can
    indicate that the camera view is physically blocked or the scene is
    completely static.

    Note: A static but unobstructed scene (e.g. an empty corridor) will also
    produce high similarity scores.  This check therefore returns UNCERTAIN
    rather than OBSTRUCTED for moderately high similarity values.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray
        Two consecutive frames in BGR colour order.
    similarity_high : float
        Correlation above this threshold is flagged as high-confidence frozen.
    similarity_uncertain : float
        Correlation above this (but below similarity_high) is flagged UNCERTAIN.
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Resize b to match a if dimensions differ (e.g. after sampling)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))

    # Mean-subtract to focus on structure rather than absolute brightness
    a_norm = gray_a - gray_a.mean()
    b_norm = gray_b - gray_b.mean()

    std_a = float(np.std(a_norm))
    std_b = float(np.std(b_norm))

    if std_a < 1e-6 or std_b < 1e-6:
        # At least one frame is completely uniform – treat as high confidence
        similarity = 1.0
    else:
        # Normalised cross-correlation coefficient
        similarity = float(np.mean(a_norm * b_norm) / (std_a * std_b))
        similarity = max(-1.0, min(1.0, similarity))  # clamp to [-1, 1]

    # Map similarity to confidence (higher similarity = more suspicious)
    if similarity >= similarity_high:
        confidence = 1.0
        status = ObstructionStatus.OBSTRUCTED
    elif similarity >= similarity_uncertain:
        confidence = 0.5
        status = ObstructionStatus.UNCERTAIN
    else:
        confidence = 0.0
        status = ObstructionStatus.CLEAR

    issues: list[str] = []
    recommendations: list[str] = []

    if status == ObstructionStatus.OBSTRUCTED:
        issues.append(
            f"Consecutive frames are nearly identical (similarity={similarity:.4f}). "
            "Camera view may be frozen or physically blocked."
        )
        recommendations.append(
            "Verify that the camera is operational and the scene has not been blocked."
        )
    elif status == ObstructionStatus.UNCERTAIN:
        issues.append(
            f"Consecutive frames are very similar (similarity={similarity:.4f}). "
            "This may indicate a frozen camera or a naturally static scene."
        )

    return ObstructionResult(
        status=status,
        confidence=round(abs(confidence), 3),
        details={
            "frame_similarity": round(similarity, 5),
            "similarity_high_threshold": similarity_high,
            "similarity_uncertain_threshold": similarity_uncertain,
        },
        issues=issues,
        recommendations=recommendations,
    )
