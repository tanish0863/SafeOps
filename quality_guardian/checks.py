"""
checks.py
---------
Individual camera quality check functions.

Each check receives a BGR image (NumPy ndarray) and returns a dict
containing the raw measurement, a boolean pass/fail flag, optional
recommendations, and any extra diagnostic fields.

All thresholds are passed as arguments so that callers can make them
configurable without touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Return-value containers
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Structured result returned by every quality check."""

    name: str
    passed: bool
    value: float                          # Primary numeric measurement
    threshold: float                      # Threshold used for pass/fail
    details: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    uncertain: bool = False


# ---------------------------------------------------------------------------
# A. Darkness Detection
# ---------------------------------------------------------------------------

def check_darkness(
    image_bgr: np.ndarray,
    mean_intensity_threshold: float = 60.0,
    dark_pixel_fraction_threshold: float = 0.40,
    dark_pixel_value_threshold: int = 50,
) -> CheckResult:
    """
    Detect whether an image is too dark for reliable PPE detection.

    Approach
    --------
    1. Convert the image to grayscale.
    2. Calculate the mean pixel intensity (0 = black, 255 = white).
    3. Count the fraction of pixels whose intensity is below
       ``dark_pixel_value_threshold``.
    4. Flag the image as too dark when EITHER:
       - mean intensity < ``mean_intensity_threshold``, OR
       - dark pixel fraction > ``dark_pixel_fraction_threshold``.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order (as loaded by cv2.imread).
    mean_intensity_threshold : float
        Minimum acceptable mean grayscale intensity (default 60).
    dark_pixel_fraction_threshold : float
        Maximum acceptable fraction of dark pixels (default 0.40 = 40 %).
    dark_pixel_value_threshold : int
        Pixel intensity below which a pixel is considered "dark" (default 50).

    Returns
    -------
    CheckResult
        name       = "darkness"
        value      = mean grayscale intensity
        threshold  = mean_intensity_threshold
        passed     = True when the image has sufficient lighting
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mean_intensity: float = float(np.mean(gray))

    # Fraction of pixels that are very dark.
    # np.sum() returns numpy.int64; dividing by gray.size (also numpy int)
    # yields numpy.float64 – explicitly cast to Python float so the value
    # is JSON-serialisable without relying solely on _safe_serialise.
    dark_pixels = int(np.sum(gray < dark_pixel_value_threshold))
    total_pixels = int(gray.size)
    dark_pixel_fraction: float = float(dark_pixels / total_pixels)

    # The image fails if either criterion is violated.
    # Comparing a float to a float always produces a Python bool here
    # because mean_intensity was already cast to float above.
    too_dark_by_mean: bool = bool(mean_intensity < mean_intensity_threshold)
    too_dark_by_fraction: bool = bool(dark_pixel_fraction > dark_pixel_fraction_threshold)

    passed = not (too_dark_by_mean or too_dark_by_fraction)

    issues: list[str] = []
    recommendations: list[str] = []

    if not passed:
        issues.append(
            f"Insufficient lighting detected "
            f"(mean intensity={mean_intensity:.1f}, "
            f"dark pixel fraction={dark_pixel_fraction:.1%})"
        )
        recommendations.append(
            "Improve the lighting in the monitored area or increase camera exposure."
        )
        recommendations.append(
            "Consider adding supplementary lighting or repositioning the camera "
            "away from very dark backgrounds."
        )

    return CheckResult(
        name="darkness",
        passed=passed,
        value=mean_intensity,
        threshold=mean_intensity_threshold,
        details={
            "mean_intensity": round(mean_intensity, 2),
            "dark_pixel_fraction": round(dark_pixel_fraction, 4),
            "dark_pixel_value_threshold": dark_pixel_value_threshold,
            "dark_pixel_fraction_threshold": dark_pixel_fraction_threshold,
            "too_dark_by_mean": too_dark_by_mean,
            "too_dark_by_fraction": too_dark_by_fraction,
        },
        issues=issues,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# B. Blur Detection
# ---------------------------------------------------------------------------

def check_blur(
    image_bgr: np.ndarray,
    laplacian_variance_threshold: float = 100.0,
    min_evaluable_brightness: float = 35.0,
    max_evaluable_brightness: float = 225.0,
    min_evaluable_std: float = 0.1,
) -> CheckResult:
    """
    Estimate image sharpness using the Laplacian operator.

    Approach
    --------
    1. Convert the image to grayscale.
    2. Apply the Laplacian operator (a second-order derivative filter).
       Sharp images contain strong edges -> high Laplacian responses.
    3. Calculate the variance of the Laplacian output.
       Low variance ≈ few or weak edges ≈ blurry image.

    Reliability Context
    -------------------
    Laplacian variance requires adequate visual detail and lighting to be
    meaningful. If an image is severely underexposed (mean < 35), severely
    overexposed (mean > 225), or completely featureless (std < 0.1), low
    variance is an artifact of the lighting or uniform surface, not necessarily
    an out-of-focus camera. In these conditions, the result is marked
    ``uncertain = True`` so that false penalties are avoided.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    laplacian_variance_threshold : float
        Minimum acceptable Laplacian variance (default 100).
        Values below this are flagged as potentially blurry.
    min_evaluable_brightness : float
        Minimum mean intensity required to reliably evaluate blur (default 35).
    max_evaluable_brightness : float
        Maximum mean intensity above which overexposure impairs blur evaluation (default 225).
    min_evaluable_std : float
        Minimum standard deviation required to consider an image non-featureless (default 0.1).

    Returns
    -------
    CheckResult
        name      = "blur"
        value     = Laplacian variance
        threshold = laplacian_variance_threshold
        passed    = True when the image appears sharp enough
        uncertain = True when extreme lighting/featurelessness prevents reliable judgment
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Laplacian filter: highlights rapid intensity changes (edges)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance: float = float(laplacian.var())

    gray_mean = float(np.mean(gray))
    gray_std = float(np.std(gray))

    passed = laplacian_variance >= laplacian_variance_threshold
    uncertain = False

    issues: list[str] = []
    recommendations: list[str] = []

    if not passed:
        if gray_mean < min_evaluable_brightness:
            uncertain = True
            issues.append(
                f"Focus/blur cannot be reliably assessed due to severe underexposure "
                f"(mean intensity={gray_mean:.1f}, threshold={min_evaluable_brightness:.1f})"
            )
            recommendations.append(
                "Ensure adequate scene lighting before evaluating camera focus."
            )
        elif gray_mean > max_evaluable_brightness:
            uncertain = True
            issues.append(
                f"Focus/blur cannot be reliably assessed due to severe overexposure "
                f"(mean intensity={gray_mean:.1f}, threshold={max_evaluable_brightness:.1f})"
            )
            recommendations.append(
                "Reduce exposure or mitigate glare before evaluating camera focus."
            )
        elif gray_std < min_evaluable_std:
            uncertain = True
            issues.append(
                "Focus/blur cannot be reliably assessed on a completely featureless / solid-colour image."
            )
            recommendations.append(
                "Ensure the camera view includes visible scene texture to evaluate focus."
            )
        else:
            # Genuine blur detected in evaluable scene
            issues.append(
                f"Image appears blurry "
                f"(Laplacian variance={laplacian_variance:.2f}, "
                f"threshold={laplacian_variance_threshold})"
            )
            recommendations.append(
                "Clean the camera lens and ensure it is free from smudges or condensation."
            )
            recommendations.append(
                "Adjust the camera focus. If using an auto-focus camera, "
                "ensure the subject is within the focus range."
            )

    return CheckResult(
        name="blur",
        passed=passed,
        value=round(laplacian_variance, 2),
        threshold=laplacian_variance_threshold,
        details={
            "laplacian_variance": round(laplacian_variance, 2),
            "threshold": laplacian_variance_threshold,
            "gray_mean": round(gray_mean, 2),
            "gray_std": round(gray_std, 2),
            "uncertain": uncertain,
        },
        issues=issues,
        recommendations=recommendations,
        uncertain=uncertain,
    )


# ---------------------------------------------------------------------------
# C. Resolution Check
# ---------------------------------------------------------------------------

def check_resolution(
    image_bgr: np.ndarray,
    min_width: int = 640,
    min_height: int = 480,
) -> CheckResult:
    """
    Verify that the image meets minimum resolution requirements.

    Approach
    --------
    Read the image dimensions from the NumPy array shape and compare
    against configurable minimum width and height values.

    Why resolution matters for PPE detection
    ----------------------------------------
    Reliable object detection typically requires that the target object
    (e.g. a hard hat or safety vest) occupies a sufficient number of
    pixels.  Very low-resolution images make it hard for detectors to
    distinguish small safety-critical details.

    Note: This check only verifies frame resolution.  Evaluating whether
    workers occupy enough pixels would require a connected person detector,
    which is not part of this module.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    min_width : int
        Minimum acceptable image width in pixels (default 640).
    min_height : int
        Minimum acceptable image height in pixels (default 480).

    Returns
    -------
    CheckResult
        name      = "resolution"
        value     = min(width, height)   (weakest dimension)
        threshold = min(min_width, min_height)
        passed    = True when both dimensions meet requirements
    """
    # ndarray.shape values are numpy.intp (a NumPy integer subtype).
    # Cast to plain int so comparisons yield Python bool, not numpy.bool_.
    height: int = int(image_bgr.shape[0])
    width: int = int(image_bgr.shape[1])

    width_ok: bool = bool(width >= min_width)
    height_ok: bool = bool(height >= min_height)
    passed = width_ok and height_ok

    issues: list[str] = []
    recommendations: list[str] = []

    if not passed:
        failed_dims = []
        if not width_ok:
            failed_dims.append(f"width {width}px < required {min_width}px")
        if not height_ok:
            failed_dims.append(f"height {height}px < required {min_height}px")
        issues.append(f"Resolution too low: {', '.join(failed_dims)}")
        recommendations.append(
            f"Use a camera that supports at least {min_width}×{min_height} pixels."
        )
        recommendations.append(
            "Check camera settings and ensure the video stream is not being "
            "scaled down unnecessarily."
        )

    return CheckResult(
        name="resolution",
        passed=passed,
        value=float(min(width, height)),
        threshold=float(min(min_width, min_height)),
        details={
            "width": width,
            "height": height,
            "min_width": min_width,
            "min_height": min_height,
            "width_ok": width_ok,
            "height_ok": height_ok,
        },
        issues=issues,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# D. Glare Detection
# ---------------------------------------------------------------------------

def check_glare(
    image_bgr: np.ndarray,
    bright_value_threshold: int = 230,
    bright_saturation_threshold: int = 30,
    bright_pixel_fraction_threshold: float = 0.05,
) -> CheckResult:
    """
    Detect excessive brightness or glare regions.

    Approach
    --------
    1. Convert the image to HSV colour space.
       HSV separates colour (Hue), vividness (Saturation), and brightness (Value).
    2. Identify overexposed pixels: very high Value AND very low Saturation.
       Overexposed / glare pixels typically appear white (high V, low S).
    3. Calculate the fraction of such pixels in the image.
    4. Flag the image when the fraction exceeds ``bright_pixel_fraction_threshold``.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order.
    bright_value_threshold : int
        HSV Value channel threshold above which a pixel is considered very bright
        (0–255, default 230).
    bright_saturation_threshold : int
        HSV Saturation channel threshold below which a bright pixel is considered
        overexposed / washed-out (0–255, default 30).
    bright_pixel_fraction_threshold : float
        Maximum acceptable fraction of glare/overexposed pixels (default 0.05 = 5 %).

    Returns
    -------
    CheckResult
        name      = "glare"
        value     = fraction of overexposed pixels (0.0–1.0)
        threshold = bright_pixel_fraction_threshold
        passed    = True when glare is within acceptable limits
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Split into individual channels
    _hue, saturation, value = cv2.split(hsv)

    # Glare mask: pixels that are both very bright AND nearly desaturated (white-ish)
    glare_mask = (value >= bright_value_threshold) & (saturation <= bright_saturation_threshold)

    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    glare_pixel_count = int(np.sum(glare_mask))
    bright_pixel_fraction: float = glare_pixel_count / total_pixels

    passed = bright_pixel_fraction <= bright_pixel_fraction_threshold

    issues: list[str] = []
    recommendations: list[str] = []

    if not passed:
        issues.append(
            f"Excessive glare or overexposure detected "
            f"(bright pixel fraction={bright_pixel_fraction:.1%}, "
            f"threshold={bright_pixel_fraction_threshold:.1%})"
        )
        recommendations.append(
            "Adjust the camera angle to avoid direct light sources in the field of view."
        )
        recommendations.append(
            "Reduce camera exposure or use a polarising filter to mitigate glare."
        )
        recommendations.append(
            "Reposition overhead lighting so it does not shine directly into the lens."
        )

    return CheckResult(
        name="glare",
        passed=passed,
        value=round(bright_pixel_fraction, 4),
        threshold=bright_pixel_fraction_threshold,
        details={
            "bright_pixel_fraction": round(bright_pixel_fraction, 4),
            "glare_pixel_count": glare_pixel_count,
            "total_pixels": total_pixels,
            "bright_value_threshold": bright_value_threshold,
            "bright_saturation_threshold": bright_saturation_threshold,
            "bright_pixel_fraction_threshold": bright_pixel_fraction_threshold,
        },
        issues=issues,
        recommendations=recommendations,
        # Store the mask so callers can use it for visual annotation
    )
