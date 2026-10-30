"""
assessment.py
-------------
Rule-based quality scoring engine.

HOW THE SCORE IS CALCULATED
----------------------------
The overall quality score (0–100) is computed by starting at 100 and
deducting penalty points for each failed check.

Each check has a configurable maximum penalty weight.  When a check
fails the full penalty is applied.  When a check produces an UNCERTAIN
result (currently only obstruction detection), no penalty is applied –
the uncertain result is surfaced as an informational warning only.

Default penalty weights
    Darkness     : 30 points
    Blur         : 25 points
    Resolution   : 20 points
    Glare        : 15 points
    Obstruction  : 10 points  (only when status == OBSTRUCTED, not UNCERTAIN)

Overall status thresholds
    GOOD    : score >= 80
    WARNING : score >= 50
    POOR    : score <  50

IMPORTANT DISCLAIMER
--------------------
This score is a rule-based heuristic derived from configurable thresholds
and penalty weights.  It does NOT represent a scientifically validated
probability of detection success.  It is intended as an operational
guideline to help flag frames that are likely to degrade PPE detector
performance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .checks import CheckResult, check_darkness, check_blur, check_resolution, check_glare
from .obstruction import ObstructionResult, ObstructionStatus, check_obstruction_single_frame


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AssessmentConfig:
    """
    All configurable thresholds and penalty weights in one place.
    Override the defaults by passing keyword arguments.
    """

    # --- Darkness ---
    darkness_mean_threshold: float = 60.0
    darkness_dark_pixel_fraction: float = 0.40
    darkness_dark_pixel_value: int = 50
    darkness_penalty: float = 30.0

    # --- Blur ---
    blur_laplacian_variance_threshold: float = 100.0
    blur_penalty: float = 25.0

    # --- Resolution ---
    resolution_min_width: int = 640
    resolution_min_height: int = 480
    resolution_penalty: float = 20.0

    # --- Glare ---
    glare_bright_value_threshold: int = 230
    glare_bright_saturation_threshold: int = 30
    glare_bright_pixel_fraction_threshold: float = 0.05
    glare_penalty: float = 15.0

    # --- Obstruction ---
    obstruction_edge_density_low: float = 0.0008
    obstruction_edge_density_uncertain: float = 0.004
    obstruction_std_low: float = 3.5
    obstruction_std_uncertain: float = 10.0
    obstruction_penalty: float = 10.0  # only applied for OBSTRUCTED, not UNCERTAIN

    # --- Overall status thresholds ---
    score_good_threshold: float = 80.0
    score_warning_threshold: float = 50.0


# ---------------------------------------------------------------------------
# Overall assessment result
# ---------------------------------------------------------------------------

@dataclass
class AssessmentResult:
    """Complete quality assessment for a single image frame."""

    overall_score: float
    overall_status: str            # "GOOD", "WARNING", or "POOR"
    checks: dict[str, CheckResult]
    obstruction: ObstructionResult
    all_issues: list[str]
    all_recommendations: list[str]
    score_breakdown: dict          # Details on how the score was computed
    disclaimer: str = (
        "This score is a rule-based heuristic. It is not a scientifically "
        "validated probability of PPE detection success."
    )


# ---------------------------------------------------------------------------
# Assessment engine
# ---------------------------------------------------------------------------

def assess_frame(
    image_bgr,
    config: Optional[AssessmentConfig] = None,
) -> AssessmentResult:
    """
    Run all five quality checks on a single BGR image and return a scored result.

    Parameters
    ----------
    image_bgr : np.ndarray
        Input image in BGR colour order (as returned by cv2.imread or VideoCapture).
    config : AssessmentConfig, optional
        Threshold and penalty configuration.  Uses sensible defaults when None.

    Returns
    -------
    AssessmentResult
        Contains the overall score, overall status, individual check results,
        all detected issues and recommendations, and a score breakdown.
    """
    if config is None:
        config = AssessmentConfig()

    # ------------------------------------------------------------------
    # Run the five checks
    # ------------------------------------------------------------------
    darkness_result = check_darkness(
        image_bgr,
        mean_intensity_threshold=config.darkness_mean_threshold,
        dark_pixel_fraction_threshold=config.darkness_dark_pixel_fraction,
        dark_pixel_value_threshold=config.darkness_dark_pixel_value,
    )

    blur_result = check_blur(
        image_bgr,
        laplacian_variance_threshold=config.blur_laplacian_variance_threshold,
    )

    resolution_result = check_resolution(
        image_bgr,
        min_width=config.resolution_min_width,
        min_height=config.resolution_min_height,
    )

    glare_result = check_glare(
        image_bgr,
        bright_value_threshold=config.glare_bright_value_threshold,
        bright_saturation_threshold=config.glare_bright_saturation_threshold,
        bright_pixel_fraction_threshold=config.glare_bright_pixel_fraction_threshold,
    )

    # Context derivation for obstruction heuristic
    is_dark = not darkness_result.passed
    is_bright = not glare_result.passed
    is_blurry = not blur_result.passed and not getattr(blur_result, "uncertain", False)

    obstruction_result = check_obstruction_single_frame(
        image_bgr,
        edge_density_low=config.obstruction_edge_density_low,
        edge_density_uncertain=config.obstruction_edge_density_uncertain,
        std_low=config.obstruction_std_low,
        std_uncertain=config.obstruction_std_uncertain,
        is_dark=is_dark,
        is_bright=is_bright,
        is_blurry=is_blurry,
    )

    # ------------------------------------------------------------------
    # Score calculation
    # ------------------------------------------------------------------
    score = 100.0
    score_breakdown: dict = {}

    for check, penalty_weight in [
        (darkness_result, config.darkness_penalty),
        (blur_result, config.blur_penalty),
        (resolution_result, config.resolution_penalty),
        (glare_result, config.glare_penalty),
    ]:
        if not check.passed:
            if getattr(check, "uncertain", False):
                score_breakdown[check.name] = {
                    "failed": False,
                    "uncertain": True,
                    "penalty_applied": 0.0,
                    "note": "Result is uncertain due to insufficient visual detail; no penalty applied.",
                }
            else:
                score -= penalty_weight
                score_breakdown[check.name] = {
                    "failed": True,
                    "uncertain": False,
                    "penalty_applied": penalty_weight,
                }
        else:
            score_breakdown[check.name] = {
                "failed": False,
                "uncertain": False,
                "penalty_applied": 0.0,
            }

    # Obstruction: only penalise if OBSTRUCTED (not UNCERTAIN)
    obstruction_penalty_applied = 0.0
    if obstruction_result.status == ObstructionStatus.OBSTRUCTED:
        score -= config.obstruction_penalty
        obstruction_penalty_applied = config.obstruction_penalty

    score_breakdown["obstruction"] = {
        "status": obstruction_result.status.value,
        "penalty_applied": obstruction_penalty_applied,
        "note": (
            "Penalty is only applied when status is OBSTRUCTED. "
            "UNCERTAIN results do not reduce the score."
        ),
    }

    # Clamp score to [0, 100]
    score = max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    if score >= config.score_good_threshold:
        overall_status = "GOOD"
    elif score >= config.score_warning_threshold:
        overall_status = "WARNING"
    else:
        overall_status = "POOR"

    # ------------------------------------------------------------------
    # Collect all issues and recommendations
    # ------------------------------------------------------------------
    all_issues: list[str] = []
    all_recommendations: list[str] = []

    for check in [darkness_result, blur_result, resolution_result, glare_result]:
        all_issues.extend(check.issues)
        all_recommendations.extend(check.recommendations)

    all_issues.extend(obstruction_result.issues)
    all_recommendations.extend(obstruction_result.recommendations)

    # De-duplicate recommendations while preserving order
    seen: set[str] = set()
    unique_recommendations: list[str] = []
    for rec in all_recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)

    return AssessmentResult(
        overall_score=round(score, 1),
        overall_status=overall_status,
        checks={
            "darkness": darkness_result,
            "blur": blur_result,
            "resolution": resolution_result,
            "glare": glare_result,
        },
        obstruction=obstruction_result,
        all_issues=all_issues,
        all_recommendations=unique_recommendations,
        score_breakdown=score_breakdown,
    )


def status_emoji(status: str) -> str:
    """Return a simple emoji for the given overall status string."""
    return {"GOOD": "✅", "WARNING": "⚠️", "POOR": "❌"}.get(status, "?")
