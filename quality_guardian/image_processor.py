"""
image_processor.py
------------------
Single-image analysis pipeline.

Usage
-----
    from quality_guardian.image_processor import process_image

    result, report_path, annotated_path = process_image("samples/frame.jpg")
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .assessment import AssessmentConfig, AssessmentResult, assess_frame, status_emoji
from .report_generator import generate_json_report, save_annotated_image

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk and return a BGR NumPy array.

    Raises
    ------
    FileNotFoundError
        When the file does not exist.
    ValueError
        When the file extension is not supported or the image cannot be decoded.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{path.suffix}'. "
            f"Supported formats: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        )

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(
            f"Could not decode image: {path}. "
            "The file may be corrupted or in an unsupported encoding."
        )

    return image


def process_image(
    image_path: str | Path,
    config: Optional[AssessmentConfig] = None,
    save_report: bool = True,
    save_annotated: bool = True,
    show_display: bool = False,
) -> Tuple[AssessmentResult, Optional[Path], Optional[Path]]:
    """
    Run the full quality assessment pipeline on a single image file.

    Steps
    -----
    1. Load and validate the image.
    2. Run the assessment engine (all five checks).
    3. Optionally save a JSON report.
    4. Optionally save an annotated copy of the image.
    5. Optionally display the annotated result using matplotlib.

    Parameters
    ----------
    image_path : str | Path
        Path to the input image (JPG, JPEG, or PNG).
    config : AssessmentConfig, optional
        Custom thresholds and penalty weights.
    save_report : bool
        Save a JSON report to the reports/ directory (default True).
    save_annotated : bool
        Save an annotated image to the reports/ directory (default True).
    show_display : bool
        Display the annotated result in a matplotlib window (default False).

    Returns
    -------
    (AssessmentResult, json_report_path | None, annotated_image_path | None)
    """
    path = Path(image_path)
    image = load_image(path)

    ts = datetime.now()
    result = assess_frame(image, config=config)

    json_path: Optional[Path] = None
    annotated_path: Optional[Path] = None

    if save_report:
        json_path = generate_json_report(
            result=result,
            source_name=path.name,
            image_shape=image.shape,
            timestamp=ts,
        )

    if save_annotated:
        annotated_path = save_annotated_image(
            image_bgr=image,
            result=result,
            source_name=path.name,
            timestamp=ts,
        )

    if show_display:
        _display_result(image, result, annotated_path)

    return result, json_path, annotated_path


def _display_result(
    original: np.ndarray,
    result: AssessmentResult,
    annotated_path: Optional[Path],
) -> None:
    """
    Display the assessment result using matplotlib.
    Falls back gracefully if matplotlib is unavailable or display is unsupported.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[INFO] matplotlib not available – skipping display.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Camera Quality Guardian  |  Score: {result.overall_score}/100  "
        f"[{result.overall_status}]",
        fontsize=14,
        fontweight="bold",
    )

    # Original image (convert BGR → RGB for matplotlib)
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Annotated image (if saved)
    if annotated_path and annotated_path.exists():
        annotated = cv2.imread(str(annotated_path))
        axes[1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Annotated Result")
    axes[1].axis("off")

    # Summary text panel
    summary_lines = [f"Overall: {result.overall_score}/100 [{result.overall_status}]", ""]
    for name, check in result.checks.items():
        icon = "✓" if check.passed else "✗"
        summary_lines.append(f"{icon} {name.title()}: {check.value:.2f} (threshold {check.threshold})")

    obs = result.obstruction
    obs_icon = {"CLEAR": "✓", "UNCERTAIN": "?", "OBSTRUCTED": "✗"}.get(obs.status.value, "?")
    summary_lines.append(f"{obs_icon} Obstruction: {obs.status.value}")

    if result.all_issues:
        summary_lines.append("\nIssues:")
        for issue in result.all_issues:
            summary_lines.append(f"  • {issue}")

    fig.text(
        0.5, 0.01, "\n".join(summary_lines),
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
    )

    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    finally:
        plt.close(fig)


def print_result_summary(result: AssessmentResult, source: str = "") -> None:
    """Print a human-readable summary of an AssessmentResult to stdout."""
    divider = "-" * 60
    print(f"\n{divider}")
    print(f"  Camera Quality Guardian - Assessment Report")
    if source:
        print(f"  Source : {source}")
    print(divider)
    print(f"  Overall Score  : {result.overall_score:.1f} / 100")
    print(f"  Overall Status : {result.overall_status}")
    print(divider)
    print("  Individual Checks:")
    for name, check in result.checks.items():
        if check.passed:
            icon = "[OK]"
        elif getattr(check, "uncertain", False):
            icon = "[??]"
        else:
            icon = "[!!]"
        print(f"    {icon}  {name.title():12s}  value={check.value:.2f}  threshold={check.threshold}")
    obs = result.obstruction
    obs_icon = {"CLEAR": "[OK]", "UNCERTAIN": "[??]", "OBSTRUCTED": "[!!]"}.get(obs.status.value, "[ ]")
    print(f"    {obs_icon}  Obstruction   status={obs.status.value}  confidence={obs.confidence:.2f}")
    if result.all_issues:
        print(divider)
        print("  Issues Detected:")
        for issue in result.all_issues:
            print(f"    * {issue}")
    if result.all_recommendations:
        print(divider)
        print("  Recommendations:")
        for rec in result.all_recommendations:
            print(f"    -> {rec}")
    print(divider)
    print(f"  {result.disclaimer}")
    print(divider + "\n")
