"""
config.py
---------
Configuration parameters for the Safety Vest Detection module.
Defines inference thresholds, hardware device selection, torso bounding criteria,
and label normalization mappings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class VestDetectorConfig:
    """
    Centralized configuration for vest detection and torso compliance verification.
    """
    # Hardware and execution
    device: str = "cuda"                        # "cuda", "cpu", or device index like "cuda:0"
    fallback_to_cpu: bool = True               # If CUDA unavailable, gracefully fallback to CPU
    image_size: int = 640                      # YOLO standard inference dimension
    half_precision: bool = True                # Use FP16 on CUDA if supported

    # Model inference thresholds
    confidence_threshold: float = 0.35         # General fallback confidence threshold
    vest_confidence_threshold: float = 0.35    # Threshold for WITH_VEST -> PRESENT
    no_vest_confidence_threshold: float = 0.35 # Threshold for NO_VEST -> ABSENT
    iou_threshold: float = 0.45                # Non-maximum suppression IoU threshold

    # Torso bounding box validation criteria
    min_torso_width: float = 35.0              # Minimum width (px) for reliable evaluation
    min_torso_height: float = 45.0             # Minimum height (px) for reliable evaluation
    min_torso_area: float = 2000.0             # Minimum area (px²) for reliable evaluation
    padding_fraction: float = 0.08             # Context padding applied to torso crop (8%)

    # Torso-vest spatial overlap criteria
    min_overlap_iou: float = 0.18              # Minimum bounding box IoU
    min_intersection_ratio: float = 0.30       # Fraction of vest box that must lie inside torso

    # Image quality bounds for three-state assessment
    min_intensity_mean: float = 20.0           # Underexposed / dark cutoff
    max_intensity_mean: float = 248.0          # Overexposed / glare cutoff
    min_laplacian_variance: float = 3.5        # Severe blur cutoff

    # Default model path
    default_model_filename: str = "yolov8n_vest_best.pt"
    models_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "models"
    )

    # Numeric class IDs from model checkpoint
    vest_class_ids: Set[int] = field(default_factory=lambda: {7})
    no_vest_class_ids: Set[int] = field(default_factory=lambda: {4})

    # Label normalization mappings
    vest_label_aliases: Set[str] = field(
        default_factory=lambda: {
            "safety vest",
            "safety_vest",
            "safety-vest",
            "vest",
            "with_vest",
            "with-vest",
            "with vest",
            "with_safety_vest",
            "with-safety-vest",
            "with safety vest",
            "reflective_vest",
            "reflective-vest",
            "reflective vest",
            "hi_vis_vest",
            "hi-vis-vest",
            "hi vis vest",
            "7",
        }
    )
    no_vest_label_aliases: Set[str] = field(
        default_factory=lambda: {
            "no-safety vest",
            "no_safety_vest",
            "no safety vest",
            "no-safety-vest",
            "no_vest",
            "no-vest",
            "no vest",
            "without_vest",
            "without-vest",
            "without vest",
            "without_safety_vest",
            "without-safety-vest",
            "without safety vest",
            "4",
        }
    )

    def normalize_label(self, raw_label: Union[str, int, float]) -> Optional[str]:
        """
        Normalize a raw model class string or numeric class ID into
        canonical 'WITH_VEST', 'NO_VEST', or None.
        """
        if isinstance(raw_label, (int, float)):
            int_label = int(raw_label)
            if int_label in self.vest_class_ids:
                return "WITH_VEST"
            if int_label in self.no_vest_class_ids:
                return "NO_VEST"
            raw_str = str(int_label)
        else:
            raw_str = str(raw_label).strip()

        cleaned = raw_str.lower().strip()
        # Direct alias check
        if cleaned in self.vest_label_aliases:
            return "WITH_VEST"
        if cleaned in self.no_vest_label_aliases:
            return "NO_VEST"

        # Check with hyphens/underscores converted to spaces
        normalized_spacing = cleaned.replace("-", " ").replace("_", " ").strip()
        while "  " in normalized_spacing:
            normalized_spacing = normalized_spacing.replace("  ", " ")

        if normalized_spacing in self.vest_label_aliases or normalized_spacing == "safety vest":
            return "WITH_VEST"
        if normalized_spacing in self.no_vest_label_aliases or normalized_spacing in ("no safety vest", "no vest"):
            return "NO_VEST"

        return None
