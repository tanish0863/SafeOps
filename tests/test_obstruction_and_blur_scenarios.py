"""
test_obstruction_and_blur_scenarios.py
---------------------------------------
Comprehensive tests for:
1. Normal image with no obstruction (CLEAR, score GOOD)
2. Completely dark image (Darkness detected; blur & obstruction UNCERTAIN with zero score penalty)
3. Heavily blurred image (Blur detected; obstruction UNCERTAIN with zero score penalty)
4. Overexposed / glare image (Glare detected; blur & obstruction UNCERTAIN with zero score penalty)
5. Genuinely obstructed image (Uniform surface under normal illumination -> OBSTRUCTED)
6. Low edge density alone must not trigger OBSTRUCTED
7. Real sample images (normal.jpg, dark.jpg, blurry.jpg, bright.jpg)
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from quality_guardian.assessment import AssessmentConfig, assess_frame
from quality_guardian.checks import check_blur
from quality_guardian.obstruction import ObstructionStatus, check_obstruction_single_frame


class TestObstructionAndBlurScenarios:
    """Validate obstruction and blur behavior across all edge conditions."""

    def test_normal_image_unobstructed(self, normal_image):
        """Normal image must be CLEAR of obstruction, sharp, and score GOOD."""
        result = assess_frame(normal_image)
        assert result.obstruction.status == ObstructionStatus.CLEAR
        assert result.checks["blur"].passed is True
        assert result.overall_status == "GOOD"
        assert result.overall_score >= 80.0
        assert result.score_breakdown["obstruction"]["penalty_applied"] == 0.0

    def test_completely_dark_image_uncertain_obstruction_and_blur(self, very_dark_image):
        """
        In a completely dark image, darkness must be flagged,
        while blur and obstruction must be marked UNCERTAIN
        without causing misleading additional penalties.
        """
        result = assess_frame(very_dark_image)
        assert result.checks["darkness"].passed is False
        assert result.checks["blur"].uncertain is True
        assert result.obstruction.status == ObstructionStatus.UNCERTAIN
        # Only darkness penalty should be applied (30 points)
        assert result.score_breakdown["darkness"]["penalty_applied"] == 30.0
        assert result.score_breakdown["blur"]["penalty_applied"] == 0.0
        assert result.score_breakdown["obstruction"]["penalty_applied"] == 0.0
        assert result.overall_score == 70.0
        assert result.overall_status == "WARNING"

    def test_heavily_blurred_image_detected_without_obstruction_penalty(self):
        """
        An intentionally blurred image must be detected as blurry,
        but must not be falsely flagged as physically obstructed.
        """
        # Create an image with content (rectangles) and apply heavy blur
        img = np.full((480, 640, 3), 150, dtype=np.uint8)
        cv2.rectangle(img, (150, 100), (490, 380), (220, 180, 130), -1)
        blurred = cv2.GaussianBlur(img, (51, 51), sigmaX=15)

        result = assess_frame(blurred)
        assert result.checks["blur"].passed is False
        assert result.checks["blur"].uncertain is False  # Confidently blurry!
        assert result.obstruction.status in (ObstructionStatus.CLEAR, ObstructionStatus.UNCERTAIN)
        assert result.obstruction.status != ObstructionStatus.OBSTRUCTED
        assert result.score_breakdown["blur"]["penalty_applied"] == 25.0
        assert result.score_breakdown["obstruction"]["penalty_applied"] == 0.0
        assert result.overall_score == 75.0

    def test_overexposed_image_uncertain_obstruction_and_blur(self, bright_glare_image):
        """
        An overexposed image must detect glare, but blur and obstruction
        must be reported as UNCERTAIN without additional false penalties.
        """
        result = assess_frame(bright_glare_image)
        assert result.checks["glare"].passed is False
        assert result.checks["blur"].uncertain is True
        assert result.obstruction.status == ObstructionStatus.UNCERTAIN
        assert result.score_breakdown["glare"]["penalty_applied"] == 15.0
        assert result.score_breakdown["blur"]["penalty_applied"] == 0.0
        assert result.score_breakdown["obstruction"]["penalty_applied"] == 0.0
        assert result.overall_score == 85.0
        assert result.overall_status == "GOOD"

    def test_genuinely_obstructed_image(self):
        """
        A camera lens that is physically covered in an illuminated area
        produces near-zero edges and near-zero variance under normal lighting.
        This must be detected as OBSTRUCTED with high confidence.
        """
        # Uniform gray image (e.g., lens cap or cardboard in normal room lighting)
        obstructed_img = np.full((480, 640, 3), 100, dtype=np.uint8)
        result = assess_frame(obstructed_img)

        assert result.obstruction.status == ObstructionStatus.OBSTRUCTED
        assert result.obstruction.confidence >= 0.90
        assert result.score_breakdown["obstruction"]["penalty_applied"] == 10.0
        assert result.overall_score == 90.0

    def test_low_edge_density_alone_does_not_flag_obstruction(self):
        """
        An image with low edge density but significant contrast/variation
        must not be classified as OBSTRUCTED.
        """
        # Create a smooth vertical gradient: std is high (~73), but edge detector yields 0 edges
        gradient = np.tile(np.linspace(50, 200, 480, dtype=np.uint8)[:, None], (1, 640))
        img = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)

        obs_res = check_obstruction_single_frame(img)
        assert obs_res.status != ObstructionStatus.OBSTRUCTED
        assert obs_res.status in (ObstructionStatus.CLEAR, ObstructionStatus.UNCERTAIN)


class TestRealSampleFiles:
    """Verify that the four generated sample files in samples/ behave as expected."""

    @pytest.fixture(autouse=True)
    def setup_samples(self):
        self.samples_dir = Path("samples")
        if not (self.samples_dir / "normal.jpg").exists():
            pytest.skip("Sample files not present")

    def test_sample_normal_jpg(self):
        img = cv2.imread(str(self.samples_dir / "normal.jpg"))
        res = assess_frame(img)
        assert res.overall_status == "GOOD"
        assert res.overall_score == 100.0
        assert res.obstruction.status == ObstructionStatus.CLEAR
        assert res.checks["blur"].passed is True

    def test_sample_dark_jpg(self):
        img = cv2.imread(str(self.samples_dir / "dark.jpg"))
        res = assess_frame(img)
        assert res.overall_status == "WARNING"
        assert res.overall_score == 70.0
        assert res.checks["darkness"].passed is False
        assert res.checks["blur"].uncertain is True
        assert res.obstruction.status == ObstructionStatus.UNCERTAIN
        assert res.score_breakdown["obstruction"]["penalty_applied"] == 0.0

    def test_sample_blurry_jpg(self):
        img = cv2.imread(str(self.samples_dir / "blurry.jpg"))
        res = assess_frame(img)
        assert res.overall_status == "WARNING"
        assert res.overall_score == 75.0
        assert res.checks["blur"].passed is False
        assert res.checks["blur"].uncertain is False
        assert res.obstruction.status == ObstructionStatus.UNCERTAIN
        assert res.score_breakdown["obstruction"]["penalty_applied"] == 0.0

    def test_sample_bright_jpg(self):
        img = cv2.imread(str(self.samples_dir / "bright.jpg"))
        res = assess_frame(img)
        assert res.overall_status == "GOOD"
        assert res.overall_score == 85.0
        assert res.checks["glare"].passed is False
        assert res.checks["blur"].uncertain is True
        assert res.obstruction.status == ObstructionStatus.UNCERTAIN
        assert res.score_breakdown["obstruction"]["penalty_applied"] == 0.0
