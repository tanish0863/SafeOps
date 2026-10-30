"""
test_glare.py
-------------
Tests for the glare / overexposure detection check.
"""

import numpy as np
import pytest

from quality_guardian.checks import check_glare


class TestGlareNormalImages:
    """Normal images should pass the glare check."""

    def test_normal_image_passes(self, normal_image):
        result = check_glare(normal_image)
        assert result.passed is True

    def test_dark_image_passes_glare_check(self, dark_image):
        """Dark images have no bright pixels → should pass glare check."""
        result = check_glare(dark_image)
        assert result.passed is True

    def test_no_issues_for_normal_image(self, normal_image):
        result = check_glare(normal_image)
        assert len(result.issues) == 0
        assert len(result.recommendations) == 0


class TestGlareExcessiveBrightRegions:
    """Images with large bright/white regions should fail."""

    def test_white_image_fails(self, bright_glare_image):
        result = check_glare(bright_glare_image)
        assert result.passed is False

    def test_white_image_has_high_fraction(self, bright_glare_image):
        result = check_glare(bright_glare_image)
        assert result.value > 0.05

    def test_partial_glare_image_fails(self, partial_glare_image):
        """An image with ~10% glare patch should exceed the 5% default threshold."""
        result = check_glare(partial_glare_image)
        assert result.passed is False

    def test_issues_reported_for_glare(self, bright_glare_image):
        result = check_glare(bright_glare_image)
        assert len(result.issues) > 0

    def test_recommendations_given_for_glare(self, bright_glare_image):
        result = check_glare(bright_glare_image)
        assert len(result.recommendations) > 0


class TestGlareCustomThresholds:
    """Custom threshold values should be respected."""

    def test_strict_threshold_fails_partial_glare(self, partial_glare_image):
        # Require 0 glare pixels – even normal images with 1 bright pixel fail
        result = check_glare(partial_glare_image, bright_pixel_fraction_threshold=0.001)
        assert result.passed is False

    def test_lenient_threshold_passes_white_image(self, bright_glare_image):
        # Allow up to 100% bright pixels
        result = check_glare(bright_glare_image, bright_pixel_fraction_threshold=1.0)
        assert result.passed is True

    def test_result_value_is_fraction(self, normal_image):
        result = check_glare(normal_image)
        assert 0.0 <= result.value <= 1.0

    def test_details_contains_expected_keys(self, normal_image):
        result = check_glare(normal_image)
        for key in ("bright_pixel_fraction", "glare_pixel_count",
                    "total_pixels", "bright_value_threshold"):
            assert key in result.details


class TestGlareEdgeCases:
    """Edge cases for glare detection."""

    def test_single_black_pixel_passes(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = check_glare(img)
        assert result.passed is True
        assert result.value == 0.0

    def test_single_white_pixel_fails(self):
        img = np.full((1, 1, 3), 255, dtype=np.uint8)
        result = check_glare(img)
        # 100% of pixels are bright and desaturated → fraction = 1.0 > default 0.05
        assert result.passed is False

    def test_highly_saturated_bright_pixels_do_not_count_as_glare(self):
        """
        Bright red/green/blue pixels (high V, high S) are NOT glare.
        Glare is white/washed-out (high V, low S).
        """
        # Pure red in BGR: (0, 0, 255) → in HSV: Hue=0, Sat=255, Val=255
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :] = (0, 0, 255)  # solid red
        result = check_glare(
            img,
            bright_value_threshold=230,
            bright_saturation_threshold=30,
        )
        # Red is highly saturated → saturation >> 30 → NOT counted as glare
        assert result.passed is True
        assert result.value < 0.01
