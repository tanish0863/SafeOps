"""
test_darkness.py
----------------
Tests for the darkness detection check.
"""

import numpy as np
import pytest

from quality_guardian.checks import check_darkness


class TestDarknessNormalLighting:
    """Normal images should pass the darkness check."""

    def test_normal_noise_image_passes(self, normal_image):
        result = check_darkness(normal_image)
        assert result.passed is True
        assert result.name == "darkness"

    def test_mean_intensity_is_reasonable(self, normal_image):
        result = check_darkness(normal_image)
        assert 40 < result.value < 220, "Mean intensity should be mid-range for a normal image"

    def test_no_issues_for_good_image(self, normal_image):
        result = check_darkness(normal_image)
        assert len(result.issues) == 0
        assert len(result.recommendations) == 0

    def test_high_res_normal_image_passes(self, high_res_image):
        result = check_darkness(high_res_image)
        assert result.passed is True


class TestDarknessVeryDarkImages:
    """Dark images should fail the darkness check."""

    def test_black_image_fails(self, very_dark_image):
        result = check_darkness(very_dark_image)
        assert result.passed is False

    def test_dark_image_fails(self, dark_image):
        result = check_darkness(dark_image)
        assert result.passed is False

    def test_mean_intensity_near_zero_for_black(self, very_dark_image):
        result = check_darkness(very_dark_image)
        assert result.value < 20

    def test_issues_reported_for_dark_image(self, dark_image):
        result = check_darkness(dark_image)
        assert len(result.issues) > 0

    def test_recommendations_given_for_dark_image(self, dark_image):
        result = check_darkness(dark_image)
        assert len(result.recommendations) > 0

    def test_dark_pixel_fraction_high_for_black_image(self, very_dark_image):
        result = check_darkness(very_dark_image)
        fraction = result.details["dark_pixel_fraction"]
        assert fraction > 0.9, "Almost all pixels should be considered dark"


class TestDarknessCustomThresholds:
    """Custom threshold values should be respected."""

    def test_very_strict_threshold_fails_normal_image(self, normal_image):
        # Require mean > 200: a ~128-mean image should fail
        result = check_darkness(normal_image, mean_intensity_threshold=200.0)
        assert result.passed is False

    def test_very_lenient_threshold_passes_dark_image(self, dark_image):
        # Require mean > 5 only: dark_image (mean ~10) should pass.
        # Set dark_pixel_fraction_threshold to 1.01 (above the maximum possible
        # fraction of 1.0) so the fraction check never triggers.
        result = check_darkness(dark_image, mean_intensity_threshold=5.0,
                                dark_pixel_fraction_threshold=1.01)
        assert result.passed is True

    def test_details_contains_expected_keys(self, normal_image):
        result = check_darkness(normal_image)
        for key in ("mean_intensity", "dark_pixel_fraction",
                    "dark_pixel_value_threshold", "too_dark_by_mean"):
            assert key in result.details


class TestDarknessEdgeCases:
    """Edge cases for darkness detection."""

    def test_single_pixel_image(self):
        img = np.array([[[50, 50, 50]]], dtype=np.uint8)
        result = check_darkness(img, mean_intensity_threshold=60.0)
        # Single pixel at intensity 50 should fail
        assert result.passed is False

    def test_white_image_passes(self):
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        result = check_darkness(img)
        assert result.passed is True

    def test_result_value_equals_mean_intensity(self):
        # Create image with known exact mean
        img = np.full((100, 100, 3), 80, dtype=np.uint8)
        result = check_darkness(img, mean_intensity_threshold=60.0)
        assert abs(result.value - 80.0) < 1.0
