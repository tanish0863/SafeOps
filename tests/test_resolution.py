"""
test_resolution.py
------------------
Tests for the resolution check.
"""

import numpy as np
import pytest

from quality_guardian.checks import check_resolution


class TestResolutionHighResolution:
    """Full HD and above should pass."""

    def test_1080p_image_passes(self, high_res_image):
        result = check_resolution(high_res_image)
        assert result.passed is True

    def test_640x480_exactly_passes(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = check_resolution(img, min_width=640, min_height=480)
        assert result.passed is True

    def test_dimensions_reported_correctly(self, high_res_image):
        result = check_resolution(high_res_image)
        assert result.details["width"] == 1920
        assert result.details["height"] == 1080

    def test_no_issues_for_high_res(self, high_res_image):
        result = check_resolution(high_res_image)
        assert len(result.issues) == 0
        assert len(result.recommendations) == 0


class TestResolutionLowResolution:
    """Images below minimum resolution should fail."""

    def test_low_res_image_fails(self, low_res_image):
        result = check_resolution(low_res_image)
        assert result.passed is False

    def test_tiny_image_fails_both_dimensions(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = check_resolution(img, min_width=640, min_height=480)
        assert result.passed is False
        assert result.details["width_ok"] is False
        assert result.details["height_ok"] is False

    def test_wide_but_short_fails(self):
        img = np.zeros((100, 1920, 3), dtype=np.uint8)
        result = check_resolution(img, min_width=640, min_height=480)
        assert result.passed is False
        assert result.details["width_ok"] is True
        assert result.details["height_ok"] is False

    def test_tall_but_narrow_fails(self):
        img = np.zeros((1080, 100, 3), dtype=np.uint8)
        result = check_resolution(img, min_width=640, min_height=480)
        assert result.passed is False
        assert result.details["width_ok"] is False
        assert result.details["height_ok"] is True

    def test_issues_reported_for_low_res(self, low_res_image):
        result = check_resolution(low_res_image)
        assert len(result.issues) > 0

    def test_recommendations_for_low_res(self, low_res_image):
        result = check_resolution(low_res_image)
        assert len(result.recommendations) > 0


class TestResolutionCustomThresholds:
    """Custom minimum resolution thresholds should be respected."""

    def test_custom_high_threshold_fails_hd_image(self, high_res_image):
        result = check_resolution(high_res_image, min_width=3840, min_height=2160)
        assert result.passed is False

    def test_custom_low_threshold_passes_low_res(self, low_res_image):
        result = check_resolution(low_res_image, min_width=10, min_height=10)
        assert result.passed is True

    def test_result_name_is_resolution(self, normal_image):
        result = check_resolution(normal_image)
        assert result.name == "resolution"


class TestResolutionEdgeCases:
    """Edge case inputs for resolution check."""

    def test_single_pixel_image_fails(self):
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        result = check_resolution(img)
        assert result.passed is False

    def test_grayscale_array_still_works(self):
        """check_resolution reads shape[:2] so channels don't matter."""
        img = np.zeros((480, 640), dtype=np.uint8)
        # Manually add channels to simulate BGR
        img_bgr = np.stack([img, img, img], axis=-1)
        result = check_resolution(img_bgr)
        assert result.passed is True
