"""
test_blur.py
------------
Tests for the blur detection check.
"""

import numpy as np
import pytest

from quality_guardian.checks import check_blur


class TestBlurSharpImages:
    """Sharp images should pass the blur check."""

    def test_checkerboard_passes(self, sharp_image):
        """Checkerboard has very high Laplacian variance."""
        result = check_blur(sharp_image)
        assert result.passed is True

    def test_sharp_laplacian_variance_is_high(self, sharp_image):
        result = check_blur(sharp_image)
        # A perfect checkerboard should have variance well above 100
        assert result.value > 100

    def test_no_issues_for_sharp_image(self, sharp_image):
        result = check_blur(sharp_image)
        assert len(result.issues) == 0
        assert len(result.recommendations) == 0

    def test_normal_noise_image_passes(self, normal_image):
        """Random noise is inherently high-frequency → should be sharp."""
        result = check_blur(normal_image)
        assert result.passed is True


class TestBlurBlurryImages:
    """Blurry (smooth) images should fail the blur check."""

    def test_smooth_image_fails(self, blurry_image):
        result = check_blur(blurry_image)
        assert result.passed is False

    def test_blurry_laplacian_variance_is_low(self, blurry_image):
        result = check_blur(blurry_image)
        assert result.value < 100

    def test_issues_reported_for_blurry_image(self, blurry_image):
        result = check_blur(blurry_image)
        assert len(result.issues) > 0

    def test_recommendations_for_blurry_image(self, blurry_image):
        result = check_blur(blurry_image)
        assert len(result.recommendations) > 0


class TestBlurCustomThresholds:
    """Custom thresholds should be respected."""

    def test_strict_threshold_fails_normal_image(self, normal_image):
        # Require Laplacian variance > 99999: almost anything will fail
        result = check_blur(normal_image, laplacian_variance_threshold=99999.0)
        assert result.passed is False

    def test_zero_threshold_passes_everything(self, blurry_image):
        result = check_blur(blurry_image, laplacian_variance_threshold=0.0)
        assert result.passed is True

    def test_result_name_is_blur(self, normal_image):
        result = check_blur(normal_image)
        assert result.name == "blur"

    def test_details_contains_laplacian_variance(self, sharp_image):
        result = check_blur(sharp_image)
        assert "laplacian_variance" in result.details
        assert result.details["laplacian_variance"] == result.value


class TestBlurEdgeCases:
    """Edge case inputs for blur detection."""

    def test_single_pixel_image(self):
        """A single pixel has no edges; variance should be 0."""
        img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        result = check_blur(img)
        assert result.value == 0.0
        assert result.passed is False  # 0 < default threshold 100

    def test_solid_colour_image_fails(self):
        """A completely solid image has zero Laplacian variance."""
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = check_blur(img)
        assert result.value == 0.0
        assert result.passed is False
