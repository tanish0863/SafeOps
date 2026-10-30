"""
test_assessment.py
------------------
Tests for the quality assessment engine (assessment.py).
Also covers:
    - Invalid image path handling (image_processor.py)
    - Corrupted image handling
    - Obstruction detection (obstruction.py)
    - Score calculation logic
"""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from quality_guardian.assessment import (
    AssessmentConfig,
    AssessmentResult,
    assess_frame,
)
from quality_guardian.obstruction import (
    ObstructionStatus,
    check_obstruction_single_frame,
    check_frame_similarity,
)
from quality_guardian.image_processor import load_image


# ---------------------------------------------------------------------------
# A. Score and status tests
# ---------------------------------------------------------------------------

class TestAssessmentScore:
    """Score and status are derived correctly from check results."""

    def test_normal_image_scores_high(self, normal_image):
        result = assess_frame(normal_image)
        assert result.overall_score >= 70, (
            f"Normal noise image should score well, got {result.overall_score}"
        )

    def test_good_status_score_range(self, normal_image):
        config = AssessmentConfig()
        result = assess_frame(normal_image, config)
        if result.overall_status == "GOOD":
            assert result.overall_score >= config.score_good_threshold

    def test_black_image_scores_low(self, very_dark_image):
        """A completely black image should fail darkness check and score poorly."""
        result = assess_frame(very_dark_image)
        assert result.overall_score <= 70

    def test_score_is_in_valid_range(self, normal_image):
        result = assess_frame(normal_image)
        assert 0.0 <= result.overall_score <= 100.0

    def test_score_is_in_valid_range_dark(self, very_dark_image):
        result = assess_frame(very_dark_image)
        assert 0.0 <= result.overall_score <= 100.0

    def test_overall_status_is_valid_string(self, normal_image):
        result = assess_frame(normal_image)
        assert result.overall_status in ("GOOD", "WARNING", "POOR")

    def test_score_breakdown_has_all_checks(self, normal_image):
        result = assess_frame(normal_image)
        for key in ("darkness", "blur", "resolution", "glare", "obstruction"):
            assert key in result.score_breakdown

    def test_multiple_failures_reduce_score(self, very_dark_image):
        """
        A black image should fail darkness AND potentially other checks,
        resulting in a lower score than just one failure.
        """
        # Assess a normal image and a dark image; dark should score lower
        normal_result = assess_frame(np.full((480, 640, 3), 128, dtype=np.uint8))
        dark_result = assess_frame(very_dark_image)
        assert dark_result.overall_score < normal_result.overall_score


class TestAssessmentPenalties:
    """Penalty weights are applied correctly."""

    def test_darkness_penalty_applied_for_dark_image(self, very_dark_image):
        config = AssessmentConfig()
        result = assess_frame(very_dark_image, config)
        breakdown = result.score_breakdown["darkness"]
        assert breakdown["failed"] is True
        assert breakdown["penalty_applied"] == config.darkness_penalty

    def test_no_penalty_for_passing_darkness(self, normal_image):
        result = assess_frame(normal_image)
        if result.score_breakdown["darkness"]["failed"] is False:
            assert result.score_breakdown["darkness"]["penalty_applied"] == 0

    def test_uncertain_obstruction_does_not_penalise(self, uniform_obstruction_image):
        """
        A uniform grey image may produce UNCERTAIN obstruction.
        UNCERTAIN must not reduce the score.
        """
        result = assess_frame(uniform_obstruction_image)
        breakdown = result.score_breakdown["obstruction"]
        if result.obstruction.status == ObstructionStatus.UNCERTAIN:
            assert breakdown["penalty_applied"] == 0.0

    def test_score_clamped_to_zero_minimum(self):
        """
        Even with maximum penalties the score should not go below 0.
        """
        # Manufacture a config with impossibly strict thresholds so all checks fail
        config = AssessmentConfig(
            darkness_mean_threshold=300.0,    # impossible
            blur_laplacian_variance_threshold=999999.0,
            resolution_min_width=10000,
            resolution_min_height=10000,
            glare_bright_pixel_fraction_threshold=0.0,
            darkness_penalty=50.0,
            blur_penalty=50.0,
            resolution_penalty=50.0,
            glare_penalty=50.0,
            obstruction_penalty=50.0,
        )
        img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = assess_frame(img, config)
        assert result.overall_score >= 0.0

    def test_custom_config_respected(self, normal_image):
        config = AssessmentConfig(score_good_threshold=99.0)
        result = assess_frame(normal_image, config)
        # With a threshold of 99, even a normal image is unlikely to be GOOD
        if result.overall_score < 99.0:
            assert result.overall_status in ("WARNING", "POOR")


class TestAssessmentOutput:
    """Output structure tests."""

    def test_result_has_all_checks(self, normal_image):
        result = assess_frame(normal_image)
        for name in ("darkness", "blur", "resolution", "glare"):
            assert name in result.checks

    def test_result_has_obstruction(self, normal_image):
        result = assess_frame(normal_image)
        assert result.obstruction is not None

    def test_disclaimer_is_present(self, normal_image):
        result = assess_frame(normal_image)
        assert isinstance(result.disclaimer, str)
        assert len(result.disclaimer) > 0

    def test_issues_and_recommendations_are_lists(self, very_dark_image):
        result = assess_frame(very_dark_image)
        assert isinstance(result.all_issues, list)
        assert isinstance(result.all_recommendations, list)


# ---------------------------------------------------------------------------
# B. Obstruction detection tests
# ---------------------------------------------------------------------------

class TestObstructionSingleFrame:
    """Single-frame obstruction heuristic tests."""

    def test_normal_image_is_clear(self, normal_image):
        result = check_obstruction_single_frame(normal_image)
        assert result.status in (ObstructionStatus.CLEAR, ObstructionStatus.UNCERTAIN)

    def test_sharp_image_is_clear(self, sharp_image):
        result = check_obstruction_single_frame(sharp_image)
        assert result.status == ObstructionStatus.CLEAR

    def test_uniform_grey_is_suspicious(self, uniform_obstruction_image):
        """A completely uniform image has no edges → should be UNCERTAIN or OBSTRUCTED."""
        result = check_obstruction_single_frame(uniform_obstruction_image)
        assert result.status in (ObstructionStatus.UNCERTAIN, ObstructionStatus.OBSTRUCTED)

    def test_result_has_confidence(self, normal_image):
        result = check_obstruction_single_frame(normal_image)
        assert 0.0 <= result.confidence <= 1.0

    def test_note_is_present(self, normal_image):
        result = check_obstruction_single_frame(normal_image)
        assert isinstance(result.note, str) and len(result.note) > 0

    def test_details_has_edge_density(self, normal_image):
        result = check_obstruction_single_frame(normal_image)
        assert "edge_density" in result.details
        assert "intensity_std" in result.details


class TestFrameSimilarity:
    """Consecutive frame similarity obstruction tests."""

    def test_identical_frames_flagged(self, uniform_obstruction_image):
        result = check_frame_similarity(
            uniform_obstruction_image, uniform_obstruction_image,
            similarity_high=0.99,
        )
        assert result.status in (ObstructionStatus.UNCERTAIN, ObstructionStatus.OBSTRUCTED)

    def test_different_frames_are_clear(self):
        """Two clearly different high-variance frames should have low similarity."""
        np.random.seed(1)
        frame_a = np.random.randint(0, 128, (480, 640, 3), dtype=np.uint8)   # dark-range noise
        np.random.seed(2)
        frame_b = np.random.randint(128, 255, (480, 640, 3), dtype=np.uint8) # light-range noise
        result = check_frame_similarity(frame_a, frame_b)
        assert result.status == ObstructionStatus.CLEAR

    def test_similar_noisy_frames_are_suspicious(self):
        """Two frames from the same distribution should be very similar."""
        np.random.seed(42)
        frame_a = np.random.randint(100, 110, (480, 640, 3), dtype=np.uint8)
        frame_b = frame_a.copy()
        result = check_frame_similarity(frame_a, frame_b, similarity_high=0.995)
        assert result.status in (ObstructionStatus.UNCERTAIN, ObstructionStatus.OBSTRUCTED)

    def test_similarity_result_has_confidence(self, normal_image):
        result = check_frame_similarity(normal_image, normal_image)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# C. Invalid path and corrupted input tests
# ---------------------------------------------------------------------------

class TestInvalidInputHandling:
    """Error handling for bad inputs."""

    def test_missing_file_raises_file_not_found(self, tmp_path):
        from quality_guardian.image_processor import load_image
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image(tmp_path / "nonexistent.jpg")

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        fake_file = tmp_path / "image.bmp"
        fake_file.write_bytes(b"fake data")
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(fake_file)

    def test_corrupted_image_raises_value_error(self, tmp_path):
        """A file with a .jpg extension but garbage content should fail to decode."""
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_bytes(b"\x00\x01\x02\x03\xFF\xFE garbage data not jpeg")
        with pytest.raises(ValueError, match="Could not decode"):
            load_image(corrupted)

    def test_video_missing_file_raises(self):
        from quality_guardian.video_processor import process_video
        with pytest.raises(FileNotFoundError):
            process_video("nonexistent_video.mp4")

    def test_video_unsupported_extension_raises(self, tmp_path):
        from quality_guardian.video_processor import process_video
        bad_video = tmp_path / "clip.mkv"
        bad_video.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            process_video(bad_video)
