"""Unit tests for quality metrics and their reporting formulas."""
import numpy as np
import pytest

from backend.metrics import (
    calculate_metrics,
    denoising_improvement,
    estimated_latent_ratio,
)


def test_denoising_improvement_sign_matches_error_change():
    improved_error = denoising_improvement(120.0, 80.0)
    increased_error = denoising_improvement(80.0, 120.0)

    assert improved_error == 40.0
    assert increased_error == -40.0


def test_latent_ratio_uses_working_image_rgb_dimensions():
    expected_ratio = (200 * 100 * 24) / (1300 * 8)

    ratio = estimated_latent_ratio(200, 100, 1300, 8)

    assert ratio == pytest.approx(expected_ratio)


def test_identical_images_have_zero_error_and_perfect_similarity():
    image = np.full((16, 16, 3), 90, dtype=np.uint8)

    result = calculate_metrics(image, image.copy())

    assert result["mse"] == 0
    assert result["psnr"] is None
    assert result["ssim"] == 1


def test_changed_pixels_produce_real_metric_values():
    original = np.zeros((16, 16, 3), dtype=np.uint8)
    changed = original.copy()
    changed[4:12, 4:12] = 40

    result = calculate_metrics(original, changed)

    assert result["mse"] > 0
    assert 0 <= result["ssim"] < 1
    assert result["psnr"] > 0


def test_metrics_reject_images_with_different_dimensions():
    first_image = np.zeros((8, 8, 3), dtype=np.uint8)
    second_image = np.zeros((9, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_metrics(first_image, second_image)
