"""Unit tests for image preprocessing, noise, filters and latent quantization."""
from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from backend.filters import add_gaussian_noise, apply_filter
from backend.preprocessing import decode_image, prepare_for_vae, resize_for_model
from backend.quantization import quantize_latents


def png_bytes(pixels: np.ndarray) -> bytes:
    """Encode test pixels as PNG bytes for the preprocessing tests."""
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_zero_noise_returns_an_identical_copy():
    image = np.full((12, 10, 3), 100, dtype=np.uint8)

    result = add_gaussian_noise(image, 0)

    assert np.array_equal(image, result)
    assert result is not image


def test_noise_is_repeatable_and_clipped_to_byte_range():
    image = np.full((20, 20, 3), 250, dtype=np.uint8)

    first_result = add_gaussian_noise(image, 50, seed=4)
    second_result = add_gaussian_noise(image, 50, seed=4)

    assert np.array_equal(first_result, second_result)
    assert first_result.max() <= 255


def test_noise_level_above_supported_range_is_rejected():
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        add_gaussian_noise(image, 51)


def test_decode_keeps_original_dimensions_and_aspect_ratio():
    pixels = np.zeros((80, 160, 3), dtype=np.uint8)

    decoded, original_size = decode_image(png_bytes(pixels))

    assert decoded.shape[:2] == (80, 160)
    assert original_size == (160, 80)


def test_model_resize_preserves_aspect_ratio():
    image = np.zeros((800, 1600, 3), dtype=np.uint8)

    resized = resize_for_model(image)

    assert resized.shape[:2] == (256, 512)


def test_vae_preprocessing_pads_dimensions_to_multiple_of_eight():
    image = np.ones((11, 17, 3), dtype=np.uint8)

    padded, original_shape = prepare_for_vae(image)

    assert padded.shape == (16, 24, 3)
    assert original_shape == (11, 17)
    assert np.array_equal(padded[:11, :17], image)


@pytest.mark.parametrize("filter_name", ["None", "Median Filter", "Gaussian Filter"])
def test_filters_keep_image_shape_and_dtype(filter_name):
    image = np.full((9, 9, 3), 120, dtype=np.uint8)

    filtered = apply_filter(image, filter_name)

    assert filtered.shape == image.shape
    assert filtered.dtype == np.uint8


@pytest.mark.parametrize("bit_depth", [4, 8, 16])
def test_quantization_returns_finite_latents(bit_depth):
    latent = torch.linspace(-2, 3, 100).reshape(1, 1, 10, 10)

    restored, details = quantize_latents(latent, bit_depth)

    assert restored.shape == latent.shape
    assert torch.isfinite(restored).all()
    assert details["levels"] == 2**bit_depth - 1


def test_constant_latents_do_not_divide_by_zero():
    latent = torch.full((1, 4, 2, 2), 2.5)

    restored, _ = quantize_latents(latent, 4)

    assert torch.equal(restored, latent)


def test_unsupported_bit_depth_is_rejected():
    with pytest.raises(ValueError):
        quantize_latents(torch.ones(2), 5)
