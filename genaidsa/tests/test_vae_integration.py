"""Real model integration tests; no mocked encoder, decoder or output images."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
import torch

from backend.image_pipeline import process_image
from backend.metrics import calculate_metrics
from backend.model_loader import load_vae
from backend.quantization import quantize_latents


@pytest.fixture(scope="module")
def real_vae():
    """Load the actual pretrained VAE once for this integration test module."""
    return load_vae()


@pytest.fixture(scope="module")
def sample_png():
    """Create a small, deterministic RGB image for actual model inference."""
    y_coordinates, x_coordinates = np.mgrid[0:32, 0:40]
    pixels = np.stack(
        (
            x_coordinates * 6,
            y_coordinates * 7,
            (x_coordinates + y_coordinates) * 3,
        ),
        axis=-1,
    )
    pixels = pixels.clip(0, 255).astype(np.uint8)

    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def run_real_image_pipeline(
    image_bytes: bytes,
    model_and_device,
    bit_depth: int = 8,
    noise_level: int = 0,
    filter_name: str = "None",
) -> dict:
    """Run the actual image pipeline using the pretrained VAE fixture."""
    model, device = model_and_device
    return process_image(
        image_bytes,
        noise_level,
        filter_name,
        bit_depth,
        model,
        device,
    )


def test_latent_compression_uses_real_vae_dimensions(real_vae, sample_png):
    result = run_real_image_pipeline(sample_png, real_vae, bit_depth=8)
    latent_shape = result["metadata"]["latent_dimensions"]
    latent_element_count = int(np.prod(latent_shape))
    working_dimensions = result["metadata"]["working_image_dimensions"]

    expected_ratio = (
        working_dimensions["width"]
        * working_dimensions["height"]
        * 24
        / (latent_element_count * 8)
    )

    assert result["metrics"]["estimated_working_image_latent_ratio"] == pytest.approx(
        expected_ratio
    )


def test_information_loss_is_measured_on_real_encoded_latents(real_vae, sample_png):
    model, device = real_vae
    image = np.array(Image.open(BytesIO(sample_png)).convert("RGB"), copy=True)
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.float().to(device).div(127.5).sub(1)

    # Run the actual encoder, quantizers and decoder at both precision settings.
    with torch.inference_mode():
        posterior = model.encode(image_tensor).latent_dist
        encoded_latent = posterior.mode() * model.config.scaling_factor
        four_bit_latent, _ = quantize_latents(encoded_latent, 4)
        sixteen_bit_latent, _ = quantize_latents(encoded_latent, 16)
        four_bit_image = model.decode(
            four_bit_latent / model.config.scaling_factor
        ).sample
        sixteen_bit_image = model.decode(
            sixteen_bit_latent / model.config.scaling_factor
        ).sample

    four_bit_error = torch.mean((encoded_latent - four_bit_latent) ** 2).item()
    sixteen_bit_error = torch.mean((encoded_latent - sixteen_bit_latent) ** 2).item()

    assert four_bit_error > 0
    assert sixteen_bit_error <= four_bit_error
    assert not torch.equal(four_bit_image, sixteen_bit_image)


def test_denoising_reports_real_error_change_without_guaranteeing_improvement(
    real_vae,
    sample_png,
):
    result = run_real_image_pipeline(
        sample_png,
        real_vae,
        bit_depth=8,
        noise_level=35,
        filter_name="Median Filter",
    )
    metrics = result["metrics"]
    expected_change = (
        metrics["original_vs_noisy_mse"]
        - metrics["original_vs_reconstructed_mse"]
    )

    assert metrics["original_vs_noisy_mse"] > 0
    assert metrics["denoising_improvement"] == pytest.approx(expected_change)
    assert np.isfinite(metrics["original_vs_reconstructed_mse"])


def test_reconstruction_quality_matches_returned_real_images(real_vae, sample_png):
    result = run_real_image_pipeline(sample_png, real_vae, bit_depth=8)
    original_image = np.asarray(
        Image.open(BytesIO(base64.b64decode(result["original_png"]))).convert("RGB")
    )
    reconstructed_image = np.asarray(
        Image.open(BytesIO(base64.b64decode(result["reconstructed_png"]))).convert("RGB")
    )

    actual_metrics = calculate_metrics(original_image, reconstructed_image)

    assert reconstructed_image.shape == original_image.shape == (32, 40, 3)
    assert result["metrics"]["original_vs_reconstructed_mse"] == pytest.approx(
        actual_metrics["mse"]
    )
    assert result["metrics"]["ssim"] == pytest.approx(actual_metrics["ssim"])
    assert result["metrics"]["psnr"] == pytest.approx(actual_metrics["psnr"])
