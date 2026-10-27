"""Reusable image workflow, separate from the HTTP endpoint and Streamlit UI."""
import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

try:  # Package import in pytest, top-level module import in Azure Functions.
    from .filters import add_gaussian_noise, apply_filter
    from .metrics import calculate_metrics, denoising_improvement, estimated_latent_ratio
    from .preprocessing import decode_image, prepare_for_vae, resize_for_model
    from .quantization import quantize_latents
except ImportError:  # pragma: no cover - exercised by the Functions host
    from filters import add_gaussian_noise, apply_filter
    from metrics import calculate_metrics, denoising_improvement, estimated_latent_ratio
    from preprocessing import decode_image, prepare_for_vae, resize_for_model
    from quantization import quantize_latents

VALID_FILTERS = {"None", "Median Filter", "Gaussian Filter"}
VALID_BITS = {4, 8, 16}


def _validate_settings(noise_level: float, filter_name: str, bit_depth: int) -> None:
    """Reject unsupported controls before processing an image."""
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, float))
        or not 0 <= noise_level <= 50
    ):
        raise ValueError("noise_level must be a number from 0 to 50.")
    if filter_name not in VALID_FILTERS:
        raise ValueError("filter must be None, Median Filter or Gaussian Filter.")
    if isinstance(bit_depth, bool) or bit_depth not in VALID_BITS:
        raise ValueError("bit_depth must be 4, 8 or 16.")


def _png_base64(array: np.ndarray) -> str:
    """Encode an RGB image array as a Base64 PNG string."""
    buffer = BytesIO()
    image = Image.fromarray(array.astype(np.uint8))
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _to_model_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert RGB uint8 pixels into a batch normalized to [-1, 1]."""
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    image_tensor = image_tensor.to(device=device, dtype=torch.float32)
    return image_tensor.div(127.5).sub(1.0)


def _decode_latent_image(
    model,
    quantized_latent: torch.Tensor,
) -> np.ndarray:
    """Decode quantized latents and return RGB uint8 pixels."""
    decoded = model.decode(quantized_latent / model.config.scaling_factor).sample
    decoded = decoded[0].detach().float().clamp(-1, 1)
    decoded = decoded.add(1).mul(127.5).round().byte()
    return decoded.permute(1, 2, 0).to(device="cpu").numpy()


def process_image(
    raw: bytes,
    noise_level: float,
    filter_name: str,
    bit_depth: int,
    model,
    device: torch.device,
) -> dict:
    """Run preprocessing, VAE compression, reconstruction and metrics."""
    _validate_settings(noise_level, filter_name, bit_depth)

    # Keep the full-size image for the final comparison; resize only its VAE copy.
    original, (original_w, original_h) = decode_image(raw)
    working_image = resize_for_model(original)
    noisy = add_gaussian_noise(working_image, float(noise_level))
    filtered = apply_filter(noisy, filter_name)
    padded_image, (working_height, working_width) = prepare_for_vae(filtered)
    model_input = _to_model_tensor(padded_image, device)

    # Use the posterior mode for repeatable encoding, then apply the VAE scale.
    with torch.inference_mode():
        posterior = model.encode(model_input).latent_dist
        latent = posterior.mode() * model.config.scaling_factor
        quantized_latent, _ = quantize_latents(latent, bit_depth)
        reconstructed_working_image = _decode_latent_image(model, quantized_latent)

    # Remove VAE padding, then restore the uploaded image's original dimensions.
    reconstructed_working_image = reconstructed_working_image[:working_height, :working_width]
    reconstruction = Image.fromarray(reconstructed_working_image).resize(
        (original_w, original_h), Image.Resampling.LANCZOS
    )
    reconstruction = np.asarray(reconstruction, dtype=np.uint8)

    noisy_at_original_size = Image.fromarray(noisy).resize(
        (original_w, original_h), Image.Resampling.LANCZOS
    )
    noisy_at_original_size = np.asarray(noisy_at_original_size, dtype=np.uint8)

    noisy_metrics = calculate_metrics(original, noisy_at_original_size)
    reconstruction_metrics = calculate_metrics(original, reconstruction)

    # Estimate bits for the resized RGB working image versus the quantized latent.
    latent_elements = int(latent.numel())
    compression_ratio = estimated_latent_ratio(
        working_width,
        working_height,
        latent_elements,
        bit_depth,
    )

    metrics = {
        "original_vs_noisy_mse": noisy_metrics["mse"],
        "original_vs_reconstructed_mse": reconstruction_metrics["mse"],
        "psnr": reconstruction_metrics["psnr"],
        "ssim": reconstruction_metrics["ssim"],
        "estimated_working_image_latent_ratio": compression_ratio,
        "denoising_improvement": denoising_improvement(
            noisy_metrics["mse"], reconstruction_metrics["mse"]
        ),
    }
    metadata = {
        "input_dimensions": {"width": original_w, "height": original_h},
        "latent_dimensions": list(latent.shape),
        "working_image_dimensions": {"width": working_width, "height": working_height},
        "noise_level": float(noise_level),
        "filter": filter_name,
        "bit_depth": bit_depth,
        "device": str(device),
        "model": "stabilityai/sd-vae-ft-mse",
    }

    return {
        "original_png": _png_base64(original),
        "noisy_png": _png_base64(noisy_at_original_size),
        "reconstructed_png": _png_base64(reconstruction),
        "metrics": metrics,
        "metadata": metadata,
    }
