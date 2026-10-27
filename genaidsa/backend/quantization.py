"""Uniformly quantize latent values and dequantize them for the VAE decoder."""
import torch


def quantize_latents(latents: torch.Tensor, bit_depth: int) -> tuple[torch.Tensor, dict]:
    """Map finite latent values to the requested levels and back to floats."""
    if bit_depth not in (4, 8, 16):
        raise ValueError("Bit depth must be 4, 8 or 16.")
    if not torch.isfinite(latents).all():
        raise ValueError("Latent values must be finite.")
    low, high = latents.amin(), latents.amax()
    levels = (1 << bit_depth) - 1
    span = high - low
    if span.item() == 0:
        restored = latents.clone()
    else:
        # Quantize to integer-like levels, then restore values to the latent range.
        indices = torch.round((latents - low) / span * levels)
        restored = indices / levels * span + low
    details = {"min": float(low.item()), "max": float(high.item()), "levels": levels}
    return restored, details
