"""Image quality metrics for matching RGB uint8 images."""
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def denoising_improvement(noisy_mse: float, reconstructed_mse: float) -> float:
    """Positive means reconstruction reduced error; negative means it increased it."""
    return float(noisy_mse - reconstructed_mse)


def estimated_latent_ratio(width: int, height: int, latent_elements: int, bit_depth: int) -> float:
    """Estimate working-image RGB bits divided by quantized latent bits."""
    if width <= 0 or height <= 0 or latent_elements <= 0:
        raise ValueError("Image dimensions and latent element count must be positive.")
    if bit_depth not in (4, 8, 16):
        raise ValueError("Bit depth must be 4, 8 or 16.")
    return float((width * height * 24) / (latent_elements * bit_depth))


def calculate_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Calculate MSE, PSNR and SSIM for two same-size RGB uint8 images."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Metrics require matching RGB image dimensions.")
    a, b = reference.astype(np.uint8), candidate.astype(np.uint8)
    mse = float(mean_squared_error(a, b))
    # JSON has no standard representation for infinity; None means perfect PSNR.
    psnr = None if mse == 0 else float(peak_signal_noise_ratio(a, b, data_range=255))
    min_side = min(a.shape[:2])
    if np.array_equal(a, b):
        ssim = 1.0
    else:
        # SSIM needs a neighborhood. Upscale unusually tiny but valid images
        # just for the metric so that its smallest supported window is 3x3.
        if min_side < 3:
            target = (max(3, a.shape[1]), max(3, a.shape[0]))
            a = np.asarray(Image.fromarray(a).resize(target), dtype=np.uint8)
            b = np.asarray(Image.fromarray(b).resize(target), dtype=np.uint8)
            min_side = min(a.shape[:2])
        window_size = min(7, min_side if min_side % 2 else min_side - 1)
        ssim = float(
            structural_similarity(
                a,
                b,
                channel_axis=2,
                data_range=255,
                win_size=window_size,
            )
        )
    return {"mse": mse, "psnr": psnr, "ssim": ssim}
