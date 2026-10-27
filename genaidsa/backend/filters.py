"""Add pixel-space Gaussian noise and apply optional denoising filters."""
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def add_gaussian_noise(image: np.ndarray, noise_level: float, seed: int | None = None) -> np.ndarray:
    """Add clipped Gaussian noise whose standard deviation is noise_level."""
    if not 0 <= noise_level <= 50:
        raise ValueError("Noise level must be between 0 and 50.")
    if noise_level == 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0, noise_level, image.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def apply_filter(image: np.ndarray, filter_name: str) -> np.ndarray:
    """Apply no filter, a 3x3 median filter or a Gaussian filter."""
    if filter_name == "None":
        return image.copy()
    if filter_name == "Median Filter":
        return median_filter(image, size=(3, 3, 1)).astype(np.uint8)
    if filter_name == "Gaussian Filter":
        blurred = gaussian_filter(
            image.astype(np.float32),
            sigma=(0.8, 0.8, 0),
        )
        return np.clip(np.rint(blurred), 0, 255).astype(np.uint8)
    raise ValueError("Filter must be None, Median Filter or Gaussian Filter.")
