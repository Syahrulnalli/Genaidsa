"""Image validation, aspect-preserving resize and VAE-compatible padding."""
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SIDE = 512
MAX_PIXELS = 25_000_000


def decode_image(raw: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Validate encoded image bytes and return RGB pixels and (width, height)."""
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or larger than the 20 MB limit.")
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ValueError("Use a PNG, JPG/JPEG or WebP image.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("Image dimensions exceed the 25-megapixel limit.")
            image.load()
            rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid PNG, JPEG or WebP image.") from exc
    original_width, original_height = rgb.size
    return np.asarray(rgb, dtype=np.uint8), (original_width, original_height)


def resize_for_model(image: np.ndarray) -> np.ndarray:
    """Downscale only the VAE working copy, keeping the full reference image."""
    pil = Image.fromarray(image)
    pil.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def prepare_for_vae(image: np.ndarray, factor: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad at bottom/right; return padded RGB image and unpadded (height,width)."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image.")
    h, w = image.shape[:2]
    padded_height = max(factor, ((h + factor - 1) // factor) * factor)
    padded_width = max(factor, ((w + factor - 1) // factor) * factor)
    padded = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    padded[:h, :w] = image
    return padded, (h, w)
