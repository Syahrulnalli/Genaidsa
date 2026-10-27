"""Load the pretrained VAE once and keep it available for later requests."""
from functools import lru_cache

import torch
from diffusers import AutoencoderKL

MODEL_ID = "stabilityai/sd-vae-ft-mse"

@lru_cache(maxsize=1)
def load_vae():
    """Return the cached VAE and the best available inference device.

    The first call downloads the model if it is not already cached by Hugging Face.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoencoderKL.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, device
