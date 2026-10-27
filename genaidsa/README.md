# Image Denoising & Compression Lab

A single Streamlit application for studying image noise, denoising filters, VAE reconstruction, and reduced-precision latent values. The app calls the existing Python image pipeline directly. There is no HTTP backend. The pipeline uses the pretrained `stabilityai/sd-vae-ft-mse` model through Diffusers `AutoencoderKL` and PyTorch.

## Features

- Upload PNG, JPG/JPEG, or WebP images.
- Add adjustable Gaussian noise and optionally apply median or Gaussian filtering.
- Process the image with the real pretrained VAE encoder and decoder.
- Quantize and dequantize VAE latent values at 4, 8, or 16 bits.
- Compare original, noisy, and reconstructed images.
- Calculate MSE, PSNR, SSIM, signed denoising improvement, and the estimated working-image latent ratio from the actual output.
- Download the reconstructed PNG and view image dimensions, latent dimensions, settings, and inference device.

## Project structure

```text
genaidsa/
├── frontend/
│   └── app.py                 # Streamlit UI and direct pipeline call
├── backend/
│   ├── image_pipeline.py      # End-to-end image processing and VAE calls
│   ├── model_loader.py        # Cached pretrained VAE and device selection
│   ├── preprocessing.py       # Validation, RGB conversion, resize and padding
│   ├── filters.py             # Gaussian noise and optional filters
│   ├── quantization.py        # Latent quantization/dequantization
│   └── metrics.py             # Image metrics and ratio formulas
├── tests/
│   ├── test_image_pipeline.py
│   ├── test_metrics.py
│   └── test_vae_integration.py # Four real-model integration tests
├── .streamlit/config.toml
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
└── VIVA_NOTES.md
```

## How the VAE pipeline works

1. The pipeline validates the upload, converts it to RGB, and keeps the uploaded-size image for comparison.
2. It makes an aspect-preserving VAE working copy no larger than 512 × 512, adds Gaussian noise, and applies the selected filter.
3. It pads the working image to dimensions divisible by eight and normalizes the RGB values to the VAE's expected range of -1 to 1.
4. The real VAE encoder produces latent values using the posterior mode. The configured scaling factor is applied before quantization.
5. Latents are quantized to the selected 4-, 8-, or 16-bit levels and dequantized to floating-point values.
6. The real VAE decoder reconstructs the image. Padding is cropped, then the image is resized to the uploaded dimensions for comparison.

The model is downloaded on the first run if it is not already in the Hugging Face cache. An internet connection and enough disk space are required for that download. The cached model is reused by the Python process. The app uses CUDA when PyTorch detects it; otherwise, it uses CPU.

## Metrics and compression estimate

- **MSE** measures average squared pixel difference. Lower values mean closer pixels.
- **PSNR** expresses pixel error in decibels. Higher values generally indicate a closer match. An exact match displays as infinity.
- **SSIM** compares local image structure; values closer to 1 indicate greater similarity.
- **Denoising improvement** is `noisy MSE - reconstructed MSE`. The signed value is shown. Positive means error decreased; negative means reconstruction error increased.
- **Estimated working-image latent ratio** is `(working width × working height × 24) / (quantized latent element count × bit depth)`. The working-image RGB dimensions come from the actual VAE input after resizing; the latent element count comes from the actual encoded tensor. The estimate does not count preprocessing downscaling, quantizer metadata, file/container overhead, or entropy coding. It is not the size of a compressed image file.

## Install and run locally (Windows PowerShell)

Use 64-bit Python 3.11. From the project root:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Open [http://localhost:8501](http://localhost:8501). Upload an image, choose noise, filter, and latent precision, then click **Process Image**. The first processing run may download the pretrained model.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository you control. Keep `.venv`, Hugging Face model weights, caches, `.env`, and Streamlit secrets out of Git; `.gitignore` excludes these files.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select your repository and branch. Set the app file path to `frontend/app.py`.
4. In **Advanced settings**, select Python 3.11. No secrets or backend URL settings are required.
5. Deploy, wait for the first install and model download, then open the assigned `*.streamlit.app` address and process a real image to verify it.

The Streamlit server runs the VAE inference itself. The first cloud startup may take longer because dependencies and model weights need to be installed/downloaded. See [DEPLOYMENT.md](DEPLOYMENT.md) for publishing steps and current local GitHub readiness.

## Tests

Run the unit tests without loading model weights:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_image_pipeline.py tests\test_metrics.py
```

Run the four real-VAE integration tests (the model must be cached or downloadable):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_vae_integration.py
```

Run the full suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The integration tests use the actual pretrained encoder, latent quantization/dequantization, and decoder. They do not substitute a mock VAE or assume that denoising always improves the error.

## Limitations and troubleshooting

- This pretrained VAE is a general-purpose autoencoder, not a model trained specifically to remove noise. Reconstruction quality depends on the image and settings and can be worse than the noisy image.
- The latent ratio estimates bit counts only. The project does not implement entropy coding or create a compressed image file.
- **Model download fails:** check internet access and available disk space for the Hugging Face cache.
- **Out of memory:** use a smaller input image, close other GPU workloads, or allow CPU inference to finish (which can be slower).
- **PowerShell cannot find Python:** install 64-bit Python 3.11 and reopen the terminal.
