# Image Denoising & Compression Lab

A local BCA AI/ML project for studying image noise, denoising filters, VAE latent representations and reduced-precision latent values. The Streamlit frontend sends image data to a local Azure Functions HTTP endpoint. The backend uses the real pretrained Diffusers model `stabilityai/sd-vae-ft-mse` to encode and reconstruct images.

## Features and technologies

- Upload PNG, JPG/JPEG or WebP images.
- Add adjustable Gaussian noise and optionally apply a median or Gaussian filter.
- Encode and decode with the pretrained Diffusers `AutoencoderKL` VAE.
- Quantize latent values to 4-bit, 8-bit or 16-bit precision, then dequantize them for the decoder.
- Compare original, noisy and reconstructed images; calculate MSE, PSNR and SSIM from those image pixels.
- Download the reconstructed PNG and inspect the actual image, latent and device dimensions.
- Python, PyTorch, Pillow, NumPy, SciPy, scikit-image, Streamlit, Azure Functions Core Tools and pytest.

## Project structure

```text
genaidsa/
├── frontend/
│   ├── app.py
│   └── requirements.txt
├── backend/
│   ├── function_app.py
│   ├── image_pipeline.py
│   ├── model_loader.py
│   ├── preprocessing.py
│   ├── filters.py
│   ├── quantization.py
│   ├── metrics.py
│   ├── host.json
│   ├── local.settings.json.example
│   ├── requirements.txt
│   └── README.md
├── tests/
│   ├── test_image_pipeline.py
│   ├── test_metrics.py
│   └── test_vae_integration.py
├── screenshots/
├── .streamlit/config.toml
├── .env.example
├── .gitignore
├── requirements.txt              # Streamlit Community Cloud dependencies
├── DEPLOYMENT.md
├── README.md
└── VIVA_NOTES.md
```

`backend/local.settings.json` is a local-only Functions configuration and is ignored by Git. Copy it from the example if it is missing. The `.venv` environment and Hugging Face model cache are also outside the submission files.

## How the VAE pipeline works

1. The backend checks the uploaded file, converts it to RGB and keeps the full-resolution copy for the final comparison.
2. It creates an aspect-preserving working copy no larger than 512 × 512, adds Gaussian noise and applies the selected filter.
3. The image is padded to dimensions divisible by eight. Pixels are converted from 0–255 to the VAE's expected range of -1 to 1.
4. The real VAE encoder produces a latent representation. The posterior mode is used for deterministic encoding. The VAE's `scaling_factor` is applied before quantization.
5. Latent values are uniformly quantized to the selected precision and dequantized. The scaling factor is reversed before passing values to the actual VAE decoder.
6. The decoded image is cropped to remove padding, resized to the uploaded image dimensions, and compared with the original.

The VAE weights are downloaded on the first run if they are not already in the Hugging Face cache. That run needs internet access. Later requests reuse the cached model in the backend process. CUDA is used when PyTorch detects it; otherwise inference runs on CPU.

## Metrics and compression estimate

- **MSE** is the mean squared difference between matching image pixels. Lower means the pixels are closer.
- **PSNR** expresses pixel error in decibels. Higher means closer. An identical image has infinite PSNR, represented as `null` in JSON and shown as ∞ in the app.
- **SSIM** compares local image structure. A value closer to 1 generally means more structural similarity.
- **Denoising improvement** is `noisy MSE − reconstructed MSE`. The app displays the signed value. Positive means the error decreased; negative means reconstruction error increased.
- **Estimated working-image latent ratio** is `(working width × working height × 24) ÷ (latent element count × bit depth)`. It compares the resized VAE working image with the quantized latent values. It does not count preprocessing downscaling as compression, and it is not an entropy-coded image-file size.

The estimate excludes quantizer ranges and file/container overhead. Latent quantization reduces precision; this project does not implement an entropy coder or write a compact compressed-image format.

## Install on Windows

Use 64-bit Python 3.11 and Azure Functions Core Tools v4. The project-local virtual environment is used directly, so PowerShell activation policy does not need to be changed.

Run from the project root in PowerShell:

```powershell
if (-not (Test-Path .venv\Scripts\python.exe)) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r frontend\requirements.txt
```

If `backend\local.settings.json` is missing, create it from the example:

```powershell
Copy-Item backend\local.settings.json.example backend\local.settings.json
```

The local example does not contain project secrets. To override the frontend's default backend URL, set this environment variable before starting Streamlit:

```powershell
$env:BACKEND_URL = 'http://localhost:7071/api/process-image'
```

## Run locally

Use two VS Code terminals, both opened at the project root.

**Terminal 1 — Azure Functions backend:** the `cmd.exe` wrapper activates the venv without changing PowerShell's execution policy.

```powershell
cmd.exe /k ".venv\Scripts\activate.bat && cd backend && func start"
```

The Functions endpoint is `http://localhost:7071/api/process-image`. Core Tools should list `process_image_http` as a POST route.

**Terminal 2 — Streamlit frontend:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Open [http://localhost:8501](http://localhost:8501), upload an image, select the noise/filter/quantization settings and click **Process Image**. The frontend calls the backend; it does not run model inference itself.

The JSON endpoint request and response example is documented in [backend/README.md](backend/README.md).

## Online deployment preparation

The Streamlit entry point, root dependency file, secret lookup and Azure Functions package exclusions are prepared for deployment. No public deployment or Azure backend URL exists yet. See [DEPLOYMENT.md](DEPLOYMENT.md) for the Community Cloud settings, backend publishing steps and the current account/tool blocker. The root `requirements.txt` is for the Streamlit frontend; Azure Functions uses `backend/requirements.txt`.

## Tests

Run the weight-free unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_image_pipeline.py tests\test_metrics.py
```

Run the four real-model integration tests (the first run may download the pretrained weights):

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_vae_integration.py
```

They exercise the real encoder, quantization and decoder to check: (1) the working-image latent ratio, (2) information loss at 4-bit versus 16-bit precision, (3) actual noisy/reconstructed error reporting without requiring improvement, and (4) quality metrics recalculated from the returned reconstruction.

Run every test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Demonstration and screenshots

For a short demo, show both local terminals, upload an image, explain the controls, process it, discuss the signed denoising metric and working-image ratio, then download the reconstruction. Capture the uploaded image and settings, the three-image comparison, the metrics and dimensions, and the download control in `screenshots/` or your submission folder.

## Limitations and troubleshooting

- The pretrained VAE is a general image autoencoder, not a model trained specifically to remove noise. Reconstruction quality varies and can be worse than the noisy image.
- Quantization demonstrates lower latent precision. It does not entropy-code the data or create a compressed file.
- The local endpoint is intended for local development; this project does not deploy to Azure.
- **`func` is not found:** install Azure Functions Core Tools v4, reopen VS Code and check `func --version`.
- **Backend connection error:** run Terminal 1 from the project root, confirm the endpoint is on port 7071, and check `BACKEND_URL`.
- **Port already in use:** stop the conflicting local process or select another Streamlit port and update the backend URL if you change the Functions port.
- **Model download or memory error:** keep internet available for the first model download, allow disk space for model weights, try a smaller input image and close other GPU workloads. CPU processing can take longer than GPU processing.
- **PowerShell blocks activation:** the documented backend command uses `activate.bat` through `cmd.exe`; the frontend command invokes the venv Python directly.
