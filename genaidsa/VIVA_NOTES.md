# Viva Notes — Image Denoising & Compression Lab

This guide describes the code that is in this project. It is written as a study aid; use your own words when presenting it.

## A. Short project introduction

“My project is an Image Denoising and Compression Lab. It lets a user upload an image, add a chosen amount of Gaussian noise, optionally apply a denoising filter, and explore how reducing the precision of a VAE's latent values affects reconstruction. The backend uses the pretrained `stabilityai/sd-vae-ft-mse` model from Hugging Face Diffusers. The Streamlit frontend sends the image and settings to a local Azure Functions API. The application returns a reconstructed image, calculates image-quality metrics, and lets the user download the result. The compression value is an estimate of latent bit precision, not a complete compressed image file.”

## B. Project workflow

```text
User uploads an image
        ↓
Frontend sends Base64 image and settings to local API
        ↓
Backend receives and validates the request
        ↓
Image is opened as RGB and a VAE working copy is resized
        ↓
Gaussian noise is added
        ↓
Optional median or Gaussian filter
        ↓
VAE encoder creates latent representation
        ↓
Latent values are quantized to 4, 8 or 16 bits
        ↓
Quantized values are dequantized for the model
        ↓
VAE decoder creates a reconstructed image
        ↓
Image is resized to the uploaded dimensions
        ↓
MSE, PSNR, SSIM and estimated ratio are calculated
        ↓
Frontend displays results and provides a PNG download
```

Noise and the selected filter are applied to the resized working image before it reaches the encoder. The original full-resolution image is kept for the comparison. The VAE input is padded to dimensions divisible by eight; that padding is removed after decoding.

## C. Important files

| File | Purpose |
| --- | --- |
| `frontend/app.py` | Shows the upload control and settings, calls the local API, displays returned images/metrics and offers the reconstructed PNG download. |
| `frontend/requirements.txt` | Lists the Streamlit and HTTP client packages used by the frontend. |
| `backend/function_app.py` | Defines the Azure Functions Python v2 POST route, validates request fields, loads the VAE and returns JSON or a clear error. |
| `backend/image_pipeline.py` | Coordinates image preprocessing, noise/filtering, real VAE encoding/decoding, quantization, metric calculation and response creation. |
| `backend/model_loader.py` | Loads `stabilityai/sd-vae-ft-mse`, selects CUDA when available or CPU otherwise, and caches the model in memory. |
| `backend/preprocessing.py` | Validates supported image formats and size, converts to RGB, resizes the VAE working copy and pads it for the model. |
| `backend/filters.py` | Adds Gaussian noise and applies no filter, a median filter or a Gaussian filter. |
| `backend/quantization.py` | Maps latent values to a limited set of values and restores them as floats the decoder can accept. |
| `backend/metrics.py` | Calculates MSE, PSNR, SSIM, signed denoising improvement and the working-image latent ratio. |
| `backend/host.json` | Azure Functions host configuration for local development. |
| `backend/local.settings.json.example` | Safe example settings for running Functions locally. The actual `local.settings.json` is local-only and ignored by Git. |
| `backend/requirements.txt` | Lists Azure Functions, PyTorch, Diffusers, image processing and test dependencies. |
| `backend/README.md` | Documents the HTTP request/response fields and gives a PowerShell API example. |
| `tests/test_image_pipeline.py` | Tests preprocessing, aspect ratio, noise, filters and latent quantization without loading model weights. |
| `tests/test_metrics.py` | Tests metric edge cases and the signed improvement and ratio formulas. |
| `tests/test_vae_integration.py` | Uses the real pretrained encoder and decoder to check compression estimate, information loss, noisy error reporting and reconstruction metrics. |
| `.streamlit/config.toml` | Sets the app's simple blue/light appearance using native Streamlit theme settings. |
| `.env.example` | Shows the optional backend URL setting; no key is needed. |
| `requirements.txt` | Lists frontend dependencies where Streamlit Community Cloud can find them at repository root. |
| `DEPLOYMENT.md` | Explains the prepared cloud settings and the account/resource steps still needed before a public deployment. |
| `.gitignore` | Keeps the virtual environment, local settings, secrets and generated Python/test files out of the submission. |
| `README.md` | Main setup, run, test, workflow, screenshot and troubleshooting instructions. |
| `screenshots/` | A place to save screenshots for the assignment. |

## D. Important concepts

### VAE, encoder, decoder and latent values

A Variational Autoencoder (VAE) is a neural network that learns to represent an input using a smaller set of values, then create an image from those values. Its **encoder** maps the image into a latent representation. Its **decoder** maps latent values back into image pixels. A latent representation is the model's internal numeric description of the image; it is not an ordinary image file.

This project uses Diffusers `AutoencoderKL` and the real pretrained `stabilityai/sd-vae-ft-mse` weights. It uses the encoder's posterior mode to make the latent selection deterministic. The image is normalized to the model's expected -1 to 1 range. The model's configured scaling factor is applied before quantization and undone before decoding.

### Quantization and estimated compression

Quantization rounds values to a smaller number of possible values. Four-bit, eight-bit and sixteen-bit settings provide different precision. Dequantization converts the chosen values back into floating-point numbers the VAE decoder accepts. Lower precision can lose more information; the four-bit reconstruction is not guaranteed to have the same quality as the sixteen-bit one.

The **estimated working-image latent ratio** compares the resized VAE working image's RGB bits (24 bits per pixel) with the number of latent elements multiplied by the chosen bit depth. It uses the working-image size, not the full uploaded-image size. It is a bit-count estimate only: this application does not entropy-code the latents or produce a compact compressed-image file.

### Noise and filters

Gaussian noise adds random pixel changes drawn from a Gaussian (bell-shaped) distribution. The slider controls its standard deviation in pixel-value units. It creates a repeatable experiment for comparing filters and reconstruction; it does not represent every real-world kind of image noise.

A **median filter** replaces a pixel with the median of its nearby values. It often removes isolated specks while keeping some edges. A **Gaussian filter** averages nearby values with weights that decrease with distance. It smooths noise but can blur edges and small details. Either filter can change the final error, and neither guarantees a better reconstruction.

### Image-quality metrics

- **MSE (Mean Squared Error):** average of the squared pixel differences. Zero means matching pixels; lower values are closer.
- **PSNR (Peak Signal-to-Noise Ratio):** expresses pixel error in decibels. Larger values generally indicate a closer match. Identical images have infinite PSNR, represented by `null` in JSON and ∞ in the app.
- **SSIM (Structural Similarity Index):** compares local brightness, contrast and structure. Values closer to 1 usually indicate more structural similarity.
- **Denoising improvement:** noisy-image MSE minus reconstructed-image MSE. Positive means the reconstruction error is lower; negative means it is higher. A negative value is a valid result and must not be presented as improvement.

The metrics compare matching RGB images at the uploaded image dimensions. They use actual image pixels, not hardcoded example values.

### Why the reconstruction can look worse

The VAE is a general pretrained autoencoder, not a dedicated denoising model. Noise, optional filtering, resizing and latent quantization can all remove or change information. The decoder may also introduce its own reconstruction differences. Therefore the reconstructed image can have higher error than the noisy image. The app reports the measured result instead of promising improvement.

### Frontend, backend and local Functions

Streamlit handles controls and presentation. Azure Functions handles the HTTP request and calls the Python processing pipeline. Keeping these roles separate makes it clear where the interface ends and image/model processing begins. Azure Functions Core Tools runs the Functions host locally, so the endpoint can be used on the same computer without deploying the project to Azure.

## E. Likely viva questions

1. **What problem does your project explore?** — It explores how added noise, optional filters and latent precision affect VAE image reconstruction.
2. **What does VAE stand for?** — Variational Autoencoder.
3. **What does the encoder do?** — It maps normalized image pixels into a latent representation.
4. **What does the decoder do?** — It maps latent values back into image pixels.
5. **What is a latent representation?** — It is the set of internal numeric values the VAE uses to describe an image.
6. **Which pretrained model is used?** — `stabilityai/sd-vae-ft-mse`, loaded with Diffusers `AutoencoderKL`.
7. **Why use the posterior mode?** — It makes the encoder's latent choice deterministic for the same input.
8. **Why normalize the input?** — The pretrained VAE expects pixel values in the range -1 to 1.
9. **What does quantization do?** — It reduces the number of distinct values used to represent each latent value.
10. **What does dequantization do?** — It restores quantized values to floating-point values that can be passed to the decoder.
11. **What is the difference between four-bit and sixteen-bit settings?** — Sixteen-bit has finer precision; four-bit has fewer possible latent values and can lose more information.
12. **Why add Gaussian noise?** — It gives the application a controlled noisy input so the effect of filtering and reconstruction can be studied.
13. **How do the two filters differ?** — Median filtering selects a neighborhood median; Gaussian filtering makes a weighted local average and may blur edges.
14. **What does MSE measure?** — Average squared pixel error; zero means identical pixel values.
15. **What does PSNR measure?** — Pixel error on a decibel scale; larger usually means closer images.
16. **What does SSIM measure?** — Similarity in local image structure, brightness and contrast.
17. **What does a negative denoising improvement mean?** — Reconstructed-image MSE is greater than noisy-image MSE, so the error increased.
18. **Is the compression ratio the final file-size reduction?** — No. It estimates RGB bit count versus quantized latent bit count; there is no entropy coder or compressed output file.
19. **Why can reconstruction be worse than the noisy input?** — The VAE is a general autoencoder, and resizing, filtering and quantization can lose information; it is not guaranteed to denoise.
20. **Why are the frontend and backend separate?** — The frontend collects settings and displays results; the backend validates requests and performs image/model work.
21. **Does the project require Azure deployment?** — No. Azure Functions Core Tools runs the HTTP function locally.
22. **How do the integration tests differ from unit tests?** — Unit tests check small functions without model weights; integration tests load the actual VAE and run its encoder and decoder.
23. **How does the project test information loss?** — It encodes an image with the real VAE, quantizes the same latent at 4 and 16 bits, compares latent errors, then decodes both.
24. **What device does inference use?** — CUDA when PyTorch reports it is available; otherwise CPU.
25. **What happens on the first model run?** — Diffusers downloads the pretrained model if it is not already cached; subsequent requests reuse the loaded model in memory.

## F. 60–90 second demonstration script

“This is my Image Denoising and Compression Lab. It uses Streamlit for the controls and a local Azure Functions endpoint for image processing. I will upload an image and choose a noise level, a filter and a latent bit depth. When I press Process Image, the backend makes a resized working copy, adds Gaussian noise, applies the selected filter, and sends the image through the pretrained VAE encoder. The latent values are quantized and dequantized before the real VAE decoder reconstructs the image. Here I can compare the original, noisy and reconstructed results. MSE, PSNR and SSIM are calculated from the actual images. The signed denoising value can be positive or negative, so the reconstruction is not assumed to improve every input. The ratio is an estimate for the resized working image and latent values; it is not a compressed-file size. I can download the reconstructed PNG here.”
