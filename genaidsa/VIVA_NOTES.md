# Viva Notes: Image Denoising & Compression Lab

This is a study aid for the code in this project. Explain the ideas in your own words during the presentation.

## Short introduction

“My project is an Image Denoising and Compression Lab. It lets a user upload an image, add a chosen amount of Gaussian noise, optionally apply a denoising filter, and explore how latent precision affects reconstruction. The Streamlit app calls the Python processing pipeline directly. The pipeline uses the pretrained `stabilityai/sd-vae-ft-mse` model through Diffusers `AutoencoderKL`. The application calculates image-quality metrics and lets the user download the reconstruction. The compression value estimates latent bit precision; it is not a compressed image file.”

## Project workflow

```text
Upload an image in Streamlit
        |
The app calls backend.image_pipeline.process_image(...)
        |
Validate image, convert to RGB, and resize a VAE working copy
        |
Add Gaussian noise and apply the selected optional filter
        |
Pad and normalize the working image
        |
Run the real VAE encoder to create latent values
        |
Quantize and dequantize latent values at 4, 8 or 16 bits
        |
Run the real VAE decoder to reconstruct the image
        |
Crop padding, resize to uploaded dimensions, calculate metrics
        |
Show the original/noisy/reconstructed images and download
```

Noise and the selected filter are applied to the resized working image before encoding. The original full-resolution image is kept for comparison. The VAE input is padded to dimensions divisible by eight; padding is removed after decoding.

## Important files

| File | Purpose |
| --- | --- |
| `streamlit/streamlit_app.py` | Streamlit interface; loads the cached VAE and calls the pipeline directly. |
| `backend/image_pipeline.py` | Coordinates preprocessing, noise/filtering, VAE encoding/decoding, quantization and metrics. |
| `backend/model_loader.py` | Loads `stabilityai/sd-vae-ft-mse`, selects CUDA or CPU, and caches the model. |
| `backend/preprocessing.py` | Validates supported image formats and size, converts to RGB, resizes and pads. |
| `backend/filters.py` | Adds Gaussian noise or applies no filter, a median filter or a Gaussian filter. |
| `backend/quantization.py` | Quantizes latent values and restores floating-point values for the decoder. |
| `backend/metrics.py` | Calculates MSE, PSNR, SSIM, signed improvement and the working-image latent ratio. |
| `tests/test_image_pipeline.py` | Tests preprocessing, noise, filters and quantization without loading model weights. |
| `tests/test_metrics.py` | Tests metric edge cases, signed improvement and the ratio formula. |
| `tests/test_vae_integration.py` | Four tests that use the real pretrained VAE encoder and decoder. |
| `requirements.txt` | Dependencies for local use and Streamlit Community Cloud. |
| `README.md` | Setup, run, test, model workflow and deployment instructions. |
| `DEPLOYMENT.md` | GitHub and Community Cloud publishing steps. |

## Important concepts

### VAE, encoder, decoder and latents

A Variational Autoencoder learns a numeric representation of an input and can create an image from that representation. The encoder maps pixels into latent values. The decoder maps latent values back into pixels. Latents are model values, not an ordinary image file.

This project uses Diffusers `AutoencoderKL` with the actual pretrained `stabilityai/sd-vae-ft-mse` weights. It uses the encoder posterior mode for deterministic encoding. Input pixels are normalized to the model's expected -1 to 1 range. The VAE scaling factor is applied before quantization and undone before decoding.

### Quantization and estimated compression

Quantization rounds values to a smaller set of possible levels. Four-, eight- and sixteen-bit settings have different precision. Dequantization converts these values back to floating-point values for the decoder. Lower precision can lose more information, so four-bit reconstruction is not guaranteed to match sixteen-bit quality.

The **estimated working-image latent ratio** compares the resized working image's RGB bit count (24 bits per pixel) with the actual latent element count multiplied by the selected bit depth. It does not count preprocessing downscaling. It is only a bit-count estimate: the project does not entropy-code the latents or write a compact compressed-image file.

### Noise and filters

Gaussian noise adds random pixel changes drawn from a bell-shaped distribution. The slider controls the standard deviation in pixel-value units. A median filter replaces a pixel with the neighborhood median and can remove isolated specks. A Gaussian filter averages nearby pixels with weights that decrease with distance; it smooths noise but can blur details. Neither filter guarantees a lower reconstruction error.

### Image-quality metrics

- **MSE (Mean Squared Error):** average squared pixel difference. Zero means matching pixels; lower is closer.
- **PSNR (Peak Signal-to-Noise Ratio):** pixel error expressed in decibels. Larger usually means a closer match. Identical images display infinity.
- **SSIM (Structural Similarity Index):** compares local brightness, contrast and structure. Values closer to 1 usually indicate greater similarity.
- **Denoising improvement:** noisy-image MSE minus reconstructed-image MSE. Positive means error decreased; negative means it increased. A negative value is a valid result, not an improvement.

Metrics compare matching RGB images at the uploaded image dimensions and use measured image pixels.

### Why reconstruction can be worse

The VAE is a general pretrained autoencoder, not a model trained specifically to remove noise. Noise, filtering, resizing, quantization and decoding can change or remove information. The reconstruction can have greater error than the noisy image, and the app reports the actual result.

### Streamlit and the processing pipeline

Streamlit handles the controls and presentation. When the user presses **Process Image**, the app passes the uploaded bytes and settings to `backend.image_pipeline.process_image(...)`, along with the cached VAE and device. The pipeline validates and processes the image and returns the real images and metrics. Both parts run in one Python app process; no HTTP backend is used.

## Likely viva questions

1. **What problem does the project explore?** How added noise, optional filters and latent precision affect VAE reconstruction.
2. **What does VAE stand for?** Variational Autoencoder.
3. **What does the encoder do?** Maps normalized image pixels into a latent representation.
4. **What does the decoder do?** Maps latent values back into image pixels.
5. **Which pretrained model is used?** `stabilityai/sd-vae-ft-mse`, loaded with Diffusers `AutoencoderKL`.
6. **Why use the posterior mode?** It makes the latent selection deterministic for the same input.
7. **What does quantization do?** Reduces the number of distinct values used to represent each latent value.
8. **What does dequantization do?** Converts quantized values back to decoder-compatible floating-point values.
9. **What does a negative denoising improvement mean?** Reconstructed-image MSE is greater than noisy-image MSE, so error increased.
10. **Is the compression ratio a final file-size reduction?** No. It estimates RGB bits versus quantized latent bits; there is no entropy coder or compressed image file.
11. **Why can reconstruction be worse than the noisy input?** A general autoencoder plus resizing, filtering and quantization can change information; denoising is not guaranteed.
12. **How does Streamlit run the image pipeline?** It passes the uploaded bytes, settings and cached VAE to the Python function directly.
13. **How do the integration tests differ from unit tests?** Unit tests cover small functions without model weights; integration tests run the real VAE encoder and decoder.
14. **What device does inference use?** CUDA when PyTorch reports it is available; otherwise CPU.
15. **What happens on the first model run?** Diffusers downloads the model if it is not cached; later runs reuse it in memory.

## 60–90 second demonstration script

“This is my Image Denoising and Compression Lab. It uses Streamlit for controls and results. I will upload an image and choose a noise level, filter and latent bit depth. When I press Process Image, Streamlit calls the Python pipeline directly. The pipeline makes a resized working copy, adds Gaussian noise, applies the selected filter, and runs the image through the pretrained VAE encoder. The latent values are quantized and dequantized before the real VAE decoder reconstructs the image. Here I can compare the original, noisy and reconstructed results. MSE, PSNR and SSIM are calculated from the actual images. The signed denoising value can be positive or negative, so the reconstruction is not assumed to improve every input. The ratio estimates the resized working image and latent values; it is not a compressed-file size. I can download the reconstructed PNG here.”
