"""Streamlit page for uploading images and viewing backend results."""
import base64
import os

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://localhost:7071/api/process-image"


def _get_backend_url() -> str:
    """Read the backend URL from environment or Streamlit Cloud secrets."""
    environment_url = os.getenv("BACKEND_URL")
    if environment_url:
        return environment_url

    try:
        secret_url = st.secrets.get("BACKEND_URL")
    except FileNotFoundError:
        # Local development does not need a secrets.toml file.
        secret_url = None

    return secret_url or DEFAULT_BACKEND_URL


def _show_comparison(result: dict) -> None:
    """Display the original, noisy and reconstructed images side by side."""
    st.subheader("Image comparison")
    image_columns = st.columns(3)
    image_fields = (
        ("original_png", "Original image"),
        ("noisy_png", "Noisy image"),
        ("reconstructed_png", "Reconstructed image"),
    )

    for column, (field_name, label) in zip(image_columns, image_fields):
        with column:
            st.markdown(f"**{label}**")
            image_bytes = base64.b64decode(result[field_name])
            st.image(image_bytes, width="stretch", alt=label)

    reconstructed_bytes = base64.b64decode(result["reconstructed_png"])
    st.download_button(
        "Download reconstructed image",
        reconstructed_bytes,
        "reconstructed.png",
        "image/png",
        width="stretch",
    )


def _show_metrics(metrics: dict) -> None:
    """Display calculated quality values and explain their direction."""
    st.subheader("Quality metrics")
    st.caption(
        "MSE measures average squared pixel error (lower is closer). "
        "PSNR expresses pixel error in decibels (higher is closer). "
        "SSIM compares image structure (closer to 1 is more similar)."
    )

    first_row = st.columns(3)
    first_row[0].metric(
        "Original vs noisy · MSE",
        f"{metrics['original_vs_noisy_mse']:.3f}",
    )
    first_row[1].metric(
        "Original vs reconstructed · MSE",
        f"{metrics['original_vs_reconstructed_mse']:.3f}",
    )
    psnr = metrics["psnr"]
    psnr_text = "∞ (identical)" if psnr is None else f"{psnr:.2f} dB"
    first_row[2].metric("PSNR", psnr_text)

    second_row = st.columns(3)
    second_row[0].metric("SSIM", f"{metrics['ssim']:.4f}")
    second_row[1].metric(
        "Estimated working-image latent ratio",
        f"{metrics['estimated_working_image_latent_ratio']:.2f}:1",
    )

    improvement = metrics["denoising_improvement"]
    if improvement > 0:
        improvement_status = "error decreased"
    elif improvement < 0:
        improvement_status = "error increased"
    else:
        improvement_status = "no change"

    second_row[2].metric(
        "Denoising improvement",
        f"{improvement:+.3f} MSE",
        improvement_status,
    )
    st.caption(
        "Improvement = noisy-image MSE − reconstructed-image MSE. "
        "A positive value means error decreased; a negative value means "
        "reconstruction error increased. The ratio compares resized VAE "
        "working-image RGB bits with quantized latent bits. It is not a "
        "packaged compressed file size."
    )


def _show_processing_details(metadata: dict) -> None:
    """Show the image sizes, latent shape, settings and inference device."""
    uploaded_size = metadata["input_dimensions"]
    working_size = metadata["working_image_dimensions"]
    st.subheader("Processing details")
    st.write(
        f"**Uploaded image:** {uploaded_size['width']} × {uploaded_size['height']} px  ·  "
        f"**VAE working image:** {working_size['width']} × {working_size['height']} px  ·  "
        f"**Latent tensor:** {metadata['latent_dimensions']}  ·  "
        f"**Device:** {metadata['device']}"
    )
    st.write(
        f"**Noise:** {metadata['noise_level']}  ·  "
        f"**Filter:** {metadata['filter']}  ·  "
        f"**Quantization:** {metadata['bit_depth']}-bit"
    )
    st.info(
        "The pretrained VAE is a general image autoencoder, not a dedicated "
        "denoiser. Results depend on the image and settings."
    )


def _show_results(result: dict) -> None:
    """Render all image outputs, metrics and processing details."""
    st.divider()
    _show_comparison(result)
    _show_metrics(result["metrics"])
    _show_processing_details(result["metadata"])


st.set_page_config(
    page_title="Image Denoising & Compression Lab",
    page_icon=":material/image:",
    layout="wide",
)
st.title("Image Denoising & Compression Lab")
st.write(
    "Add noise, explore denoising filters, and study latent quantization "
    "with a pretrained VAE."
)
st.caption(
    "Inference runs through the local Azure Functions backend. "
    "The first model load may download weights."
)

backend_url = _get_backend_url()
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg", "webp"],
)

noise_column, filter_column, quantization_column = st.columns(3)
with noise_column:
    noise_level = st.slider("Gaussian noise level (standard deviation)", 0, 50, 20)
with filter_column:
    filter_name = st.selectbox(
        "Denoising filter",
        ["None", "Median Filter", "Gaussian Filter"],
    )
with quantization_column:
    bit_depth = st.selectbox(
        "Latent quantization",
        [4, 8, 16],
        index=1,
        format_func=lambda value: f"{value}-bit",
    )

if uploaded_file is not None:
    st.subheader("Original image")
    st.image(
        uploaded_file.getvalue(),
        caption=f"{uploaded_file.name} · {uploaded_file.size:,} bytes",
        width="stretch",
        alt="The uploaded source image",
    )

if st.button("Process Image", type="primary", disabled=uploaded_file is None):
    request_body = {
        "image_base64": base64.b64encode(uploaded_file.getvalue()).decode("ascii"),
        "filename": uploaded_file.name,
        "noise_level": noise_level,
        "filter": filter_name,
        "bit_depth": bit_depth,
    }

    try:
        with st.spinner("Calling the local backend and running VAE inference…"):
            response = requests.post(
                backend_url,
                json=request_body,
                timeout=(10, 900),
            )

        if not response.ok:
            try:
                error_message = response.json().get("error", response.text)
            except ValueError:
                error_message = response.text
            st.error(f"Backend returned HTTP {response.status_code}: {error_message}")
        else:
            st.session_state["result"] = response.json()
    except requests.RequestException as exc:
        if backend_url == DEFAULT_BACKEND_URL:
            connection_message = (
                "Could not reach the local image-processing backend. "
                "Start Azure Functions with `func start` and try again."
            )
        else:
            connection_message = (
                f"Could not reach the configured backend at {backend_url}. "
                "Check the BACKEND_URL setting and confirm that the backend is online."
            )
        st.error(f"{connection_message} Details: {exc}")

result = st.session_state.get("result")
if result:
    _show_results(result)
