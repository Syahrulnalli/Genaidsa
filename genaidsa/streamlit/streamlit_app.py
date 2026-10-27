"""Streamlit interface for the image denoising and compression lab."""
import base64
import logging
import sys
from pathlib import Path

import streamlit as st

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in APP_PATH.parents
        if (parent / "backend" / "__init__.py").is_file()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError(
        f"Could not find the backend package above {APP_PATH}. "
        "Deploy the complete project, including the backend/ and streamlit/ folders."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.image_pipeline import process_image
from backend.model_loader import load_vae


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
    "Image processing runs directly in this Streamlit app. "
    "The first model load may download pretrained weights."
)

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
    try:
        with st.spinner("Loading the pretrained VAE and processing the image…"):
            model, device = load_vae()
            result = process_image(
                uploaded_file.getvalue(),
                noise_level,
                filter_name,
                bit_depth,
                model,
                device,
            )
        st.session_state["result"] = result
    except ValueError as exc:
        st.error(str(exc))
    except Exception:
        logging.exception("Image processing failed")
        st.error(
            "Image processing failed. Check your internet connection for the "
            "first model download, available memory, and the uploaded image, "
            "then try again."
        )

result = st.session_state.get("result")
if result:
    _show_results(result)
