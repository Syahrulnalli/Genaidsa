"""Expose the image-processing pipeline through a local HTTP endpoint."""
import base64
import binascii
import json
import logging

import azure.functions as func

try:  # Functions host imports function_app as a top-level module.
    from .image_pipeline import process_image
    from .model_loader import load_vae
except ImportError:  # pragma: no cover - used by the Functions Core Tools host
    from image_pipeline import process_image
    from model_loader import load_vae

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


def _validate_request(body: object) -> tuple[bytes, float, str, int]:
    """Check request fields and return decoded image bytes and settings."""
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")

    encoded_image = body.get("image_base64")
    if not isinstance(encoded_image, str) or not encoded_image:
        raise ValueError("image_base64 is required and must contain a Base64-encoded image.")

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid Base64.") from exc

    maximum_image_bytes = 20 * 1024 * 1024
    if len(image_bytes) > maximum_image_bytes:
        raise ValueError("Image exceeds the 20 MB limit.")

    noise_level = body.get("noise_level")
    if (
        isinstance(noise_level, bool)
        or not isinstance(noise_level, (int, float))
        or not 0 <= noise_level <= 50
    ):
        raise ValueError("noise_level must be a number from 0 to 50.")

    filter_name = body.get("filter")
    allowed_filters = {"None", "Median Filter", "Gaussian Filter"}
    if filter_name not in allowed_filters:
        raise ValueError("filter must be None, Median Filter or Gaussian Filter.")

    bit_depth = body.get("bit_depth")
    if isinstance(bit_depth, bool) or bit_depth not in (4, 8, 16):
        raise ValueError("bit_depth must be 4, 8 or 16.")

    return image_bytes, float(noise_level), filter_name, bit_depth


@app.route(route="process-image", methods=["POST"])
def process_image_http(req: func.HttpRequest) -> func.HttpResponse:
    """Validate one request, run inference and return JSON results."""
    try:
        image_bytes, noise_level, filter_name, bit_depth = _validate_request(req.get_json())
        model, device = load_vae()
        result = process_image(image_bytes, noise_level, filter_name, bit_depth, model, device)
        response_body = json.dumps(result, allow_nan=False)
        return func.HttpResponse(
            response_body,
            mimetype="application/json",
            status_code=200,
        )
    except ValueError as exc:
        return func.HttpResponse(json.dumps({"error": str(exc)}), mimetype="application/json", status_code=400)
    except Exception:
        logging.exception("Image processing request failed")
        message = (
            "The VAE model could not be loaded or image inference failed. "
            "Check internet, model cache and available memory, then retry."
        )
        return func.HttpResponse(
            json.dumps({"error": message}),
            mimetype="application/json",
            status_code=500,
        )
