"""PlantSense HTTP service using local, zero-cost ONNX inference."""

from __future__ import annotations

import math
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from plantsense_ai import (
    InvalidImageError,
    ModelUnavailableError,
    PlantClassifier,
    build_plant_assessment,
)


load_dotenv()

app = Flask(__name__)

# ESP32-CAM JPEGs are normally much smaller than this. The limit prevents a
# malformed request from consuming all of the server's memory.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MAX_IMAGE_BYTES = 4 * 1024 * 1024
classifier = PlantClassifier.from_environment(
    Path(__file__).resolve().parent / "models" / "current"
)


class RequestValidationError(ValueError):
    """Raised when the ESP32 sends an invalid multipart request."""


def analyze_plant(
    image_bytes,
    temperature,
    humidity,
    soil_moisture,
    light_lux,
    pump_state,
    grow_light_state,
):
    """Classify locally, then combine the result with sensor evidence."""
    vision = None
    model_error = None

    try:
        vision = classifier.classify(image_bytes)
    except ModelUnavailableError as error:
        # Sensor assessment remains useful while a model is being trained or
        # if the cached public model is temporarily unavailable.
        model_error = str(error)

    return build_plant_assessment(
        vision=vision,
        temperature=temperature,
        humidity=humidity,
        soil_moisture=soil_moisture,
        light_lux=light_lux,
        pump_state=pump_state,
        grow_light_state=grow_light_state,
        model_error=model_error,
    )


def parse_number(name, minimum=None, maximum=None):
    raw_value = request.form.get(name)

    if raw_value is None:
        raise RequestValidationError(
            f'Missing form field: "{name}".'
        )

    try:
        value = float(raw_value)
    except ValueError as error:
        raise RequestValidationError(
            f'Form field "{name}" must be a number.'
        ) from error

    if not math.isfinite(value):
        raise RequestValidationError(
            f'Form field "{name}" must be finite.'
        )

    if minimum is not None and value < minimum:
        raise RequestValidationError(
            f'Form field "{name}" must be at least {minimum}.'
        )

    if maximum is not None and value > maximum:
        raise RequestValidationError(
            f'Form field "{name}" must not exceed {maximum}.'
        )

    return value


def parse_actuator_state(name):
    value = request.form.get(name)

    if value is None:
        raise RequestValidationError(
            f'Missing form field: "{name}".'
        )

    value = value.strip().upper()

    if value not in {"ON", "OFF"}:
        raise RequestValidationError(
            f'Form field "{name}" must be ON or OFF.'
        )

    return value


@app.get("/health")
def health():
    model_status = classifier.describe()

    return jsonify({
        "service": "PlantSense AI",
        "status": (
            "ok" if model_status["configured"] else "degraded"
        ),
        "analysis_mode": "local_plant_health_and_sensor_rules",
        "classifier": model_status,
    })


@app.post("/analyze")
def analyze_endpoint():
    try:
        image = request.files.get("image")

        if image is None:
            raise RequestValidationError(
                'Missing multipart image field: "image".'
            )

        if image.mimetype not in {"image/jpeg", "image/jpg"}:
            raise RequestValidationError(
                "The image must use the JPEG content type."
            )

        image_bytes = image.read(MAX_IMAGE_BYTES + 1)

        if not image_bytes:
            raise RequestValidationError(
                "The uploaded image is empty."
            )

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise RequestValidationError(
                "The uploaded image exceeds the 4 MiB limit."
            )

        result = analyze_plant(
            image_bytes=image_bytes,
            temperature=parse_number(
                "temperature",
                -50.0,
                100.0,
            ),
            humidity=parse_number("humidity", 0.0, 100.0),
            soil_moisture=parse_number(
                "soil_moisture",
                0.0,
                100.0,
            ),
            light_lux=parse_number(
                "light_lux",
                0.0,
                200000.0,
            ),
            pump_state=parse_actuator_state("pump_state"),
            grow_light_state=parse_actuator_state(
                "grow_light_state"
            ),
        )

        return jsonify(result)

    except (RequestValidationError, InvalidImageError) as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Plant analysis failed")

        return jsonify({
            "error": "Plant analysis service failed."
        }), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({
        "error": "The request exceeds the 5 MiB limit."
    }), 413


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))

    app.run(
        host=host,
        port=port,
        debug=False,
    )
