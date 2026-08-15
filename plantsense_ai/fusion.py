"""Conservative, deterministic fusion of vision and sensor evidence."""

from __future__ import annotations

from typing import Any


def assess_range(
    value: float,
    minimum: float,
    maximum: float,
) -> str:
    if value < minimum:
        return "low"
    if value > maximum:
        return "high"
    return "normal"


def assess_sensors(
    temperature: float,
    humidity: float,
    soil_moisture: float,
    light_lux: float,
) -> dict[str, str]:
    return {
        "temperature": assess_range(temperature, 15.0, 32.0),
        "humidity": assess_range(humidity, 40.0, 85.0),
        "soil_moisture": assess_range(
            soil_moisture,
            30.0,
            75.0,
        ),
        "light": "low" if light_lux < 2000.0 else "normal",
    }


def build_plant_assessment(
    vision: dict[str, Any] | None,
    temperature: float,
    humidity: float,
    soil_moisture: float,
    light_lux: float,
    pump_state: str,
    grow_light_state: str,
    model_error: str | None = None,
) -> dict[str, Any]:
    sensors = assess_sensors(
        temperature,
        humidity,
        soil_moisture,
        light_lux,
    )
    abnormal_sensors = [
        name
        for name, status in sensors.items()
        if status != "normal"
    ]

    if len(abnormal_sensors) >= 3:
        stress = "high"
    elif len(abnormal_sensors) == 2:
        stress = "medium"
    elif abnormal_sensors:
        stress = "low"
    else:
        stress = "none"

    condition = "uncertain"
    disease = "unknown"
    confidence = 0.0
    visible_symptoms: list[str] = []
    possible_causes = sensor_causes(sensors)
    recommendations = sensor_recommendations(sensors)

    if vision is None:
        recommendations.insert(
            0,
            "The local vision model is unavailable; do not make a "
            "disease-specific treatment decision.",
        )
    elif vision.get("status") == "rejected":
        recommendations.insert(
            0,
            "Retake a close, well-lit image with a leaf occupying most "
            "of the frame.",
        )
    elif vision.get("status") == "uncertain":
        confidence = float(vision.get("confidence", 0.0))
        recommendations.insert(
            0,
            "The image prediction is uncertain; retake the image and "
            "inspect the plant manually.",
        )
    elif vision.get("status") == "accepted":
        confidence = float(vision.get("confidence", 0.0))

        if vision.get("healthy"):
            disease = "none"
            condition = (
                "possible_stress"
                if abnormal_sensors
                else "healthy"
            )
        elif (
            vision.get("task") == "binary_health"
            or vision.get("classification") == "abnormal"
            and str(vision.get("disease", "unknown")) == "unknown"
        ):
            condition = "possible_stress"
            possible_causes.insert(
                0,
                "The image contains a visual pattern classified as abnormal",
            )
            recommendations.insert(
                0,
                "Inspect the leaves, stems, soil, and pests manually; "
                "the generic screen cannot identify the cause or disease.",
            )
        else:
            disease = str(vision.get("disease", "unknown"))
            condition = "possible_disease"
            possible_causes.insert(
                0,
                f"Visual pattern associated with {disease}",
            )
            recommendations.insert(
                0,
                f"Inspect for signs consistent with {disease} and "
                "confirm before applying treatment.",
            )

    if condition == "uncertain" and abnormal_sensors:
        condition = "possible_stress"

    if not recommendations:
        recommendations.append(
            "Continue routine monitoring and capture another image "
            "if symptoms appear."
        )

    return {
        "plant_condition": condition,
        "disease": disease,
        "stress": stress,
        "visible_symptoms": visible_symptoms,
        "possible_causes": possible_causes,
        "confidence": round(confidence, 6),
        "sensor_assessment": sensors,
        "recommendation": " ".join(recommendations),
        "urgent_action": False,
        "vision": vision or {
            "status": "unavailable",
            "error": model_error,
        },
        "actuators": {
            "pump": pump_state,
            "grow_light": grow_light_state,
        },
        "analysis_mode": "local_plant_health_and_sensor_rules",
    }


def sensor_causes(sensors: dict[str, str]) -> list[str]:
    causes = []

    for sensor, status in sensors.items():
        if status != "normal":
            causes.append(
                f"{sensor.replace('_', ' ')} is {status}"
            )

    return causes


def sensor_recommendations(sensors: dict[str, str]) -> list[str]:
    recommendations = []

    if sensors["soil_moisture"] == "low":
        recommendations.append(
            "Check the watering system and soil-moisture calibration."
        )
    elif sensors["soil_moisture"] == "high":
        recommendations.append(
            "Pause watering and check drainage."
        )

    if sensors["light"] == "low":
        recommendations.append(
            "Check available light while preserving a daily dark period."
        )

    if sensors["temperature"] != "normal":
        recommendations.append(
            "Move the plant toward its preferred temperature range."
        )

    if sensors["humidity"] != "normal":
        recommendations.append(
            "Check ventilation and ambient humidity."
        )

    return recommendations
