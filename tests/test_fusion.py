import unittest

from plantsense_ai.fusion import (
    assess_sensors,
    build_plant_assessment,
)


class SensorAssessmentTests(unittest.TestCase):
    def test_normal_sensor_values(self):
        self.assertEqual(
            assess_sensors(25.0, 60.0, 50.0, 5000.0),
            {
                "temperature": "normal",
                "humidity": "normal",
                "soil_moisture": "normal",
                "light": "normal",
            },
        )

    def test_sensor_stress_without_model(self):
        result = build_plant_assessment(
            vision=None,
            temperature=38.0,
            humidity=30.0,
            soil_moisture=12.0,
            light_lux=5000.0,
            pump_state="ON",
            grow_light_state="OFF",
            model_error="model missing",
        )

        self.assertEqual(result["plant_condition"], "possible_stress")
        self.assertEqual(result["stress"], "high")
        self.assertEqual(result["disease"], "unknown")
        self.assertEqual(result["vision"]["status"], "unavailable")

    def test_healthy_image_with_normal_sensors(self):
        result = build_plant_assessment(
            vision={
                "status": "accepted",
                "healthy": True,
                "disease": "healthy",
                "confidence": 0.93,
            },
            temperature=25.0,
            humidity=60.0,
            soil_moisture=50.0,
            light_lux=5000.0,
            pump_state="OFF",
            grow_light_state="OFF",
        )

        self.assertEqual(result["plant_condition"], "healthy")
        self.assertEqual(result["disease"], "none")
        self.assertEqual(result["stress"], "none")

    def test_disease_prediction_is_worded_as_possible(self):
        result = build_plant_assessment(
            vision={
                "status": "accepted",
                "healthy": False,
                "disease": "Early blight",
                "confidence": 0.88,
            },
            temperature=25.0,
            humidity=60.0,
            soil_moisture=50.0,
            light_lux=5000.0,
            pump_state="OFF",
            grow_light_state="OFF",
        )

        self.assertEqual(
            result["plant_condition"],
            "possible_disease",
        )
        self.assertIn("confirm", result["recommendation"])
        self.assertFalse(result["urgent_action"])

    def test_generic_abnormal_prediction_does_not_name_disease(self):
        result = build_plant_assessment(
            vision={
                "status": "accepted",
                "task": "binary_health",
                "classification": "abnormal",
                "healthy": False,
                "disease": "unknown",
                "confidence": 0.86,
            },
            temperature=25.0,
            humidity=60.0,
            soil_moisture=50.0,
            light_lux=5000.0,
            pump_state="OFF",
            grow_light_state="OFF",
        )

        self.assertEqual(result["plant_condition"], "possible_stress")
        self.assertEqual(result["disease"], "unknown")
        self.assertIn(
            "cannot identify the cause or disease",
            result["recommendation"],
        )


if __name__ == "__main__":
    unittest.main()
