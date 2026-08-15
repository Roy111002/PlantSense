import io
import unittest

try:
    from PIL import Image
    import main
    from plantsense_ai import PlantClassifier

    API_DEPS_AVAILABLE = True
except ImportError:
    API_DEPS_AVAILABLE = False


@unittest.skipUnless(
    API_DEPS_AVAILABLE,
    "Flask and Pillow are runtime dependencies",
)
class ApiTests(unittest.TestCase):
    def setUp(self):
        self.original_classifier = main.classifier
        main.classifier = PlantClassifier(
            model_dir="definitely-not-a-model-directory",
        )
        self.client = main.app.test_client()
        buffer = io.BytesIO()
        Image.new(
            "RGB",
            (320, 240),
            color=(70, 150, 70),
        ).save(buffer, format="JPEG")
        self.jpeg = buffer.getvalue()

    def tearDown(self):
        main.classifier = self.original_classifier

    def request_data(self):
        return {
            "temperature": "26.5",
            "humidity": "61.2",
            "soil_moisture": "48.0",
            "light_lux": "7200",
            "pump_state": "OFF",
            "grow_light_state": "OFF",
            "image": (
                io.BytesIO(self.jpeg),
                "plant.jpg",
                "image/jpeg",
            ),
        }

    def test_health_reports_analysis_mode(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["analysis_mode"],
            "local_plant_health_and_sensor_rules",
        )

    def test_analyze_falls_back_to_sensor_rules_without_model(self):
        response = self.client.post(
            "/analyze",
            data=self.request_data(),
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result["vision"]["status"], "unavailable")
        self.assertEqual(result["disease"], "unknown")

    def test_invalid_image_is_rejected(self):
        data = self.request_data()
        data["image"] = (
            io.BytesIO(b"not-a-jpeg"),
            "plant.jpg",
            "image/jpeg",
        )

        response = self.client.post("/analyze", data=data)

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
