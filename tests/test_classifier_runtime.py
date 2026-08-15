import io
import unittest

try:
    import numpy as np
    from PIL import Image

    RUNTIME_DEPS_AVAILABLE = True
except ImportError:
    RUNTIME_DEPS_AVAILABLE = False

from plantsense_ai.classifier import PlantClassifier


class FakeSession:
    def __init__(self, logits):
        self.logits = logits

    def run(self, _outputs, inputs):
        self.last_input = inputs["image"]
        return [self.logits]


@unittest.skipUnless(
    RUNTIME_DEPS_AVAILABLE,
    "NumPy and Pillow are runtime dependencies",
)
class ClassifierRuntimeTests(unittest.TestCase):
    def make_jpeg(self):
        buffer = io.BytesIO()
        Image.new(
            "RGB",
            (320, 240),
            color=(70, 150, 70),
        ).save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_accepted_local_prediction(self):
        classifier = PlantClassifier(
            minimum_confidence=0.65,
            minimum_margin=0.10,
            plant_crop="Tomato",
        )
        classifier._labels = [
            "Tomato___healthy",
            "Tomato___Early_blight",
            "Potato___healthy",
        ]
        classifier._metadata = {
            "model_version": "test",
            "image_size": 224,
            "resize_shorter": 256,
        }
        classifier._input_name = "image"
        classifier._session = FakeSession(
            np.asarray([[5.0, 1.0, 20.0]], dtype=np.float32)
        )

        result = classifier.classify(self.make_jpeg())

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["label"], "Tomato___healthy")
        self.assertTrue(result["healthy"])
        self.assertEqual(
            classifier._session.last_input.shape,
            (1, 3, 224, 224),
        )

    def test_close_predictions_are_uncertain(self):
        classifier = PlantClassifier(
            minimum_confidence=0.50,
            minimum_margin=0.20,
        )
        classifier._labels = [
            "Tomato___healthy",
            "Tomato___Early_blight",
        ]
        classifier._metadata = {}
        classifier._input_name = "image"
        classifier._session = FakeSession(
            np.asarray([[1.0, 0.9]], dtype=np.float32)
        )

        result = classifier.classify(self.make_jpeg())

        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(result["reason"], "low_model_confidence")

    def test_generic_binary_abnormal_prediction_names_no_disease(self):
        classifier = PlantClassifier(
            minimum_confidence=0.60,
            minimum_margin=0.10,
            plant_crop="Tomato",
        )
        classifier._labels = ["healthy", "abnormal"]
        classifier._metadata = {
            "task": "binary_health",
            "model_version": "binary-test",
        }
        classifier._input_name = "image"
        classifier._session = FakeSession(
            np.asarray([[0.1, 3.0]], dtype=np.float32)
        )

        result = classifier.classify(self.make_jpeg())

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["classification"], "abnormal")
        self.assertEqual(result["disease"], "unknown")
        self.assertIsNone(result["crop"])
        self.assertFalse(result["healthy"])


if __name__ == "__main__":
    unittest.main()
