import unittest

from plantsense_ai.classifier import (
    PlantClassifier,
    humanize_label,
    normalize_crop,
    split_label,
)


class ClassifierHelperTests(unittest.TestCase):
    def test_split_plantvillage_label(self):
        self.assertEqual(
            split_label("Tomato___Early_blight"),
            ("Tomato", "Early_blight"),
        )

    def test_humanize_label(self):
        self.assertEqual(
            humanize_label("Tomato_Yellow_Leaf_Curl_Virus"),
            "Tomato Yellow Leaf Curl Virus",
        )

    def test_crop_normalization(self):
        self.assertEqual(
            normalize_crop("Pepper, bell"),
            "pepperbell",
        )

    def test_unconfigured_model_status(self):
        classifier = PlantClassifier(
            model_dir="definitely-not-a-model-directory",
        )
        status = classifier.describe()
        self.assertFalse(status["configured"])
        self.assertEqual(status["source"], "unconfigured")

    def test_remote_model_requires_pinned_revision(self):
        classifier = PlantClassifier(
            model_dir="definitely-not-a-model-directory",
            repo_id="example/plantsense",
            revision="main",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "40-character",
        ):
            classifier._resolve_artifacts()


if __name__ == "__main__":
    unittest.main()
