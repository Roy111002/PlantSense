import json
import unittest
from pathlib import Path

from training.train import (
    model_card,
    select_labels,
    target_label_for_source,
    target_labels_for_task,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingManifestTests(unittest.TestCase):
    def test_dataset_revision_is_immutable_commit(self):
        for filename in (
            "dataset_manifest.json",
            "dataset_manifest_smoke.json",
        ):
            manifest = json.loads(
                (ROOT / "training" / filename).read_text(
                    encoding="utf-8"
                )
            )

            self.assertRegex(manifest["revision"], r"^[0-9a-f]{40}$")
            self.assertFalse(manifest["streaming"])
            self.assertEqual(
                manifest["cache_policy"],
                "disposable_project_cache",
            )

    def test_labels_are_unique(self):
        labels = json.loads(
            (ROOT / "training" / "plantvillage_labels.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(len(labels), 38)
        self.assertEqual(len(labels), len(set(labels)))

    def test_crop_selection(self):
        labels = [
            "Tomato___healthy",
            "Tomato___Early_blight",
            "Potato___healthy",
        ]

        self.assertEqual(
            select_labels(labels, ["Tomato"]),
            ["Tomato___healthy", "Tomato___Early_blight"],
        )

    def test_binary_task_has_stable_generic_labels(self):
        source_labels = [
            "Tomato___healthy",
            "Potato___Early_blight",
        ]

        self.assertEqual(
            target_labels_for_task("binary-health", source_labels),
            ["healthy", "abnormal"],
        )
        self.assertEqual(
            target_label_for_source(
                "Tomato___healthy",
                "binary-health",
            ),
            "healthy",
        )
        self.assertEqual(
            target_label_for_source(
                "Potato___Early_blight",
                "binary-health",
            ),
            "abnormal",
        )

    def test_model_card_uses_manifest_license(self):
        card = model_card({
            "dataset_license": "CC0-1.0",
            "dataset_revision": "0" * 40,
            "sha256": "0" * 64,
            "metrics": {
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "samples": 1,
            },
        })

        self.assertIn("license: cc0-1.0", card)


if __name__ == "__main__":
    unittest.main()
