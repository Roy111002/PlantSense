"""ONNX plant-health classification with versioned model resolution."""

from __future__ import annotations

import io
import json
import os
import re
import threading
from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """Raised when no usable local classifier artifact can be resolved."""


class InvalidImageError(ValueError):
    """Raised when uploaded bytes are not a usable plant image."""


class PlantClassifier:
    """Lazily load and run a compact image classifier with ONNX Runtime."""

    REQUIRED_FILES = (
        "model.onnx",
        "labels.json",
        "model_metadata.json",
    )

    def __init__(
        self,
        model_dir: str | Path = "models/current",
        repo_id: str = "",
        revision: str = "",
        minimum_confidence: float = 0.65,
        minimum_margin: float = 0.10,
        plant_crop: str = "",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.repo_id = repo_id.strip()
        self.revision = revision.strip()
        self.minimum_confidence = minimum_confidence
        self.minimum_margin = minimum_margin
        self.plant_crop = plant_crop.strip()

        self._session: Any | None = None
        self._labels: list[str] = []
        self._metadata: dict[str, Any] = {}
        self._input_name = ""
        self._load_error = ""
        self._lock = threading.Lock()

    @classmethod
    def from_environment(
        cls,
        default_model_dir: str | Path = "models/current",
    ) -> "PlantClassifier":
        return cls(
            model_dir=os.getenv(
                "PLANTSENSE_MODEL_DIR",
                str(default_model_dir),
            ),
            repo_id=os.getenv("PLANTSENSE_MODEL_REPO", ""),
            revision=os.getenv("PLANTSENSE_MODEL_REVISION", ""),
            minimum_confidence=float(
                os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.65")
            ),
            minimum_margin=float(
                os.getenv("CLASSIFIER_MIN_MARGIN", "0.10")
            ),
            plant_crop=os.getenv("PLANT_CROP", ""),
        )

    def describe(self) -> dict[str, Any]:
        local_available = all(
            (self.model_dir / name).is_file()
            for name in self.REQUIRED_FILES
        )
        remote_configured = bool(
            self.repo_id
            and re.fullmatch(
                r"[0-9a-fA-F]{40}",
                self.revision,
            )
        )

        return {
            "configured": bool(
                local_available or remote_configured
            ),
            "loaded": self._session is not None,
            "source": (
                "local"
                if local_available
                else "hugging_face"
                if remote_configured
                else "unconfigured"
            ),
            "repo_id": self.repo_id or None,
            "revision": self.revision or None,
            "crop_filter": self.plant_crop or None,
            "task": self._task if self._metadata else None,
            "model_version": self._metadata.get("model_version"),
            "last_error": self._load_error or None,
        }

    def classify(self, image_bytes: bytes) -> dict[str, Any]:
        image, quality = self._decode_and_check_image(image_bytes)

        if quality["rejected"]:
            return {
                "status": "rejected",
                "reason": quality["reason"],
                "quality": quality,
                "top_predictions": [],
                "confidence": 0.0,
                "margin": 0.0,
            }

        self._ensure_loaded()

        try:
            import numpy as np
        except ImportError as error:
            raise ModelUnavailableError(
                "NumPy is not installed. Install requirements.txt."
            ) from error

        model_input = self._preprocess(image, np)
        model_output = self._session.run(
            None,
            {self._input_name: model_input},
        )[0]
        logits = np.asarray(model_output).reshape(-1)

        if len(logits) != len(self._labels):
            raise ModelUnavailableError(
                "Model output size does not match labels.json."
            )

        eligible_indices = self._eligible_label_indices()
        eligible_logits = logits[eligible_indices]
        eligible_logits = eligible_logits - np.max(eligible_logits)
        probabilities = np.exp(eligible_logits)
        probabilities = probabilities / np.sum(probabilities)
        ranking = np.argsort(probabilities)[::-1]

        predictions = []

        for rank_index in ranking[:3]:
            label_index = eligible_indices[int(rank_index)]
            label = self._labels[label_index]
            predictions.append(self._format_prediction(
                label,
                round(float(probabilities[int(rank_index)]), 6),
            ))

        confidence = predictions[0]["score"]
        second_score = (
            predictions[1]["score"]
            if len(predictions) > 1
            else 0.0
        )
        margin = confidence - second_score
        accepted = (
            confidence >= self.minimum_confidence
            and margin >= self.minimum_margin
        )

        return {
            "status": "accepted" if accepted else "uncertain",
            "reason": None if accepted else "low_model_confidence",
            "label": predictions[0]["label"],
            "classification": predictions[0]["classification"],
            "crop": predictions[0]["crop"],
            "disease": predictions[0]["disease"],
            "healthy": predictions[0]["healthy"],
            "confidence": confidence,
            "margin": round(margin, 6),
            "quality": quality,
            "top_predictions": predictions,
            "task": self._task,
            "model_version": self._metadata.get("model_version"),
        }

    @property
    def _task(self) -> str:
        configured = str(self._metadata.get("task", ""))
        if configured:
            return configured
        if {label.casefold() for label in self._labels} == {
            "healthy",
            "abnormal",
        }:
            return "binary_health"
        return "multiclass_disease"

    def _format_prediction(
        self,
        label: str,
        score: float,
    ) -> dict[str, Any]:
        if self._task == "binary_health":
            healthy = label.casefold() == "healthy"
            return {
                "label": label,
                "classification": "healthy" if healthy else "abnormal",
                "crop": None,
                "disease": "none" if healthy else "unknown",
                "healthy": healthy,
                "score": score,
            }

        crop, disease = split_label(label)
        healthy = disease.casefold() == "healthy"
        return {
            "label": label,
            "classification": "healthy" if healthy else "abnormal",
            "crop": humanize_label(crop),
            "disease": humanize_label(disease),
            "healthy": healthy,
            "score": score,
        }

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return

        with self._lock:
            if self._session is not None:
                return

            try:
                artifacts = self._resolve_artifacts()
                self._load_artifacts(artifacts)
                self._load_error = ""
            except Exception as error:
                self._load_error = str(error)

                if isinstance(error, ModelUnavailableError):
                    raise

                raise ModelUnavailableError(str(error)) from error

    def _resolve_artifacts(self) -> dict[str, Path]:
        local_artifacts = {
            name: self.model_dir / name
            for name in self.REQUIRED_FILES
        }

        if all(path.is_file() for path in local_artifacts.values()):
            return local_artifacts

        if not self.repo_id:
            raise ModelUnavailableError(
                "No trained model is available. Train into models/current "
                "or configure PLANTSENSE_MODEL_REPO and a pinned revision."
            )

        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ModelUnavailableError(
                "PLANTSENSE_MODEL_REVISION must be a full 40-character "
                "commit hash so every machine loads the same model."
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise ModelUnavailableError(
                "huggingface-hub is not installed. Install requirements.txt."
            ) from error

        return {
            name: Path(hf_hub_download(
                repo_id=self.repo_id,
                filename=name,
                revision=self.revision,
            ))
            for name in self.REQUIRED_FILES
        }

    def _load_artifacts(self, artifacts: dict[str, Path]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ModelUnavailableError(
                "onnxruntime is not installed. Install requirements.txt."
            ) from error

        labels = json.loads(
            artifacts["labels.json"].read_text(encoding="utf-8")
        )
        metadata = json.loads(
            artifacts["model_metadata.json"].read_text(
                encoding="utf-8"
            )
        )

        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) for label in labels)
        ):
            raise ModelUnavailableError(
                "labels.json must contain a non-empty string array."
            )

        session = ort.InferenceSession(
            str(artifacts["model.onnx"]),
            providers=["CPUExecutionProvider"],
        )

        if len(session.get_inputs()) != 1:
            raise ModelUnavailableError(
                "PlantSense expects an ONNX model with one image input."
            )

        self._labels = labels
        self._metadata = metadata
        self._input_name = session.get_inputs()[0].name
        self._session = session

        # Validate the crop filter while loading rather than during a request.
        self._eligible_label_indices()

    def _eligible_label_indices(self) -> list[int]:
        # A generic binary model has no crop-specific output labels.
        if self._task == "binary_health" or not self.plant_crop:
            return list(range(len(self._labels)))

        requested_crop = normalize_crop(self.plant_crop)
        indices = [
            index
            for index, label in enumerate(self._labels)
            if normalize_crop(split_label(label)[0]) == requested_crop
        ]

        if not indices:
            raise ModelUnavailableError(
                f'No model labels match PLANT_CROP="{self.plant_crop}".'
            )

        return indices

    def _decode_and_check_image(self, image_bytes: bytes):
        try:
            from PIL import Image, ImageStat
        except ImportError as error:
            raise ModelUnavailableError(
                "Pillow is not installed. Install requirements.txt."
            ) from error

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
            image = image.convert("RGB")
        except Exception as error:
            raise InvalidImageError(
                "The uploaded bytes are not a readable image."
            ) from error

        width, height = image.size
        sample = image.copy()
        sample.thumbnail((64, 64))
        brightness = float(
            ImageStat.Stat(sample.convert("L")).mean[0]
        )

        reason = None

        if min(width, height) < 96:
            reason = "image_resolution_too_low"
        elif brightness < 8:
            reason = "image_too_dark"
        elif brightness > 247:
            reason = "image_overexposed"

        quality = {
            "rejected": reason is not None,
            "reason": reason,
            "width": width,
            "height": height,
            "brightness": round(brightness, 2),
        }

        return image, quality

    def _preprocess(self, image, np):
        image_size = int(self._metadata.get("image_size", 224))
        resize_shorter = int(
            self._metadata.get("resize_shorter", 256)
        )
        mean = np.asarray(
            self._metadata.get(
                "normalization_mean",
                [0.485, 0.456, 0.406],
            ),
            dtype=np.float32,
        )
        std = np.asarray(
            self._metadata.get(
                "normalization_std",
                [0.229, 0.224, 0.225],
            ),
            dtype=np.float32,
        )

        width, height = image.size
        scale = resize_shorter / min(width, height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        image = image.resize(
            (resized_width, resized_height),
            resample=2,
        )
        left = (resized_width - image_size) // 2
        top = (resized_height - image_size) // 2
        image = image.crop(
            (left, top, left + image_size, top + image_size)
        )

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - mean) / std
        array = np.transpose(array, (2, 0, 1))

        return np.expand_dims(array, axis=0).astype(np.float32)


def split_label(label: str) -> tuple[str, str]:
    crop, separator, disease = label.partition("___")

    if not separator:
        return "unknown", label

    return crop, disease


def humanize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()


def normalize_crop(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())
