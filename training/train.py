"""Train MobileNetV3 from a local dataset cache and export ONNX."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
import io
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "training" / "dataset_manifest.json"
DEFAULT_LABELS = ROOT / "training" / "plantvillage_labels.json"
DEFAULT_OUTPUT = ROOT / "models" / "current"
BINARY_LABELS = ["healthy", "abnormal"]
TRAINING_CACHE = ROOT / "data" / "training-cache"

# Keep all reproducible download caches inside the ignored project cache.
# Callers can still override either location explicitly.
os.environ.setdefault(
    "HF_HOME",
    str(TRAINING_CACHE / "huggingface"),
)
os.environ.setdefault(
    "TORCH_HOME",
    str(ROOT / ".cache" / "torch"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a zero-cost PlantSense image classifier from a "
            "revision-pinned dataset with a disposable local cache."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--task",
        choices=("binary-health", "multiclass-disease"),
        default="binary-health",
        help=(
            "Train the generic healthy/abnormal screen by default. "
            "Use multiclass-disease only for PlantVillage disease names."
        ),
    )
    parser.add_argument(
        "--crops",
        nargs="*",
        default=[],
        help=(
            "Optional crop names such as Tomato Potato. By default all "
            "PlantVillage classes are used."
        ),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--max-training-minutes",
        type=float,
        default=30.0,
        help=(
            "Abort at the next batch if total preparation/training time "
            "exceeds this bound. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--keep-dataset-cache",
        action="store_true",
        help="Keep data/training-cache after the run for repeated experiments.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Also keep the PyTorch state dict for later training work.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fine-tune-backbone",
        action="store_true",
        help=(
            "Train all MobileNet weights. The default trains only the "
            "classification head and is more practical on a CPU."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def require_training_dependencies():
    try:
        import numpy as np
        import torch
        from datasets import load_dataset
        from torchvision import transforms
        from torchvision.models import (
            MobileNet_V3_Small_Weights,
            mobilenet_v3_small,
        )
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install them with: "
            "python -m pip install -r requirements-training.txt"
        ) from error

    return {
        "np": np,
        "torch": torch,
        "load_dataset": load_dataset,
        "transforms": transforms,
        "weights": MobileNet_V3_Small_Weights,
        "model_factory": mobilenet_v3_small,
    }


def normalize_crop(value: str) -> str:
    return "".join(
        character
        for character in value.casefold()
        if character.isalnum()
    )


def select_labels(all_labels: list[str], crops: list[str]):
    if not crops:
        return all_labels

    requested = {normalize_crop(crop) for crop in crops}
    selected = [
        label
        for label in all_labels
        if normalize_crop(label.split("___", 1)[0]) in requested
    ]

    if not selected:
        available = sorted({
            label.split("___", 1)[0]
            for label in all_labels
        })
        raise SystemExit(
            "No labels matched --crops. Available crops: "
            + ", ".join(available)
        )

    return selected


def target_labels_for_task(
    task: str,
    selected_source_labels: list[str],
) -> list[str]:
    """Return stable output labels for the selected training task."""
    if task == "binary-health":
        return BINARY_LABELS.copy()
    if task == "multiclass-disease":
        return selected_source_labels
    raise ValueError(f"Unsupported task: {task}")


def target_label_for_source(source_label: str, task: str) -> str:
    """Collapse any PlantVillage condition into a binary health label."""
    if task == "binary-health":
        condition = source_label.partition("___")[2]
        return (
            "healthy"
            if condition.casefold() == "healthy"
            else "abnormal"
        )
    if task == "multiclass-disease":
        return source_label
    raise ValueError(f"Unsupported task: {task}")


def decode_image(image_value):
    """Decode both Hugging Face Image objects and raw Parquet structs."""
    from PIL import Image

    if hasattr(image_value, "convert"):
        return image_value.convert("RGB")
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    if isinstance(image_value, dict) and image_value.get("path"):
        return Image.open(image_value["path"]).convert("RGB")
    raise RuntimeError(
        "The dataset image has an unsupported representation."
    )


class TransformDataset:
    """Map-style adapter so DataLoader workers can decode in parallel."""

    def __init__(
        self,
        source,
        transform,
        source_labels,
        target_label_to_index,
        task,
        label_field,
    ):
        self.source = source
        self.transform = transform
        self.source_labels = source_labels
        self.target_label_to_index = target_label_to_index
        self.task = task
        self.label_field = label_field

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        example = self.source[index]
        raw_label = example[self.label_field]
        source_label = (
            self.source_labels[raw_label]
            if isinstance(raw_label, int)
            else str(raw_label)
        )
        target_label = target_label_for_source(
            source_label,
            self.task,
        )
        return (
            self.transform(decode_image(example["image"])),
            self.target_label_to_index[target_label],
        )


def select_balanced_subset(
    source,
    maximum: int | None,
    label_field: str,
    source_labels: list[str],
    task: str,
    target_label_to_index: dict[str, int],
):
    """Select a deterministic, class-balanced debug subset without images."""
    if maximum is None or maximum >= len(source):
        return source

    if task != "binary-health":
        return source.select(range(maximum))

    per_class = max(1, maximum // len(target_label_to_index))
    counts = {
        index: 0
        for index in target_label_to_index.values()
    }
    indices = []

    for index, raw_label in enumerate(source[label_field]):
        source_label = (
            source_labels[raw_label]
            if isinstance(raw_label, int)
            else str(raw_label)
        )
        target_label = target_label_for_source(source_label, task)
        target_index = target_label_to_index[target_label]
        if counts[target_index] >= per_class:
            continue
        counts[target_index] += 1
        indices.append(index)
        if all(count >= per_class for count in counts.values()):
            break

    if not indices:
        raise RuntimeError("No samples matched the requested subset.")
    return source.select(indices)


def choose_device(torch, requested: str):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def train_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    torch,
    freeze_backbone=False,
    scaler=None,
    use_amp=False,
    deadline=None,
    log_every=100,
    epoch_number=1,
):
    model.train()
    if freeze_backbone:
        # Frozen batch-normalization statistics must not drift while only the
        # new classification head is being trained.
        model.features.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (images, labels) in enumerate(loader, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                "Training exceeded --max-training-minutes; aborting "
                "before another batch."
            )

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = loss_function(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(1) == labels).sum().item())
        total += batch_size

        if log_every and batch_index % log_every == 0:
            print(
                f"Epoch {epoch_number}: batch {batch_index}/"
                f"{len(loader)} samples={total}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError("The training dataset produced no samples.")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "samples": total,
    }


def extract_embeddings(
    model,
    loader,
    device,
    torch,
    use_amp,
    deadline,
    log_every,
    stage,
):
    """Run the frozen backbone once so head-only epochs are inexpensive."""
    model.eval()
    embedding_batches = []
    label_batches = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Training exceeded --max-training-minutes during "
                    f"{stage} feature extraction."
                )
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                embeddings = model.features(images)
                embeddings = model.avgpool(embeddings)
                embeddings = torch.flatten(embeddings, 1)
                embeddings = model.classifier[:-1](embeddings)
            embedding_batches.append(embeddings.float().cpu())
            label_batches.append(labels.clone())

            if log_every and batch_index % log_every == 0:
                print(
                    f"{stage}: batch {batch_index}/{len(loader)}",
                    flush=True,
                )

    return (
        torch.cat(embedding_batches),
        torch.cat(label_batches),
    )


def evaluate(
    model,
    loader,
    device,
    class_labels,
    torch,
    use_amp=False,
):
    class_count = len(class_labels)
    model.eval()
    confusion = [
        [0 for _ in range(class_count)]
        for _ in range(class_count)
    ]

    with torch.no_grad():
        for images, batch_labels in loader:
            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                logits = model(
                    images.to(device, non_blocking=True)
                )
            predictions = logits.argmax(1).cpu()

            for truth, prediction in zip(batch_labels, predictions):
                confusion[int(truth)][int(prediction)] += 1

    total = sum(sum(row) for row in confusion)

    if total == 0:
        raise RuntimeError("The test dataset produced no samples.")

    correct = sum(
        confusion[index][index]
        for index in range(class_count)
    )
    f1_scores = []
    per_class = {}

    for index in range(class_count):
        true_positive = confusion[index][index]
        false_positive = sum(
            confusion[row][index]
            for row in range(class_count)
            if row != index
        )
        false_negative = sum(confusion[index]) - true_positive
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_scores.append(f1)
        per_class[class_labels[index]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[index]),
        }

    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "samples": total,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def balanced_cross_entropy(logits, labels, torch):
    """Give each class present in a batch equal influence on the loss."""
    sample_losses = torch.nn.functional.cross_entropy(
        logits,
        labels,
        reduction="none",
    )
    class_losses = [
        sample_losses[labels == class_index].mean()
        for class_index in torch.unique(labels)
    ]
    return torch.stack(class_losses).mean()


def export_artifacts(
    model,
    labels,
    output_dir,
    manifest,
    args,
    metrics,
    torch,
    weights,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    onnx_path = output_dir / "model.onnx"
    if args.save_checkpoint:
        torch.save(
            model.state_dict(),
            output_dir / "training_checkpoint.pt",
        )
    dummy_input = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )

    model_hash = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    model_version = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    metadata = {
        "model_version": model_version,
        "architecture": "mobilenet_v3_small",
        "pretrained_weights": str(weights),
        "format": "ONNX",
        "sha256": model_hash,
        "image_size": 224,
        "resize_shorter": 256,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "class_count": len(labels),
        "task": args.task.replace("-", "_"),
        "labels_file": "labels.json",
        "dataset_id": manifest["dataset_id"],
        "dataset_configuration": (
            manifest.get("configuration") or "default"
        ),
        "dataset_revision": manifest["revision"],
        "dataset_license": manifest["license"],
        "selected_crops": args.crops or "all",
        "fine_tuned_backbone": args.fine_tune_backbone,
        "training_strategy": (
            "full_backbone_fine_tune"
            if args.fine_tune_backbone
            else "single_pass_frozen_backbone_embeddings"
        ),
        "data_loader_workers": args.num_workers,
        "max_training_minutes": args.max_training_minutes,
        "epochs": args.epochs,
        "metrics": metrics,
        "limitations": [
            "PlantVillage uses mostly controlled backgrounds.",
            "Generic abnormal output does not identify a cause or disease.",
            "Plants unlike the training crops may be outside the model domain.",
            "The model requires validation on real ESP32-CAM images.",
            "Output is screening evidence, not a definitive diagnosis."
        ]
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        model_card(metadata),
        encoding="utf-8",
    )

    return metadata


def model_card(metadata):
    metrics = metadata["metrics"]

    return f"""---
license: {metadata['dataset_license'].lower()}
pipeline_tag: image-classification
---

# PlantSense plant-health classifier

MobileNetV3-Small classifier trained for the PlantSense educational project.
Its default task is a generic visual screen: `healthy` versus `abnormal`.

## Evaluation

- Test accuracy: {metrics['accuracy']:.4f}
- Test macro F1: {metrics['macro_f1']:.4f}
- Test samples: {metrics['samples']}
- Dataset revision: `{metadata['dataset_revision']}`
- Model SHA-256: `{metadata['sha256']}`

## Intended use

The model provides conservative screening evidence for PlantSense. An
`abnormal` result does not identify a disease or cause. It must not trigger
pesticide or disease treatment without expert confirmation.

## Limitations

- PlantVillage images largely use controlled backgrounds.
- Plants unlike the PlantVillage training crops may be outside its domain.
- `abnormal` combines many diseases and is not a diagnosis.
- Real ESP32-CAM validation and target-domain fine-tuning are still required.
"""


def main():
    args = parse_args()
    dependencies = require_training_dependencies()
    np = dependencies["np"]
    torch = dependencies["torch"]
    load_dataset = dependencies["load_dataset"]
    transforms = dependencies["transforms"]

    if args.num_workers < 0:
        raise SystemExit("--num-workers must be zero or greater.")
    if args.max_training_minutes < 0:
        raise SystemExit("--max-training-minutes cannot be negative.")

    if TRAINING_CACHE.is_dir():
        shutil.rmtree(TRAINING_CACHE)
    TRAINING_CACHE.mkdir(parents=True, exist_ok=True)

    def cleanup_training_cache():
        if not args.keep_dataset_cache:
            shutil.rmtree(TRAINING_CACHE, ignore_errors=True)

    atexit.register(cleanup_training_cache)
    started_at = time.monotonic()
    deadline = (
        started_at + args.max_training_minutes * 60
        if args.max_training_minutes
        else None
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.hub.set_dir(str(ROOT / ".cache" / "torch"))

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    all_labels = json.loads(args.labels.read_text(encoding="utf-8"))
    selected_source_labels = select_labels(all_labels, args.crops)
    labels = target_labels_for_task(
        args.task,
        selected_source_labels,
    )
    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    dataset_args = [manifest["dataset_id"]]
    if manifest.get("configuration"):
        dataset_args.append(manifest["configuration"])
    dataset = load_dataset(
        *dataset_args,
        revision=manifest["revision"],
        streaming=False,
        cache_dir=TRAINING_CACHE / "datasets",
    )
    label_field = manifest.get("label_field", "label")

    if manifest.get("source_split"):
        all_source = dataset[manifest["source_split"]]
        split_field = manifest["split_field"]
        train_value = manifest["train_split_value"]
        test_value = manifest["test_split_value"]
        train_source = all_source.filter(
            lambda example: example[split_field] == train_value
        )
        test_source = all_source.filter(
            lambda example: example[split_field] == test_value
        )
    else:
        train_source = dataset[manifest["train_split"]]
        test_source = dataset[manifest["test_split"]]

    source_feature = train_source.features.get(label_field)
    source_labels = (
        list(source_feature.names)
        if hasattr(source_feature, "names")
        else all_labels
    )
    selected_source_set = set(selected_source_labels)

    def source_label(example):
        raw_label = example[label_field]
        return (
            source_labels[raw_label]
            if isinstance(raw_label, int)
            else str(raw_label)
        )

    if len(selected_source_set) != len(all_labels):
        train_source = train_source.filter(
            lambda example: source_label(example)
            in selected_source_set,
            desc="Filtering training crops",
        )
        test_source = test_source.filter(
            lambda example: source_label(example)
            in selected_source_set,
            desc="Filtering test crops",
        )

    train_source = train_source.shuffle(seed=args.seed)
    test_source = test_source.shuffle(seed=args.seed)
    train_source = select_balanced_subset(
        train_source,
        args.max_train_samples,
        label_field,
        source_labels,
        args.task,
        label_to_index,
    )
    test_source = select_balanced_subset(
        test_source,
        args.max_test_samples,
        label_field,
        source_labels,
        args.task,
        label_to_index,
    )

    print(
        f"Prepared {len(train_source)} training and "
        f"{len(test_source)} test images in "
        f"{time.monotonic() - started_at:.1f}s.",
        flush=True,
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])
    train_dataset = TransformDataset(
        train_source,
        train_transform,
        source_labels,
        label_to_index,
        args.task,
        label_field,
    )
    test_dataset = TransformDataset(
        test_source,
        test_transform,
        source_labels,
        label_to_index,
        args.task,
        label_field,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    weights = dependencies["weights"].DEFAULT
    model = dependencies["model_factory"](weights=weights)

    if not args.fine_tune_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(
        input_features,
        len(labels),
    )
    device = choose_device(torch, args.device)
    model = model.to(device)
    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )
    loss_function = (
        lambda logits, targets: balanced_cross_entropy(
            logits,
            targets,
            torch,
        )
        if args.task == "binary-health"
        else torch.nn.functional.cross_entropy(logits, targets)
    )

    print(
        f"Training task={args.task} with {len(labels)} classes "
        f"on {device}; workers={args.num_workers}; amp={use_amp}; "
        f"fine_tune_backbone={args.fine_tune_backbone}.",
        flush=True,
    )

    model_for_training = model
    training_use_amp = use_amp

    if not args.fine_tune_backbone:
        train_embeddings, train_targets = extract_embeddings(
            model,
            train_loader,
            device,
            torch,
            use_amp,
            deadline,
            args.log_every,
            "Training features",
        )
        test_embeddings, test_targets = extract_embeddings(
            model,
            test_loader,
            device,
            torch,
            use_amp,
            deadline,
            args.log_every,
            "Test features",
        )
        del train_loader, test_loader
        gc.collect()

        feature_batch_size = min(
            4096,
            max(256, args.batch_size * 16),
        )
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_embeddings,
                train_targets,
            ),
            batch_size=feature_batch_size,
            shuffle=True,
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                test_embeddings,
                test_targets,
            ),
            batch_size=feature_batch_size,
            pin_memory=True,
        )
        model_for_training = model.classifier[-1]
        training_use_amp = False
        print(
            "Frozen backbone features cached in memory; head epochs no "
            "longer decode or reload images.",
            flush=True,
        )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=training_use_amp,
    )

    for epoch in range(args.epochs):
        result = train_epoch(
            model_for_training,
            train_loader,
            optimizer,
            loss_function,
            device,
            torch,
            freeze_backbone=False,
            scaler=scaler,
            use_amp=training_use_amp,
            deadline=deadline,
            log_every=args.log_every,
            epoch_number=epoch + 1,
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"loss={result['loss']:.4f} "
            f"accuracy={result['accuracy']:.4f} "
            f"samples={result['samples']}",
            flush=True,
        )

    metrics = evaluate(
        model_for_training,
        test_loader,
        device,
        labels,
        torch,
        use_amp=training_use_amp,
    )
    metadata = export_artifacts(
        model,
        labels,
        args.output_dir,
        manifest,
        args,
        metrics,
        torch,
        weights,
    )

    print(
        f"Test accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}",
        flush=True,
    )
    print(
        f"Artifacts written to {args.output_dir.resolve()}",
        flush=True,
    )
    print(f"Model SHA-256: {metadata['sha256']}", flush=True)

    del train_loader, test_loader, train_dataset, test_dataset
    del train_source, test_source, dataset
    gc.collect()
    cleanup_training_cache()


if __name__ == "__main__":
    main()
