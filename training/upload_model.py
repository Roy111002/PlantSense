"""Publish only trained artifacts to a public Hugging Face model repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_FILES = (
    "model.onnx",
    "labels.json",
    "model_metadata.json",
    "README.md",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("models/current"),
    )
    args = parser.parse_args()

    missing = [
        name
        for name in ALLOWED_FILES
        if not (args.artifact_dir / name).is_file()
    ]

    if missing:
        raise SystemExit(
            "Missing model artifacts: " + ", ".join(missing)
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise SystemExit(
            "Install requirements-training.txt before uploading."
        ) from error

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=args.artifact_dir,
        allow_patterns=[*ALLOWED_FILES],
        commit_message="Publish PlantSense ONNX model",
    )

    print(json.dumps({
        "repo_id": args.repo_id,
        "commit_url": commit.commit_url,
        "revision": commit.oid,
    }, indent=2))


if __name__ == "__main__":
    main()
