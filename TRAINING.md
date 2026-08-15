# PlantSense generic local vision model

The training job downloads revision-pinned Parquet shards into the disposable
`data/training-cache` directory once, then reuses them locally across epochs.
The cache is deleted automatically on normal completion or failure and is
cleared at the start of the next run after a forced stop. The dataset is never
part of the Flask server, firmware, Git repository, or deployed runtime.

## Environment

Create a separate training environment because PyTorch is not required by the
deployed server:

```powershell
python -m venv .venv-training
.\.venv-training\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-training.txt -c constraints-tested.txt
```

## Classification target

The default task combines all PlantVillage crops into two output classes:

- `healthy`: every source label ending in `___healthy`
- `abnormal`: every other source label

This makes the camera check generic: the plant species does not need to be
entered first. The output is only a visual anomaly screen. It cannot identify
the disease, nutrient deficiency, pest, or environmental cause.

## First smoke run

This verifies the full download, training, evaluation, and ONNX export path
with a deliberately small sample. It does not produce a useful model:

```powershell
python training\train.py `
  --manifest training\dataset_manifest_smoke.json `
  --task binary-health `
  --epochs 1 `
  --output-dir models\smoke-binary
```

The smoke manifest is a pinned, 1,900-image stratified debug subset with the
same schema. Its metrics are not evidence of production performance.

## Full initial binary model

Train all crops for a stronger generic baseline. Class-balanced batch loss is
used so the larger abnormal group does not dominate the healthy group. The
default freezes the ImageNet backbone, extracts every image feature once, and
then trains the two-class head for five fast in-memory epochs:

```powershell
python training\train.py `
  --task binary-health `
  --epochs 5 `
  --batch-size 64 `
  --learning-rate 0.001 `
  --num-workers 4
```

The command downloads the revision pinned in
`training/dataset_manifest.json` and exports runtime files under
`models/current`.

The script defaults to two data-loader workers. The recommended full command
above uses four. CUDA mixed precision is enabled when CUDA is available, with
live progress every 100 batches and a 30-minute safety bound.
Override those controls with `--num-workers`, `--log-every`, or
`--max-training-minutes`. Use `--keep-dataset-cache` only when deliberately
running repeated experiments; otherwise no dataset copy remains afterward.

Use `--fine-tune-backbone --learning-rate 0.0001` only for a deliberate full
fine-tuning experiment. It decodes and augments every image on every epoch and
is not the time-bounded default baseline.

`model_metadata.json` records accuracy, macro F1, the confusion matrix, and
precision/recall/F1/support for both classes. Abnormal recall is especially
important: high overall accuracy alone is not sufficient.

The optional old-style disease experiment remains available with
`--task multiclass-disease --crops Tomato`, but it is not the generic model.

## Publish without a paid endpoint

Authenticate the Hugging Face CLI, create a public model repository, and run:

```powershell
python training\upload_model.py `
  --repo-id YOUR_HF_USERNAME/plantsense-health-model
```

The command prints the immutable commit hash. Put that hash and repository ID
in `.env` as `PLANTSENSE_MODEL_REVISION` and `PLANTSENSE_MODEL_REPO`. Runtime
machines then download only the ONNX model, labels, and metadata on first use.

The default export does not create a PyTorch checkpoint because it is not
needed for inference. Add `--save-checkpoint` only when you intend to resume
training, and do not publish that checkpoint with the runtime artifacts.

## Required real-camera work

PlantVillage is only a baseline. Before reporting field reliability, collect
and label ESP32-CAM images as `healthy` or `abnormal`, keep entire
plants/capture sessions in only one split, fine-tune on those images, and
report both per-class recall and macro F1. Include visually normal plants and
non-disease stress so the model learns the actual deployment environment.
