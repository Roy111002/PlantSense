# PlantSense AI pipeline

PlantSense performs image inference locally. No dataset is deployed with the
server and no paid inference API is called.

## Runtime flow

1. The RHYX M21-45 captures a QVGA RGB565 frame. Firmware converts it to JPEG
   and posts it with sensor readings to `/analyze`.
2. The server rejects unreadable, extremely dark, overexposed, or undersized
   images.
3. ONNX Runtime classifies an acceptable image as `healthy` or `abnormal`.
4. Low-confidence or closely tied predictions become `uncertain`.
5. Deterministic rules combine the visual prediction with the four sensor
   assessments.
6. The server returns JSON compatible with the ESP32 firmware.

The generic model does not name a disease or prescribe treatment. An abnormal
result triggers manual inspection and is combined with sensor evidence.

## Runtime setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Choose one model source in `.env`:

- Leave `PLANTSENSE_MODEL_REPO` empty to load the compact, versionable
  `models/current` runtime artifact shipped with the project.
- Set `PLANTSENSE_MODEL_REPO` and its full 40-character commit hash to
  download a public model artifact once and use the Hugging Face cache after
  that.

Leave `PLANT_CROP` blank for the generic binary model. It is used only if you
deliberately train the optional multiclass disease model.

Start the API:

```powershell
python main.py
Invoke-RestMethod http://localhost:5000/health
```

The checked-in model makes `/health` report `ok` after a normal fresh setup.
If that artifact is removed or invalid, `/health` reports `degraded` and
`/analyze` returns sensor-based assessment with vision marked `unavailable`.

The checked-in runtime artifact is about 5.81 MiB. Dataset files, PyTorch
checkpoints, caches, and smoke models remain ignored, so a new machine needs
only the project files and runtime dependencies, not the 54,304-image dataset.

## Training

See `TRAINING.md`. Training uses a separate environment and a disposable local
cache of the revision-pinned dataset so multiple epochs do not repeat network
downloads. The deployed environment installs neither PyTorch nor the dataset.
