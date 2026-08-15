# PlantSense AI Pod

PlantSense is a zero-cost plant monitoring prototype. An ESP32-CAM reads a
DHT22, BH1750, and ADS1115-connected soil sensor, controls a pump and grow
light, sends images to a local Flask API, and publishes readings to Blynk.
The included MobileNetV3-Small ONNX model performs a generic `healthy` versus
`abnormal` visual screen; it is not a disease diagnosis.

## 1. Install

- [Git for Windows](https://git-scm.com/download/win)
- [64-bit Python 3.10](https://www.python.org/downloads/release/python-31011/)
- Internet access for the first dependency and training-dataset download
- Optional: an NVIDIA GPU with a current driver for faster training
- For firmware: ESP32-CAM, USB-to-serial adapter, and the connected sensors
- For dashboards: a [Blynk Free](https://www.blynk.io/pricing) account

The commands below use Windows PowerShell from the project directory. Python
3.9-3.12 is supported, but Python 3.10 is the tested version.

## 2. Clone

```powershell
git clone https://github.com/Roy111002/PlantSense.git
Set-Location PlantSense
```

## 3. Run the AI server

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints-tested.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py
```

In a second PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health
```

Expected result: `status` is `ok` and the model source is `local`. The trained
5.81 MiB ONNX model is included, so inference does not download the dataset or
call a paid API.

For the ESP32-CAM, find the computer's local IPv4 address with `ipconfig`, keep
both devices on the same network, and allow Python on private networks when
Windows Firewall prompts.

## 4. Train the generic classifier

Create a separate environment:

```powershell
py -3.10 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install --upgrade pip
```

Install one PyTorch build.

NVIDIA GPU, matching the tested environment:

```powershell
.\.venv-training\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

CPU fallback:

```powershell
.\.venv-training\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

Install the remaining dependencies and verify the environment:

```powershell
.\.venv-training\Scripts\python.exe -m pip install -r requirements.txt -r requirements-training.txt -c constraints-tested.txt
.\.venv-training\Scripts\python.exe -m pip check
.\.venv-training\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -v
```

Run the small pipeline check:

```powershell
.\.venv-training\Scripts\python.exe -u training\train.py --manifest training\dataset_manifest_smoke.json --task binary-health --epochs 1 --batch-size 64 --num-workers 2 --max-training-minutes 5 --output-dir models\smoke-binary --device auto
```

Run the full five-epoch training:

```powershell
.\.venv-training\Scripts\python.exe -u training\train.py --task binary-health --epochs 5 --batch-size 64 --learning-rate 0.001 --num-workers 4 --max-training-minutes 30 --output-dir models\current --device auto
```

- The revision-pinned PlantVillage dataset is downloaded once per run.
- The first full download is approximately 855 MiB.
- `data/training-cache` is removed automatically on success or failure.
- The 30-minute guard aborts a slow run at the next batch.
- `--device auto` uses CUDA when available and otherwise uses the CPU.
- Only `models/current` is committed; datasets, caches, smoke models, and
  checkpoints are ignored.

The included model was trained on 54,304 images and has 1,519,906 parameters.
Its held-out accuracy is 93.31%, macro F1 is 92.08%, and abnormal recall is
91.27%. See [TRAINING.md](TRAINING.md) for methodology and limitations.

## 5. Configure Blynk and the ESP32-CAM

Install an isolated PlatformIO CLI:

```powershell
py -3.10 -m venv .venv-platformio
.\.venv-platformio\Scripts\python.exe -m pip install "platformio==6.1.19"
Copy-Item src\blynk_credentials.example.h src\blynk_credentials.h
```

Edit `src\blynk_credentials.h` and set:

- `BLYNK_TEMPLATE_ID`
- `BLYNK_AUTH_TOKEN`
- `PLANTSENSE_WIFI_SSID`
- `PLANTSENSE_WIFI_PASSWORD`
- `PLANTSENSE_AI_SERVER_URL`, using the computer's IPv4 address and `/analyze`

Example server URL: `http://192.168.1.10:5000/analyze`.

Before powering actuators, replace every GPIO value of `99` in `src/main.cpp`
with the final circuit pin map and calibrate `DRY_VALUE`, `WET_VALUE`, and the
control thresholds. The ADS1115 reads soil moisture on channel A0.

Create the six Blynk datastreams and dashboard widgets described in
[BLYNK_SETUP.md](BLYNK_SETUP.md). Then build and upload, replacing `COM5` with
the adapter's port:

```powershell
.\.venv-platformio\Scripts\platformio.exe run -e esp32cam
.\.venv-platformio\Scripts\platformio.exe run -e esp32cam -t upload --upload-port COM5
.\.venv-platformio\Scripts\platformio.exe device monitor --port COM5 --baud 115200
```

For a typical USB-to-serial adapter, connect GPIO0 to GND while flashing,
reset the ESP32-CAM, upload, then disconnect GPIO0 from GND and reset again.
Use a stable 5 V supply and a common ground.

## 6. Run one complete project cycle — commands only

### PowerShell 1

```powershell
.\.venv\Scripts\python.exe main.py
```

### PowerShell 2

```powershell
Invoke-RestMethod http://127.0.0.1:5000/health | ConvertTo-Json -Depth 4
$PlantSensePort = "COM5"
.\.venv-platformio\Scripts\platformio.exe run -e esp32cam
.\.venv-platformio\Scripts\platformio.exe run -e esp32cam -t upload --upload-port $PlantSensePort
.\.venv-platformio\Scripts\platformio.exe device monitor --port $PlantSensePort --baud 115200
```

### PowerShell 3

```powershell
Start-Process "https://blynk.cloud"
Start-Sleep -Seconds 310
```

## 7. Share a newly trained model

```powershell
git pull
git add models/current
git commit -m "Update trained plant-health model"
git push
```

Never commit `.env` or `src/blynk_credentials.h`. Virtual environments,
datasets, PlatformIO builds, caches, and training checkpoints are also ignored.

## Project references

- [AI_PIPELINE.md](AI_PIPELINE.md): API and inference flow
- [TRAINING.md](TRAINING.md): dataset, training strategy, and evaluation
- [BLYNK_SETUP.md](BLYNK_SETUP.md): Blynk datastreams and widgets
- [Project report](PlantSense%20AI%20Pod-%20An%20IoT-Based%20Smart%20Plant%20Stress%20%26%20Disease%20Detection%20System.pdf)
