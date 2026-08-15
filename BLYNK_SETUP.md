# PlantSense Blynk setup

## 1. Create the template and device

1. Sign in to Blynk.Console and enable Developer Mode.
2. Create a template named **PlantSense AI Pod** with hardware **ESP32** and
   connection type **WiFi**.
3. Create a device from that template.
4. Copy `src/blynk_credentials.example.h` to `src/blynk_credentials.h`.
5. Add the Template ID, Auth Token, Wi-Fi details, and local AI server URL to
   `src/blynk_credentials.h`.

Do not commit `src/blynk_credentials.h`; it contains the device secret.

## 2. Create template datastreams

Create these six Virtual Pin datastreams in the template:

| Name | Pin | Type | Units | Minimum | Maximum |
| --- | --- | --- | --- | ---: | ---: |
| Temperature | V0 | Double | Celsius | -20 | 80 |
| Humidity | V1 | Double | Percentage | 0 | 100 |
| Soil Moisture | V2 | Double | Percentage | 0 | 100 |
| Light | V3 | Double | Lux | 0 | 100000 |
| Pump State | V4 | Integer | None | 0 | 1 |
| Grow Light State | V5 | Integer | None | 0 | 1 |

Use one or two decimal places for the four sensor datastreams. Pump and grow
light use `0` for OFF and `1` for ON.

## 3. Build the dashboards

Add gauge, label, or chart widgets for V0 through V3. Add LED widgets for V4
and V5. Assign each widget to the datastream with the corresponding virtual
pin. Web and mobile dashboards are configured separately in Blynk.

## 4. Build and upload

PlatformIO installs all firmware libraries from `platformio.ini`:

```powershell
platformio run -e esp32cam
platformio run -e esp32cam -t upload --upload-port COM5
platformio device monitor --port COM5 --baud 115200
```

Replace `COM5` with the USB-to-serial adapter port. When WiFi and Blynk are
available, the six
datastreams are updated after every sensor cycle (currently every five
seconds). Local pump and grow-light control continues if Blynk is offline.
