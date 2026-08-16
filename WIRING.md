# PlantSense wiring

## ESP32-CAM signal map

| Component | Connection | ESP32-CAM |
| --- | --- | --- |
| BH1750 | SDA | GPIO13 |
| BH1750 | SCL | GPIO14 |
| ADS1115 | SDA | GPIO13 |
| ADS1115 | SCL | GPIO14 |
| Soil sensor | AOUT | ADS1115 A0 |
| DHT22 | DATA | GPIO2 |
| Grow-light MOSFET | Gate | GPIO15 |
| Pump MOSFET | Gate | GPIO4 |
| FTDI | TX | U0R / GPIO3 |
| FTDI | RX | U0T / GPIO1 |

## Power and addresses

- Connect every module, MOSFET source, motor supply, and FTDI ground together.
- Power the BH1750 and ADS1115 from 3.3 V so their I2C pull-ups cannot place
  5 V on the ESP32 pins.
- Power the soil sensor from the same 3.3 V rail as the ADS1115.
- Power the DHT22 from 3.3 V and add a 4.7-10 kΩ pull-up from DATA to 3.3 V.
- Connect BH1750 ADDR low/GND for address `0x23`.
- Connect ADS1115 ADDR to GND for address `0x48`.
- Never apply more than the ADS1115 supply voltage to A0.

## MOSFET loads

- Use 3.3 V logic-level N-channel MOSFETs as low-side switches.
- Add a 100-220 Ω series resistor at each gate and a 10 kΩ gate-to-GND
  pull-down so the pump and light remain off during reset.
- Connect each MOSFET source to GND and drain to the load negative terminal.
- Connect each load positive terminal to its rated supply.
- Put a flyback diode across the pump motor: cathode to motor positive and
  anode to motor negative/MOSFET drain.
- Use the correct current limiting or driver for the grow light.
- Do not power the pump or grow light from an ESP32 GPIO or 3.3 V pin.

## ESP32-CAM restrictions

- The RHYX M21-45 camera uses RGB565 capture with firmware JPEG conversion;
  keep the camera ribbon fully seated and use a stable 5 V supply.
- The microSD slot cannot be used because GPIO2, GPIO4, GPIO13, GPIO14, and
  GPIO15 are assigned to this circuit.
- GPIO4 also drives the onboard flash LED, so it lights when the pump output
  is active.
- GPIO2 and GPIO15 are boot-strapping pins. The gate pull-down on GPIO15 is
  required. If flashing fails, temporarily disconnect the DHT22 DATA wire and
  its GPIO2 pull-up.
- Use 3.3 V UART logic. Connect FTDI TX to GPIO3 and FTDI RX to GPIO1.
- Hold GPIO0 to GND while uploading, reset the board, then remove GPIO0 from
  GND and reset again to run the firmware.
- Use a stable 5 V supply for the ESP32-CAM and do not power the motor through
  the FTDI adapter.

## Calibration

- ADS1115 channel A0 is already selected in `readSoilMoisture()`.
- Replace `DRY_VALUE` and `WET_VALUE` with measurements from the installed
  soil sensor before relying on the displayed percentage.
