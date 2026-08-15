/*
   PlantSense AI Pod
   =================

   ESP32-CAM firmware

   Responsibilities:
   - Capture plant image
   - Read DHT22
   - Read soil moisture through ADS1115
   - Read BH1750
   - Control water pump
   - Control grow light
   - Apply local threshold rules
   - Send image + sensor data to Python AI server

   IMPORTANT:
   This pin map requires the microSD slot to remain unused.

*/

#ifdef __has_include
#if __has_include("blynk_credentials.h")
#include "blynk_credentials.h"
#endif
#endif

// Arduino IDE fallback. Edit these values only when the credentials header
// is not included with the sketch.
#ifndef BLYNK_TEMPLATE_ID
#define BLYNK_TEMPLATE_ID "YOUR_BLYNK_TEMPLATE_ID"
#endif

#ifndef BLYNK_TEMPLATE_NAME
#define BLYNK_TEMPLATE_NAME "PlantSense AI Pod"
#endif

#ifndef BLYNK_AUTH_TOKEN
#define BLYNK_AUTH_TOKEN "YOUR_BLYNK_DEVICE_AUTH_TOKEN"
#endif

#ifndef PLANTSENSE_WIFI_SSID
#define PLANTSENSE_WIFI_SSID "YOUR_WIFI_SSID"
#endif

#ifndef PLANTSENSE_WIFI_PASSWORD
#define PLANTSENSE_WIFI_PASSWORD "YOUR_WIFI_PASSWORD"
#endif

#ifndef PLANTSENSE_AI_SERVER_URL
#define PLANTSENSE_AI_SERVER_URL "http://YOUR_SERVER_IP:5000/analyze"
#endif

#define BLYNK_PRINT Serial

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BlynkSimpleEsp32.h>

#include "esp_camera.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include <DHT.h>
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_ADS1X15.h>


const char* WIFI_SSID = PLANTSENSE_WIFI_SSID;
const char* WIFI_PASSWORD = PLANTSENSE_WIFI_PASSWORD;
const char* AI_SERVER_URL = PLANTSENSE_AI_SERVER_URL;


// ============================================================
//                     BLYNK DATASTREAMS
// ============================================================

// These virtual pins must match the datastreams configured in the
// PlantSense template in Blynk.Console.

const uint8_t BLYNK_TEMPERATURE_VPIN = V0;
const uint8_t BLYNK_HUMIDITY_VPIN = V1;
const uint8_t BLYNK_SOIL_MOISTURE_VPIN = V2;
const uint8_t BLYNK_LIGHT_VPIN = V3;
const uint8_t BLYNK_PUMP_STATE_VPIN = V4;
const uint8_t BLYNK_GROW_LIGHT_STATE_VPIN = V5;


// ============================================================
//                     GPIO CONFIGURATION
// ============================================================

// GPIO2, GPIO4, GPIO13, GPIO14, and GPIO15 are shared with the
// AI Thinker ESP32-CAM microSD interface. Do not initialize microSD.

#define DHT_PIN               2

#define PUMP_PIN              4
#define GROW_LIGHT_PIN       15

#define I2C_SDA_PIN          13
#define I2C_SCL_PIN          14

const uint8_t BH1750_I2C_ADDRESS = 0x23;
const uint8_t ADS1115_I2C_ADDRESS = 0x48;


// ============================================================
//                     CAMERA PINS
// ============================================================

// AI Thinker ESP32-CAM camera configuration

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


// ============================================================
//                     SENSOR OBJECTS
// ============================================================

#define DHT_TYPE DHT22

DHT dht(DHT_PIN, DHT_TYPE);

BH1750 lightMeter;

Adafruit_ADS1115 ads;


// ============================================================
//                     THRESHOLDS
// ============================================================

// These are PLACEHOLDER values.
// They must be calibrated experimentally for the actual plant.

float MIN_TEMPERATURE = 15.0;
float MAX_TEMPERATURE = 32.0;

float MIN_HUMIDITY = 40.0;
float MAX_HUMIDITY = 85.0;

float MIN_LIGHT_LUX = 2000.0;

// Soil moisture is represented as percentage after calibration.

float SOIL_DRY_THRESHOLD = 30.0;
float SOIL_WET_THRESHOLD = 75.0;


// ============================================================
//                     TIMING
// ============================================================

unsigned long lastSensorRead = 0;
unsigned long lastImageUpload = 0;

const unsigned long SENSOR_INTERVAL = 5000;

// Image capture every 5 minutes

const unsigned long IMAGE_INTERVAL = 300000;

// Retry Blynk without blocking the local plant-control loop.

const unsigned long BLYNK_RECONNECT_INTERVAL = 30000;

unsigned long lastBlynkReconnectAttempt = 0;


// ============================================================
//                     GLOBAL SENSOR VALUES
// ============================================================

float temperature = 0;
float humidity = 0;

float lightLux = 0;
float soilMoisture = 0;


// ============================================================
//                     STATE
// ============================================================

bool pumpState = false;
bool growLightState = false;

bool sensorDataAvailable = false;
bool blynkConfigured = false;


// ============================================================
//                     CAMERA INITIALIZATION
// ============================================================

bool initializeCamera()
{
    // Zero-initialize the full ESP32 camera configuration. Newer versions of
    // the ESP32 Arduino core add fields to this struct.
    camera_config_t config = {};

    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;

    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;

    config.pin_xclk = XCLK_GPIO_NUM;

    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;

    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;

    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;

    config.xclk_freq_hz = 20000000;

    config.pixel_format = PIXFORMAT_JPEG;

    if (psramFound())
    {
        config.frame_size = FRAMESIZE_VGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
    }
    else
    {
        config.frame_size = FRAMESIZE_QVGA;
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);

    if (err != ESP_OK)
    {
        Serial.printf(
            "Camera initialization failed: 0x%x\n",
            err
        );

        return false;
    }

    Serial.println("Camera initialized.");

    return true;
}


// ============================================================
//                     WIFI CONNECTION
// ============================================================

void connectWiFi()
{
    Serial.print("Connecting to WiFi");

    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.persistent(false);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    int attempts = 0;

    while (WiFi.status() != WL_CONNECTED && attempts < 30)
    {
        delay(500);

        Serial.print(".");

        attempts++;
    }

    Serial.println();

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("WiFi connected.");

        Serial.print("ESP32 IP: ");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("WiFi connection failed.");
    }
}


// ============================================================
//                     BLYNK CONNECTION
// ============================================================

bool blynkCredentialsAreConfigured()
{
    return
        strcmp(
            BLYNK_TEMPLATE_ID,
            "YOUR_BLYNK_TEMPLATE_ID"
        ) != 0 &&
        strcmp(
            BLYNK_AUTH_TOKEN,
            "YOUR_BLYNK_DEVICE_AUTH_TOKEN"
        ) != 0;
}


void publishSensorDataToBlynk()
{
    if (
        !blynkConfigured ||
        !sensorDataAvailable ||
        !Blynk.connected()
    )
    {
        return;
    }

    Blynk.virtualWrite(
        BLYNK_TEMPERATURE_VPIN,
        temperature
    );

    Blynk.virtualWrite(
        BLYNK_HUMIDITY_VPIN,
        humidity
    );

    Blynk.virtualWrite(
        BLYNK_SOIL_MOISTURE_VPIN,
        soilMoisture
    );

    Blynk.virtualWrite(
        BLYNK_LIGHT_VPIN,
        lightLux
    );

    Blynk.virtualWrite(
        BLYNK_PUMP_STATE_VPIN,
        pumpState ? 1 : 0
    );

    Blynk.virtualWrite(
        BLYNK_GROW_LIGHT_STATE_VPIN,
        growLightState ? 1 : 0
    );
}


void initializeBlynk()
{
    blynkConfigured =
        blynkCredentialsAreConfigured();

    if (!blynkConfigured)
    {
        Serial.println(
            "Blynk credentials are not configured; "
            "dashboard publishing is disabled."
        );

        return;
    }

    Blynk.config(BLYNK_AUTH_TOKEN);

    if (
        WiFi.status() == WL_CONNECTED &&
        Blynk.connect(5000)
    )
    {
        Serial.println("Blynk connected.");
    }
    else
    {
        Serial.println(
            "Blynk connection failed; will retry."
        );
    }

    lastBlynkReconnectAttempt = millis();
}


void maintainBlynkConnection(unsigned long now)
{
    if (!blynkConfigured)
    {
        return;
    }

    if (WiFi.status() != WL_CONNECTED)
    {
        return;
    }

    if (Blynk.connected())
    {
        Blynk.run();

        return;
    }

    if (
        now - lastBlynkReconnectAttempt <
        BLYNK_RECONNECT_INTERVAL
    )
    {
        return;
    }

    lastBlynkReconnectAttempt = now;

    Serial.println("Reconnecting to Blynk...");

    if (Blynk.connect(2000))
    {
        Serial.println("Blynk reconnected.");

        publishSensorDataToBlynk();
    }
    else
    {
        Serial.println("Blynk reconnect failed.");
    }
}


// ============================================================
//                     SENSOR INITIALIZATION
// ============================================================

void initializeSensors()
{
    dht.begin();

    Wire.begin(
        I2C_SDA_PIN,
        I2C_SCL_PIN
    );

    if (lightMeter.begin(
        BH1750::CONTINUOUS_HIGH_RES_MODE,
        BH1750_I2C_ADDRESS,
        &Wire
    ))
    {
        Serial.println("BH1750 initialized.");
    }
    else
    {
        Serial.println("BH1750 initialization failed.");
    }

    if (ads.begin(
        ADS1115_I2C_ADDRESS,
        &Wire
    ))
    {
        Serial.println("ADS1115 initialized.");

        ads.setGain(GAIN_ONE);
    }
    else
    {
        Serial.println("ADS1115 initialization failed.");
    }
}


// ============================================================
//                     SOIL MOISTURE
// ============================================================

float readSoilMoisture()
{
    /*
       ADS1115 channel 0 is used here.

       Calibration required:

       DRY_VALUE = ADC reading in completely dry soil
       WET_VALUE = ADC reading in fully wet soil

       Replace these after testing.
    */

    int16_t raw = ads.readADC_SingleEnded(0);

    const float DRY_VALUE = 20000.0;
    const float WET_VALUE = 8000.0;

    float percentage =
        100.0 *
        (DRY_VALUE - raw) /
        (DRY_VALUE - WET_VALUE);

    percentage = constrain(
        percentage,
        0.0,
        100.0
    );

    return percentage;
}


// ============================================================
//                     READ ALL SENSORS
// ============================================================

void readSensors()
{
    float newTemperature = dht.readTemperature();
    float newHumidity = dht.readHumidity();

    if (!isnan(newTemperature))
    {
        temperature = newTemperature;
    }

    if (!isnan(newHumidity))
    {
        humidity = newHumidity;
    }

    lightLux = lightMeter.readLightLevel();

    soilMoisture = readSoilMoisture();

    sensorDataAvailable = true;


    Serial.println();
    Serial.println("========== SENSOR DATA ==========");

    Serial.print("Temperature: ");
    Serial.print(temperature);
    Serial.println(" C");

    Serial.print("Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");

    Serial.print("Light: ");
    Serial.print(lightLux);
    Serial.println(" lux");

    Serial.print("Soil Moisture: ");
    Serial.print(soilMoisture);
    Serial.println(" %");

    Serial.println("=================================");
}


// ============================================================
//                     LOCAL DECISION ENGINE
// ============================================================

void controlPlant()
{
    // --------------------------------------------------------
    // SOIL MOISTURE
    // --------------------------------------------------------

    if (soilMoisture < SOIL_DRY_THRESHOLD)
    {
        digitalWrite(PUMP_PIN, HIGH);

        pumpState = true;

        Serial.println(
            "[CONTROL] Soil dry -> PUMP ON"
        );
    }

    else if (soilMoisture > SOIL_WET_THRESHOLD)
    {
        digitalWrite(PUMP_PIN, LOW);

        pumpState = false;

        Serial.println(
            "[CONTROL] Soil sufficiently wet -> PUMP OFF"
        );
    }


    // --------------------------------------------------------
    // GROW LIGHT
    // --------------------------------------------------------

    if (lightLux < MIN_LIGHT_LUX)
    {
        digitalWrite(GROW_LIGHT_PIN, HIGH);

        growLightState = true;

        Serial.println(
            "[CONTROL] Low light -> GROW LIGHT ON"
        );
    }

    else
    {
        digitalWrite(GROW_LIGHT_PIN, LOW);

        growLightState = false;

        Serial.println(
            "[CONTROL] Sufficient light -> GROW LIGHT OFF"
        );
    }


    // --------------------------------------------------------
    // TEMPERATURE WARNING
    // --------------------------------------------------------

    if (
        temperature < MIN_TEMPERATURE ||
        temperature > MAX_TEMPERATURE
    )
    {
        Serial.println(
            "[WARNING] Temperature outside preferred range"
        );
    }


    // --------------------------------------------------------
    // HUMIDITY WARNING
    // --------------------------------------------------------

    if (
        humidity < MIN_HUMIDITY ||
        humidity > MAX_HUMIDITY
    )
    {
        Serial.println(
            "[WARNING] Humidity outside preferred range"
        );
    }
}


// ============================================================
//                     CAPTURE IMAGE
// ============================================================

camera_fb_t* captureImage()
{
    camera_fb_t* fb = esp_camera_fb_get();

    if (!fb)
    {
        Serial.println(
            "Camera capture failed."
        );

        return nullptr;
    }

    Serial.print("Image captured: ");

    Serial.print(fb->len);

    Serial.println(" bytes");

    return fb;
}


// ============================================================
//                     SEND IMAGE + DATA
// ============================================================

bool sendToAI(camera_fb_t* fb)
{
    if (!fb)
        return false;

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println(
            "WiFi unavailable."
        );

        return false;
    }


    HTTPClient http;

    http.begin(AI_SERVER_URL);

    http.setTimeout(30000);


    // Multipart request

    String boundary =
        "----PlantSenseBoundary";


    String bodyStart =
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"temperature\"\r\n\r\n" +
        String(temperature) +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"humidity\"\r\n\r\n" +
        String(humidity) +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"soil_moisture\"\r\n\r\n" +
        String(soilMoisture) +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"light_lux\"\r\n\r\n" +
        String(lightLux) +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"pump_state\"\r\n\r\n" +
        String(pumpState ? "ON" : "OFF") +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"grow_light_state\"\r\n\r\n" +
        String(growLightState ? "ON" : "OFF") +
        "\r\n" +

        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; "
        "name=\"image\"; filename=\"plant.jpg\"\r\n"
        "Content-Type: image/jpeg\r\n\r\n";


    String bodyEnd =
        "\r\n--" +
        boundary +
        "--\r\n";


    size_t totalLength =
        bodyStart.length() +
        fb->len +
        bodyEnd.length();


    // Allocate complete request

    uint8_t* request =
        (uint8_t*) malloc(totalLength);


    if (!request)
    {
        Serial.println(
            "Memory allocation failed."
        );

        http.end();

        return false;
    }


    size_t offset = 0;


    memcpy(
        request + offset,
        bodyStart.c_str(),
        bodyStart.length()
    );

    offset += bodyStart.length();


    memcpy(
        request + offset,
        fb->buf,
        fb->len
    );

    offset += fb->len;


    memcpy(
        request + offset,
        bodyEnd.c_str(),
        bodyEnd.length()
    );


    String contentType =
        "multipart/form-data; boundary=" +
        boundary;


    http.addHeader(
        "Content-Type",
        contentType
    );


    Serial.println(
        "[AI] Sending image to server..."
    );


    int responseCode =
        http.POST(
            request,
            totalLength
        );


    free(request);


    if (responseCode > 0)
    {
        Serial.print(
            "[AI] HTTP response: "
        );

        Serial.println(responseCode);


        String response =
            http.getString();


        Serial.println(
            "[AI] Server response:"
        );

        Serial.println(response);


        // ----------------------------------------------------
        // Parse AI response
        // ----------------------------------------------------

        StaticJsonDocument<4096> doc;

        DeserializationError error =
            deserializeJson(
                doc,
                response
            );


        if (!error)
        {
            const char* disease =
                doc["disease"];

            const char* stress =
                doc["stress"];

            float confidence =
                doc["confidence"];

            const char* recommendation =
                doc["recommendation"];


            Serial.println();
            Serial.println(
                "========== AI RESULT =========="
            );

            Serial.print(
                "Disease: "
            );

            Serial.println(disease);


            Serial.print(
                "Stress: "
            );

            Serial.println(stress);


            Serial.print(
                "Confidence: "
            );

            Serial.println(confidence);


            Serial.print(
                "Recommendation: "
            );

            Serial.println(recommendation);


            Serial.println(
                "==============================="
            );
        }


        http.end();

        return true;
    }


    Serial.print(
        "[AI] HTTP error: "
    );

    Serial.println(responseCode);


    http.end();

    return false;
}


// ============================================================
//                         SETUP
// ============================================================

void setup()
{
    Serial.begin(115200);

    delay(1000);


    // Prevent brownout reset during development.
    // Revisit this for the final power design.

    WRITE_PERI_REG(
        RTC_CNTL_BROWN_OUT_REG,
        0
    );


    Serial.println();
    Serial.println(
        "======================================"
    );

    Serial.println(
        "       PlantSense AI Pod"
    );

    Serial.println(
        "======================================"
    );


    // Output pins

    pinMode(
        PUMP_PIN,
        OUTPUT
    );

    pinMode(
        GROW_LIGHT_PIN,
        OUTPUT
    );


    digitalWrite(
        PUMP_PIN,
        LOW
    );

    digitalWrite(
        GROW_LIGHT_PIN,
        LOW
    );


    initializeCamera();

    initializeSensors();

    connectWiFi();

    initializeBlynk();
}


// ============================================================
//                          LOOP
// ============================================================

void loop()
{
    unsigned long now =
        millis();

    maintainBlynkConnection(now);


    // --------------------------------------------------------
    // SENSOR PROCESSING
    // --------------------------------------------------------

    if (
        now - lastSensorRead >=
        SENSOR_INTERVAL
    )
    {
        lastSensorRead = now;

        readSensors();

        controlPlant();

        publishSensorDataToBlynk();
    }


    // --------------------------------------------------------
    // IMAGE + AI PROCESSING
    // --------------------------------------------------------

    if (
        now - lastImageUpload >=
        IMAGE_INTERVAL
    )
    {
        lastImageUpload = now;


        camera_fb_t* fb =
            captureImage();


        if (fb)
        {
            sendToAI(fb);

            esp_camera_fb_return(fb);
        }
    }


    delay(100);
}
