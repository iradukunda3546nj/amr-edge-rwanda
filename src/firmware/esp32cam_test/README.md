# ESP32-CAM model test

Runs the AMR-Edge INT8 digit model on an AI-Thinker ESP32-CAM from Arduino IDE. It tests the model only: no LTE modem and no telemetry. The production firmware is [../esp32_edge_ocr.cpp](../esp32_edge_ocr.cpp), which targets the ESP32-S3.

## Build

- Arduino IDE 2 with the **esp32 by Espressif** boards package, version 3.x.
- Tools > Board: **AI Thinker ESP32-CAM**. Keep the default partition scheme (Huge APP) and leave PSRAM enabled.
- No extra libraries are needed. The INT8 kernels are plain C++ in `amr_model.cpp`.

Optional: set `WIFI_SSID` and `WIFI_PASS` in `esp32cam_test.ino`. Otherwise the board starts its own access point, `AMR-ESP32CAM` (password `amredge123`), at http://192.168.4.1.

## What it checks

At boot, it prints the following to the Serial Monitor (115200 baud):

- **Kernels:** the INT8 output (logits) for 64 golden inputs, compared byte for byte with TensorFlow Lite's reference kernels.
- **Preprocessing:** 8 raw crops, compared pixel for pixel with the Python training transform.
- **Latency:** inference and preprocessing time per wheel, measured on this chip.

The web page captures a frame. Drag a box around the meter's digit window and set the wheel count. The board then:

- splits the box into wheels;
- centres a crop on each digit with the training framing ("Fit to digit"; the glyph is about 62% of the crop height);
- reports the reading, confidences, the crops (green boxes), the 24×32 model inputs and per-stage timings.

**Mirror** and **Flip** set the sensor orientation and are saved on the board.

## Test material

The model reads printed meter-wheel digits. It was trained on one meter model's dataset plus synthetic wheels in common fonts, and has never seen handwriting. Point the camera at either:

- a real water meter, or
- the printed [test-card.html](test-card.html) (print at 100% on paper).

Avoid filming a screen, which adds moiré.

## After retraining

```
node tools/gen-esp32cam.cjs
```

This copies `digit_model_int8.h` into this folder and regenerates `golden_vectors.h`.
