# AMR-Edge Rwanda

**An INT8 model on the meter reads the register and sends a 20-byte authenticated reading over cellular to WASAC and the customer's phone.**

Hack-Nation 7th Global AI Hackathon · *Small AI for Development* (World Bank Group) · UN SDG 6

---

## Why

Rwanda loses roughly 38–40% of treated water as non-revenue water (WASAC). Meters are read by hand every 30–90 days, so a leak behind the meter runs for a whole cycle. Low-income households end up with bills they cannot pay and are disconnected. Supply rationing and water fetching fall mostly on women and girls. Revenue that is never collected is money not spent extending the network.

AMR-Edge clips onto the existing mechanical meter, so no replacement or plumbing is needed. It reads the meter on the device and reports daily, which brings leak detection down from months to about a day.

## What is real in this repository

| Component | Status |
|---|---|
| Digit recogniser | Trained here. INT8 full-integer post-training quantisation, **20.1 KB** TFLite flatbuffer, **7.9 KB** weights, 7,867 parameters, 493k MACs per wheel |
| Browser INT8 engine | Port of the TFLite reference integer kernels. **Bit-exact** with the TFLite interpreter on golden vectors (`tools/test`) |
| Preprocessing | Python (training) and JS (inference) produce **bit-identical** output (tested) |
| Register localisation | Classical CV: adaptive thresholding in both polarities, glyph chaining, pitch prior, orientation search. Operator override available |
| Payload crypto | AES-128-CTR + AES-CMAC (RFC 4493 vectors pass), WebCrypto in the browser, mbedTLS on the device |
| MQTT 3.1.1 | Real CONNECT / PUBLISH / PUBACK bytes; the back end parses the same frame the modem sent |
| Cellular session | Simulated: SIM7600 AT commands, LTE RRC/NAS attach, TCP, TLS 1.2, MQTT, with nominal timings |
| WASAC back end, SMS | Simulated: CMAC verification, replay rejection, ledger validation, block tariff, 160-character GSM-7 SMS |
| Firmware | Reference ESP-IDF code (ESP32-S3 + SIM7600 + ESP-NN). Not yet built or flashed |

## Model

| | |
|---|---|
| Task | Classify one odometer wheel crop: 10 digits + 1 **reject class** (CE marks, unit symbols, letters next to the register) |
| Input | 24×32 grayscale, polarity-normalised, 2–98th percentile stretch, int8 (scale 1/255, zero point −128) |
| Architecture | Conv3×3(16) → 3 depthwise-separable blocks (32, 48, 64) → AvgPool → FC(11); BN folded at conversion |
| Quantisation | Per-channel symmetric int8 weights, per-tensor asymmetric int8 activations, int32 bias, fixed-point requantisation |
| Data | 22,286 real meter-wheel crops (Nikić et al., *Future Internet* 16(11):402, 2024, GPL-3.0) with augmentation, plus synthetic register crops in 15 typefaces |
| Validation | FP32 100.0% / INT8 100.0% on 3,183 held-out real crops (split by blocks of consecutive video frames to avoid leakage); 98.3% INT8 on held-out synthetic crops in other fonts |
| Runtime on device | ESP-NN kernels called directly from a generated layer table. TFLM's interpreter would cost more flash than the model itself |

The real dataset comes from one meter type, so 100% validation measures fit, not generalisation. The synthetic held-out set and the photographs in `src/web/samples/` are the out-of-domain checks.

## Architecture

```
 Edge node (ESP32-S3)                         Cellular                        WASAC
 ┌───────────────────────────────┐   UART   ┌─────────┐  LTE Cat-1 / 3G  ┌──────────────────────┐
 │ OV2640 → wheel ROIs →         │ ───AT──▶ │ SIM7600 │ ── eNodeB ─ EPC ─▶│ MQTT broker (TLS)    │
 │ normalise → INT8 CNN (ESP-NN) │          └─────────┘                   │  verify CMAC, decrypt│
 │ → validate, night-flow leak   │      MQTT PUBLISH QoS 1, 50 B frame    │  replay + ledger     │
 │ → AES-CTR + CMAC → 20 B       │      (20 B payload)                    │  tariff, alert rules │
 └───────────────────────────────┘                                        └──────┬───────────────┘
                                                                                 │ SMS gateway
                                                                                 ▼
                                                                     Customer feature phone (GSM-7 SMS)
```

**Payload (20 B):** `hdr(1) · meter_id(4) · epoch(4) · reading dL(4)🔒 · night flow L/h(2)🔒 · battery(1)🔒 · CMAC-32(4)`. 🔒 = AES-128-CTR. The header, meter id and epoch are authenticated in clear, and the epoch doubles as the CTR nonce and the replay counter.

**Leak detection:** the device reads at 02:00 and 04:00. Continuous flow above 15 L/h while the household sleeps (minimum night flow) sets the leak flag and transmits immediately. The back end also alerts on consumption above 1.5 m³/day.

## Repository

```
src/web/                 Dashboard (static; no build, no framework, no third-party JS)
  index.html, css/app.css, js/app.js
  js/core/               int8-engine, digit-preprocess, image-ops, counter-locator, telemetry, mqtt
  js/sim/                cellular-link (session model), wasac-cloud (back end)
  model/digit-int8.js    Generated model (graph + int8 weights)
  samples/               Test images: Baylan meter (Wikimedia Commons, CC BY-SA 4.0) and an ARAD meter (web image, demo only)
src/model/               train.py (train, quantise, export), preprocess.py, artifacts/ (tflite, report.json)
src/firmware/            esp32_edge_ocr.cpp, digit_model.{h,cpp}, digit_model_int8.h (generated)
tools/test/              Node test suite (no dependencies)
deploy/                  Linode diagnostics, Nginx site, atomic deploy script
docs/DEPLOYMENT.md       Cloudflare + Nginx guide
```

## Run and test

**Dashboard:** open `src/web/index.html` in Chrome, Edge or Firefox (it works from `file://`), or serve it:

```bash
python -m http.server 8000 -d src/web      # then http://localhost:8000
```

1. Pick a meter account and click **Browse meter image**. Use `src/web/samples/` (`meter-c-baylan.jpg`, `Wmeter.png`) or your own close-up photo of the register.
2. The pipeline runs by itself: localisation (blue boxes are integer wheels, red boxes are fraction wheels), per-wheel INT8 classification, payload and MQTT frame, the cellular session with its AT transcript, the WASAC decision and the SMS.
3. If the boxes are wrong, click **Draw register region**, drag around the digit window, and optionally set **Wheels**.
4. Upload a second image for the same account to see consumption, billing and the high-use alert. **Resend same packet** demonstrates replay rejection.

**Tests** (Node ≥ 20):

```bash
cd tools && npm test       # 22 tests: INT8 bit-exactness, preprocessing parity, RFC 4493 CMAC, MQTT, back end, locator
```

**Reproduce the model** (Python 3.12, TensorFlow 2.18):

```bash
pip install "tensorflow==2.18.*" tflite==2.18.0 pillow numpy
# dataset: https://github.com/dbortnik/digitsDataset (unzip; folders 0-9)
python src/model/train.py --data <path>/new_dataset_staro      # ~10 min on CPU
```

This regenerates the TFLite model, the browser model, the firmware header and the test fixtures.

## Deploy

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Short version: point a proxied Cloudflare `A` record `amr-edge` at the Linode, set SSL mode to Full (strict) with an Origin certificate, install `deploy/nginx/amr-edge.zolilabs.com.conf`, then run `DEPLOY_HOST=user@LINODE_IP ./deploy/deploy.sh`.

## Limitations and next steps

- Auto-localisation works on close-ups. On wide shots, or where a printed serial number looks like a register, the operator draws the region. In the field this is a one-time installation step because the camera is fixed.
- Pointer dials (sub-litre resolution) are not read yet. The integer and red fraction wheels are.
- The firmware has not been compiled or flashed, and no on-device latency or power measurements have been taken.
- Next: a 90-day pilot on about 200 meters across Gasabo, Kicukiro and Nyarugenge, measuring time to leak detection, billing disputes and NRW against a control group.

Accounts, the tariff and the `*150#` short code are illustrative. Verify NRW figures against current WASAC/RURA publications before quoting them.
