# AMR-Edge Rwanda

**Keeping families connected to safe water. A tiny AI model on the water meter catches hidden leaks within a day, before they turn into unpayable bills, disconnections and waterborne disease.**

Hack-Nation 7th Global AI Hackathon · *Small AI for Development* (World Bank Group) · **Health track** · UN SDG 3 and SDG 6

| | |
|---|---|
| Live demo | https://amr-edge.zolilabs.com (mirror: https://amr-edge-rwanda.vercel.app) |
| Source code | https://github.com/iradukunda3546nj/amr-edge-rwanda |

---

## Health starts at the tap

Safe, continuous piped water is one of the most basic determinants of health. Handwashing, safe food preparation, hygiene during childbirth and infection control in clinics all depend on a tap that keeps running. When households lose piped water, they fall back on unprotected sources, and diarrhoeal disease is among the leading causes of death in children under five; WHO attributes most of it to unsafe water and poor hygiene.

In Rwanda, WASAC delivers that water. A gap in how it is metered quietly pushes households off it.

## The problem: one hidden leak, five steps to disease

| Step | What happens |
|---|---|
| 1. Hidden leak | A pipe breaks behind the customer's meter. Nothing is visible, and water runs day and night. |
| 2. Huge bill | Meters are read by hand every 30 to 90 days, so the leak surfaces weeks later as one unaffordable bill. |
| 3. Disconnection | The household cannot pay and the tap is closed. Children, most often girls, start fetching water. |
| 4. Unsafe water | Swamps, open wells and informal vendors replace treated piped water. |
| 5. Disease | Diarrhoea, typhoid and cholera, carried hardest by children under five and the poorest households. |

**Root cause: there is no automated, real-time metering.** The same gap hurts the system as a whole:

- **Non-revenue water.** About 38–40% of treated water is lost to leaks or never billed (WASAC).
- **Delayed revenue.** Billing waits for the next manual visit, 30 to 90 days later.
- **Slower expansion.** Revenue that is lost or late is money not spent extending safe water to unserved homes, schools and health facilities.

## The solution: break the chain at step one

AMR-Edge is a battery-powered clip-on reader for the mechanical meters already installed, so no meter replacement or plumbing is needed.

1. A camera captures the meter face, and a **20 KB INT8 neural network reads the register on the device**.
2. The reading travels as a **20-byte encrypted, signed packet** over LTE/3G to WASAC. Images never leave the home.
3. Readings at 02:00 and 04:00 detect **continuous night flow**, the standard signature of a leak, and alert immediately.
4. WASAC validates and bills the reading, and the customer gets a **plain SMS on any basic phone**.

| Outcome | Today | With AMR-Edge |
|---|---|---|
| Leak detected after | 30–90 days | under 24 hours |
| Water lost to a 1.2 m³/h leak | ~2,600 m³ over a 90-day cycle | ~29 m³ in one day |
| Customer finds out from | an unpayable bill | an SMS the same day |
| Household outcome | debt, disconnection, unsafe water | leak repaired, tap stays on |
| Utility revenue | late, partly lost | daily, verified |

**Health impact pathway:** fewer bill shocks lead to fewer disconnections, which keeps households on treated water, which reduces exposure to waterborne disease. Recovered revenue extends safe water to more households, schools and clinics.

Figures in the table are illustrative and will be validated in a pilot.

## Why Small AI, and why on the device

- **Where meters are.** Meters sit in pits and compounds, often with weak 2G/3G coverage and no power outlet. A cloud vision API needs a ~100 KB photo upload, good connectivity and mains power.
- **Cost and power.** INT8 integer arithmetic runs on a low-cost ESP32-S3 microcontroller on a battery, and the model fits in its memory.
- **Privacy.** The image is processed inside the device. Only the reading leaves the home.

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
| Output | 11 class probabilities (digits 0–9, reject) |
| Architecture | Conv3×3(16) → 3 depthwise-separable blocks (32, 48, 64) → AvgPool → FC(11); BN folded at conversion |
| Quantisation | Per-channel symmetric int8 weights, per-tensor asymmetric int8 activations, int32 bias, fixed-point requantisation |
| Data | 22,286 real meter-wheel crops (Nikić et al., *Future Internet* 16(11):402, 2024, GPL-3.0) with augmentation, plus synthetic register crops in 15 typefaces |
| Validation | FP32 100.0% / INT8 100.0% on 3,183 held-out real crops (split by blocks of consecutive video frames to avoid leakage); 98.3% INT8 on held-out synthetic crops in other fonts |
| Runtime on device | ESP-NN kernels called directly from a generated layer table. TFLM's interpreter would cost more flash than the model itself |

The real dataset comes from one meter type, so 100% validation measures fit, not generalisation. The synthetic held-out set and real meter photographs are the out-of-domain checks.

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

**Payload (20 B):** `hdr(1) · meter_id(4) · epoch(4) · reading dL(4) · night flow L/h(2) · battery(1) · CMAC-32(4)`. The reading, night flow and battery fields are encrypted with AES-128-CTR. The header, meter id and epoch are authenticated in clear, and the epoch doubles as the CTR nonce and the replay counter.

**Leak detection:** the device reads at 02:00 and 04:00. Continuous flow above 15 L/h while the household sleeps (minimum night flow) sets the leak flag and transmits immediately. The back end also alerts on consumption above 1.5 m³/day.

## Repository

```
src/web/                 Dashboard (static; no build, no framework, no third-party JS)
  index.html, css/app.css, js/app.js
  js/core/               int8-engine, digit-preprocess, image-ops, counter-locator, telemetry, mqtt
  js/sim/                cellular-link (session model), wasac-cloud (back end)
  model/digit-int8.js    Generated model (graph + int8 weights)
  samples/               Test image: Baylan meter (Wikimedia Commons, CC BY-SA 4.0)
src/model/               train.py (train, quantise, export), preprocess.py, artifacts/ (tflite, report.json)
src/firmware/            esp32_edge_ocr.cpp, digit_model.{h,cpp}, digit_model_int8.h (generated)
tools/test/              Node test suite (no dependencies)
deploy/                  Linode diagnostics, Nginx site, atomic deploy script
docs/DEPLOYMENT.md       Cloudflare + Nginx guide
vercel.json              Static hosting config (serves src/web with security headers)
```

## Run and test

**Dashboard:** use the live demo, or open `src/web/index.html` in Chrome, Edge or Firefox (it works from `file://`), or serve it:

```bash
python -m http.server 8000 -d src/web      # then http://localhost:8000
```

1. Pick a meter account and click **Browse meter image**. Use `src/web/samples/meter-c-baylan.jpg` or your own close-up photo of the register.
2. The pipeline runs by itself: localisation (blue boxes are integer wheels, red boxes are fraction wheels), per-wheel INT8 classification, payload and MQTT frame, the cellular session with its AT transcript, the WASAC decision and the SMS.
3. If the boxes are wrong, click **Draw register region**, drag around the digit window, and optionally set **Wheels**.
4. Upload a second image for the same account to see consumption, billing and the high-use alert. **Resend same packet** demonstrates replay rejection.

**Tests** (Node 20 or later):

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

The live site is hosted on Vercel: every push to `main` redeploys `src/web` as configured in `vercel.json`, and `amr-edge.zolilabs.com` points to it through a DNS-only Cloudflare CNAME. A self-hosted alternative (Linode, Nginx, Cloudflare Full strict) is documented in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Partnership and next steps

We are looking to partner with **WASAC Group** to take AMR-Edge from prototype to field:

1. **Retrain on real meters.** Collect images of the meter types installed in Rwanda and retrain the model.
2. **Run a supervised pilot.** About 200 meters across Gasabo, Kicukiro and Nyarugenge for 90 days, measuring time to leak detection, disconnections avoided, billing disputes and non-revenue water against a control group.
3. **Integrate.** Connect to WASAC billing and an approved SMS short code, with priority alerts for schools and health facilities.

## Limitations

- Auto-localisation works on close-ups. On wide shots, or where a printed serial number looks like a register, the operator draws the region. In the field this is a one-time installation step because the camera is fixed.
- Pointer dials (sub-litre resolution) are not read yet. The integer and red fraction wheels are.
- The firmware has not been compiled or flashed, and no on-device latency or power measurements have been taken.
- Accounts, the tariff and the `*150#` short code are illustrative. Verify NRW figures against current WASAC/RURA publications before quoting them.
