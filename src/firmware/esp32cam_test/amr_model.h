// Portable INT8 digit recogniser for the ESP32-CAM test sketch.
// Same arithmetic as src/web/js/core/int8-engine.js and the TFLite reference
// integer kernels; no ESP-NN or TFLite Micro dependency.
#pragma once
#include <stdint.h>

struct DigitResult {
    int cls;            // 0-9, or DIGIT_REJECT_CLASS for a non-digit glyph
    float confidence;   // softmax probability of `cls`
    int runner_up;
    float runner_up_confidence;
};

/**
 * Allocate buffers (~74 KB) and copy weights into internal SRAM. Call early in
 * setup(), before Wi-Fi takes its share of internal memory. Idempotent; the
 * functions below call it themselves if needed.
 */
bool amr_init();

/** Grayscale crop (w x h, h <= 256) -> 24x32 int8 model input. Mirrors src/model/preprocess.py. */
void amr_normalize(const uint8_t *crop, int w, int h, int8_t *out);

/** Run the INT8 graph on one 24x32 input; writes DIGIT_NUM_CLASSES int8 logits. */
void amr_invoke(const int8_t *input, int8_t *logits);

/** Softmax over dequantised logits: top class and runner-up. */
DigitResult amr_decode(const int8_t *logits);

/** Multiply-accumulates per inference, from the layer table. */
uint32_t amr_macs();

/** Otsu threshold of n grey levels (same routine preprocessing uses). */
int amr_otsu(const uint8_t *v, int n);
