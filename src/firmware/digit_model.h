// On-device digit recogniser: crop normalisation + INT8 inference on ESP-NN.
#pragma once
#include <stdint.h>

struct DigitResult {
    int cls;            // 0-9, or DIGIT_REJECT_CLASS for a non-digit glyph
    float confidence;   // softmax probability of `cls`
    int runner_up;
    float runner_up_confidence;
};

/** One-time setup (scratch buffers for the ESP-NN kernels). Returns false on allocation failure. */
bool digit_model_init();

/**
 * Normalise a grayscale wheel crop (any size) and classify it.
 * Mirrors src/model/preprocess.py and src/web/js/core/digit-preprocess.js.
 */
DigitResult digit_model_classify(const uint8_t *crop, int w, int h);
