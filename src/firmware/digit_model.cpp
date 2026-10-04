/**
 * INT8 digit recogniser on ESP-NN (no TFLite Micro interpreter).
 *
 * The 11-class model is ~8 KB of weights. TFLM's interpreter, flatbuffer
 * parser and op registry would cost several times that in flash, so the
 * generated layer table (digit_model_int8.h) is executed directly with
 * Espressif's ESP-NN kernels, which are bit-exact with the TFLite reference
 * integer kernels and use the ESP32-S3 SIMD (PIE) instructions. The browser
 * engine (src/web/js/core/int8-engine.js) implements the same kernels and is
 * tested bit-exact against TFLite in tools/test.
 *
 * Preprocessing uses double precision to stay bit-identical with training;
 * on the S3 that is software floating point, a few ms per wheel.
 */
#include "digit_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_nn.h"

#include "digit_model_int8.h"

static const char *TAG = "digit";

static constexpr int OUT_W = DIGIT_IN_W;  // 24
static constexpr int OUT_H = DIGIT_IN_H;  // 32

static int8_t *s_ping, *s_pong;  // activation buffers, DIGIT_MAX_TENSOR_BYTES each
static void *s_scratch;          // shared conv / depthwise scratch

// ---------------------------------------------------------------------------
// Preprocessing (keep in lock-step with preprocess.py / digit-preprocess.js)
// ---------------------------------------------------------------------------
struct Taps {
    int j0;
    int n;
    double w[16];
};

static void area_taps(int src, int dst, Taps *taps) {
    const double scale = double(src) / dst;
    for (int i = 0; i < dst; ++i) {
        const double lo = i * scale, hi = (i + 1) * scale;
        const int j0 = int(floor(lo)), j1 = int(fmin(ceil(hi), double(src)));
        Taps &t = taps[i];
        t.j0 = j0;
        t.n = j1 - j0 < 16 ? j1 - j0 : 16;  // crops wider than 16x the model input are clipped
        double total = 0;
        for (int k = 0; k < t.n; ++k) {
            t.w[k] = fmin(hi, double(j0 + k + 1)) - fmax(lo, double(j0 + k));
            total += t.w[k];
        }
        for (int k = 0; k < t.n; ++k) t.w[k] /= total;
    }
}

static int otsu(const uint8_t *v, int n) {
    double hist[256] = {0};
    for (int i = 0; i < n; ++i) hist[v[i]] += 1;
    double sum_all = 0;
    for (int t = 0; t < 256; ++t) sum_all += t * hist[t];
    double w_b = 0, sum_b = 0, best_var = -1;
    int best_t = 0;
    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (w_b == 0) continue;
        const double w_f = n - w_b;
        if (w_f == 0) break;
        sum_b += t * hist[t];
        const double m_b = sum_b / w_b, m_f = (sum_all - sum_b) / w_f;
        const double var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if (var > best_var) { best_var = var; best_t = t; }
    }
    return best_t;
}

/** crop (w x h) -> OUT_W x OUT_H bright-on-dark, contrast-stretched, as int8 model input. */
static void normalize(const uint8_t *crop, int w, int h, int8_t *out) {
    static Taps tx[OUT_W], ty[OUT_H];
    static double tmp[256 * OUT_W];  // supports crops up to 256 rows
    if (h > 256) h = 256;
    area_taps(w, OUT_W, tx);
    area_taps(h, OUT_H, ty);
    for (int y = 0; y < h; ++y)
        for (int i = 0; i < OUT_W; ++i) {
            double acc = 0;
            for (int k = 0; k < tx[i].n; ++k) acc += crop[y * w + tx[i].j0 + k] * tx[i].w[k];
            tmp[y * OUT_W + i] = acc;
        }
    uint8_t px[OUT_W * OUT_H];
    for (int i = 0; i < OUT_H; ++i)
        for (int x = 0; x < OUT_W; ++x) {
            double acc = 0;
            for (int k = 0; k < ty[i].n; ++k) acc += tmp[(ty[i].j0 + k) * OUT_W + x] * ty[i].w[k];
            px[i * OUT_W + x] = uint8_t(fmin(255.0, fmax(0.0, floor(acc + 0.5))));
        }

    // Polarity from the central window: the model expects a bright digit on a dark drum.
    uint8_t centre[OUT_W * OUT_H];
    int n = 0;
    for (int y = int(OUT_H * 0.15); y < int(OUT_H * 0.85); ++y)
        for (int x = int(OUT_W * 0.2); x < int(OUT_W * 0.8); ++x) centre[n++] = px[y * OUT_W + x];
    const int t = otsu(centre, n);
    int above = 0;
    for (int i = 0; i < n; ++i) above += centre[i] > t;
    if (above * 2 > n)
        for (uint8_t &v : px) v = 255 - v;

    // 2nd..98th percentile stretch.
    uint32_t hist[256] = {0};
    for (uint8_t v : px) hist[v]++;
    const int k_lo = int(ceil(0.02 * sizeof px)), k_hi = int(ceil(0.98 * sizeof px));
    int lo = -1, hi = -1;
    uint32_t cum = 0;
    for (int v = 0; v < 256; ++v) {
        cum += hist[v];
        if (lo < 0 && int(cum) >= k_lo) lo = v;
        if (hi < 0 && int(cum) >= k_hi) { hi = v; break; }
    }
    if (hi <= lo) hi = lo + 1;
    for (size_t i = 0; i < sizeof px; ++i) {
        const double s = (double(px[i]) - lo) * 255.0 / (hi - lo);
        const int q = int(fmin(255.0, fmax(0.0, floor(s + 0.5))));
        out[i] = int8_t(q + DIGIT_IN_ZERO_POINT);  // input scale 1/255, zero point -128
    }
}

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------
static void run_layer(const amr_layer_t &L, const int8_t *in, int8_t *out) {
    data_dims_t in_dims = {L.in[1], L.in[0], L.in[2], 1};
    data_dims_t out_dims = {L.out[1], L.out[0], L.out[2], 1};
    data_dims_t filter_dims = {L.k_w, L.k_h, 0, 0};
    quant_data_t q = {const_cast<int32_t *>(L.shift), const_cast<int32_t *>(L.mult)};

    switch (L.op) {
        case AMR_OP_CONV: {
            conv_params_t p = {};
            p.in_offset = L.in_offset;
            p.out_offset = L.out_offset;
            p.stride = {L.stride_w, L.stride_h};
            p.padding = {L.pad_l, L.pad_t};
            p.dilation = {1, 1};
            p.activation = {L.act_min, L.act_max};
            esp_nn_conv_s8(&in_dims, in, &filter_dims, L.w, L.bias, &out_dims, out, &p, &q);
            break;
        }
        case AMR_OP_DWCONV: {
            dw_conv_params_t p = {};
            p.in_offset = L.in_offset;
            p.out_offset = L.out_offset;
            p.ch_mult = L.out[2] / L.in[2];
            p.stride = {L.stride_w, L.stride_h};
            p.padding = {L.pad_l, L.pad_t};
            p.dilation = {1, 1};
            p.activation = {L.act_min, L.act_max};
            esp_nn_depthwise_conv_s8(&in_dims, in, &filter_dims, L.w, L.bias, &out_dims, out, &p, &q);
            break;
        }
        case AMR_OP_AVGPOOL:
            esp_nn_avg_pool_s8(in, L.in[1], L.in[0], out, L.out[1], L.out[0], L.stride_w, L.stride_h,
                               L.k_w, L.k_h, L.pad_l, L.pad_t, L.act_min, L.act_max, L.in[2]);
            break;
        case AMR_OP_FC: {
            // ESP-NN's FC kernel takes one multiplier; call it per output channel
            // when the converter quantised the dense layer per channel.
            const int n_in = L.in[2], n_out = L.out[2];
            if (L.n_quant == 1) {
                esp_nn_fully_connected_s8(in, L.in_offset, n_in, L.w, 0, L.bias, out, n_out, L.out_offset,
                                          L.shift[0], L.mult[0], L.act_min, L.act_max);
            } else {
                for (int o = 0; o < n_out; ++o)
                    esp_nn_fully_connected_s8(in, L.in_offset, n_in, L.w + o * n_in, 0, L.bias + o, out + o, 1,
                                              L.out_offset, L.shift[o], L.mult[o], L.act_min, L.act_max);
            }
            break;
        }
    }
}

bool digit_model_init() {
    size_t scratch = 0;
    for (const amr_layer_t &L : DIGIT_LAYERS) {
        data_dims_t in_dims = {L.in[1], L.in[0], L.in[2], 1};
        data_dims_t out_dims = {L.out[1], L.out[0], L.out[2], 1};
        data_dims_t filter_dims = {L.k_w, L.k_h, 0, 0};
        if (L.op == AMR_OP_CONV) {
            conv_params_t p = {L.in_offset, L.out_offset, {L.stride_w, L.stride_h}, {L.pad_l, L.pad_t}, {1, 1}, {L.act_min, L.act_max}};
            scratch = std::max<size_t>(scratch, esp_nn_get_conv_scratch_size(&in_dims, &filter_dims, &out_dims, &p));
        } else if (L.op == AMR_OP_DWCONV) {
            dw_conv_params_t p = {L.in_offset, L.out_offset, L.out[2] / L.in[2], {L.stride_w, L.stride_h}, {L.pad_l, L.pad_t}, {1, 1}, {L.act_min, L.act_max}};
            scratch = std::max<size_t>(scratch, esp_nn_get_depthwise_conv_scratch_size(&in_dims, &filter_dims, &out_dims, &p));
        }
    }
    s_ping = static_cast<int8_t *>(heap_caps_aligned_alloc(16, DIGIT_MAX_TENSOR_BYTES, MALLOC_CAP_INTERNAL));
    s_pong = static_cast<int8_t *>(heap_caps_aligned_alloc(16, DIGIT_MAX_TENSOR_BYTES, MALLOC_CAP_INTERNAL));
    s_scratch = scratch ? heap_caps_aligned_alloc(16, scratch, MALLOC_CAP_INTERNAL) : nullptr;
    if (!s_ping || !s_pong || (scratch && !s_scratch)) return false;
    esp_nn_set_conv_scratch_buf(s_scratch);
    esp_nn_set_depthwise_conv_scratch_buf(s_scratch);
    ESP_LOGI(TAG, "ready: %d layers, 2 x %d B activations, %u B scratch", DIGIT_NUM_LAYERS, DIGIT_MAX_TENSOR_BYTES, unsigned(scratch));
    return true;
}

DigitResult digit_model_classify(const uint8_t *crop, int w, int h) {
    normalize(crop, w, h, s_ping);
    int8_t *in = s_ping, *out = s_pong;
    for (const amr_layer_t &L : DIGIT_LAYERS) {
        run_layer(L, in, out);
        std::swap(in, out);
    }
    // `in` now holds the int8 logits; softmax on dequantised values.
    float z[DIGIT_NUM_CLASSES], zmax = -1e30f, sum = 0;
    for (int i = 0; i < DIGIT_NUM_CLASSES; ++i) {
        z[i] = (in[i] - DIGIT_OUT_ZERO_POINT) * DIGIT_OUT_SCALE;
        zmax = fmaxf(zmax, z[i]);
    }
    for (int i = 0; i < DIGIT_NUM_CLASSES; ++i) sum += (z[i] = expf(z[i] - zmax));
    DigitResult r = {0, 0, 1, 0};
    for (int i = 0; i < DIGIT_NUM_CLASSES; ++i) {
        const float p = z[i] / sum;
        if (p > r.confidence) {
            r.runner_up = r.cls;
            r.runner_up_confidence = r.confidence;
            r.cls = i;
            r.confidence = p;
        } else if (p > r.runner_up_confidence) {
            r.runner_up = i;
            r.runner_up_confidence = p;
        }
    }
    return r;
}
