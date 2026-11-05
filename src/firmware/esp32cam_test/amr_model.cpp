/**
 * Portable INT8 digit recogniser (plain C++, any ESP32 variant).
 *
 * Arithmetic is the TFLite reference integer ops (conv, depthwise conv,
 * average pool, fully connected) with gemmlowp requantisation, as in
 * src/web/js/core/int8-engine.js; the sketch's boot self-test checks the
 * output bit-exact against TFLite on the golden vectors. Preprocessing is
 * copied from src/firmware/digit_model.cpp; keep the two in lock-step.
 *
 * Speed on the original ESP32 (no int8 SIMD) comes from memory placement and
 * loop structure, not from changing the maths:
 *   - activations and weights live in internal SRAM, never in PSRAM;
 *   - 1x1 (pointwise) convolutions, 65% of the MACs, fold the input offset
 *     into the bias once at init: sum((x+o)*w) + b == sum(x*w) + (b + o*sum(w));
 *   - layer parameters are hoisted into locals and buffers are __restrict,
 *     so int8 stores do not force the compiler to reload them;
 *   - depthwise accumulates a whole pixel's channels at once (contiguous HWC).
 */
#pragma GCC optimize("O3")

#include "amr_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "esp_heap_caps.h"

#include "digit_model_int8.h"

static constexpr int OUT_W = DIGIT_IN_W;  // 24
static constexpr int OUT_H = DIGIT_IN_H;  // 32
static constexpr int MAX_ROWS = 256;      // tallest crop amr_normalize accepts
static constexpr int MAX_CHANNELS = 256;

struct Layer {
    const amr_layer_t *L;
    const int8_t *w;      // copy in internal SRAM
    const int32_t *bias;  // pointwise: bias + in_offset * sum(w), else the table's bias
    bool pointwise;
};

static Layer s_layers[DIGIT_NUM_LAYERS];
static int8_t *s_ping, *s_pong;  // activation buffers, DIGIT_MAX_TENSOR_BYTES each
static double *s_rows;           // preprocessing: horizontally resampled rows
static bool s_ready;

static void *fast_alloc(size_t n) {
    void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    return p ? p : malloc(n);
}

static size_t weight_count(const amr_layer_t &L) {
    switch (L.op) {
        case AMR_OP_CONV: return size_t(L.out[2]) * L.k_h * L.k_w * L.in[2];
        case AMR_OP_DWCONV: return size_t(L.k_h) * L.k_w * L.out[2];
        case AMR_OP_FC: return size_t(L.in[0]) * L.in[1] * L.in[2] * L.out[2];
        default: return 0;
    }
}

bool amr_init() {
    if (s_ready) return true;
    s_ping = (int8_t *)fast_alloc(DIGIT_MAX_TENSOR_BYTES);
    s_pong = (int8_t *)fast_alloc(DIGIT_MAX_TENSOR_BYTES);
    s_rows = (double *)fast_alloc(sizeof(double) * MAX_ROWS * OUT_W);
    if (!s_ping || !s_pong || !s_rows) return false;
    for (int i = 0; i < DIGIT_NUM_LAYERS; ++i) {
        const amr_layer_t &L = DIGIT_LAYERS[i];
        Layer &y = s_layers[i];
        y.L = &L;
        y.bias = L.bias;
        y.pointwise = L.op == AMR_OP_CONV && L.k_h == 1 && L.k_w == 1 && L.stride_h == 1 && L.stride_w == 1 &&
                      L.pad_t == 0 && L.pad_l == 0 && L.in[2] % 4 == 0;
        if (L.out[2] > MAX_CHANNELS) return false;
        const size_t n = weight_count(L);
        if (!n) continue;
        int8_t *w = (int8_t *)fast_alloc(n);
        if (!w) return false;
        memcpy(w, L.w, n);
        y.w = w;
        if (y.pointwise) {
            const int C = L.in[2], OC = L.out[2];
            int32_t *b = (int32_t *)fast_alloc(sizeof(int32_t) * OC);
            if (!b) return false;
            for (int oc = 0; oc < OC; ++oc) {
                int32_t s = 0;
                for (int ic = 0; ic < C; ++ic) s += w[oc * C + ic];
                b[oc] = L.bias[oc] + L.in_offset * s;
            }
            y.bias = b;
        }
    }
    s_ready = true;
    return true;
}

// ---------------------------------------------------------------------------
// Fixed-point helpers (tensorflow/lite/kernels/internal/common.h)
// ---------------------------------------------------------------------------
static inline int32_t sat_rounding_doubling_high_mul(int32_t a, int32_t b) {
    if (a == b && a == INT32_MIN) return INT32_MAX;
    const int64_t ab = int64_t(a) * int64_t(b);
    const int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return int32_t((ab + nudge) / (int64_t(1) << 31));
}

static inline int32_t rounding_divide_by_pot(int32_t x, int exponent) {
    const int32_t mask = int32_t((int64_t(1) << exponent) - 1);
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}

static inline int8_t requantize(int32_t acc, int32_t mult, int32_t shift, int32_t out_offset, int32_t lo, int32_t hi) {
    const int left = shift > 0 ? shift : 0, right = shift > 0 ? 0 : -shift;
    int32_t v = rounding_divide_by_pot(sat_rounding_doubling_high_mul(acc * (1 << left), mult), right) + out_offset;
    return int8_t(v < lo ? lo : v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// Kernels. Layouts follow TFLite: conv OHWI, depthwise 1HWC, FC [out][in]; activations HWC.
// ---------------------------------------------------------------------------
static void conv_pointwise(const Layer &y, const int8_t *__restrict in, int8_t *__restrict out) {
    const amr_layer_t &L = *y.L;
    const int P = L.in[0] * L.in[1], C = L.in[2], OC = L.out[2];
    const int32_t ooff = L.out_offset, lo = L.act_min, hi = L.act_max;
    const int8_t *__restrict w = y.w;
    const int32_t *__restrict bias = y.bias, *__restrict mult = L.mult, *__restrict shift = L.shift;
    for (int p = 0; p < P; ++p) {
        const int8_t *__restrict x = in + p * C;
        for (int oc = 0; oc < OC; ++oc) {
            const int8_t *__restrict wo = w + oc * C;
            int32_t acc = bias[oc];
            for (int ic = 0; ic < C; ic += 4)
                acc += x[ic] * wo[ic] + x[ic + 1] * wo[ic + 1] + x[ic + 2] * wo[ic + 2] + x[ic + 3] * wo[ic + 3];
            *out++ = requantize(acc, mult[oc], shift[oc], ooff, lo, hi);
        }
    }
}

static void conv_generic(const Layer &y, const int8_t *__restrict in, int8_t *__restrict out) {
    const amr_layer_t &L = *y.L;
    const int H = L.in[0], W = L.in[1], C = L.in[2], OH = L.out[0], OW = L.out[1], OC = L.out[2];
    const int KH = L.k_h, KW = L.k_w, SH = L.stride_h, SW = L.stride_w, PT = L.pad_t, PL = L.pad_l;
    const int32_t ioff = L.in_offset, ooff = L.out_offset, lo = L.act_min, hi = L.act_max;
    const int8_t *__restrict w = y.w;
    const int32_t *__restrict bias = y.bias, *__restrict mult = L.mult, *__restrict shift = L.shift;
    for (int oy = 0; oy < OH; ++oy) {
        const int iy0 = oy * SH - PT;
        const int ky0 = iy0 < 0 ? -iy0 : 0, ky1 = H - iy0 < KH ? H - iy0 : KH;
        for (int ox = 0; ox < OW; ++ox) {
            const int ix0 = ox * SW - PL;
            const int kx0 = ix0 < 0 ? -ix0 : 0, kx1 = W - ix0 < KW ? W - ix0 : KW;
            for (int oc = 0; oc < OC; ++oc) {
                const int8_t *__restrict wo = w + oc * KH * KW * C;
                int32_t acc = bias[oc];
                for (int ky = ky0; ky < ky1; ++ky) {
                    const int xb = ((iy0 + ky) * W + ix0) * C, wb = ky * KW * C;
                    for (int kx = kx0; kx < kx1; ++kx)
                        for (int ic = 0; ic < C; ++ic) acc += (in[xb + kx * C + ic] + ioff) * wo[wb + kx * C + ic];
                }
                *out++ = requantize(acc, mult[oc], shift[oc], ooff, lo, hi);
            }
        }
    }
}

static void dwconv(const Layer &y, const int8_t *__restrict in, int8_t *__restrict out) {
    const amr_layer_t &L = *y.L;
    const int H = L.in[0], W = L.in[1], C = L.in[2], OH = L.out[0], OW = L.out[1], OC = L.out[2];
    const int M = OC / C;
    const int KH = L.k_h, KW = L.k_w, SH = L.stride_h, SW = L.stride_w, PT = L.pad_t, PL = L.pad_l;
    const int32_t ioff = L.in_offset, ooff = L.out_offset, lo = L.act_min, hi = L.act_max;
    const int8_t *__restrict w = y.w;
    const int32_t *__restrict bias = y.bias, *__restrict mult = L.mult, *__restrict shift = L.shift;
    int32_t acc[MAX_CHANNELS];
    for (int oy = 0; oy < OH; ++oy) {
        const int iy0 = oy * SH - PT;
        const int ky0 = iy0 < 0 ? -iy0 : 0, ky1 = H - iy0 < KH ? H - iy0 : KH;
        for (int ox = 0; ox < OW; ++ox) {
            const int ix0 = ox * SW - PL;
            const int kx0 = ix0 < 0 ? -ix0 : 0, kx1 = W - ix0 < KW ? W - ix0 : KW;
            for (int oc = 0; oc < OC; ++oc) acc[oc] = bias[oc];
            for (int ky = ky0; ky < ky1; ++ky)
                for (int kx = kx0; kx < kx1; ++kx) {
                    const int8_t *__restrict x = in + ((iy0 + ky) * W + ix0 + kx) * C;
                    const int8_t *__restrict wk = w + (ky * KW + kx) * OC;
                    if (M == 1) {
                        for (int c = 0; c < C; ++c) acc[c] += (x[c] + ioff) * wk[c];
                    } else {
                        for (int c = 0; c < C; ++c)
                            for (int m = 0; m < M; ++m) acc[c * M + m] += (x[c] + ioff) * wk[c * M + m];
                    }
                }
            for (int oc = 0; oc < OC; ++oc) *out++ = requantize(acc[oc], mult[oc], shift[oc], ooff, lo, hi);
        }
    }
}

static void avgpool(const amr_layer_t &L, const int8_t *__restrict in, int8_t *__restrict out) {
    const int H = L.in[0], W = L.in[1], C = L.in[2], OH = L.out[0], OW = L.out[1];
    for (int oy = 0; oy < OH; ++oy)
        for (int ox = 0; ox < OW; ++ox) {
            const int ys = oy * L.stride_h - L.pad_t, xs = ox * L.stride_w - L.pad_l;
            const int y0 = ys > 0 ? ys : 0, y1 = ys + L.k_h < H ? ys + L.k_h : H;
            const int x0 = xs > 0 ? xs : 0, x1 = xs + L.k_w < W ? xs + L.k_w : W;
            const int count = (y1 - y0) * (x1 - x0), half = count / 2;
            for (int c = 0; c < C; ++c) {
                int32_t acc = 0;
                for (int y = y0; y < y1; ++y)
                    for (int x = x0; x < x1; ++x) acc += in[(y * W + x) * C + c];
                acc = acc > 0 ? (acc + half) / count : (acc - half) / count;
                out[(oy * OW + ox) * C + c] = int8_t(acc < L.act_min ? L.act_min : acc > L.act_max ? L.act_max : acc);
            }
        }
}

static void fully_connected(const Layer &y, const int8_t *__restrict in, int8_t *__restrict out) {
    const amr_layer_t &L = *y.L;
    const int N = L.in[0] * L.in[1] * L.in[2], OUT = L.out[2];
    for (int o = 0; o < OUT; ++o) {
        int32_t acc = y.bias[o];
        const int8_t *__restrict w = y.w + o * N;
        for (int i = 0; i < N; ++i) acc += (in[i] + L.in_offset) * w[i];
        const int q = L.n_quant > 1 ? o : 0;
        out[o] = requantize(acc, L.mult[q], L.shift[q], L.out_offset, L.act_min, L.act_max);
    }
}

void amr_invoke(const int8_t *input, int8_t *logits) {
    if (!amr_init()) {
        memset(logits, -128, DIGIT_NUM_CLASSES);
        return;
    }
    memcpy(s_ping, input, OUT_W * OUT_H);
    int8_t *in = s_ping, *out = s_pong;
    for (const Layer &y : s_layers) {
        switch (y.L->op) {
            case AMR_OP_CONV: y.pointwise ? conv_pointwise(y, in, out) : conv_generic(y, in, out); break;
            case AMR_OP_DWCONV: dwconv(y, in, out); break;
            case AMR_OP_AVGPOOL: avgpool(*y.L, in, out); break;
            case AMR_OP_FC: fully_connected(y, in, out); break;
        }
        int8_t *t = in;
        in = out;
        out = t;
    }
    memcpy(logits, in, DIGIT_NUM_CLASSES);
}

uint32_t amr_macs() {
    uint32_t macs = 0;
    for (const amr_layer_t &L : DIGIT_LAYERS) {
        const uint32_t outs = uint32_t(L.out[0]) * L.out[1] * L.out[2];
        if (L.op == AMR_OP_CONV) macs += outs * L.k_h * L.k_w * L.in[2];
        else if (L.op == AMR_OP_DWCONV) macs += outs * L.k_h * L.k_w;
        else if (L.op == AMR_OP_FC) macs += uint32_t(L.in[0]) * L.in[1] * L.in[2] * L.out[2];
    }
    return macs;
}

DigitResult amr_decode(const int8_t *logits) {
    float z[DIGIT_NUM_CLASSES], zmax = -1e30f, sum = 0;
    for (int i = 0; i < DIGIT_NUM_CLASSES; ++i) {
        z[i] = (logits[i] - DIGIT_OUT_ZERO_POINT) * DIGIT_OUT_SCALE;
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

// ---------------------------------------------------------------------------
// Preprocessing (same arithmetic as src/firmware/digit_model.cpp normalize()).
// Double precision keeps it bit-identical with the Python training transform;
// the ESP32 has no double FPU, so this is software floating point.
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

int amr_otsu(const uint8_t *v, int n) {
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

void amr_normalize(const uint8_t *crop, int w, int h, int8_t *out) {
    static Taps tx[OUT_W], ty[OUT_H];
    if (!amr_init()) {
        memset(out, DIGIT_IN_ZERO_POINT, OUT_W * OUT_H);
        return;
    }
    double *tmp = s_rows;
    if (h > MAX_ROWS) h = MAX_ROWS;
    area_taps(w, OUT_W, tx);
    area_taps(h, OUT_H, ty);
    for (int y = 0; y < h; ++y) {
        const uint8_t *row = crop + y * w;
        for (int i = 0; i < OUT_W; ++i) {
            const Taps &t = tx[i];
            double acc = 0;
            for (int k = 0; k < t.n; ++k) acc += row[t.j0 + k] * t.w[k];
            tmp[y * OUT_W + i] = acc;
        }
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
    const int t = amr_otsu(centre, n);
    int above = 0;
    for (int i = 0; i < n; ++i) above += centre[i] > t;
    if (above * 2 > n)
        for (uint8_t &v : px) v = 255 - v;

    // 2nd..98th percentile stretch, through a 256-entry table (same formula per grey level).
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
    int8_t lut[256];
    for (int v = 0; v < 256; ++v) {
        const double s = (double(v) - lo) * 255.0 / (hi - lo);
        const int q = int(fmin(255.0, fmax(0.0, floor(s + 0.5))));
        lut[v] = int8_t(q + DIGIT_IN_ZERO_POINT);  // input scale 1/255, zero point -128
    }
    for (size_t i = 0; i < sizeof px; ++i) out[i] = lut[px[i]];
}
