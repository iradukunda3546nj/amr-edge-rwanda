/*
 * INT8 inference engine: a JavaScript port of the TensorFlow Lite reference
 * integer kernels (tensorflow/lite/kernels/internal/reference/integer_ops).
 *
 * Executes the graph exported by src/model/train.py with the same fixed-point
 * arithmetic as TFLite Micro / ESP-NN on the device: int8 activations, int8
 * per-channel weights, int32 accumulators and gemmlowp-style requantisation.
 * tools/test/int8-engine.test.mjs asserts bit-exact agreement with the TFLite
 * interpreter (BUILTIN_REF kernels) on golden vectors.
 *
 * Tensors are HWC (batch of 1), row-major, Int8Array.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).int8 = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  const INT32_MIN = -2147483648;
  const INT32_MAX = 2147483647;

  /**
   * gemmlowp SaturatingRoundingDoublingHighMul: (a * b + nudge) / 2^31,
   * truncated toward zero. The 62-bit product is evaluated exactly in float64
   * by splitting |a| into 16-bit halves.
   */
  function saturatingRoundingDoublingHighMul(a, b) {
    if (a === INT32_MIN && b === INT32_MIN) return INT32_MAX;
    const negative = (a < 0) !== (b < 0) && a !== 0 && b !== 0;
    const A = Math.abs(a);
    const B = Math.abs(b);
    const aHi = Math.floor(A / 65536);
    const aLo = A - aHi * 65536;
    const nudge = negative ? 1073741823 : 1073741824; // 2^30 - 1 : 2^30
    const low = aLo * B + nudge;
    const total = aHi * B + Math.floor(low / 65536);
    const magnitude = Math.floor(total / 32768);
    return negative ? -magnitude | 0 : magnitude | 0;
  }

  /** gemmlowp RoundingDivideByPOT: x / 2^exponent, rounding half away from zero. */
  function roundingDivideByPOT(x, exponent) {
    if (exponent === 0) return x;
    const mask = exponent >= 31 ? 0x7fffffff : (1 << exponent) - 1;
    const remainder = x & mask;
    const threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
  }

  /** tflite::MultiplyByQuantizedMultiplier (double-rounding variant, the TFLite default). */
  function multiplyByQuantizedMultiplier(x, multiplier, shift) {
    const left = shift > 0 ? shift : 0;
    const right = shift > 0 ? 0 : -shift;
    return roundingDivideByPOT(saturatingRoundingDoublingHighMul(x << left, multiplier), right);
  }

  /** tflite::ComputePaddingWithOffset: leading padding for SAME, 0 for VALID. */
  function leadingPad(padding, inSize, filterSize, stride, outSize) {
    if (padding !== 'SAME') return 0;
    const total = Math.max((outSize - 1) * stride + filterSize - inSize, 0);
    return Math.floor(total / 2);
  }

  const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

  function conv2d(op, input) {
    const [H, W, C] = op.in_shape;
    const [OH, OW, OC] = op.out_shape;
    const [, KH, KW, IC] = op.filter_shape; // OHWI
    const [sh, sw] = op.stride;
    const pt = leadingPad(op.padding, H, KH, sh, OH);
    const pl = leadingPad(op.padding, W, KW, sw, OW);
    const { weights: w, bias, mult, shift, in_offset: inOff, out_offset: outOff } = op;
    const [amin, amax] = op.act;
    const out = new Int8Array(OH * OW * OC);
    let o = 0;
    for (let oy = 0; oy < OH; oy++) {
      const iy0 = oy * sh - pt;
      for (let ox = 0; ox < OW; ox++) {
        const ix0 = ox * sw - pl;
        for (let oc = 0; oc < OC; oc++) {
          let acc = 0;
          for (let ky = 0; ky < KH; ky++) {
            const iy = iy0 + ky;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < KW; kx++) {
              const ix = ix0 + kx;
              if (ix < 0 || ix >= W) continue;
              const ib = (iy * W + ix) * C;
              const wb = ((oc * KH + ky) * KW + kx) * IC;
              for (let ic = 0; ic < IC; ic++) acc += (input[ib + ic] + inOff) * w[wb + ic];
            }
          }
          acc = multiplyByQuantizedMultiplier(acc + bias[oc], mult[oc], shift[oc]) + outOff;
          out[o++] = clamp(acc, amin, amax);
        }
      }
    }
    return out;
  }

  function depthwiseConv2d(op, input) {
    const [H, W, C] = op.in_shape;
    const [OH, OW, OC] = op.out_shape;
    const [, KH, KW] = op.filter_shape; // 1 x KH x KW x OC
    const M = op.depth_multiplier || OC / C;
    const [sh, sw] = op.stride;
    const pt = leadingPad(op.padding, H, KH, sh, OH);
    const pl = leadingPad(op.padding, W, KW, sw, OW);
    const { weights: w, bias, mult, shift, in_offset: inOff, out_offset: outOff } = op;
    const [amin, amax] = op.act;
    const out = new Int8Array(OH * OW * OC);
    for (let oy = 0; oy < OH; oy++) {
      const iy0 = oy * sh - pt;
      for (let ox = 0; ox < OW; ox++) {
        const ix0 = ox * sw - pl;
        const ob = (oy * OW + ox) * OC;
        for (let ic = 0; ic < C; ic++) {
          for (let m = 0; m < M; m++) {
            const oc = ic * M + m;
            let acc = 0;
            for (let ky = 0; ky < KH; ky++) {
              const iy = iy0 + ky;
              if (iy < 0 || iy >= H) continue;
              for (let kx = 0; kx < KW; kx++) {
                const ix = ix0 + kx;
                if (ix < 0 || ix >= W) continue;
                acc += (input[(iy * W + ix) * C + ic] + inOff) * w[(ky * KW + kx) * OC + oc];
              }
            }
            acc = multiplyByQuantizedMultiplier(acc + bias[oc], mult[oc], shift[oc]) + outOff;
            out[ob + oc] = clamp(acc, amin, amax);
          }
        }
      }
    }
    return out;
  }

  function averagePool2d(op, input) {
    const [H, W, C] = op.in_shape;
    const [OH, OW] = op.out_shape;
    const [FH, FW] = op.filter;
    const [sh, sw] = op.stride;
    const pt = leadingPad(op.padding || 'VALID', H, FH, sh, OH);
    const pl = leadingPad(op.padding || 'VALID', W, FW, sw, OW);
    const [amin, amax] = op.act;
    const out = new Int8Array(OH * OW * C);
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        const y0 = Math.max(oy * sh - pt, 0), y1 = Math.min(oy * sh - pt + FH, H);
        const x0 = Math.max(ox * sw - pl, 0), x1 = Math.min(ox * sw - pl + FW, W);
        const count = (y1 - y0) * (x1 - x0);
        const half = Math.floor(count / 2);
        for (let c = 0; c < C; c++) {
          let acc = 0;
          for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) acc += input[(y * W + x) * C + c];
          acc = acc > 0 ? Math.trunc((acc + half) / count) : Math.trunc((acc - half) / count);
          out[(oy * OW + ox) * C + c] = clamp(acc, amin, amax);
        }
      }
    }
    return out;
  }

  function fullyConnected(op, input) {
    const N = op.in_features;
    const OUT = op.out_shape[op.out_shape.length - 1];
    const { weights: w, bias, mult, shift, in_offset: inOff, out_offset: outOff } = op;
    const perChannel = mult.length > 1;
    const [amin, amax] = op.act;
    const out = new Int8Array(OUT);
    for (let o = 0; o < OUT; o++) {
      let acc = 0;
      const wb = o * N;
      for (let i = 0; i < N; i++) acc += (input[i] + inOff) * w[wb + i];
      const q = perChannel ? o : 0;
      out[o] = clamp(multiplyByQuantizedMultiplier(acc + bias[o], mult[q], shift[q]) + outOff, amin, amax);
    }
    return out;
  }

  const KERNELS = {
    CONV_2D: conv2d,
    DEPTHWISE_CONV_2D: depthwiseConv2d,
    AVERAGE_POOL_2D: averagePool2d,
    FULLY_CONNECTED: fullyConnected,
    RESHAPE: (op, input) => input,
  };

  function decodeBase64(b64) {
    if (typeof Buffer !== 'undefined') {
      const buf = Buffer.from(b64, 'base64');
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    }
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  /** Typed views over little-endian buffers (all supported browsers and Node are LE). */
  function materialise(op) {
    const r = Object.assign({}, op);
    if (op.weights) r.weights = new Int8Array(decodeBase64(op.weights));
    for (const k of ['bias', 'mult', 'shift']) if (op[k]) r[k] = new Int32Array(decodeBase64(op[k]));
    if (!KERNELS[r.op]) throw new Error(`unsupported op ${r.op}`);
    return r;
  }

  function opMacs(op) {
    const n = (a) => a.reduce((x, y) => x * y, 1);
    if (op.op === 'CONV_2D') return n(op.out_shape) * op.filter_shape[1] * op.filter_shape[2] * op.filter_shape[3];
    if (op.op === 'DEPTHWISE_CONV_2D') return n(op.out_shape) * op.filter_shape[1] * op.filter_shape[2];
    if (op.op === 'FULLY_CONNECTED') return op.in_features * op.out_shape[op.out_shape.length - 1];
    return 0;
  }

  class Int8Model {
    constructor(doc) {
      if (doc.format !== 'amr-int8-graph/1') throw new Error(`unknown model format ${doc.format}`);
      this.input = doc.input;
      this.output = doc.output;
      this.classes = doc.classes;
      this.report = doc.report || {};
      this.ops = doc.ops.map(materialise);
      this.macs = this.ops.reduce((s, op) => s + opMacs(op), 0);
      this.weightBytes = this.ops.reduce(
        (s, op) => s + (op.weights ? op.weights.byteLength + op.bias.byteLength + op.mult.byteLength + op.shift.byteLength : 0), 0);
    }

    /** Layer table for display: op, output shape, MACs. */
    describe() {
      return this.ops.map((op) => ({ op: op.op, out: op.out_shape, macs: opMacs(op) }));
    }

    /** @param {Int8Array} x  quantised input, HWC  @returns {Int8Array} int8 logits */
    invoke(x) {
      let t = x;
      for (const op of this.ops) t = KERNELS[op.op](op, t);
      return t;
    }

    /** Pixel (0..255) to model input: q = round(p / 255 / scale) + zero_point. */
    quantizeInput(pixels) {
      const { scale, zero_point: zp } = this.input;
      const q = new Int8Array(pixels.length);
      for (let i = 0; i < pixels.length; i++) q[i] = clamp(Math.round(pixels[i] / 255 / scale) + zp, -128, 127);
      return q;
    }

    /** Dequantised logits through a numerically stable softmax. */
    probabilities(logits) {
      const { scale, zero_point: zp } = this.output;
      const z = Array.from(logits, (q) => (q - zp) * scale);
      const max = Math.max(...z);
      const e = z.map((v) => Math.exp(v - max));
      const sum = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / sum);
    }

    classify(pixels) {
      const logits = this.invoke(this.quantizeInput(pixels));
      const p = this.probabilities(logits);
      const order = p.map((v, i) => i).sort((a, b) => p[b] - p[a]);
      return { digit: order[0], confidence: p[order[0]], runnerUp: order[1], runnerUpConfidence: p[order[1]], probabilities: p, logits };
    }
  }

  return { Int8Model, multiplyByQuantizedMultiplier, saturatingRoundingDoublingHighMul, roundingDivideByPOT };
});
