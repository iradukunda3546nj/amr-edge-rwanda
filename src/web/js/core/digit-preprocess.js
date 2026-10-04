/*
 * Digit-crop normalisation: JavaScript port of src/model/preprocess.py.
 * The model was trained on exactly this transform; keep both in sync.
 * Parity is checked by tools/test/preprocess.test.mjs.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).digitPreprocess = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  const OUT_W = 24;
  const OUT_H = 32;

  /** Sparse area-coverage weights: for each output index, [firstSourceIndex, weights[]]. */
  function areaWeights(src, dst) {
    const scale = src / dst;
    const rows = [];
    for (let i = 0; i < dst; i++) {
      const lo = i * scale, hi = (i + 1) * scale;
      const j0 = Math.floor(lo), j1 = Math.min(Math.ceil(hi), src);
      const w = [];
      let sum = 0;
      for (let j = j0; j < j1; j++) {
        const v = Math.min(hi, j + 1) - Math.max(lo, j);
        w.push(v);
        sum += v;
      }
      rows.push([j0, w.map((v) => v / sum)]);
    }
    return rows;
  }

  /** Separable area-average resize of a grayscale image (Uint8Array, row-major). */
  function areaResize(src, w, h, outW = OUT_W, outH = OUT_H) {
    const wx = areaWeights(w, outW);
    const wy = areaWeights(h, outH);
    const tmp = new Float64Array(h * outW);
    for (let y = 0; y < h; y++) {
      for (let i = 0; i < outW; i++) {
        const [j0, ws] = wx[i];
        let acc = 0;
        for (let k = 0; k < ws.length; k++) acc += src[y * w + j0 + k] * ws[k];
        tmp[y * outW + i] = acc;
      }
    }
    const out = new Uint8Array(outW * outH);
    for (let i = 0; i < outH; i++) {
      const [j0, ws] = wy[i];
      for (let x = 0; x < outW; x++) {
        let acc = 0;
        for (let k = 0; k < ws.length; k++) acc += tmp[(j0 + k) * outW + x] * ws[k];
        out[i * outW + x] = Math.min(255, Math.max(0, Math.floor(acc + 0.5)));
      }
    }
    return out;
  }

  function otsuThreshold(values) {
    const hist = new Float64Array(256);
    for (const v of values) hist[v]++;
    const total = values.length;
    let sumAll = 0;
    for (let t = 0; t < 256; t++) sumAll += t * hist[t];
    let wB = 0, sumB = 0, bestT = 0, bestVar = -1;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (wB === 0) continue;
      const wF = total - wB;
      if (wF === 0) break;
      sumB += t * hist[t];
      const mB = sumB / wB, mF = (sumAll - sumB) / wF;
      const v = wB * wF * (mB - mF) * (mB - mF);
      if (v > bestVar) { bestVar = v; bestT = t; }
    }
    return bestT;
  }

  function percentileBounds(img, loFrac = 0.02, hiFrac = 0.98) {
    const hist = new Uint32Array(256);
    for (const v of img) hist[v]++;
    const kLo = Math.ceil(loFrac * img.length), kHi = Math.ceil(hiFrac * img.length);
    let cum = 0, lo = -1, hi = -1;
    for (let v = 0; v < 256; v++) {
      cum += hist[v];
      if (lo < 0 && cum >= kLo) lo = v;
      if (hi < 0 && cum >= kHi) { hi = v; break; }
    }
    return [lo, Math.max(hi, lo + 1)];
  }

  /**
   * @param {Uint8Array} crop grayscale crop of a single counter wheel
   * @returns {{pixels: Uint8Array, inverted: boolean}} 24x32 normalised digit
   */
  function normalizeDigit(crop, w, h) {
    let x = areaResize(crop, w, h);
    const cy0 = Math.floor(OUT_H * 0.15), cy1 = Math.floor(OUT_H * 0.85);
    const cx0 = Math.floor(OUT_W * 0.2), cx1 = Math.floor(OUT_W * 0.8);
    const centre = [];
    for (let y = cy0; y < cy1; y++) for (let i = cx0; i < cx1; i++) centre.push(x[y * OUT_W + i]);
    const t = otsuThreshold(centre);
    let above = 0;
    for (const v of centre) if (v > t) above++;
    const inverted = above * 2 > centre.length;
    if (inverted) x = x.map((v) => 255 - v);
    const [lo, hi] = percentileBounds(x);
    const out = new Uint8Array(x.length);
    for (let i = 0; i < x.length; i++) {
      out[i] = Math.min(255, Math.max(0, Math.floor(((x[i] - lo) * 255) / (hi - lo) + 0.5)));
    }
    return { pixels: out, inverted };
  }

  return { OUT_W, OUT_H, areaResize, otsuThreshold, percentileBounds, normalizeDigit };
});
