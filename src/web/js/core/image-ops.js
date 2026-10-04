/*
 * Image primitives used by the counter locator: luma conversion, local
 * adaptive thresholding (integral images) and connected-component labelling.
 * Pure functions over typed arrays; no DOM dependency, so they run in Node tests.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).imageOps = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  /** RGBA -> luma (ITU-R BT.601 integer approximation). */
  function toGray(rgba, w, h) {
    const g = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < g.length; i++, p += 4) {
      g[i] = (rgba[p] * 299 + rgba[p + 1] * 587 + rgba[p + 2] * 114 + 500) / 1000;
    }
    return g;
  }

  /** Summed-area tables of v and v^2 with a zero row/column, (w+1) x (h+1). */
  function integrals(gray, w, h) {
    const W = w + 1;
    const sum = new Float64Array(W * (h + 1));
    const sq = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
      let rs = 0, rq = 0;
      for (let x = 0; x < w; x++) {
        const v = gray[y * w + x];
        rs += v;
        rq += v * v;
        sum[(y + 1) * W + x + 1] = sum[y * W + x + 1] + rs;
        sq[(y + 1) * W + x + 1] = sq[y * W + x + 1] + rq;
      }
    }
    return { sum, sq, W };
  }

  /**
   * Local-statistics binarisation in both polarities. A pixel is "dark" when it
   * is significantly below its neighbourhood mean (Niblack-style offset scaled
   * by local standard deviation), "bright" when significantly above.
   */
  function adaptiveMasks(gray, w, h, radius, k = 0.3, minDelta = 10) {
    const { sum, sq, W } = integrals(gray, w, h);
    const dark = new Uint8Array(w * h);
    const bright = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - radius), y1 = Math.min(h, y + radius + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - radius), x1 = Math.min(w, x + radius + 1);
        const n = (y1 - y0) * (x1 - x0);
        const s = sum[y1 * W + x1] - sum[y0 * W + x1] - sum[y1 * W + x0] + sum[y0 * W + x0];
        const q = sq[y1 * W + x1] - sq[y0 * W + x1] - sq[y1 * W + x0] + sq[y0 * W + x0];
        const mean = s / n;
        const std = Math.sqrt(Math.max(0, q / n - mean * mean));
        const delta = minDelta + k * std;
        const v = gray[y * w + x];
        if (v < mean - delta) dark[y * w + x] = 1;
        else if (v > mean + delta) bright[y * w + x] = 1;
      }
    }
    return { dark, bright };
  }

  /** 8-connected component labelling (two-pass, union-find). Returns bounding boxes and areas. */
  function connectedComponents(mask, w, h) {
    const labels = new Int32Array(w * h);
    const parent = [0];
    const find = (a) => {
      while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; }
      return a;
    };
    const union = (a, b) => {
      a = find(a); b = find(b);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    };
    let next = 1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (!mask[i]) continue;
        const nb = [];
        if (x > 0 && labels[i - 1]) nb.push(labels[i - 1]);
        if (y > 0) {
          if (labels[i - w]) nb.push(labels[i - w]);
          if (x > 0 && labels[i - w - 1]) nb.push(labels[i - w - 1]);
          if (x < w - 1 && labels[i - w + 1]) nb.push(labels[i - w + 1]);
        }
        if (nb.length === 0) {
          parent.push(next);
          labels[i] = next++;
        } else {
          let m = nb[0];
          for (const l of nb) if (l < m) m = l;
          labels[i] = m;
          for (const l of nb) union(m, l);
        }
      }
    }
    const boxes = new Map();
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const l = labels[y * w + x];
        if (!l) continue;
        const r = find(l);
        let b = boxes.get(r);
        if (!b) boxes.set(r, (b = { x0: x, y0: y, x1: x, y1: y, area: 0 }));
        if (x < b.x0) b.x0 = x;
        if (x > b.x1) b.x1 = x;
        if (y < b.y0) b.y0 = y;
        if (y > b.y1) b.y1 = y;
        b.area++;
      }
    }
    return [...boxes.values()].map((b) => ({
      x: b.x0, y: b.y0, w: b.x1 - b.x0 + 1, h: b.y1 - b.y0 + 1, area: b.area,
    }));
  }

  /** Copy a rectangular region (clamped to the image) out of a grayscale image. */
  function crop(gray, w, h, box) {
    const x0 = Math.max(0, Math.round(box.x)), y0 = Math.max(0, Math.round(box.y));
    const x1 = Math.min(w, Math.round(box.x + box.w)), y1 = Math.min(h, Math.round(box.y + box.h));
    const cw = Math.max(1, x1 - x0), ch = Math.max(1, y1 - y0);
    const out = new Uint8Array(cw * ch);
    for (let y = 0; y < ch; y++) out.set(gray.subarray((y0 + y) * w + x0, (y0 + y) * w + x0 + cw), y * cw);
    return { pixels: out, w: cw, h: ch };
  }

  return { toGray, adaptiveMasks, connectedComponents, crop };
});
