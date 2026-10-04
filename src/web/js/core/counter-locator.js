/*
 * Counter-register localisation.
 *
 * Finds the row of odometer wheels on a meter photo without a learned
 * detector: the register is the tallest horizontally aligned, evenly pitched
 * chain of glyph-shaped connected components, in either print polarity.
 *
 *   1. Local adaptive binarisation, dark-on-light and light-on-dark.
 *   2. Connected components filtered by glyph geometry (height, aspect, fill).
 *   3. Greedy left-to-right chaining with height, baseline and pitch constraints.
 *   4. Chains scored by total glyph height x pitch regularity x baseline alignment.
 *   5. One ROI per wheel, sized to the training-crop geometry (aspect 0.8,
 *      glyph ~62% of crop height), with gaps of exactly one pitch filled.
 *   6. Red wheels (sub-m3 fractions on most meters) are flagged from colour.
 *
 * On a field device the ROIs are calibrated once at installation because the
 * camera is fixed; this locator stands in for that step on arbitrary photos.
 * A user-drawn region (locateInRegion) overrides it.
 */
(function (root, factory) {
  const ops = typeof module === 'object' && module.exports ? require('./image-ops.js') : root.AMR.imageOps;
  const api = factory(ops);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).counterLocator = api;
})(typeof self !== 'undefined' ? self : globalThis, function (ops) {
  'use strict';

  const ROI_HEIGHT_PER_GLYPH = 1.6; // training crops: glyph ~62% of crop height
  const ROI_ASPECT = 0.8;           // training crops are 80 x 100 px

  const median = (a) => {
    const s = [...a].sort((x, y) => x - y);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const cv = (a) => {
    if (a.length < 2) return 0;
    const m = mean(a);
    return Math.sqrt(mean(a.map((v) => (v - m) ** 2))) / m;
  };

  function isGlyph(c, w, h, minH, maxH) {
    if (c.h < minH || c.h > maxH || c.w < 2) return false;
    if (c.x <= 0 || c.y <= 0 || c.x + c.w >= w || c.y + c.h >= h) return false;
    const aspect = c.w / c.h;
    const fill = c.area / (c.w * c.h);
    if (aspect < 0.12 || aspect > 1.0) return false;
    // A "1" is a near-solid bar; every other digit is an open stroke shape.
    // The 0.62 ceiling rejects solid drum faces and window frames.
    return aspect < 0.33 ? fill >= 0.3 && fill <= 0.95 : fill >= 0.15 && fill <= 0.62;
  }

  function buildChains(cands) {
    const items = cands.map((c) => ({ ...c, cx: c.x + c.w / 2, cy: c.y + c.h / 2 })).sort((a, b) => a.cx - b.cx);
    const chains = [];
    for (let i = 0; i < items.length; i++) {
      const chain = [items[i]];
      for (;;) {
        const last = chain[chain.length - 1];
        const hRef = median(chain.map((c) => c.h));
        const pitches = chain.slice(1).map((c, k) => c.cx - chain[k].cx);
        const pRef = pitches.length ? median(pitches) : null;
        let best = null;
        for (let j = 0; j < items.length; j++) {
          const c = items[j];
          if (c.cx <= last.cx + 0.25 * hRef) continue;
          if (c.h / hRef < 0.78 || c.h / hRef > 1.28) continue;
          if (Math.abs(c.cy - last.cy) > 0.22 * hRef) continue;
          const gap = c.x - (last.x + last.w);
          if (gap < -0.15 * hRef || gap > 1.1 * hRef) continue;
          const p = c.cx - last.cx;
          if (pRef && Math.abs(p - pRef) / pRef > 0.35) continue;
          if (!best || p < best.cx - last.cx) best = c;
        }
        if (!best) break;
        chain.push(best);
      }
      if (chain.length >= 3) chains.push(chain);
    }
    return chains;
  }

  /**
   * Odometer wheels are wider than printed glyphs: pitch/height is ~0.8-1.1 on
   * registers but ~0.55-0.7 for serial numbers and labels. This prior is what
   * separates the register from the serial number printed next to it.
   */
  function pitchPrior(ratio) {
    if (ratio < 0.55) return 0.25;
    if (ratio < 0.78) return 0.25 + (0.75 * (ratio - 0.55)) / 0.23;
    if (ratio <= 1.5) return 1;
    return Math.max(0.3, 1 - (ratio - 1.5));
  }

  function scoreChain(chain) {
    const hs = chain.map((c) => c.h);
    const pitches = chain.slice(1).map((c, k) => c.cx - chain[k].cx);
    const hMed = median(hs);
    const baseline = Math.sqrt(mean(chain.map((c) => (c.cy - mean(chain.map((d) => d.cy))) ** 2))) / hMed;
    const n = Math.min(chain.length, 9);
    return ((n * hMed) / (1 + 3 * cv(pitches)) / (1 + 5 * baseline) / (1 + 0.5 * cv(hs))) * pitchPrior(median(pitches) / hMed);
  }

  /** Least-squares line through glyph centres: tolerates a few degrees of camera roll. */
  function fitBaseline(chain) {
    const xs = chain.map((c) => c.cx), ys = chain.map((c) => c.cy);
    const mx = mean(xs), my = mean(ys);
    let num = 0, den = 0;
    for (let i = 0; i < xs.length; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
    const slope = den > 0 ? num / den : 0;
    return (x) => my + slope * (x - mx);
  }

  function redFraction(rgba, w, h, box) {
    if (!rgba) return 0;
    let red = 0, n = 0;
    const x0 = Math.max(0, Math.round(box.x)), x1 = Math.min(w, Math.round(box.x + box.w));
    const y0 = Math.max(0, Math.round(box.y)), y1 = Math.min(h, Math.round(box.y + box.h));
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        const p = (y * w + x) * 4;
        const r = rgba[p], g = rgba[p + 1], b = rgba[p + 2];
        if (r > 90 && r > 1.45 * g && r > 1.45 * b) red++;
        n++;
      }
    }
    return n ? red / n : 0;
  }

  /** Turn a glyph chain into wheel ROIs, filling single missing wheels. */
  /**
   * @param {object} [fit] optional {count, region}: extrapolate to `count` wheels
   *   at the measured pitch, toward whichever side of `region` has room.
   */
  function chainToRois(chain, w, h, rgba, fit) {
    const pitches = chain.slice(1).map((c, k) => c.cx - chain[k].cx);
    const pitch = median(pitches);
    const glyphH = median(chain.map((c) => c.h));
    const centres = [chain[0].cx];
    for (let k = 1; k < chain.length; k++) {
      const steps = Math.round((chain[k].cx - chain[k - 1].cx) / pitch);
      for (let s = 1; s < steps; s++) centres.push(chain[k - 1].cx + (s * (chain[k].cx - chain[k - 1].cx)) / steps);
      centres.push(chain[k].cx);
    }
    const yAt = fitBaseline(chain);
    const roiH = ROI_HEIGHT_PER_GLYPH * glyphH;
    const roiW = Math.min(ROI_ASPECT * roiH, 1.04 * pitch);
    if (fit && fit.count) {
      const right = fit.region.x + fit.region.w, left = fit.region.x;
      while (centres.length < fit.count) {
        const nextR = centres[centres.length - 1] + pitch, nextL = centres[0] - pitch;
        const roomR = right - nextR, roomL = nextL - left;
        if (roomR < -roiW / 2 && roomL < -roiW / 2) break;
        if (roomR >= roomL) centres.push(nextR);
        else centres.unshift(nextL);
      }
    }
    const rois = centres.map((cx) => {
      const box = { x: cx - roiW / 2, y: yAt(cx) - roiH / 2, w: roiW, h: roiH };
      return { ...box, red: redFraction(rgba, w, h, box) };
    });
    markFractional(rois);
    return { rois, pitch, glyphH };
  }

  /** Trailing red wheels are the fractional register (litres); leading ones are m3. */
  function markFractional(rois) {
    let k = rois.length;
    while (k > 0 && rois[k - 1].red > 0.12) k--;
    rois.forEach((r, i) => { r.fractional = i >= k && k > 0; });
  }

  function unionBox(rois) {
    const x0 = Math.min(...rois.map((r) => r.x)), y0 = Math.min(...rois.map((r) => r.y));
    const x1 = Math.max(...rois.map((r) => r.x + r.w)), y1 = Math.max(...rois.map((r) => r.y + r.h));
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  /**
   * Glyph candidates from both polarities feed a single chain search: many
   * registers print integer wheels dark-on-light and fraction wheels
   * light-on-dark. A glyph and its own counter-hole never chain together: the
   * chain step requires a horizontal advance of a quarter glyph height.
   */
  function detect(gray, w, h, minH, maxH, radius) {
    const masks = ops.adaptiveMasks(gray, w, h, radius);
    const candidates = [];
    for (const polarity of ['dark', 'bright']) {
      for (const c of ops.connectedComponents(masks[polarity], w, h)) {
        if (isGlyph(c, w, h, minH, maxH)) candidates.push({ ...c, polarity });
      }
    }
    let best = null;
    for (const chain of buildChains(candidates)) {
      const score = scoreChain(chain);
      if (!best || score > best.score) best = { chain, score };
    }
    if (best) {
      const dark = best.chain.filter((c) => c.polarity === 'dark').length;
      best.polarity = dark === best.chain.length ? 'dark' : dark === 0 ? 'bright' : 'mixed';
    }
    return { best, candidates };
  }

  /** Rotate a grayscale (channels=1) or RGBA (channels=4) image by 90 degree steps clockwise. */
  function rotate(src, w, h, quarterTurns, channels) {
    const q = ((quarterTurns % 4) + 4) % 4;
    if (q === 0) return { data: src, w, h };
    const nw = q % 2 ? h : w, nh = q % 2 ? w : h;
    const out = new (src.constructor)(src.length);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let nx, ny;
        if (q === 1) { nx = h - 1 - y; ny = x; }
        else if (q === 2) { nx = w - 1 - x; ny = h - 1 - y; }
        else { nx = y; ny = w - 1 - x; }
        const s = (y * w + x) * channels, d = (ny * nw + nx) * channels;
        for (let c = 0; c < channels; c++) out[d + c] = src[s + c];
      }
    }
    return { data: out, w: nw, h: nh };
  }

  /**
   * Automatic localisation. Upright and both 90-degree orientations are searched
   * (meters are often installed with the register vertical); the best-scoring
   * orientation wins and the rotated image is returned for cropping and display.
   */
  function locate(gray, w, h, rgba) {
    // Registers are almost always photographed upright, so a rotated hypothesis
    // must beat the upright one by a wide margin (or upright must find nothing).
    const ROTATION_MARGIN = 1.8;
    let best = null;
    for (const turns of [0, 1, 3]) {
      const g = rotate(gray, w, h, turns, 1);
      const radius = Math.max(8, Math.round(Math.min(g.w, g.h) / 20));
      const r = detect(g.data, g.w, g.h, Math.max(10, 0.03 * g.h), 0.4 * g.h, radius);
      const score = r.best ? r.best.score * (turns === 0 ? ROTATION_MARGIN : 1) : 0;
      if (!best || score > best.score) best = { turns, g, r, score };
    }
    const { turns, g, r } = best;
    const frame = { turns, gray: g.data, w: g.w, h: g.h, rgba: rgba ? rotate(rgba, w, h, turns, 4).data : null };
    if (!r.best) return { ok: false, rois: [], candidates: r.candidates, method: 'auto', frame, reason: 'no aligned digit chain found' };
    const { rois, pitch, glyphH } = chainToRois(r.best.chain, frame.w, frame.h, frame.rgba);
    return {
      ok: true, method: 'auto', rois, window: unionBox(rois), candidates: r.candidates, chain: r.best.chain,
      polarity: r.best.polarity, pitch, glyphH, score: r.best.score, frame,
    };
  }

  /**
   * Localisation inside an operator-drawn region (installation-time calibration).
   * Falls back to an even split when glyphs cannot be separated.
   */
  function locateInRegion(gray, w, h, rgba, region, expectedCount) {
    // Search a margin around the drawn box: operators draw tight boxes, and
    // glyphs touching the search border are discarded by isGlyph().
    const m = 0.25 * region.h;
    const search = { x: region.x - m, y: region.y - m, w: region.w + 2 * m, h: region.h + 2 * m };
    const sub = ops.crop(gray, w, h, search);
    const ox = Math.max(0, Math.round(search.x)), oy = Math.max(0, Math.round(search.y));
    const radius = Math.max(6, Math.round(region.h / 2));
    const { best, candidates } = detect(sub.pixels, sub.w, sub.h, 0.3 * region.h, 1.1 * region.h, radius);
    const shift = (b) => ({ ...b, x: b.x + ox, y: b.y + oy, cx: b.cx !== undefined ? b.cx + ox : undefined, cy: b.cy !== undefined ? b.cy + oy : undefined });
    if (best && (!expectedCount || best.chain.length >= Math.ceil(expectedCount / 2))) {
      const chain = best.chain.map(shift);
      const { rois, pitch, glyphH } = chainToRois(chain, w, h, rgba, { count: expectedCount, region });
      return { ok: true, method: 'region', rois, window: region, candidates: candidates.map(shift), chain, polarity: best.polarity, pitch, glyphH };
    }
    const n = expectedCount || Math.max(1, Math.round(region.w / (ROI_ASPECT * region.h)));
    const cw = region.w / n;
    const rois = Array.from({ length: n }, (_, i) => {
      const box = { x: region.x + i * cw, y: region.y, w: cw, h: region.h };
      return { ...box, red: redFraction(rgba, w, h, box) };
    });
    markFractional(rois);
    return { ok: true, method: 'grid', rois, window: region, candidates: candidates.map(shift) };
  }

  return { locate, locateInRegion, rotate };
});
