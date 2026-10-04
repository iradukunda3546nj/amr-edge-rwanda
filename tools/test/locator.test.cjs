// Register localisation on a synthetic five-wheel register and edge cases.
const test = require('node:test');
const assert = require('node:assert/strict');
const ops = require('../../src/web/js/core/image-ops.js');
const { locate, rotate } = require('../../src/web/js/core/counter-locator.js');

/** Light background with n dark "0"-shaped glyphs at a fixed pitch. */
function syntheticRegister(w, h, n, glyphH, pitch, x0, y0) {
  const g = new Uint8Array(w * h).fill(220);
  const gw = Math.round(glyphH * 0.55), t = Math.max(2, Math.round(glyphH * 0.1)); // stroke ~10% of height, like DIN digits
  for (let k = 0; k < n; k++) {
    const gx = x0 + k * pitch;
    for (let y = y0; y < y0 + glyphH; y++) {
      for (let x = gx; x < gx + gw; x++) {
        if (x < gx + t || x >= gx + gw - t || y < y0 + t || y >= y0 + glyphH - t) g[y * w + x] = 30;
      }
    }
  }
  return g;
}

test('locates an evenly pitched five-wheel register in reading order', () => {
  const gray = syntheticRegister(640, 480, 5, 60, 52, 150, 200);
  const r = locate(gray, 640, 480, null);
  assert.ok(r.ok, r.reason);
  assert.equal(r.rois.length, 5);
  for (let k = 1; k < 5; k++) assert.ok(r.rois[k].x > r.rois[k - 1].x);
  assert.ok(Math.abs(r.pitch - 52) < 3, `pitch ${r.pitch}`);
});

test('finds a register mounted vertically', () => {
  const gray = syntheticRegister(640, 480, 5, 60, 52, 150, 200);
  const turned = rotate(gray, 640, 480, 3, 1);
  const r = locate(turned.data, turned.w, turned.h, null);
  assert.ok(r.ok, r.reason);
  assert.equal(r.rois.length, 5);
  assert.notEqual(r.frame.turns, 0);
});

test('reports failure on a featureless image', () => {
  const r = locate(new Uint8Array(320 * 240).fill(128), 320, 240, null);
  assert.equal(r.ok, false);
});

test('8-connected labelling separates blobs and joins diagonals', () => {
  const w = 20, h = 10, m = new Uint8Array(w * h);
  m[1 * w + 1] = m[2 * w + 2] = 1; // diagonal neighbours: one component
  m[5 * w + 10] = 1;               // isolated pixel
  assert.equal(ops.connectedComponents(m, w, h).length, 2);
});
