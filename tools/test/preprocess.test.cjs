// JS digit normalisation vs src/model/preprocess.py on real dataset crops.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { normalizeDigit } = require('../../src/web/js/core/digit-preprocess.js');

const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/preprocess.json'), 'utf8')).cases;
const bytes = (b64) => new Uint8Array(Buffer.from(b64, 'base64'));

test('normalisation is bit-identical to the Python training transform', () => {
  let px = 0, off = 0, maxDiff = 0;
  for (const c of cases) {
    const expected = bytes(c.expected);
    const { pixels } = normalizeDigit(bytes(c.src), c.w, c.h);
    for (let i = 0; i < expected.length; i++) {
      const d = Math.abs(pixels[i] - expected[i]);
      if (d) off++;
      if (d > maxDiff) maxDiff = d;
      px++;
    }
  }
  // Both implementations accumulate in the same order, so the output must be identical.
  assert.equal(maxDiff, 0, `${off}/${px} pixels differ, max difference ${maxDiff}`);
});

test('print polarity is normalised away', () => {
  const c = cases[0];
  const src = bytes(c.src);
  const a = normalizeDigit(src, c.w, c.h);
  const b = normalizeDigit(src.map((v) => 255 - v), c.w, c.h);
  assert.notEqual(a.inverted, b.inverted);
  const meanDiff = a.pixels.reduce((s, v, i) => s + Math.abs(v - b.pixels[i]), 0) / a.pixels.length;
  assert.ok(meanDiff < 3, `mean abs diff ${meanDiff}`);
});
