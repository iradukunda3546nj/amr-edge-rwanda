// INT8 engine vs TFLite interpreter (BUILTIN_REF kernels): logits must match exactly.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '../../src/web');
require(path.join(WEB, 'model/digit-int8.js'));
const { Int8Model, multiplyByQuantizedMultiplier, saturatingRoundingDoublingHighMul } = require(path.join(WEB, 'js/core/int8-engine.js'));

const model = new Int8Model(globalThis.AMR.digitModel);
const golden = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/golden.json'), 'utf8'));

test('logits are bit-exact with TFLite reference kernels on 64 golden inputs', () => {
  golden.inputs.forEach((b64, i) => {
    const x = new Int8Array(Buffer.from(b64, 'base64'));
    assert.deepEqual(Array.from(model.invoke(x)), golden.logits[i], `vector ${i}`);
  });
});

test('golden vectors are classified correctly', () => {
  const correct = golden.inputs.filter((b64, i) => {
    const logits = Array.from(model.invoke(new Int8Array(Buffer.from(b64, 'base64'))));
    return logits.indexOf(Math.max(...logits)) === golden.labels[i];
  }).length;
  assert.equal(correct, golden.inputs.length);
});

test('SaturatingRoundingDoublingHighMul matches an exact BigInt reference', () => {
  const reference = (a, b) => {
    if (a === -2147483648 && b === -2147483648) return 2147483647;
    const ab = BigInt(a) * BigInt(b);
    const nudge = ab >= 0n ? 1n << 30n : 1n - (1n << 30n);
    return Number((ab + nudge) / (1n << 31n)); // BigInt division truncates toward zero, like C++
  };
  let seed = 7;
  const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) | 0);
  for (let i = 0; i < 20000; i++) {
    const a = next(), b = next();
    assert.equal(saturatingRoundingDoublingHighMul(a, b), reference(a, b), `${a} * ${b}`);
  }
});

test('MultiplyByQuantizedMultiplier rounding', () => {
  assert.equal(multiplyByQuantizedMultiplier(1000, 1 << 30, 0), 500);   // x * 0.5
  assert.equal(multiplyByQuantizedMultiplier(-1000, 1 << 30, -1), -250);
  assert.equal(multiplyByQuantizedMultiplier(3, 1 << 30, -1), 1);       // 0.75 rounds to 1
});

test('model metadata is consistent with the training report', () => {
  assert.equal(model.input.zero_point, -128);
  assert.equal(model.macs, model.report.macs);
});
