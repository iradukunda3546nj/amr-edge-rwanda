const test = require('node:test');
const assert = require('node:assert/strict');
const { aesCmac, encode, decode, FLAGS } = require('../../src/web/js/core/telemetry.js');

const hex = (s) => Uint8Array.from(s.match(/../g).map((h) => parseInt(h, 16)));
const toHex = (b) => Buffer.from(b).toString('hex');
const KEY = hex('2b7e151628aed2a6abf7158809cf4f3c');

test('AES-CMAC matches RFC 4493 section 4 test vectors', async () => {
  const msg = hex(
    '6bc1bee22e409f96e93d7e117393172aae2d8a571e03ac9c9eb76fac45af8e51' +
    '30c81c46a35ce411e5fbc1191a0a52eff69f2445df4f9b17ad2b417be66c3710');
  assert.equal(toHex(await aesCmac(KEY, msg.subarray(0, 0))), 'bb1d6929e95937287fa37d129b756746');
  assert.equal(toHex(await aesCmac(KEY, msg.subarray(0, 16))), '070a16b46b4d4144f79bdd9dd04a287c');
  assert.equal(toHex(await aesCmac(KEY, msg.subarray(0, 40))), 'dfa66747de9ae63030ca32611497c827');
  assert.equal(toHex(await aesCmac(KEY, msg)), '51f0bebf7e3b9d92fc49741779363cfe');
});

test('payload round-trips and is exactly 20 bytes', async () => {
  const m = { meterId: 10432, epoch: 1791100800, readingM3: 752.1386, flowLph: null, batteryMv: 3620, flags: FLAGS.LEAK };
  const { packet } = await encode(m, KEY);
  assert.equal(packet.length, 20);
  const d = await decode(packet, KEY);
  assert.deepEqual(
    { meterId: d.meterId, epoch: d.epoch, readingM3: d.readingM3, flowLph: d.flowLph, batteryMv: d.batteryMv, flags: d.flags, version: d.version },
    { meterId: 10432, epoch: 1791100800, readingM3: 752.1386, flowLph: null, batteryMv: 3620, flags: FLAGS.LEAK, version: 1 });
});

test('flipping any bit of the packet is rejected', async () => {
  const { packet } = await encode({ meterId: 1, epoch: 2, readingM3: 3, batteryMv: 3600, flags: 0 }, KEY);
  for (let i = 0; i < 20; i++) {
    const p = packet.slice();
    p[i] ^= 0x01;
    await assert.rejects(decode(p, KEY), /CMAC/, `byte ${i}`);
  }
});

test('measurement fields are encrypted on air, header stays clear', async () => {
  const { packet, plaintext } = await encode({ meterId: 1, epoch: 99, readingM3: 75.3, batteryMv: 3600, flags: 0 }, KEY);
  assert.notDeepEqual(Array.from(packet.subarray(9, 16)), Array.from(plaintext.subarray(9, 16)));
  assert.deepEqual(Array.from(packet.subarray(0, 9)), Array.from(plaintext.subarray(0, 9)));
});

test('readings outside the uint32 decilitre range are refused, not wrapped', async () => {
  const { MAX_READING_M3 } = require('../../src/web/js/core/telemetry.js');
  await assert.rejects(encode({ meterId: 1, epoch: 1, readingM3: 600027, batteryMv: 3600, flags: 0 }, KEY), RangeError);
  await assert.rejects(encode({ meterId: 1, epoch: 1, readingM3: -1, batteryMv: 3600, flags: 0 }, KEY), RangeError);
  const { packet } = await encode({ meterId: 1, epoch: 1, readingM3: 99999.9999, batteryMv: 3600, flags: 0 }, KEY);
  assert.equal((await decode(packet, KEY)).readingM3, 99999.9999);
  assert.ok(MAX_READING_M3 > 429496 && MAX_READING_M3 < 429497);
});
