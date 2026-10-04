const test = require('node:test');
const assert = require('node:assert/strict');
const { encode } = require('../../src/web/js/core/telemetry.js');
const mqtt = require('../../src/web/js/core/mqtt.js');
const { WasacCloud, charge } = require('../../src/web/js/sim/wasac-cloud.js');

const KEY = new Uint8Array(16).fill(7);
const ACCOUNT = { meterId: 10432, label: 'MTR-10432', customer: 'Test', zone: 'Kicukiro', msisdn: '+250 78x xxx 432', key: KEY };

async function frame(epoch, m3, key = KEY) {
  const { packet } = await encode({ meterId: 10432, epoch, readingM3: m3, batteryMv: 3600, flags: 0 }, key);
  return mqtt.publish({ topic: 'wasac/v1/meters/10432/up', payload: packet, qos: 1, packetId: 1 });
}

test('increasing-block tariff', () => {
  assert.equal(charge(0), 0);
  assert.equal(charge(5), 5 * 402);
  assert.equal(charge(6), 5 * 402 + 852);
});

test('baseline, normal use, leak alert, replay and rollback', async () => {
  const cloud = new WasacCloud([ACCOUNT]);
  const day = 86400;
  assert.equal((await cloud.ingest(await frame(1000, 75))).decision, 'baseline');
  assert.equal((await cloud.ingest(await frame(1000 + day, 75.4))).decision, 'ok');
  const leak = await cloud.ingest(await frame(1000 + 2 * day, 104.2));
  assert.equal(leak.decision, 'alert');
  assert.match(leak.sms, /ALERT/);
  for (const e of cloud.log) if (e.sms) assert.ok(e.sms.length <= 160, `${e.sms.length} chars: ${e.sms}`);
  assert.equal((await cloud.ingest(await frame(1000, 200))).decision, 'rejected');         // replayed epoch
  assert.equal((await cloud.ingest(await frame(1000 + 3 * day, 50))).decision, 'review');  // register went backwards
});

test('payload sealed with the wrong key is rejected', async () => {
  const cloud = new WasacCloud([ACCOUNT]);
  const e = await cloud.ingest(await frame(5000, 10, new Uint8Array(16).fill(9)));
  assert.equal(e.decision, 'rejected');
  assert.match(e.reason, /CMAC/);
});
