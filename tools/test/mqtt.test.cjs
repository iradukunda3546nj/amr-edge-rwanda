const test = require('node:test');
const assert = require('node:assert/strict');
const mqtt = require('../../src/web/js/core/mqtt.js');

test('remaining-length varint at the MQTT 3.1.1 boundary values', () => {
  const cases = [[0, [0x00]], [127, [0x7f]], [128, [0x80, 0x01]], [16383, [0xff, 0x7f]],
    [16384, [0x80, 0x80, 0x01]], [268435455, [0xff, 0xff, 0xff, 0x7f]]];
  for (const [n, enc] of cases) {
    assert.deepEqual(mqtt.encodeLength(n), enc);
    assert.equal(mqtt.decodeLength(Uint8Array.from(enc), 0).value, n);
  }
  assert.throws(() => mqtt.encodeLength(268435456));
});

test('PUBLISH QoS 1 layout and round trip', () => {
  const payload = Uint8Array.from({ length: 20 }, (_, i) => i);
  const topic = 'wasac/v1/meters/10432/up';
  const f = mqtt.publish({ topic, payload, qos: 1, packetId: 1 });
  assert.equal(f[0], 0x32);                                  // type 3, QoS 1
  assert.equal(f[1], 2 + topic.length + 2 + payload.length); // remaining length
  const p = mqtt.parse(f);
  assert.equal(p.type, 'PUBLISH');
  assert.equal(p.topic, topic);
  assert.equal(p.packetId, 1);
  assert.deepEqual(Array.from(p.payload), Array.from(payload));
});

test('CONNECT, CONNACK, PUBACK and DISCONNECT encodings', () => {
  const c = mqtt.parse(mqtt.connect({ clientId: 'MTR-10432', keepAlive: 60 }));
  assert.equal(c.protocol, 'MQTT');
  assert.equal(c.level, 4);
  assert.equal(c.keepAlive, 60);
  assert.equal(c.clientId, 'MTR-10432');
  assert.deepEqual(Array.from(mqtt.connack(false, 0)), [0x20, 0x02, 0x00, 0x00]);
  assert.deepEqual(Array.from(mqtt.puback(1)), [0x40, 0x02, 0x00, 0x01]);
  assert.deepEqual(Array.from(mqtt.disconnect()), [0xe0, 0x00]);
});
