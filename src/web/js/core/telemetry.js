/*
 * Uplink payload: 20 bytes, authenticated and partially encrypted.
 *
 *   off len field        encoding                              protection
 *   0   1   hdr          version(3b) | flags(5b)               authenticated
 *   1   4   meter_id     uint32 LE                             authenticated
 *   5   4   epoch        uint32 LE, CTR nonce + replay counter authenticated
 *   9   4   reading      uint32 LE, decilitres (0.0001 m3)     AES-128-CTR
 *   13  2   flow_lph     int16 LE, min night flow, 0x7FFF = n/a AES-128-CTR
 *   15  1   battery      (cell_mV - 2000) / 10                 AES-128-CTR
 *   16  4   tag          AES-CMAC(bytes 0..15), first 4 bytes  -
 *
 * Same layout as TelemetryPacket in src/firmware/esp32_edge_ocr.cpp.
 * Crypto uses WebCrypto (browser and Node >= 20); CMAC follows RFC 4493 and is
 * verified against the RFC test vectors in tools/test/telemetry.test.mjs.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).telemetry = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  const subtle = (typeof crypto !== 'undefined' ? crypto : require('crypto').webcrypto).subtle;
  const VERSION = 1;
  const FLOW_UNAVAILABLE = 0x7fff;
  const FLAGS = { LEAK: 0x01, ENHANCED: 0x02, LOW_CONFIDENCE: 0x04, RATE_FAULT: 0x08, LOW_BATTERY: 0x10 };

  const FIELDS = [
    { name: 'hdr', offset: 0, length: 1, protection: 'auth' },
    { name: 'meter_id', offset: 1, length: 4, protection: 'auth' },
    { name: 'epoch', offset: 5, length: 4, protection: 'auth' },
    { name: 'reading', offset: 9, length: 4, protection: 'enc' },
    { name: 'flow_lph', offset: 13, length: 2, protection: 'enc' },
    { name: 'battery', offset: 15, length: 1, protection: 'enc' },
    { name: 'tag', offset: 16, length: 4, protection: 'mac' },
  ];

  const importKey = (raw, alg) => subtle.importKey('raw', raw, alg, false, ['encrypt']);

  /** Single-block AES-128 encryption (ECB) via CBC with a zero IV. */
  async function aesBlock(keyCbc, block) {
    const out = await subtle.encrypt({ name: 'AES-CBC', iv: new Uint8Array(16) }, keyCbc, block);
    return new Uint8Array(out, 0, 16);
  }

  function leftShift1(b) {
    const out = new Uint8Array(16);
    for (let i = 0; i < 16; i++) out[i] = ((b[i] << 1) | (i < 15 ? b[i + 1] >> 7 : 0)) & 0xff;
    if (b[0] & 0x80) out[15] ^= 0x87;
    return out;
  }

  /** AES-CMAC (RFC 4493 / NIST SP 800-38B), full 16-byte tag. */
  async function aesCmac(keyBytes, message) {
    const key = await importKey(keyBytes, 'AES-CBC');
    const k1 = leftShift1(await aesBlock(key, new Uint8Array(16)));
    const k2 = leftShift1(k1);
    const n = Math.max(1, Math.ceil(message.length / 16));
    const complete = message.length > 0 && message.length % 16 === 0;
    const last = new Uint8Array(16);
    const tail = message.subarray((n - 1) * 16);
    last.set(tail);
    if (!complete) last[tail.length] = 0x80;
    const sub = complete ? k1 : k2;
    for (let i = 0; i < 16; i++) last[i] ^= sub[i];
    let x = new Uint8Array(16);
    for (let b = 0; b < n - 1; b++) {
      const y = new Uint8Array(16);
      for (let i = 0; i < 16; i++) y[i] = x[i] ^ message[b * 16 + i];
      x = await aesBlock(key, y);
    }
    for (let i = 0; i < 16; i++) last[i] ^= x[i];
    return aesBlock(key, last);
  }

  /** AES-128-CTR with initial counter block meter_id || epoch || 0^8 (bytes 1..8 of the packet). */
  async function ctrTransform(keyBytes, packet) {
    const key = await importKey(keyBytes, 'AES-CTR');
    const counter = new Uint8Array(16);
    counter.set(packet.subarray(1, 9));
    const body = packet.slice(9, 16);
    const out = await subtle.encrypt({ name: 'AES-CTR', counter, length: 64 }, key, body);
    return new Uint8Array(out);
  }

  /**
   * @param {{meterId:number, epoch:number, readingM3:number, flowLph?:number, batteryMv:number, flags:number}} m
   * @param {Uint8Array} key 16-byte device key
   */
  const MAX_READING_M3 = 0xffffffff / 10000; // uint32 decilitres

  async function encode(m, key) {
    if (!(m.readingM3 >= 0 && m.readingM3 <= MAX_READING_M3)) {
      throw new RangeError(`reading ${m.readingM3} m3 is outside the payload range 0..${MAX_READING_M3} m3`);
    }
    const p = new Uint8Array(20);
    const dv = new DataView(p.buffer);
    p[0] = ((VERSION & 0x07) << 5) | (m.flags & 0x1f);
    dv.setUint32(1, m.meterId >>> 0, true);
    dv.setUint32(5, m.epoch >>> 0, true);
    dv.setUint32(9, Math.round(m.readingM3 * 10000) >>> 0, true);
    dv.setInt16(13, m.flowLph === undefined || m.flowLph === null ? FLOW_UNAVAILABLE : m.flowLph, true);
    p[15] = Math.max(0, Math.min(255, Math.round((m.batteryMv - 2000) / 10)));
    const plaintext = p.slice();
    p.set(await ctrTransform(key, p), 9);
    p.set((await aesCmac(key, p.subarray(0, 16))).subarray(0, 4), 16);
    return { packet: p, plaintext };
  }

  /** Verify the tag, then decrypt. Throws on authentication failure. */
  async function decode(packet, key) {
    if (packet.length !== 20) throw new Error(`payload must be 20 bytes, got ${packet.length}`);
    const tag = (await aesCmac(key, packet.subarray(0, 16))).subarray(0, 4);
    let diff = 0;
    for (let i = 0; i < 4; i++) diff |= tag[i] ^ packet[16 + i];
    if (diff) throw new Error('CMAC verification failed');
    const p = packet.slice();
    p.set(await ctrTransform(key, p), 9);
    const dv = new DataView(p.buffer);
    const flow = dv.getInt16(13, true);
    return {
      version: p[0] >> 5,
      flags: p[0] & 0x1f,
      meterId: dv.getUint32(1, true),
      epoch: dv.getUint32(5, true),
      readingM3: dv.getUint32(9, true) / 10000,
      flowLph: flow === FLOW_UNAVAILABLE ? null : flow,
      batteryMv: p[15] * 10 + 2000,
    };
  }

  return { encode, decode, aesCmac, FIELDS, FLAGS, VERSION, MAX_READING_M3 };
});
