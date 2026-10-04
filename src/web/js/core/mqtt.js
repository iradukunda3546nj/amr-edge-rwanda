/*
 * MQTT 3.1.1 control-packet codec (OASIS standard, sections 2-3): the subset a
 * telemetry device uses. Produces and parses the exact bytes that cross the
 * TLS session between the modem and the WASAC broker.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).mqtt = api;
})(typeof self !== 'undefined' ? self : globalThis, function () {
  'use strict';

  const TYPE = { CONNECT: 1, CONNACK: 2, PUBLISH: 3, PUBACK: 4, PINGREQ: 12, PINGRESP: 13, DISCONNECT: 14 };
  const TYPE_NAME = Object.fromEntries(Object.entries(TYPE).map(([k, v]) => [v, k]));
  const utf8 = new TextEncoder();
  const utf8dec = new TextDecoder();

  /** Variable-length "Remaining Length" (1-4 bytes, 7 bits per byte). */
  function encodeLength(n) {
    if (n < 0 || n > 268435455) throw new RangeError('remaining length out of range');
    const out = [];
    do {
      let b = n % 128;
      n = Math.floor(n / 128);
      if (n > 0) b |= 0x80;
      out.push(b);
    } while (n > 0);
    return out;
  }

  function decodeLength(buf, offset) {
    let mult = 1, value = 0, i = offset, b;
    do {
      if (i - offset >= 4) throw new Error('malformed remaining length');
      b = buf[i++];
      value += (b & 0x7f) * mult;
      mult *= 128;
    } while (b & 0x80);
    return { value, bytes: i - offset };
  }

  const str = (s) => { const b = utf8.encode(s); return [b.length >> 8, b.length & 0xff, ...b]; };
  const u16 = (n) => [(n >> 8) & 0xff, n & 0xff];

  function packet(firstByte, body) {
    return Uint8Array.from([firstByte, ...encodeLength(body.length), ...body]);
  }

  function connect({ clientId, username, password, keepAlive = 60, cleanSession = true }) {
    let flags = cleanSession ? 0x02 : 0;
    if (username !== undefined) flags |= 0x80;
    if (password !== undefined) flags |= 0x40;
    const body = [...str('MQTT'), 0x04, flags, ...u16(keepAlive), ...str(clientId)];
    if (username !== undefined) body.push(...str(username));
    if (password !== undefined) body.push(...str(password));
    return packet(TYPE.CONNECT << 4, body);
  }

  const connack = (sessionPresent = false, returnCode = 0) => packet(TYPE.CONNACK << 4, [sessionPresent ? 1 : 0, returnCode]);

  function publish({ topic, payload, qos = 1, packetId = 1, retain = false, dup = false }) {
    const body = [...str(topic)];
    if (qos > 0) body.push(...u16(packetId));
    body.push(...payload);
    return packet((TYPE.PUBLISH << 4) | (dup ? 8 : 0) | (qos << 1) | (retain ? 1 : 0), body);
  }

  const puback = (packetId) => packet(TYPE.PUBACK << 4, u16(packetId));
  const disconnect = () => packet(TYPE.DISCONNECT << 4, []);

  /** Parse one control packet; returns its type and fields plus the header/body split for display. */
  function parse(buf) {
    const type = buf[0] >> 4;
    const { value: length, bytes } = decodeLength(buf, 1);
    const start = 1 + bytes;
    const body = buf.subarray(start, start + length);
    const out = { type: TYPE_NAME[type], headerBytes: start, length };
    const readStr = (o) => { const n = (body[o] << 8) | body[o + 1]; return [utf8dec.decode(body.subarray(o + 2, o + 2 + n)), o + 2 + n]; };
    if (type === TYPE.PUBLISH) {
      const qos = (buf[0] >> 1) & 3;
      const [topic, o] = readStr(0);
      out.qos = qos;
      out.topic = topic;
      out.packetId = qos ? (body[o] << 8) | body[o + 1] : null;
      out.payload = body.subarray(qos ? o + 2 : o);
    } else if (type === TYPE.CONNACK) {
      out.sessionPresent = !!(body[0] & 1);
      out.returnCode = body[1];
    } else if (type === TYPE.PUBACK) {
      out.packetId = (body[0] << 8) | body[1];
    } else if (type === TYPE.CONNECT) {
      const [proto, o] = readStr(0);
      out.protocol = proto;
      out.level = body[o];
      out.keepAlive = (body[o + 2] << 8) | body[o + 3];
      out.clientId = readStr(o + 4)[0];
    }
    return out;
  }

  return { connect, connack, publish, puback, disconnect, parse, encodeLength, decodeLength, TYPE };
});
