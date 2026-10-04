/*
 * WASAC telemetry back end (simulated): consumes the MQTT PUBLISH bytes the
 * device sent, authenticates and decrypts the payload, validates the reading
 * against the account ledger, prices the consumption and composes the
 * customer SMS. All state is in memory for the session.
 */
(function (root, factory) {
  const deps = typeof module === 'object' && module.exports
    ? { mqtt: require('../core/mqtt.js'), telemetry: require('../core/telemetry.js') }
    : { mqtt: root.AMR.mqtt, telemetry: root.AMR.telemetry };
  const api = factory(deps);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).wasacCloud = api;
})(typeof self !== 'undefined' ? self : globalThis, function (deps) {
  'use strict';

  const { mqtt, telemetry } = deps;

  /** Illustrative increasing-block residential tariff (RWF per m3). Replace with the current RURA schedule. */
  const TARIFF = [
    { upTo: 5, rate: 402 },
    { upTo: 20, rate: 852 },
    { upTo: 50, rate: 1288 },
    { upTo: Infinity, rate: 1486 },
  ];
  const HIGH_USE_M3_PER_DAY = 1.5;   // ~5x typical Kigali household demand
  const TOPIC = /^wasac\/v1\/meters\/(\d+)\/up$/;

  function charge(m3) {
    let rest = m3, prev = 0, total = 0;
    for (const b of TARIFF) {
      const inBand = Math.min(rest, b.upTo - prev);
      if (inBand <= 0) break;
      total += inBand * b.rate;
      rest -= inBand;
      prev = b.upTo;
    }
    return Math.round(total);
  }

  const GSM7 = /^[A-Za-z0-9 @£$¥èéùìòÇØøÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ!"#¤%&'()*+,\-./:;<=>?¡ÄÖÑÜ§¿äöñüà\n\r]*$/;
  const CAT_OFFSET_S = 2 * 3600; // customer-facing times are local (UTC+2)
  const fmtDate = (epoch) => {
    const d = new Date((epoch + CAT_OFFSET_S) * 1000);
    const p = (n) => String(n).padStart(2, '0');
    return `${p(d.getUTCDate())}/${p(d.getUTCMonth() + 1)} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
  };
  const fmtNum = (n) => Math.round(n).toLocaleString('en-US');
  const fmtM3 = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, ''));

  class WasacCloud {
    /** @param {Array<{meterId:number, label:string, customer:string, zone:string, msisdn:string, key:Uint8Array}>} accounts */
    constructor(accounts) {
      this.accounts = new Map(accounts.map((a) => [a.meterId, { ...a, readings: [], lastEpoch: 0 }]));
      this.log = [];
    }

    account(meterId) { return this.accounts.get(meterId); }

    /** @param {Uint8Array} publishFrame raw MQTT PUBLISH packet */
    async ingest(publishFrame) {
      const frame = mqtt.parse(publishFrame);
      const entry = { topic: frame.topic, bytes: publishFrame.length, steps: [] };
      const reject = (reason) => { entry.decision = 'rejected'; entry.reason = reason; this.log.unshift(entry); return entry; };
      entry.steps.push(`MQTT ${frame.type} QoS ${frame.qos}, packet id ${frame.packetId}, ${frame.payload.length} B payload`);

      const m = TOPIC.exec(frame.topic || '');
      if (frame.type !== 'PUBLISH' || !m) return reject('unexpected topic');
      const acct = this.accounts.get(Number(m[1]));
      if (!acct) return reject(`unknown meter ${m[1]}`);
      entry.meterId = acct.meterId;

      let rec;
      try {
        rec = await telemetry.decode(frame.payload, acct.key);
      } catch (e) {
        return reject(e.message);
      }
      entry.steps.push('AES-CMAC tag verified, payload decrypted (AES-128-CTR)');
      entry.record = rec;
      if (rec.meterId !== acct.meterId) return reject('meter id does not match topic');
      if (rec.epoch <= acct.lastEpoch) return reject('replayed or stale epoch');
      entry.steps.push(`epoch ${rec.epoch} is newer than last accepted ${acct.lastEpoch || 'none'}`);

      const prev = acct.readings[acct.readings.length - 1];
      const r = { epoch: rec.epoch, m3: rec.readingM3, flags: rec.flags };
      if (rec.flags & telemetry.FLAGS.LOW_CONFIDENCE) {
        r.status = 'review';
        entry.steps.push('device flagged low OCR confidence; held for manual review');
      } else if (!prev) {
        r.status = 'baseline';
        r.usedM3 = 0;
        entry.steps.push('first reading on account: registered as billing baseline');
      } else if (rec.readingM3 < prev.m3) {
        r.status = 'review';
        entry.steps.push(`register decreased (${fmtM3(prev.m3)} to ${fmtM3(rec.readingM3)} m3): possible misread or meter exchange, held for review`);
      } else {
        const days = (rec.epoch - prev.epoch) / 86400;
        r.usedM3 = rec.readingM3 - prev.m3;
        r.perDay = r.usedM3 / days;
        r.status = r.perDay > HIGH_USE_M3_PER_DAY || rec.flags & telemetry.FLAGS.LEAK ? 'alert' : 'ok';
        entry.steps.push(`consumption ${fmtM3(r.usedM3)} m3 over ${days.toFixed(1)} days = ${r.perDay.toFixed(2)} m3/day (alert above ${HIGH_USE_M3_PER_DAY})`);
      }
      const billable = acct.readings.filter((x) => x.usedM3).reduce((s, x) => s + x.usedM3, 0) + (r.usedM3 || 0);
      r.billToDate = charge(billable);
      acct.readings.push(r);
      acct.lastEpoch = rec.epoch;

      entry.decision = r.status;
      entry.reading = r;
      entry.sms = this.composeSms(acct, r);
      entry.steps.push(`SMS queued to ${acct.msisdn} (${entry.sms.length} chars, ${GSM7.test(entry.sms) ? 'GSM-7' : 'UCS-2'})`);
      this.log.unshift(entry);
      return entry;
    }

    /** One 160-character GSM-7 segment, plain text. */
    composeSms(acct, r) {
      const id = acct.label;
      const reg = fmtM3(r.m3);
      switch (r.status) {
        case 'baseline':
          return `WASAC: Meter ${id} now reads automatically. Reading ${reg} m3 on ${fmtDate(r.epoch)}. Info: *150#`;
        case 'alert':
          return `WASAC ALERT: Meter ${id} used ${fmtM3(r.usedM3)} m3 in ${Math.round(r.usedM3 / r.perDay)} day(s), above normal. Possible leak: check taps and pipes. Info: *150#`;
        case 'review':
          return `WASAC: Meter ${id} reading on ${fmtDate(r.epoch)} needs verification. No charge until confirmed. Info: *150#`;
        default:
          return `WASAC: Meter ${id} read ${reg} m3 on ${fmtDate(r.epoch)}. Used ${fmtM3(r.usedM3)} m3. Bill to date ${fmtNum(r.billToDate)} RWF. Info: *150#`;
      }
    }
  }

  return { WasacCloud, charge, TARIFF, HIGH_USE_M3_PER_DAY };
});
