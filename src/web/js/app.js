/*
 * AMR-Edge dashboard controller.
 *
 * Image -> register localisation -> INT8 inference -> 20-byte sealed payload
 * -> MQTT PUBLISH -> simulated LTE/MQTT session -> WASAC ingest -> SMS.
 * Every stage consumes the previous stage's real output: the cloud parses the
 * same PUBLISH bytes the "modem" sent, and the SMS comes from the cloud's
 * decision on the decrypted payload.
 */
(function () {
  'use strict';

  const { Int8Model } = AMR.int8;
  const { normalizeDigit, OUT_W, OUT_H } = AMR.digitPreprocess;
  const ops = AMR.imageOps;
  const locator = AMR.counterLocator;
  const telemetry = AMR.telemetry;
  const mqtt = AMR.mqtt;
  const link = AMR.cellularLink;

  const MAX_SIDE = 720;            // working resolution, matches the locator tuning
  const LOW_CONFIDENCE = 0.5;      // below this a wheel is flagged for review
  const CAT_OFFSET_S = 2 * 3600;   // Rwanda, Central Africa Time (UTC+2)
  const BATTERY_MV = 3610;

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const hex = (b) => b.toString(16).padStart(2, '0').toUpperCase();
  const fmt = (n) => Math.round(n).toLocaleString('en-US');
  const fmtM3 = (v) => (v === undefined || v === null ? '—' : Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, '').replace(/\.$/, ''));
  const pad2 = (n) => String(n).padStart(2, '0');
  const utc = (epoch) => { const d = new Date(epoch * 1000); return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`; };
  const localClock = (epoch) => { const d = new Date((epoch + CAT_OFFSET_S) * 1000); return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`; };

  function setKv(dl, rows) {
    dl.replaceChildren();
    for (const [k, v] of rows) { dl.append(el('dt', '', k), el('dd', '', v)); }
  }

  /* ---------------------------------------------------------------- stages */
  const STAGES = ['acquire', 'locate', 'infer', 'packet', 'uplink', 'cloud'];
  function stage(name, state, text = '') {
    const li = document.querySelector(`.stages li[data-stage="${name}"]`);
    li.classList.remove('active', 'done', 'failed');
    if (state) li.classList.add(state);
    li.querySelector('.s').textContent = text;
  }
  function resetStages(from) {
    for (const s of STAGES.slice(STAGES.indexOf(from))) stage(s, null, '');
  }

  /* ---------------------------------------------------------------- state */
  const accounts = [
    { meterId: 10432, label: 'MTR-10432', customer: 'Household, Gikondo', zone: 'Kicukiro', msisdn: '+250 78x xxx 432' },
    { meterId: 20517, label: 'MTR-20517', customer: 'Household, Kimironko', zone: 'Gasabo', msisdn: '+250 78x xxx 517' },
    { meterId: 30981, label: 'MTR-30981', customer: 'Primary school, Nyamirambo', zone: 'Nyarugenge', msisdn: '+250 73x xxx 981' },
  ].map((a) => ({ ...a, key: crypto.getRandomValues(new Uint8Array(16)) })); // provisioned per device; shared with the back end

  const cloud = new AMR.wasacCloud.WasacCloud(accounts);
  const state = { model: null, image: null, frame: null, located: null, wheels: null, reading: null, sealed: null, publish: null, run: 0 };
  const clock = new Map(); // meterId -> last simulated capture epoch

  /* ---------------------------------------------------------------- model */
  function initModel() {
    try {
      state.model = new Int8Model(AMR.digitModel);
      const r = state.model.report;
      $('modelStatus').textContent = `INT8 model ready · ${(r.int8_bytes / 1024).toFixed(1)} KB · val ${(r.val_acc_int8 * 100).toFixed(1)}%`;
      $('modelStatus').classList.add('ready');
      renderModelCard();
    } catch (e) {
      $('modelStatus').textContent = `Model failed to load: ${e.message}`;
      $('modelStatus').classList.add('error');
    }
  }

  function renderModelCard() {
    const tbody = $('layerTable').tBodies[0];
    tbody.replaceChildren();
    for (const l of state.model.describe()) {
      const tr = el('tr');
      tr.append(el('td', '', l.op), el('td', '', l.out.join('×')), el('td', 'r', l.macs ? fmt(l.macs) : '—'));
      tbody.append(tr);
    }
    const r = state.model.report;
    setKv($('modelInfo'), [
      ['Input', `${OUT_W}×${OUT_H} grayscale, int8 (scale 1/255, zp ${state.model.input.zero_point})`],
      ['Parameters', fmt(r.params)],
      ['MACs per wheel', fmt(state.model.macs)],
      ['TFLite INT8 flatbuffer', `${fmt(r.int8_bytes)} B (FP32 ${fmt(r.fp32_bytes)} B)`],
      ['Weights + requant params', `${fmt(state.model.weightBytes)} B`],
      ['Quantisation', 'per-channel int8 weights, int8 activations, int32 bias'],
      ['Validation (real, FP32 / INT8)', `${(r.val_acc_fp32 * 100).toFixed(2)}% / ${(r.val_acc_int8 * 100).toFixed(2)}% on ${fmt(r.val_samples)} held-out crops`],
      ['Validation (synthetic fonts)', r.synthetic_val_acc_int8 ? `${(r.synthetic_val_acc_int8 * 100).toFixed(1)}%` : '—'],
      ['Kernels', 'TFLite reference integer ops, bit-exact (tools/test)'],
    ]);
  }

  /* ---------------------------------------------------------------- 01 acquisition */
  function initAccounts() {
    const sel = $('meterSelect');
    for (const a of accounts) {
      const o = el('option', '', `${a.label} · ${a.customer} (${a.zone})`);
      o.value = String(a.meterId);
      sel.append(o);
    }
    sel.addEventListener('change', renderAccount);
  }

  const currentAccount = () => cloud.account(Number($('meterSelect').value));

  async function loadFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    const run = ++state.run;
    resetStages('acquire');
    stage('acquire', 'active', 'decoding');
    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch (e) {
      stage('acquire', 'failed', 'unreadable image');
      return;
    }
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const w = Math.round(bitmap.width * scale), h = Math.round(bitmap.height * scale);
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    state.image = { name: file.name, bytes: file.size, natural: [bitmap.width, bitmap.height], w, h, rgba, gray: ops.toGray(rgba, w, h) };
    bitmap.close();
    setKv($('imageInfo'), [
      ['File', file.name],
      ['Size', `${fmt(file.size / 1024)} KB`],
      ['Resolution', `${state.image.natural[0]}×${state.image.natural[1]}`],
      ['Working frame', `${w}×${h} luma`],
    ]);
    stage('acquire', 'done', `${w}×${h}`);
    $('autoBtn').disabled = false;
    $('drawBtn').disabled = false;
    if (run === state.run) locate();
  }

  function initAcquisition() {
    const input = $('fileInput');
    $('browseBtn').addEventListener('click', () => input.click());
    input.addEventListener('change', () => { loadFile(input.files[0]); input.value = ''; });
    const dz = $('dropzone');
    dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('over'); });
    dz.addEventListener('dragleave', () => dz.classList.remove('over'));
    dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('over'); loadFile(e.dataTransfer.files[0]); });
  }

  /* ---------------------------------------------------------------- 02 localisation */
  const canvas = $('visionCanvas');
  const vctx = canvas.getContext('2d');
  let frameBitmap = null;
  let drag = null;

  function locate(region) {
    const img = state.image;
    if (!img) return;
    resetStages('locate');
    stage('locate', 'active');
    const count = Number($('wheelCount').value) || undefined;
    const t0 = performance.now();
    let r;
    if (region) {
      const f = state.frame;
      r = locator.locateInRegion(f.gray, f.w, f.h, f.rgba, region, count);
      r.frame = f;
    } else {
      r = locator.locate(img.gray, img.w, img.h, img.rgba);
      state.frame = r.frame;
      paintFrameBitmap();
    }
    const ms = performance.now() - t0;
    state.located = r;
    drawVision();
    const status = $('locateStatus');
    if (!r.ok) {
      stage('locate', 'failed', 'not found');
      status.className = 'status-line warn';
      status.textContent = `No register found automatically (${ms.toFixed(0)} ms). Drag a box around the digit window with "Draw register region".`;
      clearDownstream();
      return;
    }
    const nFrac = r.rois.filter((x) => x.fractional).length;
    const orient = r.frame.turns ? `rotated ${r.frame.turns * 90}°` : 'upright';
    status.className = 'status-line';
    status.textContent = `${r.method === 'auto' ? 'Auto' : r.method === 'region' ? 'Operator region' : 'Operator region, even split'}: ` +
      `${r.rois.length} wheels (${r.rois.length - nFrac} integer, ${nFrac} fraction) · ${orient} · ` +
      `${r.polarity || 'n/a'} print · ${r.candidates.length} glyph candidates · ${ms.toFixed(0)} ms. ` +
      'If boxes are wrong, draw the register region and set the wheel count.';
    stage('locate', 'done', `${r.rois.length} wheels`);
    infer();
  }

  function paintFrameBitmap() {
    const f = state.frame;
    const c = new OffscreenCanvas(f.w, f.h);
    c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(f.rgba), f.w, f.h), 0, 0);
    frameBitmap = c;
    canvas.width = f.w;
    canvas.height = f.h;
    $('visionEmpty').hidden = true;
  }

  function drawVision() {
    if (!frameBitmap) return;
    vctx.drawImage(frameBitmap, 0, 0);
    const r = state.located;
    const lw = Math.max(1.5, canvas.width / 400);
    if (r && $('showCandidates').checked) {
      vctx.strokeStyle = 'rgba(148,163,184,0.7)';
      vctx.lineWidth = 1;
      for (const c of r.candidates) vctx.strokeRect(c.x + 0.5, c.y + 0.5, c.w, c.h);
    }
    if (r && r.ok) {
      const win = r.window;
      vctx.setLineDash([6, 4]);
      vctx.strokeStyle = 'rgba(229,231,235,0.9)';
      vctx.lineWidth = lw;
      vctx.strokeRect(win.x - 4, win.y - 4, win.w + 8, win.h + 8);
      vctx.setLineDash([]);
      r.rois.forEach((b, i) => {
        vctx.strokeStyle = b.fractional ? '#ef4444' : '#3b82f6';
        vctx.lineWidth = lw * 1.3;
        vctx.strokeRect(b.x, b.y, b.w, b.h);
        const w = state.wheels && state.wheels[i];
        if (w) {
          vctx.font = `600 ${Math.max(11, Math.round(b.h * 0.22))}px JetBrains Mono, monospace`;
          const full = `${w.label} ${(w.confidence * 100).toFixed(0)}%`;
          const label = vctx.measureText(full).width + 6 <= b.w ? full : w.label; // avoid overlapping neighbours
          const tw = vctx.measureText(label).width + 6, th = Math.max(14, Math.round(b.h * 0.28));
          vctx.fillStyle = b.fractional ? '#ef4444' : '#3b82f6';
          vctx.fillRect(b.x, b.y - th - 2, tw, th);
          vctx.fillStyle = '#fff';
          vctx.textBaseline = 'middle';
          vctx.fillText(label, b.x + 3, b.y - th / 2 - 2);
        }
      });
    }
    if (drag) {
      vctx.strokeStyle = '#38bdf8';
      vctx.lineWidth = lw;
      vctx.setLineDash([4, 3]);
      vctx.strokeRect(drag.x, drag.y, drag.w, drag.h);
      vctx.setLineDash([]);
    }
  }

  function canvasPoint(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: ((e.clientX - rect.left) / rect.width) * canvas.width, y: ((e.clientY - rect.top) / rect.height) * canvas.height };
  }

  function initLocalisation() {
    $('autoBtn').addEventListener('click', () => locate());
    $('showCandidates').addEventListener('change', drawVision);
    const drawBtn = $('drawBtn');
    drawBtn.addEventListener('click', () => {
      const on = drawBtn.getAttribute('aria-pressed') !== 'true';
      drawBtn.setAttribute('aria-pressed', String(on));
      $('canvasWrap').classList.toggle('drawing', on);
      if (on) {
        $('locateStatus').className = 'status-line';
        $('locateStatus').textContent = 'Drag a box tightly around the digit window. Set "Wheels" to the number of digits for a more reliable split.';
      }
    });
    canvas.addEventListener('pointerdown', (e) => {
      if (drawBtn.getAttribute('aria-pressed') !== 'true' || !state.frame) return;
      const p = canvasPoint(e);
      drag = { x0: p.x, y0: p.y, x: p.x, y: p.y, w: 0, h: 0 };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const p = canvasPoint(e);
      Object.assign(drag, { x: Math.min(drag.x0, p.x), y: Math.min(drag.y0, p.y), w: Math.abs(p.x - drag.x0), h: Math.abs(p.y - drag.y0) });
      drawVision();
    });
    canvas.addEventListener('pointerup', () => {
      if (!drag) return;
      const region = { x: drag.x, y: drag.y, w: drag.w, h: drag.h };
      drag = null;
      drawBtn.setAttribute('aria-pressed', 'false');
      $('canvasWrap').classList.remove('drawing');
      if (region.w > 12 && region.h > 8) locate(region);
      else drawVision();
    });
  }

  /* ---------------------------------------------------------------- 03 inference */
  function infer() {
    const r = state.located, f = state.frame;
    stage('infer', 'active');
    const t0 = performance.now();
    const classes = state.model.classes;
    const reject = classes.indexOf('X');
    let wheels = r.rois.map((roi) => {
      const c = ops.crop(f.gray, f.w, f.h, roi);
      const n = normalizeDigit(c.pixels, c.w, c.h);
      const res = state.model.classify(n.pixels);
      return { ...res, label: classes[res.digit], runnerUpLabel: classes[res.runnerUp], input: n.pixels, inverted: n.inverted, roi };
    });
    // Marks beside the register (CE, m3, maker letters) can join the glyph chain;
    // the model's reject class identifies them, and they are trimmed from the ends.
    // Only for automatic localisation: an operator-drawn region is authoritative.
    let trimmed = 0;
    if (r.method === 'auto') {
      while (wheels.length > 3 && wheels[0].digit === reject) { wheels.shift(); trimmed++; }
      while (wheels.length > 3 && wheels[wheels.length - 1].digit === reject) { wheels.pop(); trimmed++; }
    }
    r.rois = wheels.map((w) => w.roi);
    r.window = r.rois.length ? {
      x: Math.min(...r.rois.map((b) => b.x)), y: Math.min(...r.rois.map((b) => b.y)),
      w: Math.max(...r.rois.map((b) => b.x + b.w)) - Math.min(...r.rois.map((b) => b.x)),
      h: Math.max(...r.rois.map((b) => b.y + b.h)) - Math.min(...r.rois.map((b) => b.y)),
    } : r.window;
    wheels = wheels.map((w) => ({ ...w, fractional: w.roi.fractional }));
    state.wheels = wheels;
    const ms = performance.now() - t0;

    const unreadable = wheels.some((w) => w.digit === reject);
    const digitOf = (w) => (w.digit === reject ? '?' : w.label);
    const intDigits = wheels.filter((w) => !w.fractional).map(digitOf).join('');
    const fracDigits = wheels.filter((w) => w.fractional).map(digitOf).join('');
    const num = (s) => Number(s.replace(/\?/g, '0'));
    const value = num(intDigits || '0') + (fracDigits ? num(fracDigits) / 10 ** fracDigits.length : 0);
    const minConf = Math.min(...wheels.map((w) => w.confidence));
    state.reading = { intDigits, fracDigits, value, minConf, trimmed, unreadable, lowConfidence: unreadable || minConf < LOW_CONFIDENCE };

    renderTiles();
    renderReading(ms);
    drawVision();
    $('inferMeta').textContent = `${state.wheels.length} × ${fmt(state.model.macs)} MACs · ${ms.toFixed(1)} ms in browser`;
    stage('infer', state.reading.lowConfidence ? 'failed' : 'done', `${intDigits}${fracDigits ? '.' + fracDigits : ''}`);
    seal();
  }

  function renderTiles() {
    const box = $('tiles');
    box.replaceChildren();
    for (const w of state.wheels) {
      const t = el('div', `tile${w.fractional ? ' frac' : ''}${w.confidence < LOW_CONFIDENCE ? ' low' : ''}`);
      const c = el('canvas');
      c.width = OUT_W;
      c.height = OUT_H;
      const img = new ImageData(OUT_W, OUT_H);
      w.input.forEach((v, i) => { img.data.set([v, v, v, 255], i * 4); });
      c.getContext('2d').putImageData(img, 0, 0);
      const bar = el('div', 'bar');
      const fill = el('span');
      fill.style.width = `${(w.confidence * 100).toFixed(1)}%`;
      bar.append(fill);
      t.append(c, el('div', 'digit', w.label === 'X' ? 'non-digit' : w.label), bar,
        el('div', 'conf', `p = ${w.confidence.toFixed(3)}`),
        el('div', 'alt', `next: ${w.runnerUpLabel} (${w.runnerUpConfidence.toFixed(3)})`));
      box.append(t);
    }
  }

  function renderReading(ms) {
    const r = state.reading;
    const odo = $('odometer');
    odo.replaceChildren();
    for (const d of r.intDigits) odo.append(el('span', 'cell', d));
    if (r.fracDigits) {
      odo.append(el('span', 'sep', '.'));
      for (const d of r.fracDigits) odo.append(el('span', 'cell frac', d));
    }
    odo.append(el('span', 'unit', 'm³'));
    $('readingValue').textContent = `${fmtM3(r.value)} m³`;
    setKv($('readingInfo'), [
      ['Wheels', `${r.intDigits.length} integer + ${r.fracDigits.length} fraction`],
      ['Lowest wheel confidence', r.minConf.toFixed(3)],
      ['Non-digit glyphs trimmed', String(r.trimmed)],
      ['Status', r.unreadable ? 'flagged: unreadable wheel' : r.lowConfidence ? `flagged: below ${LOW_CONFIDENCE}` : 'accepted'],
      ['Inference time (browser)', `${ms.toFixed(1)} ms`],
    ]);
  }

  /* ---------------------------------------------------------------- 04 packet */
  async function seal() {
    const run = state.run;
    stage('packet', 'active');
    const acct = currentAccount();
    const interval = Number($('intervalSelect').value);
    const last = clock.get(acct.meterId);
    // First capture of the session lands on the 06:00 CAT report slot (04:00 UTC) today.
    const today = Math.floor(Date.now() / 86400000) * 86400 + 4 * 3600;
    const epoch = last ? last + interval : today;
    clock.set(acct.meterId, epoch);

    const flags = state.reading.lowConfidence ? telemetry.FLAGS.LOW_CONFIDENCE : 0;
    const m = { meterId: acct.meterId, epoch, readingM3: state.reading.value, flowLph: null, batteryMv: BATTERY_MV, flags };
    let sealed;
    try {
      sealed = await telemetry.encode(m, acct.key);
    } catch (e) {
      clock.set(acct.meterId, last);
      stage('packet', 'failed', 'out of range');
      $('packetMeta').textContent = e.message;
      return;
    }
    const { packet, plaintext } = sealed;
    if (run !== state.run) return;
    const topic = `wasac/v1/meters/${acct.meterId}/up`;
    const frame = mqtt.publish({ topic, payload: packet, qos: 1, packetId: 1 });
    state.sealed = { m, packet, plaintext, topic, acct };
    state.publish = frame;
    renderPacket();
    stage('packet', 'done', `${packet.length} B payload`);
    uplink();
  }

  function renderPacket() {
    const { m, packet, topic } = state.sealed;
    const values = {
      hdr: `v${telemetry.VERSION}, flags 0x${hex(m.flags)}`,
      meter_id: String(m.meterId),
      epoch: `${m.epoch} (${utc(m.epoch)} UTC)`,
      reading: `${fmtM3(m.readingM3)} m³ = ${Math.round(m.readingM3 * 10000)} dL`,
      flow_lph: 'n/a (0x7FFF)',
      battery: `${m.batteryMv} mV`,
      tag: 'CMAC over bytes 0-15',
    };
    const prot = { auth: ['authenticated', 'prot-auth'], enc: ['AES-128-CTR', 'prot-enc'], mac: ['AES-CMAC-32', 'prot-mac'] };
    const tbody = $('fieldTable').tBodies[0];
    tbody.replaceChildren();
    for (const f of telemetry.FIELDS) {
      const tr = el('tr');
      const bytes = Array.from(packet.subarray(f.offset, f.offset + f.length), hex).join(' ');
      tr.append(el('td', '', `${f.name} [${f.offset}]`), el('td', '', values[f.name]), el('td', '', bytes), el('td', prot[f.protection][1], prot[f.protection][0]));
      tbody.append(tr);
    }
    // PUBLISH frame, colour-coded by section
    const frame = state.publish;
    const parsed = mqtt.parse(frame);
    const topicStart = parsed.headerBytes, topicEnd = topicStart + 2 + topic.length;
    const pidEnd = topicEnd + 2, tagStart = frame.length - 4;
    const cls = (i) => (i < topicStart ? 'h-fixed' : i < topicEnd ? 'h-topic' : i < pidEnd ? 'h-pid' : i < tagStart ? 'h-pay' : 'h-tag');
    const hd = $('frameHex');
    hd.replaceChildren(...Array.from(frame, (b, i) => el('span', cls(i), hex(b))));
    $('frameMeta').textContent = `${frame.length} B, topic ${topic}`;
    const lg = $('frameLegend');
    lg.replaceChildren();
    for (const [c, label] of [['h-fixed', 'fixed header + remaining length'], ['h-topic', 'topic'], ['h-pid', 'packet id'], ['h-pay', 'payload'], ['h-tag', 'CMAC tag']]) {
      const s = el('span');
      s.append(el('i', `sw ${c}`), document.createTextNode(label));
      lg.append(s);
    }
    $('packetMeta').textContent = `20 B payload · ${frame.length} B MQTT frame · a VGA JPEG would be ~60-100 KB`;
  }

  /* ---------------------------------------------------------------- 05 uplink */
  const SVGNS = 'http://www.w3.org/2000/svg';
  const LAYER_COLOR = { AT: '#94a3b8', PHY: '#a78bfa', RRC: '#a78bfa', NAS: '#f472b6', TCP: '#fbbf24', TLS: '#34d399', MQTT: '#38bdf8', APP: '#94a3b8' };
  const svgEl = (tag, attrs, text) => {
    const e = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function buildSequence(session) {
    const svg = $('sequence');
    const lanes = link.LANES;
    const left = 72, right = 24, top = 54, rowH = 22, width = 900;
    const laneX = (id) => left + ((width - left - right) / (lanes.length - 1)) * lanes.findIndex((l) => l.id === id);
    const height = top + session.events.length * rowH + 16;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.style.height = `${height}px`;
    svg.replaceChildren();
    const defs = svgEl('defs', {});
    for (const [layer, color] of Object.entries(LAYER_COLOR)) {
      const m = svgEl('marker', { id: `ah-${layer}`, viewBox: '0 0 8 8', refX: '7', refY: '4', markerWidth: '7', markerHeight: '7', orient: 'auto-start-reverse' });
      m.append(svgEl('path', { d: 'M0,0 L8,4 L0,8 z', fill: color }));
      defs.append(m);
    }
    svg.append(defs);
    for (const l of lanes) {
      const x = laneX(l.id);
      svg.append(svgEl('text', { x, y: 18, 'text-anchor': 'middle', class: 'lane-label' }, l.label));
      svg.append(svgEl('text', { x, y: 33, 'text-anchor': 'middle', class: 'lane-sub' }, l.sub));
      svg.append(svgEl('line', { x1: x, y1: 42, x2: x, y2: height - 8, class: 'lifeline' }));
    }
    const rows = session.events.map((e, i) => {
      const y = top + i * rowH + 12;
      const g = svgEl('g', { opacity: '0.12' });
      const color = LAYER_COLOR[e.layer];
      g.append(svgEl('text', { x: 8, y: y + 4, class: 'layer', fill: color }, e.layer));
      const x1 = laneX(e.from), x2 = laneX(e.to);
      const size = e.bytes ? ` (${e.bytes.length} B)` : e.size ? ` (${e.size} B)` : '';
      if (x1 === x2) {
        g.append(svgEl('text', { x: x1 + 8, y: y + 4, class: 'msg' }, e.label));
      } else {
        g.append(svgEl('line', { x1, y1: y, x2, y2: y, class: 'arrow', stroke: color, 'marker-end': `url(#ah-${e.layer})` }));
        const mid = (x1 + x2) / 2;
        g.append(svgEl('text', { x: mid, y: y - 4, 'text-anchor': 'middle', class: 'msg' }, e.label + size));
      }
      svg.append(g);
      return { g, y };
    });
    return rows;
  }

  function appendAt(line) {
    const pre = $('atLog');
    const span = el('span', line.startsWith('>') ? 'tx' : 'rx', line + '\n');
    pre.append(span);
    pre.scrollTop = pre.scrollHeight;
  }

  async function uplink() {
    const run = state.run;
    const { acct, topic, packet } = state.sealed;
    stage('uplink', 'active', 'attaching');
    const session = link.buildSession({ clientId: acct.label, topic, payload: packet, apn: 'internet.mtn.rw' });
    // The modem must carry exactly the frame built in stage 04.
    const pubEvent = session.events.find((e) => e.publish);
    pubEvent.bytes = state.publish;
    const rows = buildSequence(session);
    $('atLog').replaceChildren();
    $('replayBtn').disabled = true;
    $('uplinkMeta').textContent = `${session.events.length} messages · ~${(session.totalMs / 1000).toFixed(1)} s nominal · ${fmt(session.airBytes)} B on the bearer`;
    const wrap = document.querySelector('.sequence-wrap');
    const speed = Number($('speedSelect').value);
    let delivered = null;
    for (let i = 0; i < session.events.length; i++) {
      if (run !== state.run) return;
      const e = session.events[i];
      rows[i].g.setAttribute('opacity', '1');
      if (rows[i].y > wrap.scrollTop + wrap.clientHeight - 40) wrap.scrollTop = rows[i].y - wrap.clientHeight / 2;
      if (e.at) appendAt(e.at);
      stage('uplink', 'active', e.layer);
      if (e.publish) delivered = cloud.ingest(e.bytes); // broker hands the frame to the back end on receipt
      await new Promise((r) => setTimeout(r, Math.max(8, e.dur / speed)));
    }
    if (run !== state.run) return;
    stage('uplink', 'done', `${(session.totalMs / 1000).toFixed(1)} s`);
    $('replayBtn').disabled = false;
    renderCloud(await delivered);
  }

  /* ---------------------------------------------------------------- 06 cloud + phone */
  function renderAccount() {
    const a = currentAccount();
    const last = a.readings[a.readings.length - 1];
    setKv($('accountInfo'), [
      ['Meter', a.label],
      ['Customer', a.customer],
      ['Zone', a.zone],
      ['SMS recipient', a.msisdn],
      ['Last accepted reading', last ? `${fmtM3(last.m3)} m³` : 'none'],
      ['Readings on ledger', String(a.readings.length)],
    ]);
    const tbody = $('ledger').tBodies[0];
    tbody.replaceChildren();
    for (const r of [...a.readings].reverse()) {
      const tr = el('tr');
      tr.append(el('td', '', utc(r.epoch)), el('td', 'r', fmtM3(r.m3)), el('td', 'r', fmtM3(r.usedM3)),
        el('td', 'r', r.perDay !== undefined ? r.perDay.toFixed(2) : '—'), el('td', 'r', r.billToDate !== undefined ? fmt(r.billToDate) : '—'),
        el('td', `st-${r.status}`, r.status));
      tbody.append(tr);
    }
    if (!a.readings.length) {
      const tr = el('tr');
      const td = el('td', 'muted', 'No readings yet for this meter.');
      td.colSpan = 6;
      tr.append(td);
      tbody.append(tr);
    }
  }

  function renderCloud(entry) {
    stage('cloud', entry.decision === 'rejected' ? 'failed' : 'done', entry.decision);
    const d = $('decision');
    d.className = `decision ${entry.decision}`;
    d.textContent = entry.decision === 'rejected' ? `REJECTED: ${entry.reason}` : entry.decision.toUpperCase();
    $('ingestSteps').replaceChildren(...entry.steps.map((s) => el('li', '', s)));
    renderAccount();
    if (entry.sms) showSms(entry.sms, entry.reading.epoch);
  }

  function showSms(text, epoch) {
    const body = $('phoneBody');
    body.replaceChildren(el('div', 'from', 'WASAC'), el('div', 'when', `${localClock(epoch)} CAT`), el('div', '', text));
    $('phoneClock').textContent = localClock(epoch);
    $('smsMeta').textContent = `${text.length}/160 chars · 1 SMS`;
    const phone = $('phone');
    phone.classList.remove('incoming');
    void phone.offsetWidth;
    phone.classList.add('incoming');
  }

  function clearDownstream() {
    state.wheels = null;
    $('tiles').replaceChildren(el('p', 'placeholder', 'Waiting for wheel regions.'));
    $('odometer').replaceChildren();
    $('readingValue').textContent = '—';
    $('readingInfo').replaceChildren();
  }

  /* ---------------------------------------------------------------- boot */
  function boot() {
    initModel();
    initAccounts();
    initAcquisition();
    initLocalisation();
    renderAccount();
    $('replayBtn').addEventListener('click', () => { if (state.sealed) { state.run++; uplink(); } });
    if (!window.crypto || !crypto.subtle) {
      $('modelStatus').textContent = 'WebCrypto unavailable: open over https or http://localhost';
      $('modelStatus').classList.add('error');
    }
  }

  boot();
})();
