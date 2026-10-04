/*
 * Cellular uplink session model: SIMCom SIM7600 (LTE Cat-1, 3G fallback)
 * driven over UART AT commands, attaching to an eNodeB and publishing over
 * MQTT/TLS to the WASAC broker.
 *
 * Produces an ordered event timeline for visualisation. Radio and core
 * procedures are summarised per 3GPP TS 36.331 (RRC) and TS 24.301 (NAS);
 * timings are nominal for good coverage. MQTT frames are the real bytes from
 * js/core/mqtt.js; TLS record sizes are representative.
 */
(function (root, factory) {
  const mqtt = typeof module === 'object' && module.exports ? require('../core/mqtt.js') : root.AMR.mqtt;
  const api = factory(mqtt);
  if (typeof module === 'object' && module.exports) module.exports = api;
  else (root.AMR = root.AMR || {}).cellularLink = api;
})(typeof self !== 'undefined' ? self : globalThis, function (mqtt) {
  'use strict';

  const LANES = [
    { id: 'mcu', label: 'ESP32-S3', sub: 'edge node' },
    { id: 'modem', label: 'SIM7600', sub: 'LTE Cat-1 modem' },
    { id: 'enb', label: 'eNodeB', sub: 'cell site (BTS)' },
    { id: 'epc', label: 'EPC core', sub: 'MME / S-GW / P-GW' },
    { id: 'broker', label: 'WASAC broker', sub: 'MQTT over TLS :8883' },
  ];

  const BROKER = 'mqtts://telemetry.wasac.example:8883';

  /**
   * @param {{clientId:string, topic:string, payload:Uint8Array, apn:string}} s
   * @returns {{events:Array, frames:object, totalMs:number}}
   */
  function buildSession(s) {
    const frames = {
      connect: mqtt.connect({ clientId: s.clientId, username: s.clientId, password: 'device-token', keepAlive: 60 }),
      connack: mqtt.connack(false, 0),
      publish: mqtt.publish({ topic: s.topic, payload: s.payload, qos: 1, packetId: 1 }),
      puback: mqtt.puback(1),
      disconnect: mqtt.disconnect(),
    };
    const ev = [];
    let t = 0;
    const add = (dur, layer, from, to, label, extra = {}) => { ev.push({ t, dur, layer, from, to, label, ...extra }); t += dur; };
    const at = (cmd, reply, dur) => {
      add(Math.round(dur * 0.4), 'AT', 'mcu', 'modem', cmd, { at: `> ${cmd}` });
      add(Math.round(dur * 0.6), 'AT', 'modem', 'mcu', reply, { at: `< ${reply}` });
    };

    // 1. Modem bring-up
    at('AT', 'OK', 60);
    at('AT+CPIN?', '+CPIN: READY', 80);
    at('AT+CSQ', '+CSQ: 21,99', 80);

    // 2. LTE cell search and RRC connection (random access, Msg1-Msg5)
    add(420, 'PHY', 'enb', 'modem', 'PSS/SSS sync, MIB, SIB1/SIB2', { note: 'cell search: EARFCN 1300, PCI 214' });
    add(30, 'RRC', 'modem', 'enb', 'PRACH preamble (Msg1)');
    add(25, 'RRC', 'enb', 'modem', 'Random Access Response (Msg2)');
    add(30, 'RRC', 'modem', 'enb', 'RRCConnectionRequest (Msg3)');
    add(40, 'RRC', 'enb', 'modem', 'RRCConnectionSetup (Msg4)');
    add(30, 'RRC', 'modem', 'enb', 'RRCConnectionSetupComplete + Attach Request');

    // 3. NAS attach: EPS-AKA authentication, security, default bearer
    add(60, 'NAS', 'enb', 'epc', 'Initial UE Message (Attach Request)');
    add(90, 'NAS', 'epc', 'modem', 'Authentication Request (RAND, AUTN)');
    add(60, 'NAS', 'modem', 'epc', 'Authentication Response (RES)');
    add(50, 'NAS', 'epc', 'modem', 'Security Mode Command');
    add(40, 'NAS', 'modem', 'epc', 'Security Mode Complete');
    add(120, 'NAS', 'epc', 'modem', 'Attach Accept, default EPS bearer', { note: `APN ${s.apn}, IPv4 10.64.18.37` });
    add(40, 'NAS', 'modem', 'epc', 'Attach Complete');
    at('AT+CEREG?', '+CEREG: 0,1', 60);
    at(`AT+CGDCONT=1,"IP","${s.apn}"`, 'OK', 60);
    at('AT+CMQTTSTART', '+CMQTTSTART: 0', 120);
    at(`AT+CMQTTACCQ=0,"${s.clientId}",1`, 'OK', 60);
    at(`AT+CMQTTCONNECT=0,"${BROKER}",60,1`, 'OK', 60);

    // 4. TCP three-way handshake through the P-GW
    add(70, 'TCP', 'modem', 'broker', 'SYN', { size: 60 });
    add(70, 'TCP', 'broker', 'modem', 'SYN-ACK', { size: 60 });
    add(10, 'TCP', 'modem', 'broker', 'ACK', { size: 52 });

    // 5. TLS 1.2 (ECDHE-ECDSA-AES128-GCM-SHA256), mutual authentication
    add(80, 'TLS', 'modem', 'broker', 'ClientHello', { size: 196 });
    add(110, 'TLS', 'broker', 'modem', 'ServerHello, Certificate, ServerKeyExchange, CertificateRequest', { size: 1412 });
    add(160, 'TLS', 'modem', 'broker', 'Certificate, ClientKeyExchange, CertificateVerify, Finished', { size: 742 });
    add(80, 'TLS', 'broker', 'modem', 'ChangeCipherSpec, Finished', { size: 51 });

    // 6. MQTT session: real control-packet bytes
    add(70, 'MQTT', 'modem', 'broker', 'CONNECT', { bytes: frames.connect });
    add(60, 'MQTT', 'broker', 'modem', 'CONNACK (accepted)', { bytes: frames.connack });
    add(40, 'AT', 'modem', 'mcu', '+CMQTTCONNECT: 0,0', { at: '< +CMQTTCONNECT: 0,0' });
    at(`AT+CMQTTTOPIC=0,${s.topic.length}`, '>', 30);
    at(`AT+CMQTTPAYLOAD=0,${s.payload.length}`, '>', 30);
    at('AT+CMQTTPUB=0,1,60', 'OK', 40);
    add(80, 'MQTT', 'modem', 'broker', `PUBLISH QoS 1, ${s.topic}`, { bytes: frames.publish, publish: true });
    add(70, 'MQTT', 'broker', 'modem', 'PUBACK', { bytes: frames.puback, delivered: true });
    add(30, 'AT', 'modem', 'mcu', '+CMQTTPUB: 0,0', { at: '< +CMQTTPUB: 0,0' });

    // 7. Teardown and power down
    add(40, 'MQTT', 'modem', 'broker', 'DISCONNECT', { bytes: frames.disconnect });
    at('AT+CMQTTDISC=0,60', '+CMQTTDISC: 0,0', 60);
    at('AT+CMQTTREL=0', 'OK', 30);
    at('AT+CMQTTSTOP', '+CMQTTSTOP: 0', 60);
    at('AT+CPOF', 'OK', 80);
    add(1, 'APP', 'mcu', 'mcu', 'Deep sleep until next slot');

    const airBytes = ev.reduce((sum, e) => sum + (e.bytes ? e.bytes.length : e.size || 0), 0);
    return { events: ev, frames, totalMs: t, airBytes };
  }

  return { LANES, BROKER, buildSession };
});
