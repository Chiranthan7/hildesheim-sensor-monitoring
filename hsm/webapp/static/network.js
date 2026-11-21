/* Radio network features: sensor focus mode, "Improve network" page (weak-signal map, new-gateway simulator,
   buildings) and the visit-route planner. All signal values come from measured messages (/api/network). */

const SIG = {
  strong: {t: 'Strong', c: '#1a9850', r: 3}, ok: {t: 'Good', c: '#66bd63', r: 2},
  weak: {t: 'Weak', c: '#f39c12', r: 1}, very_weak: {t: 'Very weak', c: '#d73027', r: 0}, none: {t: 'No signal data', c: '#9aa4b0', r: -1}};
const sigOf = snr => snr == null ? 'none' : snr >= 0 ? 'strong' : snr >= -7.5 ? 'ok' : snr >= -15 ? 'weak' : 'very_weak';
const pct = x => Math.round(100 * x) + '%';
const km = m => m >= 1000 ? (m / 1000).toFixed(1).replace('.0', '') + ' km' : Math.round(m / 10) * 10 + ' m';
const gwName = id => id.startsWith('unlisted_') ? 'Gateway ' + id.slice(9).replace('_', ' / ') : id;
function hav(a, b, c, d) {
  const r = Math.PI / 180, x = Math.sin((c - a) * r / 2) ** 2 + Math.cos(a * r) * Math.cos(c * r) * Math.sin((d - b) * r / 2) ** 2;
  return 12742000 * Math.asin(Math.sqrt(x));
}
let NET = null;
async function net() { if (!NET) NET = await api('/api/network'); return NET; }

// ---------------------------------------------------------------- 1. sensor focus mode (main map)
map.createPane('focus'); map.getPane('focus').style.zIndex = 640;          // above sensors, below gateways
const focusLayer = L.layerGroup().addTo(map);
function signalKV(id) {                       // one line for the sensor panel
  const s = NET && NET.sensors[id]; if (!s) return '';
  return `<b>Radio signal</b><span><span class="sigdot" style="background:${SIG[s.cls].c}"></span>${SIG[s.cls].t} · heard regularly by ${s.n_gw} gateway${s.n_gw > 1 ? 's' : ''}</span>`;
}
function dimGateways() {
  for (const [id, m] of Object.entries(S.gwMarkers)) m.setOpacity(S.focus && !S.focusGws.has(id) ? .25 : 1);
}
function clearFocus() {
  focusLayer.clearLayers(); S.focus = null; S.focusGws = new Set();
  map.getContainer().classList.remove('focus-mode'); dimGateways();
}
async function focusSensor(c, refit) {
  const N = await net(); const same = S.focus === c.id;
  focusLayer.clearLayers();
  if (c.lat == null) { clearFocus(); return; }
  S.focus = c.id; S.focusGws = new Set(); map.getContainer().classList.add('focus-mode');
  L.marker([c.lat, c.lon], {pane: 'focus', interactive: false, icon: L.divIcon({className: '', iconSize: [44, 58], iconAnchor: [22, 29],
    html: `<div class="mk ${c.light} focusmk"><div class="bub">${ico(c.icon_key, 28)}</div></div>`})}).addTo(focusLayer);
  const pts = [[c.lat, c.lon]], s = N.sensors[c.id];
  const links = s ? s.links : (c.gateway && N.gateways[c.gateway] ? [{gw: c.gateway, cls: 'none', share: 1}] : []);
  for (const l of links) {
    const g = N.gateways[l.gw]; if (!g) continue;
    S.focusGws.add(l.gw); pts.push([g.lat, g.lon]);
    L.polyline([[c.lat, c.lon], [g.lat, g.lon]], {pane: 'focus', color: SIG[l.cls].c, weight: 3 + 7 * l.share, opacity: .95, dashArray: l.share < .2 ? '7 7' : null, lineCap: 'round'})
      .bindTooltip(`<b>${ico('gateway', 16)} ${esc(gwName(l.gw))}</b><br>${l.cls === 'none' ? 'Main gateway (no signal strength in the data)' : `${SIG[l.cls].t} signal · receives ${pct(l.share)} of its messages`}`, {sticky: true})
      .addTo(focusLayer);
  }
  dimGateways();
  if (refit && !same && $('#view-map').classList.contains('active'))
    map.flyToBounds(pts, {paddingTopLeft: [50, 50], paddingBottomRight: [Math.min(600, innerWidth * .45), 50], maxZoom: 17, duration: .7});
}
// focus legend on the main map
const focusNote = L.control({position: 'topleft'});
focusNote.onAdd = () => { const d = L.DomUtil.create('div', 'legend focuslegend'); d.innerHTML =
  `<b>📶 Selected sensor</b><br>Lines go to the gateways that received its messages.<br>
   <span class="sw" style="background:${SIG.strong.c}"></span>Strong <span class="sw" style="background:${SIG.ok.c}"></span>Good
   <span class="sw" style="background:${SIG.weak.c}"></span>Weak <span class="sw" style="background:${SIG.very_weak.c}"></span>Very weak<br>
   <span class="mute">Thick line = most messages · dashed = rarely</span>`; return d; };
const _focusCls = new MutationObserver(() => { const on = map.getContainer().classList.contains('focus-mode'); on ? focusNote.addTo(map) : focusNote.remove(); });
_focusCls.observe(map.getContainer(), {attributes: true, attributeFilter: ['class']});

// ---------------------------------------------------------------- 2. Improve network page
const NW = {map: null, heat: null, dots: null, gw: null, cand: null, lines: null, bld: null, bldData: null, bldBox: null, res: null};
const sensorsWithSignal = () => S.state.sensors.filter(c => c.lat != null && NET.sensors[c.id]);

function predictAt(lat, lon) {
  /* For every sensor with measured signal: expected SNR from a gateway at (lat, lon), using the measured
     within-sensor distance effect (dB per tenfold distance) relative to the sensor's best existing gateway. */
  const m = NET.model, out = {better: [], backup: [], gain: 0};
  for (const c of sensorsWithSignal()) {
    const s = NET.sensors[c.id]; if (s.best_d == null) continue;
    const d = Math.max(30, hav(lat, lon, c.lat, c.lon)); if (d > m.typical_range_m) continue;
    const pred = s.best_snr + m.snr_db_per_decade * Math.log10(d / Math.max(30, s.best_d)), gain = pred - s.best_snr;
    const now = s.cls, then = sigOf(pred);
    const row = {c, s, d, pred, gain, now, then};
    if ((now === 'weak' || now === 'very_weak') && gain >= 3 && SIG[then].r > SIG[now].r) { out.better.push(row); out.gain += gain; }
    else if (s.n_gw === 1 && pred >= -7.5) out.backup.push(row);
  }
  out.better.sort((a, b) => b.gain - a.gain); out.backup.sort((a, b) => a.d - b.d);
  out.score = 2 * out.better.length + out.backup.length;
  return out;
}
function bestSpot() {
  const cs = sensorsWithSignal(); if (!cs.length) return null;
  const lats = cs.map(c => c.lat), lons = cs.map(c => c.lon);
  const la0 = Math.min(...lats) - .005, la1 = Math.max(...lats) + .005, lo0 = Math.min(...lons) - .008, lo1 = Math.max(...lons) + .008;
  let best = null; const st = .0007;                                     // about 75 m steps
  for (let la = la0; la <= la1; la += st) for (let lo = lo0; lo <= lo1; lo += st * 1.6) {
    const r = predictAt(la, lo); const sc = r.score + r.gain / 100;
    if (!best || sc > best.sc) best = {la, lo, sc, r};
  }
  return best;
}
function gwPin() {
  return L.divIcon({className: '', iconSize: [70, 82], iconAnchor: [35, 70],
    html: `<div class="newgw"><div class="newgwbub">${ico('gateway', 34)}<span>NEW</span></div><div class="newgwtip"></div></div>`});
}
async function setupNetwork() {
  await net();
  if (NW.map) { NW.map.invalidateSize(); drawNetSensors(); evaluateCand(); return; }
  NW.map = L.map('netMap', {zoomControl: true}).setView([52.152, 9.95], 13);
  baseLayers(NW.map);
  NW.map.createPane('bld'); NW.map.getPane('bld').style.zIndex = 380;
  NW.map.createPane('lines'); NW.map.getPane('lines').style.zIndex = 420;
  NW.bld = L.layerGroup().addTo(NW.map); NW.lines = L.layerGroup().addTo(NW.map);
  NW.dots = L.layerGroup().addTo(NW.map); NW.gw = L.layerGroup().addTo(NW.map);
  const lg = L.control({position: 'bottomleft'});
  lg.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.innerHTML = `<b>Signal of each sensor</b><br>` +
    ['strong', 'ok', 'weak', 'very_weak', 'none'].map(k => `<span class="dot" style="background:${SIG[k].c}"></span> ${SIG[k].t}`).join('<br>') +
    `<br>${ico('gateway', 18)} Existing gateway`; return d; };
  lg.addTo(NW.map);
  for (const [id, g] of Object.entries(NET.gateways))
    L.marker([g.lat, g.lon], {icon: L.divIcon({className: '', iconSize: [34, 34], iconAnchor: [17, 17], html: `<div class="gwsmall">${ico('gateway', 24)}</div>`})})
      .bindTooltip(`<b>${esc(gwName(id))}</b><br>existing gateway`, {direction: 'top'}).addTo(NW.gw);
  drawNetSensors();
  const weak = sensorsWithSignal().filter(c => ['weak', 'very_weak'].includes(NET.sensors[c.id].cls));
  const start = weak.length ? [weak.reduce((a, c) => a + c.lat, 0) / weak.length, weak.reduce((a, c) => a + c.lon, 0) / weak.length] : [52.152, 9.95];
  NW.cand = L.marker(start, {draggable: true, autoPan: true, icon: gwPin(), zIndexOffset: 9000}).addTo(NW.map)
    .bindTooltip('Drag me to test a spot for a new gateway', {direction: 'top', offset: [0, -66]});
  let tm = null;
  NW.cand.on('drag', () => { clearTimeout(tm); tm = setTimeout(() => evaluateCand(true), 60); });
  NW.cand.on('dragend', () => evaluateCand());
  NW.map.on('click', e => { NW.cand.setLatLng(e.latlng); evaluateCand(); });
  const pts = sensorsWithSignal().map(c => [c.lat, c.lon]); if (pts.length) NW.map.fitBounds(pts, {padding: [30, 30]});
  evaluateCand();
}
function drawNetSensors() {
  NW.dots.clearLayers(); if (NW.heat) { NW.map.removeLayer(NW.heat); NW.heat = null; }
  const heatPts = [];
  for (const c of S.state.sensors) {
    if (c.lat == null) continue;
    const s = NET.sensors[c.id], k = s ? s.cls : 'none';
    const mob = ['Bigbelly', 'OttSensor'].includes(c.family);
    L.circleMarker([c.lat, c.lon], {radius: s ? 7 : 5, color: '#fff', weight: 2, fillColor: SIG[k].c, fillOpacity: s ? .95 : .6})
      .bindTooltip(`<b>${ico(c.icon_key, 16)} ${esc(c.measures)}</b> <span class="mute">${esc(c.id)}</span><br>` + (s
        ? `Signal: <b>${SIG[k].t}</b> · heard by ${s.n_gw} gateway${s.n_gw > 1 ? 's' : ''}${s.weak_share != null ? `<br>${pct(s.weak_share)} of its messages arrived with a weak signal` : ''}`
        : mob ? 'Uses the mobile phone network, not the LoRa gateways' : 'No signal strength in the data for this sensor'), {direction: 'top'})
      .on('click', () => openSensor(c.id)).addTo(NW.dots);
    if (s && s.weak_share != null && s.weak_share > .05) heatPts.push([c.lat, c.lon, Math.min(1, s.weak_share + (s.very_weak_share || 0))]);
  }
  NW.heat = L.heatLayer(heatPts, {radius: 40, blur: 32, maxZoom: 13, max: 1.6, minOpacity: .1, gradient: {.25: '#fee08b', .55: '#fc8d59', .85: '#d73027'}});
  if ($('#heatTgl').checked) NW.heat.addTo(NW.map);
  const all = sensorsWithSignal(), cnt = k => all.filter(c => NET.sensors[c.id].cls === k).length;
  const single = all.filter(c => NET.sensors[c.id].n_gw === 1).length;
  $('#netStats').innerHTML = `<div class="ns red"><b>${cnt('very_weak') + cnt('weak')}</b><span>sensors with a weak signal</span></div>
    <div class="ns amber"><b>${single}</b><span>depend on a single gateway</span></div>
    <div class="ns green"><b>${cnt('ok') + cnt('strong')}</b><span>good or strong signal</span></div>`;
  const m = NET.model;
  $('#netMethod').innerHTML = `<p>Based on <b>${S.state.sensors.length ? Object.keys(NET.sensors).length : 0} sensors</b> and every message they sent between ${esc(NET.period.first)} and ${esc(NET.period.last)}: which gateway received it and how clearly (SNR).</p>
    <p><b>Signal classes</b> use the LoRa radio limits from the chip maker (Semtech SX127x datasheet): <b>Good</b> = at least −7.5 dB (works at the fastest radio setting), <b>Weak</b> = −7.5 to −15 dB (needs slow settings, more battery), <b>Very weak</b> = below −15 dB (close to the −20 dB limit; messages get lost).</p>
    <p><b>New gateway estimate</b>: sensors heard by several gateways show how the signal changes with distance. On average it improves by about <b>${Math.abs(m.snr_db_per_decade)} dB each time the distance becomes 10× shorter</b> (±${m.se} dB; ${m.n_links} links of ${m.n_sensors} sensors). Typical error of one estimate: ±${m.residual_sd_db} dB. Only sensors within ${km(m.typical_range_m)} are considered (90% of the measured links are shorter).</p>
    <p>A sensor counts as <b>better</b> when its signal is weak today and the new gateway is expected to give at least 3 dB more and a better class. It gets a <b>backup</b> when it depends on one gateway today and the new one is expected to receive it with a good signal.</p>
    <p class="mute">This is an estimate from measurements, not a radio-coverage simulation. Buildings, trees and the antenna height matter a lot. Before buying, test the spot with a temporary gateway for a few days. Sensors without signal data (wind stations, water stations, waste bins, salinity probe) are not included.</p>`;
}
$('#heatTgl').onchange = e => { if (!NW.heat) return; e.target.checked ? NW.heat.addTo(NW.map) : NW.map.removeLayer(NW.heat); };
$('#bestSpotBtn').onclick = () => {
  const b = bestSpot(); if (!b) return;
  NW.cand.setLatLng([b.la, b.lo]); NW.map.flyTo([b.la, b.lo], 15, {duration: .7}); evaluateCand();
};
function evaluateCand(quick) {
  if (!NW.cand) return;
  const p = NW.cand.getLatLng(), r = predictAt(p.lat, p.lng); NW.res = r;
  NW.lines.clearLayers();
  for (const x of r.better) L.polyline([[p.lat, p.lng], [x.c.lat, x.c.lon]], {pane: 'lines', color: '#1a9850', weight: 3, opacity: .9}).addTo(NW.lines);
  for (const x of r.backup) L.polyline([[p.lat, p.lng], [x.c.lat, x.c.lon]], {pane: 'lines', color: '#2563eb', weight: 1.5, opacity: .45, dashArray: '5 6'}).addTo(NW.lines);
  if (quick) { $('#candHead') && ($('#candHead').innerHTML = candHead(r)); return; }
  if ($('#bldTgl').checked) loadBuildings(p);
  drawCandResult();
}
function candHead(r) {
  return `<div class="cbig"><span class="n good">${r.better.length}</span> sensor${r.better.length === 1 ? '' : 's'} get a <b>better signal</b></div>
          <div class="cbig"><span class="n backup">${r.backup.length}</span> sensor${r.backup.length === 1 ? '' : 's'} get a <b>backup gateway</b></div>`;
}
function drawCandResult() {
  const r = NW.res, p = NW.cand.getLatLng();
  const crossing = x => NW.bldData ? countCrossings(p, x.c) : null;
  const row = (x, kind) => {
    const n = crossing(x);
    return `<div class="crow" data-id="${esc(x.c.id)}"><span class="ic">${ico(x.c.icon_key, 26)}</span><div style="flex:1;min-width:0"><div class="nm">${esc(x.c.measures)} <span class="mute">${esc(x.c.id)}</span></div>
      <div class="cdet">${kind === 'better' ? `<span class="sigpill" style="background:${SIG[x.now].c}">${SIG[x.now].t}</span> → <span class="sigpill" style="background:${SIG[x.then].c}">${SIG[x.then].t}</span> <span class="mute">≈ +${x.gain.toFixed(0)} dB</span>`
        : `today only ${esc(gwName(x.s.best_gw))} · new one: <span class="sigpill" style="background:${SIG[x.then].c}">${SIG[x.then].t}</span>`}
        <span class="mute"> · ${km(x.d)} away${n != null ? ` · 🏢 ${n} building${n === 1 ? '' : 's'} in the way` : ''}</span></div></div></div>`;
  };
  $('#candResult').innerHTML = `<div class="card candcard"><h3>📍 New gateway at this spot</h3><div id="candHead">${candHead(r)}</div>
    ${r.better.length + r.backup.length ? '' : '<p class="mute">No sensor would clearly benefit here. Drag the pin closer to the red areas or press ✨ Suggest best spot.</p>'}
    ${r.better.length ? `<div class="csub">✅ Better signal</div>${r.better.map(x => row(x, 'better')).join('')}` : ''}
    ${r.backup.length ? `<div class="csub">🛟 Backup gateway <span class="mute">(they keep working if their only gateway fails)</span></div>${r.backup.map(x => row(x, 'backup')).join('')}` : ''}
    <div id="bldNote" class="mute" style="font-size:12px;margin-top:6px">${NW.bldMsg || ''}</div>
    <div class="actions" style="margin-top:8px"><a class="btn small green" target="_blank" rel="noopener" href="${gmaps(p.lat.toFixed(6), p.lng.toFixed(6))}">🧭 Directions to this spot</a>
    <span class="mute" style="font-size:12px;align-self:center">${p.lat.toFixed(5)}, ${p.lng.toFixed(5)}</span></div></div>`;
  $$('#candResult .crow').forEach(e => e.onclick = () => openSensor(e.dataset.id));
}

// ---- buildings (2D footprints from OpenStreetMap, loaded in the browser via the Overpass API)
const BLD_R = 800;                                                      // metres around the new gateway
async function loadBuildings(p) {
  if (NW.bldBox && hav(p.lat, p.lng, NW.bldBox[0], NW.bldBox[1]) < BLD_R / 3) { drawCandResult(); return; }
  const dLa = BLD_R / 111320, dLo = BLD_R / (111320 * Math.cos(p.lat * Math.PI / 180));
  const bb = [p.lat - dLa, p.lng - dLo, p.lat + dLa, p.lng + dLo].map(x => x.toFixed(5)).join(',');
  NW.bldMsg = 'Loading buildings…'; drawCandResult();
  try {
    const r = await fetch('https://overpass-api.de/api/interpreter', {method: 'POST', body: 'data=' + encodeURIComponent(`[out:json][timeout:25];way["building"](${bb});out geom;`)});
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    NW.bldData = j.elements.filter(e => e.geometry && e.geometry.length > 2).map(e => e.geometry.map(g => [g.lat, g.lon]));
    NW.bldBox = [p.lat, p.lng];
    NW.bldMsg = `🏢 ${NW.bldData.length} buildings loaded within ${km(BLD_R)} of the new gateway (OpenStreetMap). Buildings further away are not counted.`;
  } catch (e) { NW.bldData = null; NW.bldBox = null; NW.bldMsg = '⚠️ Building data could not be loaded right now (OpenStreetMap service not reachable). Nothing is counted.'; }
  drawBuildings(); drawCandResult();
}
function drawBuildings() {
  NW.bld.clearLayers();
  if (!$('#bldTgl').checked || !NW.bldData) return;
  for (const poly of NW.bldData) L.polygon(poly, {pane: 'bld', color: '#4b5563', weight: .6, fillColor: '#64748b', fillOpacity: .45, interactive: false}).addTo(NW.bld);
}
$('#bldTgl').onchange = e => { if (e.target.checked) loadBuildings(NW.cand.getLatLng()); else { NW.bld.clearLayers(); NW.bldData = null; NW.bldBox = null; NW.bldMsg = ''; drawCandResult(); } };
function segX(a, b, c, d) {                 // do segments a-b and c-d cross? points as [lat, lon]
  const o = (p, q, r) => Math.sign((q[1] - p[1]) * (r[0] - p[0]) - (q[0] - p[0]) * (r[1] - p[1]));
  return o(a, b, c) !== o(a, b, d) && o(c, d, a) !== o(c, d, b);
}
function inPoly(pt, poly) {
  let ins = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++)
    if ((poly[i][0] > pt[0]) !== (poly[j][0] > pt[0]) && pt[1] < (poly[j][1] - poly[i][1]) * (pt[0] - poly[i][0]) / (poly[j][0] - poly[i][0]) + poly[i][1]) ins = !ins;
  return ins;
}
function countCrossings(p, c) {
  const a = [p.lat, p.lng], b = [c.lat, c.lon]; let n = 0;
  const [la0, la1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])], [lo0, lo1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
  for (const poly of NW.bldData) {
    if (poly.every(q => q[0] < la0) || poly.every(q => q[0] > la1) || poly.every(q => q[1] < lo0) || poly.every(q => q[1] > lo1)) continue;
    if (inPoly(a, poly) || inPoly(b, poly)) continue;                    // the building the gateway / sensor sits on
    for (let i = 0; i < poly.length - 1; i++) if (segX(a, b, poly[i], poly[i + 1])) { n++; break; }
  }
  return n;
}

// ---------------------------------------------------------------- 3. visit-route planner ("Check these first")
const RT = {map: null, layer: null, me: null};
$$('.seg button').forEach(b => b.onclick = () => {
  $$('.seg button').forEach(x => x.classList.toggle('on', x === b));
  const route = b.dataset.mode === 'route';
  $('#firstListBox').classList.toggle('hidden', route); $('#routeBox').classList.toggle('hidden', !route);
  if (route) drawRoute();
});
$('#routeStart').onchange = () => drawRoute(); $('#routeWhich').onchange = () => drawRoute();
function orderStops(pts, start) {           // nearest neighbour, then 2-opt (open path, straight-line distances)
  const D = (p, q) => hav(p.lat, p.lon, q.lat, q.lon);
  let rest = pts.slice(), cur = start || rest.shift(), path = start ? [] : [cur];
  while (rest.length) { let k = 0; rest.forEach((p, i) => { if (D(cur, p) < D(cur, rest[k])) k = i; }); cur = rest.splice(k, 1)[0]; path.push(cur); }
  const full = start ? [start, ...path] : path, n = full.length; let imp = true;
  while (imp) { imp = false;                                             // first point stays fixed, the end is free
    for (let i = 1; i < n - 1; i++) for (let j = i + 1; j < n; j++) {
      const before = D(full[i - 1], full[i]) + (j < n - 1 ? D(full[j], full[j + 1]) : 0);
      const after = D(full[i - 1], full[j]) + (j < n - 1 ? D(full[i], full[j + 1]) : 0);
      if (after + 1 < before) { full.splice(i, j - i + 1, ...full.slice(i, j + 1).reverse()); imp = true; }
    }
  }
  return start ? full.slice(1) : full;
}
async function drawRoute() {
  if (!RT.map) {
    RT.map = L.map('routeMap').setView([52.152, 9.95], 13);
    baseLayers(RT.map);
    RT.layer = L.layerGroup().addTo(RT.map);
  }
  setTimeout(() => RT.map.invalidateSize(), 50);
  const onlyRed = $('#routeWhich').value === 'red';
  const need = S.state.sensors.filter(c => c.visit && (!onlyRed || c.light === 'red')), loc = need.filter(c => c.lat != null).sort((a, b) => (b.priority || 0) - (a.priority || 0));
  let start = null;
  if ($('#routeStart').value === 'me') {
    try { const pos = await new Promise((ok, no) => navigator.geolocation.getCurrentPosition(ok, no, {timeout: 8000}));
          start = {lat: pos.coords.latitude, lon: pos.coords.longitude, me: true}; }
    catch (e) { alertBox('Your location is not available (permission denied or not supported). Starting from the most urgent sensor instead.'); $('#routeStart').value = 'first'; }
  }
  const stops = loc.length ? orderStops(loc, start) : [];
  RT.layer.clearLayers();
  if (!stops.length) { $('#routeSummary').innerHTML = '<p>✅ No sensor needs a site visit right now.</p>'; $('#routeStops').innerHTML = ''; $('#routeLinks').innerHTML = ''; return; }
  const seq = start ? [start, ...stops] : stops;
  let dist = 0; for (let i = 1; i < seq.length; i++) dist += hav(seq[i - 1].lat, seq[i - 1].lon, seq[i].lat, seq[i].lon);
  L.polyline(seq.map(p => [p.lat, p.lon]), {color: '#2563eb', weight: 4, opacity: .75}).addTo(RT.layer);
  if (start) L.marker([start.lat, start.lon], {icon: L.divIcon({className: '', iconSize: [30, 30], iconAnchor: [15, 15], html: '<div class="stopnum me">●</div>'})}).bindTooltip('You are here').addTo(RT.layer);
  stops.forEach((c, i) => L.marker([c.lat, c.lon], {zIndexOffset: 1000 - i, icon: L.divIcon({className: '', iconSize: [30, 30], iconAnchor: [15, 15], html: `<div class="stopnum ${c.light}">${i + 1}</div>`})})
    .bindTooltip(`<b>${i + 1}. ${esc(c.measures)}</b><br>${esc(c.status)}<br><span class="mute">${esc(c.id)}</span>`, {direction: 'top'}).on('click', () => openSensor(c.id)).addTo(RT.layer));
  RT.map.fitBounds(seq.map(p => [p.lat, p.lon]), {padding: [40, 40]});
  const noGps = need.length - loc.length;
  $('#routeSummary').innerHTML = `<div class="rbig"><b>${stops.length}</b> stops · about <b>${km(dist)}</b></div>
    <div class="mute" style="font-size:12px">Straight-line distance; the road route is longer.${noGps ? ` ${noGps} more sensor${noGps > 1 ? 's need' : ' needs'} a visit but ${noGps > 1 ? 'have' : 'has'} no GPS position.` : ''}</div>`;
  const legs = []; const all = start ? [null, ...stops] : stops;          // null = current location (Google uses "my location")
  for (let i = 0; i < all.length - 1; i += 9) legs.push(all.slice(i, i + 10));
  const ll = p => `${p.lat},${p.lon}`;
  $('#routeLinks').innerHTML = legs.map((g, k) => {
    const o = g[0], dst = g[g.length - 1], w = g.slice(1, -1);
    const url = `https://www.google.com/maps/dir/?api=1${o ? '&origin=' + ll(o) : ''}&destination=${ll(dst)}${w.length ? '&waypoints=' + w.map(ll).join('%7C') : ''}&travelmode=driving`;
    const num = e => start ? e : e + 1, e0 = k * 9, e1 = e0 + g.length - 1;
    const a = Math.max(1, num(e0)), b = num(e1);
    return `<a class="btn green" target="_blank" rel="noopener" href="${url}">🧭 Open in Google Maps${legs.length > 1 ? ` · part ${k + 1} (stops ${a}–${b})` : ''}</a>`;
  }).join('');
  $('#routeStops').innerHTML = stops.map((c, i) => `<li data-id="${esc(c.id)}"><span class="stopnum ${c.light}">${i + 1}</span>${ico(c.icon_key, 24)}<div><b>${esc(c.measures)}</b><br><span class="mute">${esc(c.status)}</span></div></li>`).join('');
  $$('#routeStops li').forEach(e => e.onclick = () => openSensor(e.dataset.id));
}
