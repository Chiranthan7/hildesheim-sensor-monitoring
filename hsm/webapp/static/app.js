/* Smart Sensor Monitoring - interactive front end (Leaflet map + plain-language panels) */
const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => (s ?? '').toString().replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const api = async (u, opt) => { $('#loading').classList.remove('hidden'); try { const r = await fetch(u, opt); if (!r.ok) throw new Error(r.status); return await r.json(); } finally { $('#loading').classList.add('hidden'); } };
const S = {meta: null, state: null, asOf: null, hours: [], light: 'all', types: new Set(), q: '', markers: {}, gwMarkers: {}, chart: null, openId: null, openKind: null, playing: null};
const COLORS = {red: '#d32f2f', yellow: '#f2a900', green: '#2e9e5b', grey: '#9aa4b0'};
const ico = (key, px = 24, cls = '') => `<img class="ico ${cls}" src="/static/icons/${key}.svg" width="${px}" height="${px}" alt="">`;
const gmaps = (lat, lon) => `https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}&travelmode=driving`;

// ---------------------------------------------------------------- map
const map = L.map('map', {zoomControl: true}).setView([52.152, 9.95], 13);
// base maps: TopPlusOpen Light Grey (BKG, official German map service, WMS) as default, other TopPlusOpen variants; OpenStreetMap and satellite as alternatives
const BKG_WMS = 'https://sgx.geodatenzentrum.de/wms_topplus_open';
const BKG_ATTR = 'Kartendarstellung: © <a href="https://www.bkg.bund.de" target="_blank" rel="noopener">BKG</a> (2026) <a href="https://www.govdata.de/dl-de/by-2-0" target="_blank" rel="noopener">dl-de/by-2-0</a>, <a href="https://sgx.geodatenzentrum.de/web_public/gdz/datenquellen/datenquellen_topplusopen.html" target="_blank" rel="noopener">Datenquellen</a>';
const bkg = layer => L.tileLayer.wms(BKG_WMS, {layers: layer, format: 'image/png', transparent: false, version: '1.3.0', maxZoom: 19, attribution: BKG_ATTR});
function baseLayers(m) {             // adds the default base map to map m and a switcher; returns the layers
  const b = {'Light grey (TopPlusOpen)': bkg('web_light_grau'), 'Light (TopPlusOpen)': bkg('web_light'), 'Colour (TopPlusOpen)': bkg('web'), 'Grey (TopPlusOpen)': bkg('web_grau'),
    'OpenStreetMap': L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© OpenStreetMap contributors'}),
    'Satellite': L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {maxZoom: 19, attribution: 'Imagery © Esri'})};
  b['Light grey (TopPlusOpen)'].addTo(m); L.control.layers(b).addTo(m); return b;     // default: light grey, so the coloured markers stand out
}
baseLayers(map);
const RANK = {red: 3, yellow: 2, grey: 1, green: 0};
const sensorLayer = L.markerClusterGroup({disableClusteringAtZoom: 18, maxClusterRadius: 70, spiderfyOnMaxZoom: true, showCoverageOnHover: false,
  iconCreateFunction: cl => {
    const ms = cl.getAllChildMarkers(); const worst = ms.reduce((w, m) => RANK[m.options.light] > RANK[w] ? m.options.light : w, 'green');
    const kinds = {}; ms.forEach(m => kinds[m.options.kind] = (kinds[m.options.kind] || 0) + 1);
    const ks = Object.entries(kinds).sort((a, b) => b[1] - a[1]);
    const nr = ms.filter(m => m.options.light === 'red').length;
    const icons = ks.slice(0, 2).map(([k, n]) => `<span>${ico(k, 15)}${n > 1 ? '×' + n : ''}</span>`).join('') + (ks.length > 2 ? '<span>+</span>' : '');
    return L.divIcon({className: '', iconSize: [62, 50], iconAnchor: [31, 25],
      html: `<div class="cluster ${worst}" title="${ms.length} sensors${nr ? ', ' + nr + ' need action' : ''}. Click to zoom in."><b>${ms.length}</b><div class="kinds">${icons}</div></div>`});
  }}).addTo(map);
const zoomCls = () => map.getContainer().classList.toggle('zoomed-out', map.getZoom() < 13); map.on('zoomend', zoomCls); zoomCls();
map.createPane('gateways'); map.getPane('gateways').style.zIndex = 650;      // gateways above sensor markers and groups
const gwLayer = L.layerGroup([], {pane: 'gateways'}).addTo(map);
const legend = L.control({position: 'bottomleft'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.innerHTML = '<span class="dot red"></span> Action needed<br><span class="dot yellow"></span> Attention soon<br><span class="dot green"></span> Working fine<br><span class="dot grey"></span> No information<br>' + ico('gateway', 20) + ' Gateway (receiver)'; return d; };
legend.addTo(map);

function batteryHTML(b, big) {
  if (b.kind === 'solar' || b.kind === 'sealed') {
    const k = b.kind === 'solar' ? 'sun' : 'lock';
    return big ? `<div class="bigbat"><div class="powerbig">${ico(k, 56)}</div><div><div class="val" style="font-size:22px">${esc(b.text)}</div><div class="lab">battery not reported</div></div></div>`
               : `<div class="bat power" title="${esc(b.note)}">${ico(k, 13)}</div>`;
  }
  const lv = 'lv-' + (b.level || 'unknown') + (b.stale ? ' stale' : '');
  if (big) return `<div class="bigbat ${lv}"><div class="batbody"><div class="batfill" style="--w:${b.fill}%"></div></div><div><div class="val">${esc(b.label)}</div><div class="lab">${esc(b.text)}</div></div></div>`;
  return `<div class="bat ${lv}"><div class="batbody"><div class="batfill" style="--w:${b.fill}%"></div></div><span>${esc(b.label)}</span></div>`;
}
function markerIcon(c) {
  return L.divIcon({className: '', iconSize: [44, 58], iconAnchor: [22, 29], popupAnchor: [0, -26],
    // on the map: battery level only for sensors that report one; no solar / sealed-battery / unknown symbols
    html: `<div class="mk ${c.light}"><div class="pulse"></div><div class="bub">${ico(c.icon_key, 28)}</div>${['band', 'percent', 'lithium'].includes(c.battery.kind) ? batteryHTML(c.battery) : ''}</div>`});
}
function visible(c) {
  if (S.light !== 'all' && c.light !== S.light) return false;
  if (S.types.size && !S.types.has(c.family)) return false;
  if (S.q) { const q = S.q.toLowerCase(); if (!(c.id.toLowerCase().includes(q) || c.measures.toLowerCase().includes(q) || c.family.toLowerCase().includes(q) || c.status.toLowerCase().includes(q))) return false; }
  return true;
}
function drawMap() {
  sensorLayer.clearLayers(); gwLayer.clearLayers(); S.markers = {}; S.gwMarkers = {};
  for (const c of S.state.sensors) {
    if (c.lat == null || !visible(c)) continue;
    const m = L.marker([c.lat, c.lon], {icon: markerIcon(c), light: c.light, kind: c.icon_key, riseOnHover: true, zIndexOffset: c.light === 'red' ? 1000 : c.light === 'yellow' ? 500 : 0});
    m.bindTooltip(`<b>${ico(c.icon_key, 18)} ${esc(c.measures)}</b><br>${esc(c.status)}<br><span style="color:#666">${esc(c.id)}</span>`, {direction: 'top', offset: [0, -24]});
    m.on('click', () => openSensor(c.id)); m.addTo(sensorLayer); S.markers[c.id] = m;
  }
  for (const g of S.state.gateways) {
    if (g.lat == null) continue;
    const m = L.marker([g.lat, g.lon], {pane: 'gateways', zIndexOffset: 5000, riseOnHover: true, icon: L.divIcon({className: '', iconSize: [64, 86], iconAnchor: [32, 29],
      html: `<div class="gwwrap"><div class="gwmk ${g.light}">${ico('gateway', 40)}</div><div class="gwlab">${esc(g.name.split(' / ')[0].replace('Unlisted gateway ', 'Gateway ').slice(0, 18))}</div></div>`})});
    m.bindTooltip(`<b>${ico('gateway', 18)} ${esc(g.name)}</b><br>${esc(g.status)}`, {direction: 'top'}); m.on('click', () => openGateway(g.id)); m.addTo(gwLayer); S.gwMarkers[g.id] = m;
  }
}
function itemHTML(c) {
  return `<div class="item" data-id="${esc(c.id)}"><span class="dot ${c.light}"></span><span class="ic">${ico(c.icon_key, 30)}</span><div style="flex:1;min-width:0"><div class="nm">${esc(c.measures)}</div><div class="st">${esc(c.status)} · ${esc(c.id)}</div></div>${batteryHTML(c.battery)}</div>`;
}
function drawChipCounts() {
  if (!S.groups) return;
  $$('#typeChips .chip').forEach(c => {
    const g = S.groups[+c.dataset.g]; const all = S.state.sensors.filter(x => g.families.includes(x.family));
    const noGps = all.filter(x => x.lat == null).length;
    c.querySelector('.cnt').textContent = all.length; c.title = noGps ? `${noGps} of ${all.length} have no GPS position in the data (listed under "Sensors without GPS position")` : `${all.length} sensors`;
  });
}
function drawList() {
  const L_ = S.state.sensors.filter(c => c.lat != null && visible(c)).sort((a, b) => (b.priority || 0) - (a.priority || 0));
  const U0 = S.state.unlocated.filter(visible).length;
  $('#sensorList').innerHTML = L_.map(itemHTML).join('') || `<p class="mute">${U0 ? 'None of these sensors has a GPS position. See the list below.' : 'No sensors match.'}</p>`;
  $('#listCount').textContent = `${L_.length} sensors on the map`;
  const U = S.state.unlocated.filter(visible); $('#unlocN').textContent = U.length; $('#unlocList').innerHTML = U.map(itemHTML).join('');
  if (S.types.size || S.q) $('.unloc').open = U.length > 0;
  // note on the map when some or all of the selected sensors have no position in the data
  const note = $('#mapNote');
  if (U.length && (S.types.size || S.q || S.light !== 'all')) {
    note.innerHTML = L_.length ? `📍 ${U.length} of the selected sensors have <b>no GPS position in the data</b>. They are listed on the left under "Sensors without GPS position".`
                               : `📍 The selected sensors have <b>no GPS position in the data</b>, so they cannot be shown on the map. They are listed on the left; their Directions button leads to the area of the gateway that receives them.`;
    note.classList.remove('hidden');
  } else note.classList.add('hidden');
  $$('#sensorList .item, #unlocList .item').forEach(e => e.onclick = () => openSensor(e.dataset.id, true));
}
function drawTiles() {
  const c = S.state.counts; $('#cRed').textContent = c.red; $('#cYellow').textContent = c.yellow; $('#cGreen').textContent = c.green;
  $('#cAll').textContent = S.state.sensors.length; $('#cAlerts').textContent = S.state.open_alerts;
  $$('.tile[data-filter]').forEach(t => t.classList.toggle('sel', t.dataset.filter === S.light));
}
function drawFirst() {
  const P = S.state.priority;
  $('#firstList').innerHTML = P.map((c, i) => `<div class="prow ${c.light}" data-open="${esc(c.id)}">
      <span class="rank">${i + 1}</span><span class="dot ${c.light}"></span>${ico(c.icon_key, 34)}
      <span class="pmain"><b>${esc(c.measures)}</b><span class="mute pid">${esc(c.id)}</span></span>
      <span class="pstat">${esc(c.status)}</span>
      <button class="btn small">Details</button></div>`).join('') || '<p>Nothing to check right now. 🎉</p>';
  $$('#firstList [data-open]').forEach(r => r.onclick = () => openSensor(r.dataset.open, true));
}
function drawGateways() {
  $('#gwList').innerHTML = S.state.gateways.map(g => `<div class="card fcard ${g.light}"><div class="top1">${ico('gateway', 40)}<div style="flex:1"><b>${esc(g.name)}</b><div class="mute" style="font-size:13px">${g.official ? 'official gateway' : 'not in the official list'}</div></div></div>
      <div><b>${esc(g.status)}</b></div><div class="mute">Last message received ${esc(g.last_heard)}</div>
      <div class="actions"><button class="btn small" data-gw="${esc(g.id)}">Details</button>${g.lat != null ? `<a class="btn small green" target="_blank" rel="noopener" href="${gmaps(g.lat, g.lon)}">🧭 Directions</a>` : ''}</div></div>`).join('');
  $$('#gwList [data-gw]').forEach(b => b.onclick = () => openGateway(b.dataset.gw));
}

// ---------------------------------------------------------------- detail drawer
function openDrawer(html) { $('#drawerBody').innerHTML = html; $('#drawer').classList.add('open'); }
$('#closeDrawer').onclick = () => { $('#drawer').classList.remove('open'); S.openId = null; if (window.clearFocus) clearFocus(); };
function predHTML(p) {
  const col = p.level === 'High' ? COLORS.red : p.level === 'Medium' ? COLORS.yellow : COLORS.green;
  return `<div class="pred"><div class="q">${esc(p.q)}</div><div class="meter"><div style="width:${Math.max(2, p.p * 100)}%;background:${col}"></div></div><span class="lvl ${p.level}">${p.level}</span> <span class="mute">(${p.pct})</span></div>`;
}
async function openSensor(id, fly) {
  S.openId = id; S.openKind = 'sensor'; await net();
  const d = await api(`/api/sensor/${encodeURIComponent(id)}?as_of=${encodeURIComponent(S.asOf)}`); const c = d.card, s = d.status;
  if (fly && c.lat != null) showTab('map');
  focusSensor(c, true);                       // fade the others, draw lines to the gateways that received it
  const dir = d.directions.lat != null ? `<a class="btn green" target="_blank" rel="noopener" href="${gmaps(d.directions.lat, d.directions.lon)}" title="${esc(d.directions.label)}">🧭 Directions${d.directions.approximate ? ' (area)' : ''}</a>` : '';
  const ser = d.series; const ctabs = [['delivered', 'Data received'], ['meas', 'Measurements'], ['battery', 'Battery'], ['risk7', 'Breakdown risk']]
    .filter(([k]) => k === 'delivered' || (k === 'meas' && ser.m1_label && ser.m1.some(x => x != null)) || (k === 'battery' && (ser.battery_pct.some(x => x != null) || ser.battery_v.some(x => x != null))) || (k === 'risk7' && ser.risk7.some(x => x != null)));
  openDrawer(`
    <div class="dhead">${ico(c.icon_key, 56)}<div><div class="t1">${esc(c.measures)}</div><div class="t2">${esc(c.manufacturer)} · ${esc(c.model)}<br>ID ${esc(c.id)}</div></div></div>
    <div class="actions actions-top">${dir}${c.lat != null ? `<button class="btn ghost" id="flyBtn">📍 Show on map</button>` : ''}</div>
    <div class="statuscard ${c.light}"><div class="sc-head"><div class="b1">${esc(c.light_text)}: ${esc(s.title)}</div><div>${esc(s.text)}</div></div>
      ${s.cause ? `<div class="sc-body"><div class="sc-row"><span class="sc-lab">🔎 Why</span><span>${esc(s.cause)}${s.cause_certainty ? ` <span class="mute">(${esc(s.cause_certainty)})</span>` : ''}</span></div>
        <div class="sc-row"><span class="sc-lab">👉 What to do</span><b>${esc(s.todo)}</b></div>
        <div class="sc-row"><span class="sc-lab">🚐 Site visit</span>${s.visit ? '<span class="badge visit">Needed</span>' : '<span class="badge novisit">Not needed</span>'}</div></div>` : ''}</div>
    ${d.directions.approximate && d.directions.lat != null ? `<p class="mute" style="font-size:13px">${esc(d.directions.label)}</p>` : ''}
    ${d.moved ? `<div class="sec"><h3>🚚 This sensor has been moved</h3><p style="margin:4px 0">It is at its current place since <b>${esc(d.moved.current_since.slice(0, 10))}</b>. The map and the directions always use the position at the selected time.</p>
      <table><tr><th>Since</th><th>Position</th><th></th></tr>${d.moved.history.map(h => `<tr><td>${esc(h.since.slice(0, 10))}</td><td>${h.lat.toFixed(5)}, ${h.lon.toFixed(5)}</td><td><a target="_blank" rel="noopener" href="https://www.google.com/maps?q=${h.lat},${h.lon}">map</a></td></tr>`).join('')}</table></div>` : ''}
    <div class="sec"><h3>🔋 Battery</h3>${batteryHTML(c.battery, true)}<p class="mute" style="font-size:13px">${esc(c.battery.note)}</p>
      ${d.battery_detail ? `${d.battery_detail.trend ? `<div>Trend: <b>${esc(d.battery_detail.trend)}</b></div>` : ''}${d.battery_detail.forecast ? `<div class="todo" style="font-weight:400">🔮 ${esc(d.battery_detail.forecast)}</div>` : ''}` : ''}</div>
    <div class="sec aisec"><h3>${ico('ai', 30)} AI check <span class="mute" style="font-size:13px;font-weight:400">forecast and measurement check</span></h3>
      <div class="aisub"><div class="aih">🔮 Looking ahead</div>${d.predictions.items.map(predHTML).join('') || '<p class="mute">No forecast available.</p>'}
        <p class="mute" style="font-size:13px;margin-bottom:0">ℹ️ ${esc(d.predictions.quality)}</p></div>
      <div class="aisub"><div class="aih">🧪 Are the measurements reliable?</div>${(c.problems.length || d.anomalies.length || d.offsets.length) ?
      `<ul class="plain">${[...c.problems.filter(p => !d.anomalies.some(a => a.what === p)), ...d.anomalies.map(a => `${a.what} (${a.days} day${a.days > 1 ? 's' : ''}, last ${a.last.slice(0, 10)})`)].map(x => `<li>${esc(x)}</li>`).join('')}${d.offsets.map(o => `<li>${esc(o.what)} <span class="mute">${esc(o.note || '')}</span></li>`).join('')}</ul>`
      : '<p style="margin:4px 0">✅ No unusual measurements found.</p>'}</div></div>
    <details class="sec histsec" id="histDrop"><summary><span>📜 History &amp; last contact</span><span class="mute hsum">Last message ${esc(c.last_heard)} · ${d.outages.length} past breakdown${d.outages.length === 1 ? '' : 's'}</span></summary>
    <div class="hsub"><div class="hsh">📶 Last contact</div><div class="kv"><b>Last message received</b><span>${esc(c.last_heard)}</span><b>Last data saved</b><span>${esc(c.last_saved)}</span>
      <b>Normally reports every</b><span>${esc(d.interval)}</span><b>Main gateway</b><span>${esc(c.gateway || '–')}</span>${signalKV(c.id)}</div></div>
    <div class="hsub"><div class="hsh">📈 Last 120 days</div><div class="ctabs">${ctabs.map(([k, l], i) => `<button class="ctab${i ? '' : ' on'}" data-c="${k}">${l}</button>`).join('')}</div><div class="chartbox"><canvas id="chart"></canvas></div></div>
    <div class="hsub"><div class="hsh">⛔ Past breakdowns</div>${d.outages.length ? `<table><tr><th>Started</th><th>Ended</th><th>Duration</th><th>Affected</th></tr>${d.outages.map(o => `<tr><td>${esc(o.start)}</td><td>${esc(o.end || 'still ongoing')}</td><td>${esc(o.duration)}</td><td>${esc(o.scope)}</td></tr>`).join('')}</table>` : '<p class="mute">No breakdowns recorded.</p>'}</div></details>
    <div class="sec"><h3>✉️ E-mails sent about this sensor</h3>${d.emails.length ? `<table>${d.emails.map(m => `<tr class="click" data-mail="${m.id}"><td>${esc(m.when)}</td><td>${esc(m.kind)}</td></tr>`).join('')}</table>` : '<p class="mute">No e-mails.</p>'}</div>
    <details class="sec"><summary><b>⚙️ Technical details (for engineers)</b></summary><div class="kv" style="margin-top:8px">${Object.entries(d.tech).map(([k, v]) => `<b>${esc(k.replace(/_/g, ' '))}</b><span>${esc(v ?? '–')}</span>`).join('')}</div></details>`);
  if ($('#flyBtn')) $('#flyBtn').onclick = () => { showTab('map'); map.flyTo([c.lat, c.lon], 17, {duration: .8}); };
  $$('#drawerBody [data-mail]').forEach(r => r.onclick = () => showMail(r.dataset.mail));
  $$('#drawerBody .ctab').forEach(b => b.onclick = () => { $$('#drawerBody .ctab').forEach(x => x.classList.remove('on')); b.classList.add('on'); drawChart(ser, b.dataset.c); });
  $('#histDrop').addEventListener('toggle', e => { if (e.target.open) drawChart(ser, $('#drawerBody .ctab.on')?.dataset.c || 'delivered'); });
}
function drawChart(ser, kind) {
  if (S.chart) S.chart.destroy();
  const ds = [];
  if (kind === 'delivered') ds.push({label: 'Data received (% of expected)', data: ser.delivered, borderColor: '#2563eb', backgroundColor: '#2563eb33', fill: true, stepped: true});
  if (kind === 'meas') { ds.push({label: ser.m1_label, data: ser.m1, borderColor: '#0f766e'}); if (ser.m2_label) ds.push({label: ser.m2_label, data: ser.m2, borderColor: '#f59e0b', yAxisID: 'y2'}); }
  if (kind === 'battery') {
    if (ser.battery_pct.some(x => x != null)) ds.push({label: ser.battery_pct_estimated ? 'Battery charge (%, estimated)' : 'Battery charge (%, reported by the sensor)', data: ser.battery_pct, borderColor: '#16a34a', backgroundColor: '#16a34a22', fill: true});
    else if (ser.battery_v.some(x => x != null)) ds.push({label: 'Battery voltage (lithium: stays flat until nearly empty)', data: ser.battery_v, borderColor: '#16a34a'});
  }
  if (kind === 'risk7') ds.push({label: 'Chance of a long breakdown in the next 7 days (%)', data: ser.risk7, borderColor: '#d32f2f', backgroundColor: '#d32f2f22', fill: true});
  ds.forEach(d => { d.pointRadius = 0; d.borderWidth = 2; d.spanGaps = true; d.tension = .2; });
  const scales = {x: {ticks: {maxTicksLimit: 6, callback: function (v) { return (this.getLabelForValue(v) || '').slice(5, 10); }}}};
  if (kind === 'meas') { ds[0].yAxisID = 'y'; scales.y = {position: 'left', title: {display: true, text: ser.m1_label}}; }
  if (kind === 'meas' && ser.m2_label) scales.y2 = {position: 'right', grid: {drawOnChartArea: false}, title: {display: true, text: ser.m2_label}};
  if (kind === 'delivered' || kind === 'risk7' || (kind === 'battery' && ser.battery_pct.some(x => x != null))) scales.y = {min: 0, max: 100};
  S.chart = new Chart($('#chart'), {type: 'line', data: {labels: ser.day, datasets: ds}, options: {responsive: true, maintainAspectRatio: false, interaction: {mode: 'index', intersect: false}, plugins: {legend: {labels: {boxWidth: 12}}}, scales}});
}
async function openGateway(id) {
  S.openId = id; S.openKind = 'gateway'; clearFocus();
  const d = await api(`/api/gateway/${encodeURIComponent(id)}?as_of=${encodeURIComponent(S.asOf)}`); const g = d.gateway;
  openDrawer(`<div class="dhead">${ico('gateway', 56)}<div><div class="t1">${esc(g.name)}</div><div class="t2">Gateway (receiver) · ${g.official ? 'official' : 'not in the official list'}</div></div></div>
    <div class="actions actions-top">${g.lat != null ? `<a class="btn green" target="_blank" rel="noopener" href="${gmaps(g.lat, g.lon)}">🧭 Directions</a>` : ''}</div>
    <div class="statuscard ${g.light}"><div class="sc-head"><div class="b1">${esc(g.status)}</div><div>Last message received ${esc(g.last_heard)}</div></div>
      ${g.light === 'red' ? `<div class="sc-body"><div class="sc-row"><span class="sc-lab">🔎 Why</span><span>The gateway is not passing on any messages, so the sensors around it cannot send their data.</span></div>
        <div class="sc-row"><span class="sc-lab">👉 What to do</span><b>Check the gateway's power supply and internet connection.</b></div>
        <div class="sc-row"><span class="sc-lab">🚐 Site visit</span><span class="badge visit">Needed</span></div></div>` : ''}</div>
    ${d.history_rate ? `<div class="sec"><h3>🔮 How often is it off?</h3><p>${esc(d.history_rate)}</p></div>` : ''}
    <div class="sec"><h3>📈 Messages received per day (last 28 days)</h3><div class="chartbox"><canvas id="chart"></canvas></div></div>
    <div class="sec"><h3>Sensors that mainly use this gateway (${d.sensors.length})</h3><div class="list">${d.sensors.map(itemHTML).join('') || '<p class="mute">None at the moment.</p>'}</div></div>
    <div class="sec"><h3>⛔ Past interruptions</h3>${d.events.length ? `<table><tr><th>Started</th><th>Duration</th><th>What</th></tr>${d.events.map(e => `<tr><td>${esc(e.start)}</td><td>${esc(e.duration)}</td><td>${esc(e.what)}</td></tr>`).join('')}</table>` : '<p class="mute">None recorded.</p>'}</div>`);
  $$('#drawerBody .item').forEach(e => e.onclick = () => openSensor(e.dataset.id, true));
  if (S.chart) S.chart.destroy();
  S.chart = new Chart($('#chart'), {type: 'bar', data: {labels: d.daily.day.map(x => x.slice(5, 10)), datasets: [{label: 'Messages', data: d.daily.records, backgroundColor: '#2563eb'}]}, options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}}});
}
async function showMail(id) {
  const m = await api(`/api/email/${id}`);
  $('#modalBody').innerHTML = `<h3>${esc(m.subject)}</h3><p class="mute">${esc(m.when)} · ${m.status === 'dry_run' ? 'test mode: written to the outbox, not really sent' : esc(m.status)}</p><pre>${esc(m.text || '(text not available)')}</pre>`;
  $('#modal').classList.remove('hidden');
}
$('#closeModal').onclick = () => $('#modal').classList.add('hidden');
$('#modal').onclick = e => { if (e.target.id === 'modal') $('#modal').classList.add('hidden'); };

// ---------------------------------------------------------------- other tabs
async function loadAlerts() {
  const a = await api(`/api/alerts?as_of=${encodeURIComponent(S.asOf)}`);
  $('#openAlerts').innerHTML = a.open.length ? `<table><tr><th>Problem</th><th>Sensor / gateway</th><th>Since</th><th>E-mails</th></tr>${a.open.map(i => `<tr class="click" data-ent="${esc(i.entity)}" data-kind="${esc(i.kind)}"><td><span class="dot ${i.severity === 'critical' ? 'red' : 'yellow'}"></span> ${esc(i.what)}</td><td>${esc(i.entity)}</td><td>${esc(i.since)}</td><td>${i.emails}</td></tr>`).join('')}</table>` : '<p style="padding:10px">No open alerts. 🎉</p>';
  $('#mailNote').textContent = a.sent.length && a.sent[0].status === 'dry_run' ? '(test mode: e-mails are written to the outbox, not really sent)' : '';
  $('#sentMails').innerHTML = `<table><tr><th>When</th><th>Type</th><th>Subject</th></tr>${a.sent.map(m => `<tr class="click" data-mail="${m.id}"><td>${esc(m.when)}</td><td>${esc(m.kind)}</td><td>${esc(m.subject)}</td></tr>`).join('')}</table>`;
  $$('#openAlerts [data-ent]').forEach(r => r.onclick = () => r.dataset.kind === 'sensor' ? openSensor(r.dataset.ent, true) : r.dataset.kind === 'gateway' ? openGateway(r.dataset.ent) : null);
  $$('#sentMails [data-mail]').forEach(r => r.onclick = () => showMail(r.dataset.mail));
}
async function loadQuality() {
  const q = await api('/api/quality');
  $('#qualityCards').innerHTML = q.cards.map(c => `<div class="card"><span class="badge grade ${c.grade}">${esc(c.grade)}</span><h3>${esc(c.title)}</h3><p>${esc(c.text)}</p></div>`).join('');
  $('#demoList').innerHTML = q.demonstrations.length ? `<table><tr><th>What happened</th><th>Detected</th><th>System's diagnosis</th><th>E-mails</th></tr>${q.demonstrations.filter(d => d.found).map(d => `<tr><td>${esc(d.description)}</td><td>${esc(d.detected_at)}</td><td>${esc(d.knew_cause || '–')}</td><td>${esc(d.n_emails)}</td></tr>`).join('')}</table>` : '';
}
function showTab(t) {
  $$('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === t)); $$('.view').forEach(v => v.classList.toggle('active', v.id === 'view-' + t));
  if (t === 'map') setTimeout(() => map.invalidateSize(), 50);
  if (t === 'alerts') loadAlerts(); if (t === 'quality') loadQuality(); if (t === 'network') setTimeout(setupNetwork, 30);
  if (t === 'first' && !$('#routeBox').classList.contains('hidden')) drawRoute();
}
$$('.tab').forEach(b => b.onclick = () => showTab(b.dataset.tab));
$$('.tile').forEach(t => t.onclick = () => { if (t.dataset.tab) return showTab(t.dataset.tab); S.light = t.dataset.filter; showTab('map'); render(true); });
let qT = null; $('#search').oninput = e => { S.q = e.target.value.trim(); clearTimeout(qT); qT = setTimeout(() => render(true), 300); };

// ---------------------------------------------------------------- time
function fmt(t) { return new Date(t.replace(' ', 'T') + 'Z').toLocaleString('en-GB', {day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZone: 'Europe/Berlin'}); }
// ---- time: pick a day (calendar), then move through its hours or play the day
let tmr = null;
const BERLIN = new Intl.DateTimeFormat('en-CA', {timeZone: 'Europe/Berlin', year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', hourCycle: 'h23'});
function berlinDay(h) { const p = Object.fromEntries(BERLIN.formatToParts(new Date(h.replace(' ', 'T') + 'Z')).map(x => [x.type, x.value])); return [`${p.year}-${p.month}-${p.day}`, +p.hour]; }
function dayHours() { return S.days[S.day] || []; }
function drawTicks() {                 // labels under the hour slider follow the hours that exist on that day
  const H = dayHours(); if (!H.length) return;
  const idx = [...new Set([0, 1, 2, 3, 4].map(k => Math.round(k * (H.length - 1) / 4)))];
  $('.hourticks').innerHTML = idx.map(i => `<span>${String(berlinDay(H[i])[1]).padStart(2, '0')}:00</span>`).join('');
}
function showHour(i, load = true) {
  const H = dayHours(); i = Math.max(0, Math.min(i, H.length - 1)); $('#hourSlider').value = i;
  const t = H[i]; $('#timeLabel').textContent = fmt(t);
  if (load) { clearTimeout(tmr); tmr = setTimeout(() => loadState(t), 200); }
}
function setDay(day, hour) {           // hour: local hour to keep, or 'last'
  stopPlay(); S.day = day; $('#dayPick').value = day;
  const H = dayHours(); $('#hourSlider').max = H.length - 1; drawTicks();
  let i = H.length - 1;
  if (hour !== 'last' && hour != null) { const j = H.findIndex(h => berlinDay(h)[1] >= hour); if (j >= 0) i = j; }
  showHour(i);
}
$('#hourSlider').oninput = e => { stopPlay(); showHour(+e.target.value); };
$('#dayPick').onchange = e => { if (S.days[e.target.value]) setDay(e.target.value, berlinDay(S.asOf)[1]); else alertBox('No data for that day. Choose a day between ' + S.dayKeys[0] + ' and ' + S.dayKeys.at(-1) + '.'); };
$('#calBtn').onclick = e => { e.preventDefault(); try { $('#dayPick').showPicker(); } catch (_) { $('#dayPick').focus(); } };
$('#nowBtn').onclick = () => setDay(S.dayKeys.at(-1), 'last');
function stopPlay() { clearInterval(S.playing); S.playing = null; $('#playBtn').textContent = '▶'; $('#playBtn').title = 'Play this day hour by hour'; }
$('#playBtn').onclick = () => {
  if (S.playing) return stopPlay();
  const sl = $('#hourSlider'); if (+sl.value >= +sl.max) showHour(0, true);       // at the end of the day: start from midnight
  $('#playBtn').textContent = '⏸'; $('#playBtn').title = 'Pause';
  S.playing = setInterval(() => { const v = +sl.value + 1; if (v > +sl.max) return stopPlay(); showHour(v); }, 1300);
};
$('#testMailBtn').onclick = async () => {
  $('#testMailMsg').textContent = 'Sending…';
  try { const r = await fetch('/api/admin/test-emails', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({token: $('#adminCode').value})});
        const j = await r.json(); $('#testMailMsg').textContent = (j.ok ? '✅ ' : '⚠️ ') + j.message; }
  catch (e) { $('#testMailMsg').textContent = '⚠️ ' + e; }
};
$('#checkBtn').onclick = async () => { const r = await api('/api/check', {method: 'POST'}); alertBox(r.message); };
function alertBox(msg) { $('#modalBody').innerHTML = `<p>${esc(msg)}</p>`; $('#modal').classList.remove('hidden'); }

function render(fit) {
  drawTiles(); drawMap(); drawList(); drawFirst(); drawGateways(); drawChipCounts(); if (window.dimGateways) dimGateways();
  if (NW.map && $('#view-network').classList.contains('active')) drawNetSensors();
  if (RT.map && !$('#routeBox').classList.contains('hidden')) drawRoute();
  if (fit) {                                                  // zoom to the sensors that match the filter
    const pts = S.state.sensors.filter(c => c.lat != null && visible(c)).map(c => [c.lat, c.lon]);
    if (pts.length === 1) map.flyTo(pts[0], 17, {duration: .6}); else if (pts.length) map.flyToBounds(pts, {padding: [50, 50], maxZoom: 17, duration: .6});
  }
}
async function loadState(t) {
  S.asOf = t; $('#timeLabel').textContent = fmt(t);
  S.state = await api(`/api/state?as_of=${encodeURIComponent(t)}`); render();
  if (S.openId && $('#drawer').classList.contains('open')) (S.openKind === 'gateway' ? openGateway : openSensor)(S.openId);
  if ($('#view-alerts').classList.contains('active')) loadAlerts();
}
async function init() {
  S.meta = await api('/api/meta');
  const a = new Date(S.meta.first.replace(' ', 'T') + 'Z'), b = new Date(S.meta.last.replace(' ', 'T') + 'Z');
  for (let x = a; x <= b; x = new Date(x.getTime() + 3600e3)) S.hours.push(x.toISOString().slice(0, 16).replace('T', ' '));
  S.days = {}; for (const h of S.hours) { const [d] = berlinDay(h); (S.days[d] = S.days[d] || []).push(h); }
  S.dayKeys = Object.keys(S.days).sort(); $('#dayPick').min = S.dayKeys[0]; $('#dayPick').max = S.dayKeys.at(-1);
  S.day = S.dayKeys.at(-1); $('#dayPick').value = S.day; $('#hourSlider').max = dayHours().length - 1; $('#hourSlider').value = dayHours().length - 1; drawTicks();
  const groups = {};
  for (const t of S.meta.types) { const k = t.measures.split(' (')[0]; (groups[k] = groups[k] || {label: k, icon: t.icon_key, families: []}).families.push(t.family); }
  S.groups = Object.values(groups);
  $('#typeChips').innerHTML = S.groups.map((g, i) => `<span class="chip" data-g="${i}">${ico(g.icon, 18)} ${esc(g.label)} <b class="cnt"></b></span>`).join('');
  $$('#typeChips .chip').forEach(c => c.onclick = () => {
    const g = S.groups[+c.dataset.g], on = !c.classList.contains('on'); c.classList.toggle('on', on);
    g.families.forEach(f => on ? S.types.add(f) : S.types.delete(f)); render(true);
  });
  $('#helpTypes').innerHTML = S.meta.types.map(t => `<p>${ico(t.icon_key, 26)} ${esc(t.measures)} <span class="mute">(${esc(t.family)})</span></p>`).join('') + `<p>${ico('gateway', 26)} Gateway (receiver)</p>`;
  if (S.meta.live) { $('#liveBadge').classList.remove('hidden'); $('#checkBtn').classList.remove('hidden'); setInterval(async () => { const m = await fetch('/api/meta').then(r => r.json()); if (m.last !== S.meta.last) { S.meta = m; location.reload(); } }, 60000); }
  await loadState(S.hours[S.hours.length - 1]);
  const pts = S.state.sensors.filter(c => c.lat != null).map(c => [c.lat, c.lon]); if (pts.length) map.fitBounds(pts, {padding: [30, 30]});
}
init();
