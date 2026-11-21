"""Hildesheim Smart Sensor Monitoring: interactive web application.

    python -m hsm.webapp.app --data appdata [--port 8050] [--host 0.0.0.0] [--live-config config/hsm.yaml --models models]

Without --live-config the app shows the prepared data (the historical replay up to the export end); a time slider shows the
situation at any past hour. With --live-config a background thread runs the live monitoring cycle (ingest exports, models,
e-mail notifications) every poll interval, refreshes `--data` and the app picks up the new state automatically."""
import argparse, json, math, threading, time, os, traceback
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template, abort
from flask.json.provider import DefaultJSONProvider
from .. import system
from ..config import NOMINAL_MIN
from . import plain

def _j(v):
    if v is None: return None
    if isinstance(v, (np.floating, float)): return None if np.isnan(v) else round(float(v), 6)
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Timestamp): return None if pd.isna(v) else local(v)
    if v is pd.NaT: return None
    return v

TZ = 'Europe/Berlin'
def local(t):
    """data times are UTC; everything shown to people is local time (Hildesheim)"""
    if t is None or (not isinstance(t, str) and pd.isna(t)): return None
    t = pd.Timestamp(t)
    if pd.isna(t): return None
    return (t.tz_localize('UTC') if t.tzinfo is None else t).tz_convert(TZ).strftime('%Y-%m-%d %H:%M')

def clean(o):
    """recursively make any value JSON-safe (NaN -> null, numpy -> python, timestamps -> local time)"""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)): return None
    if isinstance(o, (np.generic, pd.Timestamp)) or o is pd.NaT: return _j(o.item() if isinstance(o, np.generic) and not isinstance(o, (np.floating,)) else o)
    return o

class SafeJSON(DefaultJSONProvider):
    sort_keys = False
    def dumps(self, obj, **kw): return super().dumps(clean(obj), **kw)

def recs(df):
    return [{k: _j(v) for k, v in r.items()} for r in df.to_dict('records')]


def compact(df):
    """memory-lean copy (fits small hosting plans): repeated texts -> categories, floats -> float32; values unchanged for display"""
    for c in df.columns:
        if df[c].dtype == 'float64': df[c] = df[c].astype('float32')
        elif df[c].dtype == object or str(df[c].dtype).startswith(('string', 'str')):
            if df[c].nunique(dropna=True) < 0.2 * len(df): df[c] = df[c].astype('category')
    return df

class Store:
    """all prepared tables in memory; reloads when appdata/meta.json changes (live mode)"""
    def __init__(self, data_dir):
        self.dir = Path(data_dir); self.mtime = None; self.lock = threading.Lock(); self.load()

    def load(self):
        rd = lambda n: pd.read_parquet(self.dir / f'{n}.parquet') if (self.dir / f'{n}.parquet').exists() else pd.DataFrame()
        S = rd('snapshots')                                             # prepared compact (categories, float32) and pre-sorted
        if S.dtypes.eq('float64').any(): S = compact(S).sort_values(['hour_end', 'device_id']).reset_index(drop=True)   # older appdata
        with self.lock:
            self.S = S; self.D = rd('devices').set_index('device_id'); self.Y = rd('daily'); self.E = rd('outages'); self.A = rd('anomalies')
            self.OFF = rd('offsets'); self.BA = rd('battery_assessment'); self.GH = rd('gateway_hour'); self.GC = rd('gateways'); self.GE = rd('gateway_events')
            self.I = rd('incidents'); self.N = rd('notifications'); self.M = rd('emails'); self.DEMO = rd('demonstrations')
            self.ev = json.load(open(self.dir / 'evaluation.json')) if (self.dir / 'evaluation.json').exists() else {}
            P = rd('positions')
            self.pos = {str(d): (g.t_start.values, g.lat.values, g.lon.values) for d, g in P.sort_values('t_start').groupby('device_id')} if len(P) else {}
            self.net = json.load(open(self.dir / 'network.json')) if (self.dir / 'network.json').exists() else {}
            self.meta = json.load(open(self.dir / 'meta.json')); self.mtime = os.path.getmtime(self.dir / 'meta.json')
            self.hours = np.array(sorted(S.hour_end.unique())); self._cache = {}
            # battery readings per device, to show the last known value of sensors that stopped sending
            Bv = S.loc[S.battery_value.notna() & ~S.battery_condition.astype(str).isin(['no_recent_reading', 'not_measurable']), ['device_id', 'hour_end', 'battery_condition', 'battery_value']]
            self.bat = {str(d): (g.hour_end.values, g.battery_condition.astype(str).values, g.battery_value.values) for d, g in Bv.groupby('device_id', observed=True)}

    def maybe_reload(self):
        try:
            if os.path.getmtime(self.dir / 'meta.json') != self.mtime: self.load()
        except OSError: pass

    def as_of(self, s):
        t = pd.Timestamp(s) if s else pd.Timestamp(self.hours[-1])
        i = np.searchsorted(self.hours, np.datetime64(t), 'right') - 1
        return pd.Timestamp(self.hours[max(i, 0)])

    def current(self, t):
        """latest complete hour of each device at time t, with priority score"""
        if t in self._cache: return self._cache[t]
        h = self.S.hour_end.values                                     # sorted: slice instead of a full boolean copy
        S = self.S.iloc[np.searchsorted(h, np.datetime64(t - pd.Timedelta(days=60)), 'right'):np.searchsorted(h, np.datetime64(t), 'right')]
        cur = S.groupby('device_id', observed=True).tail(1).copy()
        for c in cur.columns:
            if isinstance(cur[c].dtype, pd.CategoricalDtype): cur[c] = cur[c].astype(object)
            elif cur[c].dtype == 'float32': cur[c] = cur[c].astype('float64')
        cur = system.priority(cur)
        if len(self._cache) > 64: self._cache.clear()
        self._cache[t] = cur; return cur


def last_battery(st, dev, t):
    b = st.bat.get(str(dev))
    if b is None: return None
    i = np.searchsorted(b[0], np.datetime64(t), 'right') - 1
    return None if i < 0 else (b[1][i], float(b[2][i]), local(pd.Timestamp(b[0][i])))

def position_at(store, dev, t):
    """(lat, lon, moves) where the device stood at time t; moves = list of earlier positions with dates"""
    p = store.pos.get(str(dev)) if store is not None else None
    if p is None: return None
    i = max(np.searchsorted(p[0], np.datetime64(t), 'right') - 1, 0)
    moves = [dict(since=local(pd.Timestamp(p[0][k])), lat=float(p[1][k]), lon=float(p[2][k])) for k in range(len(p[0]))]
    return float(p[1][i]), float(p[2][i]), moves, i

def sensor_card(r, D, t, store=None):
    fam = r['family']; d = D.loc[r['device_id']] if r['device_id'] in D.index else {}
    meas, icon, _ = plain.MEASURES.get(fam, (fam, '📟', (None, None)))
    r = dict(r); r['_interval'] = NOMINAL_MIN.get(fam, 60)
    st = plain.status(r, t, fam); light = r.get('overall_light') or 'grey'
    lat, lon = _j(d.get('lat')) if len(d) else None, _j(d.get('lon')) if len(d) else None
    pa = position_at(store, r['device_id'], t)
    if pa: lat, lon = round(pa[0], 6), round(pa[1], 6)              # position at the selected time (devices can be moved)
    return dict(id=r['device_id'], family=fam, manufacturer=_j(d.get('manufacturer')) if len(d) else '', model=_j(d.get('model')) if len(d) else '',
                measures=meas, icon=icon, icon_key=plain.ICON_KEY.get(fam, 'soil'), light=light, light_text=plain.LIGHT.get(light, ('', ''))[0], lat=lat, lon=lon,
                status=st['title'], status_text=st['text'], cause=st['cause'], todo=st['todo'], visit=st['visit'],
                battery=plain.battery(r, fam, last_battery(store, r['device_id'], t) if store is not None and r.get('battery_condition') == 'no_recent_reading' else None), last_heard=plain.ago(r.get('rx_last_arrival'), t), last_saved=plain.ago(r.get('svc_last_arrival'), t),
                priority=_j(r.get('priority_score')), why=r.get('why') or '', gateway=r.get('primary_gateway'),
                problems=plain.anomaly_text(r.get('anomaly_kinds')), risk7=_j(r.get('risk_prolonged_7d')))


def create_app(data_dir, live=None):
    app = Flask(__name__, template_folder=str(Path(__file__).parent / 'templates'), static_folder=str(Path(__file__).parent / 'static'))
    app.json = SafeJSON(app)
    st = Store(data_dir); app.config['store'] = st; app.config['live'] = live

    @app.before_request
    def _reload(): st.maybe_reload()

    ver = str(int(max(os.path.getmtime(p) for p in Path(app.static_folder).rglob('*') if p.is_file())))
    @app.get('/')
    def index(): return render_template('index.html', v=ver)

    @app.get('/api/meta')
    def meta():
        fams = sorted(st.D.family.unique())
        return jsonify(meta=st.meta, first=pd.Timestamp(st.hours[0]).strftime('%Y-%m-%d %H:%M'), last=pd.Timestamp(st.hours[-1]).strftime('%Y-%m-%d %H:%M'),   # UTC (slider)
                       types=[dict(family=f, measures=plain.MEASURES.get(f, (f, '📟'))[0], icon=plain.MEASURES.get(f, (f, '📟'))[1], icon_key=plain.ICON_KEY.get(f, 'soil')) for f in fams],
                       live=bool(live), live_status=live.status if live else None)

    @app.get('/api/state')
    def state():
        t = st.as_of(request.args.get('as_of')); cur = st.current(t)
        cards = [sensor_card(r, st.D, t, st) for r in cur.to_dict('records')]
        G = gateways_at(t)
        counts = {k: sum(c['light'] == k for c in cards) for k in ['red', 'yellow', 'green', 'grey']}
        I = st.I; opened = pd.to_datetime(I.opened_at); closed = pd.to_datetime(I.closed_at)
        open_now = I[(opened <= t) & (closed.isna() | (closed > t))]
        prio = [c for c in sorted(cards, key=lambda c: -(c['priority'] or 0)) if c['light'] in ('red', 'yellow')]
        return jsonify(as_of=_j(t), sensors=cards, gateways=G, counts=counts, open_alerts=int(len(open_now)),
                       priority=prio[:60], unlocated=[c for c in cards if c['lat'] is None])

    def gateways_at(t):
        GH = st.GH[st.GH.hour_end <= t]
        if not len(GH): return []
        cur = GH.sort_values('hour_end').groupby('gw_site').tail(1).set_index('gw_site')
        last_act = GH[GH.records > 0].groupby('gw_site').hour_end.max()
        out = []
        for s, r in cur.iterrows():
            g = st.GC[st.GC.gw_site == s]; g = g.iloc[0] if len(g) else {}
            txt, col = plain.GATEWAY.get(r.condition, (r.condition, 'grey'))
            out.append(dict(id=s, name=s.replace('unlisted_', 'Unlisted gateway ').replace('_', ' / '), lat=_j(g.get('lat')) if len(g) else None,
                            lon=_j(g.get('lon')) if len(g) else None, status=txt, light=col, last_heard=plain.ago(last_act.get(s), t),
                            official=bool(g.get('official')) if len(g) else None, expected_per_hour=_j(r.mu), received_last_hour=_j(r.records)))
        return out

    @app.get('/api/sensor/<dev>')
    def sensor(dev):
        t = st.as_of(request.args.get('as_of')); cur = st.current(t)
        row = cur[cur.device_id == dev]
        if not len(row): abort(404)
        r = row.iloc[0].to_dict(); fam = r['family']; card = sensor_card(r, st.D, t, st)
        r['_interval'] = NOMINAL_MIN.get(fam, 60)
        stt = plain.status(r, t, fam); pr = plain.predictions(r, fam)
        labels = plain.MEASURES.get(fam, (fam, '', (None, None)))[2]
        Y = st.Y[(st.Y.device_id == dev) & (st.Y.day <= t) & (st.Y.day > t - pd.Timedelta(days=120))].sort_values('day')
        conv = plain.lead_acid_pct if fam in ('Bigbelly', 'OttSensor') else plain.lithium_pct if fam in ('Agvolution', 'Dragino-SE0X-LB', 'Adeunis') else None
        series = dict(day=[_j(x) for x in Y.day], delivered=[_j(x * 100) if pd.notna(x) else None for x in Y.service_availability],
                      battery_v=[],
                      battery_pct=[conv(float(x)) if pd.notna(x) else None for x in Y.v_min] if conv and 'v_min' in Y
                                  else ([_j(x) for x in Y.level_pct] if 'level_pct' in Y else []), battery_pct_estimated=conv is not None,
                      m1=[_j(x) for x in Y.m1], m2=[_j(x) for x in Y.m2], risk7=[_j(x * 100) if pd.notna(x) else None for x in Y.risk7],
                      m1_label=labels[0], m2_label=labels[1])
        E = st.E[(st.E.device_id == dev) & (st.E.onset <= t)].sort_values('onset', ascending=False).head(15)
        outages = [dict(start=_j(e.onset), end=_j(e.recovered_at) if pd.notna(e.recovered_at) and e.recovered_at <= t else None,
                        duration=plain.hours_text(((min(e.recovered_at, t) if pd.notna(e.recovered_at) else t) - e.onset).total_seconds() / 3600),
                        scope={'individual': 'only this sensor', 'gateway': 'its gateway was down', 'network': 'whole network down',
                               'fleet': 'many sensors of this type'}.get(e.scope_at_declaration, e.scope_at_declaration)) for e in E.itertuples()]
        A = st.A[(st.A.device_id == dev) & (st.A.day <= t) & (st.A.day > t - pd.Timedelta(days=60))]
        anomalies = [dict(what=plain.anomaly_text(k)[0] if plain.anomaly_text(k) else k, first=_j(g.day.min()), last=_j(g.day.max()), days=int(len(g)),
                          severity=g.severity.iat[0]) for k, g in A.groupby('kind')]
        offs = [dict(what=f"{plain.VARIABLE.get(o.variable, o.variable).capitalize()} differs from nearby stations by {o.offset_28d:+.2f}", note=o.interpretation)
                for o in st.OFF[st.OFF.device_id == dev].itertuples()] if len(st.OFF) else []
        bat = None
        if len(st.BA) and t >= pd.Timestamp(st.hours[-1]) - pd.Timedelta(days=2):
            b = st.BA[st.BA.device_id == dev]
            if len(b):
                b = b.iloc[0]; bat = dict(trend=None, forecast=None)
                if pd.notna(b.get('slope_mV_per_week')):
                    w = b.slope_mV_per_week; bat['trend'] = ('slowly falling' if w < -5 else 'slowly rising' if w > 5 else 'stable') + ' (charge changes by a few % per month)' if abs(w) < 40 else ('falling quickly' if w < 0 else 'rising')
                if pd.notna(b.get('p_below_critical_30d')):
                    already_low = b.get('condition') in ('low', 'critical')
                    bat['forecast'] = ("The battery is already in the 'low' band. " if already_low else
                                       f"Chance it becomes low within 30 days: {b.p_below_warning_30d:.0%}. ")
                    bat['forecast'] += f"Chance it becomes critical within 30 days: {b.p_below_critical_30d:.0%}."
                    if not already_low and pd.notna(b.get('warning_crossing_day_p50')): bat['forecast'] += f" Expected to reach 'low' around {pd.Timestamp(b.warning_crossing_day_p50):%d %b %Y}."
                if pd.notna(b.get('slope_pct_per_week')): bat['trend'] = f"{b.slope_pct_per_week:+.1f} % per week"
                bat['note'] = b.get('note')
        I = st.I[(st.I.entity == dev) & (pd.to_datetime(st.I.opened_at) <= t)].sort_values('opened_at', ascending=False).head(15)
        ids = set(I.id)
        N = st.N[pd.to_datetime(st.N.sent_at) <= t]
        N = N[N.incident_ids.apply(lambda s: bool(ids & set(json.loads(s))) if isinstance(s, str) else False)].sort_values('sent_at', ascending=False).head(25)
        mails = [dict(id=int(n.id), when=local(n.sent_at), kind=n.kind.title(), subject=n.subject) for n in N.itertuples()]
        d = st.D.loc[dev] if dev in st.D.index else {}
        gw = r.get('primary_gateway')
        if not isinstance(gw, str) and fam in ('Bigbelly', 'OttSensor'): card['gateway'] = 'mobile network (no LoRa gateway)'
        gwr = st.GC[st.GC.gw_site == gw] if isinstance(gw, str) else st.GC.iloc[0:0]
        dest = (card['lat'], card['lon']) if card['lat'] is not None else ((_j(gwr.lat.iat[0]), _j(gwr.lon.iat[0])) if len(gwr) else (None, None))
        pa = position_at(st, dev, t); moved = None
        if pa and len(pa[2]) > 1:
            moved = dict(current_since=pa[2][pa[3]]['since'], history=pa[2])
        if stt['state'] == 'normal' and stt['cause']: stt['text'] = 'It sends its data as expected, but: ' + stt['cause'][0].lower() + stt['cause'][1:]
        return jsonify(card=card, status=stt, predictions=pr, moved=moved, series=series, outages=outages, anomalies=anomalies, offsets=offs, battery_detail=bat,
                       emails=mails, interval=plain.interval_text(NOMINAL_MIN.get(fam, 60)),
                       directions=dict(lat=dest[0], lon=dest[1], approximate=card['lat'] is None,
                                       label='Sensor position' if card['lat'] is not None else f'Area of gateway {gw} (exact sensor position not in the data)'),
                       tech=dict(state=r.get('state'), cause=r.get('probable_cause'), evidence=r.get('cause_evidence'), alternatives=r.get('alternative_causes'),
                                 data_24h=_j(r.get('avail_svc_24h')), data_28d=_j(r.get('avail_svc_672h')), gateway=gw, gateway_condition=r.get('gateway_condition'),
                                 signal_snr=_j(r.get('snr_mean_168h')), battery_voltage=(f"{card['battery']['volts']:.2f} V" if card['battery'].get('volts') else None),
                                 note=_j(d.get('battery_note')) if len(d) else None))

    @app.get('/api/network')
    def network():
        """measured radio links and signal quality (whole monitoring period) for the focus mode and the network page"""
        return jsonify(st.net or dict(sensors={}, gateways={}, model=None))

    @app.get('/api/gateway/<path:gid>')
    def gateway(gid):
        t = st.as_of(request.args.get('as_of')); G = [g for g in gateways_at(t) if g['id'] == gid]
        if not G: abort(404)
        cur = st.current(t); deps = cur[cur.primary_gateway == gid]
        H = st.GH[(st.GH.gw_site == gid) & (st.GH.hour_end <= t) & (st.GH.hour_end > t - pd.Timedelta(days=28))]
        daily = H.assign(day=H.hour_end.dt.floor('D')).groupby('day').records.sum()
        ev = st.GE[(st.GE.gw_site == gid) & (st.GE.start <= t)].sort_values('start', ascending=False).head(10)
        rate = st.GC[st.GC.gw_site == gid]
        rate_txt = None
        if len(rate) and pd.notna(rate.rate_per_week.iat[0]):
            n = int(rate.interruptions_total.iat[0]) if 'interruptions_total' in rate and pd.notna(rate.interruptions_total.iat[0]) else None
            rate_txt = ((f"It was off {n} time{'s' if n != 1 else ''} since monitoring began. " if n is not None else '') +
                        f"Based on that history, the chance of an interruption in the next 7 days is about {rate.p_interruption_7d.iat[0]:.0%}. "
                        "Gateway breakdowns could not be predicted from warning signs in the data, so this is the historical frequency, not a forecast.")
        return jsonify(gateway=G[0], sensors=[sensor_card(r, st.D, t, st) for r in deps.to_dict('records')], daily=dict(day=[_j(x) for x in daily.index], records=[int(x) for x in daily.values]),
                       events=[dict(start=_j(e.start), end=_j(e.end), duration=plain.hours_text(e.hours),
                                    what={'network_or_platform': 'whole network down', 'inactive_gateway': 'gateway off (sensors switched to others)',
                                          'inactive_unexplained': 'gateway and its sensors off'}.get(e.kind, e.kind)) for e in ev.itertuples()], history_rate=rate_txt)

    @app.get('/api/alerts')
    def alerts():
        t = st.as_of(request.args.get('as_of')); I = st.I.copy(); opened = pd.to_datetime(I.opened_at); closed = pd.to_datetime(I.closed_at)
        op = I[(opened <= t) & (closed.isna() | (closed > t))].sort_values('opened_at', ascending=False)
        TYPE = {'SENSOR_DOWN': 'Sensor not sending', 'BACKEND_FAILURE': 'Data not saved (server)', 'GATEWAY_DOWN': 'Gateway off', 'NETWORK_OUTAGE': 'Network down',
                'BATTERY_CRITICAL': 'Battery critical', 'BATTERY_LOW': 'Battery low', 'SEVERE_MALFUNCTION': 'Serious measurement fault',
                'DEGRADED_COMMUNICATION': 'Data getting lost', 'REDUCED_REPORTING_MODE': 'Power-saving mode', 'MEASUREMENT_DEGRADATION': 'Unusual measurements',
                'RISK_ELEVATED': 'Breakdown likely soon', 'SILENCE_DEVELOPING': 'Missed messages', 'FEED_STALE': 'Data export stopped'}
        N = st.N[pd.to_datetime(st.N.sent_at) <= t].sort_values('sent_at', ascending=False).head(200)
        return jsonify(open=[dict(id=int(i.id), what=TYPE.get(i.type, i.type), entity=i.entity, kind=i.entity_kind, severity=i.severity, since=local(i.opened_at),
                                  emails=int(i.n_notifications or 0), acknowledged=bool(isinstance(i.acknowledged_at, str))) for i in op.itertuples()],
                       sent=[dict(id=int(n.id), when=local(n.sent_at), kind=n.kind.title(), subject=n.subject.replace('[HSM ', '[').strip(), status=n.status) for n in N.itertuples()])

    @app.get('/api/email/<int:nid>')
    def mail(nid):
        m = st.M[st.M.id == nid]; n = st.N[st.N.id == nid]
        if not len(n): abort(404)
        return jsonify(subject=n.subject.iat[0], when=local(n.sent_at.iat[0]), status=n.status.iat[0], text=m.text.iat[0] if len(m) else None)

    @app.get('/api/quality')
    def quality():
        return jsonify(cards=quality_cards(st.ev), demonstrations=recs(st.DEMO[['case', 'description', 'found', 'entity', 'detected_at', 'n_emails', 'knew_cause', 'happened_outage_duration_h']])
                       if len(st.DEMO) else [])

    @app.post('/api/check')
    def check():
        if not live: return jsonify(ok=False, message='Live monitoring is not configured. Start the app with --live-config.'), 400
        live.trigger(); return jsonify(ok=True, message='Check started. The map updates automatically when it is finished.')

    last_test = {'t': 0.0}
    @app.post('/api/admin/test-emails')
    def test_emails():
        """send the two [TEST] alert e-mails to the configured recipients. Needs HSM_ADMIN_TOKEN (environment) and
        the same token in the request; recipients come only from the server configuration, never from the request."""
        tok = os.environ.get('HSM_ADMIN_TOKEN')
        if not tok: return jsonify(ok=False, message='Test e-mails are switched off: set HSM_ADMIN_TOKEN on the server.'), 403
        if (request.get_json(silent=True) or {}).get('token') != tok: return jsonify(ok=False, message='Wrong admin code.'), 403
        if time.time() - last_test['t'] < 60: return jsonify(ok=False, message='Please wait a minute between test e-mails.'), 429
        last_test['t'] = time.time()
        from ..cli import load_cfg
        from ..notify.samples import send_samples
        cfg = load_cfg(os.environ.get('HSM_CONFIG', str(Path(__file__).resolve().parents[2] / 'config' / 'alerts.yaml')))
        if not cfg['notifications'].get('recipients'): return jsonify(ok=False, message='No recipient: set HSM_ALERT_TO on the server.'), 400
        try: r = send_samples(cfg['notifications'], st.dir, os.environ.get('HSM_MODELS', str(Path(__file__).resolve().parents[2] / 'models')))
        except Exception as e: return jsonify(ok=False, message=f'{type(e).__name__}: {e}'), 500
        ok = all(x['status'] in ('sent', 'dry_run') for x in r['sent']) and r['sent']
        msg = (f"{len(r['sent'])} test e-mail(s) " + ('sent to ' + ', '.join(r['recipients']) if r['real_email'] else 'written to the outbox (real sending is off: set HSM_EMAIL_ENABLED=1)')
               if ok else 'Sending failed: ' + ' | '.join(str(x.get('error') or x.get('status')) for x in r['sent'])[:600])
        return jsonify(ok=bool(ok), message=msg, detail=r)

    @app.get('/api/live')
    def live_status(): return jsonify(live=bool(live), status=live.status if live else None)

    return app


def quality_cards(ev):
    """plain-language summary of the evaluation on the held-out test months and the replay"""
    cards = []
    risk = pd.DataFrame(ev.get('risk', []))
    def g(task, fam, col):
        x = risk[(risk.task == task) & (risk.family == fam)] if len(risk) else risk
        return x[col].iat[0] if len(x) and col in x and pd.notna(x[col].iat[0]) else None
    a = g('A_prolonged_7d', 'Sensoterra', 'auc'); rc = g('A_prolonged_7d', 'Sensoterra', 'event_recall'); ld = g('A_prolonged_7d', 'Sensoterra', 'median_lead_h')
    if a: cards.append(dict(title='Warning of long breakdowns (soil probes, Sensoterra)', grade='Good' if a >= 0.8 else 'Fair',
                            text=f"Tested on October-November data the system never saw while learning. It warned of {rc:.0%} of long breakdowns, typically {ld / 24:.0f} days ahead. "
                                 f"Ranking quality (AUC) {a:.2f}, where 0.5 is guessing and 1.0 is perfect."))
    for fam, name in [('MeteoHelix', 'weather stations'), ('MeteoWind', 'wind sensors')]:
        a = g('A_onset_long', fam, 'auc')
        if a: cards.append(dict(title=f'Will it stop sending in the next 24 hours? ({name})', grade='Good' if a >= 0.8 else 'Fair',
                                text=f"Ranking quality (AUC) {a:.2f} on the test months. Useful to know which stations to watch, but most warnings are early hints, not certainties."))
    a = g('A_onset_long', 'Sensoterra', 'auc')
    if a: cards.append(dict(title='Will a soil probe stop sending in the next 24 hours?', grade='Weak', text=f"AUC {a:.2f}: short-term breakdowns of soil probes are hard to foresee. Use the 7-day warning instead."))
    b = g('B_developing', 'Sensoterra', 'auc')
    if b: cards.append(dict(title='Is a missed message the start of a real breakdown?', grade='Good', text=f"AUC {b:.2f} (soil probes). Real breakdowns are recognised a few hours before the system officially declares them."))
    d = g('D_continuation_168h', 'Sensoterra', 'auc')
    if d: cards.append(dict(title='Once broken: will it still be broken in a week?', grade='Good', text=f"AUC {d:.2f} (soil probes). Helps decide whether a breakdown fixes itself or needs a visit."))
    rp = ev.get('replay', {})
    if rp: cards.append(dict(title='E-mail alerts (replay Aug-Nov 2026, test mode: no e-mails really sent)', grade='Good',
                             text=(f"E-mails go out only when a sensor stops working or may stop within 24 hours. Every breakdown where the sensor itself stopped "
                                   f"was e-mailed ({rp.get('outages_sensor_stopped_emailed_share', rp.get('outages_ge6h_ongoing_notified_share', 0)):.0%} of "
                                   f"{rp.get('outages_sensor_stopped_ongoing', '')}), on average {rp.get('emails_per_device_week', 0):.2f} e-mails per sensor per week. "
                                   f"{rp.get('outages_server_side_not_emailed', 0)} server-side problems (sensor fine, data not saved) are shown in the app but not e-mailed.")))
    if rp.get('warnings_24h_emailed'):
        cards.append(dict(title='"May stop within 24 hours" e-mails', grade='Fair',
                          text=f"{rp['warnings_24h_emailed']} warnings in the replay; {rp['warnings_24h_followed_by_outage_within_24h']:.0%} were followed by a breakdown within 24 hours. "
                               "Only sent for sensor types where the forecast proved reliable enough on past data (wind sensors, CLOVER), and for critical 12 V batteries."))
    A = pd.DataFrame(ev.get('anomaly', []))
    if len(A):
        c = A[(A.detector == 'alarm_cusum') & A.fault.str.startswith('bias_2.5')]
        if len(c): cards.append(dict(title='Detecting a weather station that measures wrong', grade='Good',
                                     text=f"In tests with artificially introduced errors, {c.detection_rate.mean():.0%} of small offsets were found, typically within {c.median_delay_h.median():.0f} hours."))
    B = pd.DataFrame(ev.get('battery', []))
    if len(B):
        bb = B[(B.family == 'Bigbelly') & (B.k == 14) & (B.method == 'local_linear_trend')]
        if len(bb): cards.append(dict(title='Battery forecast (waste bins, 12 V)', grade='Good',
                                      text=f"Two weeks ahead the voltage forecast was off by {bb.mae.iat[0] * 1000:.0f} mV on average. Other battery types cannot be forecast because their voltage stays flat."))
    W = pd.DataFrame(ev.get('gateway_warning', []))
    if len(W): cards.append(dict(title='Can gateway breakdowns be predicted?', grade='No',
                                 text='No. None of the warning signs tested came before gateway breakdowns, so the app shows only how often each gateway was off in the past.'))
    cards.append(dict(title='What the predictions cannot do', grade='Note',
                      text='There are no repair records, so "breakdown" means "stopped sending data". For sensor types with very few past breakdowns, '
                           'forecasts are rough guides only. Probable causes are best guesses from the evidence, not certainties.'))
    return cards


class Live:
    """background live monitoring: runs the full monitoring cycle (exports -> models -> e-mails) and refreshes the app data"""
    def __init__(self, cfg_path, models, data_dir):
        from ..cli import load_cfg
        self.cfg = load_cfg(cfg_path); self.models = models; self.data_dir = data_dir; self.status = dict(state='idle', last=None, error=None)
        self.ev = threading.Event(); threading.Thread(target=self.loop, daemon=True).start()
    def trigger(self): self.ev.set()
    def loop(self):
        from ..cli import monitor_once
        from .prepare import prepare
        while True:
            self.status['state'] = 'running'
            try:
                res = monitor_once(self.cfg, self.models)
                d = self.cfg['data']
                if Path(d.get('app_data_dir') or '_').resolve() != Path(self.data_dir).resolve():      # monitor_once already refreshed it otherwise
                    prepare(d['foundation_dir'], d.get('live_dir', str(Path(d['foundation_dir']).parent / 'live_state')), self.data_dir, d.get('eval_foundation_dir'))
                self.status.update(state='idle', last=res.get('now'), error=None, notifications=len(res.get('notifications', [])))
            except Exception as e:
                self.status.update(state='error', error=f'{type(e).__name__}: {e}'); traceback.print_exc()
            self.ev.wait(60 * self.cfg['data'].get('poll_minutes', 30)); self.ev.clear()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='appdata'); ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', type=int, default=8050)
    ap.add_argument('--live-config'); ap.add_argument('--models', default='models')
    a = ap.parse_args(argv)
    live = Live(a.live_config, a.models, a.data) if a.live_config else None
    app = create_app(a.data, live)
    print(f'Open http://{"localhost" if a.host in ("127.0.0.1", "0.0.0.0") else a.host}:{a.port} in your browser')
    try:
        from waitress import serve; serve(app, host=a.host, port=a.port, threads=4)
    except ImportError:
        app.run(host=a.host, port=a.port, threaded=True)

if __name__ == '__main__':
    main()
