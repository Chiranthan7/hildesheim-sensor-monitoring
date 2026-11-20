"""Device malfunction and measurement anomaly detection (component 4).

1. Rule layer (manufacturer / foundation flags), aggregated per device-day with a severity and probable cause. Informational
   flags that are documented normal behaviour (CLOVER dry-category GWC, MeteoWind calm direction, Sensoterra saturation) are
   reported as such, never as malfunctions.
2. Statistical layer for the weather stations (comparable neighbours, same city):
   residual r_i(h) = x_i(h) - median_{j != i} x_j(h)     (leave-one-out fleet median of hourly means)
   bias / drift : EWMA_t = lam r_t + (1 - lam) EWMA_{t-1}, lam = 0.02 (memory ~ 2 days); alarm |EWMA| > 2 x manufacturer accuracy
   CUSUM        : S+_t = max(0, S+_{t-1} + r_t - k), S-_t likewise; k = accuracy, alarm S > h = 24 x accuracy (one day of 2 x accuracy)
   spike        : |r_t - median_24h(r)| > max(6 x 1.4826 MAD_168h, 5 x accuracy)
   frozen       : identical raw value for >= FROZEN_N consecutive reports while the neighbours change
   isolation forest (comparison): daily residual features (mean, sd, max |r|, completeness) per station
   wind         : daily log ratio to the fleet median, EWMA alarm |EWMA| > log 1.5
   Thresholds are derived from the manufacturer accuracy (config.SPEC_ACCURACY), not tuned on outcomes.
3. Soil probes: robust jump detection on moisture (|delta| > 8 robust SD of the device's own changes) and the frozen-temperature rule.
Evaluation: synthetic faults injected into clean held-out station data (bias, drift, stuck, spike) -> detection rate, delay, and
false-alarm rate on the unmodified data."""
import numpy as np, pandas as pd
from .config import SPEC_ACCURACY, const

FROZEN_N = const('FROZEN_N', {'MeteoHelix': 18, 'MeteoWind': 36}, 'operational', 'features',
                 'identical consecutive raw readings (10-min reports) flagged as frozen: 3 h temperature (0.01 C resolution), 6 h wind')
EWMA_LAMBDA = const('EWMA_LAMBDA', 0.02, 'operational', 'features', 'EWMA weight per hourly residual (effective memory ~50 h)')

RULES = [  # (table, flag column, kind, severity, probable cause, action)
    ('soil', 'flag_soil_temp_out_of_spec', 'out_of_manufacturer_range', 'high', 'sensor fault or exposed / damaged probe', 'inspect probe'),
    ('soil', 'flag_temp_stuck_sofar', 'frozen_temperature', 'high', 'stuck sensor channel or firmware fault', 'inspect / replace probe'),
    ('soil', 'flag_soil_temp_exposure_suspected', 'temperature_exposure_suspected', 'medium', 'probe exposed to air / sun (reads case temperature)', 'check installation depth'),
    ('soil', 'flag_rawfreq_fault', 'raw_frequency_fault', 'high', 'CLOVER probe raw frequency out of range (open circuit / probe fault)', 'inspect probe and cable'),
    ('soil', 'flag_ec_suspected_unscaled', 'ec_decoder_scaling', 'medium', 'EC value without unit scaling (decoder / backend change)', 'check payload decoder'),
    ('soil', 'flag_ec_implausible', 'ec_implausible', 'medium', 'implausible conductivity', 'check probe / decoder'),
    ('soil', 'ingestion_failed', 'backend_insert_failure', 'high', 'backend database insert failed (data received, not stored)', 'fix backend ingestion; values recoverable'),
    ('soil', 'flag_zero_moisture', 'moisture_zero', 'medium', 'Sensoterra: no soil contact / too dry (documented); CLOVER: dry category', 'check probe contact'),
    ('weather', 'flag_bad_packet', 'corrupted_packet', 'medium', 'corrupted or reset weather-station message (impossible value / counter reset)', 'monitor; inspect if recurring'),
    ('devices', 'flag_temp_sentinel', 'channel_never_valid', 'high', 'temperature channel reports error value -99 (probe missing / error)', 'check probe connection'),
    ('devices', 'flag_battery_error_code', 'battery_error_code', 'medium', 'battery measurement error code', 'check power module'),
    ('devices', 'flag_rssi_sentinel', 'radio_sentinel', 'low', 'radio metadata sentinel value', 'none'),
]
INFO = [('soil', 'flag_gwc_censored_dry', 'clover_dry_category', 'documented: GWC beyond the dry end of the conversion table (not frozen)'),
        ('weather', 'flag_wind_calm', 'calm_direction_undefined', 'documented: wind direction undefined at calm'),
        ('soil', 'flag_saturated', 'saturated', 'documented: Sensoterra saturation point overshot')]

def rule_daily(soil, wx, devs):
    out = []
    for tab, df in [('soil', soil), ('weather', wx), ('devices', devs)]:
        d = df[df.dup_rank == 0].copy(); d['day'] = d.t.dt.floor('D')
        for t_, col, kind, sev, cause, act in RULES + [(a, b, c, 'info', e, 'none') for a, b, c, e in INFO]:
            if t_ != tab or col not in d: continue
            x = d[d[col].fillna(False).astype(bool)]
            if not len(x): continue
            g = x.groupby(['device_id', 'family', 'day']).size().rename('n').reset_index()
            tot = d.groupby(['device_id', 'day']).size().rename('n_records')
            g = g.join(tot, on=['device_id', 'day']); g['share'] = g.n / g.n_records
            g['kind'] = kind; g['severity'] = sev; g['probable_cause'] = cause; g['action'] = act; g['channel'] = col; g['layer'] = 'rule'
            out.append(g)
    return pd.concat(out, ignore_index=True)

def neighbour_residuals(wx, var, fam='MeteoHelix'):
    w = wx[(wx.dup_rank == 0) & (wx.family == fam) & ~wx.flag_bad_packet.fillna(False)].copy()
    w['hour_end'] = w.t.dt.floor('h') + pd.Timedelta('1h')
    p = w.groupby(['hour_end', 'device_id'])[var].mean().unstack()
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq='h'))
    med = p.median(axis=1); n = p.notna().sum(axis=1)
    # leave-one-out median approximated by the median of the others
    loo = pd.DataFrame({c: p.drop(columns=c).median(axis=1) for c in p.columns})
    r = (p - loo).where(n >= 5)
    return p, r

def debias(r):
    """remove the station's persistent offset: residual minus its own median over the previous 28 days (causal)"""
    return r - r.rolling(28 * 24, min_periods=72).median().shift(1)

def detectors(r, acc, lam=None, debiased=True):
    """r: hourly residual series of one station -> alarm series per detector (bool, aligned).
    Change detectors run on the de-biased residual (a static offset is reported separately by static_offsets)."""
    lam = lam or EWMA_LAMBDA
    r = debias(r) if debiased else r
    x = r.values; e = np.full(len(x), np.nan); s_p = np.zeros(len(x)); s_m = np.zeros(len(x)); ew = 0.0; sp = sm = 0.0; init = False
    for i, v in enumerate(x):
        if np.isnan(v): e[i] = ew if init else np.nan; s_p[i] = sp; s_m[i] = sm; continue
        ew = v if not init else lam * v + (1 - lam) * ew; init = True
        sp = max(0.0, sp + v - acc); sm = max(0.0, sm - v - acc); e[i] = ew; s_p[i] = sp; s_m[i] = sm
    med24 = r.rolling(24, min_periods=6).median(); mad = (r - r.rolling(168, min_periods=24).median()).abs().rolling(168, min_periods=24).median()
    spike = (r - med24).abs() > np.maximum(6 * 1.4826 * mad, 5 * acc)
    return pd.DataFrame({'ewma': e, 'cusum_pos': s_p, 'cusum_neg': s_m,
                         'alarm_ewma': np.abs(e) > 2 * acc, 'alarm_cusum': (s_p > 24 * acc) | (s_m > 24 * acc),
                         'alarm_spike': spike.values, 'alarm_static': (r.abs() > 5 * acc).values}, index=r.index)

def frozen_runs(wx, fam, var, n):
    w = wx[(wx.dup_rank == 0) & (wx.family == fam) & wx[var].notna()].sort_values(['device_id', 't'])
    if var == 'wind_ms': w = w[w[var] > 0]                                # calm (0 m/s) is a valid repeated value, not a frozen sensor
    w['chg'] = w.groupby('device_id')[var].transform(lambda s: s.ne(s.shift()).cumsum())
    g = w.groupby(['device_id', 'chg']).agg(start=('t', 'min'), end=('t', 'max'), n=('t', 'size'), value=(var, 'first')).reset_index()
    return g[g.n >= n].drop(columns='chg').assign(family=fam, variable=var)

def _daily_feats(rr):
    d = rr.resample('D').agg(['mean', 'std', lambda x: x.abs().max(), 'count']); d.columns = ['mean', 'sd', 'maxabs', 'n']
    return d[d.n >= 12].drop(columns='n')

def inject_eval(p, r_all, var, acc, test_start, seed=0, n_inj=60):
    """synthetic faults (in multiples of the manufacturer accuracy) injected into clean held-out station data.
    detection within 7 days of the fault start; delay; false-alarm episodes per station-week on the unmodified test data.
    Isolation Forest (daily residual features) is trained on the period before test_start and flags the 1 % most isolated days."""
    from sklearn.ensemble import IsolationForest
    rng = np.random.default_rng(seed); rows = []
    ts = pd.Timestamp(test_start)
    stations = [c for c in p.columns if p[c][ts:].notna().mean() > 0.8]
    tr = pd.concat([_daily_feats(r_all[c][:ts]) for c in p.columns], ignore_index=True).dropna()
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed).fit(tr)
    fa = {}
    for c in stations:
        D = detectors(r_all[c], acc)[ts:]; weeks = len(D) / 168
        for det in ['alarm_ewma', 'alarm_cusum', 'alarm_spike', 'alarm_static']:
            a = D[det].fillna(False).values; ep = int(np.sum(a[1:] & ~a[:-1]) + a[0]); fa.setdefault(det, []).append((ep, weeks))
        f = _daily_feats(r_all[c][ts:]).dropna(); a = iso.predict(f) == -1 if len(f) else np.array([], bool)
        fa.setdefault('isolation_forest', []).append((int(a.sum()), len(f) / 7))
    fa_rate = {k: sum(e for e, _ in v) / max(sum(w for _, w in v), 1e-9) for k, v in fa.items()}
    kinds = {f'bias_2.5x_acc': lambda n: np.full(n, 2.5), 'bias_5x_acc': lambda n: np.full(n, 5.0),
             'drift_5x_acc_per_week': lambda n: np.arange(n) / 168 * 5.0, 'spike_25x_acc': lambda n: np.r_[25.0, np.zeros(n - 1)]}
    idx = p[ts:].index; others_all = None
    for kname, f in kinds.items():
        for _ in range(n_inj):
            c = stations[rng.integers(len(stations))]; i0 = int(rng.integers(168, len(idx) - 24 * 8))
            j = p.index.get_loc(idx[i0]); q = p[c].copy(); add = np.zeros(len(q)); add[j:] = f(len(q) - j) * acc; q = q + add
            others = p.drop(columns=c).median(axis=1); rr = (q - others).where(p.notna().sum(axis=1) >= 5)
            D = detectors(rr, acc); win = D.iloc[j:j + 24 * 7]
            for det in ['alarm_ewma', 'alarm_cusum', 'alarm_spike', 'alarm_static']:
                hit = np.where(win[det].fillna(False).values)[0]
                rows.append(dict(variable=var, fault=kname, detector=det, detected=len(hit) > 0, delay_h=float(hit[0]) if len(hit) else np.nan))
            fd = _daily_feats(rr.iloc[j - 24 * 7: j + 24 * 7]).dropna(); fd = fd[fd.index >= rr.index[j].floor('D')]
            hit = np.where(iso.predict(fd) == -1)[0] if len(fd) else []
            rows.append(dict(variable=var, fault=kname, detector='isolation_forest', detected=len(hit) > 0, delay_h=float(hit[0] * 24 + 24) if len(hit) else np.nan))
    R = pd.DataFrame(rows)
    S = R.groupby(['variable', 'fault', 'detector']).agg(detection_rate=('detected', 'mean'), median_delay_h=('delay_h', 'median')).reset_index()
    S['false_alarms_per_station_week'] = S.detector.map(fa_rate)
    return S

def static_offsets(r_all, acc, var, fam, as_of=None):
    """persistent offset of each station vs its neighbours over the last 28 days (median residual), judged against 2 x accuracy"""
    r = r_all if as_of is None else r_all[:as_of]
    m = r.tail(28 * 24).median(); n = r.tail(28 * 24).notna().sum()
    out = pd.DataFrame({'device_id': m.index, 'family': fam, 'variable': var, 'offset_28d': m.values, 'n_hours': n.values})
    out['exceeds_2x_accuracy'] = out.offset_28d.abs() > 2 * acc
    out['interpretation'] = np.where(~out.exceeds_2x_accuracy, 'within manufacturer accuracy',
                                     'station altitude / pressure-reduction difference (siting, probably not a fault)' if var == 'pressure_hpa'
                                     else 'calibration offset or siting / radiation-shield difference')
    return out

def station_events(r_all, acc, var, fam):
    """alarm episodes per station on the full period (bias/drift: EWMA, spikes)"""
    rows = []
    for c in r_all.columns:
        D = detectors(r_all[c], acc)
        for det, kind, sev in [('alarm_cusum', 'neighbour_bias_or_drift_change', 'medium'), ('alarm_spike', 'spike_vs_neighbours', 'low')]:
            a = D[det].fillna(False); run = (a != a.shift()).cumsum()
            for _, g in D[a].groupby(run[a]):
                rows.append(dict(device_id=c, family=fam, variable=var, kind=kind, severity=sev, start=g.index.min() - pd.Timedelta('1h'), end=g.index.max(),
                                 hours=len(g), mean_ewma=float(g.ewma.mean()), detected_at=g.index.min(),
                                 probable_cause='new bias or drift relative to the station\'s own baseline (calibration drift, shield contamination, siting change or sensor degradation)' if det == 'alarm_cusum' else 'transient measurement error',
                                 action='compare with neighbours / reference; check shield and siting' if det == 'alarm_cusum' else 'monitor'))
    return pd.DataFrame(rows)

def wind_ratio_events(wx):
    w = wx[(wx.dup_rank == 0) & (wx.family == 'MeteoWind')].copy(); w['day'] = w.t.dt.floor('D')
    p = w.groupby(['day', 'device_id']).wind_ms.mean().unstack()
    lr = np.log((p + 0.1).div(p.median(axis=1) + 0.1, axis=0))
    rows = []
    for c in lr.columns:
        s = lr[c]; e = s.ewm(alpha=0.2, ignore_na=True).mean(); a = e.abs() > np.log(1.5)
        run = (a != a.shift()).cumsum()
        for _, g in e[a].groupby(run[a]):
            rows.append(dict(device_id=c, family='MeteoWind', variable='wind_ms', kind='wind_speed_ratio_shift', severity='medium', start=g.index.min(),
                             end=g.index.max() + pd.Timedelta('1D'), hours=24 * len(g), mean_ewma=float(np.exp(g.mean())), detected_at=g.index.min() + pd.Timedelta('1D'),
                             probable_cause='anemometer bearing / obstruction or siting (speed persistently different from the city median)', action='inspect anemometer'))
    return pd.DataFrame(rows)

def soil_jumps(soil, z=8.0):
    s = soil[(soil.dup_rank == 0) & soil.data_stored & soil.vwc_pct.notna() & ~soil.flag_saturated & ~soil.flag_zero_moisture].sort_values(['device_id', 'channel', 't']).copy()
    s['d'] = s.groupby(['device_id', 'channel'], dropna=False).vwc_pct.diff()
    g = s.groupby(['device_id', 'channel'], dropna=False).d
    med = g.transform(lambda x: x.rolling(500, min_periods=50).median().shift(1)); mad = g.transform(lambda x: (x - x.rolling(500, min_periods=50).median()).abs().rolling(500, min_periods=50).median().shift(1))
    s['rz'] = (s.d - med) / (1.4826 * mad.replace(0, np.nan))
    j = s[s.rz.abs() > z]
    return j[['device_id', 'family', 'channel', 't', 'vwc_pct', 'd', 'rz']].assign(kind='moisture_jump', severity='low',
             probable_cause='sudden moisture change not explained by the device\'s own variability (irrigation, disturbance or fault)', action='review')
