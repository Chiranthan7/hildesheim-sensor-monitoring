"""Monitoring dashboard (component 9): one self-contained HTML file (data embedded as JSON, no server, no external libraries).
usage: python -m hsm.dashboard.build --foundation <dir> --replay <dir> --out dashboard.html [--as-of '2026-11-01 12:00']
--as-of shows the system exactly as it was at that replay hour (states, risk, incidents and notifications up to then)."""
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from .. import system
from ..config import NOMINAL_MIN, BATTERY_OBSERVABILITY

def _clean(o):
    if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (pd.Timestamp,)): return None if pd.isna(o) else o.strftime('%Y-%m-%d %H:%M')
    if isinstance(o, (np.floating, float)): return None if (o is None or not np.isfinite(o)) else round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if o is pd.NaT: return None
    try:
        if pd.isna(o): return None
    except (TypeError, ValueError): pass
    return o

def records(df, cols=None):
    d = df if cols is None else df[[c for c in cols if c in df.columns]]
    return [_clean(r) for r in d.to_dict('records')]

def build(fd, rp, out, as_of=None, eval_dir=None):
    """fd: foundation with component outputs; rp: replay or live-state directory (sys_device_hour, incidents, notifications);
    eval_dir: foundation holding the historical evaluation tables when fd is a live foundation (default: fd)"""
    fd, rp = Path(fd), Path(rp); ed = Path(eval_dir) if eval_dir else fd
    def rd(n):
        p = fd / f'{n}.parquet'
        return pd.read_parquet(p if p.exists() else ed / f'{n}.parquet')
    S = pd.read_parquet(rp / 'sys_device_hour.parquet')
    as_of = pd.Timestamp(as_of) if as_of else S[S.hour_complete].hour_end.max()
    cur = S[(S.hour_end <= as_of) & S.hour_complete].sort_values('hour_end').groupby('device_id').tail(1)
    cur = system.priority(cur)
    cur['manufacturer'] = cur.family.map(lambda f: system.MANUFACTURER[f][0]); cur['model'] = cur.family.map(lambda f: system.MANUFACTURER[f][1])
    cur['interval_min'] = cur.family.map(NOMINAL_MIN); cur['hours_since_rx'] = (as_of - cur.rx_last_arrival).dt.total_seconds() / 3600
    cur['hours_since_stored'] = (as_of - cur.svc_last_arrival).dt.total_seconds() / 3600
    cur['deployment'] = np.where(cur.hours_since_rx > 24 * 14, 'inactive (> 14 d silent)', 'deployed')
    cur['battery_note'] = cur.family.map(lambda f: BATTERY_OBSERVABILITY[f]['note'])
    # locations
    loc = pd.concat([rd('silver_soil')[['device_id', 'sensor_lat', 'sensor_lon']].dropna().groupby('device_id').median(),
                     rd('silver_weather')[['device_id', 'sensor_lat', 'sensor_lon']].dropna().groupby('device_id').median()])
    dl = rd('silver_devices'); dl = dl[dl.loc_role == 'device_position'].groupby('device_id')[['loc_lat', 'loc_lon']].median().rename(columns={'loc_lat': 'sensor_lat', 'loc_lon': 'sensor_lon'})
    loc = pd.concat([loc, dl]); loc = loc[~loc.index.duplicated()]
    cur = cur.join(loc, on='device_id')
    # battery assessment (latest daily values up to as_of)
    BA = rd('gold_battery_assessment') if as_of >= S.hour_end.max() - pd.Timedelta('2D') else None
    # histories (daily)
    P = rd('gold_m1_period_daily'); P = P[P.period <= as_of]
    hist = {d: records(g, ['period', 'service_availability', 'reception_availability', 'valid_data_availability', 'mode_availability', 'missed_reports', 'outage_onsets'])
            for d, g in P.groupby('device_id')}
    U = rd('gold_uplinks'); U = U[(U.t <= as_of) & U.radio_metrics_available & U.snr_max.notna()]
    U['day'] = U.t.dt.floor('D'); rad = U.groupby(['device_id', 'day']).agg(snr=('snr_max', 'median'), rssi=('rssi_max', 'median'), beyond=('flag_beyond_sf12_reference_limits', 'mean')).reset_index()
    radio = {d: records(g, ['day', 'snr', 'rssi', 'beyond']) for d, g in rad.groupby('device_id')}
    BD = rd('gold_battery_daily'); BD = BD[BD.day <= as_of]; LV = rd('gold_battery_level_daily'); LV = LV[LV.day <= as_of]
    batt = {d: records(g, ['day', 'v_min', 'v_max', 'v_med']) for d, g in BD.groupby('device_id')}
    for d, g in LV.groupby('device_id'): batt[d] = records(g.rename(columns={'level_pct': 'v_med'}), ['day', 'v_med'])
    # measurement daily series (primary variable per family)
    so = rd('silver_soil'); so = so[(so.dup_rank == 0) & so.data_stored & (so.t <= as_of)]; so['day'] = so.t.dt.floor('D')
    so['m1'] = np.where(so.family == 'CloverSoilSensor', so.gwc_clean, so.vwc_pct)
    ms = so.groupby(['device_id', 'day']).agg(m1=('m1', 'mean'), m2=('soil_temp_c', 'mean')).reset_index()
    wx = rd('silver_weather'); wx = wx[(wx.dup_rank == 0) & (wx.t <= as_of) & ~wx.flag_bad_packet.fillna(False) & wx.family.isin(['MeteoHelix', 'MeteoWind'])]; wx['day'] = wx.t.dt.floor('D')
    wx['m1'] = np.where(wx.family == 'MeteoWind', wx.wind_ms, wx.temp_c); wx['m2'] = np.where(wx.family == 'MeteoWind', wx.gust_ms, wx.rh_pct)
    mw = wx.groupby(['device_id', 'day']).agg(m1=('m1', 'mean'), m2=('m2', 'mean')).reset_index()
    meas = {d: records(g, ['day', 'm1', 'm2']) for d, g in pd.concat([ms, mw]).groupby('device_id')}
    risk_d = S[S.hour_end <= as_of].assign(day=lambda x: x.hour_end.dt.floor('D')).groupby(['device_id', 'day']).agg(
        risk7=('risk_prolonged_7d', 'max'), still=('p_still_down_24h', 'max')).reset_index()
    riskh = {d: records(g, ['day', 'risk7', 'still']) for d, g in risk_d.groupby('device_id')}
    I = pd.read_parquet(rp / 'incidents.parquet'); I = I[pd.to_datetime(I.opened_at) <= as_of]
    I['closed_at'] = np.where(pd.to_datetime(I.closed_at) <= as_of, I.closed_at, None)
    N = pd.read_parquet(rp / 'notifications.parquet'); N = N[pd.to_datetime(N.sent_at) <= as_of]
    E = rd('gold_silences'); E = E[(E.status == 'declared') & (E.onset <= as_of)][['device_id', 'onset', 'declared_at', 'recovered_at', 'scope_at_declaration', 'cause_hint']]
    E['recovered_at'] = E.recovered_at.where(E.recovered_at <= as_of)
    outages = {d: records(g.tail(40)) for d, g in E.groupby('device_id')}
    AR = rd('gold_anom_rule_daily'); AR = AR[(AR.day <= as_of) & AR.severity.isin(['high', 'medium'])]
    anoms = {d: records(g.groupby(['kind', 'severity', 'probable_cause', 'action']).agg(first=('day', 'min'), last=('day', 'max'), days=('day', 'size')).reset_index())
             for d, g in AR.groupby('device_id')}
    SE = rd('gold_anom_station_events'); SE = SE[(SE.start <= as_of) & (SE.hours >= 24) & (SE.kind != 'spike_vs_neighbours')]
    for d, g in SE.groupby('device_id'):
        anoms.setdefault(d, []).extend(records(g.assign(first=g.start, last=g.end, days=(g.hours / 24).round(1))[['kind', 'severity', 'probable_cause', 'action', 'first', 'last', 'days', 'variable']].tail(10)))
    OFF = rd('gold_anom_static_offsets'); offs = {d: records(g) for d, g in OFF[OFF.exceeds_2x_accuracy].groupby('device_id')}
    alt = rd('gold_gw_alternatives'); alts = {d: records(g, ['gw_site', 'share']) for d, g in alt.groupby('device_id')}
    # gateways
    GH = rd('gold_gw_site_hour'); GH = GH[GH.hour_end <= as_of]
    gcur = GH.sort_values('hour_end').groupby('gw_site').tail(1).set_index('gw_site')
    last_act = GH[GH.records > 0].groupby('gw_site').hour_end.max()
    GC = rd('gold_gw_current').set_index('gw_site')
    gws = []
    for s, r in gcur.iterrows():
        h28 = GH[(GH.gw_site == s) & (GH.hour_end > as_of - pd.Timedelta('28D'))]
        deps = cur[cur.primary_gateway == s].device_id.tolist()
        g = dict(gw_site=s, condition=r.condition, last_activity=last_act.get(s), records_last_hour=r.records, expected_per_hour=r.mu, dependents=deps,
                 active_hour_share_28d=float((h28.records > 0).mean()), devices_per_hour_28d=float(h28.devices.mean()), snr_median_28d=float(h28.snr_med.median()),
                 rssi_median_28d=float(h28.rssi_med.median()))
        if s in GC.index: g.update({k: GC.at[s, k] for k in ['lat', 'lon', 'official', 'rate_per_week', 'rate_lo90', 'rate_hi90', 'p_interruption_24h', 'p_interruption_7d'] if k in GC.columns})
        gd = GH[GH.gw_site == s].assign(day=lambda x: x.hour_end.dt.floor('D')).groupby('day').agg(records=('records', 'sum'), devices=('devices', 'max'), snr=('snr_med', 'median')).reset_index()
        g['history'] = records(gd)
        g['probable_cause'] = {'inactive_gateway': 'gateway-specific: dependent sensors re-routed via other gateways', 'inactive_unexplained': 'undetermined: gateway and its sensors silent together',
                               'network_or_platform': 'network / platform interruption', 'reduced_activity': 'partial loss of traffic'}.get(r.condition, 'none')
        g['recommended_action'] = {'inactive_gateway': 'check gateway power, backhaul and packet forwarder', 'inactive_unexplained': 'check gateway power and backhaul, then dependent sensors',
                                   'network_or_platform': 'check network server / platform', 'reduced_activity': 'review gateway logs and antenna'}.get(r.condition, 'none')
        gws.append(g)
    GE = rd('gold_gw_events'); GE = GE[GE.start <= as_of]
    # evaluation summaries
    ev = {}
    for name, path in [('m1', fd / 'gold_m1_comparison.parquet'), ('risk', fd / 'gold_risk_test_summary.parquet'), ('anomaly', fd / 'gold_anom_injection_eval.parquet'),
                       ('battery', fd / 'gold_battery_backtest.parquet'), ('gw_warning', fd / 'gold_gw_warning_test.parquet')]:
        if not path.exists(): path = ed / path.name
        if not path.exists(): continue
        x = pd.read_parquet(path)
        if name == 'battery': x = x.groupby(['family', 'k', 'method', 'role']).agg(mae=('abs_err', 'mean'), coverage80=('covered80', 'mean'), n=('abs_err', 'size')).reset_index()
        if name == 'm1': x = x.drop(columns=[c for c in x.columns if c.endswith('ci95')])
        if name == 'risk': x = x[x.n_test > 0]
        ev[name] = records(x)
    rj = rp / 'replay_evaluation.json'
    ev['replay'] = json.load(open(rj)) if rj.exists() else {}
    cols = ['device_id', 'family', 'manufacturer', 'model', 'state', 'traffic_light', 'overall_light', 'elevated_risk', 'deployment', 'reasons', 'probable_cause', 'cause_confidence', 'cause_evidence',
            'alternative_causes', 'recommended_action', 'priority_score', 'why', 'rx_last_arrival', 'svc_last_arrival', 'hours_since_rx', 'hours_since_stored', 'interval_min',
            'rx_mode_min', 'svc_mode_meaning', 'avail_svc_24h', 'avail_rx_24h', 'avail_valid_24h', 'avail_mode_24h', 'avail_svc_672h', 'forecast_avail', 'forecast_model', 'tau',
            'primary_H', 'expected_missing_primary', 'hmm_p_bad', 'risk_short', 'risk_short_H', 'risk_long', 'risk_long_H', 'risk_prolonged_7d', 'risk_prolonged_14d',
            'p_silence_becomes_outage', 'p_still_down_24h', 'p_still_down_7d', 'unavailable_h', 'battery_condition', 'battery_value', 'battery_note', 'anomaly_kinds',
            'anomaly_high', 'primary_gateway', 'gateway_condition', 'primary_share_168h', 'n_sites_672h', 'sensor_lat', 'sensor_lon', 'declared_scope', 'beyond_ref_share_24h', 'snr_mean_168h']
    sensors = records(cur, cols)
    for s_ in sensors:
        d = s_['device_id']
        s_.update(history=hist.get(d, []), radio=radio.get(d, []), battery=batt.get(d, []), measurements=meas.get(d, []), risk_history=riskh.get(d, []),
                  outages=outages.get(d, []), anomalies=anoms.get(d, []), offsets=offs.get(d, []), alternatives=alts.get(d, []),
                  incidents=records(I[I.entity == d].tail(20), ['id', 'type', 'severity', 'opened_at', 'closed_at', 'n_notifications', 'status']))
        if BA is not None:
            b = BA[BA.device_id == d]
            if len(b): s_['battery_assessment'] = _clean(b.iloc[0].to_dict())
    open_inc = I[I.closed_at.isna()]
    gw_prio = sorted([g for g in gws if g['condition'] in ('inactive_gateway', 'inactive_unexplained', 'network_or_platform', 'reduced_activity')],
                     key=lambda g: -(len(g['dependents']) + 10))
    data = dict(as_of=_clean(as_of), generated_from=dict(foundation=str(fd), replay=str(rp)), sensors=sensors, gateways=gws, gateway_events=records(GE),
                notifications=records(N.sort_values('sent_at', ascending=False).head(400), ['id', 'kind', 'sent_at', 'subject', 'status', 'incident_ids', 'group_key']),
                open_incidents=records(open_inc, ['id', 'type', 'entity', 'severity', 'opened_at', 'n_notifications', 'group_key']), evaluation=ev, gateway_priority=gw_prio,
                battery_not_measurable=[f for f, o in BATTERY_OBSERVABILITY.items() if o['signal'] is None])
    html = (Path(__file__).parent / 'template.html').read_text().replace('/*__DATA__*/null', json.dumps(_clean(data), default=str, allow_nan=False))
    Path(out).write_text(html); return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); ap.add_argument('--replay', required=True)
    ap.add_argument('--out', default='dashboard.html'); ap.add_argument('--as-of'); a = ap.parse_args()
    print(build(a.foundation, a.replay, a.out, a.as_of))
