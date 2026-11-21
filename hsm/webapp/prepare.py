"""Condense the outputs of every component into a small, self-contained `appdata/` folder for the web app.

    python -m hsm.webapp.prepare --foundation foundation --replay replay --out appdata [--eval-foundation foundation]

`replay` may be the historical replay folder or the live-state folder written by `hsm.cli monitor` (same tables).
Nothing is re-modelled here: the app shows exactly what the models and the incident engine produced."""
import argparse, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from .. import system
from ..config import NOMINAL_MIN, BATTERY_OBSERVABILITY

SNAP_COLS = ['device_id', 'family', 'hour_end', 'state', 'traffic_light', 'overall_light', 'reasons', 'declared_scope', 'silence_svc_slots',
             'avail_svc_24h', 'avail_svc_672h', 'forecast_avail', 'tau', 'expected_missing_primary', 'primary_H', 'hmm_p_bad',
             'rx_last_arrival', 'svc_last_arrival', 'rx_mode_min', 'svc_mode_meaning', 'n_onsets_7d',
             'risk_long', 'risk_long_H', 'risk_prolonged_7d', 'risk_prolonged_14d', 'p_silence_becomes_outage', 'p_still_down_24h', 'p_still_down_7d',
             'primary_gateway', 'gateway_condition', 'primary_share_168h', 'n_sites_672h', 'battery_condition', 'battery_value',
             'anomaly_kinds', 'anomaly_high', 'probable_cause', 'cause_confidence', 'cause_evidence', 'alternative_causes', 'elevated_risk',
             'beyond_ref_share_24h', 'snr_mean_168h', 'p_stop_24h', 'in_declared_outage', 'risk_short', 'risk_short_H']

def _rd(fd, ed):
    def rd(n, cols=None):
        p = Path(fd) / f'{n}.parquet'
        p = p if p.exists() else Path(ed) / f'{n}.parquet'
        return pd.read_parquet(p, columns=cols) if p.exists() else None
    return rd

def prepare(fd, rp, out, eval_dir=None):
    fd, rp, out = Path(fd), Path(rp), Path(out); ed = Path(eval_dir) if eval_dir else fd
    out.mkdir(parents=True, exist_ok=True); rd = _rd(fd, ed)
    # ---- hourly snapshots of the integrated system (state, risks, battery, anomalies, cause) ----
    S = pd.read_parquet(rp / 'sys_device_hour.parquet')
    if 'hour_complete' in S.columns: S = S[S.hour_complete]
    S = S[[c for c in SNAP_COLS if c in S.columns]].copy()
    for c in SNAP_COLS:
        if c not in S.columns: S[c] = np.nan
    # stored compact and pre-sorted so that the web app loads it without copies (fits 512 MB hosting plans)
    S = S.sort_values(['hour_end', 'device_id']).reset_index(drop=True)
    for c in S.columns:
        if S[c].dtype == 'float64': S[c] = S[c].astype('float32')
        elif S[c].dtype == bool: pass
        elif S[c].dtype == object or str(S[c].dtype).startswith(('string', 'str')): S[c] = S[c].astype('category')
    S.to_parquet(out / 'snapshots.parquet', index=False, row_group_size=50_000)
    # ---- devices: family, manufacturer, measured quantity, GPS position from the data (never invented) ----
    POS = system.device_positions(fd)                                  # where each device stood, with the date it got there
    POS.to_parquet(out / 'positions.parquet', index=False)
    L = POS.sort_values('t_start').groupby('device_id').tail(1).set_index('device_id')[['lat', 'lon']]     # latest position
    D = S.groupby('device_id').family.first().to_frame().join(L)
    D['manufacturer'] = D.family.map(lambda f: system.MANUFACTURER.get(f, ('', ''))[0]); D['model'] = D.family.map(lambda f: system.MANUFACTURER.get(f, ('', ''))[1])
    D['interval_min'] = D.family.map(NOMINAL_MIN); D['battery_note'] = D.family.map(lambda f: BATTERY_OBSERVABILITY.get(f, {}).get('note', ''))
    D.reset_index().to_parquet(out / 'devices.parquet', index=False)
    from .. import network                                             # radio links, signal quality, distance model
    network.build(fd, D.reset_index(), out)
    # ---- daily series per device ----
    P = rd('gold_m1_period_daily')
    P = P.rename(columns={'period': 'day'})[['device_id', 'day', 'service_availability', 'reception_availability', 'valid_data_availability', 'missed_reports', 'outage_onsets']]
    BD = rd('gold_battery_daily'); LV = rd('gold_battery_level_daily')
    B = pd.concat([BD[['device_id', 'day', 'v_min', 'v_med']] if BD is not None else None,
                   LV[['device_id', 'day', 'level_pct']] if LV is not None else None])
    so = rd('silver_soil', ['device_id', 'family', 't', 'dup_rank', 'data_stored', 'gwc_clean', 'vwc_pct', 'soil_temp_c'])
    so = so[(so.dup_rank == 0) & so.data_stored]; so['day'] = so.t.dt.floor('D'); so['m1'] = np.where(so.family == 'CloverSoilSensor', so.gwc_clean, so.vwc_pct)
    ms = so.groupby(['device_id', 'day']).agg(m1=('m1', 'mean'), m2=('soil_temp_c', 'mean')).reset_index()
    wx = rd('silver_weather', ['device_id', 'family', 't', 'dup_rank', 'flag_bad_packet', 'wind_ms', 'gust_ms', 'temp_c', 'rh_pct'])
    wx = wx[(wx.dup_rank == 0) & ~wx.flag_bad_packet.fillna(False) & wx.family.isin(['MeteoHelix', 'MeteoWind'])].copy(); wx['day'] = wx.t.dt.floor('D')
    wx['m1'] = np.where(wx.family == 'MeteoWind', wx.wind_ms, wx.temp_c); wx['m2'] = np.where(wx.family == 'MeteoWind', wx.gust_ms, wx.rh_pct)
    mw = wx.groupby(['device_id', 'day']).agg(m1=('m1', 'mean'), m2=('m2', 'mean')).reset_index()
    R = S.assign(day=S.hour_end.dt.floor('D')).groupby(['device_id', 'day']).agg(risk7=('risk_prolonged_7d', 'max')).reset_index()
    Y = P.merge(B.groupby(['device_id', 'day']).first().reset_index() if B is not None else P[['device_id', 'day']], on=['device_id', 'day'], how='outer')
    Y = Y.merge(pd.concat([ms, mw]), on=['device_id', 'day'], how='left').merge(R, on=['device_id', 'day'], how='left')
    Y.to_parquet(out / 'daily.parquet', index=False)
    # ---- outages, anomalies, battery assessment ----
    E = rd('gold_silences', ['device_id', 'status', 'onset', 'declared_at', 'recovered_at', 'scope_at_declaration', 'cause_hint'])
    E[E.status == 'declared'].drop(columns='status').to_parquet(out / 'outages.parquet', index=False)
    AR = rd('gold_anom_rule_daily'); AR = AR[AR.severity.isin(['high', 'medium'])][['device_id', 'day', 'kind', 'severity']]
    SE = rd('gold_anom_station_events')
    if SE is not None and len(SE):
        SE = SE[(SE.hours >= 24) & (SE.kind != 'spike_vs_neighbours')]
        AR = pd.concat([AR, SE.assign(day=SE.start.dt.floor('D'), kind=SE.kind + ':' + SE.variable.astype(str))[['device_id', 'day', 'kind', 'severity']]])
    AR.to_parquet(out / 'anomalies.parquet', index=False)
    OFF = rd('gold_anom_static_offsets')
    if OFF is not None: OFF[OFF.exceeds_2x_accuracy].to_parquet(out / 'offsets.parquet', index=False)
    BA = rd('gold_battery_assessment')
    if BA is not None: BA[BA.device_id.notna()].to_parquet(out / 'battery_assessment.parquet', index=False)
    # ---- gateways ----
    GH = rd('gold_gw_site_hour', ['gw_site', 'hour_end', 'condition', 'records', 'mu', 'devices', 'dependents'])
    GH.to_parquet(out / 'gateway_hour.parquet', index=False)
    GC = rd('gold_gw_current'); GC.to_parquet(out / 'gateways.parquet', index=False)
    GE = rd('gold_gw_events'); GE.to_parquet(out / 'gateway_events.parquet', index=False)
    # ---- incidents and e-mails ----
    for n in ['incidents', 'notifications']:
        if (rp / f'{n}.parquet').exists(): pd.read_parquet(rp / f'{n}.parquet').to_parquet(out / f'{n}.parquet', index=False)
    if (rp / 'demonstrations.parquet').exists(): shutil.copy(rp / 'demonstrations.parquet', out)
    # e-mail texts (dry-run outbox) so the app can show exactly what was sent
    if (out / 'notifications.parquet').exists():
        import email, email.policy
        N = pd.read_parquet(out / 'notifications.parquet'); txt = []
        for i, pth in zip(N.id, N.path):
            body = None
            if isinstance(pth, str) and Path(pth).exists():
                m = email.message_from_bytes(Path(pth).read_bytes(), policy=email.policy.default)
                part = m.get_body(preferencelist=('plain',)); body = part.get_content() if part is not None else None
            txt.append(dict(id=int(i), text=body))
        pd.DataFrame(txt).to_parquet(out / 'emails.parquet', index=False)
    # ---- how good are the models (test months only) ----
    ev = {}
    T = rd('gold_risk_test_summary')
    if T is not None: ev['risk'] = T[T.n_test > 0].replace({np.nan: None}).to_dict('records')
    I = rd('gold_anom_injection_eval')
    if I is not None: ev['anomaly'] = I.replace({np.nan: None}).to_dict('records')
    Bt = rd('gold_battery_backtest')
    if Bt is not None:
        ev['battery'] = Bt[Bt.role == 'test'].groupby(['family', 'k', 'method']).agg(mae=('abs_err', 'mean'), coverage80=('covered80', 'mean')).reset_index().to_dict('records')
    W = rd('gold_gw_warning_test')
    if W is not None: ev['gateway_warning'] = W.to_dict('records')
    for j in [rp / 'replay_evaluation.json', ed / '..' / 'replay' / 'replay_evaluation.json']:
        if j.exists(): ev['replay'] = json.load(open(j)); break
    m1 = Path(__file__).resolve().parents[2] / 'docs' / 'm1_evaluation.json'
    if m1.exists(): ev['m1'] = json.load(open(m1))
    json.dump(ev, open(out / 'evaluation.json', 'w'), default=str)
    meta = dict(first_hour=str(S.hour_end.min()), last_hour=str(S.hour_end.max()), prepared_at=str(pd.Timestamp.now('UTC').tz_localize(None).floor('s')),
                source_foundation=str(fd), source_replay=str(rp), devices=int(S.device_id.nunique()))
    json.dump(meta, open(out / 'meta.json', 'w'), indent=1)
    return meta

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); ap.add_argument('--replay', required=True)
    ap.add_argument('--out', default='appdata'); ap.add_argument('--eval-foundation')
    a = ap.parse_args(); print(json.dumps(prepare(a.foundation, a.replay, a.out, a.eval_foundation), indent=1))
