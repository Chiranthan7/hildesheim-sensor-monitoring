"""Integration layer: causal device-hour status, root-cause diagnosis (component 5) and maintenance priority (component 6).

Every column is observable at the hour end. Daily quantities (battery, anomaly flags) are used from the NEXT day on
(a day's aggregate is complete only at its end).

Diagnosis: rule-based evidence model. Each candidate cause collects evidence items; the strongest supported cause is primary.
Confidence is an evidence-strength class (high / medium / low), not a calibrated probability: there are no ground-truth cause
labels. Severity (current problem), future risk and diagnostic confidence are separate columns."""
import json
import numpy as np, pandas as pd
from .config import NOMINAL_MIN, LEAD_ACID_REST_V, BATTERY_OBSERVABILITY, const

PRIORITY_WEIGHTS = const('PRIORITY_WEIGHTS', dict(severity_red=100, severity_yellow=40, severity_grey=5, duration_log=12,
                                                    expected_missing_log=6, risk=40, battery_critical=60, battery_low=20, anomaly_high=50,
                                                    anomaly_medium=15, family_importance={}), 'operational', 'descriptive',
                         'maintenance priority score; operational importance per family is configurable (default 1, unknown)')
WARNING_BUDGET = const('WARNING_BUDGET', 0.25, 'operational', 'evaluation',
                       'predictive warnings: threshold = lowest validation threshold with <= 0.25 false alert episodes per sensor-week '
                       '(revised operational policy, chosen after the held-out evaluation; documented as such)')

MANUFACTURER = {'Sensoterra': ('Sensoterra', 'LoRaWAN soil moisture probe'), 'MeteoHelix': ('BARANI DESIGN', 'MeteoHelix IoT Pro weather station'),
                'MeteoWind': ('BARANI DESIGN', 'MeteoWind IoT Pro wind sensor'), 'CloverSoilSensor': ('TEKTELIC', 'CLOVER agriculture sensor'),
                'Agvolution': ('Agvolution', 'soil sensor node (climavi)'), 'OttSensor': ('OTT HydroMet', 'hydrological station (model not identified)'),
                'Bigbelly': ('Bigbelly', 'smart waste station (12 V solar)'), 'Dragino-SE0X-LB': ('Dragino', 'SE0X-LB soil moisture & EC sensor'),
                'Adeunis': ('Adeunis', 'TEMP LoRaWAN temperature sensor')}

def battery_daily_condition(fd):
    V = pd.read_parquet(f'{fd}/gold_battery_daily.parquet').sort_values(['device_id', 'day'])
    V['rest7'] = V.groupby('device_id').v_min.transform(lambda s: s.rolling(7, min_periods=3).median())
    T = LEAD_ACID_REST_V
    la = V.family.isin(['OttSensor', 'Bigbelly'])
    V['battery_condition'] = np.where(la, np.select([V.rest7 < T['critical'], V.rest7 < T['warning'], V.rest7 < T['good']], ['critical', 'low', 'fair'], 'good'),
                                      'nominal_plateau')
    V['battery_value'] = np.where(la, V.rest7, V.v_med)
    LV = pd.read_parquet(f'{fd}/gold_battery_level_daily.parquet')
    LV = LV.assign(battery_condition=np.where(LV.level_pct >= 50, 'good', 'low'), battery_value=LV.level_pct)
    B = pd.concat([V[['device_id', 'day', 'battery_condition', 'battery_value']], LV[['device_id', 'day', 'battery_condition', 'battery_value']]])
    B['avail_day'] = B.day + pd.Timedelta('1D')                    # usable from the next day
    return B

def anomaly_daily(fd):
    R = pd.read_parquet(f'{fd}/gold_anom_rule_daily.parquet')
    R = R[R.severity.isin(['high', 'medium'])]
    a = R.groupby(['device_id', 'day']).agg(anomaly_kinds=('kind', lambda s: ','.join(sorted(set(s)))),
                                            anomaly_high=('severity', lambda s: int((s == 'high').any()))).reset_index()
    E = pd.read_parquet(f'{fd}/gold_anom_station_events.parquet')
    E = E[(E.kind != 'spike_vs_neighbours') & (E.hours >= 24)]
    rows = []
    for r in E.itertuples():
        for d in pd.date_range(r.start.floor('D'), r.end.floor('D'), freq='D'):
            rows.append(dict(device_id=r.device_id, day=d, anomaly_kinds=f'{r.kind}:{r.variable}', anomaly_high=0))
    if rows:
        a = pd.concat([a, pd.DataFrame(rows)]).groupby(['device_id', 'day']).agg(anomaly_kinds=('anomaly_kinds', lambda s: ','.join(sorted(set(','.join(s).split(','))))),
                                                                              anomaly_high=('anomaly_high', 'max')).reset_index()
    a['avail_day'] = a.day + pd.Timedelta('1D')
    return a

def device_status(fd):
    rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    T = rd('gold_m1_state')
    RF = rd('gold_risk_features')[['device_id', 'hour_end', 'beyond_ref_share_24h', 'snr_mean_168h', 'snr_drop_24_vs_168', 'gw_sites_168h',
                                   'primary_gw_share_168h', 'city_rain_6h', 'n_onsets_7d', 'gap_onset']]
    S = T.merge(RF, on=['device_id', 'hour_end'], how='left')
    DH = rd('gold_device_hour')[['device_id', 'hour_end', 'rx_last_arrival', 'svc_last_arrival', 'rx_mode_min']]
    S = S.merge(DH, on=['device_id', 'hour_end'], how='left')
    # risk (selected model per task, walk-forward predictions)
    Q = rd('gold_risk_hourly')
    for task, col in [('A_onset_short', 'risk_short'), ('A_onset_long', 'risk_long'), ('A_prolonged_7d', 'risk_prolonged_7d'),
                      ('A_prolonged_14d', 'risk_prolonged_14d'), ('B_developing', 'p_silence_becomes_outage'), ('D_continuation_24h', 'p_still_down_24h'),
                      ('D_continuation_168h', 'p_still_down_7d')]:
        q = Q[Q.task == task][['device_id', 'hour_end', 'p', 'H']].rename(columns={'p': col, 'H': col + '_H'})
        S = S.merge(q, on=['device_id', 'hour_end'], how='left')
    # gateway context
    Dp = rd('gold_gw_device_primary')[['device_id', 'hour_end', 'primary', 'primary_share_168h', 'n_sites_672h']].rename(columns={'primary': 'primary_gateway'})
    G = rd('gold_gw_site_hour')[['gw_site', 'hour_end', 'condition']].rename(columns={'gw_site': 'primary_gateway', 'condition': 'gateway_condition'})
    S = S.merge(Dp, on=['device_id', 'hour_end'], how='left').merge(G, on=['primary_gateway', 'hour_end'], how='left')
    # battery and anomalies (previous complete day)
    S['day_avail'] = S.hour_end.dt.floor('D')
    B = battery_daily_condition(fd).rename(columns={'avail_day': 'day_avail'})[['device_id', 'day_avail', 'battery_condition', 'battery_value']]
    S = S.merge(B, on=['device_id', 'day_avail'], how='left')
    S['battery_condition'] = S.battery_condition.fillna(S.family.map(lambda f: 'not_measurable' if BATTERY_OBSERVABILITY[f]['signal'] is None else 'no_recent_reading'))
    A = anomaly_daily(fd).rename(columns={'avail_day': 'day_avail'})[['device_id', 'day_avail', 'anomaly_kinds', 'anomaly_high']]
    S = S.merge(A, on=['device_id', 'day_avail'], how='left'); S['anomaly_high'] = S.anomaly_high.fillna(0)
    return S.drop(columns=['day_avail'])

def diagnose(S):
    """primary probable cause, confidence, evidence, alternatives (vectorised rules, priority order of evidence strength)"""
    st = S.state; sc = S.declared_scope.fillna(''); gwc = S.gateway_condition.fillna('')
    weak_radio = (S.beyond_ref_share_24h >= 0.2) | (S.snr_mean_168h < -15)
    contact = (S.family == 'Sensoterra') & (S.reduced_mode | (S.meas_zero_share_24h >= 0.5))
    chan = S.reasons.fillna('').str.contains('channels_missing') | S.anomaly_kinds.fillna('').str.contains('channel_never_valid|raw_frequency_fault|frozen|out_of_manufacturer')
    batt = S.battery_condition.isin(['critical', 'low'])
    problem = st.isin(['significant_interruption', 'backend_data_unavailable', 'degraded', 'reduced_reporting_mode']) | (S.anomaly_high > 0) | batt | S.anomaly_kinds.notna()
    rules = [
        (st == 'backend_data_unavailable', 'backend storage failure', 'high', 'uplinks received after the last stored measurement (sensor and radio working, data path failing)'),
        (sc.eq('network') | gwc.eq('network_or_platform'), 'network / platform failure', 'high', 'most gateway sites silent in the same hour'),
        (sc.eq('gateway') | gwc.isin(['inactive_gateway', 'inactive_unexplained']), 'gateway unavailability', 'medium', 'primary gateway inactive (anomalous zero activity)'),
        (sc.eq('fleet'), 'fleet-wide stop (possible remote deactivation or platform change)', 'medium', 'improbably many devices of the family stopped together'),
        (contact, 'poor installation / soil contact', 'medium', 'reduced daily reporting mode or majority of moisture readings = 0 (manufacturer-documented contact / dry-soil signature)'),
        (chan & problem, 'measurement-channel failure', 'medium', 'missing / invalid / frozen measurement channel'),
        (batt & st.isin(['normal', 'degraded', 'reduced_reporting_mode']), 'low battery charge (measured)', 'medium', 'resting-voltage proxy in the low / critical band of the lead-acid reference'),
        (batt & problem, 'battery deterioration (possible)', 'low', 'resting-voltage proxy in the low / critical band'),
        (weak_radio & problem, 'weak radio connectivity', 'medium', 'receptions at / beyond SF12 reference limits or low SNR'),
        ((st == 'significant_interruption') & sc.eq('individual'), 'individual sensor or local communication failure (cause undetermined)', 'low',
         'only this device silent; gateway and network active; no backend signature'),
        (st == 'degraded', 'elevated packet loss (cause undetermined)', 'low', 'availability forecast below the family threshold'),
        (S.anomaly_kinds.notna(), 'measurement anomaly', 'low', 'statistical or rule-based measurement anomaly (see anomaly list)'),
    ]
    prim = np.full(len(S), None, object); conf = np.full(len(S), None, object); evid = np.full(len(S), '', object); alts = [[] for _ in range(len(S))]
    for m, cause, c, e in rules:
        m = m.fillna(False).values & problem.values
        new = m & pd.isna(prim)
        prim[new] = cause; conf[new] = c; evid[new] = e
        for i in np.where(m & ~new)[0]: alts[i].append(cause)
    prim = np.where(problem.values & pd.isna(prim), 'unknown', prim); conf = np.where(prim == 'unknown', 'low', conf)
    S = S.copy(); S['probable_cause'] = prim; S['cause_confidence'] = conf; S['cause_evidence'] = evid; S['alternative_causes'] = [';'.join(a) for a in alts]
    return S

ACTIONS = {'low battery charge (measured)': 'check charging (solar panel, controller) and plan battery replacement; voltage is below the lead-acid ~50 % reference',
           'measurement anomaly': 'review the anomaly; compare with neighbouring sensors / reference',
           'backend storage failure': 'fix the ingestion pipeline (database insert errors); no field visit; stored values can be re-ingested',
           'network / platform failure': 'check network server / platform and backhaul; no field visit until the platform is ruled out',
           'gateway unavailability': 'check gateway power, backhaul and packet forwarder',
           'fleet-wide stop (possible remote deactivation or platform change)': 'confirm with the operator whether devices were deactivated or re-provisioned',
           'poor installation / soil contact': 're-seat the probe (manufacturer: poor pin contact or too-dry soil)',
           'measurement-channel failure': 'inspect the probe / channel wiring; replace probe if the fault persists',
           'battery deterioration (possible)': 'check battery and charging (solar panel, controller); plan replacement',
           'weak radio connectivity': 'improve the radio link (antenna position, gateway coverage); check SF / ADR settings',
           'individual sensor or local communication failure (cause undetermined)': 'site visit: check power, enclosure and antenna',
           'elevated packet loss (cause undetermined)': 'monitor; inspect antenna and gateway coverage if it persists',
           'unknown': 'review evidence'}

def overall(S, warn_thr):
    """overall traffic light = worst of communication state, battery, measurement anomalies and elevated future risk.
    communication light (Module 1) is kept as traffic_light; overall_light drives the fleet view, priority and dashboard."""
    S = S.copy(); rank = {'green': 0, 'grey': 0, 'yellow': 1, 'red': 2}; inv = {0: 'green', 1: 'yellow', 2: 'red'}
    lvl = S.traffic_light.map(rank).fillna(0).values.copy()
    t = S.family.map(lambda f: warn_thr.get(('A_prolonged_7d', f), np.nan)).astype(float)
    S['elevated_risk'] = (S.risk_prolonged_7d >= t).fillna(False) & S.state.isin(['normal', 'degraded'])
    lvl = np.maximum(lvl, np.where((S.battery_condition == 'critical') | (S.anomaly_high > 0), 2, 0))
    lvl = np.maximum(lvl, np.where((S.battery_condition == 'low') | S.anomaly_kinds.notna() | S.elevated_risk, 1, 0))
    S['overall_light'] = np.where((S.traffic_light == 'grey') & (lvl == 0), 'grey', [inv[int(x)] for x in lvl])
    return S

def priority(snap):
    """maintenance priority list ('Diese Sensoren jetzt pruefen') at one moment: one row per device, score with reasons"""
    W = PRIORITY_WEIGHTS; s = snap.copy()
    dur_h = np.where(s.state.isin(['significant_interruption', 'backend_data_unavailable']), s.silence_svc_slots * s.family.map(NOMINAL_MIN) / 60, 0)
    risk = np.nanmax(np.c_[s.risk_prolonged_7d.fillna(0), s.risk_long.fillna(0) * 0.5], axis=1)
    sev = s[('overall_light' if 'overall_light' in s else 'traffic_light')].map({'red': W['severity_red'], 'yellow': W['severity_yellow'], 'grey': W['severity_grey']}).fillna(0)
    imp = s.family.map(lambda f: W['family_importance'].get(f, 1.0))
    s['priority_score'] = imp * (sev + W['duration_log'] * np.log1p(dur_h) + W['expected_missing_log'] * np.log1p(s.expected_missing_primary.fillna(0))
                                 + W['risk'] * risk + np.where(s.battery_condition == 'critical', W['battery_critical'], np.where(s.battery_condition == 'low', W['battery_low'], 0))
                                 + np.where(s.anomaly_high > 0, W['anomaly_high'], np.where(s.anomaly_kinds.notna(), W['anomaly_medium'], 0)))
    s['unavailable_h'] = dur_h
    why = []
    for r in s.itertuples():
        w = []
        if r.state in ('significant_interruption', 'backend_data_unavailable'): w.append(f'{r.state.replace("_", " ")} for {r.unavailable_h:.0f} h')
        elif r.state == 'degraded': w.append(f'degraded: forecast {r.forecast_avail:.0%} vs family threshold {r.tau:.0%}')
        elif r.state == 'reduced_reporting_mode': w.append('reduced daily reporting mode')
        if np.isfinite(r.risk_prolonged_7d) and r.risk_prolonged_7d >= 0.3: w.append(f'prolonged-interruption risk 7 d {r.risk_prolonged_7d:.0%}')
        if r.battery_condition in ('critical', 'low'): w.append(f'battery {r.battery_condition}')
        if isinstance(r.anomaly_kinds, str): w.append('anomalies: ' + r.anomaly_kinds)
        why.append('; '.join(w))
    s['why'] = why; s['recommended_action'] = s.probable_cause.map(ACTIONS).fillna('none')
    return s.sort_values('priority_score', ascending=False)

def gateway_priority(Gc):
    g = Gc.copy()
    bad = g.condition.isin(['inactive_gateway', 'inactive_unexplained', 'network_or_platform'])
    g['priority_score'] = np.where(bad, 100 + 12 * np.log1p(g.hours_since_activity.fillna(0)) + 5 * g.dependents.fillna(0), np.where(g.condition == 'reduced_activity', 40, 0))
    return g.sort_values('priority_score', ascending=False)

def live_snapshot(fd, model_dir, warn_thr):
    """latest-hour status for live monitoring: Module 1 state from the analytic models (hierarchical EB-Markov forecast, no
    walk-forward table), production risk models, gateway / battery / anomaly components, diagnosis and overall light."""
    from .m1 import analytic, features, state
    from . import risk, production
    rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    H = rd('gold_device_hour'); L = rd('gold_silences'); U = rd('gold_uplinks')
    A = analytic.build(L, rd('gold_hmm_fits'), H); X = features.build(H, L, U, rd('silver_soil'), A)
    state.period_stats(H, U, L, 'D').to_parquet(f'{fd}/gold_m1_period_daily.parquet', index=False)
    T = state.timeline(X, pd.DataFrame(columns=['family', 'H', 'fold', 'device_id', 'hour_end']), {})
    X2 = risk.extra_features(X, L, U, rd('silver_devices'), rd('silver_soil'), rd('silver_weather'), rd('gold_site_hour'))
    X2 = pd.concat([X2, risk.current_gap(X2, L)], axis=1); X2.to_parquet(f'{fd}/gold_risk_features.parquet', index=False)
    Tc = T[T.hour_complete].sort_values(['device_id', 'hour_end'])
    brk = (Tc.state != 'degraded').groupby(Tc.device_id).cumsum()                     # consecutive degraded hours up to each hour
    Tc = Tc.assign(degraded_run_h=(Tc.state == 'degraded').astype(int).groupby([Tc.device_id, brk]).cumsum())
    last = Tc.groupby('device_id').tail(1)
    sc = production.score(fd, model_dir, hours=None)
    S = last.merge(X2[['device_id', 'hour_end', 'beyond_ref_share_24h', 'snr_mean_168h', 'snr_drop_24_vs_168', 'gw_sites_168h', 'primary_gw_share_168h',
                       'city_rain_6h', 'n_onsets_7d', 'gap_onset']], on=['device_id', 'hour_end'], how='left')
    S = S.merge(sc.drop(columns=['family', 'hour_end']), on='device_id', how='left')
    for c in ['risk_short', 'risk_prolonged_14d']: S[c] = np.nan
    S['risk_short_H'] = np.nan; S['risk_long_H'] = S.family.map(lambda f: risk.RISK_HORIZONS_H[f][1])
    DH = H[['device_id', 'hour_end', 'rx_last_arrival', 'svc_last_arrival', 'rx_mode_min']]
    S = S.merge(DH, on=['device_id', 'hour_end'], how='left')
    Dp = rd('gold_gw_device_primary')[['device_id', 'hour_end', 'primary', 'primary_share_168h', 'n_sites_672h']].rename(columns={'primary': 'primary_gateway'})
    G = rd('gold_gw_site_hour')[['gw_site', 'hour_end', 'condition']].rename(columns={'gw_site': 'primary_gateway', 'condition': 'gateway_condition'})
    S = S.merge(Dp, on=['device_id', 'hour_end'], how='left').merge(G, on=['primary_gateway', 'hour_end'], how='left')
    S['day_avail'] = S.hour_end.dt.floor('D')
    B = battery_daily_condition(fd).rename(columns={'avail_day': 'day_avail'})[['device_id', 'day_avail', 'battery_condition', 'battery_value']]
    S = S.merge(B, on=['device_id', 'day_avail'], how='left')
    S['battery_condition'] = S.battery_condition.fillna(S.family.map(lambda f: 'not_measurable' if BATTERY_OBSERVABILITY[f]['signal'] is None else 'no_recent_reading'))
    An = anomaly_daily(fd).rename(columns={'avail_day': 'day_avail'})[['device_id', 'day_avail', 'anomaly_kinds', 'anomaly_high']]
    S = S.merge(An, on=['device_id', 'day_avail'], how='left'); S['anomaly_high'] = S.anomaly_high.fillna(0)
    return overall(diagnose(S.drop(columns=['day_avail'])), warn_thr)
