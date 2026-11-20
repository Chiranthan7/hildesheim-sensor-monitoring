"""Historical causal replay (component 8): the integrated system runs hour by hour on the historical exports, seeing at each hour
only what was observable then (all inputs are causal tables), with dry-run e-mail delivery. Evaluation of notifications and
demonstration cases ('what the system knew' vs 'what happened next')."""
import json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from . import system, risk
from .notify.engine import Engine, Store
from .notify.mailer import Mailer
from .config import NOMINAL_MIN

def warning_thresholds(fd, budget):
    """revised operational policy: per family, the lowest validation threshold of the prolonged-interruption risk whose validation
    false-alert rate is <= budget episodes per sensor-week (validation predictions only)"""
    PA = pd.read_parquet(f'{fd}/gold_risk_pred_A.parquet'); E = risk.events(pd.read_parquet(f'{fd}/gold_silences.parquet'))
    R = pd.read_parquet(f'{fd}/gold_risk_test_summary.parquet'); thr = {}
    for r in R[R.task == 'A_prolonged_7d'].itertuples():
        v = PA[(PA.task == r.task) & (PA.family == r.family) & (PA.role == 'validation')]
        if not len(v) or v.y.sum() < 5: continue
        for t in np.quantile(v[r.selected].dropna(), [0.9, 0.95, 0.97, 0.98, 0.99, 0.995]):
            m = risk.event_metrics(v, r.selected, t, E, int(r.H))
            if m['false_alerts_per_sensor_week'] <= budget: thr[('A_prolonged_7d', r.family)] = float(t); break
    for r in R[R.task == 'B_developing'].itertuples():
        if np.isfinite(r.threshold): thr[('B_developing', r.family)] = float(r.threshold)
    return thr

def run(fd, out, start='2026-08-01', cfg=None):
    out = Path(out); shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    from .system import WARNING_BUDGET
    thr = warning_thresholds(fd, WARNING_BUDGET); json.dump({f'{k[0]}|{k[1]}': v for k, v in thr.items()}, open(out / 'warning_thresholds.json', 'w'), indent=1)
    S = system.overall(system.diagnose(system.device_status(fd)), thr); S.to_parquet(out / 'sys_device_hour.parquet', index=False)
    G = pd.read_parquet(f'{fd}/gold_gw_site_hour.parquet')
    cfg = cfg or dict(enabled=False, dry_run=True, outbox_dir=str(out / 'outbox'), recipients=['admin@example.org'], escalation_recipients=['lead@example.org'],
                      cooldown_h=6, min_clear_h={'SENSOR_DOWN': 2, 'BACKEND_FAILURE': 12, 'GATEWAY_DOWN': 2, 'NETWORK_OUTAGE': 1, 'REDUCED_REPORTING_MODE': 48, 'default': 24}, risk_hysteresis=0.7,
                      reminder_h={'critical': 24, 'warning': 72}, reminder_max_h=168, flapping_onsets_7d=3, flapping_clear_h=24, early_silence_warning=False, escalate_h=48, degraded_min_h=24, warnings_enabled=True, dashboard_url='')
    assert cfg.get('dry_run', True) or not cfg.get('enabled'), 'replay must never send real e-mail'
    cfg = dict(cfg, enabled=False, dry_run=True)                        # hard guarantee
    eng = Engine(cfg, Store(str(out / 'notifications.sqlite')), Mailer(cfg))
    hours = pd.date_range(pd.Timestamp(start), S.hour_end.max(), freq='h')
    Sd = {h: g for h, g in S[S.hour_end >= pd.Timestamp(start)].groupby('hour_end')}
    Gd = {h: g for h, g in G.groupby('hour_end')}
    run_h = {}; log = []
    for h in hours:
        s = Sd.get(h, S.iloc[:0]); g = Gd.get(h, G.iloc[:0])
        obs = s[s.hour_complete & (s.state != 'insufficient_information')]
        unobserved = set(s.loc[~s.hour_complete | (s.state == 'insufficient_information'), 'device_id'])
        for r in g.itertuples(): run_h[r.gw_site] = run_h.get(r.gw_site, 0) + 1 if r.condition in ('inactive_gateway', 'inactive_unexplained') else 0
        run_h['NETWORK'] = run_h.get('NETWORK', 0) + 1 if (g.condition == 'network_or_platform').any() else 0
        sent = eng.step(h, obs, g, thr, run_h, frozenset(unobserved))
        for x in sent: log.append(dict(hour=h, **x))
    L = pd.DataFrame(log); L.to_parquet(out / 'notification_log.parquet', index=False)
    I = pd.read_sql('select * from incidents', eng.store.db); I.to_parquet(out / 'incidents.parquet', index=False)
    N = pd.read_sql('select * from notifications', eng.store.db); N.to_parquet(out / 'notifications.parquet', index=False)
    return S, I, N, thr

def evaluate(fd, out, S, I, N, start='2026-08-01'):
    """notification quality on the replay"""
    E = risk.events(pd.read_parquet(f'{fd}/gold_silences.parquet')); E = E[E.onset >= pd.Timestamp(start)]
    res = {}
    res['emails_by_kind'] = N.kind.value_counts().to_dict(); res['incidents_by_type'] = I.type.value_counts().to_dict()
    res['grouped_emails'] = int((N.incident_ids.map(lambda x: len(json.loads(x))) > 1).sum())
    # duplicates: two e-mails of the same kind for the same incident closer than the reminder interval
    rows = []
    for n in N.itertuples():
        for i in json.loads(n.incident_ids): rows.append(dict(incident=i, kind=n.kind, t=pd.Timestamp(n.sent_at)))
    M = pd.DataFrame(rows).merge(I[['id', 'severity', 'type']], left_on='incident', right_on='id')
    M = M.sort_values(['incident', 'kind', 't']); gap = M.groupby(['incident', 'kind']).t.diff().dt.total_seconds() / 3600
    lim = M.severity.map({'critical': 24, 'warning': 72})
    res['duplicate_email_rate'] = float(((M.kind.isin(['OPEN', 'REMINDER'])) & (gap < lim)).sum() / max(len(M), 1))
    res['max_emails_per_incident'] = int(M.groupby('incident').size().max())
    # coverage of actual unavailability: declared outages lasting >= 6 h must produce a SENSOR_DOWN / BACKEND incident (open or reopened)
    D = I[I.type.isin(['SENSOR_DOWN', 'BACKEND_FAILURE'])]
    cov = []
    for e in E[(E.duration_h >= 6) | E.recovered_at.isna()].itertuples():
        d = D[(D.entity == e.device_id) & (pd.to_datetime(D.opened_at) <= (e.recovered_at if pd.notna(e.recovered_at) else S.hour_end.max()))
              & (pd.to_datetime(D.closed_at).fillna(pd.Timestamp('2262-01-01')) >= e.declared_at)]
        first = pd.to_datetime(d.opened_at).min() if len(d) else pd.NaT
        ongoing_at_check = (pd.isna(e.recovered_at)) or (e.recovered_at > e.declared_at.ceil('h'))
        cov.append(dict(device_id=e.device_id, family=e.family, onset=e.onset, declared_at=e.declared_at, covered=len(d) > 0, ongoing_at_hourly_check=ongoing_at_check,
                        delay_from_onset_h=(first - e.onset).total_seconds() / 3600 if len(d) else np.nan, risk_before=np.nan))
    C = pd.DataFrame(cov); C.to_parquet(Path(out) / 'outage_coverage.parquet', index=False)
    res['outages_ge6h'] = len(C); res['outages_ge6h_notified_share'] = float(C.covered.mean()) if len(C) else np.nan
    oc = C[C.ongoing_at_hourly_check]; res['outages_ge6h_ongoing_at_an_hourly_check'] = len(oc)
    res['outages_ge6h_ongoing_notified_share'] = float(oc.covered.mean()) if len(oc) else np.nan
    res['median_notification_delay_from_onset_h'] = float(C.delay_from_onset_h.median()) if len(C) else np.nan
    # premature recoveries: a recovery followed by the same incident type for the same entity within 24 h
    R = I[I.closed_at.notna()]
    pre = 0
    for r in R.itertuples():
        nxt = I[(I.key == r.key) & (pd.to_datetime(I.opened_at) > pd.Timestamp(r.closed_at)) & (pd.to_datetime(I.opened_at) <= pd.Timestamp(r.closed_at) + pd.Timedelta('24h'))]
        pre += len(nxt) > 0
    res['recoveries'] = int(len(R)); res['premature_recovery_rate_24h'] = float(pre / max(len(R), 1))
    pr = []
    for r in R.itertuples():
        pr.append(dict(type=r.type, premature=len(I[(I.key == r.key) & (pd.to_datetime(I.opened_at) > pd.Timestamp(r.closed_at)) & (pd.to_datetime(I.opened_at) <= pd.Timestamp(r.closed_at) + pd.Timedelta('24h'))]) > 0))
    res['premature_recovery_rate_by_type'] = pd.DataFrame(pr).groupby('type').premature.mean().round(3).to_dict() if pr else {}
    weeks = (S.hour_end.max() - pd.Timestamp(start)).total_seconds() / (168 * 3600); nd = S.device_id.nunique()
    res['emails_per_device_week'] = float(len(N) / (nd * weeks))
    # predictive warnings (RISK_ELEVATED): followed by a prolonged outage onset within 7 d?
    W = I[I.type == 'RISK_ELEVATED']; P7 = E[E.duration_h >= 24]
    hit = [((P7.device_id == w.entity) & (P7.onset > pd.Timestamp(w.opened_at)) & (P7.onset <= pd.Timestamp(w.opened_at) + pd.Timedelta('7D'))).any() for w in W.itertuples()]
    res['risk_warnings'] = int(len(W)); res['risk_warning_precision_7d'] = float(np.mean(hit)) if len(W) else np.nan
    pre_w = [((W.entity == p.device_id) & (pd.to_datetime(W.opened_at) < p.onset) & (pd.to_datetime(W.opened_at) >= p.onset - pd.Timedelta('7D'))).any() for p in P7.itertuples()]
    res['prolonged_outages'] = int(len(P7)); res['prolonged_outages_warned_before_share'] = float(np.mean(pre_w)) if len(P7) else np.nan
    json.dump(res, open(Path(out) / 'replay_evaluation.json', 'w'), indent=1, default=str)
    return res

CASES = [
    ('backend_sensoterra_13nov', 'BACKEND_FAILURE', 'Sensoterra', '2026-11-13', 'Sensoterra probes heard but inserts failing (ProgrammingError) since 13 Nov 12:55'),
    ('backend_dragino', 'BACKEND_FAILURE', 'Dragino-SE0X-LB', None, 'Dragino failed inserts (29 Sep - 5 Oct): heard but not stored'),
    ('network_outage_29sep', 'NETWORK_OUTAGE', None, '2026-09-29', 'platform / network outage 29 Sep 07:30-10:30'),
    ('fleet_stop', 'SENSOR_DOWN', 'FLEET', None, 'several devices of one family stopped together (fleet-wide)'),
    ('gateway_down_52.1528', 'GATEWAY_DOWN', 'unlisted_52.1528_9.9570', '2026-11-05', 'gateway site 52.1528/9.9570 silent since 5 Nov, sensors re-routed'),
    ('battery_low_bigbelly', 'BATTERY_LOW', 'Bigbelly', None, 'Bigbelly resting-voltage proxy below 12.0 V'),
    ('frozen_temperature', 'SEVERE_MALFUNCTION', 'Sensoterra', None, 'Sensoterra probe with frozen / out-of-range temperature'),
    ('station_bias', 'MEASUREMENT_DEGRADATION', 'MeteoHelix', None, 'weather station drifting against its neighbours'),
    ('predictive_warning', 'RISK_ELEVATED', 'Sensoterra', None, 'elevated prolonged-interruption risk before an outage'),
    ('reduced_mode', 'REDUCED_REPORTING_MODE', 'Sensoterra', None, 'Sensoterra switched to the daily reporting mode'),
]

def demonstrations(fd, out, S):
    out = Path(out); I = pd.read_parquet(out / 'incidents.parquet'); N = pd.read_parquet(out / 'notifications.parquet')
    E = risk.events(pd.read_parquet(f'{fd}/gold_silences.parquet'))
    fam = S.groupby('device_id').family.first()
    rows = []; (out / 'demo_emails').mkdir(exist_ok=True)
    for name, typ, who, day, desc in CASES:
        c = I[I.type == typ].copy(); c['opened'] = pd.to_datetime(c.opened_at)
        if who in (fam.unique().tolist()): c = c[c.entity.map(fam) == who]
        elif who == 'FLEET': c = c[c.group_key.fillna('').str.startswith('FLEET')]
        elif who: c = c[c.entity == who]
        c = c[c.opened > c.opened.min() + pd.Timedelta('1D')] if c.opened.min() == pd.Timestamp('2026-08-01') else c   # skip conditions inherited at replay start
        if day: c = c[c.opened.dt.floor('D') == pd.Timestamp(day)]
        if typ == 'RISK_ELEVATED':                                         # pick a warning followed by a prolonged outage
            ok = [((E.device_id == r.entity) & (E.onset > r.opened) & (E.onset <= r.opened + pd.Timedelta('7D')) & (E.duration_h >= 24)).any() for r in c.itertuples()]
            c = c[ok]
        if not len(c): rows.append(dict(case=name, description=desc, found=False)); continue
        inc = c.sort_values('opened').iloc[0]
        n = N[N.incident_ids.map(lambda x: int(inc.id) in json.loads(x))].sort_values('sent_at')
        first = n.iloc[0] if len(n) else None
        knew = S[(S.device_id == inc.entity) & (S.hour_end == inc.opened)]
        k = knew.iloc[0] if len(knew) else None
        if typ == 'RISK_ELEVATED': after = E[(E.device_id == inc.entity) & (E.onset > inc.opened) & (E.onset <= inc.opened + pd.Timedelta('7D'))].head(1)
        else: after = E[(E.device_id == inc.entity) & (E.onset <= inc.opened + pd.Timedelta('7D')) & ((E.recovered_at.isna()) | (E.recovered_at >= inc.opened - pd.Timedelta('1D')))]
        r = dict(case=name, description=desc, found=True, entity=inc.entity, type=typ, detected_at=inc.opened, closed_at=inc.closed_at, n_emails=len(n),
                 first_email=first.subject if first is not None else None, grouped_devices=len(json.loads(first.incident_ids)) if first is not None else 0,
                 knew_state=k.state if k is not None else None, knew_cause=k.probable_cause if k is not None else None,
                 knew_confidence=k.cause_confidence if k is not None else None, knew_evidence=k.cause_evidence if k is not None else None,
                 knew_risk_prolonged_7d=float(k.risk_prolonged_7d) if k is not None and pd.notna(k.risk_prolonged_7d) else None,
                 knew_battery=k.battery_condition if k is not None else None, knew_anomalies=k.anomaly_kinds if k is not None else None,
                 happened_outage_onset=after.onset.min() if len(after) else None,
                 happened_outage_duration_h=(f'{after.duration_h.max():.1f} h' if after.duration_h.notna().any() else 'ongoing at export end') if len(after) else None,
                 happened_recovered_at=after.recovered_at.max() if len(after) else None)
        if first is not None and isinstance(first.path, str): shutil.copy(first.path, out / 'demo_emails' / f'{name}.eml')
        rows.append(r)
    D = pd.DataFrame(rows); D.to_parquet(out / 'demonstrations.parquet', index=False)
    return D
