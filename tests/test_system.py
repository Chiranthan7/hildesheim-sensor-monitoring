"""Tests for components 2-10: notifications (no real e-mail is ever sent), mailer, risk utilities, gateway causality,
battery (no percentages / no invented crossing dates), anomaly detectors, integration (overall light, priority), dashboard."""
import os, re, json, smtplib
from pathlib import Path
import numpy as np, pandas as pd, pytest

from hsm.notify.mailer import Mailer
from hsm.notify.engine import Engine, Store
from hsm import risk, battery, anomaly, system, gateway

ROOT = Path(__file__).resolve().parent.parent
FD = Path(os.environ.get('HSM_FOUNDATION', '/home/claude/foundation'))
RP = Path(os.environ.get('HSM_REPLAY', '/home/claude/replay'))
need_fd = pytest.mark.skipif(not (FD / 'gold_radio_records.parquet').exists(), reason='foundation not built')
need_rp = pytest.mark.skipif(not (RP / 'sys_device_hour.parquet').exists(), reason='replay not run')

# ------------------------------------------------------------------ helpers
def cfg(tmp, **kw):
    c = dict(enabled=False, dry_run=True, outbox_dir=str(tmp / 'outbox'), recipients=['admin@example.org'], escalation_recipients=['lead@example.org'],
             cooldown_h=6, min_clear_h={'SENSOR_DOWN': 2, 'BACKEND_FAILURE': 12, 'GATEWAY_DOWN': 2, 'NETWORK_OUTAGE': 1, 'default': 24},
             risk_hysteresis=0.7, reminder_h={'critical': 24, 'warning': 72}, reminder_max_h=168, escalate_h=48, degraded_min_h=24, warnings_enabled=True)
    c.update(kw); return c

def row(dev='d1', fam='MeteoHelix', state='normal', **kw):
    r = dict(device_id=dev, family=fam, state=state, traffic_light={'normal': 'green', 'degraded': 'yellow'}.get(state, 'red'), reasons='',
             probable_cause='individual sensor or local communication failure (cause undetermined)' if state == 'significant_interruption' else None,
             cause_confidence='low', cause_evidence='', primary_gateway='siteA', gateway_condition='active', battery_condition='not_measurable',
             anomaly_high=0, anomaly_kinds=None, p_silence_becomes_outage=np.nan, risk_prolonged_7d=np.nan, risk_long=np.nan, p_still_down_24h=np.nan,
             p_still_down_7d=np.nan, n_onsets_7d=0.0, silence_svc_slots=10.0 if state == 'significant_interruption' else 0.0,
             rx_last_arrival=pd.Timestamp('2026-10-01'), svc_last_arrival=pd.Timestamp('2026-10-01'), rx_mode_min=10.0, expected_missing_primary=0.0, primary_H=72)
    r.update(kw); return r

S_ = lambda *rows: pd.DataFrame(list(rows))
G0 = pd.DataFrame(columns=['gw_site', 'condition', 'mu', 'dependents', 'dep_heard_elsewhere'])
T0 = pd.Timestamp('2026-10-01 00:00')
H = lambda h: T0 + pd.Timedelta(hours=h)

def engine(tmp, **kw):
    c = cfg(tmp, **kw); return Engine(c, Store(str(tmp / 'n.sqlite')), Mailer(c))

def kinds(sent): return [s['kind'] for s in sent]

# ------------------------------------------------------------------ mailer
def test_mailer_default_is_dry_run(tmp_path):
    m = Mailer(dict(outbox_dir=str(tmp_path)))
    assert m.dry
    r = m.send(['a@example.org'], 'subj', 'body', '<p>body</p>', tag='t1')
    assert r['status'] == 'dry_run' and Path(r['path']).exists() and b'Subject: subj' in Path(r['path']).read_bytes()

def test_mailer_enabled_but_dry_run_stays_dry(tmp_path, monkeypatch):
    monkeypatch.setattr(smtplib, 'SMTP', lambda *a, **k: (_ for _ in ()).throw(AssertionError('SMTP must not be contacted')))
    m = Mailer(dict(enabled=True, dry_run=True, outbox_dir=str(tmp_path)))
    assert m.send(['a@x'], 's', 'b')['status'] == 'dry_run'

def test_mailer_missing_host_fails_without_network(tmp_path, monkeypatch):
    for v in ['HSM_SMTP_HOST', 'HSM_SMTP_USER', 'HSM_SMTP_PASSWORD']: monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(smtplib, 'SMTP', lambda *a, **k: (_ for _ in ()).throw(AssertionError('SMTP must not be contacted')))
    r = Mailer(dict(enabled=True, dry_run=False, outbox_dir=str(tmp_path))).send(['a@x'], 's', 'b')
    assert r['status'] == 'failed' and 'not configured' in r['error']

def test_mailer_smtp_credentials_from_env_and_retries(tmp_path, monkeypatch):
    monkeypatch.setenv('HSM_SMTP_HOST', 'smtp.test'); monkeypatch.setenv('HSM_SMTP_USER', 'u'); monkeypatch.setenv('HSM_SMTP_PASSWORD', 'secret-from-env')
    calls = dict(n=0, login=None, sent=0)
    class Fake:
        def __init__(self, host, port, timeout):
            calls['n'] += 1
            if calls['n'] < 3: raise ConnectionError('temporary')
            assert host == 'smtp.test' and port == 587
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self, context): pass
        def login(self, u, p): calls['login'] = (u, p)
        def send_message(self, m): calls['sent'] += 1
    monkeypatch.setattr(smtplib, 'SMTP', Fake); slept = []
    m = Mailer(dict(enabled=True, dry_run=False, outbox_dir=str(tmp_path)), sleep=slept.append)
    r = m.send(['a@x'], 's', 'b')
    assert r['status'] == 'sent' and r['attempts'] == 3 and calls['login'] == ('u', 'secret-from-env') and calls['sent'] == 1 and slept == [2, 4]

def test_mailer_gives_up_after_max_attempts(tmp_path, monkeypatch):
    monkeypatch.setenv('HSM_SMTP_HOST', 'smtp.test')
    def boom(*a, **k): raise ConnectionError('down')
    monkeypatch.setattr(smtplib, 'SMTP', boom)
    r = Mailer(dict(enabled=True, dry_run=False, outbox_dir=str(tmp_path), smtp=dict(max_attempts=3)), sleep=lambda s: None).send(['a@x'], 's', 'b')
    assert r['status'] == 'failed' and r['attempts'] == 3 and 'down' in r['error']

def test_no_hardcoded_credentials_in_source():
    pat = re.compile(r"(password|passwd|smtp_pass)\s*[=:]\s*['\"][^'\"]+['\"]", re.I)
    for p in list((ROOT / 'hsm').rglob('*.py')) + list((ROOT / 'config').glob('*.yaml')) + list(ROOT.glob('*.py')):
        assert not pat.search(p.read_text()), f'possible hard-coded credential in {p}'
    ex = (ROOT / 'config' / 'hsm.example.yaml').read_text()
    assert 'enabled: false' in ex and 'dry_run: true' in ex

def test_replay_forces_dry_run():
    src = (ROOT / 'hsm' / 'replay.py').read_text()
    assert "dict(cfg, enabled=False, dry_run=True)" in src

# ------------------------------------------------------------------ incident lifecycle
def test_open_reminder_backoff_escalation_ack_recovery(tmp_path):
    e = engine(tmp_path); down = S_(row(state='significant_interruption'))
    log = {}
    for h in range(0, 160): log[h] = kinds(e.step(H(h), down, G0, {}, {}))
    assert log[0] == ['OPEN']
    assert all(log[h] == [] for h in range(1, 24))
    assert log[24] == ['REMINDER']
    assert 'ESCALATION' in log[48] and 'REMINDER' not in sum((log[h] for h in range(25, 72)), [])   # back-off: next reminder 48 h after the first
    # the escalation is itself a notification and counts in the back-off: next reminder 24 x 2^2 = 96 h after it
    assert [h for h, k in log.items() if 'REMINDER' in k] == [24, 144]
    assert sum(len(k) for k in log.values()) == 4                                  # OPEN, REMINDER, ESCALATION, REMINDER in 160 h
    iid = e.store.q("select id from incidents where status='open'")[0][0]; e.acknowledge(iid, H(160))
    ok = S_(row(state='normal'))
    assert kinds(e.step(H(161), ok, G0, {}, {})) == []                           # needs >= 2 h (and 6 report intervals = 1 h) clear
    assert kinds(e.step(H(162), ok, G0, {}, {})) == ['RECOVERY']
    assert e.store.q("select count(*) from incidents where status='open'")[0][0] == 0

def test_escalation_suppressed_after_acknowledgement(tmp_path):
    e = engine(tmp_path); down = S_(row(state='significant_interruption'))
    e.step(H(0), down, G0, {}, {}); e.acknowledge(1, H(1))
    sent = sum((kinds(e.step(H(h), down, G0, {}, {})) for h in range(2, 60)), [])
    assert 'ESCALATION' not in sent

def test_cooldown_flap_reopens_silently(tmp_path):
    e = engine(tmp_path); down = S_(row(state='significant_interruption')); ok = S_(row(state='normal'))
    e.step(H(0), down, G0, {}, {}); e.step(H(1), ok, G0, {}, {}); assert kinds(e.step(H(2), ok, G0, {}, {})) == ['RECOVERY']
    assert kinds(e.step(H(4), down, G0, {}, {})) == []                            # within the 6 h cooldown: silent reopen, no duplicate incident
    assert e.store.q("select count(*) from incidents")[0][0] == 1
    e.step(H(5), ok, G0, {}, {}); e.step(H(6), ok, G0, {}, {})
    assert kinds(e.step(H(20), down, G0, {}, {})) == ['OPEN']                     # after the cooldown: new incident, new e-mail

def test_flapping_sensor_needs_longer_clear(tmp_path):
    e = engine(tmp_path, flapping_onsets_7d=3, flapping_clear_h=24)
    e.step(H(0), S_(row(state='significant_interruption', n_onsets_7d=4.0)), G0, {}, {})
    ok = S_(row(state='normal', n_onsets_7d=4.0))
    sent = [kinds(e.step(H(h), ok, G0, {}, {})) for h in range(1, 26)]
    assert sent[:23] == [[]] * 23 and sent[23] == ['RECOVERY']

def test_grouping_common_cause_single_email(tmp_path):
    e = engine(tmp_path)
    rows = [row(dev=f'd{i}', state='significant_interruption', probable_cause='network / platform failure') for i in range(5)]
    sent = e.step(H(0), S_(*rows), G0, {}, {})
    assert len(sent) == 1 and sent[0]['n_items'] == 5 and '5 affected' in sent[0]['subject'] and 'network' in sent[0]['subject']

def test_unobserved_entities_are_frozen_not_recovered(tmp_path):
    e = engine(tmp_path); e.step(H(0), S_(row(state='significant_interruption')), G0, {}, {})
    for h in range(1, 30): assert kinds(e.step(H(h), S_(), G0, {}, {}, unobserved=frozenset({'d1'}))) in ([], ['REMINDER'])
    assert e.store.q("select status from incidents")[0][0] == 'open'

def test_outage_notification_independent_of_risk(tmp_path):
    e = engine(tmp_path)
    # actual outage with no / low predicted risk -> still notified
    s = e.step(H(0), S_(row(state='significant_interruption', risk_prolonged_7d=0.0, risk_long=0.0)), G0, {('A_prolonged_7d', 'MeteoHelix'): 0.5}, {})
    assert s and 'sensor down' in s[0]['subject'].lower()
    # high predicted risk on a working sensor never opens an outage incident
    e2 = engine(tmp_path / 'b')
    s2 = e2.step(H(0), S_(row(state='normal', risk_prolonged_7d=0.99)), G0, {('A_prolonged_7d', 'MeteoHelix'): 0.5}, {})
    types = {t for (t,) in e2.store.q('select type from incidents')}
    assert types == {'RISK_ELEVATED'} and 'SENSOR_DOWN' not in types and 'warning' in s2[0]['subject'].lower()

def test_risk_hysteresis(tmp_path):
    e = engine(tmp_path); thr = {('A_prolonged_7d', 'MeteoHelix'): 0.5}
    e.step(H(0), S_(row(risk_prolonged_7d=0.6)), G0, thr, {})
    for h in range(1, 40): e.step(H(h), S_(row(risk_prolonged_7d=0.4)), G0, thr, {})      # 0.4 >= 0.7 x 0.5: stays open
    assert e.store.q("select status from incidents")[0][0] == 'open'
    for h in range(40, 70): e.step(H(h), S_(row(risk_prolonged_7d=0.3)), G0, thr, {})
    assert e.store.q("select status from incidents")[0][0] == 'closed'

def test_gateway_down_requires_two_hours(tmp_path):
    e = engine(tmp_path)
    G = pd.DataFrame([dict(gw_site='siteA', condition='inactive_gateway', mu=20.0, dependents=3, dep_heard_elsewhere=1)])
    assert kinds(e.step(H(0), S_(), G, {}, {'siteA': 1})) == []
    s = e.step(H(1), S_(), G, {}, {'siteA': 2}); assert kinds(s) == ['OPEN'] and 'gateway down' in s[0]['subject'].lower()

def test_email_contains_required_fields(tmp_path):
    e = engine(tmp_path); e.step(H(0), S_(row(state='significant_interruption')), G0, {}, {})
    txt = next((tmp_path / 'outbox').glob('*.eml')).read_text()
    for k in ['d1', 'MeteoHelix', 'Last', 'interval', 'cause', 'ction']:
        assert k in txt, k

# ------------------------------------------------------------------ risk utilities
def test_kaplan_meier_known_values():
    ut, S = risk.km(np.array([1., 2., 2., 3., 4.]), np.array([True, True, False, True, False]))
    assert np.allclose(ut, [1, 2, 3]) and np.allclose(S, [0.8, 0.6, 0.3])
    assert np.allclose(risk.km_eval(ut, S, np.array([0.5, 2.5, 10])), [1.0, 0.6, 0.3])

# ------------------------------------------------------------------ battery
def test_lead_acid_bands():
    assert [battery.condition_lead_acid(v) for v in [12.5, 12.1, 11.8, 11.5]] == ['good', 'fair', 'low', 'critical']

@need_fd
def test_battery_assessment_has_no_percentages_or_lithium_dates():
    A = pd.read_parquet(FD / 'gold_battery_assessment.parquet')
    volt = A[A.family.isin(['Bigbelly', 'Ott', 'MeteoHelix', 'MeteoWind', 'Agvolution', 'Dragino', 'Sensoterra'])]
    assert not any('pct' in c or 'percent' in c for c in volt.dropna(axis=1, how='all').columns if c != 'latest_level_pct'), 'voltage converted to %'
    lith = A[A.condition.isin(['nominal_plateau', 'voltage_step_change'])]
    for c in [c for c in A.columns if 'crossing_day' in c]: assert lith[c].isna().all()
    nm = A[A.condition == 'not_measurable']; assert nm.device_id.isna().all()      # families without telemetry: no per-device estimate

# ------------------------------------------------------------------ anomaly detectors
def test_cusum_detects_bias_and_is_causal():
    rng = np.random.default_rng(0); n = 1500; acc = 0.3
    idx = pd.date_range('2026-08-01', periods=n, freq='h'); x = rng.normal(0, acc / 2, n); x[1000:] += 3 * acc
    D = anomaly.detectors(pd.Series(x, idx), acc)
    assert not D.alarm_cusum.iloc[200:1000].any() and D.alarm_cusum.iloc[1000:1040].any()
    D2 = anomaly.detectors(pd.Series(x[:1020], idx[:1020]), acc)                 # truncation: past values unchanged
    pd.testing.assert_frame_equal(D.iloc[:1020], D2, check_freq=False)

# ------------------------------------------------------------------ gateway causality
@need_fd
def test_gateway_site_hours_causal_under_truncation():
    R = pd.read_parquet(FD / 'gold_radio_records.parquet'); sites = pd.read_parquet(FD / 'gold_gateway_sites.parquet')
    cut = pd.Timestamp('2026-09-20')
    tcol = next(c for c in ['t', 't_rx', 'arrival', 'received_at'] if c in R.columns)
    A1, _ = gateway.site_hours(R, sites); A2, _ = gateway.site_hours(R[R[tcol] < cut], sites)
    k = ['gw_site', 'hour_end']; c = [x for x in ['n_records', 'mu', 'dependents'] if x in A1.columns]
    m = A1[A1.hour_end <= cut][k + c].merge(A2[A2.hour_end <= cut][k + c], on=k, suffixes=('', '_t'))
    assert len(m) > 1000
    for x in c: assert np.allclose(m[x].astype(float), m[x + '_t'].astype(float), equal_nan=True), x

# ------------------------------------------------------------------ integration
def test_overall_light_is_worst_component():
    S = S_(row(), row(dev='d2', battery_condition='low'), row(dev='d3', anomaly_high=1), row(dev='d4', risk_prolonged_7d=0.9),
           row(dev='d5', state='significant_interruption'), row(dev='d6', anomaly_kinds='frozen_value'))
    S.loc[0, 'traffic_light'] = 'green'
    o = system.overall(S, {('A_prolonged_7d', 'MeteoHelix'): 0.5}).set_index('device_id').overall_light
    assert o.to_dict() == dict(d1='green', d2='yellow', d3='red', d4='yellow', d5='red', d6='yellow')

def test_priority_ranks_outage_first():
    S = S_(row(), row(dev='d2', battery_condition='low'), row(dev='d5', state='significant_interruption'))
    for c in ['forecast_avail', 'tau']: S[c] = 0.9
    S = system.overall(S, {}); P = system.priority(S)
    assert P.device_id.iat[0] == 'd5' and P.device_id.iat[-1] == 'd1' and 'unavailable' not in P.why.iat[-1]

@need_rp
def test_replay_outputs_consistent():
    ev = json.load(open(RP / 'replay_evaluation.json')); log = pd.read_parquet(RP / 'notification_log.parquet')
    assert set(log.status) <= {'dry_run'}, 'replay must never send real e-mail'
    assert (log.groupby(['subject', 'hour']).size() <= 1).all()                # no duplicate e-mails
    inc = pd.read_parquet(RP / 'incidents.parquet')
    assert inc.groupby(['key', 'opened_at']).size().max() == 1

@need_rp
@need_fd
def test_dashboard_builds_without_nan(tmp_path):
    from hsm.dashboard import build
    out = build.build(str(FD), str(RP), str(tmp_path / 'd.html'), None)
    html = Path(out).read_text()
    i = html.find('const D='); assert i > 0; data = html[i:]
    assert 'NaN' not in data and 'Infinity' not in data and '"False"' not in data
    for v in ['Diese Sensoren jetzt', 'Gateway', 'overall_light']: assert v in html

def test_feed_stale_incident_once_then_recovery(tmp_path):
    e = engine(tmp_path)
    st = dict(stale=True, newest_record='2026-10-01 00:00', record_age_h=5.0, file_age_h=None); ok = dict(st, stale=False)
    s = e.step_feed(H(5), {'weather': st, 'soil': st})
    assert len(s) == 1 and s[0]['n_items'] == 2 and 'export not refreshed' in s[0]['subject']      # grouped, one e-mail
    assert e.step_feed(H(6), {'weather': st, 'soil': st}) == []
    assert kinds(e.step_feed(H(7), {'weather': ok, 'soil': st})) == ['RECOVERY']
    txt = ''.join(p.read_text() for p in (tmp_path / 'outbox').glob('*.eml'))
    assert 'unknown time (simulated clock)' in txt.replace('=\n', '')

def test_feed_status_record_age_and_file_time(tmp_path):
    import time as _t
    from hsm.cli import feed_status
    for fn, t in [('devices_raw.csv', '2026-10-01 09:30:00.000000 +00:00'), ('bodenfeuchte_raw.csv', '2026-10-01 02:00:00.000000 +00:00'),
                  ('weather_raw.csv', '2026-10-01 02:00:00.000000 +00:00')]:
        (tmp_path / fn).write_text(f'entity_id,time_index\nx,{t}\n')
    now = pd.Timestamp('2026-10-01 10:00')
    os.utime(tmp_path / 'weather_raw.csv', (now.timestamp() - 600, now.timestamp() - 600))          # rewritten 10 min ago: real silence
    os.utime(tmp_path / 'bodenfeuchte_raw.csv', (now.timestamp() - 9 * 3600, now.timestamp() - 9 * 3600))   # not rewritten: stale
    f = feed_status(tmp_path, now, 3)
    assert not f['devices']['stale'] and f['soil']['stale'] and not f['weather']['stale']
