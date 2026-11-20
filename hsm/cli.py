"""Command-line interface of the Hildesheim Smart Sensor Monitoring system.

  python -m hsm.cli build-foundation --raw DIR --out FD [--cutoff T]
  python -m hsm.cli analyse  --foundation FD                 # Module 1, risk, gateway, battery, anomaly components + evaluation
  python -m hsm.cli train    --foundation FD --models MD      # production models on all data
  python -m hsm.cli replay   --foundation FD --out RD          # causal historical replay, dry-run notifications, evaluation
  python -m hsm.cli dashboard --foundation FD --replay RD --out dashboard.html [--as-of T] [--serve]
  python -m hsm.cli notify-test --config CFG [--to ADDR]       # test e-mail (dry-run unless enabled in the config)
  python -m hsm.cli ack --config CFG --incident ID             # acknowledge an incident (stops escalation)
  python -m hsm.cli notify-sample --config CFG                 # the two alert e-mails from real historical cases, marked [TEST]
  python -m hsm.cli monitor --config CFG --models MD [--once] [--now T]   # live monitoring cycle(s)
  python -m hsm.cli prepare --foundation FD --replay RD --out appdata       # data for the web app
  python -m hsm.cli app --data appdata [--port 8050] [--live-config CFG --models MD]   # interactive web app
"""
import argparse, json, os, sys, time, shutil, subprocess
from pathlib import Path
import numpy as np, pandas as pd, yaml

ROOT = Path(__file__).resolve().parent.parent

def load_cfg(path):
    """config file + environment: HSM_ALERT_TO (comma-separated recipients), HSM_EMAIL_ENABLED=1 (real sending), HSM_APP_URL"""
    cfg = yaml.safe_load(open(path)); n = cfg.setdefault('notifications', {})
    if os.environ.get('HSM_ALERT_TO'): n['recipients'] = [x.strip() for x in os.environ['HSM_ALERT_TO'].split(',') if x.strip()]
    if os.environ.get('HSM_EMAIL_ENABLED') == '1': n['enabled'] = True; n['dry_run'] = False
    if os.environ.get('HSM_APP_URL'): n['dashboard_url'] = os.environ['HSM_APP_URL']
    return cfg

def cmd_build(a):
    import run_foundation; run_foundation.main(a.raw, a.out, a.cutoff)

def cmd_analyse(a):
    import run_m1, run_risk, run_gateway, run_battery, run_anomaly
    for f in (lambda: run_m1.main(a.foundation, not a.no_importance), lambda: run_gateway.main(a.foundation), lambda: run_battery.main(a.foundation),
              lambda: run_anomaly.main(a.foundation), lambda: run_risk.main(a.foundation)):
        f()

def cmd_train(a):
    from hsm import production; print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'hazard'} for k, v in production.train(a.foundation, a.models).items()}, indent=1))

def cmd_replay(a):
    import run_replay; run_replay.main(a.foundation, a.out, a.start)

def cmd_dashboard(a):
    from hsm.dashboard import build
    out = build.build(a.foundation, a.replay, a.out, a.as_of); print(out)
    if a.serve:
        import http.server, functools
        d = str(Path(out).resolve().parent); h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
        print(f'serving http://localhost:{a.port}/{Path(out).name}'); http.server.ThreadingHTTPServer(('', a.port), h).serve_forever()

def _engine(cfg, fd=None):
    from hsm.notify.engine import Engine, Store
    from hsm.notify.mailer import Mailer
    from hsm import system
    n = cfg['notifications']; loc = system.device_locations(fd) if fd else {}
    return Engine(n, Store(n.get('store', './state/notifications.sqlite')), Mailer(n), loc)

def cmd_notify_test(a):
    from hsm.notify.mailer import Mailer
    cfg = load_cfg(a.config)['notifications']; m = Mailer(cfg)
    to = [a.to] if a.to else cfg.get('recipients', [])
    r = m.send(to, '[HSM TEST] notification channel test', 'This is a test e-mail from the Hildesheim Smart Sensor Monitoring system.\nNo action required.',
               '<p>This is a <b>test e-mail</b> from the Hildesheim Smart Sensor Monitoring system. No action required.</p>', tag=f'test_{int(time.time())}')
    print(json.dumps(dict(dry_run=m.dry, recipients=to, **r), indent=1))

def cmd_notify_sample(a):
    """send the two alert e-mails (stopped working / may stop within 24 h) built from real historical cases, marked [TEST]"""
    from hsm.notify.samples import send_samples
    print(json.dumps(send_samples(load_cfg(a.config)['notifications'], a.data, a.models), indent=1, default=str))

def cmd_ack(a):
    e = _engine(load_cfg(a.config)); e.acknowledge(a.incident, pd.Timestamp.now('UTC').tz_localize(None)); print(f'incident {a.incident} acknowledged')

STREAMS = {'devices': 'devices_raw.csv', 'soil': 'bodenfeuchte_raw.csv', 'weather': 'weather_raw.csv'}

def feed_status(raw_dir, now, stale_h):
    """Is each export stream current? With file exports a stale export (not refreshed) and a real outage both show no new records.
    They are separated by the file's modification time: rewritten recently but without new records = the platform delivers and the
    silence is real (sensors / gateways / network, evaluated as usual); not rewritten = the stream is stale and its silence carries no
    information about the sensors. Stale streams end their families' observation window at their newest record, so the sensors become
    'insufficient information' (incidents frozen) instead of producing a flood of false outage alarms. When the export times are later
    than 'now' (simulating a past moment) file times carry no information and record age alone decides."""
    out = {}
    for k, fn in STREAMS.items():
        f = Path(raw_dir) / fn
        if not f.exists(): out[k] = dict(stale=True, newest_record=None, record_age_h=None, file_age_h=None); continue
        t = pd.read_csv(f, usecols=['time_index']).time_index
        t = pd.to_datetime(t, errors='coerce', utc=True, format='mixed').dt.tz_localize(None); newest = t[t <= now].max()
        mtime = pd.Timestamp(os.path.getmtime(f), unit='s')
        rec_age = (now - newest).total_seconds() / 3600 if pd.notna(newest) else np.inf
        file_age = (now - mtime).total_seconds() / 3600 if mtime <= now else np.inf
        stale = rec_age > stale_h and not (file_age <= stale_h and mtime > newest)
        out[k] = dict(stale=bool(stale), newest_record=str(newest), record_age_h=round(rec_age, 2), file_age_h=round(file_age, 2) if np.isfinite(file_age) else None)
    return out

def _write_live_state(cfg, S, fd, keep_days=28):
    """live-state directory for the dashboard: hourly snapshots (rolling window), incidents and notifications from the store"""
    import sqlite3
    ld = Path(cfg['data'].get('live_dir', Path(fd).parent / 'live_state')); ld.mkdir(parents=True, exist_ok=True)
    p = ld / 'sys_device_hour.parquet'
    H = pd.concat([pd.read_parquet(p), S], ignore_index=True) if p.exists() else S
    H = H.drop_duplicates(['device_id', 'hour_end'], keep='last'); H = H[H.hour_end > H.hour_end.max() - pd.Timedelta(days=keep_days)]
    H.to_parquet(p, index=False)
    db = sqlite3.connect(cfg['notifications'].get('store', './state/notifications.sqlite'))
    for t in ['incidents', 'notifications']: pd.read_sql(f'select * from {t}', db).to_parquet(ld / f'{t}.parquet', index=False)
    hist = cfg['data'].get('replay_dir')
    if hist and (Path(hist) / 'replay_evaluation.json').exists(): shutil.copy(Path(hist) / 'replay_evaluation.json', ld)
    return ld

def monitor_once(cfg, models, now=None):
    """one live cycle: check the feed, ingest exports up to 'now' (reporting deadlines are evaluated against now, so a silent device
    keeps accumulating missed reports), recompute components, score with the production models, notify, rebuild the dashboard."""
    import run_foundation, run_gateway, run_battery, run_anomaly
    from hsm import system
    from hsm.notify.engine import Engine, Store
    from hsm.notify.mailer import Mailer
    now = pd.Timestamp(now) if now else pd.Timestamp.now('UTC').tz_localize(None)
    fd = cfg['data']['foundation_dir']; n = cfg['notifications']
    feed = feed_status(cfg['data']['raw_dir'], now, cfg['data'].get('feed_stale_h', 3))
    eng = Engine(n, Store(n.get('store', './state/notifications.sqlite')), Mailer(n))
    sent = eng.step_feed(now, feed)                                     # FEED_STALE incident per stale export stream (opens / reminds / recovers)
    run_foundation.main(cfg['data']['raw_dir'], fd, str(now), stream_end={k: (pd.Timestamp(v['newest_record']) if v['stale'] and v['newest_record'] not in (None, 'NaT') else now)
                                                                         for k, v in feed.items()})
    run_gateway.main(fd); run_battery.main(fd); run_anomaly.main(fd)
    eng.locations = system.device_locations(fd)
    thr = {tuple(k.split('|')): v for k, v in json.load(open(Path(models) / 'warning_thresholds.json')).items()} if (Path(models) / 'warning_thresholds.json').exists() else {}
    S = system.live_snapshot(fd, models, thr)
    hour = S.hour_end.max()
    G = pd.read_parquet(f'{fd}/gold_gw_site_hour.parquet'); g = G[G.hour_end == G.hour_end.max()]
    run_h = {}
    for site, x in G.sort_values('hour_end').groupby('gw_site'):
        bad = x.condition.isin(['inactive_gateway', 'inactive_unexplained']).values[::-1]; run_h[site] = int(next((i for i, b in enumerate(bad) if not b), len(bad)))
    net = (G.groupby('hour_end').condition.apply(lambda c: (c == 'network_or_platform').any())).values[::-1]
    run_h['NETWORK'] = int(next((i for i, b in enumerate(net) if not b), len(net)))
    # devices without information at the status hour (stale export stream, outside their window): incidents frozen, no new opens
    unseen = (S.state == 'insufficient_information') | (S.hour_end < hour)
    obs = S[~unseen]; unobs = frozenset(S.loc[unseen, 'device_id'])
    sent += eng.step(hour, obs, g, thr, run_h, unobs)
    S.to_parquet(Path(fd) / 'live_snapshot.parquet', index=False)
    ld = _write_live_state(cfg, S, fd)
    from hsm.dashboard import build
    dash = build.build(fd, ld, cfg['data'].get('dashboard_out', str(Path(fd).parent / 'dashboard.html')), None, cfg['data'].get('eval_foundation_dir'))
    if cfg['data'].get('app_data_dir'):
        from hsm.webapp.prepare import prepare
        prepare(fd, ld, cfg['data']['app_data_dir'], cfg['data'].get('eval_foundation_dir'))
    return dict(now=str(now), feed=feed, status_hour=str(hour), devices=len(S), states=S.state.value_counts().to_dict(), notifications=sent, dashboard=dash)

def cmd_monitor(a):
    cfg = load_cfg(a.config)
    while True:
        try: print(json.dumps(monitor_once(cfg, a.models, a.now), indent=1, default=str), flush=True)
        except Exception as e: print(json.dumps(dict(error=f'{type(e).__name__}: {e}')), flush=True)      # keep monitoring after an error
        if a.once: break
        time.sleep(60 * cfg['data'].get('poll_minutes', 15))

def cmd_prepare(a):
    from hsm.webapp.prepare import prepare; print(json.dumps(prepare(a.foundation, a.replay, a.out, a.eval_foundation), indent=1))

def cmd_app(a):
    from hsm.webapp import app as webapp
    webapp.main(['--data', a.data, '--host', a.host, '--port', str(a.port)] + (['--live-config', a.live_config, '--models', a.models] if a.live_config else []))

def main(argv=None):
    sys.path.insert(0, str(ROOT))
    ap = argparse.ArgumentParser(prog='hsm', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('build-foundation'); p.add_argument('--raw', required=True); p.add_argument('--out', required=True); p.add_argument('--cutoff'); p.set_defaults(f=cmd_build)
    p = sp.add_parser('analyse'); p.add_argument('--foundation', required=True); p.add_argument('--no-importance', action='store_true'); p.set_defaults(f=cmd_analyse)
    p = sp.add_parser('train'); p.add_argument('--foundation', required=True); p.add_argument('--models', required=True); p.set_defaults(f=cmd_train)
    p = sp.add_parser('replay'); p.add_argument('--foundation', required=True); p.add_argument('--out', required=True); p.add_argument('--start', default='2026-08-01'); p.set_defaults(f=cmd_replay)
    p = sp.add_parser('dashboard'); p.add_argument('--foundation', required=True); p.add_argument('--replay', required=True); p.add_argument('--out', default='dashboard.html')
    p.add_argument('--as-of'); p.add_argument('--serve', action='store_true'); p.add_argument('--port', type=int, default=8050); p.set_defaults(f=cmd_dashboard)
    p = sp.add_parser('notify-test'); p.add_argument('--config', required=True); p.add_argument('--to'); p.set_defaults(f=cmd_notify_test)
    p = sp.add_parser('notify-sample'); p.add_argument('--config', required=True); p.add_argument('--data', default='appdata'); p.add_argument('--models', default='models')
    p.set_defaults(f=cmd_notify_sample)
    p = sp.add_parser('ack'); p.add_argument('--config', required=True); p.add_argument('--incident', type=int, required=True); p.set_defaults(f=cmd_ack)
    p = sp.add_parser('monitor'); p.add_argument('--config', required=True); p.add_argument('--models', required=True); p.add_argument('--once', action='store_true')
    p.add_argument('--now', help='evaluation time (default: current UTC); use a past time to simulate'); p.set_defaults(f=cmd_monitor)
    p = sp.add_parser('prepare'); p.add_argument('--foundation', required=True); p.add_argument('--replay', required=True); p.add_argument('--out', default='appdata')
    p.add_argument('--eval-foundation'); p.set_defaults(f=cmd_prepare)
    p = sp.add_parser('app'); p.add_argument('--data', default='appdata'); p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=8050)
    p.add_argument('--live-config'); p.add_argument('--models', default='models'); p.set_defaults(f=cmd_app)
    a = ap.parse_args(argv); a.f(a)

if __name__ == '__main__':
    main()
