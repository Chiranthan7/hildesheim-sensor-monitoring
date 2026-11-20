"""Test e-mails: the two alert types (sensor stopped working / may stop within 24 h) built from real historical cases,
marked [TEST], sent through the configured mailer (real e-mail only if enabled; otherwise written to the outbox)."""
import json, tempfile
from pathlib import Path
import pandas as pd
from .engine import Engine, Store
from .mailer import Mailer

def send_samples(ncfg, data_dir, models_dir):
    d = Path(data_dir)
    n = dict(ncfg, baseline_digest=False, email_types=['SENSOR_DOWN', 'FAILURE_EXPECTED_24H'], email_kinds=['OPEN'], email_style='simple', subject_prefix='[TEST] ')
    S = pd.read_parquet(d / 'snapshots.parquet'); D = pd.read_parquet(d / 'devices.parquet'); G = pd.read_parquet(d / 'gateways.parquet')
    loc = {r.device_id: (r.lat, r.lon, None) for r in D.itertuples() if pd.notna(r.lat)}
    loc.update({f'gw:{r.gw_site}': (r.lat, r.lon, None) for r in G.itertuples() if pd.notna(r.lat)})
    thr_p = Path(models_dir) / 'warning_thresholds.json'
    thr = {tuple(k.split('|')): v for k, v in json.load(open(thr_p)).items()} if thr_p.exists() else {}
    eng = Engine(n, Store(str(Path(tempfile.mkdtemp()) / 'sample.sqlite')), Mailer(n), loc)
    def plain_rows(x):
        x = x.copy()
        for c in x.columns:
            if isinstance(x[c].dtype, pd.CategoricalDtype): x[c] = x[c].astype(object)
            elif x[c].dtype == 'float32': x[c] = x[c].astype(float)
        return x
    last = S.groupby('device_id', observed=True).tail(1)                 # each sensor's latest status
    dn = last[(last.state.astype(str) == 'significant_interruption') & last.device_id.astype(str).isin(loc)]
    down = plain_rows(dn.sort_values('hour_end', ascending=False).head(1))
    I = pd.read_parquet(d / 'incidents.parquet') if (d / 'incidents.parquet').exists() else pd.DataFrame(columns=['type'])
    warn = S.iloc[:0]
    for r in I[I.type == 'FAILURE_EXPECTED_24H'].itertuples():
        x = S[(S.device_id.astype(str) == r.entity) & (S.hour_end == pd.Timestamp(r.opened_at))]
        if len(x): warn = plain_rows(x); break
    sent = []
    for rows in (down, warn):
        if not len(rows): continue
        h = rows.hour_end.iat[0]
        for c in eng.device_conditions(rows, thr):
            if c['type'] not in n['email_types']: continue
            iid = eng.store.x("insert into incidents (key, type, entity, entity_kind, severity, opened_at, status) values (?,?,?,?,?,?, 'open')",
                              f"{c['type']}|{c['entity']}", c['type'], c['entity'], 'sensor', 'critical', str(h))
            sent += eng.dispatch_simple(h, [('OPEN', iid, c, 'critical')])
    return dict(real_email=not eng.mailer.dry, via='resend' if eng.mailer.resend_key else 'smtp', recipients=n.get('recipients'), sent=sent)
