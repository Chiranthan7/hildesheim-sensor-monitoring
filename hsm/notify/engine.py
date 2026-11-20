"""Incident and notification engine (component 7).

Each condition (type, entity) has a lifecycle: OPEN -> (REMINDER)* -> (ESCALATION) -> RECOVERY -> closed.
  open        condition true and no open incident. If the same incident closed less than cooldown_h ago it is REOPENED silently
              (flap suppression), never duplicated.
  reminder    while open, every reminder_h[severity] hours (no hourly repeats).
  escalation  critical, unacknowledged, open >= escalate_h -> one e-mail to the escalation recipients.
  recovery    condition false for >= min_clear_h consecutive hours -> close; a recovery e-mail is sent if the opening was notified.
Grouping: all OPEN / RECOVERY events of one step that share a common-cause group (network, gateway site, family-wide backend, fleet)
are sent as ONE e-mail listing every affected device.
Actual outage incidents (SENSOR_DOWN, BACKEND_FAILURE, GATEWAY_DOWN, NETWORK_OUTAGE) depend only on the observed state, never on
predicted risk. Everything is persisted in SQLite (incidents, notifications)."""
import json, sqlite3, math
from pathlib import Path
import numpy as np, pandas as pd
from jinja2 import Environment, FileSystemLoader
from ..config import NOMINAL_MIN
from ..system import MANUFACTURER, ACTIONS

CRITICAL = {'SENSOR_DOWN', 'BACKEND_FAILURE', 'GATEWAY_DOWN', 'NETWORK_OUTAGE', 'BATTERY_CRITICAL', 'SEVERE_MALFUNCTION'}
SCHEMA = """
create table if not exists incidents (id integer primary key, key text, type text, entity text, entity_kind text, severity text,
  opened_at text, closed_at text, last_notified_at text, n_notifications integer default 0, escalated integer default 0,
  acknowledged_at text, status text, clear_hours integer default 0, group_key text, notified_open integer default 0, details text);
create index if not exists ix_inc_key on incidents(key, status);
create table if not exists notifications (id integer primary key, kind text, sent_at text, incident_ids text, recipients text,
  subject text, status text, attempts integer, error text, path text, group_key text);
"""

class Store:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True); self.db = sqlite3.connect(path); self.db.executescript(SCHEMA)
    def q(self, sql, *a): return self.db.execute(sql, a).fetchall()
    def x(self, sql, *a): c = self.db.execute(sql, a); self.db.commit(); return c.lastrowid

class Engine:
    def __init__(self, cfg, store, mailer):
        self.cfg = cfg; self.store = store; self.mailer = mailer; self.deg_hours = {}
        tdir = Path(__file__).parent / 'templates'
        self.env = Environment(loader=FileSystemLoader(str(tdir)), autoescape=False); self.env_html = Environment(loader=FileSystemLoader(str(tdir)), autoescape=True)

    # ---------------------------------------------------------------- conditions
    def device_conditions(self, S, thr):
        out = []; open_keys = {k for (k,) in self.store.q("select key from incidents where status='open'")}
        for r in S.itertuples():
            fam = r.family; man, model = MANUFACTURER.get(fam, ('', ''))
            cause = r.probable_cause if isinstance(r.probable_cause, str) else None
            grp = None
            if cause == 'network / platform failure': grp = 'NETWORK'
            elif cause == 'gateway unavailability' and isinstance(r.primary_gateway, str): grp = f'GATEWAY:{r.primary_gateway}'
            elif cause and cause.startswith('fleet-wide'): grp = f'FLEET:{fam}'
            elif cause == 'backend storage failure': grp = f'BACKEND:{fam}'
            base = dict(entity=r.device_id, entity_kind='sensor', family=fam, manufacturer=man, model=model, row=r)
            if r.state == 'significant_interruption': out.append(dict(base, type='SENSOR_DOWN', group=grp))
            if r.state == 'backend_data_unavailable': out.append(dict(base, type='BACKEND_FAILURE', group=f'BACKEND:{fam}'))
            if r.battery_condition == 'critical': out.append(dict(base, type='BATTERY_CRITICAL', group=None))
            if r.anomaly_high > 0: out.append(dict(base, type='SEVERE_MALFUNCTION', group=None))
            if not self.cfg.get('warnings_enabled', True): continue
            k = r.device_id
            if hasattr(r, 'degraded_run_h') and pd.notna(r.degraded_run_h): self.deg_hours[k] = int(r.degraded_run_h)   # live: from the timeline
            else: self.deg_hours[k] = self.deg_hours.get(k, 0) + 1 if r.state == 'degraded' else 0                     # replay: hourly steps
            if self.deg_hours[k] >= self.cfg.get('degraded_min_h', 24): out.append(dict(base, type='DEGRADED_COMMUNICATION', group=None))
            if r.battery_condition == 'low': out.append(dict(base, type='BATTERY_LOW', group=None))
            if r.state == 'reduced_reporting_mode': out.append(dict(base, type='REDUCED_REPORTING_MODE', group=None))
            if isinstance(r.anomaly_kinds, str) and not r.anomaly_high: out.append(dict(base, type='MEASUREMENT_DEGRADATION', group=None))
            tb = thr.get(('B_developing', fam))
            if self.cfg.get('early_silence_warning', False) and r.state in ('normal', 'degraded') and tb is not None and np.isfinite(r.p_silence_becomes_outage) \
                    and r.p_silence_becomes_outage >= tb:
                out.append(dict(base, type='SILENCE_DEVELOPING', group=None))
            t7 = thr.get(('A_prolonged_7d', fam))
            hyst = self.cfg.get('risk_hysteresis', 0.7) if f'RISK_ELEVATED|{r.device_id}' in open_keys else 1.0     # clear only below 0.7 x threshold
            if r.state in ('normal', 'degraded') and t7 is not None and np.isfinite(r.risk_prolonged_7d) and r.risk_prolonged_7d >= hyst * t7:
                out.append(dict(base, type='RISK_ELEVATED', group=None))
        return out

    def gateway_conditions(self, G, run_h):
        out = []
        net = (G.condition == 'network_or_platform').any()
        if net and run_h.get('NETWORK', 0) >= 2:
            out.append(dict(entity='network', entity_kind='network', family='network', manufacturer='', model='LoRaWAN network / platform', type='NETWORK_OUTAGE', group='NETWORK', row=None))
        for r in G.itertuples():
            if r.condition in ('inactive_gateway', 'inactive_unexplained') and run_h.get(r.gw_site, 0) >= 2:
                out.append(dict(entity=r.gw_site, entity_kind='gateway', family='gateway', manufacturer='', model='LoRaWAN gateway (observed via receptions)',
                                type='GATEWAY_DOWN', group=f'GATEWAY:{r.gw_site}', row=r))
        return out

    # ---------------------------------------------------------------- lifecycle
    def step(self, now, S, G, thr, gw_run_h, unobserved=frozenset()):
        """unobserved: entities without information this hour (outside their data window): their incidents are frozen, not recovered"""
        now_s = str(now); conds = self.device_conditions(S, thr) + self.gateway_conditions(G, gw_run_h)
        active = {f"{c['type']}|{c['entity']}": c for c in conds}
        events = []
        cool = self.cfg.get('cooldown_h', 6)
        for key, c in active.items():
            sev = 'critical' if c['type'] in CRITICAL else 'warning'
            op = self.store.q("select id, opened_at, last_notified_at, escalated, acknowledged_at, n_notifications from incidents where key=? and status='open'", key)
            if op:
                iid, opened, last, esc, ack, nn = op[0]
                self.store.x("update incidents set clear_hours=0 where id=?", iid)
                hrs_last = (now - pd.Timestamp(last)).total_seconds() / 3600 if last else 1e9
                # reminder back-off: base, 2 x base, 4 x base, ... capped at reminder_max_h (weekly); acknowledged incidents get weekly reminders
                base = self.cfg.get('reminder_h', {}).get(sev, 24); k = max((nn or 1) - 1, 0)
                due = min(base * 2 ** k, self.cfg.get('reminder_max_h', 168)) if not ack else self.cfg.get('reminder_max_h', 168)
                if last and hrs_last >= due: events.append(('REMINDER', iid, c, sev))
                if sev == 'critical' and not esc and not ack and (now - pd.Timestamp(opened)).total_seconds() / 3600 >= self.cfg.get('escalate_h', 48):
                    events.append(('ESCALATION', iid, c, sev)); self.store.x("update incidents set escalated=1 where id=?", iid)
                continue
            rec = self.store.q("select id, closed_at from incidents where key=? and status='closed' order by closed_at desc limit 1", key)
            if rec and (now - pd.Timestamp(rec[0][1])).total_seconds() / 3600 < cool:
                self.store.x("update incidents set status='open', closed_at=null, clear_hours=0 where id=?", rec[0][0]); continue   # flap: silent reopen
            iid = self.store.x("insert into incidents (key, type, entity, entity_kind, severity, opened_at, status, group_key) values (?,?,?,?,?,?, 'open', ?)",
                               key, c['type'], c['entity'], c['entity_kind'], sev, now_s, c['group'])
            events.append(('OPEN', iid, c, sev))
        info = {r.device_id: r for r in S.itertuples()} if len(S) else {}
        for iid, key, typ, entity, sev, clear, notified in self.store.q("select id, key, type, entity, severity, clear_hours, notified_open from incidents where status='open'"):
            if key in active or entity in unobserved: continue
            clear += 1; need = self.cfg.get('min_clear_h', {}).get(typ, self.cfg.get('min_clear_h', {}).get('default', 6))
            r = info.get(entity)
            if r is not None and typ in ('SENSOR_DOWN', 'BACKEND_FAILURE'):
                need = max(need, 6 * NOMINAL_MIN.get(r.family, 60) / 60)          # recovery must hold for 6 report intervals
                if np.isfinite(r.n_onsets_7d) and r.n_onsets_7d >= self.cfg.get('flapping_onsets_7d', 3): need = max(need, self.cfg.get('flapping_clear_h', 24))
            if clear >= need:
                self.store.x("update incidents set status='closed', closed_at=?, clear_hours=? where id=?", now_s, clear, iid)
                if notified: events.append(('RECOVERY', iid, dict(type=typ, entity=entity, entity_kind='', family='', manufacturer='', model='', row=None,
                                                                  group=self.store.q('select group_key from incidents where id=?', iid)[0][0]), sev))
            else:
                self.store.x("update incidents set clear_hours=? where id=?", clear, iid)
        return self.dispatch(now, events)

    # ---------------------------------------------------------------- e-mails
    def _item(self, kind, iid, c, sev, now):
        r = c.get('row'); it = dict(kind=kind, incident_id=iid, type=c['type'], severity=sev, entity=c['entity'], entity_kind=c['entity_kind'] or 'device',
                                     family=c['family'], manufacturer=c['manufacturer'], model=c['model'])
        if kind == 'RECOVERY' and c['entity_kind'] == 'feed':
            it.update(condition='export stream current again', cause='data feed restored', action='none; sensor evaluation resumed'); return it
        if kind == 'RECOVERY' and c['entity_kind'] == 'feed':
            it.update(condition='export stream current again', cause='data feed restored', action='none; sensor evaluation resumed'); return it
        if kind == 'RECOVERY':
            op = self.store.q('select opened_at from incidents where id=?', iid)[0][0]
            it.update(condition=f"recovered (incident open since {op})", unavailable_txt=f"{(now - pd.Timestamp(op)).total_seconds() / 3600:.1f} h in total"); return it
        if c['entity_kind'] == 'sensor' and r is not None:
            fam = c['family']; iv = NOMINAL_MIN.get(fam)
            un = r.silence_svc_slots * iv / 60 if r.state in ('significant_interruption', 'backend_data_unavailable') else None
            rk = []
            for lab, v in [('prolonged interruption within 7 d', r.risk_prolonged_7d), ('interruption onset (long horizon)', r.risk_long),
                           ('still unavailable in 24 h', r.p_still_down_24h), ('still unavailable in 7 d', r.p_still_down_7d)]:
                if isinstance(v, float) and np.isfinite(v): rk.append(f'{lab}: {v:.0%}')
            it.update(condition=f"{r.state.replace('_', ' ')} ({r.traffic_light})" + (f"; {r.reasons}" if isinstance(r.reasons, str) and r.reasons else ''),
                      last_rx=str(r.rx_last_arrival) if pd.notna(r.rx_last_arrival) else None, last_stored=str(r.svc_last_arrival) if pd.notna(r.svc_last_arrival) else None,
                      interval=f'{iv} min' + (f' (current mode {int(r.rx_mode_min)} min)' if pd.notna(r.rx_mode_min) and r.rx_mode_min != iv else ''),
                      unavailable_h=un, unavailable_txt=f'{un:.1f} h' if un else None,
                      missing=f'{r.silence_svc_slots:.0f} so far' if un else (f'{r.expected_missing_primary:.1f} expected next {int(r.primary_H)} h' if np.isfinite(r.expected_missing_primary) else None),
                      risk='; '.join(rk) or None, battery=r.battery_condition, gateway=(f'{r.primary_gateway} ({r.gateway_condition})' if isinstance(r.primary_gateway, str) else None),
                      cause=r.probable_cause, confidence=r.cause_confidence, evidence=r.cause_evidence, action=ACTIONS.get(r.probable_cause, 'review evidence'))
        elif c['entity_kind'] == 'feed':
            f = c['feed']; it.update(condition='sensor data export not refreshed', last_rx=f['newest_record'], last_stored=f['newest_record'],
                                    evidence=f"newest record {f['record_age_h']} h old; export file last written " + (f"{f['file_age_h']} h ago" if f.get('file_age_h') is not None else 'at an unknown time (simulated clock)'),
                                    cause='data feed / export pipeline stopped (sensor states cannot be evaluated)', confidence='high',
                                    action='check the export job / API connection; sensor incidents are frozen until data arrives')
        elif c['entity_kind'] == 'gateway' and r is not None:
            it.update(condition=r.condition.replace('_', ' '), last_rx=str(r.last_activity) if hasattr(r, 'last_activity') else None,
                      gateway=c['entity'], cause='gateway-specific (dependent sensors re-routed)' if r.condition == 'inactive_gateway' else 'gateway and dependent sensors silent together',
                      evidence=f'0 records for >= 2 h while {r.mu:.1f} records/h expected; dependents {int(r.dependents)}, re-routed {int(r.dep_heard_elsewhere)}',
                      action='check gateway power, backhaul and packet forwarder', confidence='medium')
        else:
            it.update(condition='network-level silence: most gateway sites without receptions', cause='network / platform failure', confidence='high',
                      action='check network server / platform and backhaul', evidence='>= 70 % of active gateway sites silent')
        return it

    def dispatch(self, now, events):
        sent = []
        groups = {}
        for kind, iid, c, sev in events:
            g = c.get('group') if kind in ('OPEN', 'RECOVERY') else None
            key = (kind, g) if g else (kind, f'single:{iid}')
            groups.setdefault(key, []).append((iid, c, sev))
        for (kind, g), members in groups.items():
            items = [self._item(kind, iid, c, sev, now) for iid, c, sev in members]
            sev = 'critical' if any(m[2] == 'critical' for m in members) else 'warning'
            if kind == 'RECOVERY': sev = 'recovery'
            n = len(items); first = items[0]
            reason = None if g is None or g.startswith('single') else {'NETWORK': 'network / platform interruption', 'FEED': 'data export not refreshed'}.get(g, g.replace(':', ' ').replace('GATEWAY', 'gateway').replace('BACKEND', 'backend storage failure, family').replace('FLEET', 'fleet-wide stop, family'))
            head = (f"{kind.title()}: {first['type'].replace('_', ' ').lower()} - {first['entity']}" if n == 1 else
                    f"{kind.title()}: {reason} - {n} affected ({', '.join(sorted({it['type'].replace('_', ' ').lower() for it in items}))})")
            ctx = dict(severity=sev, headline=head, now=str(now), items=items, group_reason=reason if n > 1 else None, dashboard_url=self.cfg.get('dashboard_url'))
            to = list(self.cfg.get('recipients', [])) + (list(self.cfg.get('escalation_recipients', [])) if kind == 'ESCALATION' else [])
            subj = f"[HSM {sev.upper()}] {head}"
            tag = f"{pd.Timestamp(now):%Y%m%dT%H%M}_{kind}_{members[0][0]}"
            res = self.mailer.send(to, subj, self.env.get_template('email.txt.j2').render(**ctx), self.env_html.get_template('email.html.j2').render(**ctx), tag=tag)
            ids = [m[0] for m in members]
            self.store.x("insert into notifications (kind, sent_at, incident_ids, recipients, subject, status, attempts, error, path, group_key) values (?,?,?,?,?,?,?,?,?,?)",
                         kind, str(now), json.dumps(ids), json.dumps(to), subj, res['status'], res['attempts'], res['error'], res['path'], g)
            for iid in ids:
                self.store.x("update incidents set last_notified_at=?, n_notifications=n_notifications+1, notified_open=CASE WHEN ?='OPEN' THEN 1 ELSE notified_open END where id=?",
                             str(now), kind, iid)
            sent.append(dict(kind=kind, subject=subj, n_items=n, status=res['status']))
        return sent

    def step_feed(self, now, feed):
        """data-feed incidents (one per export stream): FEED_STALE opens when the stream stopped being refreshed (critical; reminders with
        back-off) and closes when it is current again. Sensors of a stale stream are 'insufficient information' (frozen) meanwhile."""
        ev = []
        for stream, f in feed.items():
            key = f'FEED_STALE|{stream}'; op = self.store.q("select id, last_notified_at, n_notifications from incidents where key=? and status='open'", key)
            c = dict(type='FEED_STALE', entity=f'{stream} export', entity_kind='feed', family='feed', manufacturer='', model='sensor data export', group='FEED', row=None, feed=f)
            if f['stale'] and not op:
                iid = self.store.x("insert into incidents (key, type, entity, entity_kind, severity, opened_at, status, group_key) values (?,?,?,?,?,?, 'open', 'FEED')",
                                   key, 'FEED_STALE', c['entity'], 'feed', 'critical', str(now)); ev.append(('OPEN', iid, c, 'critical'))
            elif f['stale'] and op:
                iid, last, nn = op[0]; due = min(self.cfg.get('reminder_h', {}).get('critical', 24) * 2 ** max((nn or 1) - 1, 0), self.cfg.get('reminder_max_h', 168))
                if last and (now - pd.Timestamp(last)).total_seconds() / 3600 >= due: ev.append(('REMINDER', iid, c, 'critical'))
            elif op:
                self.store.x("update incidents set status='closed', closed_at=? where id=?", str(now), op[0][0]); ev.append(('RECOVERY', op[0][0], c, 'critical'))
        return self.dispatch(now, ev)

    def acknowledge(self, incident_id, now):
        self.store.x("update incidents set acknowledged_at=? where id=?", str(now), incident_id)
