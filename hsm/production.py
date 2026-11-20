"""Production models (deployment): the walk-forward evaluation produces predictions only for historical folds; for live monitoring the
selected model types are refitted on ALL labelled data available now and stored with joblib. Scoring applies them to the latest hours.
Model types follow the evaluation: shared gradient boosting for interruption risk where it was selected / supervised data suffice,
the gamma-Poisson device hazard otherwise; Kaplan-Meier (family) for continued unavailability; the M1 availability forecast uses the
individual / hierarchical analytic models (no training) unless a supervised model was selected for the family."""
import json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from . import risk
from .m1 import models as m1m
from .config import NOMINAL_MIN, SLOT_GRACE_FRAC

def train(fd, out_dir):
    fd = Path(fd); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X = pd.read_parquet(fd / 'gold_risk_features.parquet'); E = risk.events(pd.read_parquet(fd / 'gold_silences.parquet'))
    meta = {}
    base = X[X.eligible_for_prediction & ~X.in_declared_outage & (X.silence_svc_slots <= 1 + SLOT_GRACE_FRAC)].copy()
    for task, tgt in [('A_onset_long', lambda f: f'onset_{risk.RISK_HORIZONS_H[f][1]}h'), ('A_prolonged_7d', lambda f: 'prolonged_7d')]:
        B = base.copy(); B['H'] = B.family.map(lambda f: risk.RISK_HORIZONS_H[f][1] if task == 'A_onset_long' else 168).astype(float); B['y'] = np.nan
        for f in NOMINAL_MIN: B.loc[B.family == f, 'y'] = B.loc[B.family == f, tgt(f)]
        tr = B[B.y.notna()]; trs = tr[tr.hour_end.dt.hour % 2 == 0].copy()
        cols = m1m._usable(trs, risk._feat()); trs['fam_code'] = trs.family.map(risk.CODE); trs['H_feat'] = trs.H
        cs = cols + ['fam_code', 'H_feat']
        m = HistGradientBoostingClassifier(**risk.GBC, categorical_features=[cs.index('fam_code')]).fit(trs[cs].astype(float), trs.y, sample_weight=m1m._weights(trs.family))
        joblib.dump(dict(model=m, cols=cs), out / f'{task}.joblib')
        ex = tr.groupby('device_id').size().astype(float); fam = tr.groupby('device_id').family.first()
        meta[task] = dict(n_train=len(trs), positives=int(trs.y.sum()), trained_until=str(tr.hour_end.max()),
                          hazard=dict(events=E.groupby('device_id').size().to_dict(), exposure=ex.to_dict(), family=fam.to_dict()))
    # B: developing silences
    Bd = X[X.eligible_for_prediction & X.developing_silence & ~X.in_declared_outage & X.gap_status.isin(['declared', 'burst'])].copy()
    Bd['y'] = (Bd.gap_status == 'declared').astype(float); Bd['H'] = 0.0; Bd['surprise'] = 1 - Bd.hmm_silence_surprise; Bd['log_silence'] = np.log1p(Bd.silence_svc_slots)
    cols = m1m._usable(Bd, risk._feat() + ['log_silence', 'surprise']); Bd['fam_code'] = Bd.family.map(risk.CODE); Bd['H_feat'] = 0.0
    cs = cols + ['fam_code', 'H_feat']
    m = HistGradientBoostingClassifier(**risk.GBC, categorical_features=[cs.index('fam_code')]).fit(Bd[cs].astype(float), Bd.y, sample_weight=m1m._weights(Bd.family))
    joblib.dump(dict(model=m, cols=cs), out / 'B_developing.joblib'); meta['B_developing'] = dict(n_train=len(Bd), positives=int(Bd.y.sum()))
    # D: Kaplan-Meier per family on all outages (recovered = event, ongoing = censored at the last hour)
    last = X.hour_end.max(); Ev = E.copy(); end = Ev.recovered_at.fillna(last); Ev['dur'] = (end - Ev.onset).dt.total_seconds() / 3600; Ev['obs'] = Ev.recovered_at.notna()
    km = {}
    for f, g in list(Ev.groupby('family')) + [('__all__', Ev)]:
        ut, S = risk.km(g.dur.values, g.obs.values.astype(bool)); km[f] = dict(t=ut.tolist(), S=S.tolist(), n=len(g))
    json.dump(km, open(out / 'D_km.json', 'w')); json.dump(meta, open(out / 'meta.json', 'w'), indent=1, default=str)
    return meta

def score(fd, model_dir, hours=None):
    """risk columns for the given hour ends (default: the latest hour of every device)"""
    fd = Path(fd); md = Path(model_dir)
    X = pd.read_parquet(fd / 'gold_risk_features.parquet')
    if hours is None: X = X.sort_values('hour_end').groupby('device_id').tail(1).copy()
    else: X = X[X.hour_end.isin(hours)].copy()
    X['fam_code'] = X.family.map(risk.CODE)
    out = X[['device_id', 'family', 'hour_end']].copy()
    for task, col, Hf in [('A_onset_long', 'risk_long', lambda f: risk.RISK_HORIZONS_H[f][1]), ('A_prolonged_7d', 'risk_prolonged_7d', lambda f: 168)]:
        b = joblib.load(md / f'{task}.joblib'); X['H_feat'] = X.family.map(Hf).astype(float)
        out[col] = b['model'].predict_proba(X[b['cols']].astype(float))[:, 1]
    X['surprise'] = 1 - X.hmm_silence_surprise; X['log_silence'] = np.log1p(X.silence_svc_slots); X['H_feat'] = 0.0
    b = joblib.load(md / 'B_developing.joblib'); out['p_silence_becomes_outage'] = np.where(X.developing_silence, b['model'].predict_proba(X[b['cols']].astype(float))[:, 1], np.nan)
    km = json.load(open(md / 'D_km.json'))
    el = (X.hour_end - X.gap_onset).dt.total_seconds() / 3600
    for Dh, col in [(24, 'p_still_down_24h'), (168, 'p_still_down_7d')]:
        v = []
        for f, e, io in zip(X.family, el, X.in_declared_outage):
            k = km.get(f) if km.get(f, {}).get('n', 0) >= 10 else km['__all__']
            ut, S = np.array(k['t']), np.array(k['S'])
            a = risk.km_eval(ut, S, np.array([e])); b_ = risk.km_eval(ut, S, np.array([e + Dh]))
            v.append(float(b_[0] / a[0]) if (io and a[0] > 0) else np.nan)
        out[col] = v
    return out
