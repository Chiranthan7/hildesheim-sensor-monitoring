"""Component 1: sensor interruption and failure risk.  usage: python run_risk.py --foundation <dir>
Requires the Module 1 outputs. Writes gold_risk_* tables and risk_evaluation.json."""
import argparse, json, time
import numpy as np, pandas as pd
from hsm import risk, config
from hsm.m1.state import primary_h

def main(fd):
    t0 = time.time(); rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    X = rd('gold_m1_features'); L = rd('gold_silences'); U = rd('gold_uplinks')
    man = json.load(open(f'{fd}/manifest.json')); obs_end = {k: pd.Timestamp(v) for k, v in man['observation_end_service'].items()}
    X = risk.extra_features(X, L, U, rd('silver_devices'), rd('silver_soil'), rd('silver_weather'), rd('gold_site_hour'))
    print('features', round(time.time() - t0), flush=True)
    X = risk.targets(X, L, obs_end)
    X = pd.concat([X, risk.current_gap(X, L)], axis=1)
    X.to_parquet(f'{fd}/gold_risk_features.parquet', index=False); print('targets', round(time.time() - t0), flush=True)
    E = risk.events(L)
    parts = [risk.walk_forward_A(X, E, lambda f: f'onset_{risk.RISK_HORIZONS_H[f][0]}h', lambda f: risk.RISK_HORIZONS_H[f][0], 'A_onset_short'),
             risk.walk_forward_A(X, E, lambda f: f'onset_{risk.RISK_HORIZONS_H[f][1]}h', lambda f: risk.RISK_HORIZONS_H[f][1], 'A_onset_long')]
    for D in risk.PROLONGED_D:
        parts.append(risk.walk_forward_A(X, E, lambda f, D=D: f'prolonged_{D}d', lambda f, D=D: 24 * D, f'A_prolonged_{D}d'))
    PA = pd.concat(parts, ignore_index=True); PB = risk.walk_forward_B(X); PD = risk.walk_forward_D(X, E, obs_end)
    for n, P in [('A', PA), ('B', PB), ('D', PD)]: P.to_parquet(f'{fd}/gold_risk_pred_{n}.parquet', index=False)
    print('walk-forward', round(time.time() - t0), flush=True)
    finish(fd, X, E, PA, PB, PD, obs_end, t0)

def finish(fd, X, E, PA, PB, PD, obs_end, t0=None):
    t0 = t0 or time.time(); test0 = config.M1_FOLDS['test'][0]
    out = {}; all_scores = []; event_rows = []; calib = []
    for P, C, default in [(PA, risk.CAND_A, 'device_hazard'), (PB, risk.CAND_B, 'silence_length_rate'), (PD, risk.CAND_D, 'km_family')]:
        S = risk.scores(P, C); all_scores.append(S)
        ch = risk.choose(S, P, C, test0, default)
        for (task, fam, H), (cand, why) in ch.items():
            v = P[(P.task == task) & (P.family == fam) & (P.H == H) & (P.role == 'validation') & P.y.notna()]
            thr = risk.alert_threshold(v, cand) if len(v) and v.y.sum() >= 5 and v[cand].notna().all() else np.nan
            t = P[(P.task == task) & (P.family == fam) & (P.H == H) & (P.role == 'test') & P.y.notna()]
            r = dict(task=task, family=fam, H=H, selected=cand, basis=why, threshold=thr, n_test=len(t), positives_test=int(t.y.sum()) if len(t) else 0)
            if len(t) and 0 < t.y.sum() < len(t) and t[cand].notna().all():
                from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
                p = t[cand].astype(float).values; y = t.y.values
                r.update(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p), base_rate=y.mean())
                if np.isfinite(thr):
                    a = p >= thr; tp = (a & (y == 1)).sum()
                    r.update(precision=tp / a.sum() if a.sum() else np.nan, recall=tp / y.sum(), f1=2 * tp / (a.sum() + y.sum()))
                    if task.startswith('A'):
                        r.update(risk.event_metrics(t.assign(**{cand: t[cand]}), cand, thr, E, H))
                # calibration bins (test)
                b = pd.qcut(p, 10, duplicates='drop')
                cb = pd.DataFrame({'p': p, 'y': y, 'b': b}).groupby('b', observed=True).agg(pred=('p', 'mean'), obs=('y', 'mean'), n=('y', 'size')).reset_index(drop=True)
                cb['task'] = task; cb['family'] = fam; cb['H'] = H; calib.append(cb)
                r['ece'] = float((cb.n * (cb.pred - cb.obs).abs()).sum() / cb.n.sum())
            event_rows.append(r)
        out.update({f'{k[0]}|{k[1]}|{k[2]}': v[0] for k, v in ch.items()})
    S = pd.concat(all_scores, ignore_index=True); S.to_parquet(f'{fd}/gold_risk_scores.parquet', index=False)
    R = pd.DataFrame(event_rows); R.to_parquet(f'{fd}/gold_risk_test_summary.parquet', index=False)
    pd.concat(calib, ignore_index=True).to_parquet(f'{fd}/gold_risk_calibration.parquet', index=False)
    risk.detection_delay(E).reset_index().to_parquet(f'{fd}/gold_risk_detection_delay.parquet', index=False)
    as_of = max(obs_end.values())
    risk.chronic_rates(X, E, as_of).to_parquet(f'{fd}/gold_risk_chronic.parquet', index=False)
    risk.time_pattern(X, E).to_parquet(f'{fd}/gold_risk_time_pattern.parquet', index=False)
    b = R[R.task == 'B_developing']
    eb = []
    for r in b.itertuples():
        if np.isfinite(r.threshold):
            x = risk.early_detection_events(PB[PB.family == r.family], r.selected, r.threshold); eb.append(x)
    if eb: pd.concat(eb, ignore_index=True).to_parquet(f'{fd}/gold_risk_early_detection_events.parquet', index=False)
    current(fd, X, PA, PB, PD, R)
    json.dump(dict(choice=out, runtime_s=round(time.time() - t0, 1)), open(f'{fd}/risk_evaluation.json', 'w'), indent=1, default=str)
    print('done', round(time.time() - t0), flush=True)

def current(fd, X, PA, PB, PD, R):
    """risk per device-hour from the walk-forward predictions of the selected model (each fold's model was trained before the fold)"""
    sel = {(r.task, r.family, r.H): (r.selected, r.threshold) for r in R.itertuples()}
    rows = []
    for P in (PA, PB, PD):
        for (task, fam, H), d in P.groupby(['task', 'family', 'H']):
            c, thr = sel.get((task, fam, H), (None, np.nan))
            if c is None: continue
            rows.append(d[['device_id', 'family', 'hour_end']].assign(task=task, H=H, model=c, p=d[c].values, threshold=thr))
    Q = pd.concat(rows, ignore_index=True)
    Q.to_parquet(f'{fd}/gold_risk_hourly.parquet', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); main(ap.parse_args().foundation)
