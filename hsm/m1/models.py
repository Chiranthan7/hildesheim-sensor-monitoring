"""Module 1 candidate models and causal walk-forward evaluation.

Objective: forecast y_H = service availability over the next H hours for every device-hour origin (eligible, not in a declared outage).
Candidates span the pooling spectrum:
  global      : global_rate (pooled mean), gbm_global (one gradient-boosting model, no family identity, common features)
  family      : family_rate, gbm_family (only where M1_SUPERVISED_MIN is met in the fold's training data; family feature set)
  individual  : persistence (last H h), chronic28 (last 28 d incl. outages), bernoulli, markov, hmm2 (analytic, per-device fits)
  hierarchical: eb_markov (device transitions shrunk toward the family's other devices)
  shared      : gbm_shared (all families, family identity + descriptors + all features, family-balanced weights)
  hybrid      : hmm2_famcal (individual HMM2 forecast, isotonic-calibrated per family on training data)
Fold k: test origins in [s_k, s_{k+1}); training origins with hour_end + H <= s_k (purged). Model choice per family x horizon on
validation folds only; the test folds are never used for any choice."""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from ..config import NOMINAL_MIN, M1_HORIZONS_H, M1_SUPERVISED_MIN, M1_DEGRADED_Q, M1_FOLDS, RADIO_METRICS
warnings.filterwarnings('ignore')

COMMON = ['avail_svc_1h', 'avail_svc_6h', 'avail_svc_24h', 'avail_svc_72h', 'avail_svc_168h', 'avail_svc_672h', 'avail_rx_24h', 'avail_rx_168h',
          'avail_valid_24h', 'avail_mode_24h', 'silence_svc_slots', 'silence_rx_slots', 'backend_suspect_now', 'rx_minus_svc_24h', 'excess_24h',
          'history_days', 'reduced_mode', 'network_level', 'hod_sin', 'hod_cos', 'dow', 'fleet_bad_share', 'n_declared_28d', 'h_since_recovery',
          'last_outage_h', 'last_outcome', 'miss_run_slots', 'hmm_p_bad', 'hmm_p_next', 'hmm_silence_surprise', 'bern_p', 'mk_pDD', 'mk_pMD',
          'eb_pDD', 'eb_pMD', 'n28']
RADIO = ['snr_mean_24h', 'snr_mean_168h', 'rssi_mean_24h', 'rssi_mean_168h', 'beyond_ref_share_24h', 'saturation_share_24h',
         'gw_sites_per_uplink_24h', 'snr_drop_24_vs_168']
FAMILY_SPECIFIC = ['channel_completeness_24h', 'meas_zero_share_24h']
ANALYTIC = ['hmm2', 'markov', 'eb_markov', 'bernoulli']
GBM = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, random_state=0)

def horizon_feats(H):
    return [f'hmm_E{H}', f'mk_E{H}', f'eb_E{H}', f'bern_E{H}']

def family_features(fam, H):
    f = COMMON + horizon_feats(H) + FAMILY_SPECIFIC
    return f + ([] if fam in RADIO_METRICS else RADIO)

def tau(train_f, H, fam):
    """family threshold: lower decile of realized availability in normal (no declared outage) training windows"""
    n = max(int(round(H * 60 / NOMINAL_MIN[fam])), 1)
    normal = (train_f[f'y{H}_outage'] == 0) & ~train_f.reduced_mode.astype(bool)     # reduced mode is its own state, not normal variation
    y = train_f.loc[normal, f'y{H}']
    q = y.quantile(M1_DEGRADED_Q) if len(y) >= 50 else np.nan
    return float(min(q, 1 - 1 / n)) if np.isfinite(q) else 1 - 1 / n

def episodes(df, H, thr):
    """distinct degraded episodes: event origins separated by >= H hours within a device"""
    e = df[df[f'y{H}'] < thr].sort_values(['device_id', 'hour_end'])
    if not len(e): return 0
    gap = e.groupby('device_id').hour_end.diff().dt.total_seconds() / 3600
    return int((gap.isna() | (gap >= H)).sum())

def analytic_pred(df, name, H):
    col = {'hmm2': f'hmm_E{H}', 'markov': f'mk_E{H}', 'eb_markov': f'eb_E{H}', 'bernoulli': f'bern_E{H}'}[name]
    p = df[col].copy()
    cov = p.notna()
    for fb in [f'eb_E{H}', 'avail_svc_672h']: p = p.fillna(df[fb])
    return p, cov

def persistence_col(H):
    return {1: 'avail_svc_1h', 6: 'avail_svc_6h', 24: 'avail_svc_24h', 72: 'avail_svc_72h'}[H]

def _usable(df, cols):
    """features with at least one observed value in the training set (an all-missing column carries no information)"""
    return [c for c in cols if df[c].notna().any()]

def _weights(fams):
    n = fams.value_counts(); return fams.map(1 / np.sqrt(n)).values

def fit_predict_fold(X, H, s, e):
    """all candidates for one horizon and one fold; returns predictions for the fold's test origins of every family that uses H"""
    fams = [f for f, hs in M1_HORIZONS_H.items() if H in hs]
    D = X[X.family.isin(fams) & X.eligible_for_prediction & ~X.in_declared_outage]
    tr = D[(D.hour_end + pd.Timedelta(hours=H) <= s) & D[f'y{H}'].notna()]
    te = D[(D.hour_end >= s) & (D.hour_end < e)].copy()            # all eligible origins (scored where the outcome is observed)
    if not len(te) or not len(tr): return None, {}
    info = {}
    trs = tr[tr.hour_end.dt.hour % 3 == 0]                              # thin autocorrelated hourly origins for training
    y = trs[f'y{H}'].values
    # global & shared GBM
    gfeat = _usable(trs, COMMON + horizon_feats(H) + RADIO)
    gb = HistGradientBoostingRegressor(**GBM).fit(trs[gfeat].astype(float), y, sample_weight=_weights(trs.family))
    te['gbm_global'] = np.clip(gb.predict(te[gfeat].astype(float)), 0, 1)
    sfeat = gfeat + _usable(trs, FAMILY_SPECIFIC) + ['fam_code', 'nominal_min']
    code = {f: i for i, f in enumerate(sorted(NOMINAL_MIN))}
    for d_ in (trs, te): d_['fam_code'] = d_.family.map(code); d_['nominal_min'] = d_.family.map(NOMINAL_MIN)
    sh = HistGradientBoostingRegressor(**GBM, categorical_features=[sfeat.index('fam_code')]).fit(
        trs[sfeat].astype(float), y, sample_weight=_weights(trs.family))
    te['gbm_shared'] = np.clip(sh.predict(te[sfeat].astype(float)), 0, 1)
    te['global_rate'] = float(np.average(y, weights=_weights(trs.family)))
    te['persistence'] = te[persistence_col(H)]; te['chronic28'] = te.avail_svc_672h
    for a in ANALYTIC: te[a], te[f'{a}_cov'] = analytic_pred(te, a, H)
    te['gbm_family'] = np.nan; te['hmm2_famcal'] = np.nan; te['family_rate'] = np.nan; te['tau'] = np.nan
    # global isotonic (fallback for small families)
    trh, _ = analytic_pred(tr, 'hmm2', H)
    iso_g = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(trh.values, tr[f'y{H}'].values)
    for fam in fams:
        tf = tr[tr.family == fam]; m = te.family == fam
        if not m.any(): continue
        thr = tau(tf, H, fam) if len(tf) else 1 - 1 / max(int(round(H * 60 / NOMINAL_MIN[fam])), 1)
        te.loc[m, 'tau'] = thr
        te.loc[m, 'family_rate'] = tf[f'y{H}'].mean() if len(tf) else te.loc[m, 'global_rate']
        ph, _ = analytic_pred(tf, 'hmm2', H)
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(ph.values, tf[f'y{H}'].values) if len(tf) >= 500 else iso_g
        te.loc[m, 'hmm2_famcal'] = iso.predict(te.loc[m, 'hmm2'].values)
        n_dev = tf.device_id.nunique(); n_ep = episodes(tf, H, thr)
        suff = n_dev >= M1_SUPERVISED_MIN['devices'] and n_ep >= M1_SUPERVISED_MIN['degraded_episodes']
        info[fam] = dict(train_outcome_end_max=(tr.hour_end.max() + pd.Timedelta(hours=H)) if len(tr) else pd.NaT, train_devices=int(n_dev), train_degraded_episodes=n_ep, train_origins=int(len(tf)), supervised_family_model=bool(suff), tau=thr)
        if suff:
            tfs = tf[tf.hour_end.dt.hour % 3 == 0]; ff = _usable(tfs, family_features(fam, H))
            fm = HistGradientBoostingRegressor(**GBM).fit(tfs[ff].astype(float), tfs[f'y{H}'].values)
            te.loc[m, 'gbm_family'] = np.clip(fm.predict(te.loc[m, ff].astype(float)), 0, 1)
    return te, info

CANDIDATES = ['global_rate', 'gbm_global', 'family_rate', 'gbm_family', 'persistence', 'chronic28', 'bernoulli', 'markov', 'hmm2',
              'eb_markov', 'gbm_shared', 'hmm2_famcal']
LEVEL = dict(global_rate='global', gbm_global='global', family_rate='family', gbm_family='family', persistence='individual',
             chronic28='individual', bernoulli='individual', markov='individual', hmm2='individual', eb_markov='hierarchical',
             gbm_shared='shared', hmm2_famcal='hybrid')

def metrics(y, p, thr):
    ok = ~np.isnan(p) & ~np.isnan(y); y, p, thr = y[ok], p[ok], thr[ok]
    ev = y < thr
    r = dict(n=int(len(y)), rmse=float(np.sqrt(np.mean((p - y) ** 2))) if len(y) else np.nan, mae=float(np.mean(np.abs(p - y))) if len(y) else np.nan,
             bias=float(np.mean(p - y)) if len(y) else np.nan, event_rate=float(ev.mean()) if len(y) else np.nan)
    r['auc'] = float(roc_auc_score(ev, -p)) if 0 < ev.sum() < len(ev) else np.nan
    r['ap'] = float(average_precision_score(ev, -p)) if 0 < ev.sum() < len(ev) else np.nan
    pred_ev = p < thr
    r['precision'] = float(ev[pred_ev].mean()) if pred_ev.any() else np.nan
    r['recall'] = float(pred_ev[ev].mean()) if ev.any() else np.nan
    return r

def walk_forward(X):
    starts = [pd.Timestamp(x) for x in M1_FOLDS['validation'] + M1_FOLDS['test']] + [pd.Timestamp(M1_FOLDS['end'])]
    role = {pd.Timestamp(x): 'validation' for x in M1_FOLDS['validation']} | {pd.Timestamp(x): 'test' for x in M1_FOLDS['test']}
    preds, infos = [], []
    for H in sorted({h for v in M1_HORIZONS_H.values() for h in v}):
        for s, e in zip(starts[:-1], starts[1:]):
            te, info = fit_predict_fold(X, H, s, e)
            if te is None: continue
            te['H'] = H; te['fold'] = str(s.date()); te['role'] = role[s]
            keep = ['device_id', 'family', 'hour_end', 'H', 'fold', 'role', 'tau', f'y{H}', f'y{H}_outage'] + CANDIDATES + [f'{a}_cov' for a in ANALYTIC]
            preds.append(te[keep].rename(columns={f'y{H}': 'y', f'y{H}_outage': 'y_outage'}))
            for f, d in info.items(): infos.append(dict(H=H, fold=str(s.date()), family=f, **d))
            print(f'H={H} fold={s.date()} test origins={len(te)}', flush=True)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(infos)

def choose(P, decision_time, default='eb_markov'):
    """model choice in force from decision_time: uses ONLY validation origins whose outcome window closed by then
    (hour_end + H <= decision_time). Per family x horizon: lowest validation RMSE among candidates covering >= 99% of the origins
    (>= 200 origins); otherwise the universal choice (best pooled over all families); with no validation data: default."""
    d0 = pd.Timestamp(decision_time)
    V = P[(P.role == 'validation') & P.y.notna() & (P.hour_end + pd.to_timedelta(P.H, unit='h') <= d0)]
    uni, choice = {}, {}
    for H, d in V.groupby('H'):
        sc = {c: np.sqrt(np.nanmean((d[c].astype(float) - d.y) ** 2)) for c in CANDIDATES if d[c].notna().mean() > 0.99}
        if sc: uni[H] = min(sc, key=sc.get)
    for (fam, H) in P[['family', 'H']].drop_duplicates().itertuples(index=False):
        v = V[(V.family == fam) & (V.H == H)]
        if len(v) >= 200:
            sc = {c: np.sqrt(np.nanmean((v[c].astype(float) - v.y) ** 2)) for c in CANDIDATES if v[c].notna().mean() >= 0.99}
            choice[(fam, H)] = (min(sc, key=sc.get), 'validation')
        else:
            choice[(fam, H)] = (uni[H], 'universal_fallback') if H in uni else (default, 'default')
    return choice, uni

def score_table(P):
    rows = []
    for (fam, H, role), d in P[P.y.notna()].groupby(['family', 'H', 'role']):
        for c in CANDIDATES:
            r = metrics(d.y.values, d[c].values.astype(float), d.tau.values.astype(float)); rows.append(dict(family=fam, H=H, role=role, candidate=c, level=LEVEL[c], **r))
    return pd.DataFrame(rows)

def compare_on_test(P, choice, uni, n_boot=1000, seed=0):
    """test-period RMSE of the per-family selection vs (a) the universal choice applied to every family and (b) gbm_global;
    95% CI of the difference by device-block bootstrap (devices resampled with replacement within family x horizon)"""
    rng = np.random.default_rng(seed); comp = []
    T = P[(P.role == 'test') & P.y.notna()]
    for (fam, H), d in T.groupby(['family', 'H']):
        sel, why = choice[(fam, H)]
        out = dict(family=fam, H=H, selected=sel, selected_level=LEVEL[sel], selection_basis=why, universal=uni.get(H), n_test=len(d), n_devices=d.device_id.nunique())
        se = lambda c: (d[c].astype(float) - d.y) ** 2
        out['rmse_selected'] = float(np.sqrt(se(sel).mean()))
        for ref in ['gbm_global', 'universal']:
            rc = uni.get(H) if ref == 'universal' else ref
            a = se(sel); b = se(rc); ok = a.notna() & b.notna()
            dev = d.device_id[ok]; da = a[ok].groupby(dev).sum(); db = b[ok].groupby(dev).sum(); dn = ok[ok].groupby(dev).size()
            ids = da.index.values
            diffs = [np.sqrt(da[s_].sum() / dn[s_].sum()) - np.sqrt(db[s_].sum() / dn[s_].sum()) for s_ in (rng.choice(ids, len(ids)) for _ in range(n_boot))]
            out[f'rmse_{ref}'] = float(np.sqrt(b[ok].mean())); out[f'diff_vs_{ref}'] = out['rmse_selected'] - out[f'rmse_{ref}']
            out[f'diff_vs_{ref}_ci95'] = (float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975)))
        comp.append(out)
    return pd.DataFrame(comp)

def decision_schedule(P):
    """causal model choices: each fold uses the choice made at its start"""
    starts = sorted(P.fold.unique())
    test0 = pd.Timestamp(M1_FOLDS['test'][0])
    sched = {}
    for f in starts:
        ft = pd.Timestamp(f)
        sched[f] = choose(P, min(ft, test0) if ft >= test0 else ft)    # test folds use the pre-registered choice made at test start
    return sched
