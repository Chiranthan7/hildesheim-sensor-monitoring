"""Sensor interruption and failure-risk prediction (component 1).

Interruption event = onset of a causally DECLARED outage (service unavailability, whatever the cause). Prolonged = duration >= 24 h.
Tasks are kept separate (never pooled in one metric):
  A  prediction BEFORE onset: origin hours with no developing silence (silence <= 1.5 intervals), not in an outage.
     target_H = an outage onset in (h, h+H]; prolonged_D = onset of an outage lasting >= 24 h in (h, h+D days]
  B  early detection of a DEVELOPING silence: origin hours inside a silence that is not yet declared.
     target = this silence is later declared an outage (vs it recovers first)
  C  detection delay of the declaration rule: declared_at - onset (descriptive; the rule defines the outage)
  D  continued unavailability: origin hours inside a declared outage; target = still unavailable after +24 h / +7 d
  E  chronic recurrence: device onset rate (gamma-Poisson shrinkage to the family rate)
Censoring: a target is defined only when its whole window is observed (window end <= data observation end) and, for
prolonged / continuation targets, when the duration criterion is resolved before the observation end."""
import numpy as np, pandas as pd, warnings
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from .config import NOMINAL_MIN, M1_FOLDS, M1_SUPERVISED_MIN, SLOT_GRACE_FRAC, BATTERY_OBSERVABILITY, const
from .m1.models import COMMON, RADIO, FAMILY_SPECIFIC, _usable, _weights
warnings.filterwarnings('ignore')

RISK_HORIZONS_H = const('RISK_HORIZONS_H', {f: ([24, 72] if NOMINAL_MIN[f] >= 240 else [6, 24]) for f in NOMINAL_MIN}, 'operational',
                        'labels|evaluation', 'interruption-onset horizons: 6 h / 24 h; 24 h / 72 h for families reporting every >= 4 h')
PROLONGED_H = const('PROLONGED_H', 24, 'operational', 'labels|evaluation', 'an outage lasting >= 24 h is prolonged')
PROLONGED_D = const('PROLONGED_D', [7, 14], 'operational', 'labels|evaluation', 'horizons (days) for prolonged-interruption risk')
CONTINUATION_H = const('CONTINUATION_H', [24, 168], 'operational', 'labels|evaluation', 'continued-unavailability horizons')
GBC = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=0)
LOGIT_FEATS = ['avail_svc_24h', 'avail_svc_168h', 'avail_svc_672h', 'n_declared_28d', 'n_onsets_7d', 'outage_h_7d', 'h_since_recovery_log',
               'hmm_p_bad', 'eb_E24', 'rx_minus_svc_24h', 'fleet_bad_share', 'gw_sites_168h', 'primary_gw_share_168h']

# ------------------------------------------------------------------ events and extra features
def events(L):
    d = L[L.status == 'declared'][['device_id', 'family', 'gap_id', 'onset', 'declared_at', 'recovered_at', 'scope_at_declaration', 'cause_hint']].copy()
    d['duration_h'] = (d.recovered_at - d.onset).dt.total_seconds() / 3600
    return d.sort_values(['device_id', 'onset']).reset_index(drop=True)

def _per_device_window_counts(X, E, col_t, window, value=None):
    out = np.zeros(len(X))
    for dev, idx in X.groupby('device_id').indices.items():
        e = E[E.device_id == dev]
        if not len(e): continue
        t = np.sort(e[col_t].values.astype('datetime64[ns]')); h = X.hour_end.values[idx].astype('datetime64[ns]')
        out[idx] = np.searchsorted(t, h, 'left') - np.searchsorted(t, h - np.timedelta64(window, 'h'), 'left')
    return out

def extra_features(X, L, U, devs, soil, wx, site_hour):
    """outage trend (known declarations only), gateway dependence, battery, city weather, measurement anomaly shares"""
    X = X.sort_values(['device_id', 'hour_end']).reset_index(drop=True); E = events(L)
    known = E.copy()                                                    # an outage is known from its declaration time
    X['n_onsets_7d'] = _per_device_window_counts(X, known, 'declared_at', 168)
    X['onset_rate_ratio_7_28'] = (X.n_onsets_7d / 7) / ((X.n_declared_28d + 1) / 28)
    X['h_since_recovery_log'] = np.log1p(X.h_since_recovery.fillna(24 * 60))
    # outage hours in last 7 d (declared outages, counted up to h)
    oh = np.zeros(len(X))
    for dev, idx in X.groupby('device_id').indices.items():
        e = E[E.device_id == dev]
        if not len(e): continue
        h = X.hour_end.values[idx].astype('datetime64[ns]')[:, None]
        on = e.onset.values.astype('datetime64[ns]')[None, :]; de = e.declared_at.values.astype('datetime64[ns]')[None, :]
        en = np.where(np.isnat(e.recovered_at.values), np.datetime64('2262-01-01', 'ns'), e.recovered_at.values.astype('datetime64[ns]'))[None, :]
        lo = np.maximum(on, h - np.timedelta64(168, 'h')); hi = np.minimum(en, h)
        ov = ((hi - lo).astype('timedelta64[m]').astype(float) / 60).clip(0)
        oh[idx] = (ov * (de <= h)).sum(1)
    X['outage_h_7d'] = oh
    # gateway dependence (causal site at reception time)
    u = U[U.gw_site.notna() & (U.gw_site != 'unresolved')][['device_id', 't', 'gw_site', 'n_gw_sites']].copy()
    u['hour_end'] = u.t.dt.floor('h') + pd.Timedelta('1h')
    g = u.groupby(['device_id', 'hour_end', 'gw_site']).size().rename('n').reset_index()
    gw = {}
    for dev, x in g.groupby('device_id'):
        p = x.pivot_table(index='hour_end', columns='gw_site', values='n', fill_value=0)
        hrs = X.hour_end[X.device_id == dev]
        p = p.reindex(pd.DatetimeIndex(sorted(set(p.index) | set(hrs))), fill_value=0)
        r168 = p.rolling(168, min_periods=1).sum(); r24 = p.rolling(24, min_periods=1).sum()
        tot = r168.sum(1).replace(0, np.nan); prim = r168.idxmax(axis=1)
        sh = site_hour.set_index('hour_end') if 'hour_end' in site_hour else site_hour
        act = np.array([sh.at[h, s] if (h in sh.index and s in sh.columns) else np.nan for h, s in zip(r168.index, prim)])
        base = sh.rolling(168, min_periods=24).mean()
        bas = np.array([base.at[h, s] if (h in base.index and s in base.columns) else np.nan for h, s in zip(r168.index, prim)])
        gw[dev] = pd.DataFrame({'gw_sites_24h': (r24 > 0).sum(1), 'gw_sites_168h': (r168 > 0).sum(1), 'primary_gw_share_168h': r168.max(1) / tot,
                                'primary_gw_activity_ratio': act / np.where(bas > 0, bas, np.nan)}, index=r168.index).reindex(hrs).reset_index(drop=True)
    G = pd.concat([gw[d].assign(device_id=d, hour_end=X.hour_end[X.device_id == d].values) for d in gw], ignore_index=True) if gw else None
    if G is not None: X = X.merge(G, on=['device_id', 'hour_end'], how='left')
    X['gw_diversity_loss'] = (X.gw_sites_168h - X.gw_sites_24h).clip(lower=0)
    # battery (families with a voltage signal): 24 h minimum vs 7-day median
    b = devs[(devs.dup_rank == 0) & devs.battery_v.notna()][['device_id', 't', 'battery_v']].copy(); b['hour_end'] = b.t.dt.floor('h') + pd.Timedelta('1h')
    bh = b.groupby(['device_id', 'hour_end']).battery_v.agg(['min', 'median']).reset_index()
    X = X.merge(bh, on=['device_id', 'hour_end'], how='left')
    gb = X.groupby('device_id', sort=False)
    X['batt_min_24h'] = gb['min'].transform(lambda s: s.rolling(24, min_periods=1).min())
    X['batt_med_168h'] = gb['median'].transform(lambda s: s.rolling(168, min_periods=6).median())
    X['batt_drop_24_vs_168'] = X.batt_med_168h - X.batt_min_24h
    X = X.drop(columns=['min', 'median'])
    # city weather (MeteoHelix / MeteoWind fleet medians, hour <= h)
    w = wx[(wx.dup_rank == 0) & wx.family.isin(['MeteoHelix', 'MeteoWind']) & ~wx.flag_bad_packet.fillna(False)].copy()
    w['hour_end'] = w.t.dt.floor('h') + pd.Timedelta('1h')
    st = w.groupby(['device_id', 'hour_end']).agg(rain=('rain_mm', 'sum'), temp=('temp_c', 'mean'), wind=('wind_ms', 'mean'), rh=('rh_pct', 'mean')).reset_index()
    city = st.groupby('hour_end').agg(city_rain_1h=('rain', 'median'), city_temp=('temp', 'median'), city_wind=('wind', 'median'), city_rh=('rh', 'median'))
    city = city.reindex(pd.date_range(city.index.min(), city.index.max(), freq='h'))
    city['city_rain_6h'] = city.city_rain_1h.rolling(6, min_periods=1).sum()
    X = X.merge(city.rename_axis('hour_end').reset_index(), on='hour_end', how='left')
    # measurement anomaly share (record-level invalid flags) in the last 24 h
    s = soil[(soil.dup_rank == 0) & soil.data_stored].copy(); s['hour_end'] = s.t.dt.floor('h') + pd.Timedelta('1h')
    s['bad'] = (s.flag_soil_temp_out_of_spec | s.flag_rawfreq_fault | s.flag_temp_stuck_sofar).astype(float)
    sb = s.groupby(['device_id', 'hour_end']).agg(bad_s=('bad', 'sum'), bad_n=('bad', 'size')).reset_index()
    X = X.merge(sb, on=['device_id', 'hour_end'], how='left'); X[['bad_s', 'bad_n']] = X[['bad_s', 'bad_n']].fillna(0)
    gb = X.groupby('device_id', sort=False)
    X['meas_invalid_share_24h'] = gb.bad_s.transform(lambda s: s.rolling(24, min_periods=1).sum()) / gb.bad_n.transform(lambda s: s.rolling(24, min_periods=1).sum()).replace(0, np.nan)
    X = X.drop(columns=['bad_s', 'bad_n'])
    return X

EXTRA = ['n_onsets_7d', 'onset_rate_ratio_7_28', 'h_since_recovery_log', 'outage_h_7d', 'gw_sites_24h', 'gw_sites_168h', 'primary_gw_share_168h',
         'primary_gw_activity_ratio', 'gw_diversity_loss', 'batt_min_24h', 'batt_med_168h', 'batt_drop_24_vs_168', 'city_rain_1h', 'city_rain_6h',
         'city_temp', 'city_wind', 'city_rh', 'meas_invalid_share_24h']

# ------------------------------------------------------------------ targets
def targets(X, L, obs_end):
    """task A / B / D targets with censoring"""
    E = events(L); X = X.copy()
    oe = X.family.map(obs_end).astype('datetime64[ns]')
    for fam_h in sorted({h for v in RISK_HORIZONS_H.values() for h in v}):
        X[f'onset_{fam_h}h'] = np.nan
    for D in PROLONGED_D: X[f'prolonged_{D}d'] = np.nan
    for dev, idx in X.groupby('device_id').indices.items():
        h = X.hour_end.values[idx].astype('datetime64[ns]'); e = E[E.device_id == dev]
        on = np.sort(e.onset.values.astype('datetime64[ns]'))
        lp = e[(e.duration_h >= PROLONGED_H)]; onp = np.sort(lp.onset.values.astype('datetime64[ns]'))
        o_end = oe.values[idx]
        for H in sorted({h_ for v in RISK_HORIZONS_H.values() for h_ in v}):
            n = np.searchsorted(on, h + np.timedelta64(H, 'h'), 'right') - np.searchsorted(on, h, 'right')
            X.loc[X.index[idx], f'onset_{H}h'] = np.where(h + np.timedelta64(H, 'h') <= o_end, (n > 0).astype(float), np.nan)
        for D in PROLONGED_D:
            n = np.searchsorted(onp, h + np.timedelta64(D, 'D'), 'right') - np.searchsorted(onp, h, 'right')
            X.loc[X.index[idx], f'prolonged_{D}d'] = np.where(h + np.timedelta64(D, 'D') + np.timedelta64(PROLONGED_H, 'h') <= o_end, (n > 0).astype(float), np.nan)
    return X

def current_gap(X, L):
    """service gap containing each hour end (anchor_t < h <= next arrival): its eventual status and onset / declared time"""
    out = pd.DataFrame(index=X.index, columns=['gap_status', 'gap_onset', 'gap_declared_at', 'gap_recovered_at', 'gap_scope'])
    for dev, idx in X.groupby('device_id').indices.items():
        g = L[L.device_id == dev].sort_values('anchor_t')
        if not len(g): continue
        h = X.hour_end.values[idx].astype('datetime64[ns]'); j = np.searchsorted(g.anchor_t.values.astype('datetime64[ns]'), h, 'left') - 1
        ok = j >= 0; jj = np.maximum(j, 0)
        for c, src in [('gap_status', 'status'), ('gap_onset', 'onset'), ('gap_declared_at', 'declared_at'), ('gap_recovered_at', 'recovered_at'), ('gap_scope', 'scope_at_declaration')]:
            v = g[src].values[jj]; out.iloc[idx, out.columns.get_loc(c)] = np.where(ok, v, None)
    for c in ['gap_onset', 'gap_declared_at', 'gap_recovered_at']: out[c] = pd.to_datetime(out[c])
    return out

# ------------------------------------------------------------------ models
def _feat(fam=None):
    return COMMON + ['eb_E24', 'hmm_E24', 'mk_E24'] + RADIO + FAMILY_SPECIFIC + EXTRA

def device_hazard(tr_events, tr, te, H, _=None):
    """gamma-Poisson: device onset rate (onsets per exposure hour in the training data) shrunk to its family rate;
    prior strength = 168 h of exposure at the family rate. P(onset within H) = 1 - exp(-lambda H)."""
    ex_dev = tr.groupby('device_id').size().astype(float); fam_dev = tr.groupby('device_id').family.first()
    ex_fam = ex_dev.groupby(fam_dev).sum(); k_fam = tr_events.groupby('family').size().reindex(ex_fam.index, fill_value=0)
    lam_f = (k_fam / ex_fam).to_dict(); lam_all = k_fam.sum() / max(ex_fam.sum(), 1)
    k = te.device_id.map(tr_events.groupby('device_id').size()).fillna(0).values; ex = te.device_id.map(ex_dev).fillna(0).values
    lf = te.family.map(lam_f).fillna(lam_all).values
    lam = (168 * lf + k) / (168 + ex)
    return 1 - np.exp(-lam * np.asarray(H, float))

def episodes_count(df, col, H):
    e = df[df[col] == 1].sort_values(['device_id', 'hour_end'])
    if not len(e): return 0
    gap = e.groupby('device_id').hour_end.diff().dt.total_seconds() / 3600
    return int((gap.isna() | (gap > H)).sum())

FOLDS = lambda: ([pd.Timestamp(x) for x in M1_FOLDS['validation'] + M1_FOLDS['test']] + [pd.Timestamp(M1_FOLDS['end'])],
                  {pd.Timestamp(x): 'validation' for x in M1_FOLDS['validation']} | {pd.Timestamp(x): 'test' for x in M1_FOLDS['test']})
CODE = {f: i for i, f in enumerate(sorted(NOMINAL_MIN))}

def _fit_shared(trs, te, cols, extra=()):
    cs = cols + ['fam_code', 'H_feat'] + list(extra)
    for d_ in (trs, te): d_['fam_code'] = d_.family.map(CODE); d_['H_feat'] = d_.H
    m = HistGradientBoostingClassifier(**GBC, categorical_features=[cs.index('fam_code')]).fit(trs[cs].astype(float), trs.y, sample_weight=_weights(trs.family))
    return m.predict_proba(te[cs].astype(float))[:, 1]

def _fit_logit(trs, te, feats):
    lf = _usable(trs, feats)
    dm = lambda d: pd.concat([d[lf], pd.get_dummies(d.family).reindex(columns=sorted(NOMINAL_MIN), fill_value=0)], axis=1).astype(float)
    lg = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    lg.fit(dm(trs), trs.y, logisticregression__sample_weight=_weights(trs.family))
    return lg.predict_proba(dm(te))[:, 1]

def walk_forward_A(X, E, target_of, H_of, label):
    """task A (before onset): origins without a developing silence and not in an outage"""
    starts, role = FOLDS()
    base = X[X.eligible_for_prediction & ~X.in_declared_outage & (X.silence_svc_slots <= 1 + SLOT_GRACE_FRAC)]
    fams = [f for f in NOMINAL_MIN if H_of(f) is not None]
    B = base[base.family.isin(fams)].copy(); B['H'] = B.family.map(H_of).astype(float); B['y'] = np.nan
    for f in fams: B.loc[B.family == f, 'y'] = B.loc[B.family == f, target_of(f)]
    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        tr = B[(B.hour_end + pd.to_timedelta(B.H, unit='h') <= s) & B.y.notna()]
        te = B[(B.hour_end >= s) & (B.hour_end < e)].copy()
        if not len(te) or tr.y.sum() < 10: continue
        trs = tr[tr.hour_end.dt.hour % 2 == 0].copy(); cols = _usable(trs, _feat())
        te['gbm_shared'] = _fit_shared(trs, te, cols)
        te['logit_shared'] = _fit_logit(trs, te, LOGIT_FEATS)
        te['family_rate'] = te.family.map(tr.groupby('family').y.mean()).fillna(tr.y.mean())
        Ev = E[(E.onset < s) & E.device_id.isin(tr.device_id.unique()) & (E.onset >= tr.hour_end.min())]
        te['device_hazard'] = device_hazard(Ev, tr, te, te.H.values)
        te['gbm_family'] = np.nan
        for f in fams:
            tf = tr[tr.family == f]
            if not (te.family == f).any() or tf.y.nunique() < 2: continue
            if tf.device_id.nunique() >= M1_SUPERVISED_MIN['devices'] and episodes_count(tf, 'y', H_of(f)) >= M1_SUPERVISED_MIN['degraded_episodes']:
                tfs = tf[tf.hour_end.dt.hour % 2 == 0]; fc = _usable(tfs, cols)
                fm = HistGradientBoostingClassifier(**GBC).fit(tfs[fc].astype(float), tfs.y)
                te.loc[te.family == f, 'gbm_family'] = fm.predict_proba(te.loc[te.family == f, fc].astype(float))[:, 1]
        te['fold'] = str(s.date()); te['role'] = role[s]; te['task'] = label
        out.append(te[['device_id', 'family', 'hour_end', 'H', 'y', 'fold', 'role', 'task'] + CAND_A])
        print(label, s.date(), len(te), int(tr.y.sum()), flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def walk_forward_B(X, label='B_developing'):
    """task B (early detection): origins inside a developing, not yet declared silence; y = the silence is later declared"""
    starts, role = FOLDS()
    B = X[X.eligible_for_prediction & X.developing_silence & ~X.in_declared_outage & X.gap_status.isin(['declared', 'burst'])].copy()
    B['y'] = (B.gap_status == 'declared').astype(float); B['H'] = 0.0
    B['surprise'] = 1 - B.hmm_silence_surprise
    B['log_silence'] = np.log1p(B.silence_svc_slots)
    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        # a training silence must be resolved (declared or recovered) before the fold start
        res_t = B.gap_declared_at.where(B.gap_status == 'declared', B.gap_recovered_at)
        tr = B[res_t <= s]; te = B[(B.hour_end >= s) & (B.hour_end < e)].copy()
        if not len(te) or tr.y.sum() < 10: continue
        trs = tr.copy(); cols = _usable(trs, _feat() + ['log_silence', 'surprise'])
        te['gbm_shared'] = _fit_shared(trs, te, cols)
        te['logit_shared'] = _fit_logit(trs, te, ['log_silence', 'surprise', 'heard_since_last_stored', 'avail_svc_168h', 'n_declared_28d', 'fleet_bad_share'])
        te['surprise_calibrated'] = np.nan; te['silence_length_rate'] = np.nan
        bins = np.r_[0, np.unique(np.quantile(tr.log_silence, np.linspace(0, 1, 21)))[1:-1], np.inf]
        for f in te.family.unique():
            tf = tr[tr.family == f] if (tr.family == f).sum() >= 200 else tr
            ok = tf.surprise.notna()
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(tf.surprise[ok], tf.y[ok]) if ok.sum() >= 50 else None
            m = te.family == f
            if iso is not None: te.loc[m, 'surprise_calibrated'] = iso.predict(te.loc[m, 'surprise'].fillna(tf.surprise.median()))
            r = tf.groupby(pd.cut(tf.log_silence, bins)).y.mean()
            te.loc[m, 'silence_length_rate'] = pd.cut(te.loc[m, 'log_silence'], bins).map(r).astype(float).fillna(tf.y.mean()).values
        te['fold'] = str(s.date()); te['role'] = role[s]; te['task'] = label
        out.append(te[['device_id', 'family', 'hour_end', 'H', 'y', 'fold', 'role', 'task', 'gap_onset', 'gap_declared_at', 'gap_recovered_at'] + CAND_B])
        print(label, s.date(), len(te), int(tr.y.sum()), flush=True)
    return pd.concat(out, ignore_index=True)

CAND_B = ['silence_length_rate', 'surprise_calibrated', 'logit_shared', 'gbm_shared']

def km(durations, observed):
    """Kaplan-Meier: returns (times, S(t)) step function"""
    o = np.argsort(durations); t = durations[o]; d = observed[o]
    ut = np.unique(t[d]); S = []; s_ = 1.0
    for u in ut:
        n = (t >= u).sum(); k = ((t == u) & d).sum(); s_ *= 1 - k / n; S.append(s_)
    return ut, np.array(S)

def km_eval(ut, S, x):
    if len(ut) == 0: return np.ones_like(x, dtype=float)
    i = np.searchsorted(ut, x, 'right') - 1
    return np.where(i >= 0, S[np.maximum(i, 0)], 1.0)

def walk_forward_D(X, E, obs_end, label='D_continuation'):
    """task D: origins inside a declared outage; y_Delta = still unavailable at h + Delta (censored if unresolved by obs end)"""
    starts, role = FOLDS()
    D = X[X.in_declared_outage & X.gap_onset.notna()].copy()
    D['elapsed_h'] = (D.hour_end - D.gap_onset).dt.total_seconds() / 3600
    oe = D.family.map(obs_end).astype('datetime64[ns]')
    out = []
    for Dh in CONTINUATION_H:
        y = np.where(D.gap_recovered_at.notna() & (D.gap_recovered_at <= D.hour_end + pd.Timedelta(hours=Dh)), 0.0,
                     np.where(D.gap_recovered_at.notna() | (D.hour_end + pd.Timedelta(hours=Dh) <= oe), 1.0, np.nan))
        Dd = D.assign(y=y, H=float(Dh))
        for s, e in zip(starts[:-1], starts[1:]):
            te = Dd[(Dd.hour_end >= s) & (Dd.hour_end < e)].copy()
            Ev = E[E.onset < s].copy()
            if not len(te) or len(Ev) < 20: continue
            end = Ev.recovered_at.where(Ev.recovered_at <= s, s)
            Ev['dur'] = (end - Ev.onset).dt.total_seconds() / 3600; Ev['obs'] = Ev.recovered_at.notna() & (Ev.recovered_at <= s)
            def cond(sub, x):
                ut, S = km(sub.dur.values, sub.obs.values.astype(bool))
                a = km_eval(ut, S, x); b = km_eval(ut, S, x + Dh); return np.where(a > 0, b / np.maximum(a, 1e-12), 1.0)
            te['km_pooled'] = cond(Ev, te.elapsed_h.values)
            te['km_family'] = te.km_pooled.values; te['km_family_scope'] = te.km_pooled.values
            for f, idx in te.groupby('family').indices.items():
                ef = Ev[Ev.family == f]
                if len(ef) >= 10: te.iloc[idx, te.columns.get_loc('km_family')] = cond(ef, te.elapsed_h.values[idx])
                for sc, j in te.iloc[idx].groupby('gap_scope').indices.items():
                    es = ef[ef.scope_at_declaration == sc]
                    ii = idx[j]
                    te.iloc[ii, te.columns.get_loc('km_family_scope')] = cond(es, te.elapsed_h.values[ii]) if len(es) >= 10 else te.km_family.values[ii]
            # training origins for a supervised model: outage hours whose target closed before s
            tr = Dd[(Dd.hour_end + pd.Timedelta(hours=Dh) <= s) & Dd.y.notna() & (Dd.hour_end < s)]
            if tr.y.nunique() == 2 and len(tr) >= 500:
                trs = tr.sample(min(len(tr), 60000), random_state=0).copy()
                trs['log_elapsed'] = np.log1p(trs.elapsed_h); te['log_elapsed'] = np.log1p(te.elapsed_h)
                trs['scope_code'] = trs.gap_scope.map({'individual': 0, 'gateway': 1, 'fleet': 2, 'network': 3}); te['scope_code'] = te.gap_scope.map({'individual': 0, 'gateway': 1, 'fleet': 2, 'network': 3})
                cols = _usable(trs, ['log_elapsed', 'scope_code', 'heard_since_last_stored', 'avail_svc_672h', 'n_declared_28d', 'fleet_bad_share', 'rx_minus_svc_24h', 'history_days', 'last_outage_h'])
                te['gbm_shared'] = _fit_shared(trs, te, cols)
            else:
                te['gbm_shared'] = np.nan
            te['fold'] = str(s.date()); te['role'] = role[s]; te['task'] = f'{label}_{Dh}h'
            out.append(te[['device_id', 'family', 'hour_end', 'H', 'y', 'fold', 'role', 'task', 'elapsed_h', 'gap_scope', 'gap_onset'] + CAND_D])
            print(label, Dh, s.date(), len(te), flush=True)
    return pd.concat(out, ignore_index=True)

CAND_D = ['km_pooled', 'km_family', 'km_family_scope', 'gbm_shared']

def detection_delay(E):
    """task C: time from onset (first missing report) to declaration, by family and declaration rule"""
    d = E.copy(); d['delay_h'] = (d.declared_at - d.onset).dt.total_seconds() / 3600
    return d.groupby('family').delay_h.describe(percentiles=[.5, .9])[['count', '50%', '90%', 'max']].rename(columns={'50%': 'median_h', '90%': 'p90_h'})

def chronic_rates(X, E, as_of):
    """task E: device onset rate per week, gamma-Poisson posterior (prior: 1 week at the family rate), 90% interval"""
    from scipy.stats import gamma
    ex = X[X.eligible_for_prediction & ~X.in_declared_outage].groupby('device_id').size().astype(float)
    k = E[E.onset <= as_of].groupby('device_id').size()
    fam = X.groupby('device_id').family.first()
    exf = ex.groupby(fam.reindex(ex.index)).sum(); lam_f = (k.groupby(fam.reindex(k.index)).sum().reindex(exf.index, fill_value=0) / exf).to_dict()
    rows = []
    for dev in fam.index:
        a0 = 168 * lam_f.get(fam[dev], 0) + 1e-3; b0 = 168.0; kk = k.get(dev, 0); e_ = ex.get(dev, 0.0)
        a, b = a0 + kk, b0 + e_
        rows.append(dict(device_id=dev, family=fam[dev], onsets=int(kk), exposure_h=e_, rate_per_week=168 * a / b,
                         rate_lo90=168 * gamma.ppf(0.05, a, scale=1 / b), rate_hi90=168 * gamma.ppf(0.95, a, scale=1 / b), family_rate_per_week=168 * lam_f.get(fam[dev], 0)))
    return pd.DataFrame(rows)

def time_pattern(X, E):
    """hour-of-day pattern of onsets vs exposure (eligible, not in outage): chi-square GOF per family"""
    rows = []
    ex = X[X.eligible_for_prediction & ~X.in_declared_outage]
    for f, d in E.groupby('family'):
        if len(d) < 30: rows.append(dict(family=f, onsets=len(d), p_value=np.nan, elevated_hours='insufficient events (< 30 onsets)')); continue
        exp_h = ex[ex.family == f].hour_end.dt.hour.value_counts().reindex(range(24), fill_value=0)
        obs = d.onset.dt.hour.value_counts().reindex(range(24), fill_value=0)
        expc = exp_h / exp_h.sum() * obs.sum(); m = expc > 0
        stat = (((obs - expc) ** 2) / expc.where(m, 1))[m].sum(); p = chi2.sf(stat, m.sum() - 1)
        ratio = obs / expc.replace(0, np.nan)
        hi = [int(h) for h in ratio.index if ratio[h] >= 1.5 and obs[h] >= 5]
        rows.append(dict(family=f, onsets=len(d), p_value=p, elevated_hours=(','.join(f'{h:02d}' for h in hi) or 'none >= 1.5x') if p < 0.01 else 'no significant pattern'))
    return pd.DataFrame(rows)
CAND_A = ['family_rate', 'device_hazard', 'logit_shared', 'gbm_shared', 'gbm_family']

def scores(P, cands):
    rows = []
    for (task, fam, H, role), d in P[P.y.notna()].groupby(['task', 'family', 'H', 'role']):
        for c in cands:
            p = d[c].values.astype(float); ok = ~np.isnan(p); y = d.y.values[ok]; p = np.clip(p[ok], 1e-6, 1 - 1e-6)
            r = dict(task=task, family=fam, H=H, role=role, candidate=c, n=int(ok.sum()), positives=int(y.sum()))
            if ok.sum() and 0 < y.sum() < len(y):
                r.update(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p), logloss=log_loss(y, p), base_rate=y.mean())
            rows.append(r)
    return pd.DataFrame(rows)

def choose(S, P, cands, decision_time, default='device_hazard'):
    """per task x family x H: lowest validation log loss (n >= 200, >= 10 positives, full coverage); else best pooled; else default"""
    V = P[(P.role == 'validation') & P.y.notna() & (P.hour_end + pd.to_timedelta(P.H, unit='h') <= pd.Timestamp(decision_time))]
    choice = {}
    pooled = {}
    for task, d in V.groupby('task'):
        sc = {c: log_loss(d.y, np.clip(d[c].astype(float), 1e-6, 1 - 1e-6), labels=[0, 1]) for c in cands if d[c].notna().all() and 0 < d.y.sum() < len(d)}
        if sc: pooled[task] = min(sc, key=sc.get)
    for (task, fam, H) in P[['task', 'family', 'H']].drop_duplicates().itertuples(index=False):
        v = V[(V.task == task) & (V.family == fam) & (V.H == H)]
        sc = {}
        if len(v) >= 200 and v.y.sum() >= 10:
            sc = {c: log_loss(v.y, np.clip(v[c].astype(float), 1e-6, 1 - 1e-6), labels=[0, 1]) for c in cands if v[c].notna().mean() >= 0.99}
        choice[(task, fam, H)] = (min(sc, key=sc.get), 'validation') if sc else ((pooled[task], 'pooled_validation') if task in pooled else (default, 'default'))
    return choice

def alert_threshold(v, col):
    """threshold maximising F1 on validation origins (window level)"""
    y = v.y.values; p = v[col].values.astype(float); best = (0, 0.5)
    for t in np.unique(np.quantile(p, np.linspace(0.5, 0.995, 60))):
        a = p >= t; tp = (a & (y == 1)).sum(); f1 = 2 * tp / max(a.sum() + y.sum(), 1)
        if f1 > best[0]: best = (f1, t)
    return best[1]

def event_metrics(d, col, thr, E, H):
    """unique-event and alert-episode metrics on one family x H test set.
    event caught: an alert at an origin in [onset - H, onset); lead time = onset - first such alert.
    alert episode: consecutive alerting hours of a device; false if no onset within H after the episode start."""
    H = int(H); a = d[d[col] >= thr].sort_values(['device_id', 'hour_end'])
    ev = E[E.device_id.isin(d.device_id.unique()) & (E.onset >= d.hour_end.min()) & (E.onset <= d.hour_end.max())]
    caught, leads = 0, []
    ad = {k: g.hour_end.values for k, g in a.groupby('device_id')}
    for r in ev.itertuples():
        t = ad.get(r.device_id, np.array([], 'datetime64[ns]')); on = np.datetime64(r.onset)
        w = t[(t >= on - np.timedelta64(H, 'h')) & (t < on)]
        if len(w): caught += 1; leads.append((on - w.min()).astype('timedelta64[m]').astype(float) / 60)
    ep_false = 0; ep = 0
    for dev, g in a.groupby('device_id'):
        hh = g.hour_end.values; st = hh[np.r_[True, np.diff(hh) > np.timedelta64(1, 'h')]]
        on = np.sort(ev[ev.device_id == dev].onset.values.astype('datetime64[ns]'))
        for s0 in st:
            ep += 1; k = np.searchsorted(on, s0, 'left')
            if k >= len(on) or on[k] > s0 + np.timedelta64(H, 'h'): ep_false += 1
    sensor_weeks = d.groupby('device_id').size().sum() / 168
    return dict(events=len(ev), events_caught=caught, event_recall=caught / len(ev) if len(ev) else np.nan,
                median_lead_h=float(np.median(leads)) if leads else np.nan, alert_episodes=ep, false_episodes=ep_false,
                false_alerts_per_sensor_week=ep_false / sensor_weeks if sensor_weeks else np.nan)

def early_detection_events(PB, cand, thr):
    """task B at the level of unique silences (test): declared silences alarmed before declaration, how much earlier than the
    rule (lead over declaration) and how late after onset; recovered silences that alarmed = false alarms."""
    t = PB[(PB.role == 'test')].copy(); t['alarm'] = t[cand] >= thr
    g = t.groupby(['device_id', 'gap_onset'])
    rows = []
    for (dev, on), d in g:
        dec = d.y.iat[0] == 1; a = d[d.alarm]
        first = a.hour_end.min() if len(a) else pd.NaT
        rows.append(dict(device_id=dev, family=d.family.iat[0], declared=dec, alarmed=len(a) > 0,
                         lead_over_declaration_h=(d.gap_declared_at.iat[0] - first).total_seconds() / 3600 if (dec and len(a)) else np.nan,
                         delay_from_onset_h=(first - on).total_seconds() / 3600 if len(a) else np.nan))
    R = pd.DataFrame(rows)
    if not len(R): return pd.DataFrame()
    R['decl_alarmed'] = R.declared & R.alarmed; R['rec_alarmed'] = ~R.declared & R.alarmed
    sw = t.groupby('device_id').size().sum() / 168
    out = R.groupby('family').agg(silences=('declared', 'size'), declared=('declared', 'sum'), declared_alarmed=('decl_alarmed', 'sum'),
                                  recovered_alarmed=('rec_alarmed', 'sum'), median_lead_over_declaration_h=('lead_over_declaration_h', 'median'),
                                  median_delay_from_onset_h=('delay_from_onset_h', 'median')).reset_index()
    out['false_alarms_per_sensor_week'] = out.recovered_alarmed / sw
    return out
