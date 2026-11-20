"""Which information is informative, per family: grouped permutation importance of the shared model and (where trained) the
family model, fitted on data before the test start and evaluated on test origins of each family (analysis only)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .models import COMMON, RADIO, FAMILY_SPECIFIC, GBM, horizon_feats, family_features, _usable, _weights, tau, episodes
from ..config import NOMINAL_MIN, M1_HORIZONS_H, M1_SUPERVISED_MIN, M1_FOLDS
from .state import primary_h

GROUPS = {
    'recent_service_availability': ['avail_svc_1h', 'avail_svc_6h', 'avail_svc_24h', 'avail_svc_72h'],
    'chronic_availability': ['avail_svc_168h', 'avail_svc_672h', 'bern_p', 'n28'],
    'current_silence': ['silence_svc_slots', 'silence_rx_slots', 'miss_run_slots', 'last_outcome'],
    'individual_state_models': ['hmm_p_bad', 'hmm_p_next', 'hmm_silence_surprise', 'mk_pDD', 'mk_pMD', 'eb_pDD', 'eb_pMD'],
    'individual_forecasts': ['hmm_E', 'mk_E', 'eb_E', 'bern_E'],
    'reception_vs_storage': ['avail_rx_24h', 'avail_rx_168h', 'backend_suspect_now', 'rx_minus_svc_24h', 'excess_24h'],
    'data_validity_and_mode': ['avail_valid_24h', 'avail_mode_24h', 'reduced_mode'],
    'outage_history': ['n_declared_28d', 'h_since_recovery', 'last_outage_h', 'history_days'],
    'fleet_and_network': ['fleet_bad_share', 'network_level'],
    'radio': RADIO,
    'family_specific_measurement': FAMILY_SPECIFIC,
    'calendar': ['hod_sin', 'hod_cos', 'dow'],
    'family_identity': ['fam_code', 'nominal_min']}

def _group_cols(g, cols, H):
    out = []
    for c in GROUPS[g]:
        out += [x for x in cols if x == c or (g == 'individual_forecasts' and x == f'{c}{H}')]
    return out

def perm_importance(model, Xte, yte, cols, H, rng, reps=3):
    base = np.sqrt(np.mean((np.clip(model.predict(Xte[cols].astype(float)), 0, 1) - yte) ** 2)); res = {}
    for g in GROUPS:
        gc = _group_cols(g, cols, H)
        if not gc: continue
        d = []
        for _ in range(reps):
            Xp = Xte[cols].astype(float).copy(); idx = rng.permutation(len(Xp))
            Xp[gc] = Xp[gc].values[idx]                                    # permute the whole group jointly
            d.append(np.sqrt(np.mean((np.clip(model.predict(Xp), 0, 1) - yte) ** 2)) - base)
        res[g] = float(np.mean(d))
    return base, res

def run(X, max_rows=20000, seed=0):
    rng = np.random.default_rng(seed); s = pd.Timestamp(M1_FOLDS['test'][0]); rows = []
    code = {f: i for i, f in enumerate(sorted(NOMINAL_MIN))}
    X = X.copy(); X['fam_code'] = X.family.map(code); X['nominal_min'] = X.family.map(NOMINAL_MIN)
    for H in sorted({primary_h(f) for f in M1_HORIZONS_H} | {6}):
        fams = [f for f, hs in M1_HORIZONS_H.items() if H in hs]
        D = X[X.family.isin(fams) & X.eligible_for_prediction & ~X.in_declared_outage & X[f'y{H}'].notna()]
        tr = D[D.hour_end + pd.Timedelta(hours=H) <= s]; trs = tr[tr.hour_end.dt.hour % 3 == 0]; te = D[D.hour_end >= s]
        cols = _usable(trs, COMMON + horizon_feats(H) + RADIO) + _usable(trs, FAMILY_SPECIFIC) + ['fam_code', 'nominal_min']
        sh = HistGradientBoostingRegressor(**GBM, categorical_features=[cols.index('fam_code')]).fit(trs[cols].astype(float), trs[f'y{H}'], sample_weight=_weights(trs.family))
        for fam in fams:
            if primary_h(fam) != H and not (H == 6 and fam in ('Sensoterra', 'MeteoHelix', 'MeteoWind')): continue
            tf = te[te.family == fam]
            if len(tf) < 200: continue
            tf = tf.sample(min(len(tf), max_rows), random_state=seed)
            b, imp = perm_importance(sh, tf, tf[f'y{H}'].values, cols, H, rng)
            rows += [dict(family=fam, H=H, model='gbm_shared', base_rmse=b, group=g, rmse_increase=v) for g, v in imp.items()]
            trf = tr[tr.family == fam]; thr = tau(trf, H, fam)
            if trf.device_id.nunique() >= M1_SUPERVISED_MIN['devices'] and episodes(trf, H, thr) >= M1_SUPERVISED_MIN['degraded_episodes']:
                tfs = trf[trf.hour_end.dt.hour % 3 == 0]; ff = _usable(tfs, family_features(fam, H))
                fm = HistGradientBoostingRegressor(**GBM).fit(tfs[ff].astype(float), tfs[f'y{H}'])
                b, imp = perm_importance(fm, tf, tf[f'y{H}'].values, ff, H, rng)
                rows += [dict(family=fam, H=H, model='gbm_family', base_rmse=b, group=g, rmse_increase=v) for g, v in imp.items()]
    return pd.DataFrame(rows)
