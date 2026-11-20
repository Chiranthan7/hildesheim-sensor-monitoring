"""Module 1 tests: mathematics, causality, purging, state logic, statistics.
run:  FOUNDATION=<full foundation with M1 outputs> RAW=<raw dir> pytest -q tests/test_m1.py"""
import os, sys, subprocess, tempfile
import numpy as np, pandas as pd, pytest
from hsm.m1 import analytic, models, state
from hsm import config as C

FD = os.environ.get('FOUNDATION', '/home/claude/foundation'); RAW = os.environ.get('RAW', '/mnt/user-data/uploads')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda n, d=FD: pd.read_parquet(f'{d}/{n}.parquet')

# ---------- mathematics ----------
def test_sum_of_matrix_powers_closed_form():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.uniform(0.001, 0.999, 2); A = np.array([[1 - a, a], [b, 1 - b]]); n = int(rng.integers(1, 300))
        ref = sum(np.linalg.matrix_power(A, h) for h in range(1, n + 1))
        assert np.allclose(analytic.sum_powers(A, n), ref, atol=1e-8)

def test_markov_forecast_matches_explicit_chain():
    """add_forecasts closed form == explicit (1/n) e_last sum_h P^h [1,0]'"""
    X = rd('gold_m1_features'); r = X[X.family == 'MeteoHelix'].dropna(subset=['mk_E6', 'last_outcome']).sample(200, random_state=0)
    n = 6 * 60 // C.NOMINAL_MIN['MeteoHelix']
    for row in r.itertuples():
        if row.reduced_mode: continue
        P = np.array([[row.mk_pDD, 1 - row.mk_pDD], [row.mk_pMD, 1 - row.mk_pMD]])
        e = np.array([1.0, 0.0]) if row.last_outcome == 1 else np.array([0.0, 1.0])
        ref = e @ sum(np.linalg.matrix_power(P, h) for h in range(1, n + 1)) @ np.array([1.0, 0.0]) / n
        assert abs(ref - row.mk_E6) < 1e-9

def test_hmm_forecast_matches_filter_state():
    """hmm_E = (1/n) alpha sum_h A^h d with alpha = (1 - p_bad, p_bad) and the fit in force at the hour end"""
    X = rd('gold_m1_features'); F = rd('gold_hmm_fits')
    r = X[(X.family == 'MeteoWind') & X.hmm_available & ~X.reduced_mode].sample(100, random_state=0)
    for row in r.itertuples():
        f = F[(F.device_id == row.device_id) & (F.valid_from <= row.hour_end)].sort_values('valid_from').iloc[-1]
        A = np.array([[f.p_stay_good, 1 - f.p_stay_good], [1 - f.p_stay_bad, f.p_stay_bad]]); d = np.array([1 - f.good_miss, 1 - f.bad_miss])
        n = 24 * 60 // 10; al = np.array([1 - row.hmm_p_bad, row.hmm_p_bad])
        ref = al @ sum(np.linalg.matrix_power(A, h) for h in range(1, n + 1)) @ d / n
        assert abs(ref - row.hmm_E24) < 1e-8 and abs(al @ A @ d - row.hmm_p_next) < 1e-10

def test_probabilities_and_counts_are_valid():
    X = rd('gold_m1_features')
    for c in [c for c in X.columns if c.startswith(('hmm_E', 'mk_E', 'eb_E', 'bern_E')) or c in ('hmm_p_bad', 'hmm_p_next', 'hmm_silence_surprise')]:
        v = X[c].dropna(); assert ((v >= -1e-12) & (v <= 1 + 1e-12)).all(), c
    has = X.n28.notna()                                     # devices with a service ledger
    assert X.loc[has, ['d28', 'n_DD', 'n_DM', 'n_MD', 'n_MM']].notna().all().all()
    for c in ['n28', 'd28', 'n_DD', 'n_DM', 'n_MD', 'n_MM']: assert (X.loc[has, c] >= 0).all(), c
    assert (X.d28[has] <= X.n28[has]).all()
    assert (X.n_DD + X.n_DM + X.n_MD + X.n_MM <= X.n28 + 1e-9)[has].all()

def test_targets_recompute():
    X = rd('gold_m1_features'); H = rd('gold_device_hour').set_index(['device_id', 'hour_end']).sort_index()
    r = X.dropna(subset=['y24']).sample(300, random_state=0)
    for row in r.itertuples():
        w = H.loc[row.device_id].loc[row.hour_end + pd.Timedelta('1h'): row.hour_end + pd.Timedelta('24h')]
        assert len(w) == 24 and w.hour_complete.all()
        assert abs(w.svc_nominal_filled.sum() / w.svc_nominal_expected.sum() - row.y24) < 1e-12

# ---------- no leakage ----------
def test_features_exclude_outcomes_and_hindsight():
    feats = set(models.COMMON + models.RADIO + models.FAMILY_SPECIFIC + [f for H in (1, 6, 24, 72) for f in models.horizon_feats(H)])
    for f in feats:
        assert not f.startswith('y') and 'retro' not in f and f not in ('in_declared_outage', 'declared_scope', 'eligible_for_prediction'), f

def test_training_windows_are_purged():
    I = rd('gold_m1_fold_info')
    assert (pd.to_datetime(I.train_outcome_end_max) <= pd.to_datetime(I.fold)).all()

def test_model_choice_ignores_unclosed_outcomes():
    P = rd('gold_m1_predictions'); d = pd.Timestamp('2026-09-01')
    c1, u1 = models.choose(P, d)
    Q = P.copy(); late = Q.hour_end + pd.to_timedelta(Q.H, unit='h') > d
    Q.loc[late, 'y'] = np.random.default_rng(0).uniform(0, 1, late.sum())      # scramble everything not yet observable
    c2, u2 = models.choose(Q, d)
    assert c1 == c2 and u1 == u2

# ---------- state logic and statistics ----------
def test_state_rules():
    T = rd('gold_m1_state')
    assert set(T.state) <= set(state.LIGHT) and (T.traffic_light == T.state.map(state.LIGHT)).all()
    ins = T.state == 'insufficient_information'
    assert (ins | T.hour_complete).all()
    m = T.in_declared_outage & T.hour_complete & ~T.backend_suspect_now & T.service_started
    assert (T.state[m] == 'significant_interruption').all()
    b = T.state == 'backend_data_unavailable'; assert (T.backend_suspect_now | ~T.service_started)[b].all()
    dg = T.state == 'degraded'
    assert ((T.forecast_avail < T.tau) | (T.developing_silence & (T.hmm_silence_surprise < 0.01)))[dg].all()
    nm = T.state == 'normal'; assert ~(T.forecast_avail < T.tau)[nm].any() and ~T.reduced_mode[nm].any()

def test_period_statistics_reconcile():
    H = rd('gold_device_hour'); D = rd('gold_m1_period_daily'); W = rd('gold_m1_period_weekly')
    for S in (D, W):
        for c in ['service_availability', 'valid_data_availability', 'reception_availability', 'mode_availability']:
            v = S[c].dropna(); assert ((v >= 0) & (v <= 1)).all(), c
        assert S.expected.sum() == H.svc_nominal_expected.sum() and S.delivered.sum() == H.svc_nominal_filled.sum()
        assert (S.valid <= S.delivered).all()

# ---------- causality of the whole module (truncation) ----------
SUBSET = ['ac1f09fffe0e2411', 'ac1f09fffe0e18aa', '008000000407afe1', '0080000004082992', '647fda000001c711', 'a84041ae3e5c3250',
          '500300550', 'anl_450923', 'a84041b9706025c6', '0080000004082982', '008000000407afb0', 'ac1f09fffe0e1d5b', '0004a30b01105535']
CUT = '2026-09-20'

@pytest.fixture(scope='module')
def m1_runs():
    base = tempfile.mkdtemp(); out = {}
    for name, cut in [('full', None), ('cut', CUT)]:
        d = f'{base}/{name}'; cmd = [sys.executable, f'{ROOT}/run_foundation.py', '--raw', RAW, '--out', d, '--devices', ','.join(SUBSET)]
        if cut: cmd += ['--cutoff', cut]
        subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)
        subprocess.run([sys.executable, f'{ROOT}/run_m1.py', '--foundation', d, '--no-importance'], check=True, capture_output=True, cwd=ROOT)
        out[name] = d
    return out

def test_m1_truncation_invariance(m1_runs):
    T0 = pd.Timestamp(CUT)
    Xf = rd('gold_m1_features', m1_runs['full']); Xc = rd('gold_m1_features', m1_runs['cut'])
    cols = [c for c in Xc.columns if not c.startswith('y') and c not in ('device_id', 'hour_end', 'family')]
    m = Xf[Xf.hour_end < T0].merge(Xc[Xc.hour_end < T0], on=['device_id', 'hour_end'], suffixes=('_f', '_c'))
    assert len(m) == (Xc.hour_end < T0).sum() > 0
    for c in cols:
        a, b = m[c + '_f'], m[c + '_c']
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)) or a.dtype == bool or b.dtype == bool:
            same = (a.astype(object) == b.astype(object)) | (a.isna() & b.isna())
        else:
            same = (a == b) | (a.isna() & b.isna()) | (a.astype(float) - b.astype(float)).abs().lt(1e-9)
        assert same.all(), f'{c}: {int((~same).sum())} rows differ'
    Pf = rd('gold_m1_predictions', m1_runs['full']); Pc = rd('gold_m1_predictions', m1_runs['cut'])
    k = ['device_id', 'hour_end', 'H']
    mp = Pf[Pf.hour_end < T0].merge(Pc[Pc.hour_end < T0], on=k, suffixes=('_f', '_c'))
    assert len(mp) == (Pc.hour_end < T0).sum() > 0
    for c in models.CANDIDATES:
        a, b = mp[c + '_f'], mp[c + '_c']; assert ((a - b).abs().lt(1e-9) | (a.isna() & b.isna())).all(), c
    Tf = rd('gold_m1_state', m1_runs['full']); Tc = rd('gold_m1_state', m1_runs['cut'])
    ms = Tf[Tf.hour_end < T0].merge(Tc[Tc.hour_end < T0], on=['device_id', 'hour_end'], suffixes=('_f', '_c'))
    assert (ms.state_f == ms.state_c).all() and (ms.forecast_model_f.fillna('') == ms.forecast_model_c.fillna('')).all()

def test_m1_reproducible(m1_runs):
    d = m1_runs['full']; e = tempfile.mkdtemp()
    import shutil; shutil.copytree(d, e, dirs_exist_ok=True)
    subprocess.run([sys.executable, f'{ROOT}/run_m1.py', '--foundation', e, '--no-importance'], check=True, capture_output=True, cwd=ROOT)
    for n in ['gold_m1_features', 'gold_m1_predictions', 'gold_m1_state', 'gold_m1_device_summary']:
        pd.testing.assert_frame_equal(rd(n, d), rd(n, e))
