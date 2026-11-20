"""Foundation tests. Full-run checks use FOUNDATION (built by run_foundation.py); causality and reproducibility tests rebuild a
subset of devices into a temporary directory (with and without a cutoff) and compare.
run:  FOUNDATION=/path/to/foundation RAW=/path/to/raw pytest -q tests"""
import os, json, hashlib, subprocess, sys, tempfile
import numpy as np, pandas as pd, pytest

FD = os.environ.get('FOUNDATION', '/home/claude/foundation'); RAW = os.environ.get('RAW', '/mnt/user-data/uploads')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET = ['ac1f09fffe0e2411', 'ac1f09fffe0e18aa', '008000000407afe1', '0080000004082992', '647fda000001c711', 'a84041ae3e5c3250',
          '500300550', 'anl_450923', 'a84041b9706025c6', '0080000004082982', '008000000407afb0']
CUT = '2026-08-25'
rd = lambda name, d=FD: pd.read_parquet(f'{d}/{name}.parquet')

def test_bronze_keeps_every_raw_row():
    m = json.load(open(f'{FD}/manifest.json'))
    for src in ['devices', 'soil', 'weather']:
        assert m[src]['bronze_rows'] == m[src]['raw_rows'], src

def test_silver_rows_trace_to_bronze():
    for src in ['devices', 'soil', 'weather']:
        b = set(rd(f'bronze/bronze_{src}').raw_row_id); s = rd(f'silver_{src}')
        assert len(s) == len(b) and set(s.raw_row_id) == b, src

def test_every_record_belongs_to_one_uplink():
    links = rd('gold_uplink_records'); U = rd('gold_uplinks')
    assert links.raw_row_id.is_unique
    assert set(links.uplink_id) == set(U.uplink_id)
    assert U.n_records.sum() == len(links)

def test_availability_bounded_and_consistent():
    H = rd('gold_device_hour')
    for c in ['service_availability', 'valid_data_availability', 'mode_availability', 'reception_availability']:
        v = H[c].dropna(); assert ((v >= 0) & (v <= 1)).all(), c
    assert (H.svc_nominal_filled <= H.svc_nominal_expected).all()
    assert (H.svc_nominal_valid_filled <= H.svc_nominal_filled).all()
    assert (H.rx_nominal_filled <= H.rx_nominal_expected).all()

def test_ledger_reconciles_with_arrivals_and_hours():
    U = rd('gold_uplinks'); L = rd('gold_silences'); H = rd('gold_device_hour')
    d = U[U.data_stored]
    arrivals = d.groupby('device_id').size(); excess = d.groupby('device_id').svc_excess.sum()
    closed = L[~L.open].groupby('device_id').size()
    # every non-excess stored arrival except the first fills exactly one slot
    chk = (arrivals - excess - 1).reindex(closed.index)
    assert (chk == closed).all()
    # ledger totals == hourly totals (hours cover the whole service period)
    led = L.groupby('device_id')[['svc_filled', 'svc_missed']].sum().sum(axis=1)
    hrs = H.groupby('device_id').svc_nominal_expected.sum().reindex(led.index)
    assert (np.abs(led - hrs) <= 1).all()          # allow the boundary slot at the final partial hour

def test_declarations_only_use_their_own_time():
    L = rd('gold_silences'); dec = L[L.status == 'declared']
    assert (dec.declared_at > dec.anchor_t).all()
    rec = dec.dropna(subset=['recovered_at'])
    assert (rec.recovered_at > rec.declared_at).all()           # a declared silence lasted beyond its declaration time
    burst = L[L.status == 'burst']
    assert (burst.next_t <= burst.declared_at).all()

def test_retrospective_annotations_never_in_device_hour():
    H = rd('gold_device_hour')
    assert not any(c.startswith('retro') for c in H.columns)

def _build(out, cutoff=None):
    cmd = [sys.executable, f'{ROOT}/run_foundation.py', '--raw', RAW, '--out', out, '--devices', ','.join(SUBSET)]
    if cutoff: cmd += ['--cutoff', cutoff]
    subprocess.run(cmd, check=True, capture_output=True, cwd=ROOT)

@pytest.fixture(scope='module')
def subset_runs():
    base = tempfile.mkdtemp(); a, b, c = f'{base}/full', f'{base}/cut', f'{base}/full2'
    _build(a); _build(b, CUT); _build(c)
    return a, b, c

def test_causality_truncation(subset_runs):
    """outputs for times before the cutoff must be identical whether or not later data exists"""
    full, cut, _ = subset_runs; T = pd.Timestamp(CUT)
    Hf = rd('gold_device_hour', full); Hc = rd('gold_device_hour', cut)
    Hf = Hf[Hf.hour_end <= T.floor('h')]; Hc = Hc[Hc.hour_end <= T.floor('h')]
    cols = ['svc_nominal_expected', 'svc_nominal_filled', 'svc_mode_expected', 'svc_mode_filled', 'svc_nominal_valid_filled', 'svc_excess_reports',
            'rx_nominal_expected', 'rx_nominal_filled', 'svc_silence_min', 'svc_mode_min', 'in_declared_outage', 'developing_silence',
            'eligible_for_prediction', 'network_level', 'svc_mode_meaning']
    m = Hf.merge(Hc, on=['device_id', 'hour_end'], suffixes=('_f', '_c'))
    assert len(m) == len(Hc) > 0
    for c in cols:
        x, y = m[c + '_f'], m[c + '_c']
        same = (x == y) | (x.isna() & y.isna())
        assert same.all(), f'{c}: {int((~same).sum())} rows differ'
    Ff = rd('gold_hmm_fits', full); Fc = rd('gold_hmm_fits', cut)
    Ff = Ff[Ff.valid_from <= T].sort_values(['device_id', 'valid_from']).reset_index(drop=True)
    Fc = Fc.sort_values(['device_id', 'valid_from']).reset_index(drop=True)
    pd.testing.assert_frame_equal(Ff[['device_id', 'valid_from', 'kstar_slots']], Fc[['device_id', 'valid_from', 'kstar_slots']])
    Lf = rd('gold_silences', full); Lc = rd('gold_silences', cut)
    df = Lf[(Lf.status == 'declared') & (Lf.declared_at <= T)].set_index('gap_id'); dc = Lc[(Lc.status == 'declared') & (Lc.declared_at <= T)].set_index('gap_id')
    assert set(df.index) == set(dc.index)
    assert (df.loc[dc.index, 'scope_at_declaration'] == dc.scope_at_declaration).all()
    Uf = rd('gold_uplinks', full); Uc = rd('gold_uplinks', cut)
    m = Uf.merge(Uc, on=['device_id', 't'], suffixes=('_f', '_c'))
    assert (m.svc_excess_f == m.svc_excess_c).all() and ((m.svc_mode_min_f == m.svc_mode_min_c) | m.svc_mode_min_f.isna()).all()

def test_reproducible(subset_runs):
    a, _, c = subset_runs
    for name in ['gold_device_hour', 'gold_silences', 'gold_uplinks', 'gold_hmm_fits']:
        pd.testing.assert_frame_equal(rd(name, a), rd(name, c))

# ---------- v3 fixes: merge timing, data availability time, channel validity, reception-based mode ----------
def test_uplink_merge_is_span_anchored():
    from hsm.config import MERGE_WINDOW_S
    U = rd('gold_uplinks')
    assert ((U.t_last_record - U.t).dt.total_seconds() <= MERGE_WINDOW_S).all()

def test_service_time_is_data_availability_time():
    from hsm.config import MERGE_WINDOW_S
    U = rd('gold_uplinks'); L = rd('gold_uplink_records')
    assert (U.data_stored == U.t_data.notna()).all()
    s = U[U.data_stored]
    assert (s.t_data >= s.t).all() and ((s.t_data - s.t).dt.total_seconds() <= MERGE_WINDOW_S).all()
    # an uplink counts as stored data only if a stored record with measurement content exists
    has = set(L[L.is_data_record].uplink_id)
    assert set(s.uplink_id) == has
    assert not (U.stored_without_measurement & U.data_stored).any()

def test_channel_validity_is_strict_and_causal():
    U = rd('gold_uplinks').sort_values(['device_id', 't'])
    assert (U.data_valid <= U.data_valid_any).all()
    v = U[U.data_valid]; assert (v.n_channels_valid == v.n_channels_expected).all()
    assert (U.n_channels_stored <= U.n_channels_expected).all()
    assert (U.groupby('device_id').n_channels_seen_before.diff().fillna(0) >= 0).all()

def test_service_mode_comes_from_reception():
    U = rd('gold_uplinks')
    for dev, x in U.groupby('device_id'):
        s = x[x.data_stored]
        if not len(s): continue
        k = np.searchsorted(x.t.values, s.t_data.values, side='right') - 1
        assert (s.svc_mode_min.values == x.rx_mode_min.values[k]).all(), dev
