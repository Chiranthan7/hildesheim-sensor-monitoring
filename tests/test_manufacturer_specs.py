"""Tests for the manufacturer-informed rules (config SOURCES / SPEC_* / RADIO_* / BATTERY_OBSERVABILITY).
run:  FOUNDATION=/path/to/foundation pytest -q tests/test_manufacturer_specs.py"""
import os, re
import numpy as np, pandas as pd, pytest
from hsm import config as C

FD = os.environ.get('FOUNDATION', '/home/claude/foundation')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rd = lambda name: pd.read_parquet(f'{FD}/{name}.parquet')

@pytest.fixture(scope='module')
def soil(): return rd('silver_soil')
@pytest.fixture(scope='module')
def dev(): return rd('silver_devices')
@pytest.fixture(scope='module')
def wx(): return rd('silver_weather')

# ---------- documentation ----------
def test_every_external_constant_is_sourced_and_documented():
    doc = open(f'{ROOT}/docs/MANUFACTURER_SPECS.md').read()
    for name, c in C.C.items():
        assert set(c['sources']) <= set(C.SOURCES), name
        if c['provenance'] == 'external':
            assert c['sources'] and c['verification'] in {'verified', 'sibling_model', 'general_family', 'unverified'}, name
        if c['sources']:
            assert name in doc, f'{name} not documented'
    for sid, (title, url) in C.SOURCES.items():
        assert re.match(r'^(https://|file://)', url), sid
        assert sid in doc, f'source {sid} not listed in docs'

def test_manifest_records_sources():
    import json
    m = json.load(open(f'{FD}/manifest.json'))
    assert set(m['sources']) == set(C.SOURCES)
    assert all('sources' in v and 'verification' in v for v in m['constants'].values())

# ---------- record-level validity ----------
def test_soil_temp_validity_uses_manufacturer_range(soil):
    rng = C.SPEC_SOIL_TEMP_C; phys = C.PLAUSIBLE['soil_temp_physical_c']
    lo = soil.family.map({k: v[0] for k, v in rng.items()}).fillna(phys[0])
    hi = soil.family.map({k: v[1] for k, v in rng.items()}).fillna(phys[1])
    t = soil.soil_temp_c
    exp_oos = t.notna() & ((t < lo) | (t > hi))
    assert (soil.flag_soil_temp_out_of_spec == exp_oos).all()
    assert not (soil.flag_soil_temp_out_of_spec & soil.flag_soil_temp_exposure_suspected).any()
    assert (soil.flag_soil_temp_exposure_suspected == (t.notna() & ~exp_oos & ~t.between(*phys))).all()
    assert (soil.soil_temp_range_source == np.where(soil.family.isin(list(rng)), 'manufacturer', 'physical')).all()

def test_uplink_record_validity_follows_record_flags(soil):
    L = rd('gold_uplink_records'); L = L[L.source == 'soil'].set_index('raw_row_id')
    s = soil.set_index('raw_row_id').loc[L.index]
    exp = s.data_stored & ~(s.flag_soil_temp_out_of_spec | s.flag_rawfreq_fault | s.flag_temp_stuck_sofar)
    assert (L.valid.values == exp.values).all()
    # an exposure reading alone never invalidates a record
    only_exp = s.flag_soil_temp_exposure_suspected & s.data_stored & ~s.flag_rawfreq_fault & ~s.flag_temp_stuck_sofar
    assert L.valid[only_exp.values].all()

# ---------- field-level clean columns (raw kept unchanged) ----------
def test_clover_fields(soil, dev):
    cl = soil[soil.family == 'CloverSoilSensor']
    assert (cl.flag_gwc_censored_dry == ((cl.gwc == 0) & ~cl.flag_rawfreq_fault)).all()
    keep = ~(cl.flag_gwc_censored_dry | cl.flag_rawfreq_fault)
    assert cl.gwc_clean[~keep].isna().all()
    assert ((cl.gwc_clean[keep] == cl.gwc[keep]) | cl.gwc[keep].isna()).all()
    assert not (soil[soil.family != 'CloverSoilSensor'].flag_gwc_censored_dry).any()
    rd_ = dev.battery_remaining_days_raw
    assert (dev.flag_remaining_days_implausible == (rd_ > C.CLOVER_RATED_LIFE_DAYS)).all()
    assert dev.battery_remaining_days_clean[dev.flag_remaining_days_implausible].isna().all()
    ok = ~dev.flag_remaining_days_implausible & rd_.notna()
    assert (dev.battery_remaining_days_clean[ok] == rd_[ok]).all()
    assert dev.battery_remaining_days_clean.max() <= C.CLOVER_RATED_LIFE_DAYS

def test_dragino_ec(soil):
    g = soil.flag_ec_suspected_unscaled
    assert (soil.family[g] == 'Dragino-SE0X-LB').all()
    assert ((soil.ec[g] >= 1) & (soil.ec[g] % 1 == 0)).all()
    assert soil.ec_us_cm[g | soil.flag_ec_implausible].isna().all()
    dr = soil[(soil.family == 'Dragino-SE0X-LB') & ~g & ~soil.flag_ec_implausible & soil.ec.notna()]
    assert np.allclose(dr.ec_us_cm, dr.ec * C.DRAGINO_EC_STORAGE['to_us_cm'])
    assert soil.ec_us_cm[soil.family != 'Dragino-SE0X-LB'].isna().all()

def test_wind_direction_masked_only_in_calm(wx):
    mw = wx[wx.family == 'MeteoWind']
    calm = mw.wind_ms <= C.WIND_CALM_MS
    assert (mw.flag_wind_calm == calm).all()
    assert mw.wind_dir_clean[calm].isna().all()
    moving = ~calm & mw.wind_dir.notna()
    assert (mw.wind_dir_clean[moving] == mw.wind_dir[moving]).all()
    # calibration offset: any non-zero speed is at least ~0.36 m/s
    assert mw.wind_ms[mw.wind_ms > 0].min() >= 0.35
    assert not wx[wx.family != 'MeteoWind'].flag_wind_calm.any()

def test_sentinel_temperature(dev):
    assert dev.device_temp_c_clean[dev.flag_temp_sentinel].isna().all()
    ok = ~dev.flag_temp_sentinel & dev.device_temp_c.notna()
    assert (dev.device_temp_c_clean[ok] == dev.device_temp_c[ok]).all()

# ---------- radio ----------
def test_radio_flags(dev):
    U = rd('gold_uplinks'); R = C.RADIO_LIMITS
    assert (U.flag_snr_below_sf12_reference_floor == (U.snr_max < R['snr_floor_sf12_db'])).all()
    assert (U.flag_rssi_below_sf12_reference_sensitivity == (U.rssi_max < R['sensitivity_sf12_dbm'])).all()
    assert (U.flag_beyond_sf12_reference_limits == (U.flag_snr_below_sf12_reference_floor | U.flag_rssi_below_sf12_reference_sensitivity)).all()
    assert (U.flag_rssi_near_saturation == (U.rssi_max >= R['saturation_dbm'])).all()
    nr = U.family.isin(list(C.RADIO_METRICS))
    assert (U.radio_metrics_available == ~nr).all()
    assert U.loc[nr, ['rssi', 'snr']].isna().all().all()          # declared radio-less families really carry no radio metrics
    assert (dev.flag_snr_below_sf12_reference_floor == (dev.snr < R['snr_floor_sf12_db'])).all()

# ---------- reporting modes / battery ----------
def test_mode_meaning_only_for_documented_modes():
    H = rd('gold_device_hour')
    exp = pd.Series(list(zip(H.family, H.svc_mode_min))).map(C.MODE_MEANING)
    assert ((H.svc_mode_meaning.values == exp.values) | (pd.isna(H.svc_mode_meaning.values) & exp.isna().values)).all()
    assert set(H.svc_mode_meaning.dropna()) <= set(C.MODE_MEANING.values())

def test_battery_observability_covers_every_family(dev):
    fams = set(C.NOMINAL_MIN)
    assert set(C.BATTERY_OBSERVABILITY) == fams
    for fam, b in C.BATTERY_OBSERVABILITY.items():
        sig = b['signal']
        g = dev[dev.family == fam]
        if sig is None:
            assert g.battery_v.isna().all() and g.battery_level_pct.isna().all(), fam
        else:
            assert g[sig].notna().any(), fam
