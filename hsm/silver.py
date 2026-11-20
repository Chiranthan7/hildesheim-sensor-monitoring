"""Silver layer: typed and normalised records with anomaly flags. Nothing is deleted:
duplicates are marked (dup_rank > 0), error codes keep a flag next to the cleaned value, raw text stays in bronze (join by raw_row_id).
Two levels of validity:
  record level  (flag_* used by uplinks.data_valid): the whole packet is corrupt / beyond the manufacturer range / stuck.
  field level   (*_clean columns): one field is unusable (sentinel, censored, undefined, glitch); the raw column is kept unchanged."""
import numpy as np, pandas as pd, duckdb
from .config import (PLAUSIBLE, TEMP_STUCK_READINGS, SPEC_SOIL_TEMP_C, CLOVER_RATED_LIFE_DAYS, WIND_CALM_MS, RADIO_LIMITS,
                     DRAGINO_EC_STORAGE)

DRAGINO = 'a84041b9706025c6'

def _num(s): return pd.to_numeric(s.replace({'': None, '--': None}), errors='coerce')

def _ids(entity_id, device_eui=None):
    e = entity_id.fillna('')
    fam = e.str.split(':').str[3]
    last = e.str.replace(r':channel-\d$', '', regex=True).str.split(':').str[-1].str.lower()
    dev = last if device_eui is None else device_eui.fillna(last).str.lower()
    chan = e.str.extract(r':channel-(\d)$')[0].astype(float)
    dev = dev.where(~dev.str.startswith('channel-'), DRAGINO)
    fam = fam.where(~((dev == DRAGINO) | (fam == DRAGINO)), 'Dragino-SE0X-LB')
    variant = np.where(e.str.contains('[A-F]', regex=True) & ~e.str.contains('OpenWeather'), 'upper_eui', 'lower_eui')
    return dev, fam, chan, variant

def _xy(loc):
    xy = loc.fillna('').str.extract(r'POINT \(([-0-9.]+) ([-0-9.]+)\)').astype(float)
    lon, lat = xy[0], xy[1]
    zero = (lon == 0) & (lat == 0)
    return lat.where(~zero), lon.where(~zero), zero

def _dup_rank(df, keys, value_cols):
    """0 = first occurrence; >0 = exact duplicate of an earlier record (same keys and values); -1 = same keys, different values"""
    g = df.groupby(keys, sort=False, dropna=False)              # dropna=False: empty channel is a valid key
    rank = g.cumcount()
    nval = df.groupby(keys, dropna=False)[value_cols].transform('nunique').max(axis=1) if value_cols else 0
    return np.where((rank > 0) & (nval > 1), -1, rank), g[keys[0]].transform('size')

def devices(bronze_dir):
    d = pd.read_parquet(f'{bronze_dir}/bronze_devices.parquet')
    dev, fam, chan, variant = _ids(d.entity_id)
    s = pd.DataFrame({'raw_row_id': d.raw_row_id, 'source': 'devices', 'device_id': dev, 'family': fam, 'entity_variant': variant, 't': d.t,
                      'rssi': _num(d.rssi), 'snr': _num(d.snr), 'battery_raw': _num(d.batteryvoltage), 'battery_level_raw': _num(d.batterylevel),
                      'battery_remaining_days_raw': _num(d.batteryremainingdays), 'device_temp_c': _num(d.temperature),
                      'devicestate': d.devicestate, 'devicemodel': d.devicemodel})
    s['loc_lat'], s['loc_lon'], s['flag_loc_zero'] = _xy(d.location)
    s['loc_role'] = np.where(s.family == 'Bigbelly', 'device_position', np.where(s.loc_lat.notna(), 'receiving_gateway', None))
    s['loc_decimals'] = d.location.fillna('').str.extract(r'POINT \([-0-9]+\.([0-9]+)')[0].str.len()
    s['flag_rssi_sentinel'] = s.rssi > PLAUSIBLE['rssi_max_dbm']
    s['flag_temp_sentinel'] = s.device_temp_c == PLAUSIBLE['adeunis_temp_sentinel']
    s['device_temp_c_clean'] = s.device_temp_c.where(~s.flag_temp_sentinel)               # -99 = no valid temperature (Adeunis)
    R = RADIO_LIMITS; r = s.rssi.where(~s.flag_rssi_sentinel)
    s['flag_snr_below_sf12_reference_floor'] = s.snr < R['snr_floor_sf12_db']
    s['flag_rssi_below_sf12_reference_sensitivity'] = r < R['sensitivity_sf12_dbm']
    s['flag_rssi_near_saturation'] = r >= R['saturation_dbm']
    s['has_measurement'] = s.battery_raw.notna() | s.device_temp_c.notna()      # data content of device-table families (Ott, Bigbelly, Adeunis)
    s['meas_channel'] = 'main'
    # battery normalisation: volts; level in percent; error codes flagged
    mv = s.family.isin(['Bigbelly', 'Agvolution'])
    s['battery_v'] = np.where(mv, s.battery_raw / 1000, s.battery_raw)
    lo, hi = PLAUSIBLE['lithium_cell_v']
    s['flag_battery_error_code'] = (s.family == 'Agvolution') & ((s.battery_v > hi) | (s.battery_v < lo))
    s.loc[s.flag_battery_error_code, 'battery_v'] = np.nan
    s['battery_level_pct'] = np.where(s.family == 'CloverSoilSensor', s.battery_level_raw * 100, np.nan)
    s['flag_level_is_voltage_ratio'] = (s.family == 'Bigbelly') & s.battery_level_raw.notna()      # = V/12*100, kept raw only
    s['flag_remaining_days_implausible'] = s.battery_remaining_days_raw > CLOVER_RATED_LIFE_DAYS     # beyond rated 11-year life
    s['battery_remaining_days_clean'] = s.battery_remaining_days_raw.where(~s.flag_remaining_days_implausible)
    s['dup_rank'], s['dup_group_size'] = _dup_rank(s, ['device_id', 't'], ['rssi', 'snr', 'battery_raw', 'device_temp_c', 'loc_lat'])
    return s

import json, re
NGSI_FIELDS = {'soilMoistureVwc': 'rec_vwc', 'soilTemperature': 'rec_soil_temp', 'fieldCapacity': 'rec_field_capacity', 'wiltingPoint': 'rec_wilting_point',
               'usableFieldCapacity': 'rec_usable_fc', 'saturationPointOvershot': 'rec_saturation', 'soilMoistureEc': 'rec_ec'}

def parse_failed_insert(js):
    """embedded NGSI entity of a record the backend failed to store: error details + the original measured values"""
    out = {'ingestion_failed': False, 'error_class': None, 'error_detail': None, 'failed_batch_id': None}
    if not isinstance(js, str) or not js: return out
    try: o = json.loads(js)
    except Exception: out.update(ingestion_failed=True, error_class='unparseable'); return out
    err = o.get('error')
    if err:
        out['ingestion_failed'] = True
        out['error_class'] = re.split(r'[(]', err)[0]
        m = re.search(r"'M': '([^']+)'", err); out['error_detail'] = m.group(1) if m else err[:200]
    out['failed_batch_id'] = o.get('failedBatchID')
    data = o.get('data', {})
    for k, col in NGSI_FIELDS.items():
        v = data.get(k)
        out[col] = v.get('value') if isinstance(v, dict) else None
    return out

def soil(bronze_dir):
    d = pd.read_parquet(f'{bronze_dir}/bronze_soil.parquet')
    dev, fam, chan, variant = _ids(d.entity_id, d.device_eui.where(~d.device_eui.str.startswith('channel-', na=False)))
    n = lambda c: _num(d[c])
    s = pd.DataFrame({'raw_row_id': d.raw_row_id, 'source': 'soil', 'device_id': dev, 'family': fam, 'channel': chan, 'entity_variant': variant, 't': d.t,
                      'vwc_raw': n('soilmoisturevwc'), 'gwc': n('soilmoisturegwc'), 'rawfreq': n('soilmoisturerawfreq'),
                      'saturation_overshot': d.saturationpointovershot.map({'true': True, 'false': False}),
                      'soil_temp_raw': n('soiltemperature'), 'vwc10': n('soilmoisturevwc10cm'), 'vwc30': n('soilmoisturevwc30cm'), 'vwc45': n('soilmoisturevwc45cm'),
                      'soil_temp10': n('soiltemperature10cm'), 'soil_temp30': n('soiltemperature30cm'), 'soil_temp45': n('soiltemperature45cm'),
                      'air_temp_raw': n('airtemperature'), 'cap10': n('soilcapacitance10cm'), 'cap30': n('soilcapacitance30cm'), 'cap45': n('soilcapacitance45cm'),
                      'ec': n('soilmoistureec'), 'field_capacity': n('fieldcapacity'), 'wilting_point': n('wiltingpoint'), 'usable_fc': n('usablefieldcapacity'),
                      'copied_rssi': n('rssi'), 'copied_battery_v': n('batteryvoltage'), 'copied_device_timestamp': pd.to_datetime(d.device_timestamp, errors='coerce'),
                      'ingestion_path': np.where(d.__original_ngsi_entity__.notna(), 'embedded_ngsi', 'standard')})
    s['sensor_lat'], s['sensor_lon'], _ = _xy(d.location)
    fi = pd.DataFrame([parse_failed_insert(x) for x in d['__original_ngsi_entity__'].values], index=d.index)
    for c in fi.columns: s[c] = fi[c].values
    for c in ['rec_vwc', 'rec_soil_temp', 'rec_field_capacity', 'rec_wilting_point', 'rec_usable_fc', 'rec_ec']:
        if c in s: s[c] = pd.to_numeric(s[c], errors='coerce')
    s['data_stored'] = ~s.ingestion_failed                          # backend failed to store this record (values recoverable in rec_*)
    s['vwc_pct'] = np.where(s.family == 'Dragino-SE0X-LB', s.vwc_raw * 100, s.vwc_raw)          # Dragino reports a fraction
    s['vwc_pct'] = s.vwc_pct.fillna(s.vwc10)
    s['soil_temp_c'] = s.soil_temp_raw.fillna(s.soil_temp10)
    P = PLAUSIBLE
    s['flag_saturated'] = (s.family == 'Sensoterra') & (s.saturation_overshot == True)
    s['flag_zero_moisture'] = ((s.family == 'Sensoterra') & (s.vwc_raw == 0)) | ((s.family == 'CloverSoilSensor') & (s.gwc == 0))
    s['flag_rawfreq_fault'] = (s.family == 'CloverSoilSensor') & (s.rawfreq > 5000)
    lo = s.family.map({k: v[0] for k, v in SPEC_SOIL_TEMP_C.items()}).fillna(P['soil_temp_physical_c'][0])
    hi = s.family.map({k: v[1] for k, v in SPEC_SOIL_TEMP_C.items()}).fillna(P['soil_temp_physical_c'][1])
    s['soil_temp_range_source'] = np.where(s.family.isin(list(SPEC_SOIL_TEMP_C)), 'manufacturer', 'physical')
    s['flag_soil_temp_out_of_spec'] = s.soil_temp_c.notna() & ((s.soil_temp_c < lo) | (s.soil_temp_c > hi))     # record invalid
    s['flag_soil_temp_exposure_suspected'] = (s.soil_temp_c.notna() & ~s.flag_soil_temp_out_of_spec
                                    & ~s.soil_temp_c.between(*P['soil_temp_physical_c']))                      # valid reading, probe exposed?
    s['flag_air_temp_implausible'] = ~s.air_temp_raw.between(*P['agvolution_air_temp_c']) & s.air_temp_raw.notna()
    s['flag_ec_implausible'] = (s.ec > P['ec'][1])
    s['flag_ec_suspected_unscaled'] = (s.family == 'Dragino-SE0X-LB') & (s.ec >= 1) & (s.ec % 1 == 0)          # whole numbers: unscaled value
    s['ec_us_cm'] = np.where(s.family == 'Dragino-SE0X-LB', s.ec * DRAGINO_EC_STORAGE['to_us_cm'], np.nan)
    s.loc[s.flag_ec_suspected_unscaled | s.flag_ec_implausible, 'ec_us_cm'] = np.nan
    # CLOVER: GWC is a table conversion of a frequency; 0 = beyond the dry end of the table (censored, not a measured zero)
    s['flag_gwc_censored_dry'] = (s.family == 'CloverSoilSensor') & (s.gwc == 0) & ~s.flag_rawfreq_fault
    s['gwc_clean'] = s.gwc.where(~(s.flag_gwc_censored_dry | s.flag_rawfreq_fault))
    s['flag_copied_radio_fields'] = s.copied_rssi.notna()          # Dragino rows carry a stale copy of device metadata
    # measurement content and measurement channel: a stored record without any measured value delivers no data
    meas = ['vwc_raw', 'gwc', 'rawfreq', 'soil_temp_raw', 'ec', 'vwc10', 'vwc30', 'vwc45', 'soil_temp10', 'soil_temp30', 'soil_temp45',
            'air_temp_raw', 'cap10', 'cap30', 'cap45']
    s['has_measurement'] = s[meas].notna().any(axis=1)
    prof = s[['vwc10', 'vwc30', 'vwc45', 'soil_temp10', 'soil_temp30', 'soil_temp45']].notna().any(axis=1)
    s['meas_channel'] = np.select([s.family == 'Dragino-SE0X-LB', (s.family == 'Agvolution') & prof, s.family == 'Agvolution'],
                                  ['ch' + s.channel.fillna(0).astype(int).astype(str), 'profile', 'cap_air'], 'main')
    s.loc[~s.has_measurement, 'meas_channel'] = None
    s = s.sort_values(['device_id', 'channel', 't'], na_position='first').reset_index(drop=True)
    ch = s.groupby(['device_id', 'channel'], dropna=False).soil_temp_c.transform(lambda v: v.ne(v.shift()).cumsum())
    s['temp_run_len_sofar'] = s.groupby(['device_id', 'channel', ch], dropna=False).cumcount() + 1
    s['flag_temp_stuck_sofar'] = s.temp_run_len_sofar >= TEMP_STUCK_READINGS
    s['dup_rank'], s['dup_group_size'] = _dup_rank(s, ['device_id', 'channel', 't'], ['vwc_raw', 'gwc', 'soil_temp_c', 'ec', 'vwc10', 'air_temp_raw'])
    # stuck-run count must not double count duplicates: recompute on first occurrences only, carry to duplicates
    first = s.dup_rank == 0
    f = s[first]; ch2 = f.groupby(['device_id', 'channel'], dropna=False).soil_temp_c.transform(lambda v: v.ne(v.shift()).cumsum())
    s.loc[first, 'temp_run_len_sofar'] = (f.groupby(['device_id', 'channel', ch2], dropna=False).cumcount() + 1).values
    s['temp_run_len_sofar'] = s.groupby(['device_id', 'channel', 't'], dropna=False).temp_run_len_sofar.transform('first')
    s['flag_temp_stuck_sofar'] = s.temp_run_len_sofar >= TEMP_STUCK_READINGS
    return s

def weather(bronze_dir):
    d = pd.read_parquet(f'{bronze_dir}/bronze_weather.parquet')
    fam = d.entity_id.str.split(':').str[3]
    n = lambda c: _num(d[c])
    s = pd.DataFrame({'raw_row_id': d.raw_row_id, 'source': 'weather', 'device_id': d.device_eui.str.lower(), 'family': fam,
                      'entity_variant': np.where(d.entity_id.str.contains('[A-F]', regex=True) & (fam != 'OpenWeatherMap'), 'upper_eui', 'lower_eui'),
                      't': d.t, 'name': d['name'], 'temp_c': n('temperature'), 'rh_raw': n('relativehumidity'), 'pressure_hpa': n('atmosphericpressure'),
                      'rain_mm': n('precipitation'), 'solar_wm2': n('solarradiation'), 'solar_max': n('maximumsolarradiation'), 'solar_min': n('minimumsolarradiation'),
                      'wind_ms': n('windspeed'), 'gust_ms': n('gustspeed'), 'wind_dir': n('winddirection'), 'feels_like_c': n('feelsliketemperature'),
                      'air_temp_tsa_json': d.airtemperaturetsa})
    s['sensor_lat'], s['sensor_lon'], s['flag_loc_zero'] = _xy(d.location)
    s['rh_pct'] = np.where(s.family == 'MeteoHelix', s.rh_raw * 100, s.rh_raw)
    s['flag_wind_calm'] = (s.family == 'MeteoWind') & (s.wind_ms <= WIND_CALM_MS)
    s['wind_dir_clean'] = s.wind_dir.where(~s.flag_wind_calm)                                  # direction undefined when cups do not turn
    s['has_measurement'] = s[['temp_c', 'rh_raw', 'pressure_hpa', 'rain_mm', 'solar_wm2', 'wind_ms', 'gust_ms', 'wind_dir']].notna().any(axis=1)
    s['meas_channel'] = np.where(s.has_measurement, 'main', None)
    P = PLAUSIBLE
    s['flag_temp_implausible'] = s.temp_c.notna() & ~s.temp_c.between(*P['air_temp_c'])
    s['flag_pressure_implausible'] = s.pressure_hpa.notna() & ~s.pressure_hpa.between(*P['pressure_hpa'])
    s['flag_rain_implausible'] = s.rain_mm > P['rain_mm_per_report'][1]
    s['flag_solar_sentinel'] = s.solar_wm2.isin(P['solar_sentinels'])
    s['flag_bad_packet'] = (s.family == 'MeteoHelix') & (s.flag_temp_implausible | s.flag_pressure_implausible | s.flag_rain_implausible | s.flag_solar_sentinel)
    s['dup_rank'], s['dup_group_size'] = _dup_rank(s, ['device_id', 't'], ['temp_c', 'wind_ms'])
    return s

def gateways_official(bronze_dir):
    g = pd.read_parquet(f'{bronze_dir}/bronze_gateways.parquet')
    g['lat'] = g.lat.astype(float); g['lon'] = g['long'].astype(float)
    return g[['raw_row_id', 'name', 'lat', 'lon']]
