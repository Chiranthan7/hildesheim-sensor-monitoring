"""Every constant used by the system, with provenance and permitted use.

provenance:
  external      supplied with the challenge (gateway list) or manufacturer documentation
  declared      operator configuration / reporting schedule the service is expected to deliver
  physical      physical plausibility limits, sensor ADC limits
  operational   a declared DEFINITION (not a finding); results are conditional on it; sensitivity analysis required
  structural    property of the export itself, known at export time (e.g. last timestamp in a file)
  exploratory   chosen after looking at this dataset during exploration (degree of freedom; must be sensitivity-tested)
  retrospective hindsight annotation; NEVER a model input; evaluation/descriptive use only
allowed_use: features | labels | evaluation | descriptive
"""

C = {}
def const(name, value, provenance, allowed_use, note='', sources=(), verification=''):
    """sources: ids in SOURCES (required for provenance 'external');
    verification: verified | sibling_model | general_family | unverified (how directly the source covers OUR device)"""
    for s in sources:
        assert s in SOURCES, f'{name}: unknown source {s}'
    if provenance == 'external':
        assert sources, f'{name}: external constant needs a source'
    C[name] = dict(value=value, provenance=provenance, allowed_use=allowed_use, note=note, sources=list(sources), verification=verification)
    return value

# ---- manufacturer / reference documents (accessed 2026-11-20). Full discussion: docs/MANUFACTURER_SPECS.md ----
SOURCES = {
    'CHALLENGE_GATEWAY_LIST': ('Gateway list supplied with the challenge (d06ac58c-Gateway_Standorte.csv)', 'file://d06ac58c-Gateway_Standorte.csv'),
    'SENSOTERRA_MD_TECHSHEET': ('Sensoterra Multi Depth Tech Sheet v2.1', 'https://res.cloudinary.com/dctlrnwuz/image/upload/v1670848753/Sensoterra_MD_Tech_Sheet_v2.1_EN_yczu5f.pdf'),
    'SENSOTERRA_MD_DATASHEET': ('Sensoterra Multi Depth datasheet (iot-shop.de)', 'https://iot-shop.de/shop/sensoterra-multi-depth-lorawan-bodenfeuchtesensor-4829/document/5466'),
    'SENSOTERRA_SD_TECHSHEET': ('Sensoterra Single Depth Gen2.5 Tech Sheet', 'https://cdn.shopify.com/s/files/1/0447/5633/6807/files/Sensor_-_Sensoterra_SD_Gen2.5_Tech_Sheet_EN.pdf'),
    'SENSOTERRA_FAQ_DAILY': ('Sensoterra support: Why am I not getting hourly data from my sensor?', 'https://servicedesk-ci.freshdesk.com/en/support/solutions/articles/80001186574-why-am-i-not-getting-hourly-data-from-my-sensor-'),
    'SENSOTERRA_SD_SHOP': ('Sensoterra Single Depth product page (iot-shop.de)', 'https://iot-shop.de/en/shop/sensoterra-single-depth-bodenfeuchtesensor-4828'),
    'TEKTELIC_CLOVER_SPEC': ('TEKTELIC CLOVER Agriculture Sensor Spec Sheet (Mar 2026)', 'https://tektelic.com/wp-content/uploads/TEKTELIC_CLOVER-Agriculture_Sensor_Spec-Sheet_Mar-2026.pdf'),
    'TEKTELIC_CLOVER_QSG': ('TEKTELIC CLOVER Quick Start Guide (default reporting periods)', 'https://knowledgehub.tektelic.com/hubfs/Anuradha%20Kala%20QSG%20JULY%204/Clover%20Quick%20Start%20Guide.pdf'),
    'TEKTELIC_FAQ': ('TEKTELIC CLOVER/KIWI FAQ (frequency to GWC conversion)', 'https://knowledgehub.tektelic.com/clover-kiwi-faq'),
    'TEKTELIC_UG': ('TEKTELIC KIWI/CLOVER User Guide', 'https://tektelic.com/wp-content/uploads/TEKTELIC_KIW_UG.pdf'),
    'DRAGINO_SE0X_MANUAL': ('Dragino SE0X-LB user manual', 'https://wiki.dragino.com/xwiki/bin/view/Main/User%20Manual%20for%20LoRaWAN%20End%20Nodes/SE0X-LB--LoRaWAN_Soil_Moisture%26EC_Sensor_User_Manual'),
    'DRAGINO_LSE01_LORA_ALLIANCE': ('Dragino LSE01 LoRaWAN EU Supplementary Device Info Questionnaire (sibling model)', 'https://lora-alliance.org/wp-content/uploads/2021/11/LSE01-LoRaWAN-EU-Supplementary-Device-Info-Questionnaire-V1_0.pdf'),
    'BARANI_HELIX_DATASHEET': ('BARANI MeteoHelix IoT Pro DataSheet (2024)', 'https://barani.squarespace.com/s/MeteoHelix-IoT-Pro-DataSheet.pdf'),
    'BARANI_HELIX_DECODER': ('BARANI MeteoHelix message format / decoder', 'https://barani.squarespace.com/meteohelix-message-decoder'),
    'BARANI_HELIX_SHOP': ('MeteoHelix IoT Pro product page (iot-shop.de, accuracy)', 'https://iot-shop.de/en/shop/barani-meteohelix-iot-pro-wetterstation-5010'),
    'BARANI_WIND_DATASHEET': ('BARANI MeteoWind IoT Pro DataSheet (2024)', 'https://barani.squarespace.com/s/MeteoWind-IoT-Pro-DataSheet.pdf'),
    'BARANI_WIND_COMPACT': ('BARANI MeteoWind Compact DataSheet (calibration equation)', 'https://barani.squarespace.com/s/MeteoWind-Compact-DataSheet.pdf'),
    'ADEUNIS_TEMP_V3_UG': ('Adeunis TEMP V3 LoRaWAN user guide', 'https://www.manualslib.fr/manual/278915/Adeunis-Temp-V3-Lorawan-863.html?page=38'),
    'TTN_ADEUNIS_TEMP': ('The Things Network device repository: Adeunis Temp', 'https://www.thethingsnetwork.org/device-repository/devices/adeunis/temp/'),
    'TAGO_ADEUNIS_TEMP': ('TagoIO: Adeunis Temperature (alerts)', 'https://tago.io/devices/sigfox/adeunis-temperature.md'),
    'BIGBELLY_HC5': ('Bigbelly HC5 station data sheet (cellular connection)', 'https://www.procurement.iastate.edu/sites/default/files/Documents/Bigbelly%20Station%20Data%20Sheets%201901%20-%20HC5%20and%20SC55%20(1).pdf'),
    'BIGBELLY_SPEC_18AH': ('Bigbelly Smart Compost spec sheet (12 V, 18 Ah SLA, 40 W panel)', 'https://bigbelly.com/wp-content/uploads/2025/11/SmartCompostProductSpecSheet2507_WEB.pdf'),
    'BIGBELLY_SPEC_28AH': ('Bigbelly Smart Max product sheet (28 Ah SLA)', 'https://bigbelly.com/wp-content/uploads/2023/06/Smart-Max-Product-Sheet-2301.pdf'),
    'OTT_HYDROSYSTEMS': ('OTT HydroSystems (solar supply, cellular/satellite)', 'https://ott.com/products/system-solutions-23/ott-hydrosystems-384'),
    'OTT_ECOLOG1000': ('OTT ecoLog 1000 technical data (3.6 V lithium pack)', 'https://www.ott.com/products/water-level-1/ott-ecolog-1000-water-level-logger-2455/productAction/outputAsPdf'),
    'AGVOLUTION_CLIMAVI': ('Agvolution climavi brochure (iot-shop.de)', 'https://iot-shop.de/shop/agvolution-climavi-wetterstation-6801/document/4269'),
    'SEMTECH_LORA_FAQ': ('Semtech LoRa FAQ (SNR floor, saturation)', 'https://www.semtech.com/design-support/faq/faq-lora/P0'),
    'LORA_SNR_TABLE': ('Mobilefish LoRa tutorial 16: SNR limit per spreading factor', 'https://www.mobilefish.com/download/lora/lora_part16.pdf'),
    'LORA_SENS_TABLE': ('Blaszczyszyn et al., Analyzing LoRa ... stochastic geometry, arXiv:1811.01886 (sensitivity per SF, 125 kHz)', 'https://arxiv.org/pdf/1811.01886'),
    'OUTBACK_MX60_SOC': ('OutBack Power MX60 charge controller manual: 12 V lead-acid voltage vs state of charge (charged >12.6, ~50% 12.0, ~25% 11.7, discharged <11.4)', 'https://auto.manualsonline.com/manuals/mfg/outback_power/mx60_1.html?p=85'),
    'SAMLEX_STANDING_V': ('Samlex PST-300-12 manual, Table 7.4: state of charge vs standing voltage, 12 V lead-acid', 'https://www.manualslib.fr/manual/409057/Samplexpower-Pst-300-12.html?page=20'),
    'DYNAMIS_SLA_MANUAL': ('Dynamis sealed lead-acid technical manual (open-circuit voltage needs hours to days to stabilise)', 'https://manualmachine.com/dynamis/leadline/21505470-technical-manual'),
    'SX1276_DATASHEET': ('Semtech SX1276/77/78/79 datasheet', 'https://cdn.sparkfun.com/assets/7/7/3/2/2/SX1276_Datasheet.pdf'),
}

# ---- reporting schedules (service expectation) ----
NOMINAL_MIN = const('NOMINAL_MIN', {
    'MeteoHelix': 10, 'MeteoWind': 10, 'CloverSoilSensor': 15, 'Dragino-SE0X-LB': 20, 'Sensoterra': 60,
    'OttSensor': 60, 'Agvolution': 240, 'Bigbelly': 360, 'Adeunis': 10},
    'declared', 'features|labels|evaluation',
    'Service schedule the sensor is expected to deliver. Manufacturer defaults confirmed for MeteoHelix (10 min message format), '
    'CLOVER (15 min), Dragino SE0X-LB (20 min) and Sensoterra (24 uploads/day). Ott, Agvolution, Bigbelly, Adeunis: configured '
    'schedule evident in every device of the family (no manufacturer default found / configurable).',
    sources=['BARANI_HELIX_DECODER', 'TEKTELIC_CLOVER_QSG', 'DRAGINO_SE0X_MANUAL', 'SENSOTERRA_SD_TECHSHEET'], verification='verified')
ALT_MODES_MIN = const('ALT_MODES_MIN', {'Sensoterra': [1440]}, 'external', 'features|labels|evaluation',
                      'Reduced reporting mode of Sensoterra probes (24 h). First found in the data (coincides with vwc = 0); the manufacturer '
                      'documents once-a-day sending while in soil as poor pin contact or too-dry soil. Mode entry/exit is inferred causally.',
                      sources=['SENSOTERRA_FAQ_DAILY', 'SENSOTERRA_SD_TECHSHEET'], verification='verified')
MODE_MEANING = const('MODE_MEANING', {('Sensoterra', 1440): 'reduced_daily_mode'}, 'external', 'features|descriptive',
                     'reporting mode with a manufacturer-documented interpretation. Sensoterra once-a-day while in soil: documented possible causes '
                     'poor pin contact or too-dry soil (the manufacturer does not state other causes are excluded). Drives the recommendation '
                     '(check / re-seat the probe), never a failure label. The mode is inferred from RECEPTION (what the device sends).',
                     sources=['SENSOTERRA_FAQ_DAILY'], verification='verified')
MODE_ENTER_CONSECUTIVE = const('MODE_ENTER_CONSECUTIVE', 2, 'operational', 'features|labels|evaluation',
                               'consecutive completed intervals within tolerance of an alternative mode needed to switch mode')
MODE_TOLERANCE_FRAC = const('MODE_TOLERANCE_FRAC', 1 / 24, 'operational', 'features|labels|evaluation', '+-1 h on a 24 h mode')
MODE_EXIT_FACTOR = const('MODE_EXIT_FACTOR', 1.5, 'operational', 'features|labels|evaluation',
                         'an interval <= 1.5 x nominal returns the device to its nominal mode')

# ---- where each family's measurement DATA is stored (service availability is judged on this table) ----
DATA_SOURCE = const('DATA_SOURCE', {'Sensoterra': 'soil', 'CloverSoilSensor': 'soil', 'Agvolution': 'soil', 'Dragino-SE0X-LB': 'soil',
                                    'MeteoHelix': 'weather', 'MeteoWind': 'weather', 'OttSensor': 'devices', 'Bigbelly': 'devices', 'Adeunis': 'devices'},
                    'structural', 'features|labels|evaluation', 'measurement table of each family in the exports')

# ---- record assembly / slot accounting ----
MERGE_WINDOW_S = const('MERGE_WINDOW_S', 60, 'exploratory', 'features|labels|evaluation',
                       'records of the same device within 60 s are one uplink (cross-file offsets measured <= 0.3 s; schedules >= 10 min)')
SLOT_GRACE_FRAC = const('SLOT_GRACE_FRAC', 0.5, 'operational', 'features|labels|evaluation',
                        'an arrival less than 0.5 interval after the previous filled slot does not fill a new slot (counted as excess)')

# ---- outage declaration (definitions) ----
ALPHA = const('ALPHA', 1e-3, 'operational', 'labels|evaluation', 'silence declared outage when P(run >= k) < ALPHA under the causal model')
FLOOR_H = const('FLOOR_H', {'default': 24, 'Agvolution': 72, 'Bigbelly': 48}, 'operational', 'labels|evaluation',
                'absolute floor: silence >= this length is an outage regardless of model; for a mode interval m the floor is max(floor, 2m)')
HMM_WINDOW_D = const('HMM_WINDOW_D', 28, 'operational', 'features|labels', 'trailing window for causal HMM2 fits')
HMM_REFIT_D = const('HMM_REFIT_D', 7, 'operational', 'features|labels', 'refit cadence')
MIN_HISTORY_D = const('MIN_HISTORY_D', 14, 'operational', 'features|labels|evaluation',
                      'history needed before model-based thresholds / predictions; earlier only the floor applies')
DAILY_SLOT_PRIOR = const('DAILY_SLOT_PRIOR', (1.0, 1.0), 'operational', 'labels', 'Beta prior for loss of reduced-mode slots (replaces the full-history 0.091)')
FAMILY_LOSS_PRIOR = const('FAMILY_LOSS_PRIOR', (1.0, 1.0), 'operational', 'labels|features', 'Beta prior when a device has < 24 h history (replaces full-history family median)')
HMM_FAMILIES = const('HMM_FAMILIES', ['Sensoterra', 'MeteoHelix', 'MeteoWind', 'CloverSoilSensor', 'Agvolution', 'Dragino-SE0X-LB', 'Adeunis'],
                     'declared', 'features|labels', 'LoRaWAN families with slot-level loss; Ott and Bigbelly use the floor only (Bigbelly uses a cellular connection per manufacturer; '
                     'Ott model not identified, OTT HydroSystems stations use cellular/satellite telemetry)', sources=['BIGBELLY_HC5', 'OTT_HYDROSYSTEMS'],
                     verification='general_family')

# ---- scope (causal) ----
FLEET_WINDOW_H = const('FLEET_WINDOW_H', 2, 'operational', 'features|labels', 'same-family declarations with onsets within 2 h are concurrent')
FLEET_CHANCE_ALPHA = const('FLEET_CHANCE_ALPHA', 1e-3, 'operational', 'features|labels', 'fleet scope only if the concurrency is improbable under the family\'s own recent outage rate (Poisson, past 7 days)')
FLEET_MIN_DEVICES = const('FLEET_MIN_DEVICES', 3, 'operational', 'features|labels', 'concurrency count (including this device) for fleet scope')
NETWORK_SILENT_SHARE = const('NETWORK_SILENT_SHARE', 0.7, 'operational', 'features|labels', 'hour is network-level if >= 70% of active gateway sites silent')
NETWORK_MIN_SITES = const('NETWORK_MIN_SITES', 3, 'operational', 'features|labels', '')

# ---- gateways ----
OFFICIAL_GATEWAYS_FILE_GLOB = const('OFFICIAL_GATEWAYS_FILE_GLOB', '*Gateway_Standorte*.csv', 'external', 'features', 'gateway list supplied by the organisers',
                                    sources=['CHALLENGE_GATEWAY_LIST'], verification='verified')
SITE_RADIUS_M = const('SITE_RADIUS_M', 150, 'operational', 'features', 'packet location within 150 m of a site belongs to it')
SITE_PROMOTE_MIN_RECORDS = const('SITE_PROMOTE_MIN_RECORDS', 50, 'operational', 'features',
                                 'an unlisted location becomes a site after 50 records; earlier records stay unresolved (causal)')

# ---- physical plausibility (site physics; record-level validity) ----
PLAUSIBLE = const('PLAUSIBLE', {
    'air_temp_c': (-30, 50), 'pressure_hpa': (900, 1080), 'rain_mm_per_report': (0, 30), 'solar_sentinels': [2047, 4093, 4095],
    'soil_temp_physical_c': (-15, 40), 'vwc_pct': (0, 100), 'ec': (0, 20), 'agvolution_air_temp_c': (-30, 60),
    'lithium_cell_v': (2.0, 4.0), 'adeunis_temp_sentinel': -99, 'rssi_max_dbm': 0},
    'physical', 'features|labels|evaluation',
    'site/physics limits; stricter than the manufacturer range where physics is certain (air pressure at Hildesheim altitude, '
    'air temperature in Germany). soil_temp_physical_c is NOT a validity bound any more: a soil-probe reading inside the manufacturer '
    'range but above 40 C is a correct reading of an exposed device (TEKTELIC: direct sunlight -> case temperature) and is flagged '
    'as exposure, not as corrupt. rain > 30 mm per 10-min report: rain is the difference of consecutive counter values, so a counter '
    'reset produces one huge value. solar 2047 = 2^11-1 (saturated field; hypothesis).',
    sources=['BARANI_HELIX_DECODER', 'TEKTELIC_UG'], verification='verified')

# ---- manufacturer ranges (record/field validity) ----
SPEC_SOIL_TEMP_C = const('SPEC_SOIL_TEMP_C', {
    'Sensoterra': (-20, 60), 'CloverSoilSensor': (-15, 60), 'Dragino-SE0X-LB': (-40, 85)},
    'external', 'features|labels|evaluation',
    'soil-temperature range per manufacturer: Sensoterra operating environment (multi-depth sheet; single-depth sheet not found), '
    'CLOVER soil temperature range, Dragino LSE01 (sibling model with the same probe type). Readings outside -> out_of_spec (record invalid). '
    'Families without a spec (Agvolution) use PLAUSIBLE soil_temp_physical_c.',
    sources=['SENSOTERRA_MD_DATASHEET', 'TEKTELIC_CLOVER_SPEC', 'DRAGINO_LSE01_LORA_ALLIANCE'], verification='sibling_model')
SPEC_ACCURACY = const('SPEC_ACCURACY', {
    'MeteoHelix': {'temp_c': 0.2, 'rh_pct': 1.5, 'pressure_hpa': 1.5}, 'MeteoWind': {'wind_ms_abs': 0.2, 'wind_rel': 0.01, 'dir_deg': 2},
    'Sensoterra': {'soil_temp_c': 2.0}, 'CloverSoilSensor': {'soil_temp_c': 2.0}},
    'external', 'features|evaluation', 'reference tolerances for cross-sensor consistency checks (e.g. MeteoHelix station bias vs neighbours)',
    sources=['BARANI_HELIX_SHOP', 'BARANI_WIND_DATASHEET', 'SENSOTERRA_MD_TECHSHEET', 'TEKTELIC_CLOVER_SPEC'], verification='verified')
EFFECTIVE_RESOLUTION = const('EFFECTIVE_RESOLUTION', {'Sensoterra': {'soil_temp_c': 1.0, 'vwc_pct': 0.1}, 'MeteoHelix': {'rain_mm': 0.2},
                                                      'Dragino-SE0X-LB': {'ec_raw': 0.0001}},
    'exploratory', 'features',
    'smallest step in the data. Sensoterra temperature: every probe reports one fixed decimal (1 C steps; spec only states +-2 C accuracy); '
    'vwc 0.1 % matches the spec; rain 0.2 mm implies a 0.2 mm gauge; Dragino EC 0.0001 = 1 uS/cm spec resolution if stored in S/m. '
    'Trend/change detection must not resolve finer than these steps.',
    sources=['SENSOTERRA_MD_TECHSHEET', 'BARANI_HELIX_DATASHEET', 'DRAGINO_LSE01_LORA_ALLIANCE'], verification='sibling_model')
CLOVER_RATED_LIFE_DAYS = const('CLOVER_RATED_LIFE_DAYS', 11 * 365.25, 'external', 'features|evaluation',
    'rated battery life at 15-min reporting (SF7). A remaining-days value above it is physically impossible -> flagged and set to NaN. '
    'Replaces the earlier exploratory bound of 20000 days.', sources=['TEKTELIC_CLOVER_SPEC'], verification='verified')
WIND_CALM_MS = const('WIND_CALM_MS', 0.0, 'external', 'features|evaluation',
    'wind speed reported when the cups do not turn: the calibration m/s = 0.6335*Hz + 0.3582 gives exactly 0 only with no pulses, any pulse '
    '>= ~0.36 m/s (smallest non-zero value in the data 0.37). Direction is undefined at calm -> wind_dir_clean = NaN.',
    sources=['BARANI_WIND_COMPACT', 'BARANI_WIND_DATASHEET'], verification='verified')

# ---- radio limits (LoRa, 125 kHz) ----
RADIO_LIMITS = const('RADIO_LIMITS', {'snr_floor_sf12_db': -20.0, 'snr_floor_sf7_db': -7.5, 'sensitivity_sf12_dbm': -137.0,
                                      'saturation_dbm': -40.0},
    'external', 'features|evaluation',
    'Fixed LoRa REFERENCE values, not a link-margin estimate: the spreading factor, gateway noise figure and actual demodulation threshold '
    'are not in the exports, so link margin cannot be established. A reception beyond the SF12 reference values was received under '
    'conditions at or beyond the documented limit of the most robust setting. Gateways can still decode slightly below these values '
    '(chip sensitivity down to -148 dBm at narrow bandwidths). RSSI >= -40 dBm = receiver near saturation (RSSI uninformative).',
    sources=['SEMTECH_LORA_FAQ', 'LORA_SNR_TABLE', 'LORA_SENS_TABLE', 'SX1276_DATASHEET'], verification='general_family')
RADIO_METRICS = const('RADIO_METRICS', {'MeteoWind': 'none_in_exports', 'OttSensor': 'not_lora', 'Bigbelly': 'not_lora'},
    'external', 'features|descriptive',
    'families without RSSI/SNR: radio-quality modules skip them. Bigbelly: cellular (manufacturer). Ott: no RSSI/SNR in exports, '
    'OTT stations use cellular/satellite (model unverified). MeteoWind: LoRaWAN device, but the exports carry no radio metadata (structural).',
    sources=['BIGBELLY_HC5', 'OTT_HYDROSYSTEMS'], verification='general_family')

# ---- battery observability (what a battery forecast may use) ----
BATTERY_OBSERVABILITY = const('BATTERY_OBSERVABILITY', {
    'Sensoterra': dict(signal=None, note='non-replaceable lithium; no battery field in the exports -> not measurable'),
    'MeteoHelix': dict(signal=None, note='solar powered; no battery field in the exports -> not measurable'),
    'MeteoWind': dict(signal=None, note='solar powered; no battery field in the exports -> not measurable'),
    'CloverSoilSensor': dict(signal='battery_level_pct', note='level reported every 24 h (default); remaining-days field unreliable '
                             '(87% of values exceed the rated 11-year life, max near 65535) -> excluded from forecasting'),
    'Dragino-SE0X-LB': dict(signal='battery_v', note='Li-SOCl2 8500 mAh; flat voltage curve -> voltage is a late indicator'),
    'Agvolution': dict(signal='battery_v', note='solar node; chemistry not documented; error codes flagged'),
    'OttSensor': dict(signal='battery_v', note='12 V solar-charged system voltage; exact OTT model and battery not identified'),
    'Bigbelly': dict(signal='battery_v', note='12 V sealed lead-acid (18 or 28 Ah by model) with solar panel'),
    'Adeunis': dict(signal='battery_v', note='3.6 V (V3, SAFT LS14500) or 3.7 V lithium (newer); observed 3.78-3.87 V -> model uncertain')},
    'external', 'features|descriptive', 'no percentage or remaining life is derived from voltage (no manufacturer discharge curve available)',
    sources=['SENSOTERRA_SD_SHOP', 'TEKTELIC_CLOVER_SPEC', 'TEKTELIC_CLOVER_QSG', 'BIGBELLY_SPEC_18AH', 'BIGBELLY_SPEC_28AH',
             'OTT_HYDROSYSTEMS', 'OTT_ECOLOG1000', 'ADEUNIS_TEMP_V3_UG', 'TTN_ADEUNIS_TEMP', 'AGVOLUTION_CLIMAVI'], verification='general_family')
FIELD_NEVER_VALID = const('FIELD_NEVER_VALID', {'Adeunis': ['device_temp_c']}, 'exploratory', 'descriptive',
    'field that carried only the error sentinel in the whole export (shown as "no data", not as a fault). Record-level handling is the '
    'causal sentinel flag; this list is descriptive only.', sources=['TAGO_ADEUNIS_TEMP'], verification='unverified')
DRAGINO_EC_STORAGE = const('DRAGINO_EC_STORAGE', {'unit': 'S/m', 'to_us_cm': 1e4},
    'exploratory', 'features',
    'EC appears stored in S/m: the data step 0.0001 equals the 1 uS/cm resolution of the sibling LSE01. Whole-number EC >= 1 is '
    'treated as an unscaled-value glitch (8 records, 30 Sep 06:24-08:04, immediately before the failed-insert period) -> ec_us_cm NaN.',
    sources=['DRAGINO_LSE01_LORA_ALLIANCE'], verification='sibling_model')

TEMP_STUCK_READINGS = const('TEMP_STUCK_READINGS', 72, 'operational', 'features', 'identical consecutive readings (so far) to flag a stuck channel')

# ---- retrospective annotations (NEVER features) ----
RETRO_ANNOTATIONS = const('RETRO_ANNOTATIONS', [
    dict(kind='fleet_removal', family='CloverSoilSensor', start='2026-08-16', end='2026-09-02', note='whole fleet stopped within ~1 h on 1 Sep (2 on 17 Aug)'),
    dict(kind='fleet_removal', family='Agvolution', start='2026-09-25', end='2026-09-29', note='sequential stops 28 Sep 05:38-11:31 UTC'),
    dict(kind='remote_deactivation', family='Sensoterra', start='2026-11-13 09:00', end='2026-11-13 12:30', note='33 probes; ingestion path changed same day'),
    dict(kind='platform_outage', family='*', start='2026-09-29 07:30', end='2026-09-29 10:30', note='all gateways and soil/weather exports silent'),
    dict(kind='test_deployment', family='Adeunis', start='2026-07-07', end='2026-07-18', note='single 10-day device')],
    'retrospective', 'evaluation|descriptive', 'used only to annotate events and to report results with/without these periods')

# =====================================================================================================================
# MODULE 1: communication state and packet loss (family profiles)
# =====================================================================================================================
M1_HORIZONS_H = const('M1_HORIZONS_H', {
    'MeteoHelix': [1, 6, 24], 'MeteoWind': [1, 6, 24], 'CloverSoilSensor': [1, 6, 24], 'Dragino-SE0X-LB': [6, 24],
    'Sensoterra': [6, 24], 'OttSensor': [24], 'Agvolution': [24, 72], 'Bigbelly': [24, 72], 'Adeunis': [1, 24]},
    'operational', 'features|labels|evaluation',
    'forecast horizons per family: a horizon must contain >= 4 nominal slots (no 1 h horizon for hourly or slower families) and '
    'stay within a maintenance planning range (<= 3 days)')
M1_PRIMARY_H = const('M1_PRIMARY_H', {'Agvolution': 72, 'Bigbelly': 72}, 'operational', 'features|labels',
                     'horizon that drives the traffic-light state; 24 h unless 24 h holds fewer than 8 nominal slots')
M1_SUPERVISED_MIN = const('M1_SUPERVISED_MIN', dict(devices=8, degraded_episodes=100), 'operational', 'evaluation',
    'a family-specific supervised model is trained only if the family has >= 8 devices and >= 100 distinct degraded episodes '
    '(origins with realized availability below the family threshold, separated by >= 1 horizon) in the training data of the fold')
M1_DEGRADED_Q = const('M1_DEGRADED_Q', 0.10, 'operational', 'labels|evaluation',
    'family threshold tau_f,H = 10% quantile of realized H-window service availability in training windows WITHOUT a declared outage '
    '(capped at 1 - 1/slots(H) so one missed report is the smallest detectable degradation). Re-estimated from past data only.')
M1_FOLDS = const('M1_FOLDS', dict(validation=['2026-08-01', '2026-09-01'], test=['2026-10-01', '2026-11-01'], end='2026-11-18'),
                 'operational', 'evaluation', 'monthly walk-forward test windows; model choice per family x horizon on validation folds only')
M1_EB_KAPPA_BOUNDS = const('M1_EB_KAPPA_BOUNDS', (2.0, 5000.0), 'operational', 'features',
                           'bounds of the empirical-Bayes prior strength (method of moments across the family\'s other devices)')
M1_WINDOW_D = const('M1_WINDOW_D', 28, 'operational', 'features', 'trailing window of the individual Bernoulli / Markov / EB statistics (= HMM window)')

# =====================================================================================================================
# battery health
# =====================================================================================================================
LEAD_ACID_REST_V = const('LEAD_ACID_REST_V', dict(good=12.3, warning=12.0, critical=11.7), 'external', 'features|labels',
    'generic 12 V lead-acid resting-voltage references (~75 %, ~50 %, ~25 % state of charge). NOT Bigbelly/OTT specific; the night-time '
    'minimum of a solar-charged system is only a PROXY for resting voltage (open-circuit voltage needs hours to days to stabilise). '
    'Used as condition bands and threshold-crossing targets, never converted to a percentage.',
    sources=['OUTBACK_MX60_SOC', 'SAMLEX_STANDING_V', 'DYNAMIS_SLA_MANUAL'], verification='general_family')
BATTERY_FORECAST_DAYS = const('BATTERY_FORECAST_DAYS', [7, 14], 'operational', 'evaluation', 'backtest horizons for the resting-voltage proxy')
BATTERY_RISK_DAYS = const('BATTERY_RISK_DAYS', 30, 'operational', 'features', 'window for P(threshold crossing)')
