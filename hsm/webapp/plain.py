"""Plain-language translation of the system's technical outputs for operators and repair staff.
Every sentence here is a translation of a model output or a rule result; nothing new is inferred."""
import math
import numpy as np, pandas as pd

MEASURES = {   # family -> (what it measures, icon, measurement labels for the two daily series)
    'Sensoterra': ('Soil moisture', '💧', ('Soil moisture (%)', 'Soil temperature (°C)')),
    'CloverSoilSensor': ('Soil moisture', '💧', ('Soil moisture (GWC)', 'Soil temperature (°C)')),
    'Dragino-SE0X-LB': ('Soil moisture & salinity', '💧', ('Soil moisture (%)', 'Soil temperature (°C)')),
    'Agvolution': ('Soil moisture & temperature', '🌱', ('Soil moisture (%)', 'Soil temperature (°C)')),
    'MeteoHelix': ('Weather (temperature, humidity, air pressure)', '🌡️', ('Air temperature (°C)', 'Humidity (%)')),
    'MeteoWind': ('Wind speed & direction', '🌬️', ('Wind speed (m/s)', 'Gusts (m/s)')),
    'OttSensor': ('Water / hydrology station', '🌊', (None, None)),
    'Bigbelly': ('Waste bin (fill level)', '🗑️', (None, None)),
    'Adeunis': ('Temperature', '🌡️', (None, None)),
}
LIGHT = {'green': ('OK', 'Working normally'), 'yellow': ('Attention', 'Needs attention soon'), 'red': ('Action needed', 'Needs action now'),
         'grey': ('No information', 'No information available')}
STATUS = {
    'normal': ('Working', 'The sensor sends its data as expected.'),
    'degraded': ('Working, some data lost', 'The sensor works, but more messages than usual are getting lost.'),
    'significant_interruption': ('Not sending data', 'No data has arrived from this sensor for {h}.'),
    'backend_data_unavailable': ('Sending, but data is not saved', 'The sensor is working and is heard by the network, but its data is not being saved in the database. This is a server problem, not a sensor problem.'),
    'reduced_reporting_mode': ('Power-saving mode', 'The sensor reports only once a day. For this probe that usually means poor contact with the soil or very dry soil.'),
    'insufficient_information': ('No information', 'There is not enough data to judge this sensor at this moment.'),
}
CAUSE = {   # probable cause -> (plain explanation, what to do, site visit needed?)
    'backend storage failure': ('Server / database problem. The sensor itself is fine.', 'Inform the IT / data team. No site visit needed. The data can be re-imported later.', False),
    'network / platform failure': ('The whole network was down at the same time.', 'No site visit. Check the network server and its internet connection.', False),
    'gateway unavailability': ('The nearby receiver (gateway) is not working.', 'Check the gateway first: power supply and internet connection.', True),
    'fleet-wide stop (possible remote deactivation or platform change)': ('Many sensors of this type stopped at the same moment.', 'Probably switched off remotely or a platform change. Ask the operator before going out.', False),
    'poor installation / soil contact': ('The probe has poor contact with the soil.', 'Visit the sensor: push the probe firmly back into the soil or re-install it.', True),
    'measurement-channel failure': ('One of the measurements is missing, stuck or invalid.', 'Visit the sensor: check probe and cable for damage; replace if needed.', True),
    'low battery charge (measured)': ('The battery is getting low.', 'Clean the solar panel, check the charger and plan a battery replacement.', True),
    'battery deterioration (possible)': ('The battery may be wearing out.', 'Check the battery and charger at the next visit.', True),
    'weak radio connectivity': ('The radio signal is weak.', 'Check the antenna and new obstacles (trees, buildings). Consider moving the sensor.', True),
    'individual sensor or local communication failure (cause undetermined)': ('The sensor stopped sending. The exact reason is not known.', 'Visit the sensor: check that it is still there, undamaged and powered.', True),
    'elevated packet loss (cause undetermined)': ('Some messages are getting lost.', 'Keep an eye on it. No visit needed yet.', False),
    'measurement anomaly': ('Some measurements look unusual.', 'Compare with nearby sensors; check the sensor is clean and mounted correctly.', True),
    'unknown': ('The reason could not be determined.', 'Look at the details or visit the sensor.', True),
}
CONFIDENCE = {'high': 'very likely', 'medium': 'likely', 'low': 'possible'}
ANOMALY = {
    'moisture_zero': 'Soil moisture reads 0 (very dry soil or no soil contact)',
    'temperature_exposure_suspected': 'Very high temperature: the probe may be out of the soil and in the sun',
    'frozen_temperature': 'Temperature value is stuck at the same number',
    'out_of_manufacturer_range': 'A value is outside what the sensor can measure',
    'raw_frequency_fault': 'The sensor sends an invalid raw signal',
    'backend_insert_failure': 'Data could not be saved in the database',
    'battery_error_code': 'The sensor reports a battery error',
    'corrupted_packet': 'A damaged data message was received',
    'channel_never_valid': 'One measurement channel never delivers valid values',
    'ec_decoder_scaling': 'Salinity (EC) value is wrongly scaled',
    'ec_implausible': 'Salinity (EC) value is not plausible',
    'abrupt_voltage_change': 'Sudden jump in battery voltage',
}
VARIABLE = {'temp_c': 'temperature', 'rh_pct': 'humidity', 'pressure_hpa': 'air pressure', 'wind_ms': 'wind speed', 'vwc_pct': 'soil moisture'}
GATEWAY = {'active': ('Working', 'green'), 'low_traffic': ('Working (little traffic)', 'green'), 'insufficient_history': ('Too new to judge', 'grey'),
           'reduced_activity': ('Receiving less than usual', 'yellow'),
           'inactive_gateway': ('Off: its sensors switched to other receivers', 'red'),
           'inactive_unexplained': ('Off: its sensors are silent too', 'red'),
           'network_or_platform': ('Network down', 'red')}

def _f(x):
    try:
        x = float(x); return None if math.isnan(x) else x
    except (TypeError, ValueError): return None

def hours_text(h):
    h = _f(h)
    if h is None: return 'unknown'
    if h < 1: return 'less than an hour'
    if h < 1.5: return '1 hour'
    if h < 48: return f'{h:.0f} hours'
    return f'{h / 24:.0f} days'

def ago(t, now):
    if t is None or pd.isna(t): return 'never'
    return hours_text((pd.Timestamp(now) - pd.Timestamp(t)).total_seconds() / 3600) + ' ago'

def interval_text(minutes):
    m = int(minutes or 60)
    if m < 60: return f'{m} minutes'
    if m == 60: return 'hour'
    if m % 1440 == 0: return 'day' if m == 1440 else f'{m // 1440} days'
    return f'{m / 60:.0f} hours'

def anomaly_text(kinds):
    if not isinstance(kinds, str) or not kinds: return []
    out = []
    for k in sorted(set(kinds.replace(';', ',').split(','))):
        k = k.strip()
        if not k: continue
        if ':' in k:
            base, var = k.split(':', 1); v = VARIABLE.get(var, var)
            if 'ratio' in base: out.append(f'{v.capitalize()} differs from nearby sensors (possible obstruction or wear)')
            else: out.append(f'{v.capitalize()} differs from nearby stations (offset or slow drift)')
        else: out.append(ANOMALY.get(k, k.replace('_', ' ')))
    return list(dict.fromkeys(out))

POWER = {   # power source per family (manufacturer documentation, docs/MANUFACTURER_SPECS.md)
    'Sensoterra': ('sealed', '🔒', 'Sealed lithium battery (not replaceable). This sensor does not report its battery.'),
    'MeteoHelix': ('solar', '☀️', 'Solar powered. This station does not report its battery.'),
    'MeteoWind': ('solar', '☀️', 'Solar powered. This sensor does not report its battery.'),
}

def _band(c, v, fam, stale=None):
    fill = {'good': 90, 'fair': 60, 'low': 30, 'critical': 10}[c]
    return dict(kind='band', fill=fill, level=c, label=f'{v:.1f} V' if v else c, text={'good': 'Good', 'fair': 'Fair', 'low': 'Low', 'critical': 'Critical'}[c],
                note='12 V battery with solar panel. Shown as a band (good / fair / low / critical) from the measured voltage; '
                     'the voltage cannot be turned into an exact percentage for this battery.')

def battery(row, fam, last=None):
    """battery widget: fill level only where the measurement supports it. Lead-acid: condition band (no %),
    CLOVER: manufacturer's own %, lithium: OK / problem, no telemetry: power source. `last`: the last known reading
    (condition, value, time) when the sensor has stopped sending - shown with its date, greyed."""
    c = row.get('battery_condition'); v = _f(row.get('battery_value'))
    if c == 'no_recent_reading' and last is not None:
        lc, lv, lt = last; r = battery(dict(battery_condition=lc, battery_value=lv), fam)
        r.update(stale=True, since=lt, text=f"Last known ({lt[:10]})",
                 note=f"The sensor has stopped sending, so this is its last battery reading from {lt[:10]}. " + r['note'])
        return r
    if c in ('good', 'fair', 'low', 'critical') and fam in ('Bigbelly', 'OttSensor'): return _band(c, v, fam)
    if fam == 'CloverSoilSensor' and v is not None:
        return dict(kind='percent', fill=int(round(v)), level='good' if v >= 50 else 'low', label=f'{v:.0f} %', text='Good' if v >= 50 else 'Low',
                    note="Lithium battery (about 11 years). Level as reported by the sensor itself (manufacturer's estimate).")
    if c == 'nominal_plateau':
        return dict(kind='ok', fill=100, level='good', label=f'{v:.2f} V' if v else 'OK', text='OK',
                    note='Lithium battery: its voltage stays flat until it is almost empty, so only "OK" or "problem" can be shown, not a level.')
    if c == 'voltage_step_change':
        return dict(kind='ok', fill=40, level='low', label=f'{v:.2f} V' if v else '!', text='Check',
                    note='Lithium battery: a sudden voltage change was detected. Check the battery.')
    if fam in POWER:
        kind, ic, note = POWER[fam]
        return dict(kind=kind, fill=0, level='none', label=ic, text='Solar powered' if kind == 'solar' else 'Sealed battery', note=note)
    if c == 'no_recent_reading':
        return dict(kind='unknown', fill=0, level='unknown', label='?', text='No reading yet', note='This sensor reports its battery, but no reading has arrived yet.')
    return dict(kind='unknown', fill=0, level='unknown', label='?', text='Not reported', note='This sensor does not report its battery in the data.')

def level(p, lo=0.10, hi=0.40):
    p = _f(p)
    if p is None: return None
    return dict(p=p, pct=f'{p:.0%}', level='Low' if p < lo else 'Medium' if p < hi else 'High')

PREDICTION_QUALITY = {   # from the held-out test months (docs/SYSTEM.md); shown next to each prediction
    'Sensoterra': 'Tested on Oct-Nov data: good for long breakdowns (catches about half, ~6 days ahead); weak for the next 24 h.',
    'MeteoHelix': 'Tested on Oct-Nov data: good for the next 24 h; too few long breakdowns to test the 7-day forecast.',
    'MeteoWind': 'Tested on Oct-Nov data: good for the next 24 h; too few long breakdowns to test the 7-day forecast.',
    'CloverSoilSensor': 'Checked on Aug-Sep data (caught about 30 % of breakdowns a day ahead); these sensors stopped afterwards, so it could not be confirmed later.',
}
DEFAULT_QUALITY = 'Too few past breakdowns of this sensor type to test the forecast. Treat it as a rough guide.'

def status(row, now, fam):
    st = row.get('state') or 'insufficient_information'
    title, text = STATUS.get(st, (st, ''))
    un = None
    if st in ('significant_interruption', 'backend_data_unavailable'):
        un = (_f(row.get('silence_svc_slots')) or 0) * (row.get('_interval') or 60) / 60
    text = text.format(h=hours_text(un))
    cause = row.get('probable_cause'); cause = cause if isinstance(cause, str) and cause else None
    expl, todo, visit = CAUSE.get(cause, CAUSE['unknown']) if cause else (None, 'Nothing to do.', False)
    return dict(state=st, title=title, text=text, unavailable_h=un, unavailable_text=hours_text(un) if un else None,
                cause=expl, cause_tech=cause, cause_certainty=CONFIDENCE.get(row.get('cause_confidence'), None) if cause else None,
                todo=todo, visit=visit, evidence=row.get('cause_evidence') or None)

def predictions(row, fam):
    st = row.get('state'); out = []
    down = st in ('significant_interruption', 'backend_data_unavailable')
    if down:
        a = level(row.get('p_still_down_24h'), 0.3, 0.7); b = level(row.get('p_still_down_7d'), 0.3, 0.7)
        if a: out.append(dict(q='Chance it is still not working tomorrow', **a))
        if b: out.append(dict(q='Chance it is still not working in 7 days', **b))
    else:
        s = level(row.get('p_silence_becomes_outage'), 0.2, 0.5)
        if s: out.append(dict(q='It missed its last messages. Chance this turns into a real breakdown', **s))
        r7 = level(row.get('risk_prolonged_7d'))
        if r7: out.append(dict(q='Chance of a long breakdown (a day or more) in the next 7 days', **r7))
        rl = level(row.get('risk_long'), 0.05, 0.25); H = _f(row.get('risk_long_H')) or 24
        if rl: out.append(dict(q=f'Chance it stops sending within the next {int(H)} hours', **rl))
    return dict(items=out, quality=PREDICTION_QUALITY.get(fam, DEFAULT_QUALITY))
