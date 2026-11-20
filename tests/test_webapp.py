"""Web app: API answers for every sensor, JSON is valid (no NaN), plain language, directions, no invented positions or battery levels."""
import json, os, math
from pathlib import Path
import pandas as pd, pytest

APPDATA = Path(os.environ.get('HSM_APPDATA', '/home/claude/appdata'))
pytestmark = pytest.mark.skipif(not (APPDATA / 'snapshots.parquet').exists(), reason='appdata not prepared')

@pytest.fixture(scope='module')
def client():
    from hsm.webapp.app import create_app
    return create_app(str(APPDATA)).test_client()

def strict(r):
    assert r.status_code == 200, r.data[:300]
    return json.loads(r.data, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f'invalid JSON constant {c}')))

def test_index_and_static(client):
    r = client.get('/'); assert r.status_code == 200 and b'Smart Sensor Monitoring' in r.data
    for f in ['app.js', 'app.css', 'vendor/leaflet.js', 'vendor/chart.umd.min.js', 'vendor/leaflet.markercluster.js']:
        assert client.get(f'/static/{f}').status_code == 200, f

def test_state_counts_and_positions(client):
    d = strict(client.get('/api/state')); S = d['sensors']
    assert sum(d['counts'].values()) == len(S) > 100
    dev = pd.read_parquet(APPDATA / 'devices.parquet').set_index('device_id')
    for c in S:
        assert c['light'] in ('red', 'yellow', 'green', 'grey') and c['icon'] and c['measures'] and c['status']
        if c['lat'] is None: assert pd.isna(dev.loc[c['id'], 'lat'])                 # no invented positions
        else: assert abs(c['lat'] - dev.loc[c['id'], 'lat']) < 1e-6
    assert {u['id'] for u in d['unlocated']} == {c['id'] for c in S if c['lat'] is None}

def test_every_sensor_detail(client):
    S = strict(client.get('/api/state'))['sensors']
    for c in S:
        d = strict(client.get(f"/api/sensor/{c['id']}"))
        assert d['status']['title'] and d['predictions']['quality']
        if d['directions']['lat'] is not None:
            assert isinstance(d['directions']['lat'], float)
        for p in d['predictions']['items']: assert 0 <= p['p'] <= 1 and p['level'] in ('Low', 'Medium', 'High')

def test_battery_never_invented(client):
    S = strict(client.get('/api/state'))['sensors']
    for c in S:
        b = c['battery']
        if c['family'] in ('Sensoterra', 'MeteoHelix', 'MeteoWind'): assert b['kind'] == 'unknown' and b['label'] == '?'
        if b['kind'] == 'band': assert b['label'].endswith('V') and '%' not in b['label']      # lead-acid: voltage + band, never a percentage
        if b['kind'] == 'percent': assert c['family'] == 'CloverSoilSensor'                  # only the sensor's own reported level

def test_time_travel_changes_state(client):
    a = strict(client.get('/api/state?as_of=2026-09-23 15:00')); b = strict(client.get('/api/state'))
    assert a['counts'] != b['counts']

def test_gateways_alerts_quality_email(client):
    G = strict(client.get('/api/state'))['gateways']; assert G
    g = strict(client.get(f"/api/gateway/{G[0]['id']}")); assert 'gateway' in g
    al = strict(client.get('/api/alerts')); assert 'open' in al and al['sent']
    m = strict(client.get(f"/api/email/{al['sent'][0]['id']}")); assert m['subject']
    q = strict(client.get('/api/quality')); assert any(c['grade'] == 'No' for c in q['cards'])   # gateway prediction honestly reported as not possible

def test_check_without_live_is_refused(client):
    assert client.post('/api/check').status_code == 400
