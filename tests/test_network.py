"""Radio-network helpers: signal classes and the within-sensor distance effect."""
import numpy as np, pandas as pd
from hsm.network import signal_class, within_device_slope, haversine_m

def test_signal_classes_follow_datasheet_limits():
    assert signal_class(3) == 'strong' and signal_class(0) == 'strong'
    assert signal_class(-7.5) == 'ok' and signal_class(-7.6) == 'weak'
    assert signal_class(-15) == 'weak' and signal_class(-15.1) == 'very_weak'
    assert signal_class(None) == 'none' and signal_class(float('nan')) == 'none'

def test_haversine():
    assert abs(haversine_m(52.15, 9.95, 52.16, 9.95) - 1112) < 2

def test_within_device_slope_recovers_known_effect_despite_device_offsets():
    rng = np.random.default_rng(0); rows = []
    for k in range(40):                                   # each sensor: own offset (antenna, burial), 3-5 gateways
        off = rng.normal(0, 8)
        for d in rng.uniform(100, 5000, rng.integers(3, 6)):
            rows.append(dict(device_id=f'd{k}', d=d, snr=off - 5.0 * np.log10(d) + rng.normal(0, 1)))
    r = within_device_slope(pd.DataFrame(rows))
    assert abs(r['snr_db_per_decade'] + 5.0) < 3 * r['se'] + 0.3 and r['n_sensors'] == 40
