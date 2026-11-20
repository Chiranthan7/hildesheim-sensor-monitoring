"""Component 4: measurement anomalies.  usage: python run_anomaly.py --foundation <dir>"""
import argparse, time
import numpy as np, pandas as pd
from hsm import anomaly
from hsm.config import SPEC_ACCURACY, M1_FOLDS

def main(fd):
    t0 = time.time(); rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    soil, wx, devs = rd('silver_soil'), rd('silver_weather'), rd('silver_devices')
    rule = anomaly.rule_daily(soil, wx, devs); rule.to_parquet(f'{fd}/gold_anom_rule_daily.parquet', index=False)
    ev, inj, frz, off = [], [], [], []
    for var in ['temp_c', 'rh_pct', 'pressure_hpa']:
        acc = SPEC_ACCURACY['MeteoHelix'][var]
        p, r = anomaly.neighbour_residuals(wx, var)
        r.to_parquet(f'{fd}/gold_anom_resid_{var}.parquet')
        ev.append(anomaly.station_events(r, acc, var, 'MeteoHelix')); off.append(anomaly.static_offsets(r, acc, var, 'MeteoHelix'))
        inj.append(anomaly.inject_eval(p, r, var, acc, M1_FOLDS['test'][0]))
        print(var, round(time.time() - t0), flush=True)
    ev.append(anomaly.wind_ratio_events(wx))
    for fam, var in [('MeteoHelix', 'temp_c'), ('MeteoWind', 'wind_ms')]:
        frz.append(anomaly.frozen_runs(wx, fam, var, anomaly.FROZEN_N[fam]))
    pd.concat(ev, ignore_index=True).to_parquet(f'{fd}/gold_anom_station_events.parquet', index=False)
    pd.concat(inj, ignore_index=True).to_parquet(f'{fd}/gold_anom_injection_eval.parquet', index=False)
    pd.concat(off, ignore_index=True).to_parquet(f'{fd}/gold_anom_static_offsets.parquet', index=False)
    pd.concat(frz, ignore_index=True).to_parquet(f'{fd}/gold_anom_frozen.parquet', index=False)
    anomaly.soil_jumps(soil).to_parquet(f'{fd}/gold_anom_soil_jumps.parquet', index=False)
    print('done', round(time.time() - t0), 's')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); main(ap.parse_args().foundation)
