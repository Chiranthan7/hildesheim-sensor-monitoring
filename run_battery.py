"""Component 3: battery health.  usage: python run_battery.py --foundation <dir>"""
import argparse, json, time
import pandas as pd
from hsm import battery

def main(fd):
    t0 = time.time(); devs = pd.read_parquet(f'{fd}/silver_devices.parquet')
    man = json.load(open(f'{fd}/manifest.json')); as_of = pd.Timestamp(man['observation_end_service']['OttSensor'])
    V, LV = battery.daily(devs); B = battery.backtest(V); ch = battery.choose(B)
    A = battery.assess(V, LV, as_of, ch); An = battery.anomalies(V, devs)
    for n, d in [('daily', V), ('level_daily', LV), ('backtest', B), ('assessment', A), ('anomalies', An)]: d.to_parquet(f'{fd}/gold_battery_{n}.parquet', index=False)
    json.dump({f'{k[0]}|{k[1]}': v for k, v in ch.items()}, open(f'{fd}/battery_choice.json', 'w'), indent=1)
    print('done', round(time.time() - t0), 's')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); main(ap.parse_args().foundation)
