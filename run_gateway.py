"""Component 2: gateway monitoring.  usage: python run_gateway.py --foundation <dir>"""
import argparse, json, time
import numpy as np, pandas as pd
from hsm import gateway

def main(fd):
    t0 = time.time(); rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    R = rd('gold_radio_records'); sites = rd('gold_gateway_sites'); net = rd('gold_network_hours')
    man = json.load(open(f'{fd}/manifest.json')); as_of = pd.Timestamp(max(man['observation_end_reception'].values())).floor('h')
    A, Dp = gateway.site_hours(R, sites); A = gateway.classify(A, net)
    Ev = gateway.interruption_events(A)
    W = gateway.warning_signal_test(A, Ev)
    rates = gateway.interruption_rate(Ev, A, as_of)
    C = gateway.current(A, Ev, rates, sites, as_of, Dp)
    dep = gateway.sensor_dependence(Dp, as_of); alt = gateway.alternatives(R, as_of)
    for n, d in [('site_hour', A), ('device_primary', Dp), ('events', Ev), ('warning_test', W), ('current', C), ('sensor_dependence', dep), ('alternatives', alt)]:
        d.to_parquet(f'{fd}/gold_gw_{n}.parquet', index=False)
    print('done', round(time.time() - t0), 's; events', len(Ev))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); main(ap.parse_args().foundation)
