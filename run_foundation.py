"""Reproducible data foundation: raw exports -> bronze -> silver -> gold.
usage: python run_foundation.py --raw /mnt/user-data/uploads --out ./foundation [--cutoff '2026-08-20'] [--devices id1,id2]
--cutoff simulates the information available at that moment (all records at/after it are removed; observation ends there)."""
import argparse, hashlib, json, os, platform, sys, time, glob
import numpy as np, pandas as pd
from hsm import bronze, silver, uplinks, schedule, outages, config

def code_hash():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'hsm', '*.py'))) + [__file__]:
        h.update(open(f, 'rb').read())
    return h.hexdigest()

def main(raw, out, cutoff=None, devices=None, stream_end=None):
    """stream_end (live mode): per export stream ('devices', 'soil', 'weather') the time up to which the stream is known to be
    complete - 'now' for a current stream, its newest record for a stale one, so a stale export yields 'insufficient information',
    not missed reports."""
    t0 = time.time(); os.makedirs(out, exist_ok=True); log = {}
    man = bronze.build(raw, f'{out}/bronze', cutoff)
    S = {'devices': silver.devices(f'{out}/bronze'), 'soil': silver.soil(f'{out}/bronze'), 'weather': silver.weather(f'{out}/bronze')}
    gw_off = silver.gateways_official(f'{out}/bronze')
    if devices:
        keep = set(devices)
        S = {k: v[v.device_id.isin(keep)] for k, v in S.items()}
    for k, v in S.items(): v.to_parquet(f'{out}/silver_{k}.parquet', index=False)
    gw_off.to_parquet(f'{out}/silver_gateways_official.parquet', index=False)
    # structural observation end per family: latest export end among the files that carry the family
    ends = {k: (pd.Timestamp(cutoff) if cutoff else pd.Timestamp(man[k]['export_end'])) for k in ['devices', 'soil', 'weather']}
    if stream_end: ends.update({k: min(pd.Timestamp(v), ends[k]) for k, v in stream_end.items() if v is not None})
    fam_src = pd.concat([v[['family']].drop_duplicates().assign(src=k) for k, v in S.items()])
    obs_end_rx = fam_src.assign(end=fam_src.src.map(ends)).groupby('family').end.max().to_dict()
    obs_end = {f: ends[s] for f, s in config.DATA_SOURCE.items()}          # service: end of the family's data table
    U, links, sites, radio = uplinks.build(S['devices'], S['soil'], S['weather'], gw_off)
    U, L, LR, H = schedule.build(U, obs_end_rx, obs_end)
    # causal outage declarations per device
    parts = []; fits = []
    first = U[U.data_stored].groupby('device_id').t.min()
    for dev, G in L.groupby('device_id', sort=False):
        fam = G.family.iat[0]
        G2, f = outages.declare_device(G, fam, first[dev]); parts.append(G2)
        fits += [dict(device_id=dev, family=fam, **x) for x in f]
    L = outages.finalize_open(pd.concat(parts, ignore_index=True), obs_end)
    net, site_hour = outages.network_hours(radio)
    D = outages.scope(L[L.status == 'declared'], net, site_hour, U)
    L = L.merge(D[['gap_id', 'scope_at_declaration', 'scope_final', 'fleet_chance_p_at_declaration']], on='gap_id', how='left')
    # cause attribution: did the system still RECEIVE uplinks while stored data was missing?  (backend vs no reception)
    rxd = {d: g.t.values for d, g in U.groupby('device_id')}
    def rx_during(r):
        tt = rxd.get(r.device_id, np.array([], 'datetime64[ns]')); end = r.recovered_at if pd.notna(r.recovered_at) else obs_end[r.family]
        return int(((tt > np.datetime64(r.onset)) & (tt < np.datetime64(end))).sum())
    L['uplinks_received_during'] = [rx_during(r) if r.status == 'declared' else np.nan for r in L.itertuples()]
    L['cause_hint'] = np.where(L.status != 'declared', None, np.where(L.uplinks_received_during > 0, 'data not stored while uplinks received (backend / data path)', 'no uplinks received'))
    # retrospective annotations (evaluation / descriptive only)
    L['retro_annotation'] = None
    for a in config.RETRO_ANNOTATIONS:
        m = ((L.family == a['family']) | (a['family'] == '*')) & (L.onset >= pd.Timestamp(a['start'])) & (L.onset <= pd.Timestamp(a['end']))
        L.loc[m & (L.status == 'declared'), 'retro_annotation'] = a['kind']
    # device-hour gold: outage state and eligibility, all observable at hour end
    H = H.sort_values(['device_id', 'hour_end']).reset_index(drop=True)
    H['in_declared_outage'] = False; H['declared_scope'] = None
    dec = L[L.status == 'declared']
    for dev, idx in H.groupby('device_id').indices.items():
        he = H.hour_end.values[idx]; d = dec[dec.device_id == dev]
        for r in d.itertuples():
            end = np.datetime64(r.recovered_at) if pd.notna(r.recovered_at) else np.datetime64('2100-01-01')
            m = (he >= np.datetime64(r.declared_at)) & (he < end)
            H.loc[H.index[idx[m]], 'in_declared_outage'] = True; H.loc[H.index[idx[m]], 'declared_scope'] = r.scope_at_declaration
    H['history_days'] = (H.hour_end - H.device_id.map(first)).dt.total_seconds() / 86400
    H['service_started'] = H.svc_last_arrival.notna()
    H['developing_silence'] = (H.svc_silence_min > (1 + config.SLOT_GRACE_FRAC) * H.svc_mode_min) & ~H.in_declared_outage
    H['eligible_for_prediction'] = H.service_started & H.data_observed & (H.history_days >= config.MIN_HISTORY_D) & ~H.in_declared_outage
    H = H.merge(net[['hour_end', 'network_level']], on='hour_end', how='left'); H['network_level'] = H.network_level.fillna(False)
    # documented meaning of the causally inferred reporting mode (manufacturer source in config.MODE_MEANING)
    H['svc_mode_meaning'] = pd.Series(list(zip(H.family, H.svc_mode_min))).map(config.MODE_MEANING).values
    # write gold
    for name, df in {'uplinks': U, 'uplink_records': links, 'gateway_sites': sites, 'radio_records': radio, 'silences': L, 'reception_gaps': LR,
                     'hmm_fits': pd.DataFrame(fits), 'network_hours': net, 'device_hour': H}.items():
        df.to_parquet(f'{out}/gold_{name}.parquet', index=False)
    site_hour.reset_index().rename(columns={'index': 'hour_end'}).to_parquet(f'{out}/gold_site_hour.parquet', index=False)
    man.update(dict(cutoff=str(cutoff), devices_filter=devices, observation_end_service={k: str(v) for k, v in obs_end.items()},
                    observation_end_reception={k: str(v) for k, v in obs_end_rx.items()},
                    code_sha256=code_hash(), python=sys.version.split()[0], platform=platform.platform(),
                    packages={m: __import__(m).__version__ for m in ['pandas', 'numpy', 'duckdb', 'hmmlearn', 'sklearn']},
                    constants={k: dict(provenance=v['provenance'], allowed_use=v['allowed_use'], sources=v['sources'], verification=v['verification'])
                               for k, v in config.C.items()},
                    sources={k: dict(title=t, url=u) for k, (t, u) in config.SOURCES.items()},
                    row_counts={'uplinks': len(U), 'records': len(links), 'silences': len(L), 'declared': int((L.status == 'declared').sum()),
                                'device_hours': len(H), 'hmm_fits': len(fits)},
                    runtime_s=round(time.time() - t0, 1)))
    json.dump(man, open(f'{out}/manifest.json', 'w'), indent=2, default=str)
    return man

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--cutoff'); ap.add_argument('--devices')
    a = ap.parse_args()
    m = main(a.raw, a.out, a.cutoff, a.devices.split(',') if a.devices else None)
    print(json.dumps({k: m[k] for k in ['row_counts', 'runtime_s', 'observation_end_service']}, indent=1, default=str))
