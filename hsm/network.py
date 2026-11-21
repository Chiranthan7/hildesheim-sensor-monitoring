"""Radio-network analysis for the "Improve the network" page and the sensor focus mode.

Everything here comes from the measured radio records (which gateway received which message, with RSSI / SNR):
  * links      - for every sensor, the gateways that actually received its messages, how often, and how strong
  * signal     - per sensor: signal quality of its best gateway, share of weak messages, number of regular gateways
  * model      - how much the signal (SNR) changes with distance, estimated *within* sensors that are heard by
                 several gateways (device fixed effects), so antenna, burial depth and sensor type cancel out

Signal classes use the LoRa demodulation limits of the Semtech SX1276/77/78/79 datasheet (Table 13):
SNR limit -7.5 dB at SF7 ... -20 dB at SF12. A message at SNR >= -7.5 dB is received even at the fastest data rate;
below -15 dB only the slowest data rates (SF11/SF12) work and messages are close to being lost.
Nothing is invented: sensors without radio records (e.g. MeteoWind, OTT, Bigbelly, Dragino) are reported as
"no signal data". The distance model is an empirical average with its uncertainty, not a coverage simulation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SNR_OK, SNR_WEAK, SNR_STRONG = -7.5, -15.0, 0.0           # SX127x datasheet SF7 / SF10 limits; 0 dB = strong
MIN_LINK_RECORDS, MIN_LINK_SHARE = 20, 0.05               # a gateway "regularly" receives a sensor
SOURCE = 'Semtech SX1276/77/78/79 datasheet, Table 13 (LoRa demodulator SNR limits per spreading factor)'


def signal_class(snr):
    if snr is None or not np.isfinite(snr): return 'none'
    return 'strong' if snr >= SNR_STRONG else 'ok' if snr >= SNR_OK else 'weak' if snr >= SNR_WEAK else 'very_weak'


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371000.0 * 2 * np.arcsin(np.sqrt(h))


def within_device_slope(F):
    """SNR change per tenfold distance, device fixed effects (only sensors heard by >= 2 gateways)"""
    F = F[F.groupby('device_id').device_id.transform('size') >= 2]
    x = np.log10(F.d); xc = x - x.groupby(F.device_id).transform('mean')
    yc = F.snr - F.snr.groupby(F.device_id).transform('mean')
    b = float((xc * yc).sum() / (xc ** 2).sum()); res = yc - b * xc
    dof = len(F) - F.device_id.nunique() - 1
    se = float(np.sqrt((res ** 2).sum() / dof / (xc ** 2).sum()))
    return dict(snr_db_per_decade=round(b, 2), se=round(se, 2), residual_sd_db=round(float(res.std()), 1),
                within_r2=round(float(1 - (res ** 2).sum() / (yc ** 2).sum()), 2), n_links=int(len(F)), n_sensors=int(F.device_id.nunique()))


def build(fd, devices: pd.DataFrame, out: Path | None = None) -> dict:
    """fd: foundation dir; devices: appdata devices table (device_id, family, lat, lon = latest position)"""
    fd = Path(fd)
    R = pd.read_parquet(fd / 'gold_radio_records.parquet', columns=['device_id', 't', 'gw_site_retro', 'rssi', 'snr'])
    R = R[R.gw_site_retro.notna() & (R.gw_site_retro != 'unresolved')].rename(columns={'gw_site_retro': 'gw'})
    G = pd.read_parquet(fd / 'gold_gateway_sites.parquet').set_index('site')
    U = pd.read_parquet(fd / 'gold_uplinks.parquet', columns=['device_id', 'snr_max', 'radio_metrics_available'])
    U = U[U.radio_metrics_available.astype(bool) & U.snr_max.notna()]
    D = devices.set_index('device_id')

    L = R.groupby(['device_id', 'gw'], observed=True).agg(n=('t', 'size'), rssi=('rssi', 'median'), snr=('snr', 'median'),
                                                           first=('t', 'min'), last=('t', 'max')).reset_index()
    L['share'] = L.n / L.device_id.map(R.groupby('device_id').size())
    L = L.join(D[['lat', 'lon']], on='device_id').join(G[['lat', 'lon']].rename(columns={'lat': 'glat', 'lon': 'glon'}), on='gw')
    L['d'] = haversine_m(L.lat, L.lon, L.glat, L.glon)
    L['regular'] = (L.n >= MIN_LINK_RECORDS) & (L.share >= MIN_LINK_SHARE)

    fit = L[L.regular & L.d.notna() & (L.d > 20)]
    model = within_device_slope(fit)
    model.update(typical_range_m=int(fit.d.quantile(0.9)), max_link_m=int(fit.d.max()),
                 note='Measured messages only: links too weak to deliver anything are not in the data, so the true distance '
                      'effect is probably stronger than this average (the estimate is conservative).')

    W = U.groupby('device_id').snr_max.agg(n='size', weak=lambda s: float((s < SNR_OK).mean()), very_weak=lambda s: float((s < SNR_WEAK).mean()))
    sensors = {}
    for dev, g in L[L.regular].groupby('device_id'):
        g = g.sort_values('snr', ascending=False); best = g.iloc[0]
        w = W.loc[dev] if dev in W.index else None
        sensors[str(dev)] = dict(
            best_gw=str(best.gw), best_snr=round(float(best.snr), 1), best_rssi=round(float(best.rssi), 1),
            best_d=None if pd.isna(best.d) else round(float(best.d)), cls=signal_class(float(best.snr)),
            n_gw=int(len(g)), weak_share=None if w is None else round(w.weak, 3), very_weak_share=None if w is None else round(w.very_weak, 3),
            links=[dict(gw=str(r.gw), n=int(r.n), share=round(float(r.share), 3), snr=round(float(r.snr), 1), rssi=round(float(r.rssi), 1),
                        d=None if pd.isna(r.d) else round(float(r.d)), cls=signal_class(float(r.snr))) for r in g.itertuples()])
    period = dict(first=str(R.t.min())[:10], last=str(R.t.max())[:10])
    res = dict(model=model, sensors=sensors, period=period, thresholds=dict(ok=SNR_OK, weak=SNR_WEAK, strong=SNR_STRONG, source=SOURCE),
               gateways={str(s): dict(lat=float(r.lat), lon=float(r.lon), official=bool(r.official)) for s, r in G.iterrows() if pd.notna(r.lat)})
    if out is not None: json.dump(res, open(Path(out) / 'network.json', 'w'))
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); ap.add_argument('--appdata', required=True)
    a = ap.parse_args()
    r = build(a.foundation, pd.read_parquet(Path(a.appdata) / 'devices.parquet'), Path(a.appdata))
    cls = pd.Series([s['cls'] for s in r['sensors'].values()]).value_counts().to_dict()
    print(json.dumps(dict(model=r['model'], sensors=len(r['sensors']), classes=cls, single_gw=sum(s['n_gw'] == 1 for s in r['sensors'].values())), indent=1))
