"""Module 1 causal features and targets per device-hour (hour END h; everything uses information resolved before h).
Targets: y_H = realized SERVICE availability over (h, h+H]: sum filled / sum expected nominal slots of the next H hours; defined only
when all H hours are complete and observed (no censored outcomes)."""
import numpy as np, pandas as pd
from ..config import NOMINAL_MIN, SLOT_GRACE_FRAC, M1_HORIZONS_H, RADIO_METRICS

ROLL = [1, 6, 24, 72, 168, 672]

def _roll(g, col, w):
    return g[col].transform(lambda s: s.rolling(w, min_periods=1).sum())

def hourly_base(H):
    H = H.sort_values(['device_id', 'hour_end']).reset_index(drop=True)
    g = H.groupby('device_id', sort=False)
    out = H[['device_id', 'family', 'hour_end']].copy()
    for w in ROLL:
        e = _roll(g, 'svc_nominal_expected', w); out[f'avail_svc_{w}h'] = _roll(g, 'svc_nominal_filled', w) / e.replace(0, np.nan)
        if w in (24, 168):
            er = _roll(g, 'rx_nominal_expected', w); out[f'avail_rx_{w}h'] = _roll(g, 'rx_nominal_filled', w) / er.replace(0, np.nan)
            out[f'avail_valid_{w}h'] = _roll(g, 'svc_nominal_valid_filled', w) / e.replace(0, np.nan)
            out[f'avail_mode_{w}h'] = _roll(g, 'svc_mode_filled', w) / _roll(g, 'svc_mode_expected', w).replace(0, np.nan)
    m0 = H.family.map(NOMINAL_MIN)
    out['silence_svc_slots'] = H.svc_silence_min / m0
    out['silence_rx_slots'] = H.rx_silence_min / m0
    overdue = H.svc_silence_min > (1 + SLOT_GRACE_FRAC) * m0
    out['heard_since_last_stored'] = (H.rx_last_arrival > H.svc_last_arrival) | (H.rx_last_arrival.notna() & H.svc_last_arrival.isna())
    out['backend_suspect_now'] = overdue & out.heard_since_last_stored
    out['rx_minus_svc_24h'] = (out.avail_rx_24h - out.avail_svc_24h)
    out['excess_24h'] = _roll(g, 'svc_excess_reports', 24)
    out['history_days'] = H.history_days; out['reduced_mode'] = (H.rx_mode_min != m0) & H.rx_mode_min.notna()
    out['network_level'] = H.network_level.astype(float)
    hr = H.hour_end.dt.hour + 0.5
    out['hod_sin'] = np.sin(2 * np.pi * hr / 24); out['hod_cos'] = np.cos(2 * np.pi * hr / 24); out['dow'] = H.hour_end.dt.dayofweek
    # fleet context: share of OTHER family devices in a developing silence or declared outage at h
    bad = (H.developing_silence | H.in_declared_outage).astype(float); act = H.service_started.astype(float)
    k = H.groupby(['family', 'hour_end']); nb = k[[]].ngroup()
    sb = bad.groupby([H.family, H.hour_end]).transform('sum') - bad; sa = act.groupby([H.family, H.hour_end]).transform('sum') - act
    out['fleet_bad_share'] = sb / sa.replace(0, np.nan); out['fleet_active'] = sa
    return out

def outage_history(L, base):
    """declared outages known at h: count in last 28 d, hours since last recovery, duration of last completed outage"""
    D = L[L.status == 'declared'][['device_id', 'declared_at', 'recovered_at', 'onset']]
    res = pd.DataFrame(index=base.index, data={'n_declared_28d': 0.0, 'h_since_recovery': np.nan, 'last_outage_h': np.nan})
    for dev, idx in base.groupby('device_id').indices.items():
        d = D[D.device_id == dev]
        if not len(d): continue
        h = base.hour_end.values[idx].astype('datetime64[ns]')
        da = np.sort(d.declared_at.values.astype('datetime64[ns]'))
        res.iloc[idx, 0] = np.searchsorted(da, h, 'left') - np.searchsorted(da, h - np.timedelta64(28, 'D'), 'left')
        r = d.dropna(subset=['recovered_at']).sort_values('recovered_at')
        if len(r):
            rt = r.recovered_at.values.astype('datetime64[ns]'); j = np.searchsorted(rt, h, 'left') - 1
            res.iloc[idx, 1] = np.where(j >= 0, (h - rt[np.maximum(j, 0)]).astype('timedelta64[m]').astype(float) / 60, np.nan)
            dur = (r.recovered_at - r.onset).dt.total_seconds().values / 3600
            res.iloc[idx, 2] = np.where(j >= 0, dur[np.maximum(j, 0)], np.nan)
    return res

def radio(U, base):
    """hourly radio aggregates of received uplinks, rolled over 24 h and 168 h (families with radio metrics only)"""
    u = U[U.radio_metrics_available & U.snr_max.notna()].copy()
    u['hour_end'] = u.t.dt.floor('h') + pd.Timedelta('1h')
    a = u.groupby(['device_id', 'hour_end']).agg(n_rad=('snr_max', 'size'), snr_s=('snr_max', 'sum'), rssi_s=('rssi_max', 'sum'),
                                               beyond=('flag_beyond_sf12_reference_limits', 'sum'), sat=('flag_rssi_near_saturation', 'sum'),
                                               gws=('n_gw_sites', 'sum')).reset_index()
    b = base[['device_id', 'hour_end']].merge(a, on=['device_id', 'hour_end'], how='left').fillna(0)
    g = b.groupby('device_id', sort=False); out = pd.DataFrame(index=base.index)
    for w in (24, 168):
        n = g.n_rad.transform(lambda s: s.rolling(w, min_periods=1).sum()).replace(0, np.nan)
        out[f'snr_mean_{w}h'] = g.snr_s.transform(lambda s: s.rolling(w, min_periods=1).sum()).values / n.values
        out[f'rssi_mean_{w}h'] = g.rssi_s.transform(lambda s: s.rolling(w, min_periods=1).sum()).values / n.values
        if w == 24:
            out['beyond_ref_share_24h'] = g.beyond.transform(lambda s: s.rolling(w, min_periods=1).sum()).values / n.values
            out['saturation_share_24h'] = g.sat.transform(lambda s: s.rolling(w, min_periods=1).sum()).values / n.values
            out['gw_sites_per_uplink_24h'] = g.gws.transform(lambda s: s.rolling(w, min_periods=1).sum()).values / n.values
    out['snr_drop_24_vs_168'] = out.snr_mean_168h - out.snr_mean_24h
    out.loc[base.family.isin(list(RADIO_METRICS)).values, :] = np.nan
    return out

def channels_and_measurements(U, soil, base):
    """family-specific measurement features: channel completeness (multi-channel), Sensoterra vwc = 0 share (manufacturer:
    poor contact / dry soil), CLOVER censored-GWC share"""
    u = U[U.data_stored].copy(); u['hour_end'] = u.t_data.dt.floor('h') + pd.Timedelta('1h')
    a = u.groupby(['device_id', 'hour_end']).agg(cc_s=('channel_completeness', 'sum'), cc_n=('channel_completeness', 'count')).reset_index()
    s = soil[(soil.dup_rank == 0) & soil.data_stored].copy(); s['hour_end'] = s.t.dt.floor('h') + pd.Timedelta('1h')
    s['z'] = np.where(s.family == 'Sensoterra', s.flag_zero_moisture, np.where(s.family == 'CloverSoilSensor', s.flag_gwc_censored_dry, np.nan))
    z = s.dropna(subset=['z']).groupby(['device_id', 'hour_end']).agg(z_s=('z', 'sum'), z_n=('z', 'size')).reset_index()
    b = base[['device_id', 'hour_end', 'family']].merge(a, on=['device_id', 'hour_end'], how='left').merge(z, on=['device_id', 'hour_end'], how='left')
    b[['cc_s', 'cc_n', 'z_s', 'z_n']] = b[['cc_s', 'cc_n', 'z_s', 'z_n']].fillna(0)
    g = b.groupby('device_id', sort=False); R = lambda c: g[c].transform(lambda x: x.rolling(24, min_periods=1).sum())
    out = pd.DataFrame(index=base.index)
    out['channel_completeness_24h'] = (R('cc_s') / R('cc_n').replace(0, np.nan)).values
    out['meas_zero_share_24h'] = (R('z_s') / R('z_n').replace(0, np.nan)).values
    out.loc[~base.family.isin(['Sensoterra', 'CloverSoilSensor']).values, 'meas_zero_share_24h'] = np.nan
    out.loc[~base.family.isin(['Dragino-SE0X-LB', 'Agvolution']).values, 'channel_completeness_24h'] = np.nan
    return out

def targets(H):
    """realized service availability over the next H hours (forward windows), with observability flag"""
    H = H.sort_values(['device_id', 'hour_end']).reset_index(drop=True)
    out = H[['device_id', 'hour_end']].copy()
    ok_hour = (H.hour_complete & H.data_observed).astype(int)
    for hz in sorted({h for v in M1_HORIZONS_H.values() for h in v}):
        f = lambda c: H.groupby('device_id', sort=False)[c].transform(lambda s: s[::-1].rolling(hz, min_periods=hz).sum()[::-1].shift(-1))
        e = f('svc_nominal_expected'); fl = f('svc_nominal_filled')
        okc = H.assign(o=ok_hour).groupby('device_id', sort=False).o.transform(lambda s: s[::-1].rolling(hz, min_periods=hz).sum()[::-1].shift(-1))
        out[f'y{hz}'] = np.where((okc == hz) & (e > 0), fl / e.replace(0, np.nan), np.nan)
        out[f'y{hz}_expected'] = e
        # did a declared-outage period cover any part of the window (analysis only, never a feature)
        io = H.in_declared_outage.astype(int)
        out[f'y{hz}_outage'] = H.assign(o=io).groupby('device_id', sort=False).o.transform(lambda s: s[::-1].rolling(hz, min_periods=1).max()[::-1].shift(-1))
    return out

def build(H, L, U, soil, analytic_tab):
    base = hourly_base(H)
    X = pd.concat([base, outage_history(L, base), radio(U, base), channels_and_measurements(U, soil, base)], axis=1)
    X = X.merge(analytic_tab.drop(columns=['family']), on=['device_id', 'hour_end'], how='left')
    Hs = H.sort_values(['device_id', 'hour_end']).reset_index(drop=True)
    X['eligible_for_prediction'] = Hs.eligible_for_prediction.values; X['in_declared_outage'] = Hs.in_declared_outage.values
    X['developing_silence'] = Hs.developing_silence.values; X['declared_scope'] = Hs.declared_scope.values
    X['data_observed'] = Hs.data_observed.values; X['service_started'] = Hs.service_started.values
    X['hour_complete'] = Hs.hour_complete.values
    X['svc_mode_meaning'] = Hs.svc_mode_meaning.values
    X = X.merge(targets(H), on=['device_id', 'hour_end'], how='left')
    return X
