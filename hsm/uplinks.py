"""Uplinks: one row per transmission, assembled from all three exports. Records of the same device within MERGE_WINDOW_S
belong to one uplink. Every record is linked (uplink_records) so the evidence for each uplink stays traceable.
Gateway sites are assigned causally (a site is only known once enough records have been seen there)."""
import numpy as np, pandas as pd
from .config import MERGE_WINDOW_S, SITE_RADIUS_M, SITE_PROMOTE_MIN_RECORDS

def hav(lat1, lon1, lat2, lon2):
    p = np.radians
    return 2 * 6371000 * np.arcsin(np.sqrt(np.sin(p(lat2 - lat1) / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(p(lon2 - lon1) / 2) ** 2))

def causal_sites(records, official):
    """records: radio records with loc_lat/loc_lon (receiving gateway), sorted by time.
    Official sites known from the start; unlisted locations become sites after SITE_PROMOTE_MIN_RECORDS records within the radius.
    Returns site_at_time (causal) and site_retro (assignment with the final registry, descriptive only)."""
    sites = [dict(site=r.name, lat=r.lat, lon=r.lon, official=True, first_seen=pd.NaT) for r in official.itertuples()]
    # merge official entries with identical / near-identical coordinates (duplicate names in the supplied list)
    merged = []
    for s in sites:
        m = next((x for x in merged if hav(x['lat'], x['lon'], s['lat'], s['lon']) < 30), None)
        if m: m['site'] = m['site'] + '/' + s['site']
        else: merged.append(dict(s))
    sites = merged
    cand = []                       # candidate clusters: lat, lon, n, first_seen
    lat = records.loc_lat.values; lon = records.loc_lon.values; tt = records.t.values
    at_time = np.empty(len(records), dtype=object)
    for i in range(len(records)):
        if np.isnan(lat[i]): at_time[i] = None; continue
        S = np.array([[s['lat'], s['lon']] for s in sites])
        d = hav(S[:, 0], S[:, 1], lat[i], lon[i]); j = int(np.argmin(d))
        if d[j] <= SITE_RADIUS_M: at_time[i] = sites[j]['site']; continue
        at_time[i] = 'unresolved'
        if cand:
            Cc = np.array([[c['lat'], c['lon']] for c in cand]); dc = hav(Cc[:, 0], Cc[:, 1], lat[i], lon[i]); k = int(np.argmin(dc))
        if cand and dc[k] <= SITE_RADIUS_M:
            c = cand[k]; c['n'] += 1
            if c['n'] >= SITE_PROMOTE_MIN_RECORDS:
                sites.append(dict(site=f"unlisted_{c['lat']:.4f}_{c['lon']:.4f}", lat=c['lat'], lon=c['lon'], official=False, first_seen=pd.Timestamp(tt[i])))
                cand.pop(k)
        else:
            cand.append(dict(lat=lat[i], lon=lon[i], n=1, first_seen=tt[i]))
    S = np.array([[s['lat'], s['lon']] for s in sites]); retro = np.empty(len(records), dtype=object)
    ok = ~np.isnan(lat)
    D = np.stack([hav(S[k, 0], S[k, 1], lat[ok], lon[ok]) for k in range(len(S))], 1)
    r = np.array([s['site'] for s in sites], dtype=object)[D.argmin(1)]; r[D.min(1) > SITE_RADIUS_M] = 'unresolved'
    retro[ok] = r
    return at_time, retro, pd.DataFrame(sites)

def _span_merge(dev, t, window_s):
    """uplink sequence number: a record starts a new uplink when it is more than window_s after the FIRST record of the current
    uplink (span-anchored, no chaining). t sorted within device."""
    ts = t.astype('datetime64[ns]').astype('int64') / 1e9; seq = np.zeros(len(t), int)
    cur = -1; first = None; prev_dev = None
    for i in range(len(t)):
        if dev[i] != prev_dev or ts[i] - first > window_s:
            cur = cur + 1 if dev[i] == prev_dev else 0; first = ts[i]; prev_dev = dev[i]
        seq[i] = cur
    return seq

def build(dev_s, soil_s, wx_s, official):
    """returns uplinks, uplink_records, site registry, radio records.
    t            : first record of the uplink (time the system first knew of it)  -> reception ledger
    t_data       : first STORED MEASUREMENT record in the family's data table      -> service ledger (time the data became available)
    data_stored  : >= 1 stored record with measurement content in the data table (a stored row without any measured value is not data)
    channels     : measurement channels (Dragino ch1..ch4, Agvolution profile / cap_air, otherwise main). A channel becomes EXPECTED
                   for a device once it has appeared in an earlier stored uplink (causal registry).
    data_valid   : every expected channel stored with a valid record (strict); data_valid_any: at least one valid channel."""
    from .config import DATA_SOURCE, RADIO_LIMITS as R, RADIO_METRICS
    cols = ['raw_row_id', 'source', 'device_id', 'family', 't']
    dv = dev_s[cols + ['rssi', 'snr', 'loc_lat', 'loc_lon', 'loc_role', 'dup_rank', 'has_measurement', 'meas_channel']].copy()
    dv['stored'] = True; dv['valid'] = ~dev_s.flag_battery_error_code.values
    so = soil_s[cols + ['dup_rank', 'has_measurement', 'meas_channel']].copy(); so['stored'] = soil_s.data_stored.values
    so['valid'] = so.stored & ~(soil_s.flag_soil_temp_out_of_spec | soil_s.flag_rawfreq_fault | soil_s.flag_temp_stuck_sofar).values
    wx = wx_s[wx_s.family != 'OpenWeatherMap']; wxr = wx[cols + ['dup_rank', 'has_measurement', 'meas_channel']].copy()
    wxr['stored'] = True; wxr['valid'] = ~wx.flag_bad_packet.values
    recs = pd.concat([dv, so, wxr], ignore_index=True)
    recs['is_data_source'] = recs.source.values == recs.family.map(DATA_SOURCE).values
    recs.loc[recs.rssi > 0, 'rssi'] = np.nan                          # sentinel 127 (flag kept in silver)
    recs = recs.sort_values(['device_id', 't', 'source']).reset_index(drop=True)
    recs['uplink_seq'] = _span_merge(recs.device_id.values, recs.t.values, MERGE_WINDOW_S)
    recs['uplink_id'] = recs.device_id + '#' + (recs.uplink_seq + 1).astype(str)
    recs['is_data_record'] = recs.is_data_source & recs.stored & recs.has_measurement.fillna(False).astype(bool)
    radio = recs[(recs.loc_role == 'receiving_gateway')].sort_values('t')
    at_time, retro, sites = causal_sites(radio, official)
    recs.loc[radio.index, 'gw_site'] = at_time; recs.loc[radio.index, 'gw_site_retro'] = retro
    first_dup = recs.dup_rank == 0
    g = recs.groupby('uplink_id')
    U = pd.DataFrame({
        'device_id': g.device_id.first(), 'family': g.family.agg(lambda s: s.mode().iat[0]), 't': g.t.min(), 't_last_record': g.t.max(),
        'n_records': g.size(), 'n_records_unique': recs[first_dup].groupby('uplink_id').size(),
        'sources': g.source.agg(lambda s: '+'.join(sorted(set(s)))),
        'n_radio_records': g.rssi.count(), 'rssi': g.rssi.first(), 'snr': g.snr.first(), 'rssi_max': g.rssi.max(), 'snr_max': g.snr.max(),
        'gw_site': g.gw_site.first(), 'n_gw_sites': g.gw_site.nunique(), 'gw_site_retro': g.gw_site_retro.first()}).reset_index()
    ds = recs[recs.is_data_source]; dr = recs[recs.is_data_record]
    U = U.join(dr.groupby('uplink_id').t.min().rename('t_data'), on='uplink_id')
    U['data_stored'] = U.t_data.notna()
    U['data_record_received'] = U.uplink_id.isin(ds.uplink_id)                 # any data-table row, stored or failed, with or without values
    U['stored_without_measurement'] = U.uplink_id.isin(ds[ds.stored].uplink_id) & ~U.data_stored
    # channel registry (causal): channel expected for an uplink if it appeared in an EARLIER stored uplink of the device
    ch = dr.groupby(['uplink_id', 'meas_channel']).valid.any().reset_index()
    ch = ch.merge(U[['uplink_id', 'device_id', 't']], on='uplink_id')
    first_seen = ch.groupby(['device_id', 'meas_channel']).t.min().reset_index()
    fs = {d: np.sort(x.t.values) for d, x in first_seen.groupby('device_id')}
    U = U.sort_values(['device_id', 't']).reset_index(drop=True)
    n_prev = np.zeros(len(U), int)
    for d, idx in U.groupby('device_id').indices.items():
        if d in fs: n_prev[idx] = np.searchsorted(fs[d], U.t.values[idx], side='left')      # channels first seen strictly earlier
    U['n_channels_seen_before'] = n_prev
    per = ch.groupby('uplink_id').agg(n_channels_stored=('meas_channel', 'size'), n_channels_valid=('valid', 'sum'))
    new = ch.merge(first_seen, on=['device_id', 'meas_channel'], suffixes=('', '_first'))
    new = new[new.t == new.t_first].groupby('uplink_id').size().rename('n_channels_new')
    U = U.join(per, on='uplink_id').join(new, on='uplink_id')
    for c in ['n_channels_stored', 'n_channels_valid', 'n_channels_new']: U[c] = U[c].fillna(0).astype(int)
    U['n_channels_expected'] = np.where(U.data_stored, U.n_channels_seen_before + U.n_channels_new, U.n_channels_seen_before)
    U['channel_completeness'] = np.where(U.n_channels_expected > 0, U.n_channels_stored / U.n_channels_expected.clip(lower=1), np.nan)
    U['data_valid'] = U.data_stored & (U.n_channels_valid == U.n_channels_expected) & (U.n_channels_expected > 0)
    U['data_valid_any'] = U.data_stored & (U.n_channels_valid > 0)
    U['n_records_unique'] = U.n_records_unique.fillna(0).astype(int)
    U['repeat_reception'] = U.n_radio_records > 1
    # radio reference limits (no SF in the exports: these are fixed LoRa reference values, NOT a link-margin estimate)
    U['radio_metrics_available'] = ~U.family.isin(list(RADIO_METRICS))
    U['flag_snr_below_sf12_reference_floor'] = U.snr_max < R['snr_floor_sf12_db']
    U['flag_rssi_below_sf12_reference_sensitivity'] = U.rssi_max < R['sensitivity_sf12_dbm']
    U['flag_beyond_sf12_reference_limits'] = U.flag_snr_below_sf12_reference_floor | U.flag_rssi_below_sf12_reference_sensitivity
    U['flag_rssi_near_saturation'] = U.rssi_max >= R['saturation_dbm']
    links = recs[['uplink_id', 'raw_row_id', 'source', 't', 'dup_rank', 'gw_site', 'gw_site_retro', 'stored', 'valid', 'is_data_source',
                  'is_data_record', 'meas_channel']]
    radio_records = recs.loc[radio.index, ['device_id', 'family', 't', 'gw_site', 'gw_site_retro', 'rssi', 'snr']]
    return U, links, sites, radio_records
