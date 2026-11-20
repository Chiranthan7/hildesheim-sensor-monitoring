"""Causal reporting-mode inference and the bounded slot ledger.

Two schedules are tracked for every device:
  service schedule  = declared nominal interval (what users expect).  SERVICE availability = filled / expected service slots.
  mode schedule     = reporting mode inferred causally from the arrivals so far (e.g. Sensoterra 24 h mode).
                      MODE availability = filled / expected mode slots (link reliability given what the device is trying to send).
A slot is 'known missed' once its due time + grace has passed; an arrival within the grace window of the previous filled slot
is an EXCESS report (tracked separately, never counted as availability). Both availabilities are therefore bounded in [0, 1]."""
import numpy as np, pandas as pd
from .config import NOMINAL_MIN, ALT_MODES_MIN, MODE_ENTER_CONSECUTIVE, MODE_TOLERANCE_FRAC, MODE_EXIT_FACTOR, SLOT_GRACE_FRAC

def infer_device(t, fam, obs_end, mode_given=None):
    """t: sorted arrival times (np.datetime64) of one device. Returns per-arrival records and gap ledger (incl. open tail).
    mode_given: optional reporting mode in force at each arrival (taken from the reception ledger). The reporting mode is a property
    of what the DEVICE sends, so the service ledger uses the mode inferred from reception; storage failures cannot fake a mode."""
    m0 = NOMINAL_MIN[fam]; alts = ALT_MODES_MIN.get(fam, [])
    g = SLOT_GRACE_FRAC
    tm = (t - t[0]).astype('timedelta64[s]').astype(float) / 60          # minutes since first arrival
    n = len(t); excess = np.zeros(n, bool); mode_at = np.full(n, float(m0))
    if mode_given is not None: mode_at[0] = float(mode_given[0])
    gaps = []                                                           # anchor_idx, next_idx, dt, mode_interval at gap start
    anchor = 0; mode = float(m0) if mode_given is None else float(mode_given[0]); cnt = {a: 0 for a in alts}
    for i in range(1, n):
        dt = tm[i] - tm[anchor]
        if dt < g * m0:                                                 # within grace of previous filled slot -> excess
            excess[i] = True; mode_at[i] = mode; continue
        gaps.append((anchor, i, dt, mode))
        if mode_given is not None:
            mode = float(mode_given[i]); mode_at[i] = mode; anchor = i; continue
        # causal mode update once this interval has completed
        hit = [a for a in alts if abs(dt - a) <= MODE_TOLERANCE_FRAC * a]
        if hit:
            a = hit[0]; cnt = {k: (cnt[k] + 1 if k == a else 0) for k in cnt}
            if cnt[a] >= MODE_ENTER_CONSECUTIVE: mode = float(a)
        elif dt <= MODE_EXIT_FACTOR * m0:
            mode = float(m0); cnt = {k: 0 for k in cnt}
        else:
            cnt = {k: 0 for k in cnt}
        mode_at[i] = mode; anchor = i
    tail_dt = (obs_end - t[anchor]).astype('timedelta64[s]').astype(float) / 60
    G = pd.DataFrame(gaps, columns=['anchor_idx', 'next_idx', 'dt_min', 'mode_min'])
    G['open'] = False
    G = pd.concat([G, pd.DataFrame([dict(anchor_idx=anchor, next_idx=-1, dt_min=tail_dt, mode_min=mode, open=True)])], ignore_index=True)
    G['anchor_t'] = t[G.anchor_idx.values]
    G['next_t'] = np.where(G.open, np.datetime64('NaT'), t[np.maximum(G.next_idx.values, 0)])
    # slots: missed slots j=1..k-1 with deadline anchor + (j+g)*interval; closed gaps fill one slot at arrival
    for sch, col in [('svc', None), ('mode', 'mode_min')]:
        iv = np.full(len(G), float(m0)) if sch == 'svc' else G.mode_min.values
        k_closed = np.maximum(np.floor(G.dt_min.values / iv - g) + 1, 1)          # = round(dt/iv) for closed gaps
        missed_closed = k_closed - 1
        missed_open = np.maximum(np.floor(G.dt_min.values / iv - g), 0)          # deadlines passed by obs_end
        G[f'{sch}_missed'] = np.where(G.open, missed_open, missed_closed).astype(int)
        G[f'{sch}_filled'] = np.where(G.open, 0, 1).astype(int)
        G[f'{sch}_interval_min'] = iv
    return excess, mode_at, G

def slot_events(G, sch):
    """expand gap ledger into slot events (time when each slot is resolved: arrival for filled, deadline for missed)"""
    iv = G[f'{sch}_interval_min'].values; k = G[f'{sch}_missed'].values
    rep = np.repeat(np.arange(len(G)), k)
    j = np.concatenate([np.arange(1, kk + 1) for kk in k]) if k.sum() else np.array([], int)
    miss_t = G.anchor_t.values[rep] + ((j + SLOT_GRACE_FRAC) * iv[rep] * 60).astype('timedelta64[s]')
    fill = ~G.open.values
    ev = pd.concat([pd.DataFrame({'t': miss_t, 'filled': 0, 'gap': rep}),
                    pd.DataFrame({'t': G.next_t.values[fill], 'filled': 1, 'gap': np.arange(len(G))[fill]})], ignore_index=True)
    return ev.sort_values('t').reset_index(drop=True)

def _hourly(G, t_arr, excess, mode_at, hrs, prefix, valid=None):
    """slot counts resolved in each hour (label = hour END) for both schedules of one ledger"""
    out = {}
    for sch, name in [('svc', 'nominal'), ('mode', 'mode')]:
        ev = slot_events(G, sch); ev['h'] = ev.t.dt.floor('h') + pd.Timedelta('1h')
        a = ev.groupby('h').filled.agg(['size', 'sum']).reindex(hrs, fill_value=0)
        out[f'{prefix}_{name}_expected'] = a['size'].values; out[f'{prefix}_{name}_filled'] = a['sum'].values
        if valid is not None and sch == 'svc':
            arr = t_arr[~excess]; vv = valid[~excess]
            fe = ev[ev.filled == 1].copy(); idx = np.searchsorted(arr, fe.t.values.astype('datetime64[ns]'))
            fe['valid'] = vv[np.clip(idx, 0, len(vv) - 1)]
            out[f'{prefix}_nominal_valid_filled'] = fe.groupby('h').valid.sum().reindex(hrs, fill_value=0).values
    ex = pd.Series(1, index=pd.DatetimeIndex(t_arr[excess]).floor('h') + pd.Timedelta('1h')).groupby(level=0).sum()
    out[f'{prefix}_excess_reports'] = ex.reindex(hrs, fill_value=0).values
    arr = t_arr[~excess]; k = np.searchsorted(arr, hrs.values, side='left') - 1   # strictly before hour end: same convention as slot bins [h-1, h)
    last = np.where(k >= 0, arr[np.maximum(k, 0)], np.datetime64('NaT'))
    out[f'{prefix}_last_arrival'] = last
    out[f'{prefix}_mode_min'] = np.where(k >= 0, mode_at[~excess][np.maximum(k, 0)], np.nan)
    out[f'{prefix}_silence_min'] = (hrs.values - last).astype('timedelta64[s]').astype(float) / 60
    return out

def build(U, obs_end_rx, obs_end_data):
    """Two ledgers per device:
      service ledger  : arrivals that delivered STORED data (data_stored) -> service availability, outage declarations
      reception ledger: every arrival (any record, incl. failed inserts / radio metadata) -> reception availability, cause attribution
    Returns uplinks (with excess / mode columns for both), service gap ledger, reception gap ledger, hourly table."""
    U = U.sort_values(['device_id', 't']).reset_index(drop=True)
    for c in ['rx_excess', 'svc_excess']: U[c] = False
    U['rx_mode_min'] = np.nan; U['svc_mode_min'] = np.nan
    LS, LR, HH = [], [], []
    for dev, idx in U.groupby('device_id').indices.items():
        x = U.iloc[idx]; fam = x.family.iat[0]
        if fam not in NOMINAL_MIN: continue
        t_all = x.t.values.astype('datetime64[ns]')
        e_rx, m_rx, G_rx = infer_device(t_all, fam, np.datetime64(obs_end_rx[fam], 'ns'))
        U.loc[U.index[idx], 'rx_excess'] = e_rx; U.loc[U.index[idx], 'rx_mode_min'] = m_rx
        G_rx.insert(0, 'device_id', dev); G_rx.insert(1, 'family', fam); G_rx['gap_id'] = dev + '/rx/' + G_rx.index.astype(str); LR.append(G_rx)
        last_h = max(pd.Timestamp(obs_end_rx[fam]).ceil('h'), pd.Timestamp(obs_end_data[fam]).ceil('h'))   # include final partial hour
        hrs = pd.date_range(pd.Timestamp(t_all[0]).floor('h') + pd.Timedelta('1h'), last_h, freq='h')
        if len(hrs) == 0: continue
        H = pd.DataFrame(index=hrs); H.index.name = 'hour_end'
        for k_, v_ in _hourly(G_rx, t_all, e_rx, m_rx, hrs, 'rx').items(): H[k_] = v_
        ds = x.data_stored.values
        if ds.any():
            t_d = x.t_data.values[ds].astype('datetime64[ns]'); oe = np.datetime64(obs_end_data[fam], 'ns')
            o = np.argsort(t_d, kind='stable'); assert (o == np.arange(len(o))).all(), 'service arrivals out of order'
            kk = np.searchsorted(t_all, t_d, side='right') - 1                  # reception mode in force at the data arrival
            e_s, m_s, G_s = infer_device(t_d, fam, oe, mode_given=m_rx[kk])
            U.loc[U.index[idx[ds]], 'svc_excess'] = e_s; U.loc[U.index[idx[ds]], 'svc_mode_min'] = m_s
            G_s.insert(0, 'device_id', dev); G_s.insert(1, 'family', fam); G_s['gap_id'] = dev + '/svc/' + G_s.index.astype(str); LS.append(G_s)
            hs = hrs[(hrs > pd.Timestamp(t_d[0])) & (hrs <= pd.Timestamp(obs_end_data[fam]).ceil('h'))]
            sv = _hourly(G_s, t_d, e_s, m_s, hs, 'svc', valid=x.data_valid.values[ds])
            for k_, v_ in sv.items(): H.loc[hs, k_] = v_
        H['data_observed'] = H.index <= pd.Timestamp(obs_end_data[fam]).ceil('h')
        H['hour_complete'] = H.index <= pd.Timestamp(obs_end_data[fam])          # False for the final partial hour (and after)
        H = H.reset_index(); H.insert(0, 'device_id', dev); H.insert(1, 'family', fam); HH.append(H)
    Ls = pd.concat(LS, ignore_index=True); Lr = pd.concat(LR, ignore_index=True); Hh = pd.concat(HH, ignore_index=True)
    for c in ['svc_nominal_expected', 'svc_nominal_filled', 'svc_mode_expected', 'svc_mode_filled', 'svc_nominal_valid_filled', 'svc_excess_reports']:
        Hh[c] = Hh[c].fillna(0).astype(int)
    Hh['service_availability'] = Hh.svc_nominal_filled / Hh.svc_nominal_expected.replace(0, np.nan)
    Hh['valid_data_availability'] = Hh.svc_nominal_valid_filled / Hh.svc_nominal_expected.replace(0, np.nan)
    Hh['mode_availability'] = Hh.svc_mode_filled / Hh.svc_mode_expected.replace(0, np.nan)
    Hh['reception_availability'] = Hh.rx_nominal_filled / Hh.rx_nominal_expected.replace(0, np.nan)
    return U, Ls, Lr, Hh
