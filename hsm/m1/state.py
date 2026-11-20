"""Module 1 communication state per device-hour, statistics and evidence.

State (priority order, all observable at the hour end):
  insufficient_information  outside the data table's observation window or in its final partial hour (receptions continue in other
                            exports, so 'heard but not stored' there is an export artefact), or never heard; also < 24 h of service
                            history, but only for the normal / degraded decision (outages and backend failures are detected from the start)
  backend_data_unavailable  data overdue while the device is still HEARD (uplinks received after the last stored data) or heard
                            but never stored: the sensor and radio work, the data path does not
  significant_interruption  inside a causally declared outage (scope at declaration: individual / gateway / fleet / network)
  reduced_reporting_mode    reception-inferred reporting mode differs from nominal (Sensoterra 24 h: documented causes poor pin
                            contact / too-dry soil)
  degraded                  forecast service availability over the family's primary horizon below the family threshold tau_f,
                            or a developing silence that is improbable under the device's own normal behaviour (S(k) < 0.01)
  normal                    otherwise
Traffic light: normal green, degraded / reduced mode yellow, interruption / backend red, insufficient grey."""
import numpy as np, pandas as pd
from ..config import NOMINAL_MIN, M1_HORIZONS_H, M1_PRIMARY_H, SLOT_GRACE_FRAC
from .models import tau

LIGHT = dict(normal='green', degraded='yellow', reduced_reporting_mode='yellow', significant_interruption='red',
             backend_data_unavailable='red', insufficient_information='grey')

def primary_h(fam): return M1_PRIMARY_H.get(fam, 24)

def tau_timeline(X):
    """family threshold for the primary horizon, re-estimated at every month start from windows that closed before it"""
    rows = []
    months = pd.date_range(X.hour_end.min().floor('D').replace(day=1), X.hour_end.max() + pd.offsets.MonthBegin(1), freq='MS')
    for fam, d in X.groupby('family'):
        H = primary_h(fam)
        ok = d[d.eligible_for_prediction & ~d.in_declared_outage & d[f'y{H}'].notna()]
        for m in months:
            tr = ok[ok.hour_end + pd.Timedelta(hours=H) <= m]
            rows.append(dict(family=fam, month=m, tau=tau(tr, H, fam), n_train=len(tr)))
    return pd.DataFrame(rows)

def timeline(X, P, schedule):
    """state per device-hour. Forecast = the family's selected candidate (walk-forward prediction) where available; otherwise
    (before the first fold, < 14 d history) the hierarchical EB-Markov forecast, which borrows strength from the family."""
    T = X[['device_id', 'family', 'hour_end', 'data_observed', 'hour_complete', 'service_started', 'history_days', 'in_declared_outage', 'declared_scope',
           'developing_silence', 'backend_suspect_now', 'heard_since_last_stored', 'reduced_mode', 'svc_mode_meaning', 'silence_svc_slots',
           'miss_run_slots', 'hmm_p_bad', 'hmm_p_next', 'hmm_silence_surprise', 'avail_svc_24h', 'avail_rx_24h', 'avail_valid_24h',
           'avail_mode_24h', 'avail_svc_672h', 'rx_minus_svc_24h', 'fleet_bad_share', 'meas_zero_share_24h', 'channel_completeness_24h']].copy()
    T['primary_H'] = T.family.map(primary_h)
    sel = []
    for fold, (choice, _) in schedule.items():                         # model choice in force in each fold (causal)
        for fam, H in [(f, primary_h(f)) for f in T.family.unique()]:
            c = choice.get((fam, H), ('eb_markov', 'default'))[0]
            p = P[(P.family == fam) & (P.H == H) & (P.fold == fold)][['device_id', 'hour_end', c]].rename(columns={c: 'forecast_avail'})
            p['forecast_model'] = c; sel.append(p)
    if sel: T = T.merge(pd.concat(sel, ignore_index=True), on=['device_id', 'hour_end'], how='left')
    else: T['forecast_avail'] = np.nan; T['forecast_model'] = None                # live mode without walk-forward predictions
    fb = np.select([T.primary_H == 24, T.primary_H == 72], [X.eb_E24.values, X.eb_E72.values if 'eb_E72' in X else np.nan], np.nan)
    use_fb = T.forecast_avail.isna() & ~T.in_declared_outage
    T.loc[use_fb, 'forecast_avail'] = fb[use_fb.values]; T.loc[use_fb & T.forecast_avail.notna(), 'forecast_model'] = 'eb_markov (fallback)'
    tt = tau_timeline(X); T['month'] = T.hour_end.dt.to_period('M').dt.to_timestamp()
    T = T.merge(tt[['family', 'month', 'tau']], on=['family', 'month'], how='left').drop(columns='month')
    T.loc[T.in_declared_outage, 'forecast_avail'] = np.nan
    n_slots = T.primary_H * 60 / T.family.map(NOMINAL_MIN)
    T['expected_missing_primary'] = (1 - T.forecast_avail) * n_slots
    unusual = T.developing_silence & (T.hmm_silence_surprise < 0.01)
    insuff = (~T.hour_complete) | (~T.service_started & ~T.heard_since_last_stored)
    short = T.history_days < 1                                        # too little history to judge normal vs degraded (not outages / backend)
    backend = T.backend_suspect_now | (~T.service_started & T.heard_since_last_stored)
    degraded = (T.forecast_avail < T.tau) | unusual
    T['state'] = np.select([insuff, backend, T.in_declared_outage, T.reduced_mode, short, degraded],
                           ['insufficient_information', 'backend_data_unavailable', 'significant_interruption', 'reduced_reporting_mode',
                            'insufficient_information', 'degraded'], 'normal')
    T['traffic_light'] = T.state.map(LIGHT)
    r = []
    for row in T[['forecast_avail', 'tau', 'developing_silence', 'hmm_silence_surprise', 'avail_valid_24h', 'avail_svc_24h', 'rx_minus_svc_24h',
                  'fleet_bad_share', 'meas_zero_share_24h', 'channel_completeness_24h']].itertuples(index=False):
        x = []
        if row.forecast_avail < row.tau: x.append('forecast_below_family_threshold')
        if row.developing_silence and row.hmm_silence_surprise < 0.01: x.append('improbable_silence')
        if row.avail_svc_24h > 0 and row.avail_valid_24h < 0.9 * row.avail_svc_24h: x.append('invalid_data_share_high')
        if row.rx_minus_svc_24h > 0.1: x.append('received_not_stored_24h')
        if row.fleet_bad_share >= 0.5: x.append('family_wide_problem')
        if row.meas_zero_share_24h >= 0.5: x.append('measurement_zero_majority')
        if row.channel_completeness_24h < 0.99: x.append('channels_missing')
        r.append(';'.join(x))
    T['reasons'] = r
    return T

def period_stats(H, U, L, freq):
    """packet loss and availability per device and period (D = day, W = week)"""
    h = H.copy(); h['period'] = (h.hour_end - pd.Timedelta('1ns')).dt.to_period(freq).dt.start_time
    g = h.groupby(['device_id', 'family', 'period'])
    S = g.agg(expected=('svc_nominal_expected', 'sum'), delivered=('svc_nominal_filled', 'sum'), valid=('svc_nominal_valid_filled', 'sum'),
              received=('rx_nominal_filled', 'sum'), rx_expected=('rx_nominal_expected', 'sum'), mode_expected=('svc_mode_expected', 'sum'),
              mode_delivered=('svc_mode_filled', 'sum'), excess=('svc_excess_reports', 'sum'), outage_hours=('in_declared_outage', 'sum'),
              max_silence_h=('svc_silence_min', 'max'), hours=('hour_end', 'size')).reset_index()
    S['max_silence_h'] = S.max_silence_h / 60
    S['missed_reports'] = S.expected - S.delivered
    for a, n, d in [('service_availability', 'delivered', 'expected'), ('valid_data_availability', 'valid', 'expected'),
                    ('reception_availability', 'received', 'rx_expected'), ('mode_availability', 'mode_delivered', 'mode_expected')]:
        S[a] = S[n] / S[d].replace(0, np.nan)
    S['packet_loss_service'] = 1 - S.service_availability; S['packet_loss_reception'] = 1 - S.reception_availability
    S['packet_loss_mode'] = 1 - S.mode_availability
    u = U.copy(); u['period'] = u.t.dt.to_period(freq).dt.start_time
    ua = u.groupby(['device_id', 'period']).agg(uplinks=('uplink_id', 'size'), heard_not_stored=('data_stored', lambda s: int((~s).sum())),
                                                stored_without_measurement=('stored_without_measurement', 'sum'),
                                                channel_completeness=('channel_completeness', 'mean')).reset_index()
    S = S.merge(ua, on=['device_id', 'period'], how='left')
    d = L[L.status == 'declared'].copy(); d['period'] = d.onset.dt.to_period(freq).dt.start_time
    S = S.merge(d.groupby(['device_id', 'period']).size().rename('outage_onsets').reset_index(), on=['device_id', 'period'], how='left')
    S['outage_onsets'] = S.outage_onsets.fillna(0).astype(int)
    return S

def summary(T, X, H, U, L):
    """one row per device: lifetime and trailing statistics, behaviour, outage history, current state and evidence"""
    # current = last hour inside the device's observation window (after it the state is only 'insufficient information')
    obs = T[T.hour_complete]
    last = obs.sort_values('hour_end').groupby('device_id').tail(1).set_index('device_id')
    xl = X[X.hour_complete].sort_values('hour_end').groupby('device_id').tail(1).set_index('device_id')
    tot = H.groupby('device_id').agg(family=('family', 'first'), first_hour=('hour_end', 'min'), last_hour=('hour_end', 'max'),
                                     expected=('svc_nominal_expected', 'sum'), delivered=('svc_nominal_filled', 'sum'),
                                     valid=('svc_nominal_valid_filled', 'sum'), received=('rx_nominal_filled', 'sum'),
                                     rx_expected=('rx_nominal_expected', 'sum'), mode_expected=('svc_mode_expected', 'sum'),
                                     mode_delivered=('svc_mode_filled', 'sum'), excess=('svc_excess_reports', 'sum'))
    tot['service_availability'] = tot.delivered / tot.expected; tot['valid_data_availability'] = tot.valid / tot.expected
    tot['reception_availability'] = tot.received / tot.rx_expected; tot['mode_availability'] = tot.mode_delivered / tot.mode_expected
    tot['missed_reports'] = tot.expected - tot.delivered
    ch = U[U.data_stored].groupby('device_id').agg(channel_completeness=('channel_completeness', 'mean'), share_cycles_all_channels=('data_valid', 'mean'))
    tot = tot.join(ch)
    st = T.groupby(['device_id', 'state']).size().unstack(fill_value=0); st = st.div(st.sum(axis=1), axis=0).add_prefix('share_hours_')
    d = L[L.status == 'declared']
    oc = d.groupby(['device_id', 'scope_at_declaration']).size().unstack(fill_value=0).add_prefix('outages_')
    cause = d.groupby('device_id').cause_hint.apply(lambda s: int(s.str.startswith('data not stored').sum())).rename('outages_backend_cause')
    dur = d.assign(h=(d.recovered_at - d.onset).dt.total_seconds() / 3600).groupby('device_id').h.agg(['median', 'max']).add_prefix('outage_duration_h_')
    beh = pd.DataFrame({'p_miss_28d': 1 - xl.d28 / xl.n28.replace(0, np.nan), 'p_miss_after_miss_28d': xl.n_MM / (xl.n_MM + xl.n_MD).replace(0, np.nan)})
    beh['burstiness_ratio'] = beh.p_miss_after_miss_28d / beh.p_miss_28d.replace(0, np.nan)
    beh['mean_loss_run_slots'] = 1 / (1 - beh.p_miss_after_miss_28d).replace(0, np.nan)
    ur = U[U.radio_metrics_available].groupby('device_id').agg(snr_median=('snr_max', 'median'), rssi_median=('rssi_max', 'median'),
                                                               share_beyond_sf12_reference=('flag_beyond_sf12_reference_limits', 'mean'))
    cur = last[['state', 'traffic_light', 'reasons', 'forecast_avail', 'forecast_model', 'tau', 'primary_H', 'expected_missing_primary',
                'hmm_p_bad', 'hmm_p_next', 'silence_svc_slots', 'declared_scope', 'svc_mode_meaning', 'avail_svc_24h', 'avail_rx_24h',
                'avail_valid_24h', 'avail_svc_672h', 'hour_end']].rename(columns={'hour_end': 'as_of'}).add_prefix('current_')
    S = tot.join([beh, oc, cause, dur, ur, st, cur], how='left')
    S['evidence'] = [evidence(r) for r in S.itertuples()]
    return S.reset_index()

def _f(x, fmt='{:.1%}'): return 'n/a' if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)

def evidence(r):
    e = [f"state {r.current_state} as of {r.current_as_of:%Y-%m-%d %H:%M}",
         f"service availability last 24 h {_f(r.current_avail_svc_24h)} (reception {_f(r.current_avail_rx_24h)}, valid {_f(r.current_avail_valid_24h)}), last 28 d {_f(r.current_avail_svc_672h)}"]
    if not np.isnan(r.current_forecast_avail):
        e.append(f"forecast next {int(r.current_primary_H)} h {_f(r.current_forecast_avail)} ({r.current_forecast_model}) vs family threshold {_f(r.current_tau)}; "
                 f"expected missing reports {r.current_expected_missing_primary:.1f}")
    if not np.isnan(r.current_hmm_p_bad): e.append(f"P(degraded link state) {_f(r.current_hmm_p_bad)}, P(next report) {_f(r.current_hmm_p_next)}")
    if r.current_silence_svc_slots >= 1.5: e.append(f"silent for {r.current_silence_svc_slots:.0f} expected reports")
    if isinstance(r.current_declared_scope, str): e.append(f"declared outage, scope at declaration: {r.current_declared_scope}")
    if isinstance(r.current_svc_mode_meaning, str): e.append(f"reporting mode: {r.current_svc_mode_meaning} (check probe contact / soil)")
    if r.current_reasons: e.append('flags: ' + r.current_reasons.replace(';', ', '))
    if not np.isnan(r.burstiness_ratio): e.append(f"losses are {r.burstiness_ratio:.1f}x more likely after a loss (mean loss run {r.mean_loss_run_slots:.1f} reports)")
    return '; '.join(e)
