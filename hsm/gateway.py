"""Gateway monitoring and failure-risk prediction (component 2).

Gateways are observed INDIRECTLY: a gateway site is the receiving location attached to sensor packets (causal site registry of the
foundation). Inactivity of a site does not prove hardware failure; the module separates explanations with the dependent sensors:

  expected activity mu_s(h)  = mean records per hour at site s over the previous 168 h (causal)
  P0                         = exp(-mu)  probability of an empty hour under Poisson activity
  dependents D_s(h)          = sensors whose primary site over the previous 168 h is s
  switched                   = dependents heard via ANOTHER site in hour h (gateway-specific loss: traffic re-routed)
  silent                     = dependents heard nowhere in hour h

hour condition of a monitored site (priority):
  network_or_platform   network-level hour (>= 70 % of active sites silent)
  inactive_gateway      0 records, P0 < alpha, and >= 1 dependent heard elsewhere or other sites active (site-specific)
  inactive_unexplained  0 records, P0 < alpha, dependents silent everywhere (gateway, sensors or backend: undetermined)
  reduced_activity      records < Poisson 0.1 % quantile of mu (partial loss)
  active / low_traffic  (low_traffic: mu < 1, not monitorable hour by hour)
gateway interruption event: >= 2 consecutive hours inactive_gateway / inactive_unexplained / network_or_platform."""
import numpy as np, pandas as pd
from scipy.stats import poisson
from .config import const, NETWORK_SILENT_SHARE

GW_ALPHA = const('GW_ALPHA', 0.01, 'operational', 'features|labels', 'empty hour is anomalous if P0 = exp(-mu) < 0.01 (mu >= 4.6 records/h)')
GW_MIN_RECORDS = const('GW_MIN_RECORDS', 1000, 'operational', 'features', 'sites with >= 1000 records are monitored; others are listed as sporadic')
GW_EVENT_MIN_H = const('GW_EVENT_MIN_H', 2, 'operational', 'labels', 'gateway interruption = >= 2 consecutive anomalous inactive hours')

def site_hours(R, sites_reg):
    r = R[R.gw_site.notna() & (R.gw_site != 'unresolved')].copy(); r['hour_end'] = r.t.dt.floor('h') + pd.Timedelta('1h')
    hrs = pd.date_range(r.hour_end.min(), r.hour_end.max(), freq='h')
    A = r.groupby(['gw_site', 'hour_end']).agg(records=('device_id', 'size'), devices=('device_id', 'nunique'), rssi_med=('rssi', 'median'),
                                               snr_med=('snr', 'median'), snr_low=('snr', lambda s: (s < -15).mean())).reset_index()
    big = A.groupby('gw_site').records.sum(); mon = big[big >= GW_MIN_RECORDS].index.tolist()
    full = pd.MultiIndex.from_product([mon, hrs], names=['gw_site', 'hour_end'])
    A = A.set_index(['gw_site', 'hour_end']).reindex(full).reset_index()
    A['records'] = A.records.fillna(0); A['devices'] = A.devices.fillna(0)
    g = A.groupby('gw_site', sort=False)
    A['mu'] = g.records.transform(lambda s: s.shift(1).rolling(168, min_periods=48).mean())
    A['dev_mu'] = g.devices.transform(lambda s: s.shift(1).rolling(168, min_periods=48).mean())
    # during an inactive run the baseline is frozen at its value when the run began (otherwise a dead gateway decays into 'low traffic')
    zero = A.records == 0
    run = (zero != zero.groupby(A.gw_site).shift()).groupby(A.gw_site).cumsum()
    A['mu_live'] = A.mu
    A['mu'] = np.where(zero, A.groupby([A.gw_site, run]).mu.transform('first'), A.mu)
    A['p0'] = np.exp(-A.mu)
    A['p_low'] = poisson.cdf(A.records, A.mu.fillna(0))
    # dependents and switching: primary site per device over the previous 168 h
    dh = r.groupby(['device_id', 'hour_end', 'gw_site']).size().rename('n').reset_index()
    tot = dh.pivot_table(index=['device_id', 'hour_end'], columns='gw_site', values='n', fill_value=0)
    out = []
    for dev, x in tot.groupby(level=0):
        x = x.droplevel(0).reindex(hrs, fill_value=0)
        roll = x.shift(1).rolling(168, min_periods=1).sum().fillna(0)
        prim = roll.idxmax(axis=1).where(roll.sum(axis=1) > 0)
        heard_any = x.sum(axis=1) > 0
        out.append(pd.DataFrame({'device_id': dev, 'hour_end': hrs, 'primary': prim.values, 'heard_any': heard_any.values,
                                 'heard_primary': [x.at[h, p] > 0 if isinstance(p, str) else False for h, p in zip(hrs, prim.values)],
                                 'primary_share_168h': (roll.max(axis=1) / roll.sum(axis=1).replace(0, np.nan)).values,
                                 'n_sites_672h': (x.shift(1).rolling(672, min_periods=1).sum() > 0).sum(axis=1).values}))
    Dp = pd.concat(out, ignore_index=True)
    dep = Dp.dropna(subset=['primary']).groupby(['primary', 'hour_end']).agg(
        dependents=('device_id', 'size'), dep_heard_elsewhere=('heard_any', lambda s: 0), dep_silent=('heard_any', lambda s: int((~s).sum())))
    Dp['elsewhere'] = Dp.heard_any & ~Dp.heard_primary
    dep['dep_heard_elsewhere'] = Dp.dropna(subset=['primary']).groupby(['primary', 'hour_end']).elsewhere.sum()
    dep = dep.reset_index().rename(columns={'primary': 'gw_site'})
    A = A.merge(dep, on=['gw_site', 'hour_end'], how='left')
    for c in ['dependents', 'dep_heard_elsewhere', 'dep_silent']: A[c] = A[c].fillna(0)
    others = A.assign(act=(A.records > 0).astype(int)).groupby('hour_end').act.transform('sum') - (A.records > 0).astype(int)
    A['other_sites_active'] = others
    return A, Dp

def classify(A, net):
    A = A.merge(net[['hour_end', 'network_level']], on='hour_end', how='left'); A['network_level'] = A.network_level.fillna(False)
    empty = (A.records == 0) & (A.p0 < GW_ALPHA)
    # evidence of re-routing seen so far within the current inactive run persists for the whole run (causal cumulative max)
    zero = A.records == 0; run = (zero != zero.groupby(A.gw_site).shift()).groupby(A.gw_site).cumsum()
    rer = A.groupby([A.gw_site, run]).dep_heard_elsewhere.cummax()
    A['rerouted_in_run'] = rer
    site_specific = (rer >= 1) | ((A.dep_silent < A.dependents) & (A.other_sites_active >= 1))
    low = (A.records > 0) & (A.p_low < 0.001) & (A.mu >= 5)
    A['condition'] = np.select([A.mu.isna(), A.network_level & (A.records == 0), empty & site_specific, empty, low, A.mu < 1],
                               ['insufficient_history', 'network_or_platform', 'inactive_gateway', 'inactive_unexplained', 'reduced_activity', 'low_traffic'], 'active')
    return A

def interruption_events(A):
    bad = A.condition.isin(['inactive_gateway', 'inactive_unexplained', 'network_or_platform'])
    rows = []
    for s, x in A.assign(bad=bad).groupby('gw_site'):
        x = x.sort_values('hour_end'); run = (x.bad != x.bad.shift()).cumsum()
        for _, y in x[x.bad].groupby(run[x.bad]):
            if len(y) >= GW_EVENT_MIN_H:
                kind = y.condition.mode().iat[0]
                rows.append(dict(gw_site=s, start=y.hour_end.min() - pd.Timedelta('1h'), end=y.hour_end.max(), hours=len(y), kind=kind,
                                 dependents=int(y.dependents.max()), dependents_rerouted=int(y.dep_heard_elsewhere.max()), dependents_silent=int(y.dep_silent.max()),
                                 detected_at=y.hour_end.iloc[GW_EVENT_MIN_H - 1]))
    return pd.DataFrame(rows)

def warning_signal_test(A, Ev, lead_h=24, n_perm=500, seed=0):
    """Is there a statistically meaningful warning pattern before gateway interruptions? For each candidate signal, compare its
    value in the lead_h hours before event starts (while still active) with all other active hours of the same site (AUC), and
    test against a site-stratified permutation of event times."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    A = A[A.condition.isin(['active', 'reduced_activity'])].copy()
    A['act_ratio_24h'] = A.groupby('gw_site').records.transform(lambda s: s.rolling(24, min_periods=6).mean()) / A.mu
    A['dev_ratio_24h'] = A.groupby('gw_site').devices.transform(lambda s: s.rolling(24, min_periods=6).mean()) / A.dev_mu
    A['snr_shift'] = A.snr_med - A.groupby('gw_site').snr_med.transform(lambda s: s.shift(1).rolling(168, min_periods=24).median())
    A['reduced'] = (A.condition == 'reduced_activity').astype(float)
    A['pre'] = False
    for r in Ev[Ev.kind != 'network_or_platform'].itertuples():
        m = (A.gw_site == r.gw_site) & (A.hour_end > r.start - pd.Timedelta(hours=lead_h)) & (A.hour_end <= r.start)
        A.loc[m, 'pre'] = True
    rows = []
    for sig, direction in [('act_ratio_24h', -1), ('dev_ratio_24h', -1), ('snr_shift', -1), ('reduced', 1)]:
        d = A.dropna(subset=[sig])
        if d.pre.sum() < 10: rows.append(dict(signal=sig, pre_hours=int(d.pre.sum()), auc=np.nan, p_value=np.nan)); continue
        auc = roc_auc_score(d.pre, direction * d[sig])
        perm = []
        for _ in range(n_perm):
            pp = d.groupby('gw_site').pre.transform(lambda s: np.roll(s.values, rng.integers(len(s))))
            perm.append(roc_auc_score(pp, direction * d[sig]) if pp.any() else 0.5)
        rows.append(dict(signal=sig, pre_hours=int(d.pre.sum()), auc=auc, p_value=(np.sum(np.array(perm) >= auc) + 1) / (n_perm + 1)))
    return pd.DataFrame(rows)

def interruption_rate(Ev, A, as_of):
    """per-site interruption rate (gamma-Poisson, prior = 1 week at the pooled rate) and P(interruption within 24 h / 7 d)"""
    from scipy.stats import gamma
    ex = A[A.condition.isin(['active', 'reduced_activity'])].groupby('gw_site').size().astype(float)
    k = Ev[(Ev.kind != 'network_or_platform') & (Ev.start <= as_of)].groupby('gw_site').size().reindex(ex.index, fill_value=0)
    lam0 = k.sum() / ex.sum(); a0 = 168 * lam0 + 1e-3; b0 = 168.0
    a = a0 + k; b = b0 + ex
    out = pd.DataFrame({'events': k, 'exposure_h': ex, 'rate_per_week': 168 * a / b, 'rate_lo90': 168 * gamma.ppf(0.05, a, scale=1 / b),
                        'rate_hi90': 168 * gamma.ppf(0.95, a, scale=1 / b)})
    out['p_interruption_24h'] = 1 - np.exp(-24 * a / b); out['p_interruption_7d'] = 1 - np.exp(-168 * a / b)
    return out.reset_index()

def sensor_dependence(Dp, as_of):
    """per sensor: primary gateway, dependence share, alternative sites seen in the last 28 days"""
    last = Dp[Dp.hour_end <= as_of].sort_values('hour_end').groupby('device_id').tail(1)
    return last[['device_id', 'primary', 'primary_share_168h', 'n_sites_672h']].rename(columns={'primary': 'primary_gateway'}).assign(
        single_gateway_dependent=lambda d: (d.n_sites_672h <= 1) | (d.primary_share_168h >= 0.98))

def alternatives(R, as_of, days=28, min_share=0.05):
    r = R[(R.t > as_of - pd.Timedelta(days=days)) & (R.t <= as_of) & R.gw_site.notna() & (R.gw_site != 'unresolved')]
    sh = r.groupby(['device_id', 'gw_site']).size() / r.groupby('device_id').size()
    sh = sh[sh >= min_share].reset_index(name='share')
    return sh.sort_values(['device_id', 'share'], ascending=[True, False])

def current(A, Ev, rates, sites, as_of, Dp):
    last = A[A.hour_end <= as_of].sort_values('hour_end').groupby('gw_site').tail(1).set_index('gw_site')
    act = A[(A.records > 0) & (A.hour_end <= as_of)].groupby('gw_site').hour_end.max().rename('last_activity')
    h28 = A[(A.hour_end > as_of - pd.Timedelta(days=28)) & (A.hour_end <= as_of)]
    rel = h28.groupby('gw_site').apply(lambda x: (x.records > 0).mean()).rename('active_hour_share_28d')
    dev28 = h28.groupby('gw_site').devices.mean().rename('mean_devices_per_hour_28d')
    rs = h28.groupby('gw_site').agg(rssi_median_28d=('rssi_med', 'median'), snr_median_28d=('snr_med', 'median'))
    rs_prev = A[(A.hour_end > as_of - pd.Timedelta(days=56)) & (A.hour_end <= as_of - pd.Timedelta(days=28))].groupby('gw_site').snr_med.median().rename('snr_median_prev28d')
    deps = Dp[(Dp.hour_end == as_of)].groupby('primary').device_id.apply(list).rename('dependent_sensors')
    C = last[['condition', 'records', 'mu', 'dependents', 'dep_heard_elsewhere', 'dep_silent']].join([act, rel, dev28, rs, rs_prev, deps])
    C = C.join(rates.set_index('gw_site')).join(sites.set_index('site')[['lat', 'lon', 'official']])
    n_ev = Ev[Ev.start <= as_of].groupby('gw_site').size().rename('interruptions_total'); C = C.join(n_ev)
    C['hours_since_activity'] = (as_of - C.last_activity).dt.total_seconds() / 3600
    C['probable_cause'] = np.select([C.condition == 'network_or_platform', C.condition == 'inactive_gateway', C.condition == 'inactive_unexplained',
                                     C.condition == 'reduced_activity'],
                                    ['network or platform interruption (most sites silent)', 'gateway-specific: dependent sensors re-routed via other gateways',
                                     'undetermined: gateway and its dependent sensors silent together (gateway, power, backhaul or sensors)',
                                     'partial loss of traffic (interference, antenna or backhaul degradation, or fewer active sensors)'], 'none')
    C['recommended_action'] = np.select([C.condition == 'network_or_platform', C.condition == 'inactive_gateway', C.condition == 'inactive_unexplained',
                                         C.condition == 'reduced_activity'],
                                        ['check network server / platform and backhaul; no site visit until platform is ruled out',
                                         'check gateway power, backhaul and packet forwarder; dependent sensors currently covered by alternatives',
                                         'check gateway power and backhaul first, then the dependent sensors',
                                         'review gateway logs and antenna; monitor'], 'none')
    return C.reset_index().rename(columns={'index': 'gw_site'})
