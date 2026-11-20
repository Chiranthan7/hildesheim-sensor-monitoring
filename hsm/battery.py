"""Battery health and degradation forecasting (component 3), specific to chemistry and telemetry.

12 V solar lead-acid (Bigbelly: sealed lead-acid per manufacturer; Ott: 12 V solar system, chemistry not identified):
  daily resting-voltage proxy r_d = minimum voltage of day d (night, no charging); charge swing = daily max - min.
  forecast r_{d+k}: persistence (7-day median), Theil-Sen trend over 28 days, local-linear-trend state-space model (Kalman, 56 days).
  backtest: weekly origins; origins before the test start choose the model per family, later origins report the error.
  threshold risk: P(r_{d+k} < v) = Phi((v - mu_k)/sigma_k) from the chosen model, max over k <= 30 days; crossing range = first days
  with P >= 0.1 / 0.5 / 0.9. Thresholds: generic lead-acid resting-voltage references (config.LEAD_ACID_REST_V).
3.6 V lithium (Dragino Li-SOCl2, Adeunis, Agvolution solar node): flat discharge plateau -> no threshold-crossing forecast; report
  level, Theil-Sen slope (mV/week, 90 % CI) and robust step changes (daily median outside long-run median +- 4 MAD).
CLOVER: manufacturer battery-level estimate (%): Theil-Sen slope with CI; no crossing date unless the decline is significant.
Sensoterra, MeteoHelix, MeteoWind: no battery telemetry -> 'not measurable'."""
import numpy as np, pandas as pd, warnings
from scipy.stats import theilslopes, norm
from .config import BATTERY_OBSERVABILITY, LEAD_ACID_REST_V, BATTERY_FORECAST_DAYS, BATTERY_RISK_DAYS, M1_FOLDS
warnings.filterwarnings('ignore')
LEAD_ACID = ['OttSensor', 'Bigbelly']; LITHIUM = ['Dragino-SE0X-LB', 'Adeunis', 'Agvolution']

def daily(devs):
    d = devs[(devs.dup_rank == 0)].copy(); d['day'] = d.t.dt.floor('D')
    v = d[d.battery_v.notna()].groupby(['device_id', 'family', 'day']).battery_v.agg(v_min='min', v_max='max', v_med='median', n='size').reset_index()
    v['swing'] = v.v_max - v.v_min
    t = d[d.device_temp_c_clean.notna()].groupby(['device_id', 'day']).device_temp_c_clean.agg(temp_min='min', temp_mean='mean').reset_index()
    lv = d[d.battery_level_pct.notna()].groupby(['device_id', 'family', 'day']).battery_level_pct.mean().rename('level_pct').reset_index()
    return v.merge(t, on=['device_id', 'day'], how='left'), lv

def _ts(y):
    x = np.arange(len(y), dtype=float); m = ~np.isnan(y)
    if m.sum() < 7: return np.nan, np.nan, np.nan, np.nan
    s, i, lo, hi = theilslopes(y[m], x[m], 0.90); return s, i, lo, hi

def forecast(series, k, method):
    """series: daily resting proxy up to the origin (index = days). returns (mean, sd) for day origin + k"""
    y = series.values.astype(float)
    if method == 'persistence':
        mu = np.nanmedian(y[-7:]); r = y[-28:] - pd.Series(y[-28:]).rolling(7, min_periods=3).median().values
        return mu, max(np.nanstd(r) * np.sqrt(1 + k / 7), 0.02)
    if method == 'theil_sen':
        w = y[-28:]; s, i, lo, hi = _ts(w)
        if np.isnan(s): return np.nan, np.nan
        x0 = len(w) - 1; mu = i + s * (x0 + k)
        res = w - (i + s * np.arange(len(w))); sd = np.sqrt(np.nanvar(res) + ((hi - lo) / 3.29 * k) ** 2)
        return mu, max(sd, 0.02)
    if method == 'local_linear_trend':
        from statsmodels.tsa.statespace.structural import UnobservedComponents
        w = pd.Series(y[-56:]).interpolate(limit_direction='both').values
        if len(w) < 21: return np.nan, np.nan
        try:
            r = UnobservedComponents(w, level='local linear trend').fit(disp=False, maxiter=200)
            f = r.get_forecast(k); return float(f.predicted_mean[-1]), float(max(np.sqrt(f.var_pred_mean[-1]), 0.02))
        except Exception: return np.nan, np.nan
    raise ValueError(method)

METHODS = ['persistence', 'theil_sen', 'local_linear_trend']

def backtest(V):
    rows = []
    for dev, d in V[V.family.isin(LEAD_ACID)].groupby('device_id'):
        s = d.set_index('day').v_min.asfreq('D')
        days = s.index
        for o in days[28::7]:
            hist = s[:o]
            for k in BATTERY_FORECAST_DAYS:
                tgt = o + pd.Timedelta(days=k)
                if tgt not in s.index or np.isnan(s[tgt]): continue
                for m in METHODS:
                    mu, sd = forecast(hist, k, m)
                    if np.isnan(mu): continue
                    rows.append(dict(device_id=dev, family=d.family.iat[0], origin=o, k=k, method=m, pred=mu, sd=sd, actual=s[tgt],
                                     covered80=abs(s[tgt] - mu) <= 1.2816 * sd))
    B = pd.DataFrame(rows); B['abs_err'] = (B.pred - B.actual).abs()
    B['role'] = np.where(B.origin + pd.to_timedelta(B.k, unit='D') <= pd.Timestamp(M1_FOLDS['test'][0]), 'selection', 'test')
    return B

def choose(B):
    S = B[B.role == 'selection'].groupby(['family', 'k', 'method']).abs_err.mean().reset_index()
    return {(f, k): g.sort_values('abs_err').method.iat[0] for (f, k), g in S.groupby(['family', 'k'])}

def condition_lead_acid(v):
    T = LEAD_ACID_REST_V
    return 'critical' if v < T['critical'] else 'low' if v < T['warning'] else 'fair' if v < T['good'] else 'good'

def assess(V, LV, as_of, choice):
    rows = []
    for fam, obs in BATTERY_OBSERVABILITY.items():
        if obs['signal'] is None:
            rows.append(dict(family=fam, device_id=None, condition='not_measurable', note=obs['note'])); continue
    for dev, d in V.groupby('device_id'):
        fam = d.family.iat[0]; d = d[d.day <= as_of].sort_values('day')
        if not len(d) or fam == 'CloverSoilSensor': continue
        last = d.iloc[-1]; s = d.set_index('day').v_min.asfreq('D')
        r = dict(device_id=dev, family=fam, last_day=last.day, latest_v_min=last.v_min, latest_v_max=last.v_max, latest_v_median=last.v_med,
                 days_observed=len(d), stale_days=(as_of.floor('D') - last.day).days)
        sl, _, lo, hi = _ts(s.values[-28:]); r.update(slope_mV_per_week=sl * 7000, slope_lo90=lo * 7000, slope_hi90=hi * 7000)
        if fam in LEAD_ACID:
            r['rest_proxy_7d'] = float(np.nanmedian(s.values[-7:])); r['condition'] = condition_lead_acid(r['rest_proxy_7d'])
            r['charge_swing_7d'] = float(d.swing.tail(7).median()); r['charge_swing_prev28d'] = float(d.swing.iloc[-35:-7].median()) if len(d) > 14 else np.nan
            if d.temp_mean.notna().sum() >= 14: r['corr_rest_vs_temp'] = float(d[['v_min', 'temp_mean']].corr().iat[0, 1])
            m = choice.get((fam, 14), 'theil_sen'); r['forecast_model'] = m
            ps = {}
            for k in range(1, BATTERY_RISK_DAYS + 1):
                mu, sd = forecast(s, k, m)
                ps[k] = (mu, sd)
            mu14, sd14 = ps[14]; r.update(forecast_14d=mu14, forecast_14d_lo80=mu14 - 1.2816 * sd14, forecast_14d_hi80=mu14 + 1.2816 * sd14)
            for name in ['warning', 'critical']:
                v = LEAD_ACID_REST_V[name]; p = {k: norm.cdf((v - mu) / sd) if np.isfinite(mu) else np.nan for k, (mu, sd) in ps.items()}
                r[f'p_below_{name}_30d'] = float(np.nanmax(list(p.values())))
                for q in (0.1, 0.5, 0.9):
                    hit = [k for k, x in p.items() if x >= q]; r[f'{name}_crossing_day_p{int(q * 100)}'] = (last.day + pd.Timedelta(days=hit[0])) if hit else pd.NaT
            r['note'] = ('12 V sealed lead-acid' if fam == 'Bigbelly' else '12 V solar system, chemistry not identified (lead-acid bands are a generic reference)') + \
                        '; condition from the night-time minimum (resting-voltage proxy)'
        else:
            med = s.dropna(); base = med.iloc[:-7] if len(med) > 14 else med
            mad = (base - base.median()).abs().median() or 0.002
            step = abs(med.tail(7).median() - base.median()) > 4 * mad
            r['condition'] = 'voltage_step_change' if step else 'nominal_plateau'
            r['note'] = 'lithium cell, flat discharge plateau: voltage carries little information about remaining capacity until end of life; ' \
                        'no threshold-crossing date is estimated'
            if fam == 'Agvolution': r['note'] += '; battery error codes are flagged separately'
        rows.append(r)
    for dev, d in LV.groupby('device_id'):
        d = d[d.day <= as_of].sort_values('day'); s = d.set_index('day').level_pct.asfreq('D')
        sl, _, lo, hi = _ts(s.values[-60:])
        sig = np.isfinite(hi) and hi < 0
        rows.append(dict(device_id=dev, family=d.family.iat[0], last_day=d.day.iat[-1], latest_level_pct=d.level_pct.iat[-1], days_observed=len(d),
                         stale_days=(as_of.floor('D') - d.day.iat[-1]).days, slope_pct_per_week=sl * 7, slope_lo90=lo * 7, slope_hi90=hi * 7,
                         condition='good' if d.level_pct.iat[-1] >= 50 else 'low',
                         note='manufacturer battery-level estimate (current sense); ' + ('declining significantly' if sig else 'no significant decline; no depletion date estimated')))
    return pd.DataFrame(rows)

def anomalies(V, devs):
    """abnormal voltage changes: day-over-day change of the daily minimum beyond 5 robust SD of the device's own changes; error codes"""
    out = []
    for dev, d in V.groupby('device_id'):
        dv = d.sort_values('day').v_min.diff(); med = dv.rolling(28, min_periods=7).median().shift(1); mad = (dv - med).abs().rolling(28, min_periods=7).median().shift(1)
        z = (dv - med) / (1.4826 * mad.replace(0, np.nan))
        for i in np.where(np.abs(z) > 5)[0]:
            out.append(dict(device_id=dev, family=d.family.iat[0], day=d.day.iat[i], kind='abrupt_voltage_change', value=float(dv.iat[i]), robust_z=float(z.iat[i])))
    e = devs[(devs.dup_rank == 0) & devs.flag_battery_error_code].copy(); e['day'] = e.t.dt.floor('D')
    for (dev, day), g in e.groupby(['device_id', 'day']):
        out.append(dict(device_id=dev, family=g.family.iat[0], day=day, kind='battery_error_code', value=float(g.battery_raw.iat[0]), robust_z=np.nan))
    return pd.DataFrame(out)
