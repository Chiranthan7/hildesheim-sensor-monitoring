"""Module 1, individual statistical layer (no supervised training). Everything is evaluated at hour ends h using only slots
resolved before h (a missed slot is resolved at its deadline, a filled slot at the data arrival).

Slot process (per device, nominal reporting mode): X_k = 1 delivered, 0 missed, on the device's service schedule.
Slots of a silence are removed from the TRAINING statistics from the moment the silence is declared an outage (it is then known
to be an outage, the same rule the foundation HMM fits use); they always stay in the FILTER (state tracking).

Models (n = slots in the horizon):
  Bernoulli (chronic):  p = (d + 1) / (s + 2) over the trailing 28 days;  E[A_n] = p
  Markov-1:            states D/M, p_DD = P(X=1|prev 1), p_MD = P(X=1|prev 0) from trailing transition counts (Beta(1,1));
                       E[A_n] = (1/n) e_last . sum_{h=1..n} P^h . [1,0]'
  EB-Markov:           p_DD = (n_DD + kappa m_DD) / (n_D. + kappa), family mean m and strength kappa from the family's OTHER
                       devices at the same hour (method of moments), i.e. hierarchical shrinkage toward the family
  HMM2:                hidden good/bad link state, weekly causal fits (foundation). Forward filter alpha_k = norm(alpha_{k-1} A * e(x_k));
                       E[A_n] = (1/n) alpha . sum_{h=1..n} A^h . d,  d = P(delivered | state)
  closed form for a 2-state chain A=[[1-a,a],[b,1-b]]: sum_{h=1..n} A^h = n*1*pi + c_n (I - 1*pi),  pi=(b,a)/(a+b), lam=1-a-b,
  c_n = lam (1 - lam^n) / (1 - lam).
Silence surprise: S(k) = P(a silence lasts >= k slots | normal behaviour) under the HMM fit in force."""
import numpy as np, pandas as pd
from ..config import NOMINAL_MIN, M1_HORIZONS_H, M1_WINDOW_D, M1_EB_KAPPA_BOUNDS
from ..schedule import slot_events
from ..outages import surv

def sum_powers(A, n):
    a, b = A[0, 1], A[1, 0]
    if a + b < 1e-12: return n * np.eye(2)
    pi = np.array([b, a]) / (a + b); P = np.outer(np.ones(2), pi); lam = 1 - a - b
    c = n if abs(1 - lam) < 1e-12 else lam * (1 - lam ** n) / (1 - lam)
    return n * P + c * (np.eye(2) - P)

def stationary(A):
    a, b = A[0, 1], A[1, 0]
    return np.array([0.5, 0.5]) if a + b < 1e-12 else np.array([b, a]) / (a + b)

def _active_counts(t_add, t_end, w, hrs):
    """sum of weights w of contributions active at hour end h: t_add < h <= ... and t_end >= h (half-open [t_add, t_end))"""
    if len(t_add) == 0: return np.zeros(len(hrs))
    o1 = np.argsort(t_add); o2 = np.argsort(t_end)
    c1 = np.concatenate([[0], np.cumsum(w[o1])]); c2 = np.concatenate([[0], np.cumsum(w[o2])])
    return c1[np.searchsorted(t_add[o1], hrs, side='left')] - c2[np.searchsorted(t_end[o2], hrs, side='left')]

def device(G, fits, hrs, fam, rx_mode):
    """G: service ledger of one device; fits: HMM fits of the device; hrs: hour ends (datetime64[ns]); rx_mode: reception mode at h"""
    m0 = NOMINAL_MIN[fam]; W = np.timedelta64(M1_WINDOW_D, 'D')
    G = G.sort_values('anchor_t').reset_index(drop=True)
    nominal_gap = (G.mode_min.values == m0)
    ev = slot_events(G, 'mode')
    ev = ev[nominal_gap[ev.gap.values]].reset_index(drop=True)
    t = ev.t.values.astype('datetime64[ns]'); x = ev.filled.values.astype(int); gap = ev.gap.values
    nxt = G.next_t.values.astype('datetime64[ns]'); dec = G.declared_at.values.astype('datetime64[ns]')
    big = np.datetime64('2262-01-01', 'ns')
    reached = ~np.isnat(dec) & (np.isnat(nxt) | (dec < nxt))
    removal = np.where(reached, dec, big)[gap] if len(gap) else np.array([], 'datetime64[ns]')
    end = np.maximum(np.minimum(t + W, removal), t)        # slot resolved after its silence was declared: never counted
    out = {}
    # --- trailing slot and transition counts (Bernoulli / Markov) ---
    out['n28'] = _active_counts(t, end, np.ones(len(t)), hrs)
    out['d28'] = _active_counts(t, end, x.astype(float), hrs)
    if len(t) > 1:
        cont = np.diff(gap) <= 1                                    # consecutive nominal gaps (no reduced-mode gap between)
        tp, xp, xc = t[1:], x[:-1], x[1:]; ep = np.maximum(np.minimum(end[:-1], end[1:]), tp)
        for a in (0, 1):
            for b in (0, 1):
                m = cont & (xp == a) & (xc == b)
                out[f'n_{"DM"[1 - a]}{"DM"[1 - b]}'] = _active_counts(tp[m], ep[m], np.ones(m.sum()), hrs)
    else:
        for k in ['n_DD', 'n_DM', 'n_MD', 'n_MM']: out[k] = np.zeros(len(hrs))
    # --- last resolved slot and current run of missed slots ---
    k = np.searchsorted(t, hrs, side='left') - 1
    out['last_outcome'] = np.where(k >= 0, x[np.maximum(k, 0)], np.nan)
    run = np.zeros(len(x), int); r = 0
    for i in range(len(x)): r = r + 1 if x[i] == 0 else 0; run[i] = r
    out['miss_run_slots'] = np.where(k >= 0, run[np.maximum(k, 0)], 0)
    # --- HMM2 forward filter with the fit in force at each slot ---
    pb = np.full(len(x), np.nan); fid = np.full(len(x), -1)
    F = fits.sort_values('valid_from').reset_index(drop=True) if len(fits) else fits
    if len(F):
        vf = F.valid_from.values.astype('datetime64[ns]')
        par = [(np.array([[r.p_stay_good, 1 - r.p_stay_good], [1 - r.p_stay_bad, r.p_stay_bad]]),
                np.array([1 - r.good_miss, 1 - r.bad_miss])) for r in F.itertuples()]
        f_at = np.searchsorted(vf, t, side='right') - 1               # fit in force when the slot was resolved
        start = np.searchsorted(t, vf[0] - np.timedelta64(1, 'D'), side='left')   # warm start: last day before the first fit
        alpha = None
        for i in range(start, len(x)):
            j = max(f_at[i], 0); A, d = par[j]
            if alpha is None: alpha = stationary(A)
            e = d if x[i] == 1 else 1 - d
            a = (alpha @ A) * e; s = a.sum(); alpha = a / s if s > 0 else stationary(A)
            if f_at[i] >= 0: pb[i] = alpha[1]; fid[i] = j
        fh = np.searchsorted(vf, hrs, side='right') - 1                # fit in force at the hour end (forecasts)
        ok = (fh >= 0) & (k >= 0)
        a_bad = np.where(ok, pb[np.maximum(k, 0)], np.nan); ok &= ~np.isnan(a_bad)
        out['hmm_available'] = ok; out['hmm_p_bad'] = np.where(ok, a_bad, np.nan); out['hmm_fit_idx'] = np.where(ok, fh, -1)
        nf = len(par); Hs = M1_HORIZONS_H[fam]
        Ad = np.stack([A @ d for A, d in par])                                   # one-step: alpha . A . d
        MH = {H: np.stack([sum_powers(A, max(int(round(H * 60 / m0)), 1)) @ d / max(int(round(H * 60 / m0)), 1) for A, d in par]) for H in Hs}
        Sv = [surv(A, 1 - d, stationary(A)) for A, d in par]
        al = np.stack([1 - a_bad, a_bad], axis=1); j = np.maximum(fh, 0)
        pn = np.where(ok, (al * Ad[j]).sum(1), np.nan)
        fc = {H: np.where(ok, (al * MH[H][j]).sum(1), np.nan) for H in Hs}
        kk = out['miss_run_slots'].astype(int)
        sp = np.array([Sv[j[i]][min(kk[i], len(Sv[j[i]]) - 1)] if (ok[i] and kk[i] > 0) else (1.0 if ok[i] else np.nan) for i in range(len(hrs))])
        out['hmm_p_next'] = pn; out['hmm_silence_surprise'] = sp
        for H in fc: out[f'hmm_E{H}'] = fc[H]
    else:
        out['hmm_available'] = np.zeros(len(hrs), bool); out['hmm_p_bad'] = out['hmm_p_next'] = out['hmm_silence_surprise'] = np.full(len(hrs), np.nan)
        out['hmm_fit_idx'] = np.full(len(hrs), -1)
        for H in M1_HORIZONS_H[fam]: out[f'hmm_E{H}'] = np.full(len(hrs), np.nan)
    # --- reduced-mode loss (device's own closed reduced-mode gaps, all past) ---
    red = (~nominal_gap) & ~G.open.values
    if red.any():
        rt = nxt[red]; o = np.argsort(rt); rt = rt[o]
        mm = np.cumsum(G.mode_missed.values[red][o]); ff = np.cumsum(G.mode_filled.values[red][o])
        j = np.searchsorted(rt, hrs, side='left') - 1
        out['red_missed'] = np.where(j >= 0, mm[np.maximum(j, 0)], 0); out['red_filled'] = np.where(j >= 0, ff[np.maximum(j, 0)], 0)
    else:
        out['red_missed'] = np.zeros(len(hrs)); out['red_filled'] = np.zeros(len(hrs))
    D = pd.DataFrame(out); D.insert(0, 'hour_end', hrs); D['rx_mode_min'] = rx_mode
    return D

def add_forecasts(A):
    """Bernoulli, Markov and EB-Markov forecasts from the trailing counts; family prior from the OTHER devices at the same hour."""
    A = A.copy()
    A['n_D'] = A.n_DD + A.n_DM; A['n_M'] = A.n_MD + A.n_MM
    A['bern_p'] = (A.d28 + 1) / (A.n28 + 2)
    A['mk_pDD'] = (A.n_DD + 1) / (A.n_D + 2); A['mk_pMD'] = (A.n_MD + 1) / (A.n_M + 2)
    lo, hi = M1_EB_KAPPA_BOUNDS
    for tr, num, den in [('DD', 'n_DD', 'n_D'), ('MD', 'n_MD', 'n_M')]:
        g = A.groupby(['family', 'hour_end'])
        tot_n = g[num].transform('sum') - A[num]; tot_d = g[den].transform('sum') - A[den]
        m = (tot_n + 1) / (tot_d + 2)
        # method of moments for kappa across the family's devices with >= 30 transitions (excluding self approximately via pooled var)
        r = np.where(A[den] >= 30, A[num] / A[den].replace(0, np.nan), np.nan)
        A['_r'] = r; A['_bv'] = np.where(A[den] >= 30, r * (1 - r) / A[den].replace(0, np.nan), np.nan)
        g = A.groupby(['family', 'hour_end'])
        var_r = g['_r'].transform('var'); mean_bv = g['_bv'].transform('mean'); cnt = g['_r'].transform('count')
        vb = (var_r - mean_bv).clip(lower=1e-6)
        kappa = (m * (1 - m) / vb - 1).clip(lo, hi).where(cnt >= 3, 50.0)
        A[f'eb_p{tr}'] = (A[num] + kappa * m) / (A[den] + kappa); A[f'eb_kappa_{tr}'] = kappa
    A = A.drop(columns=['_r', '_bv'])
    for fam, idx in A.groupby('family').indices.items():
        m0 = NOMINAL_MIN[fam]
        for H in M1_HORIZONS_H[fam]:
            n = max(int(round(H * 60 / m0)), 1)
            for pre in ['mk', 'eb']:
                pDD = A[f'{pre}_pDD'].values[idx]; pMD = A[f'{pre}_pMD'].values[idx]; last = A.last_outcome.values[idx]
                a_ = 1 - pDD; b_ = pMD; lam = 1 - a_ - b_; s = a_ + b_
                pi_D = np.where(s > 1e-12, b_ / np.maximum(s, 1e-12), 0.5)
                c = np.where(np.abs(1 - lam) < 1e-12, n, lam * (1 - lam ** n) / np.where(np.abs(1 - lam) < 1e-12, 1, 1 - lam))
                eD = pi_D + c / n * (1 - pi_D); eM = pi_D - c / n * pi_D        # E[A_n | last = D], E[A_n | last = M]
                A.loc[A.index[idx], f'{pre}_E{H}'] = np.where(np.isnan(last), np.nan, np.where(last == 1, eD, eM))
            A.loc[A.index[idx], f'bern_E{H}'] = A.bern_p.values[idx]
            # reduced mode: analytic forecasts are on the nominal schedule -> device's own reduced-mode delivery x (m0 / mode)
            red = A.rx_mode_min.values[idx] != m0
            if red.any():
                pr = (A.red_filled.values[idx] + 1) / (A.red_filled.values[idx] + A.red_missed.values[idx] + 2)
                scale = m0 / np.where(red, A.rx_mode_min.values[idx], m0)
                for pre in ['hmm', 'mk', 'eb', 'bern']:
                    col = f'{pre}_E{H}'; v = A[col].values[idx].copy(); v[red] = (pr * scale)[red]
                    A.loc[A.index[idx], col] = v
    return A

def build(L, fits, H):
    """L: gold_silences, fits: gold_hmm_fits, H: gold_device_hour -> analytic table per device-hour"""
    parts = []
    Ld = {d: g for d, g in L.groupby('device_id')}; Fd = {d: g for d, g in fits.groupby('device_id')} if len(fits) else {}
    for dev, h in H.groupby('device_id'):
        fam = h.family.iat[0]
        if dev not in Ld or fam not in M1_HORIZONS_H: continue
        hrs = h.hour_end.values.astype('datetime64[ns]')
        D = device(Ld[dev], Fd.get(dev, fits.iloc[:0]), hrs, fam, h.rx_mode_min.values)
        D.insert(0, 'device_id', dev); D.insert(1, 'family', fam); parts.append(D)
    return add_forecasts(pd.concat(parts, ignore_index=True))
