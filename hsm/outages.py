"""Causal outage declaration.

For every silence (gap in the slot ledger) a declaration time is computed from information available when the silence began:
  model rule : 2-state HMM fitted on the device's previous HMM_WINDOW_D days of nominal-mode slots (refit every HMM_REFIT_D days,
               slots of silences already declared before the fit are excluded); declare when P(run >= k) < ALPHA.
  reduced-mode rule : Beta-Binomial estimate of the device's own reduced-mode slot loss (prior DAILY_SLOT_PRIOR).
  floor rule : silence >= max(FLOOR_H, 2 x mode interval)  (operational definition).
declared_at = earliest of the applicable rules. Status: 'declared' if the silence lasted beyond declared_at; 'burst' if the sensor
reported before; 'undetermined' if observation ended first. Nothing here uses the eventual length of a silence to decide
anything about an earlier moment: declared_at is a time, the outcome only says whether that time was reached."""
import numpy as np, pandas as pd, warnings
from hmmlearn.hmm import CategoricalHMM
from .config import (ALPHA, FLOOR_H, HMM_WINDOW_D, HMM_REFIT_D, MIN_HISTORY_D, DAILY_SLOT_PRIOR, NOMINAL_MIN, HMM_FAMILIES,
                     SLOT_GRACE_FRAC, FLEET_WINDOW_H, FLEET_MIN_DEVICES, FLEET_CHANCE_ALPHA, NETWORK_SILENT_SHARE, NETWORK_MIN_SITES)
from .schedule import slot_events
warnings.filterwarnings('ignore')
KMAX = 20000

def surv(A, B, pi):
    v = (pi * (1 - B)) @ A * B; v = v / v.sum(); out = np.empty(KMAX); out[0] = 1.0
    for i in range(1, KMAX):
        v = (v @ A) * B; out[i] = v.sum()
        if out[i] < 1e-12: out[i:] = out[i]; break
    return out

def fit_hmm2(m, seg, prev):
    L = pd.Series(seg).value_counts(sort=False).sort_index().values
    inits = [(np.array([.9, .1]), np.array([[.95, .05], [.2, .8]]), np.array([[.9, .1], [.3, .7]]))]
    if prev is not None: inits.append((prev.startprob_, prev.transmat_, prev.emissionprob_))
    best = None
    for sp, A, B in inits:
        h = CategoricalHMM(n_components=2, n_iter=100, tol=1e-4, init_params='', random_state=0)
        h.startprob_, h.transmat_, h.emissionprob_ = sp.copy(), A.copy(), B.copy()
        try:
            h.fit(m.reshape(-1, 1), L); ll = h.score(m.reshape(-1, 1), L)
            if best is None or ll > best[0]: best = (ll, h)
        except Exception: pass
    return None if best is None else best[1]

def floor_min(fam, mode_min):
    return max(FLOOR_H.get(fam, FLOOR_H['default']) * 60, 2 * mode_min)

def declare_device(G, fam, first_t):
    """G: gap ledger of one device (sorted). Returns G with declaration columns and the list of HMM fits."""
    m0 = NOMINAL_MIN[fam]; g = SLOT_GRACE_FRAC
    G = G.copy().reset_index(drop=True)
    nominal = G.mode_min.values == m0
    # slot events of nominal-mode gaps (for model fitting)
    ev = slot_events(G, 'mode'); ev = ev[nominal[ev.gap.values]].reset_index(drop=True)
    fits = []; prev = None
    declared_at = np.full(len(G), np.datetime64('NaT'), 'datetime64[ns]'); rule = np.empty(len(G), dtype=object)
    kstar = np.full(len(G), np.nan); p_red = np.full(len(G), np.nan)
    anchor = G.anchor_t.values.astype('datetime64[ns]')
    nxt = np.where(G.open.values, np.datetime64('NaT'), G.next_t.values.astype('datetime64[ns]')).astype('datetime64[ns]')
    taus = np.arange(np.datetime64(first_t, 'ns') + np.timedelta64(MIN_HISTORY_D, 'D'),
                     anchor.max() + np.timedelta64(HMM_REFIT_D, 'D'), np.timedelta64(HMM_REFIT_D, 'D'))
    cur_S = None; cur_k = np.nan; tau_i = 0
    order = np.argsort(anchor, kind='stable')
    def decl(i):
        """declaration time of gap i using the model state in force at its start"""
        mode = G.mode_min.iat[i]; a = anchor[i]
        cands = [(a + np.timedelta64(int(floor_min(fam, mode) * 60), 's'), 'floor')]
        if mode == m0 and cur_S is not None and np.isfinite(cur_k) and fam in HMM_FAMILIES:
            cands.append((a + np.timedelta64(int((cur_k + g) * m0 * 60), 's'), 'model'))
        elif mode != m0:
            # reduced mode: device's own past reduced-mode slots (closed gaps that ended before this gap began)
            past = G[(G.mode_min == mode) & (~G.open) & (G.next_t.values.astype('datetime64[ns]') <= a)]
            miss = past.mode_missed.sum(); fill = past.mode_filled.sum()
            p = (miss + DAILY_SLOT_PRIOR[0]) / (miss + fill + sum(DAILY_SLOT_PRIOR)); p_red[i] = p
            k = int(np.ceil(np.log(ALPHA) / np.log(p))) if p < 1 else KMAX
            cands.append((a + np.timedelta64(int((k + g) * mode * 60), 's'), 'reduced_mode_model'))
        t_, r_ = min(cands, key=lambda c: c[0]); return t_, r_
    for i in order:
        a = anchor[i]
        while tau_i < len(taus) and taus[tau_i] <= a:
            tau = taus[tau_i]
            # exclude slots of silences already DECLARED before tau (known at tau)
            # a silence is a known outage at tau only if it actually lasted until its declaration time and that time is < tau
            reached = (~np.isnat(declared_at)) & (declared_at < tau) & (np.isnat(nxt) | (nxt > declared_at))
            known_out = np.where(reached)[0]
            w = ev[(ev.t.values >= tau - np.timedelta64(HMM_WINDOW_D, 'D')) & (ev.t.values < tau) & ~ev.gap.isin(known_out).values]
            if fam in HMM_FAMILIES and len(w) >= 100 and (1 - w.filled).sum() >= 5:
                seg = (w.gap.diff().fillna(0) > 1).cumsum().values           # break where gaps were excluded / reduced mode
                h = fit_hmm2((1 - w.filled.values).astype(int), seg, prev)
                if h is not None:
                    prev = h; b = int(np.argmax(h.emissionprob_[:, 1]))
                    S = surv(h.transmat_, h.emissionprob_[:, 1], h.get_stationary_distribution())
                    below = np.where(S < ALPHA)[0]; cur_S = S; cur_k = float(below[0] + 1) if len(below) else np.nan
                    fits.append(dict(valid_from=pd.Timestamp(tau), good_miss=h.emissionprob_[1 - b, 1], bad_miss=h.emissionprob_[b, 1],
                                     p_stay_good=h.transmat_[1 - b, 1 - b], p_stay_bad=h.transmat_[b, b], kstar_slots=cur_k, n_slots=len(w),
                                     loss_window=(1 - w.filled).mean()))
            tau_i += 1
        t_d, r_d = decl(i); declared_at[i] = t_d; rule[i] = r_d; kstar[i] = cur_k if G.mode_min.iat[i] == m0 else np.nan
    G['declared_at'] = declared_at; G['declare_rule'] = rule; G['kstar_slots_at_start'] = kstar; G['p_reduced_mode_loss'] = p_red
    G['onset'] = anchor + (G.mode_min.values * 60).astype('timedelta64[s]')        # first expected report that did not arrive
    end = np.where(G.open.values, np.datetime64('NaT'), G.next_t.values.astype('datetime64[ns]'))
    G['recovered_at'] = end
    reached = np.where(G.open.values, False, end > declared_at)
    G['status'] = np.where(reached, 'declared', np.where(G.open.values, 'pending', 'burst'))
    G['history_days_at_start'] = (anchor - np.datetime64(first_t, 'ns')).astype('timedelta64[h]').astype(float) / 24
    return G, fits

def finalize_open(G, obs_end_by_family):
    """open (tail) silences: declared if observation end passed declared_at, else undetermined"""
    oe = G.family.map(obs_end_by_family).values.astype('datetime64[ns]')
    op = G.status == 'pending'
    G.loc[op & (G.declared_at.values <= oe), 'status'] = 'declared'
    G.loc[op & (G.declared_at.values > oe), 'status'] = 'undetermined'
    return G

def network_hours(records_radio):
    """hour is network-level if >= NETWORK_SILENT_SHARE of sites active in the previous 24 h have zero records (causal at hour end)"""
    r = records_radio.dropna(subset=['gw_site']); r = r[r.gw_site != 'unresolved']
    h = r.assign(h=r.t.dt.floor('h') + pd.Timedelta('1h')).groupby(['h', 'gw_site']).size().unstack(fill_value=0)
    h = h.reindex(pd.date_range(h.index.min(), h.index.max(), freq='h'), fill_value=0)
    active = (h.shift(1).rolling(24, min_periods=1).sum() > 0)
    silent = ((h == 0) & active).sum(axis=1); n_act = active.sum(axis=1)
    flag = (silent >= NETWORK_SILENT_SHARE * n_act) & (n_act >= NETWORK_MIN_SITES)
    return pd.DataFrame({'hour_end': h.index, 'active_sites': n_act.values, 'silent_sites': silent.values, 'network_level': flag.values}), h

def scope(D, net, site_hour, uplinks):
    """Scope of each declared silence.
    network : the silence STARTED within 1 h of a network-level hour.
    fleet   : at least FLEET_MIN_DEVICES devices of the family (incl. this one) started silences within +-FLEET_WINDOW_H, and that
              concurrency is improbable (Poisson tail < FLEET_CHANCE_ALPHA) given the family's declared-onset rate in the previous 7 days.
    gateway : every primary gateway site of the device (>=10% of its records in the previous 14 days) received nothing from any
              device during the complete hours between onset and declaration.
    scope_at_declaration counts only declarations known at declared_at; scope_final counts all (retrospective revision)."""
    from scipy.stats import poisson
    D = D.sort_values('onset').copy()
    neth = np.array(sorted(net.loc[net.network_level, 'hour_end'].values), dtype='datetime64[ns]')
    up = uplinks.dropna(subset=['gw_site']); up = up[up.gw_site != 'unresolved']
    upd = {d: g for d, g in up.groupby('device_id')}; empty = up.iloc[:0]
    fam_on = {f: (g.onset.values.astype('datetime64[ns]'), g.declared_at.values.astype('datetime64[ns]'), g.device_id.values) for f, g in D.groupby('family')}
    W = np.timedelta64(FLEET_WINDOW_H, 'h')
    sc_at, sc_final, p_at = [], [], []
    for r in D.itertuples():
        on = np.datetime64(r.onset, 'ns'); da = np.datetime64(r.declared_at, 'ns')
        hrs = pd.date_range(pd.Timestamp(r.onset).ceil('h'), pd.Timestamp(r.declared_at).floor('h'), freq='h')
        near = len(neth) and (np.abs(neth - (on + np.timedelta64(30, 'm'))) <= np.timedelta64(90, 'm')).any()
        if near: sc_at.append('network'); sc_final.append('network'); p_at.append(np.nan); continue
        o_, d_, dv_ = fam_on[r.family]; other = dv_ != r.device_id
        conc = other & (np.abs(o_ - on) <= W)
        n_known = int((conc & (d_ <= da)).sum()) + 1; n_all = int(conc.sum()) + 1
        past = other & (o_ < on - W) & (o_ >= on - np.timedelta64(7, 'D'))
        rate = past.sum() / (7 * 24 - FLEET_WINDOW_H); lam = rate * 2 * FLEET_WINDOW_H
        p_k = poisson.sf(n_known - 2, lam); p_a = poisson.sf(n_all - 2, lam)          # P(N_others >= n-1)
        u_ = upd.get(r.device_id, empty); hist = u_[(u_.t < r.onset) & (u_.t >= r.onset - pd.Timedelta('14D'))]
        gw_down = False
        if len(hist) >= 5 and len(hrs):
            sh = hist.gw_site.value_counts(normalize=True); prim = [p for p in sh[sh >= 0.1].index if p in site_hour.columns]
            hh = [h for h in hrs if h in site_hour.index]
            gw_down = bool(prim) and bool(hh) and (site_hour.loc[hh, prim].values.sum() == 0)
        base = 'gateway' if gw_down else 'individual'
        sc_at.append('fleet' if (n_known >= FLEET_MIN_DEVICES and p_k < FLEET_CHANCE_ALPHA) else base)
        sc_final.append('fleet' if (n_all >= FLEET_MIN_DEVICES and p_a < FLEET_CHANCE_ALPHA) else base)
        p_at.append(p_k)
    D['scope_at_declaration'] = sc_at; D['scope_final'] = sc_final; D['fleet_chance_p_at_declaration'] = p_at
    return D
