"""Module 1: communication state and packet loss.
usage: python run_m1.py --foundation <foundation dir> [--no-eval]
Writes gold_m1_* tables and m1_evaluation.json into the foundation directory."""
import argparse, json, time, hashlib, os, glob
import numpy as np, pandas as pd
from hsm.m1 import analytic, features, models, state, importance
from hsm import config

def main(fd, importance_analysis=True):
    t0 = time.time(); rd = lambda n: pd.read_parquet(f'{fd}/{n}.parquet')
    H = rd('gold_device_hour'); L = rd('gold_silences'); F = rd('gold_hmm_fits'); U = rd('gold_uplinks'); soil = rd('silver_soil')
    A = analytic.build(L, F, H); print('analytic', round(time.time() - t0), 's', flush=True)
    X = features.build(H, L, U, soil, A); print('features', round(time.time() - t0), 's', X.shape, flush=True)
    X.to_parquet(f'{fd}/gold_m1_features.parquet', index=False)
    P, info = models.walk_forward(X); print('walk-forward', round(time.time() - t0), 's', flush=True)
    P.to_parquet(f'{fd}/gold_m1_predictions.parquet', index=False); info.to_parquet(f'{fd}/gold_m1_fold_info.parquet', index=False)
    finish(fd, X, P, H, U, L, t0, importance_analysis)

def finish(fd, X, P, H, U, L, t0=None, importance_analysis=True):
    t0 = t0 or time.time()
    S = models.score_table(P); S.to_parquet(f'{fd}/gold_m1_scores.parquet', index=False)
    choice, uni = models.choose(P, config.M1_FOLDS['test'][0])          # pre-registered choice: validation outcomes closed by test start
    C = models.compare_on_test(P, choice, uni); C.to_parquet(f'{fd}/gold_m1_comparison.parquet', index=False)
    sched = models.decision_schedule(P)
    T = state.timeline(X, P, sched); T.to_parquet(f'{fd}/gold_m1_state.parquet', index=False)
    for f, n in [('D', 'daily'), ('W', 'weekly')]: state.period_stats(H, U, L, f).to_parquet(f'{fd}/gold_m1_period_{n}.parquet', index=False)
    state.summary(T, X, H, U, L).to_parquet(f'{fd}/gold_m1_device_summary.parquet', index=False)
    if importance_analysis: importance.run(X).to_parquet(f'{fd}/gold_m1_importance.parquet', index=False)
    code = hashlib.sha256(b''.join(open(f, 'rb').read() for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), 'hsm', 'm1', '*.py'))))).hexdigest()
    ev = dict(schedule={f: {f'{k[0]}|{k[1]}': v[0] for k, v in c.items()} for f, (c, u) in sched.items()}, choice={f'{k[0]}|{k[1]}': dict(candidate=v[0], basis=v[1]) for k, v in choice.items()}, universal_choice=uni,
              m1_code_sha256=code, runtime_s=round(time.time() - t0, 1))
    json.dump(ev, open(f'{fd}/m1_evaluation.json', 'w'), indent=1, default=str)
    print('done', round(time.time() - t0), 's')
    return ev

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--foundation', required=True); ap.add_argument('--no-importance', action='store_true')
    a = ap.parse_args(); main(a.foundation, not a.no_importance)
