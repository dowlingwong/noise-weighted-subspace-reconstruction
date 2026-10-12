"""Timing benchmark at the registered size on SYNTHETIC data only (no MEG data, no dev subject):
a fixture subject with C = 24, T = 20 at 1100/12 Hz (d = 480), separable or sum-of-Kronecker synthetic noise,
the dev signal model and the Ledoit-Wolf reference.

    python -m experiments.mlst2026.meg_z_20261013.benchmark --out /tmp/megz_bench.json [--C 24 --T 20]
        [--parts fit,arms,population,theta_search] [--n-random 2] [--kind kron|nonkron]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from . import common, estimators, fit, fixture, population, signal
from .preprocess import load_subject
from .protocol import load_protocol


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--C', type=int, default=24)
    ap.add_argument('--T', type=int, default=20)
    ap.add_argument('--parts', default='fit,arms,population')
    ap.add_argument('--n-random', type=int, default=2)
    ap.add_argument('--theta', type=float, default=0.25)
    ap.add_argument('--N', default='400,1600')
    ap.add_argument('--fit-arms', default='ref,iso')
    ap.add_argument('--kind', default='nonkron')
    ap.add_argument('--target', type=float, default=None)
    a = ap.parse_args(argv)
    proto = load_protocol()
    parts = a.parts.split(',')
    fs = 1100.0 / 12
    tmp = Path(a.out).with_suffix('.d')
    n = {'training': 2200, 'calibration': 1100, 'evaluation': 600}
    fixture.make_derived_subject(tmp, 'sub-99', C=a.C, T=a.T, sfreq=fs, kind=a.kind, n=n, n_ref=20000, seed=9)
    sd = load_subject(tmp / 'sub-99.h5')
    res = {'synthetic': True, 'C': a.C, 'T': a.T, 'd': a.C * a.T, 'theta': a.theta, 'kind': a.kind}
    sp = population.SubjectPopulation(sd, proto, 'megz-benchmark')
    m, info = signal.build(sd.evoked, sd.sfreq, sp.ref['cov'], a.theta, proto['signal'])
    res['signal'] = info
    W0 = fit.metric(sp.ref['cov'])
    if 'arms' in parts:
        cal = sd.records('calibration')
        for est in ('lw', 'kron'):
            for mm in (256, 768, 1024):
                t0 = time.time()
                cov, _ = estimators.arm_estimate(cal[:mm], a.C, a.T, est)
                fit.metric(cov)
                res[f'seconds_arm_{est}_{mm}'] = time.time() - t0
    if 'fit' in parts:
        Ns = [int(x) for x in a.N.split(',')]
        S_tr, _ = m.draw(max(Ns), common.rng('bench', 1))
        X = S_tr + sd.records('training')[:max(Ns)]
        S_ev, al = m.draw(500, common.rng('bench', 2))
        evl = fit.Evaluation(S_ev, sd.records('evaluation')[:500], al, m.templates(), m.variances, W0, m.second_moment())
        for N in Ns:
            for arm, W in [(x, W0 if x == 'ref' else np.eye(sd.d)) for x in a.fit_arms.split(',')]:
                t0 = time.time()
                f = fit.fit_pair(X, N, W, a.C, a.T, 2, 3, proto['learners'], ('bench', N))
                t1 = time.time()
                r = fit.score_fit(f, evl)
                res[f'fit_{arm}_N{N}'] = {'seconds_fit': t1 - t0, 'seconds_score': time.time() - t1, 'D': r['D'],
                                          'restart_status': [x['status'] for x in r['restarts']],
                                          'restart_iterations': [x['iterations'] for x in r['restarts']]}
                print(arm, N, res[f'fit_{arm}_N{N}'], flush=True)
    if 'population' in parts:
        t0 = time.time()
        row = sp.row(a.theta, a.n_random, 'bench')
        res['population'] = {k: row[k] for k in ('a_best', 'cU', 'cF', 'B', 'B_cert', 'N_star_pred', 'N_star_interval',
                                                 'n_starts', 'start_statuses', 'H_min_eig', 'restart_agreement')}
        res['population']['lambda_min'] = row['B_dagger']['lambda_min']
        res['population']['seconds'] = time.time() - t0
        print(res['population'], flush=True)
    if 'theta_search' in parts:
        t0 = time.time()
        rule = dict(proto['theta_rule'], search_random_starts=a.n_random)
        target = a.target or 430.0
        r = population.theta_for_target(sp, target, rule, a.n_random, 'A')
        res['theta_search'] = {'target': target, 'root': r['root'], 'theta': r.get('theta'), 'N_star': r.get('N_star'),
                               'evaluations': len(r['search']), 'seconds': time.time() - t0}
        print(res['theta_search'], flush=True)
    common.atomic_json(a.out, res)


if __name__ == '__main__':
    main()
