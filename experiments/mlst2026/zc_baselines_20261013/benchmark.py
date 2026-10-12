"""Development timing benchmark (never the confirmation namespace); timing only.

Generator: the T25 BASE geometry (as the confirmation run).  Records come from the development namespace
'zc-dev-20261013' (design.DEV_NAMESPACE) with its own SHA-256-derived seed; the confirmation namespace is
never touched.  Each timed task is the confirmation task (runner.fit_task: U, F, 5-fold CV, N*-rule,
SHR with its lambda path).  Only wall time, NFPA iteration counts/statuses and peak RSS are kept: no risk,
regret, choice, lambda_hat or projector is recorded or printed.

    python -m experiments.mlst2026.zc_baselines_20261013.benchmark --out /tmp/zc_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

from experiments.mlst2026.t25_production_20260930 import protocol as t25p
from . import design, runner


def dev_protocol(n_seeds: int = 1) -> dict:
    base = t25p.BASE
    return design._build(base=base, namespace=design.DEV_NAMESPACE, seed_list=design.seeds(design.DEV_NAMESPACE, n_seeds),
                         thetas=base['theta'], Ns=base['N'], m=[256, 1024], metrics_all=['oracle', 'kron256', 'kron1024'],
                         metrics_cv_only=['full256', 'full1024'], nstar_table={'0.0': None, '0.12': 5241.0, '0.25': 1221.0,
                                                                                '0.4': 488.0},
                         nstar_source={'development': 'rounded tracker values; timing only'},
                         rc1_cells=[(0.25, 400)], max_iter=base['max_iter'],
                         dependencies={'files': {}, 'in_accepted_t25_closure': []}, evidence_role='DEVELOPMENT BENCHMARK')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--theta', type=float, default=0.25)
    ap.add_argument('--N', type=int, nargs='+', default=[3200])
    ap.add_argument('--metrics', nargs='+', default=['oracle', 'kron256', 'kron1024', 'full256', 'full1024'])
    a = ap.parse_args(argv)
    proto = dev_protocol()
    assert proto['namespace'] != design.NAMESPACE
    rows = []
    for N in a.N:
        for m in a.metrics:
            t0 = time.time()
            rec, _ = runner.fit_task(proto, 0, a.theta, N, m)
            sec = time.time() - t0
            row = {'theta': a.theta, 'N': N, 'metric': m, 'seconds': sec,
                   'full_F_iterations': [r['iterations'] for r in rec['optimization']['restarts']],
                   'full_F_statuses': sorted({r['status'] for r in rec['optimization']['restarts']}),
                   'fold_F_iterations': [f['iterations'] for f in rec['cv']['fold_F']],
                   'fold_F_statuses': sorted({s for f in rec['cv']['fold_F'] for s in f['statuses']}),
                   'peak_rss_MB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
            del rec
            rows.append(row)
            print(json.dumps(row), flush=True)
    out = {'note': 'development namespace, timing only; no risk recorded', 'namespace': proto['namespace'],
           'seed': proto['seeds'][0], 'threads': {v: os.environ.get(v) for v in runner.THREAD_VARS}, 'rows': rows,
           'environment': runner.environment()}
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
