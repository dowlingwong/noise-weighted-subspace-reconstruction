"""Runtime benchmark on a NON-G2 generator of the same dimension (d = 384); timing only.

Generator: the accepted T20-S/T25 family (t25.science.geometry) re-dimensioned to C = 6, T = 64,
(k_c, k_t) = (2, 3), with the T25 spectrum, theta = 0.25 and a benchmark-only geometry seed; noise as
in G2.  Records come from the namespace 'y19-benchmark-nonG2', never the confirmation namespace.  Each
fit is the confirmation fit (runner.fit_record); only wall time, iteration counts and restart
statuses are kept; no risk, contrast or projector is recorded or printed.

    python -m experiments.mlst2026.y19_20261012.benchmark --out /tmp/y19_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from . import generator as gen
from . import runner

BENCH_GENERATOR = {'family': 't25', 'C': 6, 'T': 64, 'kc': 2, 'kt': 3,
                   'spectrum': {'top': [12.0, 11.0, 10.0, 9.0, 8.0, 7.0], 'gap': 4.0, 'tail_min': 0.02},
                   'geometry_seed': gen.seed_int('y19-benchmark-nonG2', 'geometry') % 10 ** 8,
                   'rho_c': 0.4, 'rho_t': 0.7}
NAMESPACE = 'y19-benchmark-nonG2'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--N', type=int, nargs='+', default=[302, 1208, 2917, 11668])
    ap.add_argument('--seeds', type=int, default=2)
    a = ap.parse_args(argv)
    proto = {'namespace': NAMESPACE, 'learners': {'max_iter': 6000, 'gradient_tolerance': 1e-8}}
    g = runner.geometry(BENCH_GENERATOR)
    theta = 0.25
    rows = []
    for s in range(a.seeds):
        seed = gen.seed_int(NAMESPACE, 'seed', s) % 10 ** 12
        t0 = time.time()
        pool = gen.draw(g, NAMESPACE, seed, theta, max(a.N))
        t_pool = time.time() - t0
        for N in a.N:
            t0 = time.time()
            rec, _ = runner.fit_record(g, proto, theta, seed, N, pool)
            rows.append({'seed_index': s, 'N': N, 'seconds': time.time() - t0, 'pool_seconds': t_pool,
                         'restart_iterations': [r['iterations'] for r in rec['optimization']['restarts']],
                         'restart_statuses': [r['status'] for r in rec['optimization']['restarts']]})
            print(json.dumps(rows[-1]), flush=True)
    out = {'note': 'timing only on a non-G2 generator; no risk recorded', 'generator': BENCH_GENERATOR, 'theta': theta,
           'threads': {v: os.environ.get(v) for v in runner.THREAD_VARS}, 'rows': rows,
           'environment': runner.environment()}
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)


if __name__ == '__main__':
    main()
