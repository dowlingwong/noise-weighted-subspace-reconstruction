"""Validation gate (run BEFORE any G2 use): the Y19 constants code path (predictions.constants, which
wraps core.py unchanged) must reproduce the saved T25 values.

Reference: tests/t25_reference.json, copied verbatim from MS reviews/p1nl_preparation_20261006/tr09/
weight_table.json (a, c_U, c_F, N*) and reviews/upgrade_y15_y24_2026-10-09/y15/RECOMPUTATION.json (B_cert),
with the SHA-256 of both sources.  Two passes on the T25 generator (p1nl_dev SparseLatentModel, p = 1):
  (1) the weight-table procedure (same restart stream, n_random = 6);
  (2) the Y19 restart budget (64 Newton restarts, Y19 stream) -- a must not move.

    python -m experiments.mlst2026.y19_20261012.validation_gate
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.mlst2026.p1nl_dev_20261006.generator import SparseLatentModel, rng   # T25 generator (guarded)
from . import predictions as P

REF = Path(__file__).resolve().parent / 'tests' / 't25_reference.json'
TOL = {'cU': 1e-10, 'a': 1e-8, 'cF': 1e-6, 'N_star': 1e-6, 'B_cert': 1e-8, 'zero_abs': 1e-10}


def rel(x, y):
    return abs(x - y) / abs(y)


def gate() -> dict:
    ref = json.loads(REF.read_text())
    rows = []
    for r in ref['rows']:
        th = r['theta']
        m = SparseLatentModel(theta=th, p=1.0)
        new = P.constants(m, rng('paper1-tr09-dev', 'population', th), n_random=6)
        big = P.constants(m, P.t25.rng(P.NAMESPACE, 'validation-gate-64', th), n_random=P.N_RANDOM_FINAL)
        nonsep = r['N_star_leading_order'] is not None
        c = {'theta': th,
             'cU': [new['cU'], r['cU'], rel(new['cU'], r['cU']), rel(new['cU'], r['cU']) <= TOL['cU']],
             'cF': [new['cF'], r['cF_gaussian'], rel(new['cF'], r['cF_gaussian']), rel(new['cF'], r['cF_gaussian']) <= TOL['cF']]}
        if nonsep:
            c['a'] = [new['a_best'], r['a'], rel(new['a_best'], r['a']), rel(new['a_best'], r['a']) <= TOL['a']]
            c['a_64_restarts'] = [big['a_best'], r['a'], rel(big['a_best'], r['a']), rel(big['a_best'], r['a']) <= TOL['a']]
            c['N_star'] = [new['N_star_pred'], r['N_star_leading_order'], rel(new['N_star_pred'], r['N_star_leading_order']),
                           rel(new['N_star_pred'], r['N_star_leading_order']) <= TOL['N_star']]
            c['N_star_rounded_vs_tracker'] = [round(new['N_star_pred']), ref['tracker_rounded'][str(th)],
                                              round(new['N_star_pred']) == ref['tracker_rounded'][str(th)]]
            c['B_cert'] = [new['B_cert'], r['B_cert_y15_recomputation'], rel(new['B_cert'], r['B_cert_y15_recomputation']),
                           rel(new['B_cert'], r['B_cert_y15_recomputation']) <= TOL['B_cert']]
            iv = r['N_star_interval_y15_recomputation']
            c['interval_upper'] = [new['N_star_interval'][1], iv[1], rel(new['N_star_interval'][1], iv[1]),
                                   rel(new['N_star_interval'][1], iv[1]) <= TOL['N_star']]
        else:
            c['a'] = [new['a_best'], r['a'], abs(new['a_best']), abs(new['a_best']) <= TOL['zero_abs']]
            c['a_64_restarts'] = [big['a_best'], r['a'], abs(big['a_best']), abs(big['a_best']) <= TOL['zero_abs']]
            c['B_cert'] = [new['B_cert'], r['B_cert_y15_recomputation'], abs(new['B_cert']), abs(new['B_cert']) <= TOL['zero_abs']]
            c['N_star'] = [new['N_star_pred'], None, 0.0, new['N_star_pred'] is None]
        c['cF_in_brief_range_187.7_194.8'] = 187.6 <= new['cF'] <= 194.9
        c['pass'] = bool(all(v[-1] for k, v in c.items() if isinstance(v, list)) and c['cF_in_brief_range_187.7_194.8'])
        rows.append(c)
    return {'columns': ['recomputed', 'reference', 'rel (abs for zeros)', 'ok'], 'tolerances': TOL,
            'sources': ref['sources'], 'rows': rows, 'pass': bool(all(r['pass'] for r in rows))}


if __name__ == '__main__':
    print(json.dumps(gate(), indent=1))
