"""Closed X2 (T25-NS) settings and exact allocation.

Registered in MS plan/T25NS_PROTOCOL_2026-10-05.md.  Everything not listed
here is inherited unchanged from the accepted T25 production code (generator,
estimators, shrinkage, floors, optimizer, starts, risks).
"""
from pathlib import Path
import json

BUDGET = {'max_cells': 288, 'max_charges': 9800, 'max_component_records': 340000,
          'rss_bytes': 4 * 1024 ** 3, 'scratch_bytes': 40 * 1024 ** 3, 'cell_seconds': 600,
          'campaign_seconds': 96 * 3600, 'threads': 1, 'retries': 0}
BASE = {'version': 't25ns/1', 'C': 8, 'T': 32, 'kc': 2, 'kt': 3,
        'spectrum': {'top': [12., 11., 10., 9., 8., 7.], 'gap': 4., 'tail_min': .02},
        'geometry_seed': 20260913, 'rho_c': .4, 'rho_t': .7,
        'line': {'f0': 0.125, 'ell': 8.0}, 'gain': {'amplitude': 0.2},
        'rho': [0.25, 0.5], 'theta': [0.25, 0.40], 'N': [400, 1000, 3200], 'm': [256, 1024],
        'seeds': list(range(202610050100, 202610050124)),
        'namespace': 'paper1-t25ns-v1/confirm', 'development_only': False,
        'max_iter': 6000, 'gradient_tolerance': 1e-8, 'budget': BUDGET}
FIXTURE = {**BASE, 'C': 3, 'T': 4, 'kc': 2, 'kt': 2,
           'spectrum': {'top': [6., 5., 4., 3.], 'gap': 1.5, 'tail_min': .05},
           'theta': [0.3], 'N': [40, 80], 'm': [16, 64], 'seeds': [910001, 910002, 910003],
           'namespace': 'paper1-t25ns-v1/fixture', 'development_only': True, 'max_iter': 50}
#: Production-shaped timing only: own namespace and seed, outputs discarded unviewed.
BENCHMARK = {**BASE, 'seeds': [991001], 'namespace': 'paper1-t25ns-v1/benchmark-20261005',
             'development_only': True}
BENCHMARK_CELLS = [(0.5, 0.25, 400), (0.25, 0.40, 1000), (0.5, 0.40, 3200), (0.25, 0.25, 3200)]

#: Registered namespaces of the accepted T25 package and its development runs (for
#: the disjointness check); X2 never draws from any of them.
T25_NAMESPACES = {'paper1-t25-v1/confirm': list(range(202609290100, 202609290124)),
                  'paper1-t25-v1/fixture': [900001, 900002],
                  'paper1-t25-v1/benchmark-20260930': [990001],
                  'paper1-t25-v1/solver-dev-20260930': [777]}

REPRESENTATION_CHARGES = 25   # 5 metrics x (4 structured starts + 1 unrestricted)
COVARIANCE_CHARGES = 4        # 2 calibration sizes x 2 estimators
REFERENCE_CHARGES = 5         # 4 local-domain responses + 1 truth eigenspace
CELL_CHARGES = REPRESENTATION_CHARGES + COVARIANCE_CHARGES + REFERENCE_CHARGES


def config(path):
    c = json.loads(Path(path).read_text())
    if c not in (BASE, FIXTURE):
        raise ValueError('settings differ from the registered X2 configuration')
    return c


def cells(c):
    return [{'seed': s, 'rho': r, 'theta': th, 'N': n}
            for s in c['seeds'] for r in c['rho'] for th in c['theta'] for n in c['N']]


def enumeration(c):
    rows = cells(c)
    population = len(c['rho']) + 1          # one reference charge per rho, plus rho = 0
    charges = len(rows) * CELL_CHARGES + population
    component_records = len(c['seeds']) * (2 * len(c['theta']) * max(c['N']) + max(c['m']))
    b = c['budget']
    if len(rows) > b['max_cells'] or charges > b['max_charges'] or component_records > b['max_component_records']:
        raise ValueError('allocation exceeds caps')
    return {'cells': rows, 'paired_cells': len(rows), 'cell_charges': CELL_CHARGES,
            'representation_charges': len(rows) * REPRESENTATION_CHARGES,
            'structured_starts': len(rows) * 20, 'covariance_charges': len(rows) * COVARIANCE_CHARGES,
            'reference_charges': len(rows) * REFERENCE_CHARGES + population, 'total_charges': charges,
            'unique_component_records': component_records,
            'common_random_numbers': 'signal, noise and calibration standard-normal streams are keyed by '
                                     '(namespace, seed[, theta]) only, so both rho levels share them',
            'calibration_recomputations': 'per-seed nested pool recomputed in each isolated cell; charged '
                                          'each time, never an independent replication'}
