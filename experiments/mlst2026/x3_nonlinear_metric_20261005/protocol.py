"""Closed X3 settings (MS plan/X3_NONLINEAR_METRIC_PROTOCOL_2026-10-05.md).

Generator, covariance and risks are the accepted T25 production code (oracle Sigma,
separable St (x) Sc).  The training recipe below is frozen before any seed is run.
"""
from pathlib import Path
import json

TORCH_VERSION = '2.9.1'
RECIPE = {'optimizer': 'adam', 'lr': 1e-3, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0,
          'batch': 64, 'max_epochs': 300, 'patience': 25, 'validation_fraction': 0.2, 'restarts': 4,
          'dtype': 'float64', 'threads': 1, 'deterministic': True}
BUDGET = {'max_cells': 72, 'max_trainings': 1152, 'rss_bytes': 4 * 1024 ** 3, 'scratch_bytes': 20 * 1024 ** 3,
          'cell_seconds': 3600, 'campaign_seconds': 96 * 3600, 'threads': 1, 'retries': 0}
BASE = {'version': 'x3/1', 'C': 8, 'T': 32, 'kc': 2, 'kt': 3,
        'spectrum': {'top': [12., 11., 10., 9., 8., 7.], 'gap': 4., 'tail_min': .02},
        'geometry_seed': 20260913, 'rho_c': .4, 'rho_t': .7, 'theta': 0.25, 'N': [400, 1600, 3200],
        'k': 6, 'hidden': [128, 32], 'classes': ['tied_linear', 'deep_linear', 'deep_silu'],
        'losses': ['isotropic', 'sigma_inverse'], 'recipe': RECIPE,
        'evaluation': {'records': 2048, 'hutchinson_probes': 4},
        'seeds': list(range(202610050200, 202610050224)), 'namespace': 'paper1-x3-v1/confirm',
        'development_only': False, 'torch_version': TORCH_VERSION, 'budget': BUDGET}
FIXTURE = {**BASE, 'C': 3, 'T': 4, 'kc': 2, 'kt': 2, 'spectrum': {'top': [6., 5., 4., 3.], 'gap': 1.5, 'tail_min': .05},
           'N': [40, 80], 'k': 4, 'hidden': [8, 6], 'recipe': {**RECIPE, 'max_epochs': 30, 'patience': 5, 'batch': 16},
           'evaluation': {'records': 256, 'hutchinson_probes': 4}, 'seeds': [920001, 920002, 920003],
           'namespace': 'paper1-x3-v1/fixture', 'development_only': True}
BENCHMARK = {**BASE, 'seeds': [992001], 'namespace': 'paper1-x3-v1/benchmark-20261005', 'development_only': True}
DEEP = ('deep_linear', 'deep_silu')
#: per cell: 16 trainings (2 deep classes x 2 losses x 4 restarts) + 2 closed forms + 6 evaluations
CELL_CHARGES = 24


def config(path):
    c = json.loads(Path(path).read_text())
    if c not in (BASE, FIXTURE):
        raise ValueError('settings differ from the registered X3 configuration')
    return c


def enumeration(c):
    cells = [{'seed': s, 'N': n} for s in c['seeds'] for n in c['N']]
    trainings = len(cells) * len(DEEP) * len(c['losses']) * c['recipe']['restarts']
    if len(cells) > c['budget']['max_cells'] or trainings > c['budget']['max_trainings']:
        raise ValueError('allocation exceeds caps')
    return {'cells': cells, 'paired_cells': len(cells), 'trainings': trainings,
            'total_charges': len(cells) * CELL_CHARGES,
            'closed_form_fits': len(cells) * len(c['losses']),
            'evaluated_models': len(cells) * len(c['classes']) * len(c['losses']),
            'common_random_numbers': 'per (seed, N, restart) the deep linear and deep SiLU models and both '
                                     'losses share initial weights and mini-batch order; one evaluation set '
                                     'per seed is shared across N, classes and losses'}
