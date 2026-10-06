"""TR07 pilot configuration (development namespace paper1-g1-dev).

Settled by the live tracker (MS plan/MLST_OPEN_ITEMS.md, section 5, revised 6 Oct): the T25 generator
(C = 8, T = 32, theta = 0.25, oracle Sigma), six covariance-matched Bernoulli-Gaussian latents along K's
eigenbasis plus the Gaussian tail, Route A with bottleneck k = 3 and the Sigma^-1 loss on noisy x, gate at
p = 0.1 and N = 12,800, three development seeds; p = 1 and N = 51,200 are diagnostics; <= 1 week.

Everything else below is a DECLARED PROPOSAL of the implementation agent for review (MS
plan/onthefly/TR07_PILOT_PROTOCOL.md, "Proposed decisions").  Nothing here was approved by the author.
PILOT['author_decisions_recorded'] must be set to True (with the decisions written into the protocol)
before ``pilot.py run --mode pilot`` will start.
"""
from __future__ import annotations

from ..generator import FIXTURE_GEOMETRY, T25_GEOMETRY

NAMESPACE = 'paper1-g1-dev'

RECIPE = {'optimizer': 'adam', 'lr': 1e-3, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0,
          'batch': 64, 'max_epochs': 300, 'patience': 25, 'validation_fraction': 0.2, 'restarts': 4,
          'dtype': 'float64', 'threads': 1, 'deterministic': True}          # = the frozen X3 recipe

GATE = {
    'p': 0.1, 'N': 12800,
    # G1: beats both fitted baselines in every seed, beyond seed spread and beyond Monte Carlo error
    'seed_spread_multiplier': 2.0,        # mean(d) + 2 sd(d) < 0 over the three seeds
    'mc_multiplier': 3.0,                 # every seed: d_s + 3 MCSE_s < 0
    # G2: closes >= 25% of the shrinkage-to-Bayes gap (tracker), with the denominator chosen by the author
    'gap_fraction': 0.25,
    'gap_definition': 'sparse_oracle',    # PROPOSAL; alternatives 'literal', 'sparse_coordinates' (see protocol)
    'gap_aggregation': 'min_over_seeds_after_mc_margin',
    'min_gap': 1e-3,                      # a denominator below this (whitened risk units) is 'undefined' -> fail
}

DIAGNOSTICS = {
    'eval_records': 65536, 'eval_chunk': 8192,
    'probe_fit_fraction': 0.5,             # first half of the evaluation draws fits probes, second half scores
    'recovery_threshold': 0.5,             # proposal: R^2 above which a latent counts as recovered (E-NL use)
    'interference_amplitudes': [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],   # times 1/sqrt(p), latent units
}

PILOT = {
    'version': 'tr07-pilot/20261006-v1-candidate',
    'namespace': NAMESPACE, 'geometry': T25_GEOMETRY, 'theta': 0.25, 'n_sparse': 6, 'k': 3,
    'p': [0.1, 1.0], 'N': [12800], 'N_diagnostic': [51200], 'seeds': [1, 2, 3],
    'hidden': [128, 32], 'activation': 'silu', 'recipe': RECIPE, 'gate': GATE, 'diagnostics': DIAGNOSTICS,
    'author_decisions_recorded': False,
    'budget': {'time_box_days': 7, 'trainings': 3 * 2 * 4, 'diagnostic_trainings': 3 * 2 * 4, 'retries': 0},
}

FIXTURE = {
    **PILOT, 'version': 'tr07-pilot/fixture', 'namespace': NAMESPACE + '/fixture', 'geometry': FIXTURE_GEOMETRY,
    'n_sparse': 4, 'k': 2, 'p': [0.1, 1.0], 'N': [400], 'N_diagnostic': [], 'seeds': [901, 902, 903],
    'hidden': [8, 6], 'recipe': {**RECIPE, 'max_epochs': 20, 'patience': 5, 'batch': 32},
    'gate': {**GATE, 'N': 400},
    'diagnostics': {**DIAGNOSTICS, 'eval_records': 4096, 'eval_chunk': 2048},
    'author_decisions_recorded': True,     # fixtures only
}
