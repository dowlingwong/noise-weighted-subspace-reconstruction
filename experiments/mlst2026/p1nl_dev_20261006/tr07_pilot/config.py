"""TR07 pilot configuration (development namespace paper1-g1-dev).

Settled by the live tracker (MS plan/MLST_OPEN_ITEMS.md, section 5, revised 6 Oct): the T25 generator
(C = 8, T = 32, theta = 0.25, oracle Sigma), six covariance-matched Bernoulli-Gaussian latents along K's
eigenbasis plus the Gaussian tail, Route A with bottleneck k = 3 and the Sigma^-1 loss on noisy x, gate at
p = 0.1 and N = 12,800, three development seeds; p = 1 and N = 51,200 are diagnostics; <= 1 week.

The remaining choices are the author's decisions D1-D8 of 6 Oct 2026, recorded in the dated amendment
MS plan/onthefly/TR07_PILOT_PROTOCOL.md, section 11 ("Amendment 1"):
D1 sparse_oracle normalisation with the 25% threshold and the total-risk numerator (full-gap and
sparse-coordinate closures are diagnostics); D2 every seed beyond 3 MCSE against both fitted baselines AND
mean + 2 sd < 0 (a conservative development rule, not a confidence interval or a confirmatory test);
D3 SiLU; D4 widths [128, 32], bottleneck 3, the X3-derived recipe; D5 linear baselines fitted on the AE's
training subset only; D6 65,536 shared evaluation draws; D7 no deep-linear arm; D8 representation and
interference diagnostics descriptive only.  Only N = 12,800 is planned: no N = 51,200 by default and never
to rescue a failed primary pilot.
``pilot.py run --mode pilot`` additionally requires PILOT['independent_review_accepted'] (set only after the
independent reviewer accepts the amended implementation) and P1NL_TR07_PILOT_AUTHORISED=yes.
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
    'gap_definition': 'sparse_oracle',    # D1 (author, 6 Oct); 'full_gap' and 'sparse_coordinates' are diagnostics
    'gap_aggregation': 'min_over_seeds_after_mc_margin',
    'min_gap': 1e-3,                      # a denominator below this (whitened risk units) is 'undefined' -> fail
}

DIAGNOSTICS = {
    'eval_records': 65536, 'eval_chunk': 8192,
    'probe_fit_fraction': 0.5,             # first half of the evaluation draws fits probes, second half scores
    # No recovery threshold here (D8): any R^2 threshold for 'recovered' is frozen before formal E-NL,
    # not chosen in or after the pilot; the pilot reports recovery R^2 descriptively.
    'interference_amplitudes': [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],   # times 1/sqrt(p), latent units
}

PILOT = {
    'version': 'tr07-pilot/20261006-v2-amendment1-candidate',
    'namespace': NAMESPACE, 'geometry': T25_GEOMETRY, 'theta': 0.25, 'n_sparse': 6, 'k': 3,
    'p': [0.1, 1.0], 'N': [12800], 'N_diagnostic': [], 'seeds': [1, 2, 3],      # no N = 51,200 (author, 6 Oct)
    'hidden': [128, 32], 'activation': 'silu', 'recipe': RECIPE, 'gate': GATE, 'diagnostics': DIAGNOSTICS,
    'author_decisions_recorded': True,          # D1-D8 recorded 6 Oct 2026 (protocol amendment 1)
    'independent_review_accepted': False,       # set True only after the independent re-review accepts
    'amendment': 'TR07_PILOT_PROTOCOL.md section 11, Amendment 1 (6 Oct 2026)',
    'budget': {'time_box_days': 7, 'trainings': 3 * 2 * 4, 'diagnostic_trainings': 0, 'retries': 0},
}

FIXTURE = {
    **PILOT, 'version': 'tr07-pilot/fixture', 'namespace': NAMESPACE + '/fixture', 'geometry': FIXTURE_GEOMETRY,
    'n_sparse': 4, 'k': 2, 'p': [0.1, 1.0], 'N': [400], 'N_diagnostic': [], 'seeds': [901, 902, 903],
    'hidden': [8, 6], 'recipe': {**RECIPE, 'max_epochs': 20, 'patience': 5, 'batch': 32},
    'gate': {**GATE, 'N': 400},
    'diagnostics': {**DIAGNOSTICS, 'eval_records': 4096, 'eval_chunk': 2048},
    'author_decisions_recorded': True, 'independent_review_accepted': True,     # fixtures only
}
