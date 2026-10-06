"""Mechanical TR07 gate (written before the pilot; MS plan/onthefly/TR07_PILOT_PROTOCOL.md section 5).

Gate cell: p = 0.1, N = 12,800 (cfg['gate']).  Per seed s (three development seeds):
    d_a(s) = R_AE(s) - R_proj(s),   d_b(s) = R_AE(s) - R_shrink(s)
with R_proj, R_shrink the EXACT whitened clean risks of the rank-k projector and rank-k shrinkage maps
FITTED on the same training records, and R_AE the Monte Carlo risk of the selected autoencoder (MCSE).

G1 (beats both, beyond seed spread), for X in {a, b}:
    every seed: d_X(s) + m_mc * MCSE(s) < 0,  and  mean_s d_X + m_sd * sd_s(d_X) < 0   (sd with ddof = 1).
G2 (closes >= 25% of a reference gap), per seed, with the POPULATION shrinkage reference R_sh.  The
operational normalisation is 'sparse_oracle' (author decision D1, 6 Oct 2026); the other two are reported as
diagnostics:
    'sparse_oracle':      (R_sh - R_AE) / (R_sh - R_sparse_oracle),
                          R_sparse_oracle = sum_{sparse a} r_a^Bayes + sum_{tail a} lambda_a
                          (the specified reference: Bayes on the six sparse coordinates and zero output on every
                          tail coordinate).  The total-risk numerator penalises tail degradation.  This reference
                          scale is NOT a bound for bottleneck-k models and NOT a measure of nonlinearity alone.
    'full_gap':           (R_sh - R_AE) / (R_sh - R_Bayes)   (diagnostic)
    'sparse_coordinates': sum_{sparse a} (r_a^sh - r_a^AE) / sum_{sparse a} (r_a^sh - r_a^Bayes)   (diagnostic)
On the T25 geometry at p = 0.1 the sparse_oracle reference itself closes about 24.0% of the full gap
(``full_gap_closure_of_sparse_reference``).  That is a property of this zero-tail reference, not an
upper bound: models that also recover tail energy can close more of the full gap.
    G2 holds if min over seeds of (numerator - m_mc * MCSE_numerator) / denominator >= gap_fraction.
    A denominator below gate['min_gap'] makes G2 'undefined', which fails the gate.
PASS iff G1(a), G1(b) and G2 hold and every seed produced a trained model.  PASS means only "keep
developing".  Any missing seed, failed fit or undefined quantity is a FAIL; there are no retries.
Success at N = 51,200 alone is not a pass (diagnostic only).
"""
from __future__ import annotations

import numpy as np


def full_gap_closure_of_sparse_reference(refs, model):
    """Full-gap closure attained by the zero-tail sparse reference itself (descriptive; not a bound)."""
    so = sparse_oracle(refs, model)
    return (refs['shrinkage'] - so) / (refs['shrinkage'] - refs['bayes'])


def sparse_oracle(refs, model):
    n = model.n_sparse
    return float(np.sum(refs['bayes_per_coordinate'][:n]) + np.sum(model.mu[n:]))


def evaluate(cells, refs, model, cfg):
    """cells: list of per-seed records at the gate (p, N) with keys seed, ok, R_AE, mcse, R_proj, R_shrink,
    sparse_num, sparse_num_mcse (sparse-coordinate shrinkage-minus-AE risk and its MCSE)."""
    G = cfg['gate']
    out = {'rule': __doc__, 'gate_cell': {'p': G['p'], 'N': G['N']}, 'seeds': [c['seed'] for c in cells],
           'reasons': []}
    if len(cells) != len(cfg['seeds']) or sorted(c['seed'] for c in cells) != sorted(cfg['seeds']):
        out.update(verdict='FAIL', reasons=['missing or duplicate seed at the gate cell'])
        return out
    if not all(c['ok'] for c in cells):
        out.update(verdict='FAIL', reasons=['a seed produced no trained model (no retries)'])
        return out
    R = np.array([c['R_AE'] for c in cells])
    se = np.array([c['mcse'] for c in cells])
    g1 = {}
    for name, key in (('projector', 'R_proj'), ('shrinkage', 'R_shrink')):
        d = R - np.array([c[key] for c in cells])
        every = bool(np.all(d + G['mc_multiplier'] * se < 0))
        spread = bool(d.mean() + G['seed_spread_multiplier'] * d.std(ddof=1) < 0)
        g1[name] = {'d': d.tolist(), 'mean': float(d.mean()), 'sd': float(d.std(ddof=1)),
                    'every_seed_beyond_mc': every, 'beyond_seed_spread': spread, 'holds': every and spread}
    sh = refs['shrinkage']
    dens = {'full_gap': sh - refs['bayes'], 'sparse_oracle': sh - sparse_oracle(refs, model),
            'sparse_coordinates': float(np.sum(refs['shrinkage_per_coordinate'][:model.n_sparse]
                                               - refs['bayes_per_coordinate'][:model.n_sparse]))}
    g2 = {}
    for name, den in dens.items():
        if name == 'sparse_coordinates':
            num = np.array([c['sparse_num'] for c in cells])
            nse = np.array([c['sparse_num_mcse'] for c in cells])
        else:
            num, nse = sh - R, se
        if not den > G['min_gap']:
            g2[name] = {'denominator': den, 'status': 'undefined', 'holds': False}
            continue
        lower = (num - G['mc_multiplier'] * nse) / den
        g2[name] = {'denominator': float(den), 'closure': (num / den).tolist(), 'closure_lower': lower.tolist(),
                    'min_lower': float(lower.min()), 'holds': bool(lower.min() >= G['gap_fraction'])}
    chosen = G['gap_definition']
    out.update(G1=g1, G2=g2, G2_definition_used=chosen,
               full_gap_closure_of_sparse_reference=float(full_gap_closure_of_sparse_reference(refs, model)))
    passed = g1['projector']['holds'] and g1['shrinkage']['holds'] and g2[chosen]['holds']
    if not g1['projector']['holds']:
        out['reasons'].append('G1 fails against the fitted projector')
    if not g1['shrinkage']['holds']:
        out['reasons'].append('G1 fails against the fitted shrinkage baseline')
    if not g2[chosen]['holds']:
        out['reasons'].append(f'G2 ({chosen}) fails or is undefined')
    out['verdict'] = 'PASS (keep developing only)' if passed else 'FAIL (stop at the time-box)'
    return out
