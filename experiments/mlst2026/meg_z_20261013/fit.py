"""Learners and scores of one record (design §D.2 fit.py).  The two linear classes are fitted with the T25
recipe, imported unchanged (``t25_production_20260930/science.py``, bytes checked against the accepted T25
closure by the X3 guard before import):

    x, mean, S = t25.moments(train, N)                  fitted mean, centred second moment (divisor N)
    starts     = t25.starts(S, C, T, k_c, k_t, key)     HOSVD + 3 random starts
    B_F        = t25.structured(W, S, starts, ...)      NFPA on the physical product Grassmannian, metric W
    B_U        = t25.unrestricted(W, S, k)              EMPCA population form (W-whitened top-k)
    A          = t25.map_for(B, W);  b = (I - A) mean

Scores (all in measurement coordinates, on the fixed evaluation set of the subject and theta):
    clean_ref             mean (A x + b - s)^T W0 (A x + b - s), W0 = Sigma_ref^{-1}   (primary)
    clean_euclidean       the same with the identity metric
    clean_ref_semi        t25.risks(A, A mu_n + b, R_s, Sigma_n, W0)['clean']: exact in the Gaussian signal law,
                          evaluation-noise mean mu_n and covariance Sigma_n empirical
    readout_nmse_a11/a21  GLS (W0) amplitudes of A x + b on the 6 physical templates; mean squared error of
                          alpha_11 and alpha_21 divided by their latent variances
"""
from __future__ import annotations

import time

import numpy as np

from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard as _guard

_guard.assert_direct()                                                        # accepted T25 bytes before import

from experiments.mlst2026.t25_production_20260930 import science as t25      # noqa: E402

from .common import symmetric                                                 # noqa: E402


def metric(cov: np.ndarray) -> np.ndarray:
    return symmetric(np.linalg.inv(symmetric(cov)))


def fit_pair(train: np.ndarray, N: int, W: np.ndarray, C: int, T: int, kc: int, kt: int, learners: dict, key) -> dict:
    """Fit both classes on the first N training records under the metric W."""
    t0 = time.time()
    x, mu, S = t25.moments(train, N)
    d = C * T
    initial = t25.starts(S, C, T, kc, kt, key)
    BF, diag, factors = t25.structured(W, S, initial, learners['max_iter'], learners['gradient_tolerance'])
    t1 = time.time()
    BU = t25.unrestricted(W, S, kc * kt)
    I = np.eye(d)
    maps = {}
    for cls, B in (('U', BU), ('F', BF)):
        A = t25.map_for(B, W)
        maps[cls] = (A, (I - A) @ mu)
    restart_maps = []
    for uc, ut in factors:
        A = t25.map_for(np.kron(ut, uc), W)
        restart_maps.append((A, (I - A) @ mu))
    restarts = [{'objective': r['objective'], 'gradient_norm': r['gradient_norm'], 'status': r['status'],
                 'iterations': len(r['trajectory']) - 1} for r in diag['restarts']]
    return {'maps': maps, 'restart_maps': restart_maps, 'selected': diag['selected'], 'restarts': restarts,
            'training_sha': t25.digest(x), 'seconds_F': t1 - t0, 'seconds_total': time.time() - t0}


class Evaluation:
    """Fixed evaluation set of one subject and theta: records X = S + noise, clean S, latents alpha."""

    def __init__(self, S, noise, alpha, templates, variances, W0, R_s):
        self.S, self.noise, self.alpha = S, noise, alpha
        self.X = S + noise
        self.Bs, self.var = templates, np.asarray(variances, float)
        self.W0, self.R_s = W0, R_s
        self.mu_n = noise.mean(0)
        nc = noise - self.mu_n
        self.Sigma_n = symmetric(nc.T @ nc / len(noise))
        G = templates.T @ W0 @ templates
        self.readout_op = np.linalg.solve(G, templates.T @ W0)              # 6 x d

    def primary(self, A: np.ndarray, b: np.ndarray) -> float:
        R = self.X @ A.T + b - self.S
        return float(np.mean(np.einsum('ij,ij->i', R @ self.W0, R)))

    def scores(self, A: np.ndarray, b: np.ndarray) -> dict:
        Y = self.X @ A.T + b
        R = Y - self.S
        clean = float(np.mean(np.einsum('ij,ij->i', R @ self.W0, R)))
        eucl = float(np.mean(np.einsum('ij,ij->i', R, R)))
        semi = t25.risks(A, A @ self.mu_n + b, self.R_s, self.Sigma_n, self.W0)
        ah = Y @ self.readout_op.T
        err = ah - self.alpha
        return {'clean_ref': clean, 'clean_euclidean': eucl, 'clean_ref_semi_analytic': semi['clean'],
                'semi_signal': semi['signal'], 'semi_noise': semi['noise'], 'semi_offset': semi['offset'],
                'readout_nmse_a11': float(np.mean(err[:, 0] ** 2) / self.var[0]),
                'readout_nmse_a21': float(np.mean(err[:, 3] ** 2) / self.var[3])}


DEFAULT_FLAG_RULE = {'line_search_gradient_max': 1e-6}


def optimization_flags(restarts: list, selected: int, spread: float, D: float, rule: dict | None = None) -> dict:
    """Two optimization-limited flags for one record (the T25 optimizer itself is unchanged).

    * ``optimization_limited`` (refined rule, delta review): the selected restart ended ``iteration_limit``, OR it
      ended ``line_search_stopped`` with normalised gradient norm > line_search_gradient_max (1e-6 = 100 x the
      recipe tolerance), OR the restart clean-risk spread >= |D|.  Under an ill-conditioned (Ledoit-Wolf) metric
      ``line_search_stopped`` near 1e-8 marks the floating-point floor of the Armijo search, not non-convergence.
    * ``optimization_limited_t25_rule`` (unchanged T25 rule, reported alongside): selected restart not
      ``stationary``, OR spread >= |D|."""
    rule = rule or DEFAULT_FLAG_RULE
    st = restarts[selected]['status']
    g = float(restarts[selected]['gradient_norm'])
    refined = (st == 'iteration_limit' or (st == 'line_search_stopped' and g > float(rule['line_search_gradient_max']))
               or spread >= abs(D))
    return {'optimization_limited': bool(refined), 'optimization_limited_t25_rule': bool(st != 'stationary' or spread >= abs(D)),
            'selected_status': st, 'selected_gradient_norm': g}


def score_fit(fit: dict, ev: Evaluation, primary: str = 'clean_ref', flag_rule: dict | None = None) -> dict:
    out = {cls: ev.scores(*fit['maps'][cls]) for cls in ('U', 'F')}
    if primary != 'clean_ref':
        raise ValueError('the restart spread is computed for the primary score clean_ref only')
    rr = [ev.primary(A, b) for A, b in fit['restart_maps']]
    D = out['F'][primary] - out['U'][primary]
    spread = float(max(rr) - min(rr))
    sel = fit['selected']
    return {'U': out['U'], 'F': out['F'], 'D': D,
            'D_scores': {k: out['F'][k] - out['U'][k] for k in out['U']},
            'restart_scores': rr, 'restart_spread': spread, 'selected': sel, 'restarts': fit['restarts'],
            **optimization_flags(fit['restarts'], sel, spread, D, flag_rule),
            'training_sha': fit['training_sha'], 'seconds_F': fit['seconds_F'], 'seconds_fit': fit['seconds_total']}
