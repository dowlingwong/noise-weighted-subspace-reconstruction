"""E2X+ plug-in quantities: certificate, approximation cost and first-order crossing computed from data.

Everything population-level is computed by code imported unchanged (hash-pinned in protocol.json):
  - y19_20261012.predictions.constants / certificate   (TR09 first-order path, as for the frozen predictions)
  - p1nl_dev_20261006.tr09_first_order.core             (product optimum, c_U, c_F, B)
  - t25_production_20260930.science                     (moments, estimate, powers)
This module only builds the plug-in matrix K_hat and a duck-typed model around it.  No learner is fitted.
"""
from __future__ import annotations

import numpy as np

from experiments.mlst2026.t25_production_20260930 import science as t25
from experiments.mlst2026.y19_20261012 import predictions as P

ARMS = ('oracle', 'kron256', 'kron1024', 'full256', 'full1024')
NO_DECISION = ('nonpositive_gap', 'H_not_pos_def', 'exception', 'nonfinite')


class _Starts:
    """Stands in for ``model.g`` in core.population_product_optimum: its U_c0 / U_t0 are the
    practitioner-feasible partial-trace start of the plug-in top-k projector (never the oracle factors)."""

    def __init__(self, U_c0: np.ndarray, U_t0: np.ndarray):
        self.U_c0, self.U_t0 = U_c0, U_t0


class PluginModel:
    """Duck-typed model for TR09 core.py / Y19 predictions.constants, built from a plug-in K_hat.

    K is the clipped plug-in clean second moment; mu its eigenvalues (descending), V its eigenvectors,
    M = K + I, ell = mu + 1.  Gaussian latents (kappa = 3, p = 1), as for the population predictions."""

    def __init__(self, K_hat: np.ndarray, C: int, T: int, kc: int, kt: int, label: str = 'plugin'):
        A = 0.5 * (K_hat + K_hat.T)
        w, V = np.linalg.eigh(A)
        order = np.argsort(w)[::-1]
        w, V = w[order], V[:, order]
        self.n_clipped = int(np.sum(w < 0))
        self.clipped_mass = float(-np.sum(w[w < 0]))
        mu = np.maximum(w, 0.0)
        self.C, self.T, self.kc, self.kt = int(C), int(T), int(kc), int(kt)
        self.k, self.d = self.kc * self.kt, self.C * self.T
        self.theta, self.p, self.n_sparse = label, 1.0, self.k
        self.mu, self.V = mu, V
        K = (V * mu) @ V.T
        self.K = 0.5 * (K + K.T)
        self.M = self.K + np.eye(self.d)
        self.ell = self.mu + 1.0
        self.gap = float(mu[self.k - 1] - mu[self.k])
        Uk = V[:, :self.k]
        P4 = (Uk @ Uk.T).reshape(self.C, self.T, self.C, self.T, order='F')
        Uc = np.linalg.eigh(np.einsum('ctdt->cd', P4))[1][:, -self.kc:]
        Ut = np.linalg.eigh(np.einsum('ctcs->ts', P4))[1][:, -self.kt:]
        self.g = _Starts(Uc, Ut)

    @property
    def kappa(self) -> np.ndarray:
        return np.full(self.d, 3.0)


def whiteners(sigma: np.ndarray, calibration: np.ndarray, C: int, T: int) -> dict:
    """W = Sigma^{-1/2} per arm; estimated arms use the T25 estimators unchanged (fixed 10% shrinkage)."""
    out = {'oracle': t25.powers(sigma)[1]}
    for m in (256, 1024):
        covs, _ = t25.estimate(calibration[:m], C, T)
        out[f'kron{m}'] = t25.powers(covs['kron'])[1]
        out[f'full{m}'] = t25.powers(covs['full'])[1]
    return out


def k_hat(training: np.ndarray, N: int, W: np.ndarray) -> np.ndarray:
    """K_hat = W S W - I with S the centred divisor-N sample covariance of the first N records (t25.moments)."""
    _, _, S = t25.moments(training, N)
    return W @ S @ W - np.eye(W.shape[0])


def certificate_only(Kh: np.ndarray, C: int, T: int, kc: int, kt: int) -> dict:
    m = PluginModel(Kh, C, T, kc, kt)
    c = P.certificate(m)
    return {'B_cert': c['B_cert'], 'Delta_k': c['Delta_k'], 'n_clipped': m.n_clipped,
            'clipped_mass': m.clipped_mass, 'k_minus_cap': c['partial_trace']['k_minus_cap']}


def plugin_constants(Kh: np.ndarray, C: int, T: int, kc: int, kt: int, rng) -> dict:
    """Full plug-in constants through Y19 predictions.constants (8 restarts: HOSVD, partial-trace start, 6 random)."""
    m = PluginModel(Kh, C, T, kc, kt)
    if not m.gap > 0:
        return {'available': False, 'cause': 'nonpositive_gap', 'gap': m.gap, 'n_clipped': m.n_clipped}
    try:
        row = P.constants(m, rng, n_random=6)
    except Exception as exc:                       # noqa: BLE001  (protocol v2.1 section 8: recorded, no decision)
        return {'available': False, 'cause': 'exception', 'error': f'{type(exc).__name__}: {exc}', 'n_clipped': m.n_clipped}
    if not all(np.isfinite([row['a_best'], row['cU'], row['cF'], row['B']])):
        return {'available': False, 'cause': 'nonfinite', 'n_clipped': m.n_clipped,
                **{k: (float(row[k]) if np.isfinite(row[k]) else None) for k in ('a_best', 'cU', 'cF', 'B')}}
    keep = ('a_best', 'cU', 'cF', 'B', 'B_cert', 'Delta_k', 'H_min_eig', 'N_star_pred', 'N_star_interval',
            'all_starts_reach_best', 'starts_reaching_best', 'n_starts', 'checks', 'gradient_norm_at_optimum')
    out = {k: row[k] for k in keep}
    out['N_star_plug'] = out.pop('N_star_pred')
    out['N_star_plug_interval'] = out.pop('N_star_interval')
    out['n_clipped'], out['clipped_mass'], out['gap'] = m.n_clipped, m.clipped_mass, m.gap
    if out['N_star_plug'] is not None:
        out['available'], out['cause'] = True, None
    elif not row['a_best'] > P.A_ZERO_TOL:
        out['available'], out['cause'] = False, 'a_zero'
    elif not row['B'] < 0:
        out['available'], out['cause'] = False, 'B_nonnegative'
    else:
        out['available'], out['cause'] = False, 'other'
    if not row['H_min_eig'] > 0:
        out['available'], out['cause'], out['N_star_plug'] = False, 'H_not_pos_def', None
    return out


def debias(N: int, c: dict) -> dict:
    """Debiased arm (protocol section 4): a_db = a_hat - (cU_hat - cF_hat)/N; N*_db = (cU - cF)/a_db - 1."""
    if c.get('cause') in NO_DECISION or 'a_best' not in c:
        return {'a_db': None, 'N_star_db': None, 'db_cause': c.get('cause')}
    gap_c = c['cU'] - c['cF']
    a_db = c['a_best'] - gap_c / N
    if gap_c <= 0:
        return {'a_db': a_db, 'N_star_db': None, 'db_cause': 'c_gap_nonpositive'}
    if a_db <= 0:
        return {'a_db': a_db, 'N_star_db': None, 'db_cause': 'a_db_nonpositive'}
    return {'a_db': a_db, 'N_star_db': gap_c / a_db - 1, 'db_cause': None}


def rule_choice(N: int, c: dict) -> str | None:
    """Naive arm: 'F' (restrict) iff N < N*_plug; a_hat <= 1e-9 -> 'F' (ZC convention); a_hat > 0 with
    B_hat >= 0 (first-order contrast positive at every N) -> 'U'; nonpositive gap or Hessian not positive
    definite -> None (no decision, counted against the rule)."""
    if c.get('available'):
        return 'F' if N < c['N_star_plug'] else 'U'
    if c.get('cause') == 'a_zero':
        return 'F'
    if c.get('cause') == 'B_nonnegative':
        return 'U'
    return None


def rule_choice_db(N: int, c: dict) -> str | None:
    """Debiased arm: 'F' iff a_db <= 0 or N < N*_db; (cU - cF) <= 0 -> 'U'; no-decision causes as the naive arm."""
    cause = c.get('db_cause')
    if cause in NO_DECISION or c.get('cause') in NO_DECISION:
        return None
    if cause == 'a_db_nonpositive':
        return 'F'
    if cause == 'c_gap_nonpositive':
        return 'U'
    return 'F' if N < c['N_star_db'] else 'U'
