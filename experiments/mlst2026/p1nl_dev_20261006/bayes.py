"""Population references for the sparse-latent T25 generator (development; TR07 / TR10).

Whitened, eigenbasis coordinates c_a = v_a^T y = sqrt(lambda_a) z_a + w_a are independent across a
(independent latents along K's eigenbasis, white noise), so the Bayes posterior mean factorises:

    E[s | y] = sum_a v_a E[sqrt(lambda_a) z_a | c_a].

Sparse coordinate (Bernoulli-Gaussian z = g xi / sqrt(p)): given active, c ~ N(0, lambda/p + 1); given
inactive, c ~ N(0, 1).  With gamma = (lambda/p) / (lambda/p + 1) and pi(c) = P(active | c),

    E[sqrt(lambda) z | c] = pi(c) gamma c.

Gaussian (tail) coordinate, and every coordinate when p = 1: the Wiener map lambda/(lambda + 1) c.
lambda = 0: the posterior mean is 0.

Per-coordinate Bayes risk (MMSE): r = lambda - E[(pi gamma c)^2], computed by one-dimensional quadrature
in each mixture component (y = sigma_1 t and y = t with t ~ N(0, 1)), with two independent rules
(adaptive scipy quad, and composite Simpson on [-14, 14] at two resolutions) and a reported discrepancy.  This is a
numerical-integration result with a controlled error, not a Monte Carlo estimate.

Linear population references (whitened clean risk, oracle Sigma, truth mean 0):
* rank-k projector onto the top-k eigenvectors: sum_{a in top k} 1 + sum_{a not in top k} lambda_a;
* rank-k shrinkage (reduced-rank Wiener; the optimal rank-k linear map): keep the k largest
  lambda^2/(lambda + 1) with gains lambda/(lambda + 1): sum_S lambda/(lambda+1) + sum_{not S} lambda;
* full Wiener: sum_a lambda_a/(lambda_a + 1).
The unconstrained Bayes risk is a LOWER BOUND for every estimator, including every bottleneck-k model;
it is not the bottleneck-k optimum.
"""
from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import logsumexp

LOG2PI = np.log(2 * np.pi)


def _log_normal(y, var):
    return -0.5 * (LOG2PI + np.log(var) + y * y / var)


def active_probability(c, lam, p):
    """pi(c) = P(active | c) for a Bernoulli-Gaussian coordinate; 1 when p = 1; stable in the tails."""
    c = np.asarray(c, dtype=float)
    if p >= 1.0:
        return np.ones_like(c)
    if lam <= 0:
        return np.full_like(c, p)
    v1 = lam / p + 1.0
    l1 = np.log(p) + _log_normal(c, v1)
    l0 = np.log1p(-p) + _log_normal(c, 1.0)
    return np.exp(l1 - np.logaddexp(l1, l0))


def scalar_posterior_mean(c, lam, p):
    c = np.asarray(c, dtype=float)
    if lam <= 0:
        return np.zeros_like(c)
    if p >= 1.0:
        return lam / (lam + 1.0) * c
    gamma = (lam / p) / (lam / p + 1.0)
    return active_probability(c, lam, p) * gamma * c


def _second_moment_integrands(lam, p):
    gamma = (lam / p) / (lam / p + 1.0)
    s1 = np.sqrt(lam / p + 1.0)
    phi = lambda t: np.exp(-0.5 * t * t) / np.sqrt(2 * np.pi)          # noqa: E731
    f1 = lambda t: phi(t) * (active_probability(s1 * t, lam, p) * gamma * s1 * t) ** 2   # noqa: E731
    f0 = lambda t: phi(t) * (active_probability(t, lam, p) * gamma * t) ** 2             # noqa: E731
    return f1, f0


def _simpson(f, n, half_width=14.0):
    """Composite Simpson rule on [-half_width, half_width] (the N(0,1)-weighted integrands are below
    1e-40 beyond 14 standard deviations); n intervals (even)."""
    t = np.linspace(-half_width, half_width, n + 1)
    return float(integrate.simpson(f(t), x=t))


def sparse_mmse(lam, p, quad_tol=1e-13, simpson_intervals=(20000, 40000)):
    """Bayes risk of one Bernoulli-Gaussian coordinate and its numerical-error report."""
    if lam <= 0:
        return 0.0, {'method': 'zero signal', 'error_bound': 0.0}
    if p >= 1.0:
        return lam / (lam + 1.0), {'method': 'Gaussian closed form', 'error_bound': 0.0}
    f1, f0 = _second_moment_integrands(lam, p)
    q1, e1 = integrate.quad(f1, -np.inf, np.inf, epsabs=quad_tol, epsrel=quad_tol, limit=500)
    q0, e0 = integrate.quad(f0, -np.inf, np.inf, epsabs=quad_tol, epsrel=quad_tol, limit=500)
    m2_quad = p * q1 + (1 - p) * q0
    simp = [p * _simpson(f1, n) + (1 - p) * _simpson(f0, n) for n in simpson_intervals]
    risk = lam - m2_quad
    err = max(abs(m2_quad - simp[-1]), abs(simp[0] - simp[-1]), p * e1 + (1 - p) * e0)
    return float(risk), {'method': 'quadrature', 'second_moment_quad': m2_quad, 'second_moment_simpson': simp,
                         'quad_abserr': p * e1 + (1 - p) * e0, 'error_bound': float(err)}


def bayes_risk(mu, n_sparse, p):
    """Total whitened Bayes risk (sum over coordinates) and per-coordinate risks; error bound summed."""
    per, err = [], 0.0
    for a, lam in enumerate(mu):
        if a < n_sparse:
            r, info = sparse_mmse(float(lam), p)
            err += info['error_bound']
        else:
            r = float(lam / (lam + 1.0))
        per.append(r)
    return float(np.sum(per)), np.asarray(per), err


def linear_references(mu, k):
    mu = np.asarray(mu, dtype=float)
    order = np.argsort(mu)[::-1]
    top = np.zeros(len(mu), bool)
    top[order[:k]] = True
    gain = mu ** 2 / (mu + 1.0)
    keep = np.zeros(len(mu), bool)
    keep[np.argsort(gain)[::-1][:k]] = True
    wiener = mu / (mu + 1.0)
    return {'projector': float(np.sum(np.where(top, 1.0, mu))),
            'shrinkage': float(np.sum(np.where(keep, wiener, mu))),
            'full_wiener': float(np.sum(wiener)),
            'projector_per_coordinate': np.where(top, 1.0, mu),
            'shrinkage_per_coordinate': np.where(keep, wiener, mu),
            'shrinkage_set_equals_projector_set': bool(np.array_equal(top, keep))}


def references(model, k, p=None):
    """All population references for a SparseLatentModel at bottleneck/rank k."""
    p = model.p if p is None else p
    lin = linear_references(model.mu, k)
    rb, per, err = bayes_risk(model.mu, model.n_sparse, p)
    return {**lin, 'bayes': rb, 'bayes_error_bound': err, 'bayes_per_coordinate': per,
            'gap_shrinkage_to_bayes': lin['shrinkage'] - rb}


def posterior_mean(model, Y, p=None):
    """Factorised Bayes posterior mean of s given whitened Y (rows are records)."""
    p = model.p if p is None else p
    Cc = Y @ model.V
    out = np.empty_like(Cc)
    for a, lam in enumerate(model.mu):
        out[:, a] = scalar_posterior_mean(Cc[:, a], lam, p if a < model.n_sparse else 1.0)
    return out @ model.V.T


def brute_force_posterior_mean(model, Y, p=None):
    """Reference: enumerate every activity pattern of the sparse latents in full whitened coordinates;
    s | pattern ~ N(0, K_S), posterior weights by the Gaussian marginal likelihood.  2^n_sparse patterns:
    unit-test sizes only."""
    p = model.p if p is None else p
    n = model.n_sparse
    V, mu, d = model.V, model.mu, model.d
    logs, means = [], []
    for bits in range(2 ** n):
        act = np.array([(bits >> i) & 1 for i in range(n)], bool)
        lam = mu.copy()
        lam[:n] = np.where(act, mu[:n] / p, 0.0)
        KS = (V * lam) @ V.T
        Cov = KS + np.eye(d)
        sign, logdet = np.linalg.slogdet(Cov)
        sol = np.linalg.solve(Cov, Y.T)                    # d x n_records
        prior = act.sum() * np.log(p) + (n - act.sum()) * np.log1p(-p) if p < 1 else (0.0 if act.all() else -np.inf)
        logs.append(prior - 0.5 * (logdet + np.einsum('ij,ji->i', Y, sol) + d * LOG2PI))
        means.append((KS @ sol).T)
    logs = np.array(logs)
    wts = np.exp(logs - logsumexp(logs, axis=0))
    return np.einsum('pr,prd->rd', wts, np.array(means))
