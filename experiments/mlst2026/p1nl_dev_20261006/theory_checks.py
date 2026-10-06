"""Numerical checks for MS plan/onthefly/P1NL_THEORY_NOTE.md (TR10, TR11).  Development only.

TR11.  Let x = t + n with n ~ N(0, Sigma).  Whiten with an assumed Sigma_hat:  y = W x, W = Sigma_hat^{-1/2},
so y = u + e with u = W t and e ~ N(0, Sigma_r), Sigma_r = W Sigma W^T.  For a weakly differentiable
estimator g acting on y (trained independently of this y), with Jacobian J(y):

    SURE_hat(y) = ||y - g(y)||^2 + 2 div g(y) - D            (Stein's estimate assuming e ~ N(0, I))
    R_hat       = E ||g(y) - u||^2                           (clean risk in the Sigma_hat metric)
    E[SURE_hat] - R_hat = tr Sigma_r - D - 2 E tr((Sigma_r - I) J(y)).

With Sigma_hat = Sigma: Sigma_r = I and the bias vanishes (Stein 1981).  Special cases: g = 0 gives
tr Sigma_r - D; g = identity gives D - tr Sigma_r; affine g = A y + b gives tr Sigma_r - D - 2 tr((Sigma_r - I) A).
The risk in the TRUE metric, E||Sigma^{-1/2}(f(x) - t)||^2, differs from R_hat by a change of metric that is
not a function of J alone; the identity above concerns the Sigma_hat metric.

TR10.  Population prices on the sparse-latent T25 generator (whitened, oracle Sigma): no-bottleneck price
of linearity R(full Wiener) - R(Bayes), and the bottleneck-k gain bound R(rank-k shrinkage) - R(Bayes) with
its split into sparse and tail coordinates.
"""
from __future__ import annotations

import numpy as np

from . import bayes
from .generator import SparseLatentModel


def sure_bias_check(g, jac_trace_fn, Sigma, Sigma_hat, sample_t, n_mc, gen):
    """Monte Carlo of E[SURE_hat] - R_hat and of the formula; returns both with standard errors.
    g: y -> g(y) (rows); jac_trace_fn(y, M) -> per-row tr(M J(y))."""
    D = len(Sigma)
    w, Q = np.linalg.eigh(Sigma_hat)
    W = (Q / np.sqrt(w)) @ Q.T
    Sr = W @ Sigma @ W.T
    Ls = np.linalg.cholesky(Sigma)
    t = sample_t(n_mc, gen)
    x = t + gen.normal(size=(n_mc, D)) @ Ls.T
    y, u = x @ W.T, t @ W.T
    gy = g(y)
    sure = ((y - gy) ** 2).sum(1) + 2 * jac_trace_fn(y, np.eye(D)) - D
    risk = ((gy - u) ** 2).sum(1)
    diff = sure - risk
    formula = np.trace(Sr) - D - 2 * jac_trace_fn(y, Sr - np.eye(D))
    return {'mc_bias': float(diff.mean()), 'mc_se': float(diff.std(ddof=1) / np.sqrt(n_mc)),
            'formula': float(formula.mean()), 'formula_se': float(formula.std(ddof=1) / np.sqrt(n_mc)),
            'paired_se': float((diff - formula).std(ddof=1) / np.sqrt(n_mc)), 'trace_Sigma_r': float(np.trace(Sr))}


def price_of_linearity(theta=0.25, ps=(1.0, 0.3, 0.1), ks=(3, 6)):
    rows = []
    for p in ps:
        m = SparseLatentModel(theta=theta, p=p)
        n = m.n_sparse
        for k in ks:
            r = bayes.references(m, k)
            sh_pc, b_pc = r['shrinkage_per_coordinate'], r['bayes_per_coordinate']
            rows.append({'theta': theta, 'p': p, 'k': k, 'R_projector': r['projector'], 'R_shrinkage': r['shrinkage'],
                         'R_full_wiener': r['full_wiener'], 'R_bayes': r['bayes'], 'bayes_error_estimate': r['bayes_error_estimate'],
                         'price_no_bottleneck': r['full_wiener'] - r['bayes'],
                         'gain_bound_bottleneck': r['shrinkage'] - r['bayes'],
                         'gain_bound_sparse_part': float(np.sum(sh_pc[:n] - b_pc[:n])),
                         'gain_bound_tail_part': float(np.sum(sh_pc[n:] - b_pc[n:])),
                         'projector_minus_shrinkage': r['projector'] - r['shrinkage']})
    return rows
