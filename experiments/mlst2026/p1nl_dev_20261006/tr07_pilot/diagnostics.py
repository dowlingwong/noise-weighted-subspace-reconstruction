"""TR07 risk evaluation and representation diagnostics (whitened coordinates, oracle Sigma).

* Truth-referenced whitened clean risk E||shat - s||^2: exact for affine maps (second moments), Monte Carlo
  with a standard error for the autoencoder (fresh evaluation draws, shared by all models of a seed).
* Per-eigencoordinate risk r_a = E[(v_a^T (shat - s))^2], grouped into the k retained coordinates, the
  remaining sparse coordinates and the Gaussian tail (gain decomposition: shrinkage / sparse recovery
  beyond the bottleneck / tail).
* k-matched leakage S2_k = 1 - (top-k eigenvalue sum) / tr of the whitened output covariance.  Output
  covariance beyond k dimensions alone is not evidence of superposition (curvature also produces it).
* Known-latent recovery: per sparse latent a, R^2 of z_a from (i) the output coordinate v_a^T shat
  (univariate least squares with intercept) and (ii) the k-dimensional code (linear least squares with
  intercept).  Probes are fitted on the first half of the evaluation draws and scored on the second half.
* Interference: noise-free single-latent inputs y = t sqrt(lambda_a) v_a; response of latent coordinate b,
  rho_b(t) = v_b^T (f(y) - f(0)) / sqrt(lambda_b); I_ab = sum_t rho_b(t) t / sum_t t^2.  Off-diagonal entries
  measure cross-latent interference; diagonal entries are amplitude gains.  Exact, isolated recovery of the
  latent amplitude would give the identity; a shrinking estimator such as the factorised Bayes map gives
  zero off-diagonal interference but diagonal gains below one (pi(c) gamma < 1).
"""
from __future__ import annotations

import numpy as np


def affine_per_coordinate(A, b, model):
    """Exact per-eigencoordinate risk of y -> A y + b (whitened)."""
    V, lam = model.V, model.mu
    Ae = V.T @ A @ V
    be = V.T @ b
    E = Ae - np.eye(len(Ae))
    return np.einsum('ij,j,ij->i', E, lam, E) + np.einsum('ij,ij->i', Ae, Ae) + be ** 2


def mc_risk(Shat, S, model):
    """Monte Carlo whitened clean risk, its standard error, and per-coordinate risks with SEs."""
    e = Shat - S
    per_sample = (e ** 2).sum(1)
    ec = e @ model.V
    coord = ec ** 2
    n = len(e)
    return {'risk': float(per_sample.mean()), 'mcse': float(per_sample.std(ddof=1) / np.sqrt(n)),
            'per_coordinate': coord.mean(0), 'per_coordinate_se': coord.std(0, ddof=1) / np.sqrt(n),
            'per_sample': per_sample, 'per_sample_coord': coord}


def groups(model, k):
    idx = np.arange(model.d)
    return {'retained': idx < k, 'other_sparse': (idx >= k) & (idx < model.n_sparse), 'tail': idx >= model.n_sparse}


def group_sums(per_coordinate, model, k):
    return {g: float(per_coordinate[m].sum()) for g, m in groups(model, k).items()}


def leakage(Shat, k):
    Y = Shat - Shat.mean(0)
    w = np.linalg.eigvalsh(Y.T @ Y / (len(Y) - 1))[::-1]
    tr = float(w.sum())
    return None if tr <= 0 else float(1.0 - w[:k].sum() / tr)


def _r2(xfit, yfit, xev, yev):
    Xf = np.column_stack([np.ones(len(xfit)), xfit])
    beta, *_ = np.linalg.lstsq(Xf, yfit, rcond=None)
    pred = np.column_stack([np.ones(len(xev)), xev]) @ beta
    ss = float(((yev - yev.mean()) ** 2).sum())
    return None if ss <= 0 else float(1.0 - ((yev - pred) ** 2).sum() / ss)


def recovery(Shat, codes, Z, model, fit_fraction=0.5):
    n = len(Z)
    h = int(n * fit_fraction)
    out = {'output_r2': [], 'code_r2': []}
    for a in range(model.n_sparse):
        o = Shat @ model.V[:, a]
        out['output_r2'].append(_r2(o[:h], Z[:h, a], o[h:], Z[h:, a]))
        out['code_r2'].append(None if codes is None else _r2(codes[:h], Z[:h, a], codes[h:], Z[h:, a]))
    return out


def interference(f_whitened, model, amplitudes):
    n = model.n_sparse
    lam = model.mu[:n]
    t = np.asarray(amplitudes, dtype=float) / np.sqrt(model.p)
    f0 = f_whitened(np.zeros((1, model.d)))[0]
    I = np.zeros((n, n))
    for a in range(n):
        Y = np.outer(t * np.sqrt(lam[a]), model.V[:, a])
        R = (f_whitened(Y) - f0) @ model.V[:, :n] / np.sqrt(lam)          # rho_b(t), shape (len t, n)
        I[a] = (R * t[:, None]).sum(0) / (t ** 2).sum()
    return I
