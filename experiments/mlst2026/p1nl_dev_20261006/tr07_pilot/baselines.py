"""Linear baselines for the TR07 pilot, fitted on the same training records as the autoencoder.

All maps act on WHITENED data y = Sigma^{-1/2} x (oracle Sigma) and are affine, f(y) = A y + b.  The exact
whitened clean risk of an affine map under the sparse-latent generator (truth mean 0, Cov(s) = K, noise
I, s and w independent) depends on second moments only:

    R(A, b) = tr((A - I) K (A - I)^T) + tr(A A^T) + ||b||^2.

Fitted rank-k projector (CW-PCA): A = U U^T with U the top-k eigenvectors of the centred sample covariance
Shat (divisor n); b = (I - A) ybar.
Fitted rank-k shrinkage: A = U diag(g) U^T with sample eigenvalues l_1 >= ... >= l_k and
g_i = (l_i - 1)_+ / l_i (the plug-in reduced-rank Wiener gain; noise variance is 1 in whitened units, so
l_i <= 1 means a nonpositive estimated signal variance and gives g_i = 0 without any division by a small
number); b = (I - A) ybar.  The top-k set by l equals the top-k set by the plug-in gain (l - 1)_+^2 / l,
which is nondecreasing in l.
"""
from __future__ import annotations

import numpy as np


def affine_risk(A, b, K):
    I = np.eye(len(A))
    E = A - I
    return float(np.trace(E @ K @ E.T) + np.trace(A @ A.T) + b @ b)


def sample_moments(Y):
    ybar = Y.mean(0)
    Yc = Y - ybar
    return ybar, Yc.T @ Yc / len(Y)


def fitted_projector(Y, k):
    ybar, S = sample_moments(Y)
    w, V = np.linalg.eigh(S)
    U = V[:, ::-1][:, :k]
    A = U @ U.T
    return A, (np.eye(len(A)) - A) @ ybar, {'sample_eigenvalues': w[::-1][:k + 1].tolist()}


def fitted_shrinkage(Y, k):
    ybar, S = sample_moments(Y)
    w, V = np.linalg.eigh(S)
    w, V = w[::-1], V[:, ::-1]
    lk = w[:k]
    g = np.where(lk > 1.0, (lk - 1.0) / np.where(lk > 1.0, lk, 1.0), 0.0)
    U = V[:, :k]
    A = (U * g) @ U.T
    return A, (np.eye(len(A)) - A) @ ybar, {'sample_eigenvalues': lk.tolist(), 'gains': g.tolist(),
                                            'zero_gain_count': int(np.sum(g == 0))}
