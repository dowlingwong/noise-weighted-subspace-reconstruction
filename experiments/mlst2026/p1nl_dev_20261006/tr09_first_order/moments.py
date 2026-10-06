"""Exact finite-N moments of the sample-covariance error in the eigenbasis of M (TR09 note, section 2).

Coordinates y_a = sqrt(lambda_a) z_a + w_a are independent, zero mean, symmetric, with
ell_a = lambda_a + 1 = E y_a^2 and mu4_a = E y_a^4 = lambda_a^2 kappa_a + 6 lambda_a + 3 (kappa_a = E z_a^4).

* Known mean, S0 = (1/N) sum y y^T:   E S0 = M;  Var S0_ab = ell_a ell_b / N (a != b);
  Var S0_aa = (mu4_a - ell_a^2) / N = [2 ell_a^2 + lambda_a^2 (kappa_a - 3)] / N.
* Sample-centred, divisor N (T25), Shat = S0 - ybar ybar^T:   E Shat = (N - 1)/N M;
  Var Shat_ab = ell_a ell_b (N - 1) / N^2 (a != b);
  Var Shat_aa = ((N - 1)/N)^2 [mu4_a / N - ell_a^2 (N - 3) / (N (N - 1))].
* All covariances between DISTINCT entries vanish (independence, zero mean, symmetry), for both estimators.
* But entries are not independent: for the known-mean error E = S0 - M and a != b,
  E[E_ab^2 E_aa] = ell_b (mu4_a - ell_a^2) / N^2   (depends on kappa_a).
"""
from __future__ import annotations

import numpy as np


def mu4(lam, kappa):
    return lam ** 2 * kappa + 6 * lam + 3


def known_mean(lam, kappa, N):
    ell = lam + 1
    off = np.outer(ell, ell) / N
    diag = (mu4(lam, kappa) - ell ** 2) / N
    V = off.copy()
    np.fill_diagonal(V, diag)
    return V


def centred(lam, kappa, N):
    ell = lam + 1
    V = np.outer(ell, ell) * (N - 1) / N ** 2
    diag = ((N - 1) / N) ** 2 * (mu4(lam, kappa) / N - ell ** 2 * (N - 3) / (N * (N - 1)))
    np.fill_diagonal(V, diag)
    return V


def third_cross(lam, kappa, N, a, b):
    ell = lam + 1
    return ell[b] * (mu4(lam, kappa)[a] - ell[a] ** 2) / N ** 2
