"""Second-moment-matched sparse-latent version of the accepted T25 generator (development only).

Whitened model (oracle Sigma, T25 conventions, Fortran vec index c + C t):

    y = s + w,   w ~ N(0, I_D),   s = sum_a sqrt(mu_a) z_a v_a,

where v_a are the columns of V(theta) (the eigenvectors of K(theta) = V diag(mu) V^T, mu strictly
decreasing, so column a carries eigenvalue mu_a) and the z_a are independent, zero mean, unit variance:

* the first ``n_sparse`` (= 6 = k) latents are Bernoulli-Gaussian, z = g xi / sqrt(p), g ~ N(0, 1),
  xi ~ Bernoulli(p), so E z^2 = 1 and E z^4 = 3/p (p = 1 is the Gaussian case);
* the remaining D - n_sparse latents (the tail) are standard Gaussian.

Hence Cov(s) = K(theta) for every p: all second moments, and therefore every population statement about
the linear classes, are those of T25.  Physical data are x = L y with L = Sigma^{1/2} symmetric (rows:
``Y @ L``), as in T25.  The geometry (V(theta), mu) and Sigma are taken from the accepted T25 code, whose
bytes are checked against the accepted closure (identity 01f2f82f...) before import.

T25 itself draws Gaussian latents as ``normal @ (V sqrt(mu))^T``; this module uses its own development
streams and namespaces, so its draws never coincide with any T25, X2 or X3 stream.
"""
from __future__ import annotations

import hashlib

import numpy as np

from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard as _guard

_guard.assert_direct()            # accepted T25 bytes before importing them

from experiments.mlst2026.t25_production_20260930 import science as t25   # noqa: E402

#: The T25 / X3 production geometry (T25 protocol "Population and class"; X3 BASE).
T25_GEOMETRY = {'C': 8, 'T': 32, 'kc': 2, 'kt': 3,
                'spectrum': {'top': [12.0, 11.0, 10.0, 9.0, 8.0, 7.0], 'gap': 4.0, 'tail_min': 0.02},
                'geometry_seed': 20260913, 'rho_c': 0.4, 'rho_t': 0.7}
#: A small geometry for unit tests (the X3 FIXTURE geometry).
FIXTURE_GEOMETRY = {'C': 3, 'T': 4, 'kc': 2, 'kt': 2,
                    'spectrum': {'top': [6.0, 5.0, 4.0, 3.0], 'gap': 1.5, 'tail_min': 0.05},
                    'geometry_seed': 20260913, 'rho_c': 0.4, 'rho_t': 0.7}


def rng(*key):
    """Independent stream per key (same construction as the T25 code: SHA-256 of the joined key)."""
    return np.random.default_rng(int.from_bytes(hashlib.sha256('/'.join(map(str, key)).encode()).digest()[:8], 'little'))


class SparseLatentModel:
    def __init__(self, geometry=None, theta=0.25, p=1.0, n_sparse=None):
        c = dict(T25_GEOMETRY if geometry is None else geometry)
        if not 0 < p <= 1:
            raise ValueError('activity p must lie in (0, 1]')
        self.c, self.theta, self.p = c, float(theta), float(p)
        self.g = t25.geometry(c)
        self.C, self.T, self.kc, self.kt = c['C'], c['T'], c['kc'], c['kt']
        self.k = self.kc * self.kt
        self.d = self.C * self.T
        self.n_sparse = self.k if n_sparse is None else int(n_sparse)
        self.mu = self.g.mu.copy()                     # strictly decreasing
        self.V = self.g.V(self.theta)                  # column a <-> mu_a
        self.K = self.g.K(self.theta)
        self.M = self.K + np.eye(self.d)               # whitened data covariance
        self.ell = self.mu + 1.0                       # eigenvalues of M
        self.sigma = t25.covariance(c)
        self.L, self.Li = t25.powers(self.sigma)
        self.sparse = np.arange(self.d) < self.n_sparse

    @property
    def kappa(self):
        """Fourth moment E z_a^4 of each latent (3/p sparse, 3 Gaussian)."""
        return np.where(self.sparse, 3.0 / self.p, 3.0)

    def latents(self, N, gen):
        z = gen.normal(size=(N, self.d))
        if self.p < 1:
            act = gen.random(size=(N, self.n_sparse)) < self.p
            z[:, :self.n_sparse] *= act / np.sqrt(self.p)
        return z

    def sample(self, N, gen):
        """Whitened (y, s, z): rows are records."""
        z = self.latents(N, gen)
        s = (z * np.sqrt(self.mu)) @ self.V.T
        y = s + gen.normal(size=(N, self.d))
        return y, s, z

    def sample_keyed(self, N, *key):
        """Whitened (y, s, z) from three independent keyed streams (latent amplitude, activity, noise).
        The activity stream is drawn for every p, so draws with the same key are paired across p
        (common random numbers): z_a = g_a 1[u_a < p] / sqrt(p) on the sparse coordinates."""
        g = rng(*key, 'latent').normal(size=(N, self.d))
        u = rng(*key, 'activity').random(size=(N, self.n_sparse))
        g[:, :self.n_sparse] *= (u < self.p) / np.sqrt(self.p)
        s = (g * np.sqrt(self.mu)) @ self.V.T
        y = s + rng(*key, 'noise').normal(size=(N, self.d))
        return y, s, g

    def physical(self, Y):
        return Y @ self.L
