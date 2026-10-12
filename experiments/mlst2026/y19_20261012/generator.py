"""Y19 second synthetic generator G2: channel-dependent pulse-shape mixing.

Conventions (identical to the T25 oracle arm):
* channel-by-time record Z (C x T), vectorised by column stacking, index c + C t, so
  vec(A Z B) = (B^T (x) A) vec(Z) and a product subspace is range(U_t (x) U_c);
* separable physical noise Sigma = S_t (x) S_c with AR(1) Toeplitz factors of unit diagonal,
  built by ``t25.science.covariance`` (rho_c, rho_t re-dimensioned to C = 6, T = 64);
* whitened coordinates y = x L^{-1}, with (L, L^{-1}) = ``t25.science.powers(Sigma)`` (symmetric
  square root), clean whitened second moment K, data covariance M = K + I;
* records are drawn exactly as in ``t25.science.make_data``: Gaussian latents z, whitened clean
  signal z diag(sqrt(mu)) V^T, physical record (clean + white) @ L.

G2 construction (all deterministic; no sampling seed enters K):
1. Common temporal pulse basis (k_t = 3): double-exponential pulses
   p_j(t) = exp(-(t - t0)/tau_d,j) - exp(-(t - t0)/tau_r,j) for t > t0, else 0.
2. Channel patterns (k_c = 2): q_1[c] = exp(-(c - c0)^2 / 8) (broad), q_2[c] = (c - c0)/c0
   (gradient), c0 = (C - 1)/2.
3. Channel-dependent pulse shape: channel c sees every pulse delayed by phi * delta_c samples,
   delta_c = c - c0.  Template (i, j) is Z_ij[c, t] = q_i[c] p_j(t - phi delta_c).  At phi = 0, Z_ij = q_i p_j^T and the template span is exactly the
   product subspace range(P (x) Q); for phi != 0 the span is generally not a product subspace.
4. Top-k eigenvectors of K(phi): eigenvectors of K0 = L^{-1} B Omega B^T L^{-1} (B = physical
   template matrix, Omega = diag(pulse_w (x) channel_w)), ordered by K0 eigenvalue.
   Their span is L^{-1} span(templates), so the physical top-k signal subspace is exactly the
   span of the channel-dependent pulse templates.
5. Registered spectrum, identical at every phi (phi moves geometry only, as theta does in
   T20-S/T25): lambda_i = top_scale / i for i <= k, then a gap, then the same 1/j law,
   lambda_j = (top_scale/k - gap) (k + 1)/j for j > k.
6. Tail eigenvectors: a fixed Gaussian matrix R (key ``geometry_key``) projected off the top-k
   span and orthonormalised (as the seeded completion of T20-S/T25).
Column signs are fixed (largest-magnitude entry positive) so draws are host-independent.

``build(cfg)`` also accepts ``family == 't25'`` (the accepted T25/T20-S geometry, used only by the
smoke tests and the non-G2 timing benchmark).
"""
from __future__ import annotations

import hashlib

import numpy as np

from experiments.mlst2026.x3_nonlinear_metric_20261005 import guard as _guard

_guard.assert_direct()            # accepted T25 bytes before importing them

from experiments.mlst2026.t25_production_20260930 import science as t25   # noqa: E402

FAMILY = 'y19-g2-pulse-shape-mixing'


def seed_int(*key) -> int:
    """64-bit integer from the SHA-256 of the '/'-joined key (the t25.science.rng construction)."""
    return int.from_bytes(hashlib.sha256('/'.join(map(str, key)).encode()).digest()[:8], 'little')


def fix_signs(V: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(V), axis=0)
    s = np.sign(V[idx, np.arange(V.shape[1])])
    s[s == 0] = 1.0
    return V * s


def orth(B: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(B)
    return q * np.where(np.diag(r) < 0, -1.0, 1.0)


def pulse(u, tau_r: float, tau_d: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = u > 0
    out[m] = np.exp(-u[m] / tau_d) - np.exp(-u[m] / tau_r)
    return out


def ar1(rho: float, n: int) -> np.ndarray:
    return rho ** np.abs(np.arange(n)[:, None] - np.arange(n))


def spectrum(cfg: dict, d: int, k: int) -> np.ndarray:
    s = cfg['spectrum']
    top = float(s['top_scale']) / np.arange(1, k + 1)
    first_tail = top[-1] - float(s['gap'])
    if not first_tail > 0:
        raise ValueError('gap must leave a positive tail')
    tail = first_tail * (k + 1) / np.arange(k + 1, d + 1)
    mu = np.concatenate([top, tail])
    if np.any(np.diff(mu) >= 0):
        raise ValueError('spectrum not strictly decreasing')
    return mu


class G2Geometry:
    """Frozen G2 ingredients; ``V(phi)`` is the full whitened eigenbasis (column a <-> mu_a)."""

    family = FAMILY

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.C, self.T, self.k_c, self.k_t = int(cfg['C']), int(cfg['T']), int(cfg['kc']), int(cfg['kt'])
        self.k, self.d = self.k_c * self.k_t, self.C * self.T
        self.mu = spectrum(cfg, self.d, self.k)
        self.sigma = t25.covariance(cfg)
        self.L, self.Li = t25.powers(self.sigma)
        self.R = np.random.default_rng(seed_int(cfg['geometry_key'])).standard_normal((self.d, self.d - self.k))
        c0 = (self.C - 1) / 2.0
        self.delays = np.arange(self.C) - c0
        self.Q = np.stack([np.exp(-self.delays ** 2 / 8.0), self.delays / c0], 1)          # C x k_c
        pw = np.asarray(cfg['pulses']['weights'], float)
        cw = np.asarray(cfg['channel_patterns']['weights'], float)
        self.omega = np.kron(pw, cw)                                                        # i (channel) fastest
        if len(np.unique(np.round(self.omega, 12))) != self.k:
            raise ValueError('template weights must be distinct')
        self._cache = {}

    @property
    def Delta_k(self) -> float:
        return float(self.mu[self.k - 1] - self.mu[self.k])

    def pulses(self, shift: float = 0.0) -> np.ndarray:
        t = np.arange(self.T, dtype=float)
        p = self.cfg['pulses']
        return np.stack([pulse(t - float(p['onset']) - shift, tr, td) for tr, td in p['rise_decay']], 1)   # T x k_t

    def templates(self, phi: float) -> np.ndarray:
        """Physical template matrix B (d x k): column i + k_c j is vec(Z_ij), Z_ij[c, :] = q_i[c] p_j(. - phi delta_c)."""
        cols = []
        shifted = [self.pulses(float(phi) * dc) for dc in self.delays]                      # per channel, T x k_t
        for j in range(self.k_t):
            for i in range(self.k_c):
                Z = np.stack([self.Q[c, i] * shifted[c][:, j] for c in range(self.C)], 0)    # C x T
                cols.append(Z.reshape(-1, order='F'))
        return np.stack(cols, 1)

    def top_basis(self, phi: float):
        Bw = self.Li @ self.templates(phi)
        K0 = (Bw * self.omega) @ Bw.T
        w, U = np.linalg.eigh(0.5 * (K0 + K0.T))
        order = np.argsort(w)[::-1][:self.k]
        return fix_signs(U[:, order]), w[order]

    def V(self, phi: float) -> np.ndarray:
        key = float(phi)
        if key not in self._cache:
            U6, _ = self.top_basis(key)
            Rp = self.R - U6 @ (U6.T @ self.R)
            Qt = fix_signs(orth(Rp))
            self._cache[key] = np.concatenate([U6, Qt], 1)
        return self._cache[key].copy()

    def K(self, phi: float) -> np.ndarray:
        V = self.V(phi)
        K = (V * self.mu) @ V.T
        return 0.5 * (K + K.T)

    # phi = 0 product factors in whitened coordinates (start for the population optimum)
    @property
    def U_c0(self) -> np.ndarray:
        return orth(t25.powers(ar1(self.cfg['rho_c'], self.C))[1] @ self.Q)

    @property
    def U_t0(self) -> np.ndarray:
        return orth(t25.powers(ar1(self.cfg['rho_t'], self.T))[1] @ self.pulses(0.0))


def build(cfg: dict):
    """Geometry for a generator config: G2 or (smoke/benchmark only) the accepted T25 family."""
    fam = cfg.get('family', FAMILY)
    if fam == FAMILY:
        return G2Geometry(cfg)
    if fam == 't25':
        g = t25.geometry(cfg)
        g.sigma = t25.covariance(cfg)
        g.L, g.Li = t25.powers(g.sigma)
        g.family = 't25'
        return g
    raise ValueError(f'unknown generator family {fam!r}')


class PopulationModel:
    """Duck-typed model for ``p1nl_dev_20261006/tr09_first_order/core.py`` (Gaussian latents, p = 1)."""

    def __init__(self, g, phi: float):
        self.g, self.theta, self.p = g, float(phi), 1.0
        self.C, self.T, self.kc, self.kt, self.k, self.d = g.C, g.T, g.k_c, g.k_t, g.k, g.d
        self.n_sparse = self.k
        self.mu = np.asarray(g.mu, float).copy()
        self.V = g.V(self.theta)
        K = (self.V * self.mu) @ self.V.T
        self.K = 0.5 * (K + K.T)
        self.M = self.K + np.eye(self.d)
        self.ell = self.mu + 1.0

    @property
    def kappa(self) -> np.ndarray:
        return np.full(self.d, 3.0)


def draw(g, namespace: str, seed: int, phi: float, n: int):
    """Physical training pool (n x d) exactly as t25.science.make_data (oracle arm; no calibration pool)."""
    key = (namespace, seed, float(phi))
    clean = t25.rng(*key, 'signal').normal(size=(n, g.d)) @ (g.V(phi) * np.sqrt(g.mu)).T @ g.L
    noise = t25.rng(*key, 'noise').normal(size=(n, g.d)) @ g.L
    return clean + noise
