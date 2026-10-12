"""Noise covariance estimators of the meg_z kit (record layout c + C t, see common.py).

* ``ledoit_wolf_full``   Ledoit-Wolf (2004) shrinkage of the centred sample covariance (divisor n) toward
                         mu I, mu = tr S / d; the formula of sklearn.covariance.ledoit_wolf_shrinkage.
* ``kronecker_flipflop`` maximum-likelihood S_t (x) S_c (Dutilleul 1999; Werner, Jansson & Stoica 2008),
                         computed from the centred second moment only; normalised tr S_c = C.
* ``nearest_kronecker``  Van Loan-Pitsianis Frobenius-nearest Kronecker product (G4 descriptive).
* ``reference_metric``   the reference covariance Sigma_ref from the run 1-2 statistics (kind set by the
                         protocol: 'kronecker_flipflop' (dev default) or 'ledoit_wolf_full').

All estimators work from streamed moment statistics (``MomentStats``) so that the overlapping reference
pool (tens of thousands of windows) never has to be held in memory at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common import symmetric


# ----------------------------------------------------------------------------------------------- moments
@dataclass
class MomentStats:
    """Centred moment statistics of n records x_i in R^d.

    mean = (1/n) sum x_i;  S = (1/n) sum (x_i - mean)(x_i - mean)^T  (divisor n);
    sum_q4 = sum ||x_i - mean||^4 (needed by Ledoit-Wolf only)."""
    n: int
    mean: np.ndarray
    S: np.ndarray
    sum_q4: float
    n_eff: int | None = None          # effective number of independent records (overlapping windows; F14)

    @property
    def d(self) -> int:
        return len(self.mean)


def moment_stats(X: np.ndarray) -> MomentStats:
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        raise ValueError('at least two records required')
    mu = X.mean(0)
    Xc = X - mu
    q = np.einsum('ij,ij->i', Xc, Xc)
    return MomentStats(n, mu, symmetric(Xc.T @ Xc / n), float(np.sum(q * q)))


def moment_stats_streamed(chunks_factory) -> MomentStats:
    """Two passes over ``chunks_factory()`` (a callable returning an iterable of (n_i, d) arrays):
    first the mean, then the centred second moment and the fourth-power sum.  Exact (no approximation)."""
    n, s = 0, None
    for X in chunks_factory():
        X = np.asarray(X, dtype=np.float64)
        s = X.sum(0) if s is None else s + X.sum(0)
        n += len(X)
    if n < 2:
        raise ValueError('at least two records required')
    mu = s / n
    S = np.zeros((len(mu), len(mu)))
    q4 = 0.0
    for X in chunks_factory():
        Xc = np.asarray(X, dtype=np.float64) - mu
        S += Xc.T @ Xc
        q = np.einsum('ij,ij->i', Xc, Xc)
        q4 += float(np.sum(q * q))
    return MomentStats(n, mu, symmetric(S / n), q4)


# ----------------------------------------------------------------------------------------------- Ledoit-Wolf
def ledoit_wolf_from_stats(st: MomentStats) -> tuple[np.ndarray, dict]:
    """Ledoit-Wolf shrinkage toward mu I.  With S the divisor-n centred covariance:
    mu = tr S/d, delta = ||S - mu I||_F^2/d, beta = min(delta, (sum||x||^4/n - ||S||_F^2)/(n_eff d)),
    shrinkage = beta/delta, Sigma = (1 - shrinkage) S + shrinkage mu I.
    beta estimates the variance of S around its expectation: (1/n) sum ||x x^T - S||^2 / d is the per-record
    variance and it is divided by the number of *independent* records n_eff.  For independent records
    n_eff = n (sklearn's formula exactly); for the stride-1 overlapping reference pool n_eff is the number of
    non-overlapping (stride-T) windows of the same gaps (F14), which raises the shrinkage accordingly."""
    d, n, S = st.d, st.n, st.S
    n_eff = int(st.n_eff) if st.n_eff else n
    mu = float(np.trace(S) / d)
    delta = float(np.sum((S - mu * np.eye(d)) ** 2) / d)
    beta = float((st.sum_q4 / n - np.sum(S * S)) / (n_eff * d))
    beta = min(max(beta, 0.0), delta)
    shrinkage = 0.0 if delta == 0 else beta / delta
    cov = (1.0 - shrinkage) * S + shrinkage * mu * np.eye(d)
    return symmetric(cov), {'kind': 'ledoit_wolf_full', 'shrinkage': shrinkage, 'mu': mu, 'n': n, 'n_eff': n_eff}


def ledoit_wolf_full(X: np.ndarray) -> tuple[np.ndarray, dict]:
    return ledoit_wolf_from_stats(moment_stats(X))


# ----------------------------------------------------------------------------------------------- Kronecker MLE
def _inv_spd(a: np.ndarray) -> np.ndarray:
    w, q = np.linalg.eigh(symmetric(a))
    if w[0] <= 0 or not np.isfinite(w).all():
        raise np.linalg.LinAlgError('factor not SPD')
    return (q / w) @ q.T


def kronecker_flipflop_from_S(S: np.ndarray, C: int, T: int, max_iter: int = 500, tol: float = 1e-10):
    """Maximum-likelihood separable covariance S_t (x) S_c from a centred second moment S (d x d, divisor n,
    record index c + C t).  Flip-flop:  S_c <- (1/T) sum_{t,s} (S_t^{-1})_{ts} S4[:, t, :, s],
    S_t <- (1/C) sum_{c,e} (S_c^{-1})_{ce} S4[c, :, e, :], with S4[c, t, e, s] = S[c + C t, e + C s];
    normalised tr S_c = C.  Converged when the relative Frobenius change of S_t (x) S_c is <= tol.
    Returns (S_t, S_c, info); info['negloglik_per_record'] is (1/2)(T log|S_c| + C log|S_t| + tr(Sigma^-1 S))
    without the constant, which the iteration never increases."""
    S4 = np.asarray(S).reshape(C, T, C, T, order='F')
    St = np.eye(T)
    prev = None
    history = []
    status = 'iteration_limit'
    for it in range(1, max_iter + 1):
        Sc = symmetric(np.einsum('ts,ctes->ce', _inv_spd(St), S4) / T)
        St = symmetric(np.einsum('ce,ctes->ts', _inv_spd(Sc), S4) / C)
        scale = float(np.trace(Sc) / C)
        Sc, St = Sc / scale, St * scale
        cur = np.kron(St, Sc)
        nll = _negloglik(St, Sc, S4, C, T)
        history.append(nll)
        if prev is not None and np.linalg.norm(cur - prev) <= tol * np.linalg.norm(cur):
            status = 'converged'
            break
        prev = cur
    monotone = bool(all(b <= a + 1e-9 * max(1.0, abs(a)) for a, b in zip(history, history[1:])))
    return St, Sc, {'kind': 'kronecker_flipflop', 'iterations': it, 'status': status,
                    'negloglik_per_record': history[-1], 'monotone': monotone,
                    'trace_whitened': float(np.einsum('ts,ce,ctes->', _inv_spd(St), _inv_spd(Sc), S4))}


def _negloglik(St, Sc, S4, C, T) -> float:
    lc = np.linalg.slogdet(Sc)[1]
    lt = np.linalg.slogdet(St)[1]
    tr = float(np.einsum('ts,ce,ctes->', _inv_spd(St), _inv_spd(Sc), S4))
    return 0.5 * (T * lc + C * lt + tr)


def kronecker_flipflop(X: np.ndarray, C: int, T: int, **kw):
    st = moment_stats(X)
    St, Sc, info = kronecker_flipflop_from_S(st.S, C, T, **kw)
    info['n'] = st.n
    return np.kron(St, Sc), St, Sc, info


# ----------------------------------------------------------------------------------------------- NKP
def nearest_kronecker(A: np.ndarray, C: int, T: int):
    """Van Loan-Pitsianis: Frobenius-nearest B_t (x) B_c under the c + C t index (rank-one SVD of the
    rearrangement R[(t, t'), (c, c')] = A[c + C t, c' + C t']).  Returns (B_t, B_c, relative distance)."""
    R = np.asarray(A).reshape(T, C, T, C).transpose(0, 2, 1, 3).reshape(T * T, C * C)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    Bt = (np.sqrt(s[0]) * U[:, 0]).reshape(T, T)
    Bc = (np.sqrt(s[0]) * Vt[0]).reshape(C, C)
    if np.trace(Bc) < 0:
        Bt, Bc = -Bt, -Bc
    rel = float(np.sqrt(max(float(np.sum(s[1:] ** 2)), 0.0)) / np.linalg.norm(A, 'fro'))
    return symmetric(Bt), symmetric(Bc), rel


def is_kronecker(A: np.ndarray, C: int, T: int, tol: float = 1e-10) -> bool:
    return nearest_kronecker(A, C, T)[2] <= tol


# ----------------------------------------------------------------------------------------------- reference
REFERENCE_KINDS = ('kronecker_flipflop', 'ledoit_wolf_full')


def reference_metric(st: MomentStats, C: int, T: int, kind: str, flipflop: dict | None = None) -> dict:
    """Sigma_ref from the run 1-2 reference statistics.  Returns {'cov', 'kind', 'info', and for the
    Kronecker kind 'St', 'Sc'}."""
    if kind == 'ledoit_wolf_full':
        cov, info = ledoit_wolf_from_stats(st)
        return {'cov': cov, 'kind': kind, 'info': info}
    if kind == 'kronecker_flipflop':
        St, Sc, info = kronecker_flipflop_from_S(st.S, C, T, **(flipflop or {}))
        info['n'] = st.n
        return {'cov': np.kron(St, Sc), 'kind': kind, 'info': info, 'St': St, 'Sc': Sc}
    raise ValueError(f'unknown reference kind {kind!r}; allowed {REFERENCE_KINDS}')


def arm_estimate(X: np.ndarray, C: int, T: int, kind: str, flipflop: dict | None = None) -> tuple[np.ndarray, dict]:
    """Calibration-arm estimate Sigma_hat(m) from m calibration windows (records X, m x d)."""
    if kind == 'lw':
        return ledoit_wolf_full(X)
    if kind == 'kron':
        cov, _, _, info = kronecker_flipflop(X, C, T, **(flipflop or {}))
        return cov, info
    raise ValueError(f'unknown arm estimator {kind!r}')


def whitened_error(cov_a: np.ndarray, cov_ref: np.ndarray) -> dict:
    """Errors of cov_a in the metric of cov_ref: E = Sigma_ref^{-1/2} cov_a Sigma_ref^{-1/2} - I."""
    w, q = np.linalg.eigh(symmetric(cov_ref))
    if w[0] <= 0:
        raise np.linalg.LinAlgError('reference not SPD')
    li = (q / np.sqrt(w)) @ q.T
    e = symmetric(li @ cov_a @ li) - np.eye(len(w))
    ev = np.linalg.eigvalsh(e)
    return {'operator': float(max(abs(ev[0]), abs(ev[-1]))), 'relative_frobenius': float(np.linalg.norm(e) / np.sqrt(len(w))),
            'min_eig': float(ev[0] + 1), 'max_eig': float(ev[-1] + 1)}
