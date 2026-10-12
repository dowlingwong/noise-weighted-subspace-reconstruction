"""Nearest Kronecker product (Van Loan & Pitsianis 1993) in the T25 column-stacking convention.

Convention (identical to t25_production_20260930/science.py and the Y19 generator): a channel-by-time
record Z (C x T) is vectorised by column stacking, vec index c + C t, so a separable covariance is
Sigma = Sigma_t (x) Sigma_c = np.kron(Sigma_t, Sigma_c), i.e. Sigma[c + C t, c' + C t'] =
Sigma_t[t, t'] Sigma_c[c, c'].

Rearrangement: R[(t, t'), (c, c')] = S[c + C t, c' + C t'] (row t T + t', column c C + c').  For
S = A_t (x) A_c, R = vec(A_t) vec(A_c)^T has rank one, so the leading singular triple of R returns
the factors exactly (up to the scale split).  For a general S the leading triple gives the
Frobenius-nearest Kronecker product (Van Loan & Pitsianis).  Both factors are then symmetrised and
projected onto the PSD cone (negative eigenvalues set to 0); for a PSD input the Van Loan-Pitsianis
factors are already symmetric PSD up to rounding, so the projection only removes rounding (its size
is reported).  The scale is split so that ||A_c||_F = ||A_t||_F.
"""
from __future__ import annotations

import numpy as np


def rearrange(S: np.ndarray, C: int, T: int) -> np.ndarray:
    """R (T^2 x C^2) with R[t T + t', c C + c'] = S[c + C t, c' + C t']."""
    S = np.asarray(S, float)
    if S.shape != (C * T, C * T):
        raise ValueError(f'S must be {(C * T, C * T)}, got {S.shape}')
    S4 = S.reshape(C, T, C, T, order='F')              # S4[c, t, c', t'] = S[c + C t, c' + C t']
    return np.ascontiguousarray(S4.transpose(1, 3, 0, 2)).reshape(T * T, C * C)


def _psd(a: np.ndarray) -> tuple[np.ndarray, float]:
    a = (a + a.T) / 2
    w, q = np.linalg.eigh(a)
    neg = float(-w[w < 0].sum()) if np.any(w < 0) else 0.0
    return (q * np.maximum(w, 0.0)) @ q.T, neg


def nearest_kronecker(S: np.ndarray, C: int, T: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (A_c (C x C), A_t (T x T), info) with np.kron(A_t, A_c) the PSD-projected NKP of S."""
    R = rearrange(S, C, T)
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    u, v, s0 = U[:, 0], Vt[0], float(s[0])
    At = np.sqrt(s0) * u.reshape(T, T)
    Ac = np.sqrt(s0) * v.reshape(C, C)
    if np.trace(Ac) < 0:                                 # the singular pair is defined up to a joint sign
        Ac, At = -Ac, -At
    Ac, neg_c = _psd(Ac)
    At, neg_t = _psd(At)
    nc, nt = np.linalg.norm(Ac), np.linalg.norm(At)
    if nc > 0 and nt > 0:
        r = np.sqrt(nt / nc)
        Ac, At = Ac * r, At / r
    info = {'sigma_1': s0, 'sigma_2_over_sigma_1': float(s[1] / s0) if len(s) > 1 and s0 > 0 else 0.0,
            'rank1_residual_rel': float(np.sqrt(max(np.sum(s[1:] ** 2), 0.0)) / max(np.sqrt(np.sum(s ** 2)), 1e-300)),
            'psd_clip_channel': neg_c, 'psd_clip_time': neg_t}
    return Ac, At, info


def nkp_matrix(S: np.ndarray, C: int, T: int) -> tuple[np.ndarray, dict]:
    """np.kron(A_t, A_c) for the PSD-projected nearest Kronecker product of S (d x d), plus info."""
    Ac, At, info = nearest_kronecker(S, C, T)
    return np.kron(At, Ac), info
