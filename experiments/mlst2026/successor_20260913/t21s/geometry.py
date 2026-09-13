"""T21-S frozen geometry: planted unit directions in whitened R^d with a
specified Gram matrix, and the exact population signal covariance.

Everything here is a pure function of (d, geometry arm, theta) and a frozen
construction seed.  Nothing draws data.

Construction (frozen; see PROTOCOL_T21S_v2.json ``design.geometry``):

* ``Q`` = orthonormal frame, d x J, from a QR decomposition of a Gaussian
  matrix drawn with ``FRAME_SEED`` (signs fixed so that diag(R) > 0);
* ``U = Q @ L.T`` where ``G = L @ L.T`` is the Cholesky factorisation of the
  arm's Gram matrix, so that ``U.T @ U == G`` exactly (to rounding) and every
  column ``u_j`` has unit norm because ``G_jj = 1``;
* signal covariance ``K = U diag(theta) U.T``; its non-zero spectrum is the
  spectrum of the J x J matrix ``M = diag(sqrt theta) G diag(sqrt theta)``;
* unique component ``u_j^perp`` = ``u_j`` projected off ``span{u_i : i != j}``,
  with ``||u_j^perp||^2 / ||u_j||^2 = 1 / (G^{-1})_jj`` (Schur complement identity).

This geometry is independent of the archived T21 pulse family
(``experiments/paper1/t21_excitation.py``): no channel/temporal bases, no
Jacobian columns, no AR(1)/Brownian covariance; the noise is N(0, I) so the
whitening map is the identity and "whitened coordinates" are the raw ones.
"""
from __future__ import annotations

import numpy as np

J = 4
FRAME_SEED = 20260913
UNIQUE_UNDEFINED_BELOW = 1e-3          # ||u_j^perp||^2/||u_j||^2 below this -> unique recovery undefined
GRAM_TOLERANCE = 1e-10                 # realised Gram must match the declared one to this (max abs)

# Correlated Gram: off-diagonals of both signs plus an exactly-zero pair.
# Eigenvalues 0.258, 0.758, 1.242, 1.742 (positive definite);
# unique-norm ratios 1/(G^-1)_jj = 0.464, 0.563, 0.660, 0.880.
GRAM = {
    "orthogonal": np.eye(J),
    "correlated": np.array([[1.0, 0.6, -0.4, 0.0],
                            [0.6, 1.0, 0.0, 0.0],
                            [-0.4, 0.0, 1.0, 0.3],
                            [0.0, 0.0, 0.3, 1.0]]),
}
GEOMETRIES = tuple(GRAM.keys())


def gram(geometry: str) -> np.ndarray:
    if geometry not in GRAM:
        raise ValueError(f"unknown geometry {geometry!r}; known {GEOMETRIES}")
    return GRAM[geometry].copy()


def orthonormal_frame(d: int, n: int = J, seed: int = FRAME_SEED) -> np.ndarray:
    """d x n orthonormal frame from a seeded Gaussian matrix (QR, signs fixed)."""
    if d < n:
        raise ValueError(f"need d >= {n}, got d={d}")
    A = np.random.default_rng(seed).standard_normal((d, n))
    Q, R = np.linalg.qr(A, mode="reduced")
    Q = Q * np.sign(np.diag(R))[None, :]
    return Q


def directions(d: int, geometry: str, seed: int = FRAME_SEED) -> np.ndarray:
    """d x J matrix U whose columns are the planted unit directions with
    U.T @ U == GRAM[geometry] (realised via a Cholesky factor)."""
    G = gram(geometry)
    L = np.linalg.cholesky(G)               # G = L L^T
    Q = orthonormal_frame(d, G.shape[0], seed)
    return Q @ L.T                          # U^T U = L Q^T Q L^T = L L^T = G


def unique_components(U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each column j: u_j projected off span{u_i : i != j}.
    Returns (U_perp d x J, norm_ratio J) with norm_ratio_j = ||u_j^perp||^2/||u_j||^2."""
    d, n = U.shape
    Up = np.zeros_like(U)
    ratio = np.zeros(n)
    for j in range(n):
        others = np.delete(U, j, axis=1)
        coef, *_ = np.linalg.lstsq(others, U[:, j], rcond=None)
        Up[:, j] = U[:, j] - others @ coef
        ratio[j] = float(Up[:, j] @ Up[:, j]) / float(U[:, j] @ U[:, j])
    return Up, ratio


def signal_covariance(U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """K = sum_j theta_j u_j u_j^T (d x d, exact)."""
    theta = np.asarray(theta, dtype=float)
    return (U * theta[None, :]) @ U.T


def population_spectrum_exact(G: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Non-zero eigenvalues of K via the J x J matrix diag(sqrt theta) G diag(sqrt theta), descending."""
    s = np.sqrt(np.asarray(theta, dtype=float))
    M = (s[:, None] * G) * s[None, :]
    return np.sort(np.linalg.eigvalsh(M))[::-1]


def theta_vector(*, gamma: float, kind: str, varied_direction: int | None, ratio: float | None,
                 strong_ratio: float, n_directions: int = J) -> np.ndarray:
    """Excess population eigenvalues theta_j in units set by the BBP scale
    sqrt(gamma): grid cells hold every direction at ``strong_ratio * sqrt(gamma)``
    except the varied one at ``ratio * sqrt(gamma)``; ``all_zero`` -> K = 0;
    ``all_strong`` -> every direction at ``strong_ratio * sqrt(gamma)``."""
    sg = float(np.sqrt(gamma))
    if kind == "all_zero":
        return np.zeros(n_directions)
    if kind == "all_strong":
        return np.full(n_directions, strong_ratio * sg)
    if kind == "grid":
        if varied_direction is None or ratio is None:
            raise ValueError("grid cells need a varied direction and a ratio")
        th = np.full(n_directions, strong_ratio * sg)
        th[int(varied_direction)] = float(ratio) * sg
        return th
    raise ValueError(f"unknown cell kind {kind!r}")


def check_invariants(d: int, geometry: str, theta: np.ndarray | None = None, seed: int = FRAME_SEED) -> dict:
    """Invariants the frozen construction must satisfy; returns numbers, never
    raises.  ``ok`` is True when every declared tolerance holds."""
    G = gram(geometry)
    U = directions(d, geometry, seed)
    Q = orthonormal_frame(d, G.shape[0], seed)
    Up, ratio = unique_components(U)
    Ginv = np.linalg.inv(G)
    out = {
        "d": int(d), "geometry": geometry, "n_directions": int(G.shape[0]),
        "gram_max_abs_error": float(np.max(np.abs(U.T @ U - G))),
        "frame_orthonormality_max_abs_error": float(np.max(np.abs(Q.T @ Q - np.eye(G.shape[0])))),
        "unit_norm_max_abs_error": float(np.max(np.abs(np.sum(U * U, axis=0) - 1.0))),
        "unique_norm_ratio": [float(x) for x in ratio],
        "unique_norm_ratio_schur_identity_max_abs_error": float(np.max(np.abs(ratio - 1.0 / np.diag(Ginv)))),
        "unique_orthogonal_to_others_max_abs": float(max(np.max(np.abs(np.delete(U, j, axis=1).T @ Up[:, j])) for j in range(G.shape[0]))),
        "gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(G))),
        "unique_defined": [bool(r >= UNIQUE_UNDEFINED_BELOW) for r in ratio],
    }
    if theta is not None:
        theta = np.asarray(theta, dtype=float)
        K = signal_covariance(U, theta)
        w_exact = population_spectrum_exact(G, theta)
        w_full = np.sort(np.linalg.eigvalsh(K))[::-1][: G.shape[0]]
        out["population_excess_eigenvalues_exact"] = [float(x) for x in w_exact]
        out["population_spectrum_max_abs_error"] = float(np.max(np.abs(w_full - w_exact)))
        out["population_trace_error"] = float(abs(np.trace(K) - float(np.sum(theta))))
    tol = GRAM_TOLERANCE
    out["ok"] = bool(out["gram_max_abs_error"] <= tol and out["frame_orthonormality_max_abs_error"] <= tol
                     and out["unit_norm_max_abs_error"] <= tol and out["unique_norm_ratio_schur_identity_max_abs_error"] <= 1e-9
                     and out["unique_orthogonal_to_others_max_abs"] <= 1e-9 and out["gram_min_eigenvalue"] > 0
                     and out.get("population_spectrum_max_abs_error", 0.0) <= 1e-8 * max(1.0, float(np.max(theta)) if theta is not None else 1.0))
    return out
