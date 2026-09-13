"""T21-S frozen predictor: isolated-spike overlap with loadings.

Pure functions of the exact population signal covariance K and the aspect
ratio gamma = d/N.  No coefficient is fitted on any data; the formula was
fixed in the archive (successor report of 12 September 2026) before this
package was written and is scored, not tuned, on confirmation seeds.

    c^2(theta, gamma) = (1 - gamma/theta^2) / (1 + gamma/theta)   for theta > sqrt(gamma), else 0

(Donoho, Gavish and Johnstone 2018, eq. 1.6; theta = excess population
eigenvalue above the unit noise floor, i.e. population eigenvalue ell = 1 + theta).

For a unit direction w and the population eigenpairs (lambda_i, v_i) of K,

    predicted recovery(w) = sum_i (w . v_i)^2 * c^2(lambda_i, gamma)

which for w = u_j is ``sum_i loading^2(j, i) c^2(theta_i, gamma)`` with
loading(j, i) = u_j . v_i.  The same expression applied to the normalised
unique component u_j^perp/||u_j^perp|| gives the predicted unique-component
recovery.  The predictor ignores the chance floor k/d (recorded separately).
"""
from __future__ import annotations

import numpy as np

FORMULA_ID = "dgj2018-eq1.6-isolated-spike-x-loadings/1"
FORMULA_TEXT = ("c2(theta,gamma) = (1 - gamma/theta^2)/(1 + gamma/theta) if theta > sqrt(gamma) else 0; "
                "pred(w) = sum_i (w.v_i)^2 * c2(lambda_i, gamma) over population eigenpairs (lambda_i, v_i) of the exact K")
SPIKE_FLOOR = 1e-12   # eigenvalues of K below this are the zero eigenspace (rank-deficient K)


def c2_isolated_spike(theta: float, gamma: float) -> float:
    """Squared overlap of the sample and population spike eigenvectors, DGJ eq. 1.6."""
    theta = float(theta)
    gamma = float(gamma)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if theta <= np.sqrt(gamma):
        return 0.0
    return float((1.0 - gamma / theta ** 2) / (1.0 + gamma / theta))


def population_eigenpairs(K: np.ndarray, n_top: int) -> tuple[np.ndarray, np.ndarray]:
    """Top ``n_top`` eigenpairs of the symmetric matrix K, descending eigenvalues.
    Eigenvalues below SPIKE_FLOOR are clipped to 0 (they contribute c2 = 0)."""
    w, V = np.linalg.eigh(K)
    order = np.argsort(w)[::-1][:n_top]
    lam = w[order]
    lam = np.where(lam < SPIKE_FLOOR, 0.0, lam)
    return lam, V[:, order]


def loadings(directions: np.ndarray, V: np.ndarray) -> np.ndarray:
    """loading(j, i) = u_j . v_i for unit-norm columns u_j (J x n_top)."""
    norms = np.sqrt(np.sum(directions * directions, axis=0))
    return (directions / norms[None, :]).T @ V


def predicted_recovery(w: np.ndarray, V: np.ndarray, lam: np.ndarray, gamma: float) -> float:
    """sum_i (w_hat . v_i)^2 c2(lambda_i, gamma) with w_hat = w/||w||."""
    w = np.asarray(w, dtype=float)
    nw = float(w @ w)
    if nw <= 0:
        raise ValueError("direction has zero norm")
    proj2 = (V.T @ w) ** 2 / nw
    c2 = np.array([c2_isolated_spike(l, gamma) for l in lam])
    return float(np.sum(proj2 * c2))


def predict_all(U: np.ndarray, U_perp: np.ndarray, unique_defined, K: np.ndarray, gamma: float) -> dict:
    """Ordinary and unique-component predictions for every planted direction."""
    n = U.shape[1]
    lam, V = population_eigenpairs(K, n)
    Lsq = loadings(U, V) ** 2
    c2 = [c2_isolated_spike(l, gamma) for l in lam]
    pred = [float(np.sum(Lsq[j] * np.array(c2))) for j in range(n)]
    pred_u, Lsq_u = [], []
    for j in range(n):
        if unique_defined[j]:
            pred_u.append(predicted_recovery(U_perp[:, j], V, lam, gamma))
            Lsq_u.append([float(x) for x in (V.T @ U_perp[:, j]) ** 2 / float(U_perp[:, j] @ U_perp[:, j])])
        else:
            pred_u.append(None)
            Lsq_u.append(None)
    return {"population_excess_eigenvalues": [float(x) for x in lam], "c2_per_spike": [float(x) for x in c2],
            "loadings_sq": [[float(x) for x in row] for row in Lsq], "unique_loadings_sq": Lsq_u,
            "predicted_recovery": pred, "predicted_unique_recovery": pred_u}


def sanity(gamma: float = 2.0) -> dict:
    """Data-free checks of the formula: zero at and below threshold, monotone
    above it, tends to one, never outside [0, 1]."""
    sg = np.sqrt(gamma)
    below = [c2_isolated_spike(r * sg, gamma) for r in (0.0, 0.5, 0.999, 1.0)]
    grid = np.linspace(1.001, 200.0, 4000) * sg
    vals = np.array([c2_isolated_spike(t, gamma) for t in grid])
    out = {"gamma": float(gamma), "zero_at_or_below_threshold": bool(all(v == 0.0 for v in below)),
           "monotone_above_threshold": bool(np.all(np.diff(vals) >= -1e-15)),
           "in_unit_interval": bool(np.all((vals >= 0) & (vals <= 1))),
           "value_at_theta_1e6": c2_isolated_spike(1e6 * sg, gamma),
           "tends_to_one": bool(abs(1.0 - c2_isolated_spike(1e6 * sg, gamma)) < 1e-5),
           "c2_at_ratio_2": c2_isolated_spike(2 * sg, gamma)}
    out["ok"] = bool(out["zero_at_or_below_threshold"] and out["monotone_above_threshold"] and out["in_unit_interval"] and out["tends_to_one"])
    return out
