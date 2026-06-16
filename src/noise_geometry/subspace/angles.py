"""Subspace angle metrics."""

from __future__ import annotations

import numpy as np


def _orthonormal_rows(basis: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    B = np.asarray(basis, dtype=np.float64)
    if weights is not None:
        sqrt_w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 0.0, None))
        B = B * sqrt_w[None, :]
    q, _ = np.linalg.qr(B.T)
    return q


def principal_angles(a: np.ndarray, b: np.ndarray, *, weights: np.ndarray | None = None, degrees: bool = True) -> np.ndarray:
    """Return principal angles between row-span subspaces."""
    Qa = _orthonormal_rows(a, weights)
    Qb = _orthonormal_rows(b, weights)
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    angles = np.arccos(sv)
    return np.degrees(angles) if degrees else angles
