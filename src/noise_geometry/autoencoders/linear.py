"""Closed-form tied linear autoencoder baselines."""

from __future__ import annotations

import numpy as np

from ..subspace import fit_pca, fit_weighted_pca


def tied_linear_ae_closed_form(samples: np.ndarray, rank: int, *, weights: np.ndarray | None = None, center: bool = True):
    """Return the tied linear AE subspace for MSE or weighted reconstruction loss."""
    if weights is None:
        return fit_pca(samples, rank, center=center)
    return fit_weighted_pca(samples, weights, rank, center=center)
