"""Noise-geometry reconstruction tools for Paper 1 experiments."""

from .filters import gls_amplitude, project_rank1
from .metrics import mse, weighted_residual, whitened_mse
from .noise import (
    estimate_covariance,
    estimate_psd_rfft,
    inverse_psd_weights,
    regularize_covariance,
    regularize_psd,
    whiten_with_covariance,
    unwhiten_with_covariance,
    whiten_rfft,
    generate_colored_noise,
)
from .subspace import fit_pca, fit_weighted_pca, principal_angles

__all__ = [
    "estimate_covariance",
    "estimate_psd_rfft",
    "fit_pca",
    "fit_weighted_pca",
    "generate_colored_noise",
    "gls_amplitude",
    "inverse_psd_weights",
    "mse",
    "principal_angles",
    "project_rank1",
    "regularize_covariance",
    "regularize_psd",
    "unwhiten_with_covariance",
    "weighted_residual",
    "whiten_rfft",
    "whiten_with_covariance",
    "whitened_mse",
]
