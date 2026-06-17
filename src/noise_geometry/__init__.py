"""Noise-geometry reconstruction tools for Paper 1 experiments."""

from .filters import gls_amplitude, project_rank1, psd_amplitude_variance
from .metrics import (
    gaussian_nll,
    mse,
    residual_autocorrelation,
    residual_psd,
    weighted_inner,
    weighted_norm,
    weighted_residual,
    whitened_mse,
)
from .noise import (
    block_covariance,
    estimate_covariance,
    estimate_psd_rfft,
    estimate_psd_welch,
    inverse_covariance,
    inverse_psd_weights,
    regularize_covariance,
    regularize_psd,
    unwhiten_rfft,
    whiten_with_covariance,
    unwhiten_with_covariance,
    whiten_rfft,
    generate_colored_noise,
)
from .subspace import fit_pca, fit_weighted_pca, principal_angles

__all__ = [
    "block_covariance",
    "estimate_covariance",
    "estimate_psd_rfft",
    "estimate_psd_welch",
    "fit_pca",
    "fit_weighted_pca",
    "gaussian_nll",
    "generate_colored_noise",
    "gls_amplitude",
    "inverse_covariance",
    "inverse_psd_weights",
    "mse",
    "principal_angles",
    "project_rank1",
    "psd_amplitude_variance",
    "regularize_covariance",
    "regularize_psd",
    "residual_autocorrelation",
    "residual_psd",
    "unwhiten_rfft",
    "unwhiten_with_covariance",
    "weighted_inner",
    "weighted_norm",
    "weighted_residual",
    "whiten_rfft",
    "whiten_with_covariance",
    "whitened_mse",
]
