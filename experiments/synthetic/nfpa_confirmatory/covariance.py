"""Covariance construction and numerical checks for the NFPA confirmatory study.

Per protocol section 5: ``Sigma = Sigma_t (x) Sigma_c`` in the column-major vec
convention (``vec(Z)`` with ``order="F"``), ``Sigma_t`` implements the declared
temporal PSD, and ``Sigma_c`` is an AR(1)-style channel correlation with
``rho = 0.4``. Every covariance must pass the five numerical checks before any
data generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from noise_geometry.noise import covariance_from_psd, make_powerlaw_psd

from .config import PrimaryConfig


def ar1_covariance(n_channels: int, rho: float) -> np.ndarray:
    """AR(1) correlation matrix ``Sigma_c[i,j] = rho^|i-j|``."""
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    idx = np.arange(int(n_channels))
    return np.power(rho, np.abs(idx[:, None] - idx[None, :]))


def temporal_covariance(cfg: PrimaryConfig) -> np.ndarray:
    """Stationary temporal covariance implementing the declared PSD (1/f^2)."""
    _, psd = make_powerlaw_psd(cfg.T, cfg.sampling_frequency, kind=cfg.noise_kind, level=1.0)
    return covariance_from_psd(psd, cfg.T, sampling_frequency=cfg.sampling_frequency)


def full_covariance(cfg: PrimaryConfig, regime: str) -> dict[str, np.ndarray]:
    """Return ``Sigma_c``, ``Sigma_t``, and the full ``Sigma`` for a regime.

    Regimes:

    - ``"colored"``: AR(1) channel correlation and 1/f^2 temporal PSD.
    - ``"white"``:   identity channel and temporal covariances (``Sigma = I``).
    """
    if regime == "colored":
        Sigma_c = ar1_covariance(cfg.C, cfg.channel_ar_rho)
        Sigma_t = temporal_covariance(cfg)
    elif regime == "white":
        Sigma_c = np.eye(cfg.C)
        Sigma_t = np.eye(cfg.T)
    else:
        raise ValueError(f"unknown noise regime {regime!r}")
    Sigma = np.kron(Sigma_t, Sigma_c)
    return {"Sigma_c": Sigma_c, "Sigma_t": Sigma_t, "Sigma": Sigma}


def _sqrt_inv(A: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
    vals = np.clip(vals, np.finfo(float).eps, None)
    return (vecs / np.sqrt(vals)[None, :]) @ vecs.T


def whiten_separable(X: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray) -> np.ndarray:
    """Apply ``Sigma^{-1/2}`` as ``Z = Sigma_c^{-1/2} X Sigma_t^{-1/2}``.

    ``X`` has shape ``(..., C, T)``. Axis-preserving separable whitening per
    protocol section 5.
    """
    Wc = _sqrt_inv(Sigma_c)
    Wt = _sqrt_inv(Sigma_t)
    return np.einsum("ab,...bd,de->...ae", Wc, X, Wt)


def unwhiten_separable(Z: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray) -> np.ndarray:
    """Invert :func:`whiten_separable`."""
    Sc = np.linalg.cholesky(Sigma_c)
    St = np.linalg.cholesky(Sigma_t)
    return np.einsum("ab,...bd,de->...ae", Sc, Z, St.T)


def run_numerical_checks(cfg: PrimaryConfig, Sigma: np.ndarray) -> dict[str, float]:
    """Run the five covariance checks from protocol section 5 and return a report."""
    d = Sigma.shape[0]
    symmetry_error = float(np.linalg.norm(Sigma - Sigma.T, ord="fro") / max(d, 1))

    vals = np.linalg.eigvalsh(Sigma)
    min_eig = float(vals.min())
    positive = bool(min_eig > 0.0)

    W = _sqrt_inv(Sigma)
    whitening_error = float(np.linalg.norm(W @ Sigma @ W.T - np.eye(d), ord="fro") / d)

    rng = np.random.default_rng(0)
    sqrt_Sigma = np.linalg.cholesky(Sigma)
    noise = rng.standard_normal((4096, d)) @ sqrt_Sigma.T
    empirical = (noise.T @ noise) / noise.shape[0]
    mc_error = float(np.linalg.norm(empirical - Sigma, ord="fro") / np.linalg.norm(Sigma, ord="fro"))

    return {
        "symmetry_error": symmetry_error,
        "min_eigenvalue": min_eig,
        "positive_definite": positive,
        "whitening_identity_error": whitening_error,
        "monte_carlo_covariance_relative_error": mc_error,
    }


def assert_checks_pass(report: dict[str, float], *, tol_sym: float = 1e-10, tol_whiten: float = 1e-8, tol_mc: float = 5e-2) -> None:
    if report["symmetry_error"] > tol_sym:
        raise AssertionError(f"symmetry error {report['symmetry_error']:.2e} exceeds {tol_sym:.0e}")
    if not report["positive_definite"]:
        raise AssertionError("covariance is not positive definite")
    if report["whitening_identity_error"] > tol_whiten:
        raise AssertionError(f"whitening identity error {report['whitening_identity_error']:.2e} exceeds {tol_whiten:.0e}")
    if report["monte_carlo_covariance_relative_error"] > tol_mc:
        raise AssertionError(f"MC covariance error {report['monte_carlo_covariance_relative_error']:.2e} exceeds {tol_mc:.0e}")
