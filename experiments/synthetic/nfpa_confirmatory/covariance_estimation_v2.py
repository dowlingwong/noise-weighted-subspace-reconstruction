"""Calibrated separable covariance estimation (Round 10, E10b).

The defect being corrected
--------------------------
Round 9's E10 estimated each Kronecker factor from marginal samples of the
same calibration noise. Each marginal estimate then carries the FULL average
measurement variance:

    E[Sigma_c_raw] = Sigma_c * tr(Sigma_t)/T,
    E[Sigma_t_raw] = Sigma_t * tr(Sigma_c)/C,

so their Kronecker product carries that scale twice — the E10 product's
average diagonal is ~(mean variance)^2 instead of the mean variance. NFPA's
fit and reconstruction are invariant to a global whitening scale, so E10's
*contrasts* are unaffected; but the product-scale diagnostics (scale ratio,
Kronecker Frobenius error, oracle-whitening residual) were dominated by this
uncalibrated global factor and said nothing about estimation accuracy.

This is a **product-scale calibration** issue. It is *not* the reciprocal
factor non-identifiability (Sigma_t (x) Sigma_c is unchanged under
Sigma_c -> a*Sigma_c, Sigma_t -> Sigma_t/a): that ambiguity is real but
scale-free diagnostics already handle it. The two must not be conflated.

The frozen correction (declared before any full rerun)
------------------------------------------------------
With ``v_c = trace(Sigma_c_reg)/C`` and ``v_t = trace(Sigma_t_reg)/T`` — both
computed AFTER E10's unchanged positive-definiteness regularization and
BEFORE this normalization — both estimate the same average measurement
variance. Define

    v_hat       = sqrt(v_c * v_t)
    Sigma_c_hat = Sigma_c_reg / sqrt(v_hat)
    Sigma_t_hat = Sigma_t_reg / sqrt(v_hat).

The corrected Kronecker product equals the regularized product divided by
``v_hat``; its average diagonal equals ``v_hat`` by construction. The residual
per-factor scale is reciprocal and does not alter the product. The frozen
order is: sample estimation -> E10's regularization -> this normalization.
That order is not changed from full-run outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from noise_geometry.noise import estimate_covariance

from .run_estimated_covariance import (
    channel_samples,
    estimate_axis_covariances,
    temporal_samples,
)


@dataclass(frozen=True)
class CalibratedCovarianceEstimate:
    """Named estimator result required by the Round 10 runbook (Phase 3)."""

    Sigma_c_sample: np.ndarray
    Sigma_t_sample: np.ndarray
    Sigma_c_regularized: np.ndarray
    Sigma_t_regularized: np.ndarray
    Sigma_c_hat: np.ndarray
    Sigma_t_hat: np.ndarray
    v_c: float
    v_t: float
    v_hat: float
    regularization: str = "regularize_covariance(shrinkage=0.0, floor=None)  # unchanged from E10"
    diagnostics: dict = field(default_factory=dict)


def estimate_calibrated_axis_covariances(noise: np.ndarray) -> CalibratedCovarianceEstimate:
    """Sample estimation -> E10's regularization -> symmetric scale calibration."""
    # Raw sample estimates, recorded for the audit trail (identical estimator
    # calls to E10's, before regularization).
    Sc_sample = estimate_covariance(channel_samples(noise), center=True)
    St_sample = estimate_covariance(temporal_samples(noise), center=True)

    # E10's unchanged regularization (this call performs sample estimation
    # again internally; the estimator is deterministic in `noise`).
    Sc_reg, St_reg = estimate_axis_covariances(noise)

    C = Sc_reg.shape[0]
    T = St_reg.shape[0]
    v_c = float(np.trace(Sc_reg)) / C
    v_t = float(np.trace(St_reg)) / T
    if v_c <= 0 or v_t <= 0:
        raise ValueError(f"non-positive average variances v_c={v_c}, v_t={v_t}")
    v_hat = float(np.sqrt(v_c * v_t))

    scale = float(np.sqrt(v_hat))
    Sc_hat = Sc_reg / scale
    St_hat = St_reg / scale

    return CalibratedCovarianceEstimate(
        Sigma_c_sample=Sc_sample,
        Sigma_t_sample=St_sample,
        Sigma_c_regularized=Sc_reg,
        Sigma_t_regularized=St_reg,
        Sigma_c_hat=Sc_hat,
        Sigma_t_hat=St_hat,
        v_c=v_c,
        v_t=v_t,
        v_hat=v_hat,
        diagnostics={
            "mean_diagonal_of_product": v_hat,  # by construction; verified in tests
            "calibration_identity": "kron(St_hat, Sc_hat) == kron(St_reg, Sc_reg) / v_hat",
        },
    )
