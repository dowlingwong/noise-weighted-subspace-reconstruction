"""Quadrature GLS estimators and their predicted variance.

All estimators here minimize the same Gaussian objective as the rest of the
repository,

    chi^2(s_hat) = (x - s_hat)^dagger Sigma^{-1} (x - s_hat),

specialized to the TIDMAD signal model. Two constraints are of interest:

* **rank 1, fixed phase** -- the classical single-template optimal filter. It is
  the estimator a matched-filter pipeline uses when it assumes it knows the
  phase, and it is provably suboptimal by ``cos^2(phase error)`` when it does
  not.
* **rank 2, unknown phase** -- the analytic quadrature subspace. This is the
  correct constrained maximum-likelihood estimator for a sinusoid of unknown
  phase, and it is what Paper 1's rank argument predicts a *learned* rank-2
  subspace should recover.

Weight convention: everything routes through the repository's one-sided OF
weights (``canonical.make_weights.build_of_one_sided_weights`` via
``noise_geometry.noise.inverse_psd_weights``), so DC/Nyquist handling lives in
exactly one place and matches S1/S2/S5 and CRESST.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "QuadratureFit",
    "fit_quadrature",
    "fit_fixed_phase",
    "quadrature_amplitude_variance",
    "fixed_phase_snr_loss",
]


@dataclass(frozen=True)
class QuadratureFit:
    """Result of a rank-2 quadrature GLS fit.

    Attributes
    ----------
    cos_coefficient, sin_coefficient:
        The two quadrature coefficients ``a`` and ``b`` such that the fitted
        signal is ``a cos(wt) + b sin(wt)``.
    amplitude:
        ``sqrt(a^2 + b^2)``. Note this is a *biased* estimator of the true
        amplitude at low SNR (it is Rice-distributed, not Gaussian); the bias is
        negligible when ``amplitude / amplitude_sigma`` is large. Report the
        coefficients, not just the amplitude, whenever SNR is low.
    phase:
        ``atan2(-b, a)``, the phase of ``A cos(wt + phi)``.
    weighted_residual:
        The value of the Gaussian objective at the fit, per window.
    """

    cos_coefficient: np.ndarray
    sin_coefficient: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    weighted_residual: np.ndarray


def _as_2d(X_f: np.ndarray) -> tuple[np.ndarray, bool]:
    X = np.asarray(X_f)
    if X.ndim == 1:
        return X[None, :], True
    if X.ndim != 2:
        raise ValueError("X_f must be 1D or 2D")
    return X, False


def _weighted_gram(basis_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Real part of the weighted Gram matrix of a template basis.

    For real time-domain templates the weighted inner product is Hermitian with
    a real value on this basis, so the real part is the whole story; taking it
    explicitly avoids carrying a numerically negligible imaginary residue into
    the linear solve.
    """
    gram = (basis_f * weights[None, :]) @ basis_f.conj().T
    return np.real(gram)


def fit_quadrature(
    X_f: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
) -> QuadratureFit:
    """Fit the rank-2 quadrature subspace by weighted least squares.

    Parameters
    ----------
    X_f:
        ``(n_windows, n_rfft)`` (or ``(n_rfft,)``) rFFT of the observed windows.
    basis_f:
        ``(2, n_rfft)`` rFFT of the cosine and sine templates.
    weights:
        One-sided OF weights of length ``n_rfft``.
    """
    X, single = _as_2d(X_f)
    B = np.asarray(basis_f)
    w = np.asarray(weights, dtype=np.float64)
    if B.ndim != 2 or B.shape[0] != 2:
        raise ValueError("basis_f must have shape (2, n_rfft)")
    if B.shape[1] != w.shape[0] or X.shape[1] != w.shape[0]:
        raise ValueError("bin counts of X_f, basis_f, and weights must agree")

    gram = _weighted_gram(B, w)
    rhs = np.real((X * w[None, :]) @ B.conj().T)  # (n_windows, 2)
    coeff = np.linalg.solve(gram, rhs.T).T

    residual = X - coeff @ B
    weighted_residual = np.real(np.sum((np.abs(residual) ** 2) * w[None, :], axis=1))

    a = coeff[:, 0]
    b = coeff[:, 1]
    amplitude = np.hypot(a, b)
    phase = np.arctan2(-b, a)

    if single:
        return QuadratureFit(
            cos_coefficient=a[0],
            sin_coefficient=b[0],
            amplitude=float(amplitude[0]),
            phase=float(phase[0]),
            weighted_residual=float(weighted_residual[0]),
        )
    return QuadratureFit(
        cos_coefficient=a,
        sin_coefficient=b,
        amplitude=amplitude,
        phase=phase,
        weighted_residual=weighted_residual,
    )


def fit_fixed_phase(
    X_f: np.ndarray,
    basis_f: np.ndarray,
    weights: np.ndarray,
    *,
    phase: float = 0.0,
) -> np.ndarray:
    """Rank-1 optimal filter at an assumed phase.

    This is the estimator that is *wrong by construction* when the true phase is
    unknown; Phase 3 of the roadmap uses the gap between this and
    :func:`fit_quadrature` as the real-data measurement of the rank argument.
    """
    X, single = _as_2d(X_f)
    B = np.asarray(basis_f)
    w = np.asarray(weights, dtype=np.float64)
    template = np.cos(phase) * B[0] - np.sin(phase) * B[1]
    denom = float(np.real(np.sum(np.conj(template) * template * w)))
    if denom <= 0.0:
        raise ValueError("template has zero weighted norm")
    amps = np.real(np.sum(np.conj(template)[None, :] * X * w[None, :], axis=1)) / denom
    return float(amps[0]) if single else amps


def quadrature_amplitude_variance(
    basis_f: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    *,
    trace_len: int,
    sampling_frequency: float,
) -> float:
    """Predicted per-quadrature coefficient variance under the noise model.

    Delegates to :func:`noise_geometry.filters.psd_amplitude_variance`, which
    encodes this repository's rFFT and one-sided-PSD normalization, and averages
    the two quadrature directions. For an on-grid frequency the two directions
    carry identical information, so the average is exact rather than an
    approximation; off grid it is a good approximation and the two values are
    reported separately by callers that need them.

    Interpretation. This is the variance of ``a`` and of ``b`` individually. At
    high SNR the amplitude ``sqrt(a^2+b^2)`` inherits the same variance, because
    the estimator error projects onto the (known) signal direction. At low SNR
    the amplitude is Rice-distributed and this value understates its spread --
    compare coefficients, not amplitudes, in that regime.
    """
    try:  # repo-root on path (pytest)
        from ..filters import psd_amplitude_variance
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.filters import psd_amplitude_variance  # type: ignore

    B = np.asarray(basis_f)
    variances = [
        float(
            psd_amplitude_variance(
                B[i],
                psd,
                weights,
                trace_len=int(trace_len),
                sampling_frequency=float(sampling_frequency),
            )
        )
        for i in range(B.shape[0])
    ]
    return float(np.mean(variances))


def fixed_phase_snr_loss(phase_error: float) -> float:
    """Fractional SNR retained by a fixed-phase rank-1 filter: ``cos^2``.

    Paper 1 Eq. (16) gives the effective SNR loss of a misaligned template as
    the squared cosine of the whitened-space angle. For the quadrature manifold
    that angle *is* the phase error, which makes this an exact, closed-form
    prediction to test on real data rather than a heuristic.
    """
    return float(np.cos(float(phase_error)) ** 2)
