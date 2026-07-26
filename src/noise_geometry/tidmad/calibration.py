"""Phase 4: the calibration boundary, in-domain.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 4.

Paper 1 needs one worked demonstration that *implementation correctness is not
model adequacy*. GWOSC currently supplies it, but in a different domain, with no
ground truth for what the failure costs. TIDMAD can supply the same lesson
in-domain, with injections quantifying the cost -- which is strictly better.

The structure mirrors the GWOSC stage deliberately, because that design was
sound even though its result was negative:

* a **stationary synthetic control** that must pass, otherwise any failure on
  real data is a bug rather than a finding (this is a halt gate, not a result);
* a **global** noise model evaluated on held-out stretches;
* a **time-local** model at several block lengths, predeclared before running,
  as the obvious first repair attempt;
* the cost, expressed in the units a physicist cares about -- how much the
  amplitude estimate's bias and spread degrade relative to the ideal.

Nothing here tunes the block length until the data passes. The acceptance band
is fixed in the caller and the local radii are declared in the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .estimators import fit_quadrature
from .psd import welch_psd_from_windows
from .templates import quadrature_templates_rfft

__all__ = [
    "CalibrationResult",
    "whitened_chi2_per_dof",
    "score_dispersion",
    "assess_global_model",
    "assess_local_model",
    "dispersion_standard_error",
]


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration of one noise model on one set of held-out windows."""

    label: str
    chi2_per_dof_mean: float
    chi2_per_dof_std: float
    score_dispersion: float
    score_dispersion_standard_error: float
    n_windows: int
    degrees_of_freedom: int
    metadata: dict

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "chi2_per_dof_mean": self.chi2_per_dof_mean,
            "chi2_per_dof_std": self.chi2_per_dof_std,
            "score_dispersion": self.score_dispersion,
            "score_dispersion_standard_error": self.score_dispersion_standard_error,
            "score_dispersion_z": (
                float((self.score_dispersion - 1.0) / self.score_dispersion_standard_error)
                if self.score_dispersion_standard_error > 0
                else float("nan")
            ),
            "n_windows": self.n_windows,
            "degrees_of_freedom": self.degrees_of_freedom,
            **self.metadata,
        }


def _effective_dof(n_samples: int, weights: np.ndarray) -> int:
    """Number of independent real quadratic degrees of freedom in the sum.

    Only bins with non-zero weight contribute. Interior rFFT bins carry two real
    degrees of freedom (real and imaginary parts); DC and, for even length, the
    Nyquist bin carry one. Getting this wrong rescales chi^2/dof by a constant
    and would make a perfectly calibrated model look broken.
    """
    w = np.asarray(weights, dtype=np.float64)
    n = int(n_samples)
    active = w > 0.0
    dof = 0
    if active[0]:
        dof += 1
    interior_stop = n // 2 + 1 - (n + 1) % 2
    dof += 2 * int(np.count_nonzero(active[1:interior_stop]))
    if n % 2 == 0 and active.size > 1 and active[-1]:
        dof += 1
    return int(dof)


def whitened_chi2_per_dof(
    windows: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    *,
    n_samples: int | None = None,
) -> np.ndarray:
    """Per-window whitened energy divided by its expectation under the model.

    Normalization, derived once here so it is auditable. With the repository's
    one-sided OF weights and rFFT convention:

    * interior bin: ``w_k = 2 / J_k`` and ``E|X_k|^2 = J_k f_s N / 2``, so the
      bin contributes ``f_s N`` in expectation and carries two real degrees of
      freedom (real and imaginary parts);
    * Nyquist bin (even ``N``): ``w = 1 / (2 J)`` and ``E|X|^2 = J f_s N``, so
      it contributes ``f_s N / 2`` and carries one degree of freedom;
    * DC is zero-weighted by convention and contributes nothing.

    Every case therefore contributes ``f_s N / 2`` per degree of freedom, giving
    ``E[energy] = dof * f_s * N / 2``. Dividing by that leaves a statistic with
    expectation one under a correct model. This is Paper 1's S8 statistic
    applied to noise-only stretches rather than to fit residuals.
    """
    X = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    n = int(n_samples) if n_samples is not None else X.shape[1]
    w = np.asarray(weights, dtype=np.float64)
    X_f = np.fft.rfft(X, axis=1)
    energy = np.real(np.sum((np.abs(X_f) ** 2) * w[None, :], axis=1))
    dof = _effective_dof(n, w)
    if dof <= 0:
        raise ValueError("no weighted degrees of freedom; check the weight vector")
    expectation = float(dof) * float(sampling_frequency) * float(n) / 2.0
    return energy / expectation


def score_dispersion(
    windows: np.ndarray,
    psd: np.ndarray,
    weights: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
) -> tuple[float, np.ndarray]:
    """Standard deviation of the normalized matched-filter score on noise.

    The score is each quadrature coefficient divided by its model-predicted
    sigma, so a correct noise model gives unit dispersion by construction. This
    is the TIDMAD analogue of the GWOSC null-spread ratio, and it is the
    statistic that matters operationally: it sets the false-alarm rate.

    The predicted sigma comes from the shared
    :func:`quadrature_amplitude_variance`, which encodes this repository's rFFT
    and one-sided-PSD normalization, rather than from a local re-derivation.
    """
    from .estimators import quadrature_amplitude_variance  # local: avoids a cycle

    X = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    n_samples = X.shape[1]
    basis_f = quadrature_templates_rfft(n_samples, sampling_frequency, frequency_hz)
    fit = fit_quadrature(np.fft.rfft(X, axis=1), basis_f, weights)

    predicted_sigma = float(
        np.sqrt(
            quadrature_amplitude_variance(
                basis_f,
                psd,
                weights,
                trace_len=n_samples,
                sampling_frequency=sampling_frequency,
            )
        )
    )
    coefficients = np.concatenate(
        [np.asarray(fit.cos_coefficient), np.asarray(fit.sin_coefficient)]
    )
    scores = coefficients / predicted_sigma if predicted_sigma > 0 else coefficients
    return (float(np.std(scores, ddof=1)) if scores.size > 1 else float("nan")), scores


def dispersion_standard_error(n_scores: int) -> float:
    """Standard error of a sample standard deviation of ``n`` normal scores.

    Reported next to every dispersion so a reader can tell a real calibration
    failure from a small-sample fluctuation. With only a hundred scores the
    dispersion wanders by several percent on its own, which is enough to move it
    across a naive acceptance band.

    Caveat, and it matters: this is the standard error *assuming the predicted
    sigma is exact*. It is not, because the PSD is estimated from finite
    calibration data, and that adds variance the formula does not see. In
    synthetic five-seed checks at ~1000 scores the realized seed-to-seed spread
    was about 0.03 against a formula value of 0.022. Treat the z-score as a
    lower bound on the fluctuation and prefer a seed ensemble when a dispersion
    near the acceptance boundary has to be adjudicated.
    """
    n = int(n_scores)
    return float(1.0 / np.sqrt(2.0 * max(n - 1, 1)))


def assess_global_model(
    calibration_windows: np.ndarray,
    evaluation_windows: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    label: str = "global",
    window: str = "boxcar",
    average: str = "median",
) -> CalibrationResult:
    """Fit one PSD on calibration windows; score held-out windows with it."""
    try:
        from ..noise import inverse_psd_weights
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.noise import inverse_psd_weights  # type: ignore

    calibration = np.atleast_2d(np.asarray(calibration_windows, dtype=np.float64))
    evaluation = np.atleast_2d(np.asarray(evaluation_windows, dtype=np.float64))
    n_samples = calibration.shape[1]

    estimate = welch_psd_from_windows(
        list(calibration), sampling_frequency, window=window, average=average, source=label
    )
    weights = inverse_psd_weights(estimate.psd, n_samples)

    chi2 = whitened_chi2_per_dof(evaluation, weights, sampling_frequency, n_samples=n_samples)
    dispersion, scores = score_dispersion(
        evaluation, estimate.psd, weights, sampling_frequency, frequency_hz
    )

    return CalibrationResult(
        label=label,
        chi2_per_dof_mean=float(np.mean(chi2)),
        chi2_per_dof_std=float(np.std(chi2, ddof=1)) if chi2.size > 1 else float("nan"),
        score_dispersion=dispersion,
        score_dispersion_standard_error=dispersion_standard_error(int(np.size(scores))),
        n_windows=int(evaluation.shape[0]),
        degrees_of_freedom=_effective_dof(n_samples, weights),
        metadata={
            "n_calibration_windows": int(calibration.shape[0]),
            "psd_average": average,
            "psd_window": window,
        },
    )


def assess_local_model(
    windows: np.ndarray,
    sampling_frequency: float,
    frequency_hz: float,
    *,
    radius_windows: int,
    label: str | None = None,
    window: str = "boxcar",
    average: str = "median",
) -> CalibrationResult:
    """Score each window with a PSD fitted from its temporal neighbours only.

    The neighbourhood **excludes the scored window**. Including it would let the
    noise model absorb the very fluctuation it is meant to predict, which
    produces an apparently excellent calibration that is pure leakage -- the
    single easiest way to fake a passing result here.
    """
    try:
        from ..noise import inverse_psd_weights
    except ImportError:  # pragma: no cover - src on path
        from noise_geometry.noise import inverse_psd_weights  # type: ignore

    X = np.atleast_2d(np.asarray(windows, dtype=np.float64))
    n_windows, n_samples = X.shape
    radius = int(radius_windows)
    if radius < 1:
        raise ValueError("radius_windows must be at least 1")
    if n_windows < 2 * radius + 1:
        raise ValueError(
            f"need at least {2 * radius + 1} windows for radius {radius}, got {n_windows}"
        )

    chi2_values, scores = [], []
    for i in range(n_windows):
        lo = max(0, i - radius)
        hi = min(n_windows, i + radius + 1)
        neighbours = np.concatenate([X[lo:i], X[i + 1 : hi]], axis=0)
        if neighbours.shape[0] < 2:
            continue
        estimate = welch_psd_from_windows(
            list(neighbours), sampling_frequency, window=window, average=average, source="local"
        )
        weights = inverse_psd_weights(estimate.psd, n_samples)
        chi2_values.append(
            float(
                whitened_chi2_per_dof(
                    X[i : i + 1], weights, sampling_frequency, n_samples=n_samples
                )[0]
            )
        )
        dispersion_scores = score_dispersion(
            X[i : i + 1], estimate.psd, weights, sampling_frequency, frequency_hz
        )[1]
        scores.extend(np.asarray(dispersion_scores).tolist())

    chi2_array = np.asarray(chi2_values, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64)
    reference_weights = inverse_psd_weights(
        welch_psd_from_windows(list(X[: 2 * radius]), sampling_frequency, window=window).psd,
        n_samples,
    )
    return CalibrationResult(
        label=label or f"local_radius_{radius}",
        chi2_per_dof_mean=float(np.mean(chi2_array)) if chi2_array.size else float("nan"),
        chi2_per_dof_std=float(np.std(chi2_array, ddof=1)) if chi2_array.size > 1 else float("nan"),
        score_dispersion=float(np.std(score_array, ddof=1)) if score_array.size > 1 else float("nan"),
        score_dispersion_standard_error=dispersion_standard_error(int(score_array.size)),
        n_windows=int(chi2_array.size),
        degrees_of_freedom=_effective_dof(n_samples, reference_weights),
        metadata={"radius_windows": radius, "psd_average": average, "psd_window": window},
    )


def cost_of_mismatch(
    matched: Iterable[float], mismatched: Iterable[float]
) -> dict[str, float]:
    """Express a calibration failure in estimator terms rather than in chi^2.

    A dispersion ratio is abstract. What a reader wants to know is what the
    inadequate model costs: how much wider the amplitude estimate's spread
    becomes, and therefore how much the detection threshold must move. This is
    the quantity GWOSC could not produce because it had no injections.
    """
    a = np.asarray(list(matched), dtype=np.float64)
    b = np.asarray(list(mismatched), dtype=np.float64)
    sigma_matched = float(np.std(a, ddof=1))
    sigma_mismatched = float(np.std(b, ddof=1))
    return {
        "sigma_matched": sigma_matched,
        "sigma_mismatched": sigma_mismatched,
        "sigma_inflation": float(sigma_mismatched / sigma_matched) if sigma_matched > 0 else float("nan"),
        "threshold_inflation_at_fixed_false_alarm": float(sigma_mismatched / sigma_matched)
        if sigma_matched > 0
        else float("nan"),
    }
