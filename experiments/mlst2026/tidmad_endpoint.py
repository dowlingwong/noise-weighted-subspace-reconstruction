"""T10/T11/T12 - TIDMAD quadrature endpoint, conventions and estimators.

Protocol: MS ``plan/MLST_EXPERIMENT_PROGRAMME_2026-09-10.md`` sections T10-T12
and ``plan/mlst_protocol_2026-09-10.json`` key ``tidmad``.

This module implements the *new* protocol.  It deliberately does not reuse the
historical Phase-2 runner (`noise_geometry.tidmad.analysis.run_metric_reversal`),
whose endpoint is a scalar truth-phase projection in reference units; the new
endpoint is a two-component complex quadrature in readout physical-amplitude
units.  The historical runner is left untouched.

Conventions fixed here (all part of the freeze)
-----------------------------------------------

*Unitary DFT.*  ``c = rfft(x) / sqrt(w)``.  With this normalisation Parseval
reads ``|c_0|^2 + |c_{m-1}|^2 + 2 sum_{interior} |c_k|^2 = sum_n x_n^2``, so an
rFFT bin coefficient squared is directly a *bin power* in the signal's own
squared units.  Every noise level in this module is a bin power ``N_k =
E|c_k|^2``; the conversion from a one-sided PSD density is
``N_k = (fs/2) * S_onesided(f_k)`` at an interior bin.  Keeping bin power as the
native unit is what makes the line-SNR definition of item 3 dimensionally
compatible without any implicit bandwidth factor.

*Physical amplitude (window-length normalisation).*  For a real tone
``A cos(2 pi k0 n / w + phi)`` at an **interior** bin, ``rfft[k0] = (A w / 2)
e^{i phi}``, hence the unitary coefficient is ``c_{k0} = (A sqrt(w) / 2)
e^{i phi}`` and ``|c| ~ sqrt(w)``.  Raw squared coefficient improvements are
therefore *not* comparable across window lengths.  The endpoint is expressed in
physical amplitude, ``a = 2 c / sqrt(w)``, whose modulus is the tone amplitude
in readout ADC units independent of ``w``.  Estimates and labels are transformed
identically, so the contrast is a difference of squared ADC amplitudes.

*Window / bin / off-bin handling.*  Rectangular (boxcar) window, no taper - the
diagonal-Fourier-weighting story of the paper is a boxcar-DFT statement.  The
injection bin ``k_inj`` is determined once per stretch from the
transfer-calibration windows of the reference channel and never re-estimated per
evaluation window.  ``k_inj`` is required to be interior (never DC or Nyquist),
so the factor 2 above always applies.  The tone generally sits off-bin by
``delta = f_inj w / fs - k_inj`` in [-0.5, 0.5]; the single-bin coefficient is
then attenuated by the Dirichlet factor ``D(delta) = sin(pi delta) /
(w sin(pi delta / w))``.  Estimate and label share the same bin and the same
``delta``, so the *contrast* is unaffected; the absolute amplitude scale carries
``D(delta)``, which is recorded per stretch rather than corrected away.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS_HZ = 10_000_000.0
RANK = 2

# --- representation -----------------------------------------------------------


def unitary_rfft(x: np.ndarray) -> np.ndarray:
    """Unitary rFFT along the last axis: ``rfft(x) / sqrt(w)``."""
    w = np.asarray(x).shape[-1]
    return np.fft.rfft(x, axis=-1) / np.sqrt(w)


def dirichlet_attenuation(delta: float, w: int) -> float:
    """Single-bin amplitude attenuation for a tone ``delta`` bins off-grid."""
    d = float(delta)
    if abs(d) < 1e-15:
        return 1.0
    return float(np.sin(np.pi * d) / (w * np.sin(np.pi * d / w)))


def to_real_view(C: np.ndarray) -> np.ndarray:
    """Isometric real view of unitary-rFFT rows of an even-length real signal.

    Layout ``[Re c_0, sqrt2 Re c_1, sqrt2 Im c_1, ..., Re c_{m-1}]`` of length
    ``w = 2(m-1)``.  The ``sqrt2`` on interior bins makes the map an isometry,
    so a diagonal weight ``1/N_k`` applied to both interior entries of bin ``k``
    is exactly the noise-whitening metric.
    """
    C = np.atleast_2d(np.asarray(C))
    n, m = C.shape
    out = np.empty((n, 2 * (m - 1)), dtype=np.float64)
    out[:, 0] = C[:, 0].real
    out[:, -1] = C[:, m - 1].real
    if m > 2:
        s2 = np.sqrt(2.0)
        out[:, 1:-1:2] = s2 * C[:, 1:m - 1].real
        out[:, 2:-1:2] = s2 * C[:, 1:m - 1].imag
    return out


def from_real_view(V: np.ndarray, m: int) -> np.ndarray:
    """Inverse of :func:`to_real_view`."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    n = V.shape[0]
    C = np.empty((n, m), dtype=np.complex128)
    C[:, 0] = V[:, 0]
    C[:, m - 1] = V[:, -1]
    if m > 2:
        s2 = np.sqrt(2.0)
        C[:, 1:m - 1] = (V[:, 1:-1:2] + 1j * V[:, 2:-1:2]) / s2
    return C


def real_view_weights(N: np.ndarray) -> np.ndarray:
    """Expand per-bin noise bin-powers ``N_k`` to real-view inverse weights."""
    N = np.asarray(N, dtype=np.float64)
    m = N.shape[0]
    inv = 1.0 / np.clip(N, np.finfo(float).tiny, None)
    out = np.empty(2 * (m - 1), dtype=np.float64)
    out[0] = inv[0]
    out[-1] = inv[m - 1]
    if m > 2:
        out[1:-1:2] = inv[1:m - 1]
        out[2:-1:2] = inv[1:m - 1]
    return out


def real_view_bin_slice(k: int, m: int) -> tuple[int, int]:
    """Real-view column indices ``(re, im)`` of interior bin ``k``."""
    if not (0 < k < m - 1):
        raise ValueError(f"bin {k} is not interior for m={m}")
    return 2 * k - 1, 2 * k


# --- noise / line quantities --------------------------------------------------


def bin_power_from_psd(psd_onesided: np.ndarray, fs: float = FS_HZ) -> np.ndarray:
    """Convert a one-sided PSD density to interior-bin expected power."""
    return np.asarray(psd_onesided, dtype=np.float64) * (fs / 2.0)


def mean_bin_power(C: np.ndarray) -> np.ndarray:
    """Mean ``|c_k|^2`` over rows - the native noise estimate of this module."""
    return np.mean(np.abs(np.asarray(C)) ** 2, axis=0)


def offline_noise_floor(C: np.ndarray, k: int, *, guard: int = 4,
                        half_width: int = 64) -> float:
    """Robust off-line noise bin power near bin ``k``.

    Median of ``|c|^2`` over the bins within ``half_width`` of ``k`` but more
    than ``guard`` bins away, averaged over rows first.  The median makes the
    estimate insensitive to a second tone inside the neighbourhood, which is
    exactly the multi-tone case the characterisation has to survive.
    """
    P = mean_bin_power(C)
    m = P.shape[0]
    lo, hi = max(1, k - half_width), min(m - 1, k + half_width + 1)
    idx = np.arange(lo, hi)
    idx = idx[np.abs(idx - k) > guard]
    if idx.size == 0:
        raise ValueError("empty off-line neighbourhood")
    return float(np.median(P[idx]))


def line_snr(C: np.ndarray, k: int, **kw) -> float:
    """Line SNR in compatible bin-power units: ``|c_k|^2 / N_offline(k)``.

    Both numerator and denominator are bin powers (units^2), so no PSD-density
    to bin-power conversion is implicit anywhere.  The numerator is the mean
    line bin power over the supplied windows.
    """
    P = mean_bin_power(C)
    return float(P[k] / offline_noise_floor(C, k, **kw))


# --- transfer calibration -----------------------------------------------------


@dataclass
class TransferCalibration:
    """Complex least-squares readout-on-reference gain at the injection bin."""

    g: complex
    n_windows: int
    residual_bin_power: float
    predicted_bin_power: float
    reference_bin_power: float
    split_half_g: tuple[complex, complex]
    calibration_mean_coefficient: complex
    shrinkage_lambda: float

    def as_dict(self) -> dict:
        return {
            "g_real": float(self.g.real), "g_imag": float(self.g.imag),
            "g_abs": float(abs(self.g)), "g_phase_rad": float(np.angle(self.g)),
            "n_calibration_windows": int(self.n_windows),
            "residual_bin_power": float(self.residual_bin_power),
            "predicted_readout_bin_power": float(self.predicted_bin_power),
            "reference_bin_power": float(self.reference_bin_power),
            "split_half_g_abs": [float(abs(self.split_half_g[0])),
                                 float(abs(self.split_half_g[1]))],
            "split_half_relative_difference": float(
                abs(self.split_half_g[0] - self.split_half_g[1])
                / max(abs(self.g), np.finfo(float).tiny)),
            "calibration_mean_coefficient_abs": float(abs(self.calibration_mean_coefficient)),
            "shrinkage_lambda": float(self.shrinkage_lambda),
        }


def fit_transfer(c_cal: np.ndarray, rho_cal: np.ndarray,
                 readout_noise_bin_power: float) -> TransferCalibration:
    """Complex LS gain ``g = sum conj(rho) c / sum |rho|^2`` plus the two other
    calibration-dependent quantities that the bootstrap must refit with it.

    The calibration-mean baseline ``mean(c_cal)`` and the analytical shrinkage
    weight ``lambda = S / (S + N)`` (``S`` the sample bin power of ``c`` about
    that mean, ``N`` the readout noise bin power) are estimated *here*, from the
    same windows, precisely so that a replicate that resamples the calibration
    windows refits all three together.
    """
    c = np.asarray(c_cal, dtype=np.complex128)
    r = np.asarray(rho_cal, dtype=np.complex128)
    den = float(np.sum(np.abs(r) ** 2))
    g = complex(np.sum(np.conj(r) * c) / den) if den > 0 else 0j
    resid = c - g * r
    n = c.size
    half = n // 2
    def _g(sl):
        d = float(np.sum(np.abs(r[sl]) ** 2))
        return complex(np.sum(np.conj(r[sl]) * c[sl]) / d) if d > 0 else 0j
    cbar = complex(np.mean(c))
    S = float(np.mean(np.abs(c - cbar) ** 2))
    N = float(readout_noise_bin_power)
    lam = S / (S + N) if (S + N) > 0 else 0.0
    return TransferCalibration(
        g=g, n_windows=int(n),
        residual_bin_power=float(np.mean(np.abs(resid) ** 2)),
        predicted_bin_power=float(np.mean(np.abs(g * r) ** 2)),
        reference_bin_power=float(np.mean(np.abs(r) ** 2)),
        split_half_g=(_g(slice(0, half)), _g(slice(half, n))),
        calibration_mean_coefficient=cbar,
        shrinkage_lambda=float(np.clip(lam, 0.0, 1.0)),
    )


def errors_in_variables_note(reference_snr: float) -> dict:
    """The attenuation statement, with its assumptions made explicit.

    The classical attenuation ``E[g_hat] ~ g / (1 + sigma_rho^2 / |rho|^2)``
    holds only if the reference coefficient error is (i) additive, (ii)
    uncorrelated with the true reference value, and (iii) uncorrelated with the
    readout error.  None of the three is established by the release, so a
    reference line SNR >= 100 is reported as a *necessary* condition that the
    protocol declared in advance, not as a proof that the gain bias is below
    1 %.  The split-half gain agreement and the predicted-vs-observed residual
    ratio in :class:`TransferCalibration` are the empirical calibration checks
    reported alongside it.
    """
    snr = float(reference_snr)
    return {
        "reference_line_snr": snr,
        "meets_declared_threshold_100": bool(snr >= 100.0),
        "nominal_attenuation_if_assumptions_hold": float(1.0 / (1.0 + 1.0 / snr))
        if snr > 0 else float("nan"),
        "assumptions_required": [
            "reference coefficient error is additive",
            "reference error uncorrelated with the true reference coefficient",
            "reference error uncorrelated with the readout error",
        ],
        "claim": "necessary declared condition, not a demonstrated bias bound",
    }


# --- estimators ---------------------------------------------------------------
#: Declared method list.  ``zero`` and ``calibration-mean`` are the two
#: usefulness references; ``raw-bin`` is the no-denoising quadrature;
#: ``quadrature-OF`` must equal it under diagonal Fourier weighting at a
#: bin-aligned tone; ``wiener`` and ``shrinkage-cal`` are the analytical and
#: shrinkage baselines that the protocol requires beyond zero/mean.
METHODS = ("zero", "calibration-mean", "raw-bin", "quadrature-OF",
           "pca", "empca", "wiener", "shrinkage-cal")

#: Methods whose per-window estimate changes when the calibration windows are
#: resampled, so the bootstrap must recompute them inside each replicate.
CALIBRATION_DEPENDENT = ("calibration-mean", "shrinkage-cal")


def fit_rank2_subspaces(C_fit: np.ndarray, N_bins: np.ndarray):
    """Fit the rank-2 PCA and EMPCA bases on the fit windows.

    Both fits run on the isometric real view of the unitary rFFT; they differ
    only in the metric (Euclidean vs ``1/N_k``), which is the whole point of the
    comparison.  The rank is fixed at 2 by the protocol and is never selected.
    """
    from noise_geometry.subspace import fit_pca, fit_weighted_pca

    V = to_real_view(C_fit)
    w_view = real_view_weights(N_bins)
    return {
        "pca": fit_pca(V, RANK),
        "empca": fit_weighted_pca(V, w_view, RANK),
    }, w_view


def _reconstruct_bin(C_eval: np.ndarray, fit, k: int, m: int,
                     weights: np.ndarray | None) -> np.ndarray:
    """Reconstruct evaluation windows on a fitted basis; return bin ``k``."""
    from noise_geometry.subspace import project_onto_basis

    V = to_real_view(C_eval)
    Vhat = project_onto_basis(V, fit.components, weights=weights, mean=fit.mean)
    return from_real_view(Vhat, m)[:, k]


def quadrature_of_bin(C_eval: np.ndarray, k: int, m: int, w: int,
                      N_bins: np.ndarray) -> np.ndarray:
    """PSD-weighted rank-2 quadrature OF at ``k``, evaluated the general way.

    The cos/sin templates at an exactly bin-aligned frequency are supported on
    bin ``k`` alone, so with a diagonal weight the weighted least-squares
    solution must return the raw bin coefficient exactly, whatever the weights
    are.  This is computed through the generic weighted-LS path rather than
    short-circuited, so that the equality with ``raw-bin`` is an actual check.
    """
    from noise_geometry.subspace import project_onto_basis

    t = np.arange(w, dtype=np.float64)
    omega = 2.0 * np.pi * k / w
    basis_time = np.vstack([np.cos(omega * t), np.sin(omega * t)])
    B = to_real_view(unitary_rfft(basis_time))
    V = to_real_view(C_eval)
    Vhat = project_onto_basis(V, B, weights=real_view_weights(N_bins),
                              mean=np.zeros(V.shape[1]))
    return from_real_view(Vhat, m)[:, k]


def compute_estimates(C_eval: np.ndarray, C_fit: np.ndarray, k: int, w: int,
                      N_bins: np.ndarray, cal: TransferCalibration) -> dict:
    """Per-window unitary-DFT quadrature estimates ``q_hat`` for every method."""
    m = C_eval.shape[1]
    c_i = C_eval[:, k]
    n = c_i.size
    fits, w_view = fit_rank2_subspaces(C_fit, N_bins)

    # Wiener per-bin gain, signal PSD estimated on the fit windows only.
    P_fit = mean_bin_power(C_fit)
    P_signal = np.clip(P_fit - N_bins, 0.0, None)
    beta_k = float(P_signal[k] / (P_signal[k] + N_bins[k])) if (P_signal[k] + N_bins[k]) > 0 else 0.0

    est = {
        "zero": np.zeros(n, dtype=np.complex128),
        "calibration-mean": np.full(n, cal.calibration_mean_coefficient, dtype=np.complex128),
        "raw-bin": c_i.copy(),
        "quadrature-OF": quadrature_of_bin(C_eval, k, m, w, N_bins),
        "pca": _reconstruct_bin(C_eval, fits["pca"], k, m, None),
        "empca": _reconstruct_bin(C_eval, fits["empca"], k, m, w_view),
        "wiener": beta_k * c_i,
        "shrinkage-cal": cal.shrinkage_lambda * c_i
        + (1.0 - cal.shrinkage_lambda) * cal.calibration_mean_coefficient,
    }
    diagnostics = {
        "wiener_bin_gain": beta_k,
        "fit_window_bin_power": float(P_fit[k]),
        "noise_bin_power": float(N_bins[k]),
        "shrinkage_lambda": float(cal.shrinkage_lambda),
        "of_vs_rawbin_max_abs_diff": float(
            np.max(np.abs(est["quadrature-OF"] - est["raw-bin"]))),
        "of_vs_rawbin_max_rel_diff": float(
            np.max(np.abs(est["quadrature-OF"] - est["raw-bin"]))
            / max(np.max(np.abs(est["raw-bin"])), np.finfo(float).tiny)),
    }
    return est, diagnostics


def to_physical_amplitude(q: np.ndarray, w: int) -> np.ndarray:
    """``a = 2 q / sqrt(w)`` - the window-length-normalised endpoint units."""
    return 2.0 * np.asarray(q) / np.sqrt(w)


def squared_errors(q: np.ndarray, t: np.ndarray, w: int) -> np.ndarray:
    """``e_i^2 = |a_hat_i - a_true_i|^2`` in squared readout ADC amplitude."""
    return np.abs(to_physical_amplitude(q, w) - to_physical_amplitude(t, w)) ** 2


# --- dependence assessment (item 4) -------------------------------------------

#: Declared candidate block lengths.  Evaluation stretches hold 150 windows, so
#: b = 16 leaves floor(150/16) = 9 >= 5 blocks; calibration stretches hold 64
#: windows, so the calibration grid stops at 8 (floor(64/8) = 8 >= 5).
BLOCK_CANDIDATES_EVAL = (1, 2, 4, 8, 16)
BLOCK_CANDIDATES_CAL = (1, 2, 4, 8)
MIN_BLOCKS = 5
STABILISATION_RATIO = 1.15


def circular_block_indices(rng, n: int, block: int) -> np.ndarray:
    """One circular-block-bootstrap resample of ``n`` indices."""
    if block <= 1:
        return rng.integers(0, n, size=n)
    b = min(block, n)
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=nb)
    return ((starts[:, None] + np.arange(b)[None, :]) % n).reshape(-1)[:n]


def block_variance_curve(values: np.ndarray, candidates, *, replicates: int,
                         seed: int) -> dict:
    """Block-bootstrap variance of the mean of ``values`` at each block length."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    out = {}
    for b in candidates:
        if n // b < MIN_BLOCKS:
            out[int(b)] = {"variance": float("nan"), "n_blocks": int(n // b),
                           "usable": False}
            continue
        rng = np.random.default_rng(seed + 1000 * int(b))
        means = np.empty(replicates)
        for j in range(replicates):
            means[j] = x[circular_block_indices(rng, n, int(b))].mean()
        out[int(b)] = {"variance": float(means.var(ddof=1)),
                       "n_blocks": int(n // b), "usable": True}
    return out


def autocorrelation(values: np.ndarray, max_lag: int = 30) -> dict:
    """Sample autocorrelation with Bartlett bands - reported, never the selector."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    xc = x - x.mean()
    denom = float(np.sum(xc * xc))
    if denom <= 0:
        return {"lags": [], "acf": [], "bartlett_band": []}
    acf, band, cum = [], [], 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        r = float(np.sum(xc[:-lag] * xc[lag:]) / denom)
        acf.append(r)
        band.append(float(1.96 * np.sqrt((1.0 + 2.0 * cum) / n)))
        cum += r * r
    return {"lags": list(range(1, len(acf) + 1)), "acf": acf,
            "bartlett_band": band,
            "n_exceeding_band": int(sum(abs(a) > b for a, b in zip(acf, band)))}


def select_block_length(curve: dict, candidates) -> dict:
    """Declared plateau rule (replaces the single adjacent variance ratio).

    A candidate ``b`` qualifies as a *plateau candidate* only when at least two
    larger usable candidates exist, so that "no larger candidate rises by more
    than 15 %" is a genuine plateau statement rather than a single adjacent-pair
    test.  On the declared grid ``(1, 2, 4, 8, 16)`` the plateau candidates are
    therefore ``1, 2, 4``.

    * ``stabilised`` -- the smallest plateau candidate ``b`` with
      ``max_{b' >= b} V(b') / V(b) <= 1.15``.
    * ``stabilised-at-endpoint`` -- no plateau candidate qualifies, but the top
      of the grid is locally flat in *both* of its ratios,
      ``V(8)/V(4) <= 1.15`` and ``V(16)/V(8) <= 1.15``.  ``b = 8`` is used and
      the result is flagged, because the evidence for the plateau is only the
      two topmost ratios.
    * ``inconclusive`` -- otherwise the variance is still rising where the block
      grid runs out.  ``b = b_max`` is used, the interval is flagged, and the
      affected file's verdict may be only "ordering" or "inconclusive", never
      "useful".

    Accepting the largest candidate as a plateau is never possible: there is
    nothing above it to compare against, so the condition would hold vacuously.
    That is precisely the endpoint ambiguity the protocol left open.
    """
    cands = [int(b) for b in candidates if curve.get(int(b), {}).get("usable")]
    if len(cands) < 2:
        return {"block_length": int(max(candidates)), "verdict": "inconclusive",
                "reason": "fewer than two usable block lengths",
                "plateau_candidates": cands, "ratios": {}}
    V = {b: curve[b]["variance"] for b in cands}
    bmax = cands[-1]
    plateau_candidates = cands[:-2]  # need >= 2 larger candidates
    ratios = {}
    for b in cands[:-1]:
        ratios[b] = {int(bp): float(V[bp] / V[b]) for bp in cands if bp >= b}
    for b in plateau_candidates:
        if max(ratios[b].values()) <= STABILISATION_RATIO:
            return {"block_length": b, "verdict": "stabilised",
                    "reason": (f"every larger candidate within "
                               f"{STABILISATION_RATIO} of V({b})"),
                    "plateau_candidates": plateau_candidates, "ratios": ratios}
    if len(cands) >= 3:
        top, mid, low = cands[-1], cands[-2], cands[-3]
        r_top = float(V[top] / V[mid])
        r_mid = float(V[mid] / V[low])
        if r_top <= STABILISATION_RATIO and r_mid <= STABILISATION_RATIO:
            return {"block_length": mid, "verdict": "stabilised-at-endpoint",
                    "reason": (f"no plateau candidate qualified, but V({mid})/V({low})"
                               f" = {r_mid:.3f} and V({top})/V({mid}) = {r_top:.3f}"
                               f" are both <= {STABILISATION_RATIO}; flagged"),
                    "endpoint_ratios": [r_mid, r_top],
                    "plateau_candidates": plateau_candidates, "ratios": ratios}
    endpoint_ratio = float(V[bmax] / V[cands[-2]])
    chain = [float(V[cands[i + 1]] / V[cands[i]]) for i in range(len(cands) - 1)]
    monotone = all(r >= 1.0 for r in chain)
    shape = ("variance still rising at the largest usable block" if monotone
             else "variance curve is non-monotone across the block grid, so the "
                  "large-block variance estimates are themselves unstable")
    return {"block_length": bmax, "verdict": "inconclusive",
            "reason": (f"{shape}; adjacent ratios "
                       f"{[round(r, 3) for r in chain]} over b={cands}; "
                       f"V({bmax})/V({cands[-2]}) = {endpoint_ratio:.3f}. "
                       f"Interval flagged and 'useful' verdict withheld."),
            "endpoint_ratio": endpoint_ratio,
            "adjacent_ratio_chain": chain,
            "monotone": bool(monotone),
            "plateau_candidates": plateau_candidates, "ratios": ratios}


def aggregate_block_curves(curves: list[dict], candidates) -> dict:
    """Aggregate normalised variance curves over development files/contrasts.

    Each curve is normalised by its own ``V(1)`` and the **maximum** over
    (file, contrast) is taken at every block length, so the most strongly
    dependent case drives the choice.  This is the declared aggregation rule;
    it is conservative by construction.
    """
    norm = []
    for c in curves:
        base = c.get(1, {}).get("variance")
        if base is None or not np.isfinite(base) or base <= 0:
            continue
        norm.append({int(b): (c[int(b)]["variance"] / base
                              if c.get(int(b), {}).get("usable") else np.nan)
                     for b in candidates})
    if not norm:
        return {"curve": {}, "selection": {"block_length": int(max(candidates)),
                                           "verdict": "inconclusive",
                                           "reason": "no usable curves"}}
    agg = {}
    for b in candidates:
        vals = [c[int(b)] for c in norm if np.isfinite(c.get(int(b), np.nan))]
        agg[int(b)] = {"variance": float(max(vals)) if vals else float("nan"),
                       "usable": bool(vals), "n_curves": len(vals)}
    # Per-curve selections are reported as a diagnostic. The declared decision
    # is the one taken on the aggregated (maximum) curve; because the maximum is
    # taken independently at each block length, the aggregate can be
    # non-monotone even when no individual curve is, and that is visible here.
    per_curve = []
    for c in norm:
        cc = {int(b): {"variance": c[int(b)], "usable": bool(np.isfinite(c[int(b)]))}
              for b in candidates}
        per_curve.append(select_block_length(cc, candidates))
    return {"curve": agg, "n_input_curves": len(norm),
            "per_curve_selections": per_curve,
            "per_curve_verdict_counts": {
                v: sum(1 for s_ in per_curve if s_["verdict"] == v)
                for v in sorted({s_["verdict"] for s_ in per_curve})},
            "selection": select_block_length(agg, candidates)}


# --- characterisation (T11) ---------------------------------------------------

#: Declared numeric thresholds, from plan/mlst_protocol_2026-09-10.json.
PROMINENCE_MIN = 1000.0
MULTI_TONE_HZ = 500.0
MULTI_TONE_POWER_FRACTION = 0.1
CHIRP_BINS = 0.5
COUPLED_SNR_MIN = 1.0
REFERENCE_SNR_MIN = 100.0
N_EVAL_REQUIRED = 150
OUTCOMES = ("eligible-coupled", "uncoupled", "multi-tone", "chirped", "underpowered")


def _subbin_peak(power: np.ndarray, k: int) -> float:
    """Quadratic (log-magnitude) interpolation of the peak location near ``k``."""
    if k <= 0 or k >= power.size - 1:
        return float(k)
    a, b, c = (np.log(max(power[k - 1], 1e-300)), np.log(max(power[k], 1e-300)),
               np.log(max(power[k + 1], 1e-300)))
    den = a - 2 * b + c
    return float(k) if abs(den) < 1e-300 else float(k + 0.5 * (a - c) / den)


def characterise(C_cal_ref: np.ndarray, C_cal_read: np.ndarray, w: int,
                 N_read_bins: np.ndarray, stretch_samples: int,
                 n_eval_available: int, *, search_lo_hz: float = 1000.0,
                 n_eval_required: int = N_EVAL_REQUIRED) -> dict:
    """Decide the stretch outcome from the transfer-calibration windows only.

    Uses *both* channels, because coupling is a statement about the readout.
    Every threshold is the declared one; every outcome is recorded, including
    the ones that exclude a stretch from selection and from the combined
    estimate.

    ``n_eval_required`` is the evaluation-window count the *calling* protocol
    declares.  The default is the T10-T12 allocation (150), so those runners are
    unchanged; a protocol with a longer evaluation block (T22 declares 300) must
    pass its own number, or a stretch with between 150 and that many available
    windows would escape ``underpowered`` and be scored on a truncated block.
    The value used is recorded beside the available count.
    """
    m = C_cal_ref.shape[1]
    df = FS_HZ / w
    P_ref = mean_bin_power(C_cal_ref)
    lo = max(1, int(np.ceil(search_lo_hz / df)))
    band = P_ref[lo:m - 1]
    k_inj = int(np.argmax(band)) + lo
    median_in_band = float(np.median(band))
    prominence = float(P_ref[k_inj] / median_in_band) if median_in_band > 0 else np.inf

    # sub-bin frequency and its drift across the calibration windows
    per_window = np.abs(C_cal_ref) ** 2
    ks = np.array([_subbin_peak(per_window[j], k_inj) for j in range(per_window.shape[0])])
    idx = np.arange(ks.size, dtype=np.float64)
    slope_bins_per_window = float(np.polyfit(idx, ks, 1)[0]) if ks.size > 2 else 0.0
    windows_in_stretch = stretch_samples / float(w)
    drift_bins_over_stretch = abs(slope_bins_per_window) * windows_in_stretch
    k_sub = float(np.median(ks))
    delta = float(k_sub - k_inj)

    # multi-tone: a second line within +-500 Hz carrying >= 10 % of the main line.
    # The declared band is evaluated on the bins it can actually reach at this
    # window length: `half = max(1, round(500 / df))` bins either side, with the
    # two immediately adjacent bins excluded because they carry the main line's
    # own Dirichlet leakage. At w = 4096 (df = 2441 Hz) one bin already spans
    # more than +-500 Hz, so the effective band is wider than declared and is
    # recorded as `multi_tone_band_hz_effective`; the rule is then bin-limited
    # and cannot resolve a sub-bin companion at all. That limitation is recorded
    # rather than hidden, and the sub-bin diagnostics below are reported beside
    # it (they do not gate: the gating rule is the declared one).
    guard = 1
    half = max(guard + 1, int(np.floor(MULTI_TONE_HZ / df)))
    nb = np.arange(max(1, k_inj - half), min(m - 1, k_inj + half + 1))
    nb = nb[np.abs(nb - k_inj) > guard]
    second_frac = float(P_ref[nb].max() / P_ref[k_inj]) if nb.size and P_ref[k_inj] > 0 else 0.0
    second_bin = int(nb[np.argmax(P_ref[nb])]) if nb.size else -1
    # The band can resolve a companion only if the nearest bin outside the guard
    # still lies inside the declared +-500 Hz; otherwise the smallest offset the
    # grid offers already overshoots the band and the rule is bin-limited.
    multi_tone_resolvable = bool((guard + 1) * df <= MULTI_TONE_HZ)

    # Sub-bin two-tone diagnostics (reported, never gating). Two unresolved
    # tones beat against each other, so the per-window line bin power is
    # strongly modulated across the stretch and the sub-bin peak location
    # wanders, whereas a genuine single tone gives a near-stationary pair.
    line_power_per_window = np.abs(C_cal_ref[:, k_inj]) ** 2
    lp_mean = float(np.mean(line_power_per_window))
    line_power_cv = float(np.std(line_power_per_window) / lp_mean) if lp_mean > 0 else float("nan")
    subbin_scatter = float(np.std(ks))

    ref_snr = line_snr(C_cal_ref, k_inj)
    # Leakage sensitivity of the off-line floor. A rectangular window spreads a
    # strong tone's Dirichlet skirt across the neighbourhood, so the median
    # off-line floor is biased *up* and the line SNR correspondingly *down*.
    # The bias is therefore conservative for the >= 100 eligibility gate (it can
    # never manufacture eligibility), but the size of it is reported rather than
    # assumed negligible: a wider guard excludes more of the skirt.
    ref_snr_guard16 = line_snr(C_cal_ref, k_inj, guard=16, half_width=128)
    cal = fit_transfer(C_cal_read[:, k_inj], C_cal_ref[:, k_inj], float(N_read_bins[k_inj]))
    readout_line_power = float(np.mean(np.abs(cal.g * C_cal_ref[:, k_inj]) ** 2))
    readout_snr = float(readout_line_power / N_read_bins[k_inj]) if N_read_bins[k_inj] > 0 else 0.0

    reasons = []
    n_eval_required = int(n_eval_required)
    if n_eval_available < n_eval_required:
        outcome = "underpowered"; reasons.append(f"{n_eval_available} < {n_eval_required} evaluation windows")
    elif prominence < PROMINENCE_MIN:
        outcome = "uncoupled"; reasons.append(f"reference line prominence {prominence:.3g} < {PROMINENCE_MIN}")
    elif second_frac >= MULTI_TONE_POWER_FRACTION:
        outcome = "multi-tone"; reasons.append(f"second line at bin {second_bin} carries {second_frac:.3f} of the main line")
    elif drift_bins_over_stretch > CHIRP_BINS:
        outcome = "chirped"; reasons.append(f"drift {drift_bins_over_stretch:.3f} bins > {CHIRP_BINS} over the stretch at w={w}")
    elif readout_snr < COUPLED_SNR_MIN:
        outcome = "uncoupled"; reasons.append(f"readout line SNR {readout_snr:.4g} < {COUPLED_SNR_MIN}")
    else:
        outcome = "eligible-coupled"

    return {
        "outcome": outcome, "reasons": reasons,
        "window_samples": int(w), "bin_width_hz": float(df),
        "k_inj": int(k_inj), "f_inj_hz": float(k_inj * df),
        "subbin_offset_delta": delta,
        "f_inj_subbin_hz": float(k_sub * df),
        "dirichlet_attenuation": dirichlet_attenuation(delta, w),
        "reference_line_prominence": prominence,
        "reference_line_snr": ref_snr,
        "reference_line_snr_wide_guard": ref_snr_guard16,
        "reference_line_snr_leakage_sensitivity": float(ref_snr_guard16 / ref_snr)
        if ref_snr > 0 else float("nan"),
        "readout_line_snr": readout_snr,
        "second_line_bin": second_bin,
        "second_line_power_fraction": second_frac,
        "multi_tone_band_hz_effective": float(half * df),
        "multi_tone_band_resolvable_at_declared_500hz": multi_tone_resolvable,
        "line_power_coefficient_of_variation": line_power_cv,
        "subbin_peak_scatter_bins": subbin_scatter,
        "drift_bins_per_window": slope_bins_per_window,
        "drift_bins_over_stretch": float(drift_bins_over_stretch),
        "n_evaluation_windows_available": int(n_eval_available),
        "n_evaluation_windows_required": n_eval_required,
        "readout_unit_target_supported": bool(ref_snr >= REFERENCE_SNR_MIN),
        "errors_in_variables": errors_in_variables_note(ref_snr),
        "transfer_calibration": cal.as_dict(),
    }


# --- joint bootstrap (item 2) -------------------------------------------------


def joint_bootstrap(q_est: dict, c_eval: np.ndarray, rho_eval: np.ndarray,
                    c_cal: np.ndarray, rho_cal: np.ndarray,
                    noise_bin_power: float, w: int, contrasts,
                    *, replicates: int, seed: int, b_eval: int, b_cal: int,
                    calibration_only: bool = False) -> dict:
    """Bootstrap ``mean(Delta)`` with the shared transfer error included.

    One replicate block-resamples the paired reference/readout **calibration**
    windows, refits ``g_hat`` and with it every calibration-dependent quantity
    that enters the inferential target -- the labels ``t_i = g_hat rho_i``, the
    calibration-mean baseline and the shrinkage weight -- and independently
    block-resamples the evaluation windows.

    Held conditional (not resampled, by declaration): the 350 fit windows and
    everything estimated from them (the PCA/EMPCA bases and the Wiener signal
    PSD), the science-file noise bin powers, the injection bin ``k_inj``, and
    the block lengths.  The reported interval is therefore a
    conditional-on-training interval; ``calibration_only=True`` repeats it with
    the evaluation windows held fixed so the shared-gain component is visible
    on its own.
    """
    rng = np.random.default_rng(seed)
    n_e, n_c = c_eval.size, c_cal.size
    out = {name: np.empty(replicates) for name in contrasts}
    for j in range(replicates):
        ic = circular_block_indices(rng, n_c, b_cal)
        cal_b = fit_transfer(c_cal[ic], rho_cal[ic], noise_bin_power)
        ie = (np.arange(n_e) if calibration_only
              else circular_block_indices(rng, n_e, b_eval))
        t_b = cal_b.g * rho_eval[ie]
        e2 = {}
        for name, q in q_est.items():
            if name == "calibration-mean":
                qb = np.full(ie.size, cal_b.calibration_mean_coefficient)
            elif name == "shrinkage-cal":
                qb = (cal_b.shrinkage_lambda * c_eval[ie]
                      + (1.0 - cal_b.shrinkage_lambda) * cal_b.calibration_mean_coefficient)
            else:
                qb = q[ie]
            e2[name] = squared_errors(qb, t_b, w)
        for (a, b) in contrasts:
            out[(a, b)][j] = float(np.mean(e2[a] - e2[b]))
    return out


def observed_contrast(q_est: dict, c_eval: np.ndarray, t_eval: np.ndarray,
                      w: int, a: str, b: str) -> dict:
    """Point values of the primary (squared) and secondary (absolute) contrasts."""
    ea = squared_errors(q_est[a], t_eval, w)
    eb = squared_errors(q_est[b], t_eval, w)
    ra = np.abs(to_physical_amplitude(q_est[a], w) - to_physical_amplitude(t_eval, w))
    rb = np.abs(to_physical_amplitude(q_est[b], w) - to_physical_amplitude(t_eval, w))
    return {
        "mean_delta_squared": float(np.mean(ea - eb)),
        "rmse_a": float(np.sqrt(np.mean(ea))), "rmse_b": float(np.sqrt(np.mean(eb))),
        "mean_absolute_error_contrast_secondary": float(np.mean(ra - rb)),
        "n_windows": int(ea.size),
    }


def percentile_interval(draws: np.ndarray, alpha: float = 0.05) -> dict:
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci95_lo": float(lo), "ci95_hi": float(hi),
            "bootstrap_mean": float(np.mean(draws)),
            "bootstrap_sd": float(np.std(draws, ddof=1)),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "replicates": int(draws.size)}


def usefulness_verdict(vs_zero: dict, vs_calmean: dict, dependence_verdict: str) -> dict:
    """Declared usefulness criterion, with inconclusive verdicts preserved.

    A method is *useful* only if it beats **both** references with 95 %
    intervals excluding zero.  A dependence assessment that did not stabilise
    withholds the "useful" verdict outright, however large the point estimate:
    the interval that would support it is not trusted.
    """
    both_positive = (vs_zero["mean_delta_squared"] > 0
                     and vs_calmean["mean_delta_squared"] > 0)
    both_exclude = vs_zero["ci_excludes_zero"] and vs_calmean["ci_excludes_zero"]
    if dependence_verdict == "inconclusive":
        return {"verdict": "inconclusive",
                "reason": "dependence assessment did not stabilise; the "
                          "declared rule withholds a 'useful' verdict"}
    if both_positive and both_exclude:
        return {"verdict": "useful", "reason": "beats zero and calibration-mean, "
                                               "both intervals exclude zero"}
    if both_positive:
        return {"verdict": "ordering", "reason": "positive against both references "
                                                 "but at least one interval includes zero"}
    return {"verdict": "not-useful",
            "reason": "does not beat both references on the point estimate"}


# --- across files -------------------------------------------------------------


def combine_files(per_file: list[dict]) -> dict:
    """Precision-weighted and unweighted combination with Cochran's Q.

    ``theta_IV`` is a *weighted* estimand: it is not equal-file physical
    performance, and every per-file value is reported beside it.
    """
    vals = np.array([f["mean_delta_squared"] for f in per_file], dtype=float)
    var = np.array([f["bootstrap_variance"] for f in per_file], dtype=float)
    ok = np.isfinite(vals) & np.isfinite(var) & (var > 0)
    if ok.sum() == 0:
        return {"n_files": 0, "note": "no file with a usable variance"}
    v, s = vals[ok], var[ok]
    wts = 1.0 / s
    theta_iv = float(np.sum(wts * v) / np.sum(wts))
    se_iv = float(np.sqrt(1.0 / np.sum(wts)))
    Q = float(np.sum(wts * (v - theta_iv) ** 2))
    dof = int(ok.sum() - 1)
    return {
        "n_files": int(ok.sum()),
        "theta_iv_precision_weighted": theta_iv,
        "theta_iv_se": se_iv,
        "theta_iv_ci95": [theta_iv - 1.96 * se_iv, theta_iv + 1.96 * se_iv],
        "unweighted_mean": float(np.mean(v)),
        "unweighted_sd": float(np.std(v, ddof=1)) if ok.sum() > 1 else float("nan"),
        "cochran_Q": Q, "Q_dof": dof,
        "I_squared": float(max(0.0, (Q - dof) / Q)) if Q > 0 else 0.0,
        "note": "precision-weighted estimand, not equal-file physical performance",
    }
