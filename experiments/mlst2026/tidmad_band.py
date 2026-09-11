"""T22 - narrowband representation on top of the frozen T10-T12 endpoint.

Everything about the endpoint, the labels, the calibration, the bootstrap, the
dependence assessment and the characterisation is imported unchanged from
:mod:`experiments.mlst2026.tidmad_endpoint`.  What this module adds is the one
thing T22 changes: the space the rank-2 subspace is fitted in.

T10-T12 fitted PCA and EMPCA on the whole window -- ``p = 65536`` real
coordinates against ``n = 350`` fit windows, so ``p/n = 187`` and the leading
sample eigendirections were noise-fitting artefacts about 8.6x stronger than
the injected tone (phase report 2026-09-10, section 4.7).  T22 removes the
whole-window dimension and keeps everything else: the learned space is the real
and imaginary parts of the ``B`` unitary-DFT bins centred on the injection bin,
i.e. ``2B`` real coordinates, with ``2B <= 34`` against the same 350 fit
windows.

Conventions
-----------
The band real view is the restriction of
:func:`tidmad_endpoint.to_real_view` to the band's columns, so it inherits the
``sqrt(2)`` isometry factor on interior bins: a diagonal weight ``1/N_k``
applied to both entries of bin ``k`` is exactly the noise-whitening metric, and
squared distances in the band view are squared distances in the signal's own
units.  Every band bin is required to be interior, so DC and Nyquist -- which
carry a single real coordinate each and would break the isometry -- can never
enter a band.

At ``B = 1`` the band view has two real coordinates and a rank-2 projector is
the identity, so band-PCA, band-EMPCA and the bin read must agree exactly.
:func:`assert_single_bin_identity` checks that to a relative ``1e-12`` and is
the protocol's implementation control.
"""
from __future__ import annotations

import numpy as np

from experiments.mlst2026 import tidmad_endpoint as te


def band_bins(k_inj: int, B: int, m: int) -> np.ndarray:
    """The ``B`` unitary-rFFT bins centred on ``k_inj``.

    Raises when the band would reach DC or Nyquist; the caller turns that into
    a declared ``band-at-edge`` outcome rather than silently shrinking the band.
    """
    if B < 1 or B % 2 == 0:
        raise ValueError(f"band width must be a positive odd integer, got {B}")
    half = (B - 1) // 2
    lo, hi = k_inj - half, k_inj + half
    if lo < 1 or hi > m - 2:
        raise ValueError(f"band of width {B} centred on bin {k_inj} leaves the "
                         f"interior bins [1, {m - 2}]")
    return np.arange(lo, hi + 1, dtype=int)


def band_real_columns(bins: np.ndarray) -> np.ndarray:
    """Real-view column indices of the band, interleaved ``(re, im)`` per bin."""
    cols = np.empty(2 * bins.size, dtype=int)
    for j, k in enumerate(bins):
        cols[2 * j], cols[2 * j + 1] = 2 * int(k) - 1, 2 * int(k)
    return cols


def to_band_view(C: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Isometric real view restricted to the band: ``(n, 2B)``.

    Computed directly from the complex coefficients rather than by building the
    full ``2(m-1)``-wide real view and slicing it, so the cost is ``O(nB)``
    instead of ``O(nm)``; :func:`band_view_matches_full_view` checks the two
    agree.
    """
    C = np.atleast_2d(np.asarray(C))
    s2 = np.sqrt(2.0)
    out = np.empty((C.shape[0], 2 * bins.size), dtype=np.float64)
    sub = C[:, bins]
    out[:, 0::2] = s2 * sub.real
    out[:, 1::2] = s2 * sub.imag
    return out


def from_band_view(V: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_band_view`: complex coefficients of the band bins."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    s2 = np.sqrt(2.0)
    return (V[:, 0::2] + 1j * V[:, 1::2]) / s2


def band_view_weights(N_bins: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Inverse noise bin powers expanded onto the band's real coordinates."""
    inv = 1.0 / np.clip(np.asarray(N_bins, dtype=np.float64)[bins],
                        np.finfo(float).tiny, None)
    return np.repeat(inv, 2)


def band_view_matches_full_view(C: np.ndarray, bins: np.ndarray,
                                atol: float = 1e-12) -> bool:
    """The fast band view equals the corresponding columns of the full view."""
    full = te.to_real_view(C)[:, band_real_columns(bins)]
    return bool(np.allclose(full, to_band_view(C, bins), rtol=0.0, atol=atol))


def fit_band_subspaces(C_fit: np.ndarray, bins: np.ndarray, N_bins: np.ndarray):
    """Rank-2 band-PCA and band-EMPCA on identical inputs, differing only in the
    metric.  Returns ``(fits, band_weights)``."""
    from noise_geometry.subspace import fit_pca, fit_weighted_pca

    V = to_band_view(C_fit, bins)
    w_band = band_view_weights(N_bins, bins)
    return {"pca": fit_pca(V, te.RANK),
            "empca": fit_weighted_pca(V, w_band, te.RANK)}, w_band


def _reconstruct_band(C_eval: np.ndarray, fit, bins: np.ndarray, k: int,
                      weights: np.ndarray | None) -> np.ndarray:
    from noise_geometry.subspace import project_onto_basis

    V = to_band_view(C_eval, bins)
    Vhat = project_onto_basis(V, fit.components, weights=weights, mean=fit.mean)
    centre = int(np.searchsorted(bins, k))
    return from_band_view(Vhat, bins)[:, centre]


def compute_band_estimates(C_eval: np.ndarray, C_fit: np.ndarray, k: int, B: int,
                           w: int, N_bins: np.ndarray,
                           cal: te.TransferCalibration) -> tuple[dict, dict]:
    """Per-window quadrature estimates at bin ``k`` for the eight estimators.

    Only ``pca`` and ``empca`` see the band; every other estimator is exactly
    the T10-T12 one, so the comparison isolates the representation.
    """
    m = C_eval.shape[1]
    bins = band_bins(k, B, m)
    c_i = C_eval[:, k]
    n = c_i.size
    fits, w_band = fit_band_subspaces(C_fit, bins, N_bins)

    P_fit = te.mean_bin_power(C_fit)
    P_signal = np.clip(P_fit - N_bins, 0.0, None)
    denom = P_signal[k] + N_bins[k]
    beta_k = float(P_signal[k] / denom) if denom > 0 else 0.0

    est = {
        "zero": np.zeros(n, dtype=np.complex128),
        "calibration-mean": np.full(n, cal.calibration_mean_coefficient, dtype=np.complex128),
        "raw-bin": c_i.copy(),
        "quadrature-OF": te.quadrature_of_bin(C_eval, k, m, w, N_bins),
        "pca": _reconstruct_band(C_eval, fits["pca"], bins, k, None),
        "empca": _reconstruct_band(C_eval, fits["empca"], bins, k, w_band),
        "wiener": beta_k * c_i,
        "shrinkage-cal": cal.shrinkage_lambda * c_i
        + (1.0 - cal.shrinkage_lambda) * cal.calibration_mean_coefficient,
    }
    ref = max(float(np.max(np.abs(est["raw-bin"]))), np.finfo(float).tiny)
    diagnostics = {
        "band_width": int(B),
        "band_bins": [int(b) for b in bins],
        "band_real_dimension": int(2 * bins.size),
        "fit_windows": int(C_fit.shape[0]),
        "dimension_over_samples": float(2 * bins.size / C_fit.shape[0]),
        "band_view_matches_full_view": band_view_matches_full_view(C_eval[:8], bins),
        "wiener_bin_gain": beta_k,
        "fit_window_bin_power": float(P_fit[k]),
        "noise_bin_power": float(N_bins[k]),
        "shrinkage_lambda": float(cal.shrinkage_lambda),
        "of_vs_rawbin_max_rel_diff": float(
            np.max(np.abs(est["quadrature-OF"] - est["raw-bin"])) / ref),
        "pca_vs_rawbin_max_rel_diff": float(
            np.max(np.abs(est["pca"] - est["raw-bin"])) / ref),
        "empca_vs_rawbin_max_rel_diff": float(
            np.max(np.abs(est["empca"] - est["raw-bin"])) / ref),
        "pca_vs_empca_max_rel_diff": float(
            np.max(np.abs(est["pca"] - est["empca"])) / ref),
        "band_energy_share_of_fit_windows": float(
            np.sum(P_fit[bins]) / max(float(np.sum(P_fit)), np.finfo(float).tiny)),
    }
    return est, diagnostics


IDENTITY_RTOL = 1e-12


def assert_single_bin_identity(diag: dict, label: str,
                               rtol: float = IDENTITY_RTOL) -> dict:
    """B = 1 control: band-PCA = band-EMPCA = bin read, to ``rtol``."""
    checks = {
        "pca_vs_rawbin": diag["pca_vs_rawbin_max_rel_diff"],
        "empca_vs_rawbin": diag["empca_vs_rawbin_max_rel_diff"],
        "pca_vs_empca": diag["pca_vs_empca_max_rel_diff"],
    }
    worst = max(checks.values())
    ok = worst <= rtol
    if not ok:
        raise AssertionError(
            f"B=1 identity control failed on {label}: max relative difference "
            f"{worst:.3e} > {rtol:.1e}; {checks}")
    return {"label": label, "passed": True, "tolerance": rtol,
            "max_relative_difference": worst, "checks": checks}


def dirichlet_leakage_profile(delta: float, bins: np.ndarray, k: int,
                              w: int) -> np.ndarray:
    """Predicted single-tone leakage power at each band bin, relative to ``k``.

    A tone ``delta`` bins off grid leaks into bin ``k + j`` with amplitude
    ``D(delta - j)``, so the predicted power ratio is
    ``|D(delta - j)|^2 / |D(delta)|^2``.  This is what the second-line
    admissibility test compares the observed band spectrum against, so a bin
    that is merely receiving the main line's own skirt does not count as a
    second line.
    """
    d0 = te.dirichlet_attenuation(delta, w)
    base = max(d0 * d0, np.finfo(float).tiny)
    out = np.empty(bins.size, dtype=np.float64)
    for i, b in enumerate(bins):
        d = te.dirichlet_attenuation(delta - (int(b) - int(k)), w)
        out[i] = d * d / base
    return out


#: Bins this close to the injection bin carry the main line's own skirt and
#: cannot be resolved from it at boxcar resolution; the T10-T12
#: characterisation uses the same guard of one bin.
SECOND_LINE_GUARD = 1


def second_line_excess(C_cal_ref: np.ndarray, bins: np.ndarray, k: int,
                       delta: float, w: int,
                       fraction: float = 0.1) -> dict:
    """Decide whether the band contains a second characterised line.

    **Declared test (amendment A4, 11 September 2026).**  A band bin ``b`` is a
    second line when all three hold:

    1. ``|b - k| > SECOND_LINE_GUARD`` -- the immediately adjacent bins always
       carry the main line's skirt;
    2. its mean calibration-reference bin power is at least ``fraction`` of the
       line bin power -- the protocol's ``multi_tone_power_fraction``;
    3. it is a **local maximum** relative to its neighbour one step *towards*
       ``k``, i.e. it does not merely continue the outward-decaying skirt.

    Condition 3 is what distinguishes a line from leakage.  The rule this
    replaced compared the observed band power against the *idealised boxcar
    Dirichlet* skirt of a single tone; measured TIDMAD skirts sit about four
    times above that idealisation (rectangular-window leakage plus sub-bin
    frequency jitter across the calibration windows), so that test rejected
    every band of width >= 3 on three of the four eligible development
    stretches even though each band profile decays monotonically outward on
    both sides -- the signature of leakage, not of a second line.  The
    Dirichlet profile is retained below as a reported diagnostic.
    """
    P = te.mean_bin_power(C_cal_ref)
    line = float(P[k])
    predicted = dirichlet_leakage_profile(delta, bins, k, w) * line
    observed = P[bins].astype(float)
    rel = observed / max(line, np.finfo(float).tiny)
    excess = np.clip(observed - predicted, 0.0, None) / max(line, np.finfo(float).tiny)

    idx = {int(b): j for j, b in enumerate(bins)}
    centre = idx[int(k)]
    hits = []
    for j, b in enumerate(bins):
        b = int(b)
        if abs(b - int(k)) <= SECOND_LINE_GUARD:
            continue
        inward = j + 1 if b < int(k) else j - 1
        if rel[j] >= fraction and rel[j] >= rel[inward]:
            hits.append({"bin": b, "fraction_of_line": float(rel[j]),
                         "inward_neighbour_fraction": float(rel[inward])})

    # Reported shape diagnostic: does the profile decay outward on both sides?
    def _monotone(seq):
        return all(seq[i + 1] <= seq[i] * 1.02 for i in range(len(seq) - 1))
    left = list(rel[:centre])[::-1]
    right = list(rel[centre + 1:])

    off = bins != k
    return {
        "band_bins": [int(b) for b in bins],
        "observed_bin_power": [float(v) for v in observed],
        "relative_to_line": [float(v) for v in rel],
        "predicted_leakage_bin_power": [float(v) for v in predicted],
        "excess_over_dirichlet_fraction_of_line": [float(v) for v in excess],
        "max_dirichlet_excess_off_centre": float(excess[off].max()) if off.any() else 0.0,
        "guard_bins": SECOND_LINE_GUARD,
        "fraction_threshold": float(fraction),
        "second_lines": hits,
        "has_second_line": bool(hits),
        "max_fraction_beyond_guard": float(max(
            [rel[j] for j, b in enumerate(bins)
             if abs(int(b) - int(k)) > SECOND_LINE_GUARD], default=0.0)),
        "profile_decays_outward_left": bool(_monotone(left)),
        "profile_decays_outward_right": bool(_monotone(right)),
    }
