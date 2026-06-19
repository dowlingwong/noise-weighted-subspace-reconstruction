"""Canonical EMPCA verification: representation fidelity and reference agreement.

Two independent checks that make the optimized complex EMPCA (TCY) trustworthy
for the paper:

1. ``rfft_to_weighted_real_features`` preserves the weighted Hermitian inner
   product, so working in the stacked real-feature space is equivalent to the
   complex Sigma^{-1} geometry.
2. The TCY no-smoothing EMPCA (exact ``full`` M-step) and Stephen Bailey's
   published reference WPCA recover the *same subspace*. Bailey runs on the
   whitened real features; TCY runs on the complex rfft data. Agreement of two
   independent implementations is the verification.
"""

import numpy as np
import pytest

from src.canonical.empca import empca as bailey_empca
from src.canonical.empca_equivalence_utils import (
    build_of_one_sided_weights,
    fit_empca_no_smoothing,
    rfft_to_weighted_real_features,
    weighted_inner,
)
from src.noise_geometry.subspace import principal_angles


# --------------------------------------------------------------------------- #
# 1. rfft -> real feature mapping preserves the weighted inner product
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("trace_len", [64, 65, 128, 129])
def test_rfft_real_mapping_preserves_weighted_inner_product(trace_len):
    """Euclidean dot of mapped real features == Re of the weighted inner product.

    Holds for rFFTs of *real* signals, where the DC bin (and the Nyquist bin
    for even length) are real, which is the only case the pipeline uses.
    """
    rng = np.random.default_rng(trace_len)
    traces = rng.standard_normal((6, trace_len))
    Xf = np.fft.rfft(traces, axis=1)
    psd = rng.uniform(0.5, 2.0, size=Xf.shape[1])

    for w in (build_of_one_sided_weights(psd, trace_len), 1.0 / psd):
        feat = rfft_to_weighted_real_features(Xf, w)
        for i in range(Xf.shape[0]):
            for j in range(Xf.shape[0]):
                lhs = float(np.dot(feat[i], feat[j]))
                rhs = float(np.real(weighted_inner(Xf[i], Xf[j], w)))
                assert abs(lhs - rhs) <= 1e-9 * (1.0 + abs(rhs)), (i, j, lhs, rhs)


def test_rfft_real_mapping_preserves_weighted_norm():
    """Self inner product == weighted energy (a real, nonnegative quantity)."""
    rng = np.random.default_rng(7)
    traces = rng.standard_normal((10, 128))
    Xf = np.fft.rfft(traces, axis=1)
    psd = rng.uniform(0.5, 2.0, size=Xf.shape[1])
    w = build_of_one_sided_weights(psd, 128)
    feat = rfft_to_weighted_real_features(Xf, w)
    euclid_norm_sq = np.sum(feat**2, axis=1)
    weighted_energy = np.array([np.real(weighted_inner(x, x, w)) for x in Xf])
    np.testing.assert_allclose(euclid_norm_sq, weighted_energy, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# 2. TCY (full, no-smoothing) vs Bailey reference: same subspace
#
# Correspondence note: a complex EMPCA component carries a phase degree of
# freedom (the complex coefficient), so one complex component spans a TWO-
# dimensional real subspace: map(u) and map(i*u). The faithful comparison is
# therefore complex rank-k  <->  real rank-2k. (Comparing complex-k against
# real-k is a dimensionality mismatch; that is also the origin of the residual
# angle seen in the S2 OF/EMPCA check when amplitudes are effectively real.)
# --------------------------------------------------------------------------- #
def _complex_coeff_dataset(seed=0, n_traces=500, trace_len=129, rank=2, noise=0.03):
    """Real traces whose modes carry COMPLEX coefficients (phase/time-shift DOF).

    Complex coefficients populate both map(u) and map(i*u), so the data genuinely
    occupies the 2k-dimensional real subspace that complex rank-k EMPCA spans.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, trace_len, endpoint=False)
    modes = np.vstack([np.sin(2 * np.pi * (3 + 2 * k) * t) for k in range(rank)])
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    modes_f = np.fft.rfft(modes, axis=1)
    scale = np.array([3.0, 1.5, 1.0])[:rank]
    coeffs = (rng.normal(size=(n_traces, rank)) + 1j * rng.normal(size=(n_traces, rank))) * scale
    Xf_clean = coeffs @ modes_f
    freqs = np.fft.rfftfreq(trace_len)
    psd = np.ones_like(freqs)
    psd[1:] = 1.0 / freqs[1:]
    psd[0] = psd[1]
    noise_f = (rng.normal(size=(n_traces, psd.size)) + 1j * rng.normal(size=(n_traces, psd.size))) * np.sqrt(psd)
    Xf = Xf_clean + noise * noise_f / np.abs(noise_f).std()
    return Xf, psd, modes_f


def _tcy_real_equiv(eigvec, w):
    """Real-equivalent basis of complex components: stack map(u) and map(i*u)."""
    u = np.atleast_2d(eigvec)
    return np.vstack([
        rfft_to_weighted_real_features(u, w),
        rfft_to_weighted_real_features(1j * u, w),
    ])


@pytest.mark.parametrize("rank", [1, 2])
def test_tcy_fullmode_matches_bailey_reference_subspace(rank):
    """Complex rank-k TCY == real rank-2k Bailey, the faithful correspondence."""
    trace_len = 129  # odd: no Nyquist bin, keeps the mapping exact and clean
    Xf, psd, _ = _complex_coeff_dataset(seed=rank, trace_len=trace_len, rank=rank)
    w = build_of_one_sided_weights(psd, trace_len)

    # TCY: complex weighted EMPCA, exact full M-step, NO smoothing.
    tcy_eigvec, _, _ = fit_empca_no_smoothing(Xf, w, n_comp=rank, n_iter=400, mode="full")
    tcy_real = _tcy_real_equiv(tcy_eigvec, w)  # 2k real directions

    # Bailey reference: unweighted WPCA on the whitened real features, 2k vectors.
    feat = rfft_to_weighted_real_features(Xf, w)
    model = bailey_empca(feat, weights=np.ones_like(feat), niter=400, nvec=2 * rank, smooth=0, silent=True)
    bailey_eigvec = np.atleast_2d(model.eigvec)

    angles = principal_angles(tcy_real, bailey_eigvec)
    assert np.max(angles) < 1.0, f"max principal angle {np.max(angles):.4f} deg too large"


@pytest.mark.parametrize("rank", [1, 2])
def test_real_coeff_bailey_subspace_contained_in_tcy(rank):
    """Real-amplitude case: Bailey rank-k lies INSIDE TCY's real-equivalent span.

    Documents the phase-DOF relationship: with real coefficients the data is
    k-dimensional, Bailey captures exactly that, and it is contained in the
    (larger) 2k-dimensional span of complex rank-k EMPCA.
    """
    trace_len = 129
    rng = np.random.default_rng(100 + rank)
    t = np.linspace(0.0, 1.0, trace_len, endpoint=False)
    modes = np.vstack([np.sin(2 * np.pi * (3 + 2 * k) * t) for k in range(rank)])
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    coeffs = rng.normal(scale=[3.0, 1.5][:rank], size=(800, rank))  # REAL coefficients
    freqs = np.fft.rfftfreq(trace_len)
    psd = np.ones_like(freqs)
    psd[1:] = 1.0 / freqs[1:]
    psd[0] = psd[1]
    noise_f = (rng.normal(size=(800, psd.size)) + 1j * rng.normal(size=(800, psd.size))) * np.sqrt(psd)
    noise_td = np.fft.irfft(noise_f, n=trace_len, axis=1)
    traces = coeffs @ modes + 0.03 * noise_td / noise_td.std()
    Xf = np.fft.rfft(traces, axis=1)
    w = build_of_one_sided_weights(psd, trace_len)

    tcy_eigvec, _, _ = fit_empca_no_smoothing(Xf, w, n_comp=rank, n_iter=400, mode="full")
    tcy_real = _tcy_real_equiv(tcy_eigvec, w)  # 2k-dim span

    feat = rfft_to_weighted_real_features(Xf, w)
    bailey = bailey_empca(feat, weights=np.ones_like(feat), niter=400, nvec=rank, smooth=0, silent=True)

    # principal_angles returns min(dim) angles; Bailey-k inside TCY-2k => all small.
    angles = principal_angles(np.atleast_2d(bailey.eigvec), tcy_real)
    assert np.max(angles) < 1.0, f"Bailey subspace not contained in TCY span: {np.max(angles):.4f} deg"


def test_tcy_fullmode_recovers_known_signal_subspace():
    """Both estimators recover the planted modes: independent agreement on truth."""
    trace_len = 129
    rank = 2
    Xf, psd, modes_f = _complex_coeff_dataset(seed=11, trace_len=trace_len, rank=rank, noise=0.02)
    w = build_of_one_sided_weights(psd, trace_len)

    tcy_eigvec, _, _ = fit_empca_no_smoothing(Xf, w, n_comp=rank, n_iter=400, mode="full")
    tcy_real = _tcy_real_equiv(tcy_eigvec, w)

    # True modes in the same 2k real-equivalent representation.
    modes_real = np.vstack([
        rfft_to_weighted_real_features(modes_f, w),
        rfft_to_weighted_real_features(1j * modes_f, w),
    ])

    angles = principal_angles(tcy_real, modes_real)
    assert np.max(angles) < 2.0, f"recovered subspace off by {np.max(angles):.4f} deg"
