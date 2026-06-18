"""Item 2: the central rFFT<->real representation primitive.

Pins the conventions that S1/S2/S5 and the real-data experiments all depend on,
so they live in exactly one place: faithful real round-trip, weighted-feature
identity, inner-product preservation, and a single weight convention shared
between the canonical builder and the noise_geometry wrapper.
"""

import numpy as np
import pytest

from src.canonical.empca_equivalence_utils import (
    build_of_one_sided_weights,
    complex_to_real_whitened,
    real_to_rfft,
    real_weight_vector,
    rfft_to_real,
    rfft_to_weighted_real_features,
    weighted_inner,
)
from src.noise_geometry.noise import inverse_psd_weights


@pytest.mark.parametrize("trace_len", [64, 65, 128, 129])
def test_rfft_to_real_round_trip(trace_len):
    """rfft_to_real is a faithful real representation of a real signal."""
    rng = np.random.default_rng(trace_len)
    traces = rng.standard_normal((8, trace_len))
    Xf = np.fft.rfft(traces, axis=1)
    feat = rfft_to_real(Xf)
    assert feat.shape == (8, 2 * Xf.shape[1] - 1)
    Xf_rec = real_to_rfft(feat, Xf.shape[1])
    recon = np.fft.irfft(Xf_rec, n=trace_len, axis=1)
    np.testing.assert_allclose(recon, traces, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("trace_len", [64, 129])
def test_whitened_equals_unweighted_times_sqrt_weight(trace_len):
    """complex_to_real_whitened == rfft_to_real * sqrt(real_weight_vector)."""
    rng = np.random.default_rng(trace_len + 1)
    Xf = np.fft.rfft(rng.standard_normal((5, trace_len)), axis=1)
    psd = rng.uniform(0.5, 2.0, size=Xf.shape[1])
    w = build_of_one_sided_weights(psd, trace_len)

    whitened = complex_to_real_whitened(Xf, w)
    manual = rfft_to_real(Xf) * np.sqrt(real_weight_vector(w))[None, :]
    np.testing.assert_allclose(whitened, manual, rtol=1e-12, atol=1e-12)
    # back-compat: the central name aliases the original function.
    np.testing.assert_allclose(whitened, rfft_to_weighted_real_features(Xf, w))


def test_whitened_features_preserve_weighted_inner_product():
    rng = np.random.default_rng(3)
    Xf = np.fft.rfft(rng.standard_normal((6, 128)), axis=1)
    psd = rng.uniform(0.5, 2.0, size=Xf.shape[1])
    w = build_of_one_sided_weights(psd, 128)
    feat = complex_to_real_whitened(Xf, w)
    for i in range(Xf.shape[0]):
        for j in range(Xf.shape[0]):
            lhs = float(np.dot(feat[i], feat[j]))
            rhs = float(np.real(weighted_inner(Xf[i], Xf[j], w)))
            assert abs(lhs - rhs) <= 1e-9 * (1.0 + abs(rhs))


def test_real_weight_vector_layout():
    w = np.array([0.0, 2.0, 3.0, 4.0])  # one-sided, DC zeroed
    wr = real_weight_vector(w)
    np.testing.assert_array_equal(wr, np.array([0.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0]))


def test_whitened_projection_equals_gls_amplitude_and_of():
    """S1 routing: rank-1 whitened projection == gls_amplitude == OptimumFilter.fit."""
    from src.noise_geometry.synthetic import make_rank1_pulse_dataset
    from src.noise_geometry.filters import gls_amplitude
    from src.canonical.OptimumFilter import OptimumFilter

    d = make_rank1_pulse_dataset(n_traces=128, seed=1)
    Xf = np.fft.rfft(d.traces, axis=1)
    sf = np.fft.rfft(d.template)

    feat_X = complex_to_real_whitened(Xf, d.weights)
    feat_s = complex_to_real_whitened(sf, d.weights)
    amp_primitive = (feat_X @ feat_s) / float(feat_s @ feat_s)

    amp_gls = gls_amplitude(Xf, sf, d.weights)
    of = OptimumFilter(d.template, d.psd, d.sampling_frequency)
    amp_of = np.array([of.fit(tr)[0] for tr in d.traces])

    np.testing.assert_allclose(amp_primitive, amp_gls, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(amp_primitive, amp_of, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("trace_len", [64, 65, 256, 257])
def test_weight_convention_single_source_of_truth(trace_len):
    """noise_geometry.inverse_psd_weights delegates to the canonical builder."""
    rng = np.random.default_rng(trace_len)
    psd = rng.uniform(0.5, 3.0, size=trace_len // 2 + 1)
    ng = inverse_psd_weights(psd, trace_len)            # wrapper
    canon = build_of_one_sided_weights(psd, trace_len)  # single source
    np.testing.assert_allclose(ng, canon, rtol=1e-12, atol=1e-12)
