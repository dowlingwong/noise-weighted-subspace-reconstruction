import numpy as np

from noise_weighted_sr.weights import build_of_one_sided_weights, make_inverse_psd_weights


def test_inverse_psd_weights_positive():
    psd = np.array([1.0, 2.0, 4.0])
    w = make_inverse_psd_weights(psd)
    assert np.all(w > 0)


def test_build_of_one_sided_weights_even_length():
    trace_len = 8
    psd = np.ones(trace_len // 2 + 1)
    w = build_of_one_sided_weights(psd, trace_len)
    assert w.shape == psd.shape
    assert w[0] == 0.0
    assert w[-1] == 1.0
    assert np.allclose(w[1:-1], 2.0)
