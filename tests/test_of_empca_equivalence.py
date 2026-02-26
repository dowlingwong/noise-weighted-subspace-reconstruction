import numpy as np

from noise_weighted_sr.metrics import weighted_cosine
from noise_weighted_sr.of import gls_amplitude


def test_rank1_gls_recovers_amplitude_in_low_noise():
    rng = np.random.default_rng(0)
    n = 64
    s = rng.normal(size=n) + 1j * rng.normal(size=n)
    w = np.ones(n)
    a_true = 2.5 - 0.7j
    x = a_true * s + 1e-6 * (rng.normal(size=n) + 1j * rng.normal(size=n))

    a_hat = gls_amplitude(x, s, w)
    assert np.isclose(a_hat, a_true, rtol=1e-5, atol=1e-5)


def test_weighted_cosine_is_one_for_colinear_vectors():
    rng = np.random.default_rng(1)
    a = rng.normal(size=32) + 1j * rng.normal(size=32)
    b = (3.0 - 2.0j) * a
    w = np.ones(32)
    c = weighted_cosine(a, b, w)
    assert np.isclose(c, 1.0, atol=1e-12)
