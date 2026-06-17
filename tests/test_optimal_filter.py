import numpy as np

from src.noise_geometry.filters import gls_amplitude


def test_gls_amplitude_is_unbiased_and_reaches_crb():
    rng = np.random.default_rng(21)
    template = np.exp(-np.linspace(0.0, 4.0, 64))
    sigma = 0.2
    metric = np.full(template.shape, 1.0 / sigma**2)
    amplitudes = rng.uniform(0.5, 1.5, size=6000)
    samples = amplitudes[:, None] * template + rng.normal(scale=sigma, size=(amplitudes.size, template.size))
    estimate = gls_amplitude(samples, template, metric)
    error = estimate - amplitudes
    predicted_sigma = 1.0 / np.sqrt(np.sum(template**2 * metric))

    assert abs(np.mean(error)) < 0.01
    assert 0.95 < np.std(error, ddof=1) / predicted_sigma < 1.05
