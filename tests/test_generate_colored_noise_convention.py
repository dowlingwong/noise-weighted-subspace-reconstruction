"""EXP-01: statistical contract tests for the core Fourier Gaussian sampler.

``noise_geometry.noise.generate_colored_noise`` backs S0, S1, S2, and S5 (see
``results/audits/noise_generator_provenance.json``). These tests pin the
discrete convention declared in
``noise_module/noise_module/PAPER_CONVENTION.md``: interior rFFT bins are
circular complex Gaussian with expected power ``P_k * f_s * N / 2``; DC and
(for even ``N``) Nyquist are real Gaussian with the full endpoint power
``P_k * f_s * N``. A prior version of this function scaled DC with the
interior-bin formula instead of the endpoint formula, silently halving its
expected variance; no test previously covered this, so these are regression
tests for that fix, not just contract documentation.

Repeated across a few fixed seeds with moment tolerances (not a single
p-value), per the plan's requirement that a statistical gate not depend on a
lucky draw.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.noise.synthetic import generate_colored_noise

SEEDS = [0, 1, 2]
N = 128
FS = 1.0
N_TRACES = 60_000
# Loose enough not to flake at this sample size, tight enough to have failed
# under the pre-fix DC scaling (which was off by a factor of 2 in variance).
REL_TOL = 0.05


def _flat_psd() -> np.ndarray:
    return np.ones(N // 2 + 1)


@pytest.mark.parametrize("seed", SEEDS)
def test_interior_bin_power_matches_declared_convention(seed):
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, N_TRACES)
    X = np.fft.rfft(x, axis=1)
    power = np.mean(np.abs(X) ** 2, axis=0)
    interior = power[2:-1]  # skip DC, bin 1 (still settling), and Nyquist
    expected = psd[2:-1] * FS * N / 2.0
    assert np.allclose(interior, expected, rtol=REL_TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_dc_bin_power_matches_endpoint_convention(seed):
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, N_TRACES)
    X = np.fft.rfft(x, axis=1)
    dc_power = float(np.mean(np.abs(X[:, 0]) ** 2))
    expected_endpoint = psd[0] * FS * N
    expected_interior = psd[0] * FS * N / 2.0
    assert np.isclose(dc_power, expected_endpoint, rtol=REL_TOL)
    # Guard against regressing to the old (wrong) interior-formula scaling.
    assert not np.isclose(dc_power, expected_interior, rtol=REL_TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_nyquist_bin_power_matches_endpoint_convention(seed):
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, N_TRACES)  # N even -> real Nyquist bin
    X = np.fft.rfft(x, axis=1)
    nyquist_power = float(np.mean(np.abs(X[:, -1]) ** 2))
    expected_endpoint = psd[-1] * FS * N
    assert np.isclose(nyquist_power, expected_endpoint, rtol=REL_TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_dc_and_nyquist_bins_are_real(seed):
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, 32)
    X = np.fft.rfft(x, axis=1)
    np.testing.assert_allclose(X[:, 0].imag, 0.0, atol=1e-8)
    np.testing.assert_allclose(X[:, -1].imag, 0.0, atol=1e-8)


@pytest.mark.parametrize("seed", SEEDS)
def test_expected_time_domain_variance_from_psd_integral(seed):
    """Discrete Parseval check: irfft output variance equals the declared
    one-sided PSD integral, independent of the per-bin endpoint convention.
    """
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, N_TRACES)
    empirical_variance = float(np.mean(np.var(x, axis=1)))
    # Parseval: Var[x] = (1/N^2) * (E|X_0|^2 + E|X_Nyq|^2 + 2*sum_interior E|X_k|^2)
    # with E|X_0|^2 = E|X_Nyq|^2 = P*fs*N (endpoint) and E|X_k|^2 = P*fs*N/2
    # (interior); the mirrored-pair factor of 2 exactly cancels the interior
    # bins' 1/2, leaving a plain (fs/N) * sum over one-sided PSD bins.
    expected_variance = (FS / N) * (psd[0] + psd[-1] + np.sum(psd[1:-1]))
    assert np.isclose(empirical_variance, expected_variance, rtol=REL_TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_normalized_interior_periodogram_is_exponential(seed):
    """Interior-bin |X_k|^2 / E|X_k|^2 should be Exponential(mean=1) --
    equivalently, real/imag parts standardized by scale/sqrt(2) are
    standard normal. Checked via first and second moments across many
    interior bins pooled together (avoids a single noisy chi-square draw).
    """
    rng = np.random.default_rng(seed)
    psd = _flat_psd()
    x = generate_colored_noise(rng, psd, N, FS, 4000)
    X = np.fft.rfft(x, axis=1)
    interior = X[:, 2:-1]
    expected_power = psd[2:-1] * FS * N / 2.0
    normalized = (np.abs(interior) ** 2) / expected_power[None, :]
    pooled = normalized.ravel()
    # Exponential(1): mean=1, var=1.
    assert abs(float(np.mean(pooled)) - 1.0) < 0.05
    assert abs(float(np.var(pooled)) - 1.0) < 0.15


def test_rejects_mismatched_psd_length():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_colored_noise(rng, np.ones(5), N, FS, 4)
