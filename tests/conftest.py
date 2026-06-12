import sys
from pathlib import Path

import numpy as np
import pytest

# Make repo root importable so `import src` works without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


N_SAMPLES = 1024
FS = 1.0e5
N_TRACES = 256
RNG_SEED = 7


def _template(n=N_SAMPLES, fs=FS, tau_rise=8e-4, tau_decay=8e-3):
    t = np.arange(n) / fs
    s = np.exp(-t / tau_decay) - np.exp(-t / tau_rise)
    return s / s.max()


def _pink_psd(n=N_SAMPLES, fs=FS, level=1e-4):
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    J = np.empty_like(f)
    J[1:] = level * (f[1] / f[1:])  # 1/f
    J[0] = J[1]
    return f, J


def _colored_noise(rng, J, n, fs, n_traces):
    """Generate stationary Gaussian noise with one-sided PSD J (rfft bins)."""
    n_rfft = n // 2 + 1
    # variance per rfft bin for one-sided PSD: sigma_k^2 = J_k * fs * n / 2
    # (DC and Nyquist real-only handled below)
    scale = np.sqrt(J * fs * n / 2.0)
    re = rng.standard_normal((n_traces, n_rfft))
    im = rng.standard_normal((n_traces, n_rfft))
    X = (re + 1j * im) / np.sqrt(2.0) * scale[None, :]
    X[:, 0] = re[:, 0] * scale[0]
    if n % 2 == 0:
        X[:, -1] = re[:, -1] * scale[-1]
    return np.fft.irfft(X, n=n, axis=1)


@pytest.fixture(scope="session")
def sim_data():
    """Synthetic pulses: amplitude-scaled template + pink noise."""
    rng = np.random.default_rng(RNG_SEED)
    s = _template()
    f, J = _pink_psd()
    amps = rng.uniform(0.5, 1.5, size=N_TRACES)
    noise = _colored_noise(rng, J, N_SAMPLES, FS, N_TRACES)
    traces = amps[:, None] * s[None, :] + noise
    return {
        "template": s,
        "psd": J,
        "freqs": f,
        "amps": amps,
        "traces": traces,
        "fs": FS,
        "n": N_SAMPLES,
    }
