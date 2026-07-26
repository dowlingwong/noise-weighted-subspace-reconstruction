"""Phase 5: the linear noise-aware denoiser.

Roadmap: ``docs/plans/TIDMAD_IMPLEMENTATION_ROADMAP.md`` Phase 5.

The claim being tested is *not* "a linear filter beats deep denoisers". It is:

    how much of the gap between doing nothing and a trained deep denoiser is
    explained by using the correct noise geometry with a linear model?

That sentence is written here, before the experiment runs, so the result cannot
be retrofitted into a stronger claim afterwards.

The estimator is the Wiener filter, which is the linear minimum-mean-square
estimator under the same Gaussian model Paper 1 uses -- the same Sigma, applied
to a denoising task instead of an amplitude task. Its transfer function is

    H(f) = S_signal(f) / (S_signal(f) + S_noise(f)),

so it approaches one where the signal dominates and zero where the noise does.
Both spectra are estimated from *training* files and applied blind to
validation, which is the same information the deep baselines are given.

Windowing note. Denoising a long stream by independent blocks introduces edge
discontinuities. Overlap-add with a square-root Hann window and 50% overlap
reconstructs the stream exactly (the squared windows sum to a constant), so the
filter output is continuous and no artificial edges enter the score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["WienerFilter", "estimate_signal_psd", "wiener_transfer_function", "denoise_stream"]


@dataclass(frozen=True)
class WienerFilter:
    """A frequency-domain Wiener filter with its provenance attached."""

    frequencies: np.ndarray
    transfer: np.ndarray
    signal_psd: np.ndarray
    noise_psd: np.ndarray
    window_samples: int
    sampling_frequency: float

    def as_dict(self) -> dict:
        return {
            "window_samples": int(self.window_samples),
            "sampling_frequency_hz": float(self.sampling_frequency),
            "transfer_max": float(np.max(self.transfer)),
            "transfer_median": float(np.median(self.transfer)),
            "n_bins_above_half": int(np.count_nonzero(self.transfer > 0.5)),
        }


def estimate_signal_psd(
    signal_plus_noise_psd: np.ndarray,
    noise_psd: np.ndarray,
    *,
    floor: float = 0.0,
) -> np.ndarray:
    """Signal PSD by spectral subtraction, clipped at a floor.

    ``S_signal = max(S_total - S_noise, floor)``. The clip matters: sampling
    fluctuation makes the difference negative in bins where no signal exists,
    and a negative "signal" PSD would produce a transfer function outside
    ``[0, 1]`` -- an amplifier, not a filter.
    """
    total = np.asarray(signal_plus_noise_psd, dtype=np.float64)
    noise = np.asarray(noise_psd, dtype=np.float64)
    if total.shape != noise.shape:
        raise ValueError("PSD arrays must have the same shape")
    return np.clip(total - noise, float(floor), None)


def wiener_transfer_function(
    signal_psd: np.ndarray,
    noise_psd: np.ndarray,
    *,
    epsilon: float = 1e-30,
) -> np.ndarray:
    """Return ``S / (S + N)`` bin by bin, bounded to ``[0, 1]``."""
    s = np.clip(np.asarray(signal_psd, dtype=np.float64), 0.0, None)
    n = np.clip(np.asarray(noise_psd, dtype=np.float64), 0.0, None)
    return np.clip(s / (s + n + float(epsilon)), 0.0, 1.0)


def build_wiener_filter(
    signal_windows: np.ndarray,
    noise_windows: np.ndarray,
    sampling_frequency: float,
    *,
    window: str = "hann",
    average: str = "median",
) -> WienerFilter:
    """Fit the filter from training signal-bearing and signal-free windows."""
    from .psd import welch_psd_from_windows  # local import keeps the module import light

    signal_windows = np.atleast_2d(np.asarray(signal_windows, dtype=np.float64))
    noise_windows = np.atleast_2d(np.asarray(noise_windows, dtype=np.float64))
    if signal_windows.shape[1] != noise_windows.shape[1]:
        raise ValueError("signal and noise windows must have the same length")

    total = welch_psd_from_windows(
        list(signal_windows), sampling_frequency, window=window, average=average, source="training"
    )
    noise = welch_psd_from_windows(
        list(noise_windows), sampling_frequency, window=window, average=average, source="science"
    )
    signal_psd = estimate_signal_psd(total.psd, noise.psd)
    return WienerFilter(
        frequencies=total.frequencies,
        transfer=wiener_transfer_function(signal_psd, noise.psd),
        signal_psd=signal_psd,
        noise_psd=noise.psd,
        window_samples=int(signal_windows.shape[1]),
        sampling_frequency=float(sampling_frequency),
    )


def denoise_stream(stream: np.ndarray, wiener: WienerFilter) -> np.ndarray:
    """Apply the filter to a long trace by 50%-overlap square-root-Hann OLA.

    Returns an array of the same length as ``stream``. The first and last half
    windows receive partial window coverage; they are renormalized by the
    realized window-sum so the output is unbiased everywhere rather than tapered
    at the edges.
    """
    x = np.asarray(stream, dtype=np.float64)
    n = int(wiener.window_samples)
    if x.ndim != 1:
        raise ValueError("stream must be one-dimensional")
    if x.size < n:
        raise ValueError("stream is shorter than one window")

    hop = n // 2
    window = np.sqrt(np.hanning(n) + 1e-18)

    output = np.zeros_like(x)
    normalization = np.zeros_like(x)
    for start in range(0, x.size - n + 1, hop):
        segment = x[start : start + n] * window
        filtered = np.fft.irfft(np.fft.rfft(segment) * wiener.transfer, n=n)
        output[start : start + n] += filtered * window
        normalization[start : start + n] += window**2

    tail = x.size - ((x.size - n) // hop) * hop - n
    if tail > 0:  # cover the final partial hop with one last full window
        start = x.size - n
        segment = x[start:] * window
        filtered = np.fft.irfft(np.fft.rfft(segment) * wiener.transfer, n=n)
        output[start:] += filtered * window
        normalization[start:] += window**2

    covered = normalization > 1e-12
    output[covered] /= normalization[covered]
    output[~covered] = x[~covered]
    return output
