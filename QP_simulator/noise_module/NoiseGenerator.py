"""Single-channel stationary spectral noise generator."""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

import numpy as np

from scipy.fft import irfft, rfftfreq
from scipy.interpolate import interp1d

try:
    from .utils import resolve_rng
except ImportError:  # pragma: no cover - script execution fallback
    from utils import resolve_rng


class NoiseGenerator:
    """Generate stationary single-channel Gaussian noise from a target PSD.

    Public unit convention:

    * ``build_psd_density`` returns a one-sided PSD density in ADC^2 / Hz.
    * ``build_rfft_power`` converts that density to expected rFFT-bin power
      for NumPy's unnormalised ``rfft`` / ``irfft`` convention.
    * ``build_psd`` is kept as a backward-compatible alias for
      ``build_rfft_power`` because earlier notebooks used that name for the
      synthesis-domain quantity.
    """

    def __init__(self, config: dict[str, Any], rng: Any = None, seed: int | None = None):
        self.seed = seed
        self.rng = resolve_rng(rng=rng, seed=seed)
        self.config: dict[str, Any] = {}
        self._noise_path: str | None = None
        self._set_spectra()
        self.set_config(config)

    def set_config(self, config: dict[str, Any]) -> None:
        """Validate and store a new generator configuration."""
        expected_fields = ["noise_type", "noise_power", "sampling_frequency"]
        missing = [field for field in expected_fields if field not in config]
        if missing:
            raise RuntimeError(
                f"Configuration missing required field(s): {', '.join(missing)}."
            )

        self.config = deepcopy(config)
        self.set_noise_type(config["noise_type"])
        self.set_noise_power(config["noise_power"])
        self.sampling_frequency = float(config["sampling_frequency"])

    def set_noise_type(self, noise_type: str) -> None:
        """Set the analytic noise colour or load a custom PSD file."""
        analytic_types = "white blue violet brownian pink".split()
        if isinstance(noise_type, str) and noise_type.lower() in analytic_types:
            self.noise_type = noise_type.lower()
            self._noise_path = None
        elif os.path.isfile(str(noise_type)):
            self._noise_path = os.path.abspath(str(noise_type))
            self.noise_type = "custom"
            self._load_psd()
        else:
            raise RuntimeError(
                f"Configuration noise_type field {noise_type} is neither a supported "
                "noise type nor a PSD file path."
            )
        self.spectrum = self._spectra[self.noise_type]

    def set_noise_power(self, noise_power: float) -> None:
        """Set the target integrated noise power."""
        self.psd_area = float(noise_power)

    def _set_spectra(self) -> None:
        self._spectra = {
            "white": lambda f: np.ones_like(f, dtype=float),
            "blue": lambda f: f,
            "violet": lambda f: f**2,
            "brownian": lambda f: 1.0 / np.where(f == 0, np.inf, f**2),
            "pink": lambda f: 1.0 / np.where(f == 0, np.inf, f),
        }
        self._normalize = {
            "white": self._normalize_white,
            "blue": self._normalize_blue,
            "violet": self._normalize_violet,
            "brownian": self._normalize_brownian,
            "pink": self._normalize_pink,
        }

    def _load_psd(self) -> None:
        if self._noise_path is None:
            raise RuntimeError("Custom PSD path is not set.")
        self.noise_psd_data = np.load(self._noise_path)
        self._spectra["custom"] = interp1d(
            self.noise_psd_data[0],
            self.noise_psd_data[1],
            bounds_error=False,
            fill_value=(self.noise_psd_data[1][0], self.noise_psd_data[1][-1]),
        )
        self._normalize["custom"] = lambda f: np.ones_like(f, dtype=float)

    def build_psd_density(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return the one-sided PSD density in ADC^2 / Hz.

        For analytic colours, ``noise_power`` is interpreted as the desired
        discrete integral of the one-sided PSD over this exact rFFT grid:
        ``sum(psd_density) * (fs / N) == noise_power``. For custom PSD files,
        the stored density is used directly and ``noise_power`` is metadata only.
        """
        if N <= 0:
            raise ValueError("N must be positive.")

        frequencies = rfftfreq(N, d=1.0 / self.sampling_frequency)
        if self.noise_type == "custom":
            density = np.clip(
                np.asarray(self.spectrum(frequencies), dtype=float),
                a_min=0.0,
                a_max=None,
            )
        else:
            shape = np.asarray(self.spectrum(frequencies), dtype=float)
            density = self._normalize_shape_to_power(shape, N)

        if not return_metadata:
            return frequencies, density

        df = self.sampling_frequency / int(N)
        metadata = {
            "noise_type": self.noise_type,
            "noise_power": self.psd_area,
            "sampling_frequency": self.sampling_frequency,
            "seed": self.seed,
            "n_samples": N,
            "units": "ADC^2/Hz",
            "density_integral": float(np.sum(density) * df),
            "config": deepcopy(self.config),
        }
        return frequencies, density, metadata

    def build_rfft_power(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return expected rFFT coefficient power used for synthesis."""
        if return_metadata:
            frequencies, density, density_meta = self.build_psd_density(
                N,
                return_metadata=True,
            )
            power = self.psd_density_to_rfft_power(
                density,
                self.sampling_frequency,
                N,
            )
            metadata = {
                **density_meta,
                "units": "ADC^2 rFFT-bin power",
                "psd_density_units": "ADC^2/Hz",
                "rfft_power_total": float(np.sum(power)),
            }
            return frequencies, power, metadata

        frequencies, density = self.build_psd_density(N)
        power = self.psd_density_to_rfft_power(density, self.sampling_frequency, N)
        return frequencies, power

    def build_psd(
        self, N: int, return_metadata: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Backward-compatible alias for ``build_rfft_power``.

        The returned second array is not a density; use ``build_psd_density``
        when a physical PSD in ADC^2 / Hz is needed.
        """
        return self.build_rfft_power(N, return_metadata=return_metadata)

    def sample_stationary_gaussian_from_rfft_power(
        self,
        rfft_power: np.ndarray,
        N: int | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Sample a stationary Gaussian time series from rFFT-bin power."""
        rfft_power = np.asarray(rfft_power, dtype=float)
        if rfft_power.ndim != 1:
            raise ValueError("rfft_power must be a one-dimensional array.")
        if N is None:
            N = 2 * (len(rfft_power) - 1)
        if N <= 0:
            raise ValueError("N must be positive.")

        amplitude = np.sqrt(np.clip(rfft_power, a_min=0.0, a_max=None))
        spectrum = np.zeros_like(amplitude, dtype=complex)

        # True complex-Gaussian bins (E|eta_k|^2 = psd_k, Rayleigh modulus).
        # The previous implementation used CONSTANT-modulus random-phase bins
        # (eta_k = sqrt(psd_k) e^{i phi}): correct second-order statistics,
        # but every trace then has identical per-bin power, so any
        # distributional statistic built from weighted residual energies
        # (chi^2 goodness-of-fit, KS residual-whiteness tests) is degenerate
        # and meaningless. Gaussian bins are required for those tests and
        # match the Gaussian-ML assumptions of the framework.
        if len(amplitude) > 0:
            spectrum[0] = amplitude[0] * self.rng.standard_normal()
        if len(amplitude) > 2:
            re = self.rng.standard_normal(len(amplitude) - 2)
            im = self.rng.standard_normal(len(amplitude) - 2)
            spectrum[1:-1] = amplitude[1:-1] * (re + 1j * im) / np.sqrt(2.0)
        if len(amplitude) > 1:
            if N % 2 == 0:
                spectrum[-1] = amplitude[-1] * self.rng.standard_normal()
            else:
                re, im = self.rng.standard_normal(2)
                spectrum[-1] = amplitude[-1] * (re + 1j * im) / np.sqrt(2.0)

        signal = irfft(spectrum, n=N)
        if not return_metadata:
            return signal

        metadata = {
            "n_samples": N,
            "variance": float(np.var(signal)),
            "mean": float(np.mean(signal)),
        }
        return signal, metadata

    def sample_stationary_gaussian_from_psd(
        self,
        psd: np.ndarray,
        N: int | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Backward-compatible wrapper for rFFT-bin power input.

        Earlier code used ``psd`` to mean expected rFFT coefficient power.
        New code should call ``sample_stationary_gaussian_from_rfft_power`` or
        pass physical densities through ``psd_density_to_rfft_power`` first.
        """
        return self.sample_stationary_gaussian_from_rfft_power(
            psd,
            N=N,
            return_metadata=return_metadata,
        )

    def generate_noise(
        self, N: int, return_metadata: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        """Convenience wrapper combining PSD construction and sampling."""
        if return_metadata:
            frequencies, density, density_meta = self.build_psd_density(
                N,
                return_metadata=True,
            )
            rfft_power = self.psd_density_to_rfft_power(
                density,
                self.sampling_frequency,
                N,
            )
            signal, sample_meta = self.sample_stationary_gaussian_from_rfft_power(
                rfft_power,
                N=N,
                return_metadata=True,
            )
            metadata = {
                **density_meta,
                **sample_meta,
                "frequencies": frequencies,
                "psd_density": density,
            }
            return signal, metadata

        _, rfft_power = self.build_rfft_power(N)
        return self.sample_stationary_gaussian_from_rfft_power(rfft_power, N=N)

    def _normalize_shape_to_power(self, shape: np.ndarray, N: int) -> np.ndarray:
        """Scale a non-negative one-sided shape to the configured noise power."""
        values = np.asarray(shape, dtype=float)
        values = np.clip(values, a_min=0.0, a_max=None)
        total = float(np.sum(values))
        if total <= 0.0:
            raise ValueError(f"Noise type {self.noise_type!r} has zero PSD support.")
        df = self.sampling_frequency / int(N)
        return values * (self.psd_area / (total * df))

    @staticmethod
    def psd_density_to_rfft_power(
        psd_density: np.ndarray,
        sampling_frequency: float,
        N: int,
    ) -> np.ndarray:
        """Convert one-sided PSD density to expected unnormalised rFFT power."""
        if N <= 0:
            raise ValueError("N must be positive.")
        density = np.asarray(psd_density, dtype=float)
        if density.ndim != 1:
            raise ValueError("psd_density must be one-dimensional.")
        expected_bins = int(N) // 2 + 1
        if density.shape[0] != expected_bins:
            raise ValueError(
                f"PSD density length {density.shape[0]} does not match rFFT bins {expected_bins}."
            )
        power = np.clip(density, a_min=0.0, a_max=None) * float(sampling_frequency) * int(N)
        if int(N) > 2:
            upper = int(N) // 2 + 1 - (int(N) + 1) % 2
            power[1:upper] *= 0.5
        return power

    @staticmethod
    def _normalize_white(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        return 1.0 / max(float(np.max(frequencies) - np.min(frequencies)), np.finfo(float).eps)

    @staticmethod
    def _normalize_blue(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 2 - np.min(frequencies) ** 2)
        return 2.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_violet(frequencies: np.ndarray) -> float:
        if len(frequencies) < 2:
            return 1.0
        denom = float(np.max(frequencies) ** 3 - np.min(frequencies) ** 3)
        return 3.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_brownian(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) == 0:
            return 1.0
        if len(positive) == 1:
            return float(positive[0])
        denom = float(1.0 / positive[0] - 1.0 / positive[-1])
        return 1.0 / max(denom, np.finfo(float).eps)

    @staticmethod
    def _normalize_pink(frequencies: np.ndarray) -> float:
        positive = np.sort(frequencies[frequencies > 0])
        if len(positive) < 2:
            return 1.0
        denom = float(np.log(positive[-1]) - np.log(positive[0]))
        return 1.0 / max(denom, np.finfo(float).eps)
