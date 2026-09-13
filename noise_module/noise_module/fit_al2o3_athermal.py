"""Phase 1/2 of the Al2O3_Al_athermal reverse-engineering plan.

Phase 1: fit each reference noise-budget ASD file
(``Al2O3_Al_athermal/{Johnson,SQUID,TD,Er}_noise.dat``) to a
``spectral_models`` component in PSD space, and validate the fitted
``CompositeSpectrum`` against ``total_noise.dat`` via the same
quadrature-sum check used in the original reference notebook
(``noise_sample/read_dat_1.ipynb``), at a much tighter tolerance.

Phase 2: sum the fitted (or raw) component PSDs on the dataset's native log
grid, resample onto a uniform rFFT grid for a target sampling frequency and
record length, and save the result as the two-row ``.npy`` PSD-density
artifact consumed by ``NoiseGenerator`` and by an optimal filter. The
resampling assumes a band-limited readout (in-band interpolation) unless
``--alias-fold`` is given; see ``build_psd_artifact`` for why that choice
changes the simulated resolution by a large factor.

See ``../AL2O3_AL_ATHERMAL_PLAN.md`` for background and rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

try:
    from . import psd_resampling as pr
    from .spectral_models import CompositeSpectrum, component_from_config
except ImportError:  # pragma: no cover - direct script execution fallback
    from noise_module.noise_module import psd_resampling as pr
    from noise_module.noise_module.spectral_models import (
        CompositeSpectrum,
        component_from_config,
    )


DATA_DIR = Path(__file__).resolve().parents[1] / "Al2O3_Al_athermal"
COMPONENT_FILES = {
    "Johnson": "Johnson_noise.dat",
    "SQUID": "SQUID_noise.dat",
    "TD": "TD_noise.dat",
    "Er": "Er_noise.dat",
}
TOTAL_FILE = "total_noise.dat"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_asd(filename: str, data_dir: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load a plaintext ``[freq_hz, asd]`` reference file.

    These are the raw two-column ``.dat`` files, not the ``.npy`` ``(2, N)``
    format ``psd_resampling.load_psd_density`` expects -- that loader is
    reserved for the Phase 2 artifact this script produces.
    """
    data = np.loadtxt(data_dir / filename)
    return data[:, 0], data[:, 1]


def load_all_components(data_dir: Path = DATA_DIR) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: load_asd(fname, data_dir) for name, fname in COMPONENT_FILES.items()}


# --------------------------------------------------------------------------
# Phase 1: per-component parametric fits, all performed in log10(PSD) space
# so that a fit is not dominated by whichever decade has the largest values.
# --------------------------------------------------------------------------


def _johnson_log_model(f: np.ndarray, log_s0: float) -> np.ndarray:
    return np.full_like(f, log_s0)


def _squid_log_model(f: np.ndarray, log_s0: float, log_a: float) -> np.ndarray:
    return np.log10(10.0**log_s0 + 10.0**log_a / f)


def _td_log_model(f: np.ndarray, log_s0: float, log_fc: float) -> np.ndarray:
    fc = 10.0**log_fc
    return log_s0 - np.log10(1.0 + (f / fc) ** 2)


def _er_log_model(f: np.ndarray, log_s0: float, alpha: float) -> np.ndarray:
    return log_s0 + alpha * np.log10(f)


def fit_johnson(freq: np.ndarray, psd: np.ndarray) -> dict[str, Any]:
    log_psd = np.log10(psd)
    p0 = [np.log10(np.median(psd))]
    popt, _ = curve_fit(_johnson_log_model, freq, log_psd, p0=p0)
    s0 = float(10.0 ** popt[0])
    components = [{"type": "white", "scale": s0, "name": "johnson"}]
    model_psd = np.full_like(freq, s0)
    return {"name": "Johnson", "params": {"s0": s0}, "components": components, "model_psd": model_psd}


def fit_squid(freq: np.ndarray, psd: np.ndarray) -> dict[str, Any]:
    log_psd = np.log10(psd)
    p0 = [np.log10(psd[-1]), np.log10(max(psd[0] * freq[0], 1e-12))]
    popt, _ = curve_fit(_squid_log_model, freq, log_psd, p0=p0, maxfev=20000)
    s0, a = float(10.0 ** popt[0]), float(10.0 ** popt[1])
    components = [
        {"type": "white", "scale": s0, "name": "squid_white"},
        {"type": "powerlaw", "scale": a, "exponent": -1.0, "reference_hz": 1.0, "name": "squid_1_f"},
    ]
    model_psd = s0 + a / freq
    knee_hz = a / s0
    return {
        "name": "SQUID",
        "params": {"s0": s0, "a": a, "knee_hz": knee_hz},
        "components": components,
        "model_psd": model_psd,
    }


def fit_td(freq: np.ndarray, psd: np.ndarray) -> dict[str, Any]:
    log_psd = np.log10(psd)
    p0 = [np.log10(psd[0]), np.log10(100.0)]
    popt, _ = curve_fit(_td_log_model, freq, log_psd, p0=p0, maxfev=20000)
    s0, fc = float(10.0 ** popt[0]), float(10.0 ** popt[1])
    components = [
        {"type": "rolloff", "scale": s0, "corner_hz": fc, "order": 2.0, "kind": "lowpass", "name": "td"}
    ]
    model_psd = s0 / (1.0 + (freq / fc) ** 2)
    return {"name": "TD", "params": {"s0": s0, "corner_hz": fc}, "components": components, "model_psd": model_psd}


def fit_er(freq: np.ndarray, psd: np.ndarray) -> dict[str, Any]:
    log_psd = np.log10(psd)
    log_freq = np.log10(freq)
    # Linear in log-log space; curve_fit for consistency with the other fits.
    p0 = [np.log10(psd[0]), -0.9]
    popt, _ = curve_fit(_er_log_model, freq, log_psd, p0=p0, maxfev=20000)
    s0, alpha = float(10.0 ** popt[0]), float(popt[1])
    components = [{"type": "powerlaw", "scale": s0, "exponent": alpha, "reference_hz": 1.0, "name": "er"}]
    model_psd = s0 * freq**alpha
    return {"name": "Er", "params": {"s0": s0, "alpha": alpha}, "components": components, "model_psd": model_psd}


FIT_FUNCS = {
    "Johnson": fit_johnson,
    "SQUID": fit_squid,
    "TD": fit_td,
    "Er": fit_er,
}


def fit_all_components(
    components: dict[str, tuple[np.ndarray, np.ndarray]]
) -> dict[str, dict[str, Any]]:
    results = {}
    for name, (freq, asd) in components.items():
        psd = asd**2
        results[name] = FIT_FUNCS[name](freq, psd)
    return results


def build_composite(fit_results: dict[str, dict[str, Any]]) -> CompositeSpectrum:
    component_configs = [cfg for res in fit_results.values() for cfg in res["components"]]
    return CompositeSpectrum([component_from_config(cfg) for cfg in component_configs])


# --------------------------------------------------------------------------
# Acceptance check: same quadrature-sum diagnostic as read_dat_1.ipynb
# (median/max/rms fractional deviation of total_asd / model_asd), but at a
# tolerance appropriate for a fitted model rather than the notebook's
# workflow-sanity thresholds (0.50 / 0.20).
# --------------------------------------------------------------------------


def validate_against_total(
    freq: np.ndarray,
    total_asd: np.ndarray,
    composite: CompositeSpectrum,
    *,
    max_frac_dev_threshold: float = 0.01,
    rms_frac_dev_threshold: float = 0.01,
) -> dict[str, float]:
    df = float(np.mean(np.diff(freq)))
    model_psd, _ = composite.evaluate(freq, df)
    model_asd = np.sqrt(model_psd)

    valid = (freq > 0) & np.isfinite(total_asd) & np.isfinite(model_asd) & (total_asd > 0) & (model_asd > 0)
    ratio = total_asd[valid] / model_asd[valid]
    frac_dev = np.abs(ratio - 1.0)

    diagnostics = {
        "median_ratio": float(np.median(ratio)),
        "max_frac_dev": float(np.max(frac_dev)),
        "rms_frac_dev": float(np.sqrt(np.mean((ratio - 1.0) ** 2))),
    }
    if diagnostics["max_frac_dev"] > max_frac_dev_threshold or diagnostics["rms_frac_dev"] > rms_frac_dev_threshold:
        raise RuntimeError(
            "Fitted composite spectrum does not reproduce total_noise.dat within "
            f"tolerance: {diagnostics}"
        )
    return diagnostics


# --------------------------------------------------------------------------
# Phase 2: native-grid PSD sum -> resampled rFFT-grid PSD artifact
# --------------------------------------------------------------------------


def native_grid_psd(
    components: dict[str, tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """Sum component PSDs (ASD**2) in power on the shared native log grid."""
    freq = None
    total_psd = None
    for f, asd in components.values():
        if freq is None:
            freq = f
            total_psd = asd**2
        else:
            if not np.array_equal(freq, f):
                raise ValueError("Components are not on a shared frequency grid.")
            total_psd = total_psd + asd**2
    assert freq is not None and total_psd is not None
    return freq, total_psd


def build_psd_artifact(
    freq: np.ndarray,
    native_psd: np.ndarray,
    *,
    sampling_frequency: float,
    n_samples: int,
    out_path: Path,
    alias_fold: bool = False,
    anti_alias_cutoff_hz: float | None = None,
    anti_alias_transition_hz: float = 0.0,
) -> dict[str, Any]:
    """Resample the native-grid PSD onto a uniform rFFT grid and save it.

    The reference budget extends to 3e7 Hz with a *white* SQUID+Johnson floor
    (about 0.093 uPhi0^2/Hz), while the target Nyquist is typically 0.5 MHz.
    The two ways of bringing that budget onto the DAQ grid differ by a large
    factor and encode a physical assumption about the readout chain:

    * ``alias_fold=False`` (default): in-band interpolation, optionally
      multiplied by a cosine anti-alias power response
      (``anti_alias_cutoff_hz``). This models a readout whose bandwidth is
      limited before sampling -- a flux-locked-loop SQUID (few MHz) followed
      by a fast digitizer with a decimation filter, or an analogue anti-alias
      filter (ECHo: ~100 kHz). It is the correct choice for MMC DAQs of the
      DELight/ECHo type.
    * ``alias_fold=True``: fold every alias of the 30 MHz budget into the
      target band. With no bandwidth limit that folds ~60 copies of the white
      floor into the band, raising the in-band floor ~60x in PSD and the
      optimal-filter baseline sigma ~5.7x on the default grid. Use only when
      the readout really has no bandwidth limit below the source's 30 MHz,
      or pass ``anti_alias_cutoff_hz`` to fold only the part that survives
      the filter.

    ``anti_alias_cutoff_hz`` is applied in both modes so the same readout
    assumption can be expressed either way.
    """
    if alias_fold:
        target_f, target_psd, meta = pr.alias_fold_psd_density(
            freq,
            native_psd,
            sampling_frequency,
            n_samples,
            anti_alias_cutoff_hz=anti_alias_cutoff_hz,
            anti_alias_transition_hz=anti_alias_transition_hz,
        )
    else:
        target_f, target_psd, meta = pr.inband_resample_psd_density(
            freq,
            native_psd,
            sampling_frequency,
            n_samples,
            anti_alias_cutoff_hz=anti_alias_cutoff_hz,
            anti_alias_transition_hz=anti_alias_transition_hz,
            allow_extrapolation=True,
        )
    pr.save_psd_density(out_path, target_f, target_psd)
    meta["readout_bandwidth_assumption"] = (
        "no bandwidth limit below source maximum (all aliases folded)"
        if alias_fold and anti_alias_cutoff_hz is None
        else "band-limited before sampling (in-band or filtered fold)"
    )
    return meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Al2O3_Al_athermal directory.")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Where to write fit/artifact outputs.")
    parser.add_argument("--sampling-frequency", type=float, default=1_000_000.0, help="Target DAQ fs in Hz.")
    parser.add_argument("--samples", type=int, default=131_072, help="Target record length in samples.")
    parser.add_argument(
        "--alias-fold",
        action="store_true",
        help=(
            "Fold all aliases of the 30 MHz budget into the target band (assumes a "
            "readout with no bandwidth limit before sampling). Default: in-band "
            "interpolation, i.e. a band-limited readout/DAQ."
        ),
    )
    parser.add_argument(
        "--anti-alias-cutoff-hz",
        type=float,
        default=None,
        help="Cosine anti-alias power cutoff applied to the source budget before resampling (readout/decimation bandwidth).",
    )
    parser.add_argument(
        "--anti-alias-transition-hz",
        type=float,
        default=0.0,
        help="Width of the cosine anti-alias transition band.",
    )
    parser.add_argument("--max-frac-dev", type=float, default=0.01, help="Acceptance threshold for the quadrature-sum check.")
    parser.add_argument("--rms-frac-dev", type=float, default=0.01, help="Acceptance threshold for the quadrature-sum check.")
    args = parser.parse_args()

    components = load_all_components(args.data_dir)
    total_freq, total_asd = load_asd(TOTAL_FILE, args.data_dir)

    print("Phase 1: fitting components...")
    fit_results = fit_all_components(components)
    for name, res in fit_results.items():
        print(f"  {name}: {res['params']}")

    composite = build_composite(fit_results)
    diagnostics = validate_against_total(
        total_freq,
        total_asd,
        composite,
        max_frac_dev_threshold=args.max_frac_dev,
        rms_frac_dev_threshold=args.rms_frac_dev,
    )
    print(f"  quadrature-sum check vs total_noise.dat: {diagnostics}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fit_config_path = args.out_dir / "al2o3_athermal_fit.json"
    fit_config_path.write_text(
        json.dumps(
            {
                "components": [cfg for res in fit_results.values() for cfg in res["components"]],
                "params": {name: res["params"] for name, res in fit_results.items()},
                "validation": diagnostics,
            },
            indent=2,
        )
    )
    print(f"Wrote fit config: {fit_config_path}")

    print("Phase 2: building resampled noise PSD artifact...")
    native_freq, native_psd = native_grid_psd(components)
    # Sanity check: native-grid sum should match the raw total_noise.dat file
    # (independent of the Phase 1 fit) before it is resampled.
    native_asd = np.sqrt(native_psd)
    raw_ratio = total_asd / native_asd
    print(
        "  native-grid quadrature check vs total_noise.dat: "
        f"max_frac_dev={float(np.max(np.abs(raw_ratio - 1.0))):.3e}"
    )

    artifact_path = args.out_dir / "al2o3_athermal_total_psd.npy"
    meta = build_psd_artifact(
        native_freq,
        native_psd,
        sampling_frequency=args.sampling_frequency,
        n_samples=args.samples,
        out_path=artifact_path,
        alias_fold=args.alias_fold,
        anti_alias_cutoff_hz=args.anti_alias_cutoff_hz,
        anti_alias_transition_hz=args.anti_alias_transition_hz,
    )
    print(f"Wrote PSD artifact: {artifact_path} ({meta})")


if __name__ == "__main__":
    main()
