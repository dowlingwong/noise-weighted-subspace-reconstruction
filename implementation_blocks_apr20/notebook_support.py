"""Shared utilities for the Apr 20 implementation notebooks.

The notebooks are intentionally written around one canonical analysis contract:

- mean baseline subtraction over a fixed pretrigger window;
- deterministic train/validation/test splits;
- one-sided PSD weighting for rFFT-domain comparisons;
- explicit gauge alignment before coefficient comparisons;
- graceful fallback to synthetic data when local real traces are unavailable.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy import linalg, stats
from scipy.ndimage import gaussian_filter1d

def _discover_repo_root() -> Path:
    if "__file__" in globals():
        start = Path(__file__).resolve()
        candidates = [start.parent, *start.parents]
    else:  # pragma: no cover - notebook execution fallback
        start = Path.cwd().resolve()
        candidates = [start, *start.parents]

    for candidate in candidates:
        if (candidate / "implementation_blocks_apr20").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError(f"Could not locate repo root from start path: {start}")


REPO_ROOT = _discover_repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from OptimumFilter import OptimumFilter  # noqa: E402
from PSDCalculator import calculate_psd  # noqa: E402
from empca_TCY_optimized import empca_solver, orthonormalize, smooth  # noqa: E402
from empca_equivalence_utils import (  # noqa: E402
    build_of_one_sided_weights,
    phase_align_basis,
    project_gls,
    weighted_cosine,
    weighted_inner,
    weighted_residual_energy,
)
from noise_module.NoiseGenerator import NoiseGenerator  # noqa: E402
from noise_module.artifact_injector import ArtifactInjector  # noqa: E402
from noise_module.multichannel_noise import MultiChannelNoiseGenerator  # noqa: E402
from noise_module.temporal_noise import TemporalNoiseWrapper  # noqa: E402


@dataclass(slots=True)
class CanonicalConfig:
    seed: int = 314159
    sampling_frequency: float = 1.0
    trace_len: int = 32768
    pretrigger: int = 4000
    baseline_method: str = "mean"
    split_train: float = 0.60
    split_val: float = 0.20
    split_test: float = 0.20
    empirical_rank_max: int = 6
    default_empca_iter: int = 80
    default_empca_patience: int = 12
    default_savgol_window: int = 15
    default_savgol_polyord: int = 3
    default_psd_floor_quantile: float = 0.01
    real_dataset_candidates: tuple[str, ...] = (
        "data/k_alpha_traces.h5",
        "data/sample/k_alpha_traces.h5",
        "data/sample/traces.h5",
        "k_alpha_traces.h5",
    )
    rq_dataset_candidates: tuple[str, ...] = (
        "data/k_alpha_rqs.h5",
        "k_alpha_rqs.h5",
    )
    template_candidates: tuple[str, ...] = (
        "PC_interpretation/template_K_alpha_tight.npy",
        "decoder/make_template/outputs/QP_template_smooth.npy",
        "decoder/make_template/outputs/QP_template_std.npy",
    )
    psd_candidates: tuple[str, ...] = (
        "data/weight/noise_psd_pink.npy",
        "data/weight/noise_psd_white.npy",
        "noise_module/tutorial_custom_psd.npy",
    )
    real_trace_keys: tuple[str, ...] = (
        "traces",
        "trace",
        "traces_clean",
        "traces_MMC",
    )
    results_tables_dir: str = "results/tables"
    results_figures_dir: str = "results/figures"
    results_manifests_dir: str = "results/manifests"

    def validate(self) -> "CanonicalConfig":
        total = self.split_train + self.split_val + self.split_test
        if not np.isclose(total, 1.0):
            raise ValueError(f"train/val/test fractions must sum to 1, got {total}")
        if self.pretrigger <= 0 or self.pretrigger >= self.trace_len:
            raise ValueError("pretrigger must lie inside the trace length")
        return self


@dataclass(slots=True)
class DatasetBundle:
    traces_raw: np.ndarray
    traces_baseline: np.ndarray
    traces_freq: np.ndarray
    split_indices: dict[str, np.ndarray]
    template_time: np.ndarray
    template_freq: np.ndarray
    psd_one_sided: np.ndarray
    weights_one_sided: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_results_dirs(cfg: CanonicalConfig) -> dict[str, Path]:
    dirs = {
        "tables": REPO_ROOT / cfg.results_tables_dir,
        "figures": REPO_ROOT / cfg.results_figures_dir,
        "manifests": REPO_ROOT / cfg.results_manifests_dir,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def find_first_existing(rel_paths: Iterable[str]) -> Path | None:
    for rel_path in rel_paths:
        path = REPO_ROOT / rel_path
        if path.exists():
            return path
    return None


def resolve_template_path(cfg: CanonicalConfig) -> Path:
    path = find_first_existing(cfg.template_candidates)
    if path is None:
        raise FileNotFoundError(f"No template found in candidates: {cfg.template_candidates}")
    return path


def resolve_psd_path(cfg: CanonicalConfig) -> Path:
    path = find_first_existing(cfg.psd_candidates)
    if path is None:
        raise FileNotFoundError(f"No PSD found in candidates: {cfg.psd_candidates}")
    return path


def resolve_real_trace_path(cfg: CanonicalConfig) -> Path | None:
    return find_first_existing(cfg.real_dataset_candidates)


def resolve_rq_path(cfg: CanonicalConfig) -> Path | None:
    return find_first_existing(cfg.rq_dataset_candidates)


def baseline_correct(
    traces: np.ndarray,
    pretrigger: int,
    method: str = "mean",
    return_baseline: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    traces = np.asarray(traces, dtype=np.float64)
    window = traces[..., :pretrigger]
    if method == "mean":
        baseline = np.mean(window, axis=-1, keepdims=True)
    elif method == "median":
        baseline = np.median(window, axis=-1, keepdims=True)
    else:
        raise ValueError(f"Unsupported baseline method: {method}")
    corrected = traces - baseline
    if return_baseline:
        return corrected, np.squeeze(baseline, axis=-1)
    return corrected


def normalize_template_peak(template: np.ndarray, pretrigger: int) -> np.ndarray:
    corrected = baseline_correct(np.asarray(template)[None, :], pretrigger=pretrigger)[0]
    peak = np.max(np.abs(corrected))
    if peak <= 0:
        return corrected
    return corrected / peak


def load_template(cfg: CanonicalConfig) -> tuple[np.ndarray, dict[str, Any]]:
    path = resolve_template_path(cfg)
    template = np.load(path).astype(np.float64)
    if template.ndim != 1:
        template = np.asarray(template).reshape(-1)
    if template.shape[0] != cfg.trace_len:
        raise ValueError(f"Template length {template.shape[0]} != configured trace length {cfg.trace_len}")
    template = normalize_template_peak(template, pretrigger=cfg.pretrigger)
    metadata = {
        "path": str(path.relative_to(REPO_ROOT)),
        "trace_len": int(template.shape[0]),
        "peak_abs": float(np.max(np.abs(template))),
    }
    return template, metadata


def load_psd(cfg: CanonicalConfig) -> tuple[np.ndarray, dict[str, Any]]:
    path = resolve_psd_path(cfg)
    psd = np.load(path)
    if psd.ndim == 2 and psd.shape[0] == 2:
        frequencies = psd[0].astype(np.float64)
        psd_values = psd[1].astype(np.float64)
    else:
        frequencies = np.fft.rfftfreq(cfg.trace_len, d=1.0 / cfg.sampling_frequency)
        psd_values = np.asarray(psd, dtype=np.float64)
    if psd_values.ndim != 1:
        raise ValueError(f"PSD must be 1D or stacked [freq, psd], got shape {psd.shape}")
    expected = cfg.trace_len // 2 + 1
    if psd_values.shape[0] != expected:
        raise ValueError(f"PSD length {psd_values.shape[0]} != expected rFFT bins {expected}")
    metadata = {
        "path": str(path.relative_to(REPO_ROOT)),
        "n_bins": int(psd_values.shape[0]),
        "min": float(np.min(psd_values)),
        "max": float(np.max(psd_values)),
        "has_frequency_axis": bool(psd.ndim == 2 and psd.shape[0] == 2),
    }
    return psd_values, metadata | {"frequencies": frequencies}


def deterministic_split_indices(n_samples: int, cfg: CanonicalConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n_samples)
    n_train = int(round(cfg.split_train * n_samples))
    n_val = int(round(cfg.split_val * n_samples))
    n_train = min(n_train, n_samples)
    n_val = min(n_val, n_samples - n_train)
    train = np.sort(order[:n_train])
    val = np.sort(order[n_train : n_train + n_val])
    test = np.sort(order[n_train + n_val :])
    return {"train": train, "val": val, "test": test}


def rfft_traces(traces: np.ndarray) -> np.ndarray:
    return np.fft.rfft(np.asarray(traces, dtype=np.float64), axis=-1)


def safe_sqrt_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    return np.sqrt(np.clip(weights, a_min=0.0, a_max=None))


def whiten_frequency_data(X_f: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray(X_f) * safe_sqrt_weights(weights)[None, :]


def unwhiten_basis(U_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
    sqrt_w = safe_sqrt_weights(weights)
    out = np.zeros_like(U_w, dtype=np.complex128)
    mask = sqrt_w > 0
    out[:, mask] = U_w[:, mask] / sqrt_w[mask]
    return out


def normalize_basis_weighted_unit(U: np.ndarray, weights: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=np.complex128)
    out = np.array(U, copy=True)
    if out.ndim == 1:
        out = out[None, :]
        squeeze = True
    else:
        squeeze = False
    for idx in range(out.shape[0]):
        norm = np.sqrt(np.real(weighted_inner(out[idx], out[idx], weights)))
        if norm > 0:
            out[idx] /= norm
    return out[0] if squeeze else out


def principal_angles_weighted(U_a: np.ndarray, U_b: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U_a = np.atleast_2d(np.asarray(U_a, dtype=np.complex128))
    U_b = np.atleast_2d(np.asarray(U_b, dtype=np.complex128))
    sqrt_w = safe_sqrt_weights(weights)
    A = (U_a * sqrt_w[None, :]).T
    B = (U_b * sqrt_w[None, :]).T
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    singular_values = np.linalg.svd(Qa.conj().T @ Qb, compute_uv=False)
    singular_values = np.clip(np.real(singular_values), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(np.clip(singular_values, -1.0, 1.0)))
    return singular_values, angles_deg


def exact_weighted_subspace(X_f: np.ndarray, weights: np.ndarray, k: int) -> dict[str, np.ndarray]:
    X_f = np.asarray(X_f, dtype=np.complex128)
    X_w = whiten_frequency_data(X_f, weights)
    _, _, vh = np.linalg.svd(X_w, full_matrices=False)
    U_w = vh[:k].astype(np.complex128)
    U_native = normalize_basis_weighted_unit(unwhiten_basis(U_w, weights), weights)
    return {
        "basis_whitened": U_w,
        "basis_native": U_native,
    }


def exact_isotropic_subspace(X_f: np.ndarray, k: int) -> np.ndarray:
    _, _, vh = np.linalg.svd(np.asarray(X_f, dtype=np.complex128), full_matrices=False)
    return vh[:k].astype(np.complex128)


def rankk_gls_coefficients(X_f: np.ndarray, basis_f: np.ndarray, weights: np.ndarray, return_complex: bool = True) -> np.ndarray:
    coeff = project_gls(np.asarray(X_f), np.asarray(basis_f), np.asarray(weights), return_complex=return_complex)
    return np.asarray(coeff)


def reconstruct_from_basis(coeff: np.ndarray, basis_f: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff)
    basis_f = np.asarray(basis_f)
    if basis_f.ndim == 1:
        return coeff[:, None] * basis_f[None, :]
    return coeff @ basis_f


def residual_energy_per_trace(X_f: np.ndarray, basis_f: np.ndarray, coeff: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return weighted_residual_energy(X_f, basis_f, coeff, weights)


def compute_residual_summary(
    X_f: np.ndarray,
    basis_a: np.ndarray,
    basis_b: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    coeff_a = rankk_gls_coefficients(X_f, basis_a, weights, return_complex=True)
    coeff_b = rankk_gls_coefficients(X_f, basis_b, weights, return_complex=True)
    resid_a = residual_energy_per_trace(X_f, basis_a, coeff_a, weights)
    resid_b = residual_energy_per_trace(X_f, basis_b, coeff_b, weights)
    ks = stats.ks_2samp(resid_a, resid_b)
    return {
        "residual_mean_a": float(np.mean(resid_a)),
        "residual_mean_b": float(np.mean(resid_b)),
        "relative_mean_diff": float((np.mean(resid_a) - np.mean(resid_b)) / np.mean(resid_b)),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def compute_of_amplitudes(X_time: np.ndarray, template_time: np.ndarray, psd_one_sided: np.ndarray, fs: float) -> np.ndarray:
    of = OptimumFilter(template_time, psd_one_sided, fs)
    amps = [of.fit(trace)[0] for trace in np.asarray(X_time, dtype=np.float64)]
    return np.asarray(amps, dtype=np.float64)


def compute_of_statistics(template_f: np.ndarray, weights: np.ndarray, calibration_scale: float | None = None) -> dict[str, float]:
    fisher = float(np.real(weighted_inner(template_f, template_f, weights)))
    variance = float(1.0 / fisher) if fisher > 0 else np.inf
    stats_dict = {
        "fisher_information": fisher,
        "variance_crb": variance,
        "sigma_amplitude": float(np.sqrt(variance)) if np.isfinite(variance) else np.inf,
    }
    if calibration_scale is not None:
        stats_dict["sigma_calibrated"] = float(np.sqrt(variance) * calibration_scale)
    return stats_dict


def fit_weighted_empca(
    X_f: np.ndarray,
    weights: np.ndarray,
    k: int,
    n_iter: int = 80,
    patience: int = 12,
    smoothing: bool = False,
    init: str = "random",
    template_f: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    X_f = np.asarray(X_f, dtype=np.complex128)
    weights = np.asarray(weights, dtype=np.float64)
    if seed is not None:
        np.random.seed(seed)

    solver = empca_solver(k, X_f, weights)
    init_mode = init.lower()
    if init_mode == "svd":
        solver.eigvec = exact_weighted_subspace(X_f, weights, k)["basis_native"].copy()
        solver.coeff = solver.solve_coeff()
    elif init_mode == "template" and template_f is not None and k >= 1:
        base = normalize_basis_weighted_unit(np.asarray(template_f, dtype=np.complex128), weights)[None, :]
        if k > 1:
            extra = exact_weighted_subspace(X_f, weights, k)["basis_native"][1:]
            base = np.vstack([base, extra])
        solver.eigvec = orthonormalize(base.copy())
        solver.coeff = solver.solve_coeff()
    elif init_mode != "random":
        raise ValueError(f"Unsupported init mode: {init}")

    chi2_trace: list[float] = []
    subspace_trace: list[float] = []
    best = np.inf
    stale = 0
    prev = None

    for _ in range(n_iter):
        eigvec = solver.solve_eigvec(mode="fast")
        if smoothing:
            eigvec = smooth(eigvec, window=15, polyord=3, deriv=0)
        solver.eigvec = orthonormalize(eigvec)
        solver.coeff = solver.solve_coeff()
        chi2 = float(solver.chi2())
        chi2_trace.append(chi2)

        if prev is None:
            subspace_trace.append(0.0)
        else:
            cosines, _ = principal_angles_weighted(prev, solver.eigvec, weights)
            subspace_trace.append(float(1.0 - np.min(cosines)))
        prev = solver.eigvec.copy()

        if chi2 + 1e-12 < best:
            best = chi2
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    return {
        "basis": solver.eigvec.copy(),
        "coeff": solver.coeff.copy(),
        "chi2_trace": np.asarray(chi2_trace, dtype=np.float64),
        "subspace_delta_trace": np.asarray(subspace_trace, dtype=np.float64),
        "n_iter_used": int(len(chi2_trace)),
        "smoothing": bool(smoothing),
        "init": init_mode,
    }


def weighted_rank1_alignment(empca_basis: np.ndarray, template_f: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    basis = np.asarray(empca_basis, dtype=np.complex128).reshape(-1)
    template = np.asarray(template_f, dtype=np.complex128).reshape(-1)
    aligned = phase_align_basis(basis, template, weights)
    cosine = weighted_cosine(aligned, template, weights)
    return {
        "basis_aligned": aligned,
        "weighted_cosine": float(cosine),
    }


def ks_compare(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    res = stats.ks_2samp(np.asarray(x), np.asarray(y))
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}


def audit_psd(psd_one_sided: np.ndarray, trace_len: int, floor_quantile: float = 0.01) -> dict[str, Any]:
    psd_one_sided = np.asarray(psd_one_sided, dtype=np.float64)
    floor = float(np.quantile(psd_one_sided[np.isfinite(psd_one_sided)], floor_quantile))
    clipped = np.maximum(psd_one_sided, floor)
    weights = build_of_one_sided_weights(clipped, trace_len)
    sqrt_w = safe_sqrt_weights(weights)
    return {
        "n_bins_expected": int(trace_len // 2 + 1),
        "n_bins_actual": int(psd_one_sided.shape[0]),
        "all_finite": bool(np.all(np.isfinite(psd_one_sided))),
        "all_nonnegative": bool(np.all(psd_one_sided >= 0)),
        "psd_floor_value": floor,
        "n_bins_floored": int(np.sum(psd_one_sided < floor)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "whitening_has_nan": bool(np.any(~np.isfinite(sqrt_w))),
    }


def load_trace_matrix_from_h5(path: Path, keys: Iterable[str]) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        for key in keys:
            if key in handle:
                arr = handle[key][:]
                break
        else:
            available = list(handle.keys())
            raise KeyError(f"None of {tuple(keys)} found in {path}; available keys: {available}")

    arr = np.asarray(arr, dtype=np.float64)
    metadata = {"path": str(path.relative_to(REPO_ROOT)), "key": key, "original_shape": tuple(arr.shape)}
    if arr.ndim == 3:
        arr = np.sum(arr, axis=1)
        metadata["channel_reduce"] = "sum"
    elif arr.ndim != 2:
        raise ValueError(f"Trace array must be 2D or 3D, got shape {arr.shape}")
    return arr, metadata | {"final_shape": tuple(arr.shape)}


def inspect_real_trace_candidate(path: Path, keys: Iterable[str]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT)),
        "exists": path.exists(),
        "readable": False,
        "matched_key": None,
        "error": None,
    }
    if not path.exists():
        return info
    try:
        with h5py.File(path, "r") as handle:
            available = list(handle.keys())
            info["available_keys"] = available
            for key in keys:
                if key in handle:
                    info["matched_key"] = key
                    info["shape"] = tuple(handle[key].shape)
                    info["dtype"] = str(handle[key].dtype)
                    info["readable"] = True
                    break
            if info["matched_key"] is None:
                info["error"] = f"no expected dataset key found; available keys: {available}"
    except Exception as exc:  # pragma: no cover - depends on local file state
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def resolve_usable_real_trace_path(cfg: CanonicalConfig) -> tuple[Path | None, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    for rel_path in cfg.real_dataset_candidates:
        path = REPO_ROOT / rel_path
        report = inspect_real_trace_candidate(path, cfg.real_trace_keys)
        reports.append(report)
        if report["readable"]:
            return path, reports
    return None, reports


def shift_trace_fractional(trace: np.ndarray, shift_samples: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    grid = np.arange(trace.shape[0], dtype=np.float64)
    return np.interp(grid, grid - shift_samples, trace, left=trace[0], right=trace[-1])


def broaden_trace(trace: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(trace, dtype=np.float64)
    return gaussian_filter1d(np.asarray(trace, dtype=np.float64), sigma=sigma, mode="nearest")


def build_proxy_family(template_time: np.ndarray, cfg: CanonicalConfig) -> dict[str, np.ndarray]:
    centered_template = baseline_correct(template_time[None, :], pretrigger=cfg.pretrigger)[0]
    timing_proxy = shift_trace_fractional(centered_template, +2.0) - shift_trace_fractional(centered_template, -2.0)
    width_proxy = broaden_trace(centered_template, sigma=4.0) - broaden_trace(centered_template, sigma=1.0)
    mean_proxy = centered_template / max(np.linalg.norm(centered_template), np.finfo(float).eps)
    return {
        "mean_like": mean_proxy,
        "template_like": centered_template / max(np.linalg.norm(centered_template), np.finfo(float).eps),
        "timing_like": timing_proxy / max(np.linalg.norm(timing_proxy), np.finfo(float).eps),
        "width_like": width_proxy / max(np.linalg.norm(width_proxy), np.finfo(float).eps),
    }


def sample_stationary_gaussian_from_psd(psd_one_sided: np.ndarray, n_traces: int, rng: np.random.Generator) -> np.ndarray:
    psd_one_sided = np.asarray(psd_one_sided, dtype=np.float64)
    n_bins = psd_one_sided.shape[0]
    trace_len = 2 * (n_bins - 1)
    amp = np.sqrt(np.clip(psd_one_sided, a_min=0.0, a_max=None))
    traces = np.zeros((n_traces, trace_len), dtype=np.float64)
    for idx in range(n_traces):
        spectrum = np.zeros(n_bins, dtype=np.complex128)
        spectrum[0] = amp[0]
        if n_bins > 2:
            phases = rng.uniform(0.0, 2.0 * np.pi, size=n_bins - 2)
            spectrum[1:-1] = amp[1:-1] * np.exp(1j * phases)
        if n_bins > 1:
            spectrum[-1] = amp[-1]
        traces[idx] = np.fft.irfft(spectrum, n=trace_len)
    return traces


def generate_synthetic_rank1_dataset(
    cfg: CanonicalConfig,
    n_events: int = 768,
    amplitude_range: tuple[float, float] = (0.8, 1.2),
    timing_jitter_std: float = 0.0,
    width_sigma_range: tuple[float, float] = (0.0, 0.0),
    psd_one_sided: np.ndarray | None = None,
    template_time: np.ndarray | None = None,
    return_truth: bool = False,
) -> tuple[np.ndarray, dict[str, Any]] | tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    if template_time is None:
        template_time, template_meta = load_template(cfg)
    else:
        template_meta = {"path": "provided"}
    if psd_one_sided is None:
        psd_one_sided, psd_meta = load_psd(cfg)
    else:
        psd_meta = {"path": "provided"}

    amplitudes = rng.uniform(amplitude_range[0], amplitude_range[1], size=n_events)
    timing_jitter = rng.normal(0.0, timing_jitter_std, size=n_events)
    width_sigma = rng.uniform(width_sigma_range[0], width_sigma_range[1], size=n_events)
    clean = np.zeros((n_events, cfg.trace_len), dtype=np.float64)
    for idx in range(n_events):
        pulse = shift_trace_fractional(template_time, timing_jitter[idx])
        pulse = broaden_trace(pulse, sigma=width_sigma[idx])
        pulse = baseline_correct(pulse[None, :], pretrigger=cfg.pretrigger)[0]
        clean[idx] = amplitudes[idx] * pulse
    noise = sample_stationary_gaussian_from_psd(psd_one_sided, n_events, rng=rng)
    traces = clean + noise
    metadata = {
        "source": "synthetic_rank1",
        "n_events": int(n_events),
        "amplitude_range": [float(amplitude_range[0]), float(amplitude_range[1])],
        "timing_jitter_std": float(timing_jitter_std),
        "width_sigma_range": [float(width_sigma_range[0]), float(width_sigma_range[1])],
        "template": template_meta,
        "psd": psd_meta,
    }
    truth = {
        "clean_traces": clean,
        "noise_traces": noise,
        "amplitudes": amplitudes,
        "timing_jitter": timing_jitter,
        "width_sigma": width_sigma,
    }
    if return_truth:
        return traces, metadata, truth
    return traces, metadata


def generate_rankk_signal_dataset(
    cfg: CanonicalConfig,
    n_events: int = 768,
    noise_psd: np.ndarray | None = None,
    template_time: np.ndarray | None = None,
    timing_scale: float = 0.06,
    width_scale: float = 0.08,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(cfg.seed + 11)
    if template_time is None:
        template_time, _ = load_template(cfg)
    if noise_psd is None:
        noise_psd, _ = load_psd(cfg)

    base = baseline_correct(template_time[None, :], cfg.pretrigger)[0]
    timing = shift_trace_fractional(base, +2.0) - shift_trace_fractional(base, -2.0)
    width = broaden_trace(base, sigma=4.0) - broaden_trace(base, sigma=1.0)
    clean = np.zeros((n_events, cfg.trace_len), dtype=np.float64)
    for idx in range(n_events):
        a0 = rng.uniform(0.8, 1.2)
        a1 = rng.normal(0.0, timing_scale)
        a2 = rng.normal(0.0, width_scale)
        clean[idx] = a0 * base + a1 * timing + a2 * width
    noise = sample_stationary_gaussian_from_psd(noise_psd, n_events, rng)
    return clean + noise, {
        "source": "synthetic_rankk",
        "n_events": int(n_events),
        "timing_scale": float(timing_scale),
        "width_scale": float(width_scale),
    }


def apply_structured_noise_suite(
    traces: np.ndarray,
    cfg: CanonicalConfig,
    psd_one_sided: np.ndarray,
) -> dict[str, dict[str, Any]]:
    base_cfg = {
        "noise_type": "white",
        "noise_power": float(np.mean(psd_one_sided[1:])),
        "sampling_frequency": cfg.sampling_frequency,
    }
    rng = np.random.default_rng(cfg.seed + 29)

    temporal = TemporalNoiseWrapper(
        {
            "mode": "piecewise",
            "n_segments": 4,
            "crossfade_len": 64,
            "vary_noise_power": True,
            "noise_power_scale_range": [0.7, 1.4],
            "add_drift": True,
            "drift_sigma": 0.03,
            "variance_modulation": True,
        },
        rng=rng,
    )
    artifacts = ArtifactInjector(
        {
            "sampling_frequency": cfg.sampling_frequency,
            "enable_lines": True,
            "lines": [{"freq": 0.02, "amp": [0.01, 0.04], "harmonics": [1, 2]}],
            "enable_glitches": True,
            "glitch_rate": 1.0,
            "enable_sparse_impulses": True,
            "impulse_probability": 2e-4,
            "impulse_sigma": 0.08,
        },
        rng=rng,
    )
    generator = NoiseGenerator(base_cfg, rng=rng)

    scenarios: dict[str, dict[str, Any]] = {}
    piecewise = np.vstack([temporal.apply(trace, base_generator=generator) for trace in traces])
    scenarios["piecewise_drift"] = {"traces": piecewise, "regime": "robustness-support"}
    artifacted = np.vstack([artifacts.apply(trace) for trace in traces])
    scenarios["artifact_rich"] = {"traces": artifacted, "regime": "robustness-support"}
    return scenarios


def generate_multichannel_synthetic(
    cfg: CanonicalConfig,
    n_events: int = 256,
    n_channels: int = 4,
    corr_strength: float = 0.35,
) -> dict[str, Any]:
    psd_one_sided, _ = load_psd(cfg)
    base_cfg = {
        "noise_type": "white",
        "noise_power": float(np.mean(psd_one_sided[1:])),
        "sampling_frequency": cfg.sampling_frequency,
    }
    gen = MultiChannelNoiseGenerator(
        base_cfg,
        config={"mode": "shared_private", "n_channels": n_channels, "corr_strength": corr_strength},
        seed=cfg.seed + 41,
    )
    traces = np.stack([gen.generate(cfg.trace_len, C=n_channels) for _ in range(n_events)], axis=0)
    summed = np.sum(traces, axis=1)
    return {
        "traces_multichannel": traces,
        "traces_summed": summed,
        "metadata": {
            "source": "synthetic_multichannel",
            "n_events": int(n_events),
            "n_channels": int(n_channels),
            "corr_strength": float(corr_strength),
        },
    }


def prepare_dataset(
    traces: np.ndarray,
    cfg: CanonicalConfig,
    template_time: np.ndarray | None = None,
    psd_one_sided: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    cfg.validate()
    if template_time is None:
        template_time, template_meta = load_template(cfg)
    else:
        template_meta = {"path": "provided"}
    if psd_one_sided is None:
        psd_one_sided, psd_meta = load_psd(cfg)
    else:
        psd_meta = {"path": "provided"}

    traces = np.asarray(traces, dtype=np.float64)
    if traces.ndim != 2 or traces.shape[1] != cfg.trace_len:
        raise ValueError(f"Expected traces shape (n, {cfg.trace_len}), got {traces.shape}")

    corrected, baselines = baseline_correct(
        traces,
        pretrigger=cfg.pretrigger,
        method=cfg.baseline_method,
        return_baseline=True,
    )
    freqs = rfft_traces(corrected)
    split_idx = deterministic_split_indices(corrected.shape[0], cfg)
    weights = build_of_one_sided_weights(psd_one_sided, cfg.trace_len).astype(np.float64)
    template_freq = np.fft.rfft(template_time)
    merged_metadata = {
        "n_events": int(corrected.shape[0]),
        "trace_len": int(cfg.trace_len),
        "pretrigger": int(cfg.pretrigger),
        "baseline_method": cfg.baseline_method,
        "template": template_meta,
        "psd": {k: v for k, v in psd_meta.items() if k != "frequencies"},
        "baseline_mean_mean": float(np.mean(baselines)),
        "baseline_mean_std": float(np.std(baselines)),
        "split_sizes": {k: int(v.shape[0]) for k, v in split_idx.items()},
    }
    if metadata:
        merged_metadata.update(metadata)
    return DatasetBundle(
        traces_raw=traces,
        traces_baseline=corrected,
        traces_freq=freqs,
        split_indices=split_idx,
        template_time=template_time,
        template_freq=template_freq,
        psd_one_sided=np.asarray(psd_one_sided, dtype=np.float64),
        weights_one_sided=weights,
        metadata=merged_metadata,
    )


def load_or_make_dataset(
    cfg: CanonicalConfig,
    prefer_real: bool = True,
    synthetic_events: int = 768,
    synthetic_rankk: bool = False,
    strict_real: bool = False,
) -> DatasetBundle:
    trace_reports: list[dict[str, Any]] = []
    trace_path = None
    if prefer_real:
        trace_path, trace_reports = resolve_usable_real_trace_path(cfg)
    if trace_path is not None:
        traces, trace_meta = load_trace_matrix_from_h5(trace_path, cfg.real_trace_keys)
        return prepare_dataset(
            traces,
            cfg,
            metadata={"source": "real_local", **trace_meta, "trace_candidate_reports": trace_reports},
        )

    if strict_real and prefer_real:
        raise RuntimeError(
            "No usable real trace HDF5 found. Candidate reports: "
            + json.dumps(trace_reports, indent=2, sort_keys=True)
        )

    if synthetic_rankk:
        traces, meta = generate_rankk_signal_dataset(cfg, n_events=synthetic_events)
    else:
        traces, meta = generate_synthetic_rank1_dataset(cfg, n_events=synthetic_events)
    return prepare_dataset(
        traces,
        cfg,
        metadata=meta
        | {
            "fallback_due_to_missing_real_data": True,
            "trace_candidate_reports": trace_reports,
        },
    )


def dataframe_from_claim_map() -> pd.DataFrame:
    rows = [
        (
            "one-objective hierarchy support",
            "02_unified_objective",
            "theorem-support",
            "hierarchy table completeness; preprocessing invariants frozen",
            "all required fields populated",
            "results/tables/block_01_claim_map.csv",
        ),
        (
            "OF estimator / CRB / resolution support",
            "03_optimal_filter",
            "real-support",
            "Fisher info; CRB; calibrated sigma; residual diagnostics",
            "finite fisher; variance reported; residual diagnostics saved",
            "results/tables/block_03_of_summary.json",
        ),
        (
            "rank-1 OF vs EMPCA equivalence support",
            "04_empca",
            "real-support",
            "weighted cosine; amplitude gap; residual KS statistic",
            "cosine ~ 1, low amplitude gap, non-pathological KS",
            "results/tables/block_03_rank1_comparison.csv",
        ),
        (
            "synthetic theorem verification",
            "04_empca",
            "theorem-support",
            "planted subspace recovery; amplitude bias/variance",
            "recovery improves with sample size; no systematic bias trend",
            "results/tables/block_04_synthetic_grid.csv",
        ),
        (
            "EMPCA vs linear-AE bridge support",
            "05_linear_ae",
            "theorem-support",
            "principal angles; residual gap; native-vs-whitened consistency",
            "tiny principal angles and residual mismatch",
            "results/tables/block_05_bridge_summary.json",
        ),
        (
            "convergence / initialization / rank-selection support",
            "06_convergence",
            "real-support",
            "objective traces; seed stability; held-out residual saturation",
            "stable decrease and identifiable k knee",
            "results/tables/block_06_rank_selection.csv",
        ),
        (
            "noise-aware-vs-isotropic support",
            "07_experiments",
            "real-support",
            "held-out weighted residual; chi2 proxy; rank saturation",
            "weighted objective outperforms isotropic under colored noise",
            "results/tables/block_07_ablation.csv",
        ),
        (
            "PC interpretation support",
            "08_discussion",
            "real-support",
            "weighted overlaps and coefficient correlations",
            "interpretation reported conservatively with stability notes",
            "results/tables/block_08_pc_metrics.csv",
        ),
        (
            "structured-noise robustness support",
            "07_experiments",
            "robustness-support",
            "degradation by perturbation family",
            "relative degradation summarized without theorem claims",
            "results/tables/block_09_robustness.csv",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "manuscript_claim",
            "section_label",
            "support_regime",
            "metrics",
            "acceptance_criterion",
            "artifact_path",
        ],
    )


def build_split_manifest(bundle: DatasetBundle, cfg: CanonicalConfig) -> dict[str, Any]:
    return {
        "seed": int(cfg.seed),
        "trace_len": int(cfg.trace_len),
        "pretrigger": int(cfg.pretrigger),
        "baseline_method": cfg.baseline_method,
        "split_sizes": {k: int(v.shape[0]) for k, v in bundle.split_indices.items()},
        "template_path": bundle.metadata.get("template", {}).get("path"),
        "psd_path": bundle.metadata.get("psd", {}).get("path"),
        "source": bundle.metadata.get("source"),
    }


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def manifest_row(
    block_id: str,
    regime: str,
    output_path: str,
    cfg: CanonicalConfig,
    bundle: DatasetBundle | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "block_id": block_id,
        "regime": regime,
        "output_path": output_path,
        "seed": int(cfg.seed),
        "trace_len": int(cfg.trace_len),
        "pretrigger": int(cfg.pretrigger),
    }
    if bundle is not None:
        row["data_source"] = bundle.metadata.get("source")
        row["n_events"] = int(bundle.metadata.get("n_events", bundle.traces_raw.shape[0]))
        row["template_path"] = bundle.metadata.get("template", {}).get("path")
        row["psd_path"] = bundle.metadata.get("psd", {}).get("path")
    if extra:
        row.update(extra)
    return row


def config_as_frame(cfg: CanonicalConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in asdict(cfg).items()]
    )
