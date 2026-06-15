from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    REPO_ROOT / "src" / "EMPCA",
    REPO_ROOT / "QP_simulator",
    REPO_ROOT / "implementation",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from implementation.notebook_support import (  # noqa: E402
    CanonicalConfig,
    _linear_map_from_coeff,
    _predict_from_linear_map,
    build_of_one_sided_weights,
    exact_isotropic_subspace,
    exact_weighted_subspace,
    make_clean_qp_trace,
    phase_align_basis,
    rankk_gls_coefficients,
    residual_energy_per_trace,
    stationary_noise_generator,
)
from empca_TCY_optimized import empca_solver, orthonormalize  # noqa: E402
from paper2.npml import npml_support  # noqa: E402


RESULTS_ROOT = REPO_ROOT / "NPML" / "results"
FIG_DPI = 180


@dataclass(slots=True)
class ExperimentSettings:
    seeds: tuple[int, ...] = (20260613, 20260614, 20260615)
    trace_len: int = 2048
    pretrigger: int = 256
    sampling_frequency: float = 2.5e5
    train_events: int = 160
    test_events: int = 160
    background_events: int = 240
    k_rank: int = 4
    noise_power: float = 0.05
    bootstrap_reps: int = 300


def make_cfg(seed: int, settings: ExperimentSettings) -> CanonicalConfig:
    return replace(
        CanonicalConfig(seed=int(seed)),
        trace_len=int(settings.trace_len),
        pretrigger=int(settings.pretrigger),
        sampling_frequency=float(settings.sampling_frequency),
        split_train=0.70,
        split_val=0.15,
        split_test=0.15,
        sim_events_small=int(settings.test_events),
        sim_events_medium=int(settings.train_events),
        sim_events_large=int(settings.train_events + settings.test_events),
        default_empca_iter=40,
        default_empca_patience=8,
    ).validate()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)


def save_figure(path: Path, fig) -> None:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def resample_psd(psd: np.ndarray, trace_len: int) -> np.ndarray:
    psd = np.asarray(psd, dtype=np.float64).reshape(-1)
    target = trace_len // 2 + 1
    if psd.size == target:
        out = psd.copy()
    else:
        x_old = np.linspace(0.0, 1.0, psd.size)
        x_new = np.linspace(0.0, 1.0, target)
        out = np.interp(x_new, x_old, psd)
    finite = out[np.isfinite(out) & (out > 0)]
    if finite.size == 0:
        raise ValueError("PSD contains no finite positive bins.")
    floor = max(float(np.percentile(finite, 1.0)) * 1e-3, 1e-12)
    out = np.where(np.isfinite(out) & (out > floor), out, floor)
    out[0] = out[1] if out.size > 1 else out[0]
    return out


def load_measured_mmc_psd(trace_len: int) -> np.ndarray:
    return resample_psd(np.load(REPO_ROOT / "data" / "Noise_PSD" / "noise_psd_from_MMC.npy"), trace_len)


def safe_one_sided_weights(
    psd: np.ndarray,
    trace_len: int,
    floor_fraction: float = 1e-3,
    max_weight_ratio: float = 1e6,
) -> np.ndarray:
    """Build stable one-sided precision weights from a PSD.

    The helper intentionally floors tiny PSD bins and normalizes the positive
    weights to unit mean. OF coefficients are invariant to a global weight
    scale, while normalized weights make residual plots comparable across PSDs.
    """

    psd = resample_psd(psd, trace_len)
    finite = psd[np.isfinite(psd) & (psd > 0)]
    floor = max(float(np.median(finite)) * floor_fraction, 1e-12)
    psd = np.where(np.isfinite(psd) & (psd > floor), psd, floor)
    weights = build_of_one_sided_weights(psd, trace_len)
    positive = weights[np.isfinite(weights) & (weights > 0)]
    if positive.size == 0:
        raise ValueError("PSD produced no finite positive precision weights.")
    median = float(np.median(positive))
    weights = np.clip(weights, 0.0, median * max_weight_ratio)
    positive = weights[np.isfinite(weights) & (weights > 0)]
    weights = weights / max(float(np.mean(positive)), 1e-12)
    return weights


def sample_noise_from_psd(
    psd: np.ndarray,
    trace_len: int,
    rng: np.random.Generator,
    target_std: float,
) -> np.ndarray:
    psd = resample_psd(psd, trace_len)
    n_bins = psd.size
    spectrum = np.zeros(n_bins, dtype=np.complex128)
    amp = np.sqrt(np.maximum(psd, 0.0))
    spectrum[0] = amp[0] * rng.standard_normal()
    if n_bins > 2:
        spectrum[1:-1] = amp[1:-1] * (
            rng.standard_normal(n_bins - 2) + 1j * rng.standard_normal(n_bins - 2)
        ) / np.sqrt(2.0)
    if n_bins > 1:
        spectrum[-1] = amp[-1] * rng.standard_normal()
    noise = np.fft.irfft(spectrum, n=trace_len)
    std = float(np.std(noise))
    if std > 0:
        noise = noise * (float(target_std) / std)
    return noise


def analytic_psd_and_noise(
    cfg: CanonicalConfig,
    noise_type: str,
    noise_power: float,
    n_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ng = stationary_noise_generator(cfg, noise_type=noise_type, noise_power=noise_power, rng=rng)
    _, psd = ng.build_psd(cfg.trace_len)
    noise = np.stack([ng.generate_noise(cfg.trace_len) for _ in range(n_events)], axis=0)
    return np.asarray(psd, dtype=np.float64), np.asarray(noise, dtype=np.float64)


def noise_for_condition(
    cfg: CanonicalConfig,
    condition: str,
    noise_power: float,
    n_events: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    if condition == "measured_mmc":
        psd = load_measured_mmc_psd(cfg.trace_len)
        target_std = math.sqrt(float(noise_power))
        noise = np.stack(
            [sample_noise_from_psd(psd, cfg.trace_len, rng, target_std=target_std) for _ in range(n_events)],
            axis=0,
        )
        return psd, noise, "measured_psd:data/Noise_PSD/noise_psd_from_MMC.npy"
    psd, noise = analytic_psd_and_noise(cfg, condition, noise_power, n_events, rng)
    return psd, noise, f"analytic:{condition}"


def generate_family(
    cfg: CanonicalConfig,
    n_events: int,
    *,
    seed: int,
    tau_decay_range: tuple[float, float] = (3e6, 3e6),
    t0_jitter_range: tuple[float, float] = (0.0, 0.0),
    n_qp_range: tuple[int, int] = (5000, 5000),
    noise_condition: str = "pink",
    noise_power: float = 0.05,
    clean_only: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    tau_vals = (
        np.full(n_events, tau_decay_range[0], dtype=np.float64)
        if tau_decay_range[0] == tau_decay_range[1]
        else rng.uniform(tau_decay_range[0], tau_decay_range[1], size=n_events)
    )
    t0_vals = (
        np.zeros(n_events, dtype=np.float64)
        if t0_jitter_range[0] == t0_jitter_range[1]
        else rng.uniform(t0_jitter_range[0], t0_jitter_range[1], size=n_events)
    )
    n_qp_vals = (
        np.full(n_events, n_qp_range[0], dtype=np.int64)
        if n_qp_range[0] == n_qp_range[1]
        else rng.integers(n_qp_range[0], n_qp_range[1] + 1, size=n_events)
    )
    clean = np.zeros((n_events, cfg.trace_len), dtype=np.float64)
    amp_true = np.zeros(n_events, dtype=np.float64)
    for idx in range(n_events):
        clean[idx], amp_true[idx] = make_clean_qp_trace(
            cfg,
            n_qp=int(n_qp_vals[idx]),
            tau_decay=float(tau_vals[idx]),
            t0_shift_ns=float(t0_vals[idx]),
            arrival_mode="aligned",
            rng=rng,
        )
    template, template_amp = make_clean_qp_trace(
        cfg,
        n_qp=int(np.median(n_qp_vals)),
        tau_decay=float(np.median(tau_vals)),
        t0_shift_ns=0.0,
        arrival_mode="aligned",
        rng=rng,
    )
    if clean_only:
        psd = np.ones(cfg.trace_len // 2 + 1, dtype=np.float64)
        noise = np.zeros_like(clean)
        psd_source = "none"
    else:
        psd, noise, psd_source = noise_for_condition(
            cfg,
            condition=noise_condition,
            noise_power=noise_power,
            n_events=n_events,
            rng=rng,
        )
    return {
        "x": clean + noise,
        "clean": clean,
        "noise": noise,
        "template": template,
        "template_amp": float(template_amp),
        "psd": psd,
        "weights": safe_one_sided_weights(psd, cfg.trace_len),
        "amp_true": amp_true,
        "tau_decay": tau_vals,
        "t0_shift": t0_vals,
        "n_qp": n_qp_vals,
        "noise_condition": noise_condition,
        "psd_source": psd_source,
    }


def rfft(x: np.ndarray) -> np.ndarray:
    return np.fft.rfft(np.asarray(x, dtype=np.float64), axis=-1)


def irfft(x_f: np.ndarray, trace_len: int) -> np.ndarray:
    return np.fft.irfft(x_f, n=trace_len, axis=-1)


def top_right_singular_vectors(matrix: np.ndarray, k: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    k = int(k)
    n_min = min(matrix.shape)
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= n_min - 1 or n_min <= 40:
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        return vh[:k].astype(np.complex128)
    _, singular_values, vh = svds(matrix, k=k, which="LM", return_singular_vectors=True)
    order = np.argsort(singular_values)[::-1]
    return vh[order].astype(np.complex128)


def normalize_basis_weighted(basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    basis = np.atleast_2d(np.asarray(basis, dtype=np.complex128)).copy()
    weights = np.asarray(weights, dtype=np.float64)
    for idx in range(basis.shape[0]):
        norm = np.sqrt(np.real(np.sum(np.conj(basis[idx]) * basis[idx] * weights)))
        if norm > 0:
            basis[idx] /= norm
    return basis


def fit_pca(train_x: np.ndarray, weights: np.ndarray, k: int, weighted: bool) -> np.ndarray:
    x_f = rfft(train_x)
    if not weighted:
        return top_right_singular_vectors(x_f, k=k)
    sqrt_w = np.sqrt(np.clip(np.asarray(weights, dtype=np.float64), 0.0, None))
    vh_w = top_right_singular_vectors(x_f * sqrt_w[None, :], k=k)
    basis = np.zeros_like(vh_w, dtype=np.complex128)
    mask = sqrt_w > 0
    basis[:, mask] = vh_w[:, mask] / sqrt_w[mask]
    return normalize_basis_weighted(basis, weights)


def fit_empca_basis(train_x: np.ndarray, weights: np.ndarray, k: int, seed: int) -> np.ndarray:
    x_f = rfft(train_x)
    solver = empca_solver(k, x_f, weights)
    # Use weighted-SVD initialization for stability, then run the actual EMPCA
    # alternating updates. This keeps production plots fast while avoiding the
    # previous "EMPCA == exact weighted PCA by construction" shortcut.
    solver.eigvec = fit_pca(train_x, weights, k=k, weighted=True).copy()
    solver.coeff = solver.solve_coeff()
    for _ in range(12):
        solver.eigvec = orthonormalize(solver.solve_eigvec(mode="full"))
        solver.coeff = solver.solve_coeff()
    return np.asarray(solver.eigvec, dtype=np.complex128)


def fit_named_basis(train_x: np.ndarray, weights: np.ndarray, k: int, method: str, seed: int) -> np.ndarray:
    if method == "PCA":
        return fit_pca(train_x, weights, k=k, weighted=False)
    if method == "Whitened PCA":
        return fit_pca(train_x, weights, k=k, weighted=True)
    if method == "EMPCA":
        return fit_empca_basis(train_x, weights, k=k, seed=seed)
    raise ValueError(f"Unsupported basis method: {method}")


def fit_amplitude_probe(train_x: np.ndarray, train_amp: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    coeff = rankk_gls_coefficients(rfft(train_x), basis, weights, return_complex=False)
    return _linear_map_from_coeff(np.real(coeff), train_amp)


def reduced_chi2_values(residual_energy: np.ndarray, trace_len: int) -> np.ndarray:
    return np.asarray(residual_energy, dtype=np.float64) / max(int(trace_len), 1)


def predict_basis_metrics(
    test_x: np.ndarray,
    clean_x: np.ndarray,
    amp_true: np.ndarray,
    basis: np.ndarray,
    weights: np.ndarray,
    amp_map: np.ndarray,
    trace_len: int,
) -> dict[str, float]:
    x_f = rfft(test_x)
    coeff = rankk_gls_coefficients(x_f, basis, weights, return_complex=True)
    coeff_real = np.real(coeff)
    amp_pred = _predict_from_linear_map(coeff_real, amp_map)
    resid = residual_energy_per_trace(x_f, basis, coeff, weights)
    recon_f = coeff[:, None] * basis[None, :] if np.asarray(basis).ndim == 1 else coeff @ basis
    recon_t = irfft(recon_f, trace_len)
    reduced = reduced_chi2_values(resid, trace_len)
    return {
        "amplitude_rmse": rmse(amp_pred, amp_true),
        "amplitude_bias": float(np.mean(amp_pred - amp_true)),
        "weighted_residual_mean": float(np.mean(resid)),
        "reduced_chi2_mean": float(np.mean(reduced)),
        "noise_normalized_residual_rms": float(np.sqrt(np.mean(reduced))),
        "reconstruction_mse_clean": float(np.mean((recon_t - clean_x) ** 2)),
    }


def predict_template_amplitude(
    x: np.ndarray,
    template: np.ndarray,
    template_amp: float,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    basis = rfft(template)
    coeff = rankk_gls_coefficients(rfft(x), basis, weights, return_complex=True)
    resid = residual_energy_per_trace(rfft(x), basis, coeff, weights)
    return np.real(coeff) * float(template_amp), resid


def make_template_bank(
    cfg: CanonicalConfig,
    tau_values: list[float],
    t0_values: list[float],
    n_qp: int = 5000,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    templates = []
    amps = []
    labels = []
    rng = np.random.default_rng(cfg.seed + 9001)
    for tau in tau_values:
        for t0 in t0_values:
            template, amp = make_clean_qp_trace(
                cfg,
                n_qp=n_qp,
                tau_decay=tau,
                t0_shift_ns=t0,
                arrival_mode="aligned",
                rng=rng,
            )
            templates.append(template)
            amps.append(amp)
            labels.append(f"tau={tau:.3g},t0={t0:.3g}")
    return np.asarray(templates), np.asarray(amps), labels


def predict_template_bank_amplitude(
    x: np.ndarray,
    templates: np.ndarray,
    template_amps: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_f = rfft(x)
    best_amp = np.zeros(x.shape[0], dtype=np.float64)
    best_resid = np.full(x.shape[0], np.inf, dtype=np.float64)
    for template, amp in zip(templates, template_amps):
        coeff = rankk_gls_coefficients(x_f, rfft(template), weights, return_complex=True)
        resid = residual_energy_per_trace(x_f, rfft(template), coeff, weights)
        mask = resid < best_resid
        best_resid[mask] = resid[mask]
        best_amp[mask] = np.real(coeff[mask]) * float(amp)
    return best_amp, best_resid


def first_basis_vector(basis: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis)
    return basis if basis.ndim == 1 else basis[0]


def euclidean_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    num = float(np.abs(np.vdot(a, b)))
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(num / den, 0.0, 1.0))))


def raw_time_angle_to_template_deg(basis_vec_f: np.ndarray, template_f: np.ndarray, trace_len: int) -> float:
    return euclidean_angle_deg(irfft(basis_vec_f, trace_len), irfft(template_f, trace_len))


def whitened_angle_to_template_deg(basis_vec_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray) -> float:
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    return euclidean_angle_deg(basis_vec_f * sqrt_w, template_f * sqrt_w)


def angle_to_of_filter_deg(basis_vec_f: np.ndarray, template_f: np.ndarray, weights: np.ndarray) -> float:
    # OF estimator/filter direction is proportional to C^{-1}s in frequency space.
    return euclidean_angle_deg(basis_vec_f, template_f * weights)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, reps: int = 300, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    means = np.empty(reps, dtype=np.float64)
    for idx in range(reps):
        sample = rng.choice(values, size=values.size, replace=True)
        means[idx] = np.mean(sample)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def summarize_metrics(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(12345)
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n"] = int(len(group))
        for col in value_cols:
            vals = group[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = bootstrap_ci(vals, rng)
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_sweep(
    df: pd.DataFrame,
    *,
    x: str,
    y: str,
    method_col: str = "method",
    title: str,
    ylabel: str,
    xlabel: str,
) -> Any:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    methods = list(df[method_col].dropna().unique())
    for method in methods:
        sub = df[df[method_col] == method].sort_values(x)
        mean_col = f"{y}_mean"
        lo_col = f"{y}_ci_low"
        hi_col = f"{y}_ci_high"
        ax.plot(sub[x], sub[mean_col], marker="o", linewidth=2.0, label=method)
        ax.fill_between(sub[x], sub[lo_col], sub[hi_col], alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return fig


def plot_shape_variation_residual(summary: pd.DataFrame) -> Any:
    tick_map = (
        summary[["spread_index", "tau_decay_label"]]
        .drop_duplicates()
        .sort_values("spread_index")
    )
    ticks = tick_map["spread_index"].to_numpy()
    labels = tick_map["tau_decay_label"].tolist()

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    colors = {
        "EMPCA": "#1f77b4",
        "PCA": "#ff7f0e",
        "multi-template OF": "#d62728",
        "single-template OF": "#9467bd",
    }
    markers = {"EMPCA": "o", "PCA": "s", "multi-template OF": "^", "single-template OF": "D"}

    for method in ["EMPCA", "PCA", "multi-template OF", "single-template OF"]:
        sub = summary[summary["method"] == method].sort_values("spread_index")
        if sub.empty:
            continue
        y = sub["reduced_chi2_mean_mean"].to_numpy()
        lo = sub["reduced_chi2_mean_ci_low"].to_numpy()
        hi = sub["reduced_chi2_mean_ci_high"].to_numpy()
        x = sub["spread_index"].to_numpy()
        ax.plot(
            x,
            y,
            marker=markers.get(method, "o"),
            linewidth=2.2,
            markersize=7,
            label=method,
            color=colors.get(method),
        )
        ax.fill_between(x, lo, hi, alpha=0.15, color=colors.get(method))
    ax.grid(True, alpha=0.25)
    ax.set_ylabel("mean chi^2 / N_dof")
    ax.set_title("Signal-Shape Variation: Tau-Decay Template Mismatch")
    ax.legend(frameon=False, ncol=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("template mismatch level (tau decay range, ms)")
    return fig


def plot_point_ranges(
    df: pd.DataFrame,
    *,
    category: str,
    y: str,
    hue: str,
    title: str,
    ylabel: str,
) -> Any:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    categories = list(df[category].dropna().unique())
    hues = list(df[hue].dropna().unique())
    width = 0.72 / max(len(hues), 1)
    for h_idx, h in enumerate(hues):
        sub = df[df[hue] == h]
        xs = np.arange(len(categories)) - 0.36 + width * (h_idx + 0.5)
        ys = []
        yerr_low = []
        yerr_high = []
        for cat in categories:
            row = sub[sub[category] == cat]
            if row.empty:
                ys.append(np.nan)
                yerr_low.append(0.0)
                yerr_high.append(0.0)
                continue
            row = row.iloc[0]
            mean = float(row[f"{y}_mean"])
            ys.append(mean)
            yerr_low.append(max(0.0, mean - float(row[f"{y}_ci_low"])))
            yerr_high.append(max(0.0, float(row[f"{y}_ci_high"]) - mean))
        ax.errorbar(xs, ys, yerr=[yerr_low, yerr_high], fmt="o", capsize=4, label=h)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    return fig


def experiment_dir(experiment_id: str) -> Path:
    out = ensure_dir(RESULTS_ROOT / experiment_id)
    figures = ensure_dir(out / "figures")
    for old_plot in figures.glob("*.png"):
        old_plot.unlink()
    return out


def write_manifest(out_dir: Path, experiment_id: str, settings: ExperimentSettings, extra: dict[str, Any] | None = None) -> None:
    payload = {
        "experiment_id": experiment_id,
        "settings": asdict(settings),
        "repo_root": str(REPO_ROOT),
    }
    if extra:
        payload.update(extra)
    write_json(out_dir / "manifest.json", payload)


def run_of_recovery(settings: ExperimentSettings) -> Path:
    out = experiment_dir("01_of_recovery")
    rows = []
    n_train_grid = [10, 30, 100, 300, 1000]
    n_train_grid = [n for n in n_train_grid if n <= max(settings.train_events * 4, 10)]
    for seed in settings.seeds:
        cfg = make_cfg(seed, settings)
        n_total = max(n_train_grid) + settings.test_events
        data = generate_family(
            cfg,
            n_total,
            seed=seed,
            n_qp_range=(3000, 7000),
            tau_decay_range=(3e6, 3e6),
            t0_jitter_range=(0.0, 0.0),
            noise_condition="pink",
            noise_power=settings.noise_power,
        )
        train_pool = np.arange(max(n_train_grid))
        test_idx = np.arange(max(n_train_grid), n_total)
        template_f = rfft(data["template"])
        for n_train in n_train_grid:
            train_idx = train_pool[:n_train]
            train_x = data["x"][train_idx]
            train_amp = data["amp_true"][train_idx]
            test_x = data["x"][test_idx]
            test_amp = data["amp_true"][test_idx]
            test_clean = data["clean"][test_idx]
            of_amp, of_resid = predict_template_amplitude(test_x, data["template"], data["template_amp"], data["weights"])
            rows.append(
                {
                    "seed": seed,
                    "n_train": n_train,
                    "method": "OF",
                    "raw_angle_to_template_deg": 0.0,
                    "whitened_angle_to_template_deg": 0.0,
                    "angle_to_of_filter_deg": 0.0,
                    "amplitude_rmse": rmse(of_amp, test_amp),
                    "amplitude_bias": float(np.mean(of_amp - test_amp)),
                    "weighted_residual_mean": float(np.mean(of_resid)),
                    "reduced_chi2_mean": float(np.mean(reduced_chi2_values(of_resid, cfg.trace_len))),
                    "noise_normalized_residual_rms": float(
                        np.sqrt(np.mean(reduced_chi2_values(of_resid, cfg.trace_len)))
                    ),
                    "reconstruction_mse_clean": np.nan,
                }
            )
            for method in ("PCA", "Whitened PCA", "EMPCA"):
                basis = fit_named_basis(train_x, data["weights"], k=1, method=method, seed=seed + n_train)
                basis = first_basis_vector(basis)
                basis = phase_align_basis(basis, template_f, data["weights"])
                amp_map = fit_amplitude_probe(train_x, train_amp, basis, data["weights"])
                metrics = predict_basis_metrics(
                    test_x,
                    test_clean,
                    test_amp,
                    basis,
                    data["weights"],
                    amp_map,
                    cfg.trace_len,
                )
                rows.append(
                    {
                        "seed": seed,
                        "n_train": n_train,
                        "method": method,
                        "raw_angle_to_template_deg": raw_time_angle_to_template_deg(basis, template_f, cfg.trace_len),
                        "whitened_angle_to_template_deg": whitened_angle_to_template_deg(
                            basis, template_f, data["weights"]
                        ),
                        "angle_to_of_filter_deg": angle_to_of_filter_deg(basis, template_f, data["weights"]),
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    summary = summarize_metrics(
        df,
        ["n_train", "method"],
        [
            "raw_angle_to_template_deg",
            "whitened_angle_to_template_deg",
            "angle_to_of_filter_deg",
            "amplitude_rmse",
            "reduced_chi2_mean",
        ],
    )
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    save_figure(
        out / "figures" / "of_recovery_angle_vs_ntrain.png",
        plot_metric_sweep(
            summary[summary["method"] != "OF"],
            x="n_train",
            y="whitened_angle_to_template_deg",
            title="OF Recovery: Whitened-Space Signal Direction",
            ylabel="angle to C^{-1/2} template (deg)",
            xlabel="training traces",
        ),
    )
    save_figure(
        out / "figures" / "of_recovery_angle_to_of_filter_vs_ntrain.png",
        plot_metric_sweep(
            summary[summary["method"] != "OF"],
            x="n_train",
            y="angle_to_of_filter_deg",
            title="OF Recovery: Learned Basis vs OF Filter Direction",
            ylabel="angle to C^{-1}s filter direction (deg)",
            xlabel="training traces",
        ),
    )
    save_figure(
        out / "figures" / "of_recovery_amplitude_rmse_vs_ntrain.png",
        plot_metric_sweep(
            summary,
            x="n_train",
            y="amplitude_rmse",
            title="OF Recovery: Amplitude RMSE",
            ylabel="amplitude RMSE (ADC)",
            xlabel="training traces",
        ),
    )
    write_manifest(out, "01_of_recovery", settings, {"noise_condition": "pink"})
    return out


def run_shape_variation(settings: ExperimentSettings) -> Path:
    out = experiment_dir("02_shape_variation")
    rows = []
    spreads = [
        ("rank1", "3.0", 3e6, 3e6),
        ("mild", "2.5-3.5", 2.5e6, 3.5e6),
        ("medium", "1.8-4.2", 1.8e6, 4.2e6),
        ("strong", "1.0-5.0", 1.0e6, 5.0e6),
    ]
    for seed in settings.seeds:
        cfg = make_cfg(seed, settings)
        for spread_idx, (label, tau_decay_label, tau_lo, tau_hi) in enumerate(spreads):
            train = generate_family(
                cfg,
                settings.train_events,
                seed=seed + 100 * spread_idx,
                tau_decay_range=(tau_lo, tau_hi),
                t0_jitter_range=(0.0, 0.0),
                n_qp_range=(3000, 7000),
                noise_condition="pink",
                noise_power=settings.noise_power,
            )
            test = generate_family(
                cfg,
                settings.test_events,
                seed=seed + 100 * spread_idx + 1,
                tau_decay_range=(tau_lo, tau_hi),
                t0_jitter_range=(0.0, 0.0),
                n_qp_range=(3000, 7000),
                noise_condition="pink",
                noise_power=settings.noise_power,
            )
            of_amp, of_resid = predict_template_amplitude(test["x"], train["template"], train["template_amp"], train["weights"])
            rows.append(
                {
                    "seed": seed,
                    "spread": label,
                    "spread_index": spread_idx,
                    "tau_decay_range_ns": f"{tau_lo:.0f}-{tau_hi:.0f}",
                    "tau_decay_label": tau_decay_label,
                    "method": "single-template OF",
                    "amplitude_rmse": rmse(of_amp, test["amp_true"]),
                    "amplitude_bias": float(np.mean(of_amp - test["amp_true"])),
                    "weighted_residual_mean": float(np.mean(of_resid)),
                    "reduced_chi2_mean": float(np.mean(reduced_chi2_values(of_resid, cfg.trace_len))),
                    "noise_normalized_residual_rms": float(
                        np.sqrt(np.mean(reduced_chi2_values(of_resid, cfg.trace_len)))
                    ),
                    "reconstruction_mse_clean": np.nan,
                }
            )
            tau_bank = [tau_lo, 3e6, tau_hi] if tau_lo != tau_hi else [3e6]
            t0_bank = [0.0]
            templates, amps, _ = make_template_bank(cfg, tau_bank, t0_bank)
            mt_amp, mt_resid = predict_template_bank_amplitude(test["x"], templates, amps, train["weights"])
            rows.append(
                {
                    "seed": seed,
                    "spread": label,
                    "spread_index": spread_idx,
                    "tau_decay_range_ns": f"{tau_lo:.0f}-{tau_hi:.0f}",
                    "tau_decay_label": tau_decay_label,
                    "method": "multi-template OF",
                    "amplitude_rmse": rmse(mt_amp, test["amp_true"]),
                    "amplitude_bias": float(np.mean(mt_amp - test["amp_true"])),
                    "weighted_residual_mean": float(np.mean(mt_resid)),
                    "reduced_chi2_mean": float(np.mean(reduced_chi2_values(mt_resid, cfg.trace_len))),
                    "noise_normalized_residual_rms": float(
                        np.sqrt(np.mean(reduced_chi2_values(mt_resid, cfg.trace_len)))
                    ),
                    "reconstruction_mse_clean": np.nan,
                }
            )
            for method in ("PCA", "EMPCA"):
                basis = fit_named_basis(
                    train["x"],
                    train["weights"],
                    k=settings.k_rank,
                    method=method,
                    seed=seed + spread_idx,
                )
                amp_map = fit_amplitude_probe(train["x"], train["amp_true"], basis, train["weights"])
                metrics = predict_basis_metrics(
                    test["x"],
                    test["clean"],
                    test["amp_true"],
                    basis,
                    train["weights"],
                    amp_map,
                    cfg.trace_len,
                )
                rows.append(
                    {
                        "seed": seed,
                        "spread": label,
                        "spread_index": spread_idx,
                        "tau_decay_range_ns": f"{tau_lo:.0f}-{tau_hi:.0f}",
                        "tau_decay_label": tau_decay_label,
                        "method": method,
                        **metrics,
                    }
                )
    df = pd.DataFrame(rows)
    summary = summarize_metrics(
        df,
        ["spread", "spread_index", "tau_decay_label", "method"],
        ["amplitude_rmse", "reduced_chi2_mean", "noise_normalized_residual_rms", "amplitude_bias"],
    ).sort_values(["spread_index", "method"])
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    save_figure(
        out / "figures" / "shape_variation_amplitude_rmse.png",
        plot_metric_sweep(
            summary,
            x="spread_index",
            y="amplitude_rmse",
            title="Signal-Shape Variation: Amplitude RMSE",
            ylabel="amplitude RMSE (ADC)",
            xlabel="template mismatch level",
        ),
    )
    save_figure(
        out / "figures" / "shape_variation_weighted_residual.png",
        plot_shape_variation_residual(summary),
    )
    write_manifest(
        out,
        "02_shape_variation",
        settings,
        {
            "template_mismatch_definition": (
                "Mismatch level is the event-generation tau_decay range in ms; "
                "t0 jitter is fixed to zero for this refined plot."
            ),
        },
    )
    return out


def run_covariance_ablation(settings: ExperimentSettings) -> Path:
    out = experiment_dir("03_covariance_ablation")
    rows = []
    conditions = ["white", "pink", "brownian", "measured_mmc"]
    for seed in settings.seeds:
        cfg = make_cfg(seed, settings)
        for cond_idx, condition in enumerate(conditions):
            train = generate_family(
                cfg,
                settings.train_events,
                seed=seed + cond_idx * 10,
                tau_decay_range=(1.8e6, 4.2e6),
                t0_jitter_range=(-5e4, 5e4),
                n_qp_range=(3000, 7000),
                noise_condition=condition,
                noise_power=settings.noise_power,
            )
            test = generate_family(
                cfg,
                settings.test_events,
                seed=seed + cond_idx * 10 + 1,
                tau_decay_range=(1.8e6, 4.2e6),
                t0_jitter_range=(-5e4, 5e4),
                n_qp_range=(3000, 7000),
                noise_condition=condition,
                noise_power=settings.noise_power,
            )
            pca_basis = fit_named_basis(train["x"], train["weights"], k=settings.k_rank, method="PCA", seed=seed)
            whitened_basis = fit_named_basis(
                train["x"], train["weights"], k=settings.k_rank, method="Whitened PCA", seed=seed
            )
            emp_basis = fit_named_basis(
                train["x"], train["weights"], k=settings.k_rank, method="EMPCA", seed=seed + cond_idx
            )
            cosines, angles = weighted_principal_angles(whitened_basis, pca_basis, train["weights"])
            emp_cosines, emp_angles = weighted_principal_angles(emp_basis, whitened_basis, train["weights"])
            for method, basis in (("PCA", pca_basis), ("Whitened PCA", whitened_basis), ("EMPCA", emp_basis)):
                amp_map = fit_amplitude_probe(train["x"], train["amp_true"], basis, train["weights"])
                metrics = predict_basis_metrics(
                    test["x"],
                    test["clean"],
                    test["amp_true"],
                    basis,
                    train["weights"],
                    amp_map,
                    cfg.trace_len,
                )
                rows.append(
                    {
                        "seed": seed,
                        "noise_condition": condition,
                        "method": method,
                        "mean_principal_angle_pca_whitened_deg": float(np.mean(angles)),
                        "min_principal_cosine_pca_whitened": float(np.min(cosines)),
                        "mean_principal_angle_empca_whitened_deg": float(np.mean(emp_angles)),
                        "min_principal_cosine_empca_whitened": float(np.min(emp_cosines)),
                        **metrics,
                    }
                )
            of_amp, of_resid = predict_template_amplitude(test["x"], train["template"], train["template_amp"], train["weights"])
            rows.append(
                {
                    "seed": seed,
                    "noise_condition": condition,
                    "method": "OF",
                    "mean_principal_angle_pca_whitened_deg": float(np.mean(angles)),
                    "min_principal_cosine_pca_whitened": float(np.min(cosines)),
                    "mean_principal_angle_empca_whitened_deg": float(np.mean(emp_angles)),
                    "min_principal_cosine_empca_whitened": float(np.min(emp_cosines)),
                    "amplitude_rmse": rmse(of_amp, test["amp_true"]),
                    "amplitude_bias": float(np.mean(of_amp - test["amp_true"])),
                    "weighted_residual_mean": float(np.mean(of_resid)),
                    "reduced_chi2_mean": float(np.mean(reduced_chi2_values(of_resid, cfg.trace_len))),
                    "noise_normalized_residual_rms": float(
                        np.sqrt(np.mean(reduced_chi2_values(of_resid, cfg.trace_len)))
                    ),
                    "reconstruction_mse_clean": np.nan,
                }
            )
    df = pd.DataFrame(rows)
    summary = summarize_metrics(
        df,
        ["noise_condition", "method"],
        [
            "amplitude_rmse",
            "reduced_chi2_mean",
            "mean_principal_angle_pca_whitened_deg",
            "mean_principal_angle_empca_whitened_deg",
        ],
    )
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    save_figure(
        out / "figures" / "covariance_ablation_weighted_residual.png",
        plot_point_ranges(
            summary,
            category="noise_condition",
            y="reduced_chi2_mean",
            hue="method",
            title="Noise Covariance Ablation: Reduced Weighted Residual",
            ylabel="mean chi^2 / N_dof",
        ),
    )
    save_figure(
        out / "figures" / "covariance_ablation_subspace_angle.png",
        plot_point_ranges(
            summary[summary["method"] != "OF"],
            category="noise_condition",
            y="mean_principal_angle_pca_whitened_deg",
            hue="method",
            title="Noise Covariance Ablation: Raw PCA vs Whitened Subspace",
            ylabel="weighted principal angle (deg)",
        ),
    )
    save_figure(
        out / "figures" / "covariance_ablation_amplitude_rmse.png",
        plot_point_ranges(
            summary,
            category="noise_condition",
            y="amplitude_rmse",
            hue="method",
            title="Noise Covariance Ablation: Amplitude RMSE",
            ylabel="amplitude RMSE (ADC)",
        ),
    )
    write_manifest(out, "03_covariance_ablation", settings)
    return out


def weighted_principal_angles(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.atleast_2d(np.asarray(a, dtype=np.complex128))
    b = np.atleast_2d(np.asarray(b, dtype=np.complex128))
    sqrt_w = np.sqrt(np.clip(weights, 0.0, None))
    qa, _ = np.linalg.qr((a * sqrt_w[None, :]).T)
    qb, _ = np.linalg.qr((b * sqrt_w[None, :]).T)
    s = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
    s = np.clip(np.real(s), 0.0, 1.0)
    return s, np.degrees(np.arccos(s))


def detection_score_residual_reduction(x: np.ndarray, basis: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_f = rfft(x)
    null = np.real(np.sum((np.abs(x_f) ** 2) * weights[None, :], axis=1))
    coeff = rankk_gls_coefficients(x_f, basis, weights, return_complex=True)
    resid = residual_energy_per_trace(x_f, basis, coeff, weights)
    return null - resid


def efficiency_curve_for_condition(
    settings: ExperimentSettings,
    seed: int,
    noise_condition: str,
    energy_grid: list[int],
    fpr: float = 0.01,
) -> list[dict[str, Any]]:
    cfg = make_cfg(seed, settings)
    train = generate_family(
        cfg,
        settings.train_events,
        seed=seed + 710,
        tau_decay_range=(1.8e6, 4.2e6),
        t0_jitter_range=(-5e4, 5e4),
        n_qp_range=(3000, 7000),
        noise_condition=noise_condition,
        noise_power=settings.noise_power,
    )
    bg = generate_family(
        cfg,
        settings.background_events,
        seed=seed + 711,
        n_qp_range=(0, 0),
        noise_condition=noise_condition,
        noise_power=settings.noise_power,
    )
    pca = fit_named_basis(train["x"], train["weights"], k=settings.k_rank, method="PCA", seed=seed)
    empca = fit_named_basis(train["x"], train["weights"], k=settings.k_rank, method="EMPCA", seed=seed + 17)
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "OF": lambda x: predict_template_amplitude(x, train["template"], train["template_amp"], train["weights"])[0],
        "PCA residual reduction": lambda x: detection_score_residual_reduction(x, pca, train["weights"]),
        "EMPCA residual reduction": lambda x: detection_score_residual_reduction(x, empca, train["weights"]),
    }
    rows = []
    thresholds = {}
    for method, scorer in methods.items():
        bg_score = scorer(bg["x"])
        thresholds[method] = float(np.quantile(bg_score, 1.0 - fpr))
    for n_qp in energy_grid:
        signal = generate_family(
            cfg,
            settings.test_events,
            seed=seed + 800 + n_qp,
            tau_decay_range=(1.8e6, 4.2e6),
            t0_jitter_range=(-5e4, 5e4),
            n_qp_range=(n_qp, n_qp),
            noise_condition=noise_condition,
            noise_power=settings.noise_power,
        )
        for method, scorer in methods.items():
            score = scorer(signal["x"])
            passed = score >= thresholds[method]
            rows.append(
                {
                    "seed": seed,
                    "noise_condition": noise_condition,
                    "method": method,
                    "fpr": fpr,
                    "n_qp": n_qp,
                    "injected_amplitude_mean": float(np.mean(signal["amp_true"])),
                    "threshold": thresholds[method],
                    "efficiency": float(np.mean(passed)),
                    "n_signal": int(passed.size),
                }
            )
    return rows


def run_trigger_efficiency(settings: ExperimentSettings, measured: bool = False) -> Path:
    experiment_id = "05_measured_psd_trigger" if measured else "04_trigger_efficiency"
    out = experiment_dir(experiment_id)
    rows = []
    conditions = ["pink", "measured_mmc"] if measured else ["pink"]
    energy_grid = [5, 10, 20, 50, 100, 200, 500, 1000]
    fprs = [0.001, 0.01, 0.05]
    for seed in settings.seeds:
        for condition in conditions:
            for fpr in fprs:
                rows.extend(efficiency_curve_for_condition(settings, seed, condition, energy_grid, fpr=fpr))
    df = pd.DataFrame(rows)
    summary = summarize_metrics(
        df,
        ["noise_condition", "fpr", "method", "n_qp"],
        ["efficiency"],
    ).sort_values(["noise_condition", "fpr", "method", "n_qp"])
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    for condition in conditions:
        for fpr in fprs:
            sub = summary[(summary["noise_condition"] == condition) & (summary["fpr"] == fpr)]
            label = f"{100.0 * fpr:g}%"
            fig = plot_metric_sweep(
                sub,
                x="n_qp",
                y="efficiency",
                title=f"Trigger Efficiency at {label} FPR ({condition})",
                ylabel="trigger efficiency",
                xlabel="injected QP count",
            )
            fig.axes[0].set_xscale("log")
            save_figure(
                out / "figures" / f"trigger_efficiency_{condition}_fpr_{str(fpr).replace('.', 'p')}.png",
                fig,
            )
    write_manifest(out, experiment_id, settings, {"noise_conditions": conditions, "fprs": fprs, "energy_grid_n_qp": energy_grid})
    return out


def run_sample_efficiency(settings: ExperimentSettings) -> Path:
    out = experiment_dir("06_sample_efficiency")
    rows = []
    grid = [10, 30, 100, 300, 1000]
    grid = [n for n in grid if n <= max(settings.train_events * 4, 10)]
    for seed in settings.seeds:
        cfg = make_cfg(seed, settings)
        n_total = max(grid) + settings.test_events
        data = generate_family(
            cfg,
            n_total,
            seed=seed + 1230,
            tau_decay_range=(1.8e6, 4.2e6),
            t0_jitter_range=(-5e4, 5e4),
            n_qp_range=(3000, 7000),
            noise_condition="pink",
            noise_power=settings.noise_power,
        )
        test_idx = np.arange(max(grid), n_total)
        train_pool = np.arange(max(grid))
        of_amp, of_resid = predict_template_amplitude(data["x"][test_idx], data["template"], data["template_amp"], data["weights"])
        reference_basis = fit_named_basis(
            data["x"][train_pool],
            data["weights"],
            k=settings.k_rank,
            method="Whitened PCA",
            seed=seed + 999,
        )
        for n_train in grid:
            rows.append(
                {
                    "seed": seed,
                    "n_train": n_train,
                    "method": "OF",
                    "amplitude_rmse": rmse(of_amp, data["amp_true"][test_idx]),
                    "weighted_residual_mean": float(np.mean(of_resid)),
                    "reduced_chi2_mean": float(np.mean(reduced_chi2_values(of_resid, cfg.trace_len))),
                    "mean_angle_to_reference_deg": 0.0,
                }
            )
            train_idx = train_pool[:n_train]
            for method in ("PCA", "Whitened PCA", "EMPCA"):
                basis = fit_named_basis(
                    data["x"][train_idx],
                    data["weights"],
                    k=settings.k_rank,
                    method=method,
                    seed=seed + n_train,
                )
                amp_map = fit_amplitude_probe(data["x"][train_idx], data["amp_true"][train_idx], basis, data["weights"])
                metrics = predict_basis_metrics(
                    data["x"][test_idx],
                    data["clean"][test_idx],
                    data["amp_true"][test_idx],
                    basis,
                    data["weights"],
                    amp_map,
                    cfg.trace_len,
                )
                rows.append(
                    {
                        "seed": seed,
                        "n_train": n_train,
                        "method": method,
                        "amplitude_rmse": metrics["amplitude_rmse"],
                        "weighted_residual_mean": metrics["weighted_residual_mean"],
                        "reduced_chi2_mean": metrics["reduced_chi2_mean"],
                        "mean_angle_to_reference_deg": float(
                            np.mean(weighted_principal_angles(basis, reference_basis, data["weights"])[1])
                        ),
                    }
                )
    df = pd.DataFrame(rows)
    summary = summarize_metrics(
        df,
        ["n_train", "method"],
        ["amplitude_rmse", "reduced_chi2_mean", "mean_angle_to_reference_deg"],
    )
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    fig = plot_metric_sweep(
        summary,
        x="n_train",
        y="amplitude_rmse",
        title="Calibration Sample Efficiency",
        ylabel="amplitude RMSE (ADC)",
        xlabel="training traces",
    )
    fig.axes[0].set_xscale("log")
    save_figure(out / "figures" / "sample_efficiency_amplitude_rmse.png", fig)
    fig = plot_metric_sweep(
        summary[summary["method"] != "OF"],
        x="n_train",
        y="mean_angle_to_reference_deg",
        title="Calibration Sample Efficiency: Subspace Stability",
        ylabel="angle to full-calibration weighted subspace (deg)",
        xlabel="training traces",
    )
    fig.axes[0].set_xscale("log")
    save_figure(out / "figures" / "sample_efficiency_subspace_stability.png", fig)
    write_manifest(
        out,
        "06_sample_efficiency",
        settings,
        {
            "neural_runs": "Use scripts/run_paper2_training_suite.py --suite best with copied configs for AE/transformer sample sweeps.",
        },
    )
    return out


def run_coverage_ablation(settings: ExperimentSettings) -> Path:
    out = experiment_dir("07_coverage_ablation")
    rows = []
    for seed in settings.seeds:
        result = npml_support.run_coverage_ablation(seed=seed, k=settings.k_rank)
        frame = result["coverage_df"].copy()
        frame["seed"] = seed
        rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    summary = summarize_metrics(
        df,
        ["training_condition", "method"],
        ["weighted_residual_mean", "amplitude_rmse", "timing_rmse", "position_rmse", "shape_rmse"],
    )
    save_frame(out / "metrics_by_seed.csv", df)
    save_frame(out / "metrics_summary.csv", summary)
    save_figure(
        out / "figures" / "coverage_ablation_weighted_residual.png",
        plot_point_ranges(
            summary,
            category="training_condition",
            y="weighted_residual_mean",
            hue="method",
            title="Coverage Ablation: Weighted Residual",
            ylabel="mean weighted residual",
        ),
    )
    save_figure(
        out / "figures" / "coverage_ablation_position_rmse.png",
        plot_point_ranges(
            summary,
            category="training_condition",
            y="position_rmse",
            hue="method",
            title="Coverage Ablation: Position-Proxy RMSE",
            ylabel="position-proxy RMSE",
        ),
    )
    write_manifest(out, "07_coverage_ablation", settings, {"source": "paper2.npml.npml_support.run_coverage_ablation"})
    return out


def run_nfpa_architecture(settings: ExperimentSettings) -> Path:
    out = experiment_dir("08_nfpa_architecture")
    rows = []
    for seed in settings.seeds:
        result = npml_support.run_nfpa_regime_sweep(seed=seed)
        frame = result["nfpa_df"].copy()
        frame["seed"] = seed
        rows.append(frame)
    df = pd.concat(rows, ignore_index=True)
    long_rows = []
    for row in df.itertuples(index=False):
        empca_residual = float(row.empca_weighted_residual)
        for method in ("nfpa", "empca"):
            weighted_residual = float(getattr(row, f"{method}_weighted_residual"))
            long_rows.append(
                {
                    "seed": row.seed,
                    "regime": row.regime,
                    "distortion_level": row.distortion_level,
                    "method": method.upper(),
                    "weighted_residual": weighted_residual,
                    "weighted_residual_ratio_to_empca": weighted_residual / max(empca_residual, 1e-12),
                    "reconstruction_mse": getattr(row, f"{method}_reconstruction_mse"),
                    "mean_principal_angle_deg": row.mean_principal_angle_deg,
                }
            )
    long_df = pd.DataFrame(long_rows)
    summary = summarize_metrics(
        long_df,
        ["regime", "distortion_level", "method"],
        ["weighted_residual", "weighted_residual_ratio_to_empca", "reconstruction_mse", "mean_principal_angle_deg"],
    ).sort_values(["distortion_level", "method"])
    save_frame(out / "metrics_by_seed.csv", long_df)
    save_frame(out / "metrics_summary.csv", summary)
    save_figure(
        out / "figures" / "nfpa_weighted_residual_vs_distortion.png",
        plot_metric_sweep(
            summary,
            x="distortion_level",
            y="weighted_residual_ratio_to_empca",
            title="NFPA vs EMPCA: Weighted Residual Ratio",
            ylabel="weighted residual / EMPCA residual",
            xlabel="non-separability distortion",
        ),
    )
    save_figure(
        out / "figures" / "nfpa_reconstruction_mse_vs_distortion.png",
        plot_metric_sweep(
            summary,
            x="distortion_level",
            y="reconstruction_mse",
            title="NFPA vs EMPCA: Native Reconstruction MSE",
            ylabel="native reconstruction MSE",
            xlabel="non-separability distortion",
        ),
    )
    write_manifest(out, "08_nfpa_architecture", settings, {"source": "paper2.npml.npml_support.run_nfpa_regime_sweep"})
    return out


def run_paper2_dryrun(settings: ExperimentSettings) -> Path:
    out = experiment_dir("09_paper2_neural_hooks")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_paper2_training_suite.py"),
        "--suite",
        "all",
        "--dry-run",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (out / "paper2_training_suite_dry_run.txt").write_text(proc.stdout, encoding="utf-8")
    write_manifest(
        out,
        "09_paper2_neural_hooks",
        settings,
        {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "purpose": "Documents the existing AE/transformer/NFPA training hooks. Full neural training is intentionally not run by the NPML smoke suite.",
        },
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Paper2 dry run failed; see {out / 'paper2_training_suite_dry_run.txt'}")
    return out


EXPERIMENTS: dict[str, Callable[[ExperimentSettings], Path]] = {
    "01": run_of_recovery,
    "02": run_shape_variation,
    "03": run_covariance_ablation,
    "04": lambda settings: run_trigger_efficiency(settings, measured=False),
    "05": lambda settings: run_trigger_efficiency(settings, measured=True),
    "06": run_sample_efficiency,
    "07": run_coverage_ablation,
    "08": run_nfpa_architecture,
    "09": run_paper2_dryrun,
}


def parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NPML experiment suite.")
    parser.add_argument(
        "--experiments",
        default="01,02,03,04,05,06,07,08,09",
        help="Comma-separated experiment IDs, e.g. 01,03,04. Use 'all' for all.",
    )
    parser.add_argument("--seeds", default="20260613,20260614,20260615")
    parser.add_argument("--trace-len", type=int, default=2048)
    parser.add_argument("--pretrigger", type=int, default=256)
    parser.add_argument("--train-events", type=int, default=160)
    parser.add_argument("--test-events", type=int, default=160)
    parser.add_argument("--background-events", type=int, default=240)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--noise-power", type=float, default=0.05)
    parser.add_argument("--bootstrap-reps", type=int, default=300)
    return parser


def settings_from_args(args: argparse.Namespace) -> ExperimentSettings:
    return ExperimentSettings(
        seeds=parse_seeds(args.seeds),
        trace_len=args.trace_len,
        pretrigger=args.pretrigger,
        train_events=args.train_events,
        test_events=args.test_events,
        background_events=args.background_events,
        k_rank=args.rank,
        noise_power=args.noise_power,
        bootstrap_reps=args.bootstrap_reps,
    )


def selected_experiments(value: str) -> list[str]:
    if value.lower() == "all":
        return sorted(EXPERIMENTS)
    ids = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in ids if item not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {unknown}. Known IDs: {sorted(EXPERIMENTS)}")
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = settings_from_args(args)
    ensure_dir(RESULTS_ROOT)
    outputs = []
    for exp_id in selected_experiments(args.experiments):
        print(f"[NPML] start experiment {exp_id}", flush=True)
        out = EXPERIMENTS[exp_id](settings)
        outputs.append({"experiment": exp_id, "output_dir": str(out.relative_to(REPO_ROOT))})
        print(f"[NPML] done experiment {exp_id}: {out.relative_to(REPO_ROOT)}", flush=True)
    write_json(
        RESULTS_ROOT / "suite_manifest.json",
        {
            "settings": asdict(settings),
            "outputs": outputs,
        },
    )
    print(f"[NPML] suite manifest: {RESULTS_ROOT / 'suite_manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
