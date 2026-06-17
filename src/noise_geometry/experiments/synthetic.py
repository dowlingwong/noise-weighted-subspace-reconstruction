"""Synthetic Paper 1 validation experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..autoencoders import tied_linear_ae_closed_form
from ..filters import gls_amplitude, psd_amplitude_variance
from ..metrics import amplitude_resolution, gaussian_nll, mse, weighted_residual
from ..noise import (
    block_covariance,
    generate_colored_noise,
    inverse_covariance,
    inverse_psd_weights,
    make_powerlaw_psd,
    regularize_covariance,
    whiten_with_covariance,
)
from ..subspace import fit_pca, fit_weighted_pca, principal_angles, project_onto_basis
from ..synthetic import make_rank1_pulse_dataset, run_of_empca_equivalence
from ..synthetic.pulses import exponential_pulse


def _rankk_signal(
    rng: np.random.Generator,
    n_traces: int,
    n_features: int,
    rank: int,
    *,
    jitter: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_features, endpoint=False)
    modes = []
    base = np.exp(-((t - 0.35) ** 2) / 0.004)
    base /= np.linalg.norm(base)
    modes.append(base)
    if rank > 1:
        derivative = np.gradient(base)
        derivative /= np.linalg.norm(derivative)
        modes.append(derivative)
    while len(modes) < rank:
        freq = 5 + 3 * len(modes)
        mode = np.sin(2.0 * np.pi * freq * t)
        mode -= sum(np.dot(mode, m) * m for m in modes)
        mode /= np.linalg.norm(mode)
        modes.append(mode)
    basis = np.vstack(modes[:rank])
    coeff = rng.normal(size=(n_traces, rank))
    if jitter and rank > 1:
        coeff[:, 1] *= 0.25
    clean = coeff @ basis
    return clean, basis


def _time_cov_from_psd(psd: np.ndarray, n_features: int, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = generate_colored_noise(rng, psd, n_features, 1.0, n_draws)
    return (noise.T @ noise) / max(n_draws - 1, 1)


def _ar1_cov(n_features: int, rho: float, scale: float = 1.0) -> np.ndarray:
    idx = np.arange(n_features)
    return float(scale) * float(rho) ** np.abs(idx[:, None] - idx[None, :])


def _gls_time_amplitude(samples: np.ndarray, template: np.ndarray, metric: np.ndarray) -> np.ndarray:
    denom = float(template @ metric @ template)
    return (samples @ metric @ template) / denom


def _metric_reversal_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed = int(config.get("seed", 19))
    n_traces = int(config.get("n_traces", 512))
    n_features = int(config.get("n_features", config.get("n_samples", 256)))
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_features, endpoint=False)
    signal_basis = np.vstack(
        [
            np.sin(2.0 * np.pi * 9.0 * t),
            np.cos(2.0 * np.pi * 13.0 * t),
        ]
    )
    signal_basis /= np.linalg.norm(signal_basis, axis=1, keepdims=True)
    clean = rng.normal(size=(n_traces, 2)) @ signal_basis

    freqs = np.fft.rfftfreq(n_features)
    psd = np.ones_like(freqs)
    psd[1:] = 1.0 / freqs[1:] ** 2
    psd[0] = psd[1]
    weights = inverse_psd_weights(psd, n_features)
    noise = 0.25 * generate_colored_noise(rng, psd, n_features, 1.0, n_traces)
    return clean + noise, clean, weights


def run_s0_smoke(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 0, "n_traces": 96, "n_samples": 256, "noise_kind": "white"} | config
    dataset = make_rank1_pulse_dataset(
        n_traces=int(cfg["n_traces"]),
        n_samples=int(cfg["n_samples"]),
        noise_kind=str(cfg["noise_kind"]),
        seed=int(cfg["seed"]),
    )
    summary = run_of_empca_equivalence(dataset)
    summary["experiment"] = "S0"
    summary["status"] = "smoke_passed"
    return summary


def run_s1_of_crb(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 1, "n_traces": 2048, "n_samples": 512, "noise_kind": "pink"} | config
    dataset = make_rank1_pulse_dataset(
        n_traces=int(cfg["n_traces"]),
        n_samples=int(cfg["n_samples"]),
        noise_kind=str(cfg["noise_kind"]),
        seed=int(cfg["seed"]),
    )
    X_f = np.fft.rfft(dataset.traces, axis=1)
    s_f = np.fft.rfft(dataset.template)
    amp = gls_amplitude(X_f, s_f, dataset.weights)
    crb_sigma = float(
        np.sqrt(
            psd_amplitude_variance(
                s_f,
                dataset.psd,
                dataset.weights,
                trace_len=dataset.traces.shape[1],
                sampling_frequency=dataset.sampling_frequency,
            )
        )
    )
    res = amplitude_resolution(amp, dataset.amplitudes)
    res.update(
        {
            "experiment": "S1",
            "noise_kind": str(cfg["noise_kind"]),
            "crb_sigma": crb_sigma,
            "sigma_over_crb": float(res["std"] / crb_sigma),
            "n_traces": int(cfg["n_traces"]),
        }
    )
    return res


def run_s2_of_empca(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 7, "n_traces": 256, "n_samples": 1024, "noise_kind": "pink"} | config
    dataset = make_rank1_pulse_dataset(
        n_traces=int(cfg["n_traces"]),
        n_samples=int(cfg["n_samples"]),
        noise_kind=str(cfg["noise_kind"]),
        seed=int(cfg["seed"]),
    )
    summary = run_of_empca_equivalence(dataset)
    summary.update({"experiment": "S2", "noise_kind": str(cfg["noise_kind"])})
    return summary


def run_s3_ae_bridge(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 3, "n_traces": 512, "n_features": 128, "rank": 3} | config
    rng = np.random.default_rng(int(cfg["seed"]))
    clean, _ = _rankk_signal(rng, int(cfg["n_traces"]), int(cfg["n_features"]), int(cfg["rank"]))
    variance = np.geomspace(0.02, 0.5, int(cfg["n_features"]))
    weights = 1.0 / variance
    noise = rng.normal(scale=np.sqrt(variance)[None, :], size=clean.shape)
    X = clean + noise
    empca = fit_weighted_pca(X, weights, int(cfg["rank"]))
    ae = tied_linear_ae_closed_form(X, int(cfg["rank"]), weights=weights)
    angles = principal_angles(empca.components, ae.components, weights=weights)
    empca_recon = project_onto_basis(X, empca.components, weights=weights, mean=empca.mean)
    ae_recon = project_onto_basis(X, ae.components, weights=weights, mean=ae.mean)
    return {
        "experiment": "S3",
        "max_principal_angle_deg": float(np.max(angles)),
        "mean_principal_angle_deg": float(np.mean(angles)),
        "reconstruction_rmse": float(np.sqrt(mse(empca_recon, ae_recon))),
    }


def run_s4_white_control(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 4, "n_traces": 512, "n_features": 128, "rank": 3, "noise_level": 0.05} | config
    rng = np.random.default_rng(int(cfg["seed"]))
    clean, _ = _rankk_signal(rng, int(cfg["n_traces"]), int(cfg["n_features"]), int(cfg["rank"]))
    X = clean + rng.normal(scale=float(cfg["noise_level"]), size=clean.shape)
    weights = np.ones(X.shape[1], dtype=np.float64)
    pca = fit_pca(X, int(cfg["rank"]))
    empca = fit_weighted_pca(X, weights, int(cfg["rank"]))
    pca_recon = project_onto_basis(X, pca.components, mean=pca.mean)
    empca_recon = project_onto_basis(X, empca.components, weights=weights, mean=empca.mean)
    angles = principal_angles(pca.components, empca.components)
    return {
        "experiment": "S4",
        "pca_mse": float(mse(X, pca_recon)),
        "empca_mse": float(mse(X, empca_recon)),
        "pca_weighted_residual": float(np.mean(weighted_residual(X, pca_recon, weights))),
        "empca_weighted_residual": float(np.mean(weighted_residual(X, empca_recon, weights))),
        "max_principal_angle_deg": float(np.max(angles)),
    }


def run_s5_metric_reversal(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 19, "n_traces": 512, "n_features": 256, "rank": 2} | config
    X, clean, weights_f = _metric_reversal_dataset(cfg)
    X_f = np.fft.rfft(X, axis=1).real
    clean_f = np.fft.rfft(clean, axis=1).real
    rank = int(cfg["rank"])
    pca = fit_pca(X_f, rank)
    wpca = fit_weighted_pca(X_f, weights_f, rank)
    pca_recon = project_onto_basis(X_f, pca.components, mean=pca.mean)
    wpca_recon = project_onto_basis(X_f, wpca.components, weights=weights_f, mean=wpca.mean)
    pca_wr = weighted_residual(X_f, pca_recon, weights_f)
    wpca_wr = weighted_residual(X_f, wpca_recon, weights_f)
    return {
        "experiment": "S5",
        "rank": rank,
        "pca_raw_residual_to_observed": float(mse(X_f, pca_recon)),
        "weighted_pca_raw_residual_to_observed": float(mse(X_f, wpca_recon)),
        "pca_weighted_residual_to_observed": float(np.mean(pca_wr)),
        "weighted_pca_weighted_residual_to_observed": float(np.mean(wpca_wr)),
        "pca_nll_mean": float(np.mean(gaussian_nll(X_f, pca_recon, weights_f))),
        "weighted_pca_nll_mean": float(np.mean(gaussian_nll(X_f, wpca_recon, weights_f))),
        "pca_clean_mse_diagnostic": float(mse(clean_f, pca_recon)),
        "weighted_pca_clean_mse_diagnostic": float(mse(clean_f, wpca_recon)),
        "subspace_angle_deg_max": float(np.max(principal_angles(pca.components, wpca.components, weights=weights_f))),
    }


def run_s6_timing_rank_sweep(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 6, "n_traces": 400, "n_features": 128, "rank_max": 6, "timing_jitter_samples": 3.0} | config
    rng = np.random.default_rng(int(cfg["seed"]))
    n_traces = int(cfg["n_traces"])
    n_features = int(cfg["n_features"])
    template = exponential_pulse(n_features, float(n_features), tau_rise=0.03, tau_decay=0.16)
    amplitudes = rng.uniform(0.5, 1.5, size=n_traces)
    shifts = rng.normal(scale=float(cfg["timing_jitter_samples"]), size=n_traces)
    grid = np.arange(n_features)
    clean = np.vstack(
        [amp * np.interp(grid - shift, grid, template, left=0.0, right=0.0) for amp, shift in zip(amplitudes, shifts)]
    )
    cov = _ar1_cov(n_features, rho=0.85, scale=0.01)
    metric = inverse_covariance(cov)
    X = clean + rng.multivariate_normal(np.zeros(n_features), cov, size=n_traces)

    amp_of = _gls_time_amplitude(X, template, metric)
    of_recon = amp_of[:, None] * template[None, :]
    ranks = list(range(1, int(cfg["rank_max"]) + 1))
    residuals = []
    clean_errors = []
    for rank in ranks:
        fit = fit_weighted_pca(X, metric, rank)
        recon = project_onto_basis(X, fit.components, weights=metric, mean=fit.mean)
        residuals.append(float(np.mean(weighted_residual(X, recon, metric))))
        clean_errors.append(float(mse(clean, recon)))
    return {
        "experiment": "S6",
        "ranks": ranks,
        "weighted_residual_by_rank": residuals,
        "clean_mse_by_rank": clean_errors,
        "of_weighted_residual": float(np.mean(weighted_residual(X, of_recon, metric))),
        "of_amplitude_bias": float(np.mean(amp_of - amplitudes)),
        "best_rank_by_clean_mse": int(ranks[int(np.argmin(clean_errors))]),
    }


def run_s7_covariance_robustness(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "seed": 7,
        "n_features": 64,
        "n_eval_traces": 1200,
        "n_noise_traces": [10, 30, 100, 300, 1000],
        "shrinkage": 0.05,
    } | config
    rng = np.random.default_rng(int(cfg["seed"]))
    n_features = int(cfg["n_features"])
    template = exponential_pulse(n_features, float(n_features), tau_rise=0.03, tau_decay=0.16)
    true_cov = _ar1_cov(n_features, rho=0.9, scale=0.02)
    true_metric = inverse_covariance(true_cov)
    amplitudes = rng.uniform(0.5, 1.5, size=int(cfg["n_eval_traces"]))
    eval_noise = rng.multivariate_normal(np.zeros(n_features), true_cov, size=amplitudes.size)
    X = amplitudes[:, None] * template[None, :] + eval_noise

    oracle_amp = _gls_time_amplitude(X, template, true_metric)
    oracle_sigma = float(np.std(oracle_amp - amplitudes, ddof=1))
    sample_counts = [int(v) for v in cfg["n_noise_traces"]]
    sigma_ratio = []
    covariance_error = []
    for count in sample_counts:
        calibration = rng.multivariate_normal(np.zeros(n_features), true_cov, size=count)
        estimated = regularize_covariance(
            np.cov(calibration, rowvar=False),
            shrinkage=float(cfg["shrinkage"]),
        )
        metric = inverse_covariance(estimated)
        estimate = _gls_time_amplitude(X, template, metric)
        sigma_ratio.append(float(np.std(estimate - amplitudes, ddof=1) / oracle_sigma))
        covariance_error.append(float(np.linalg.norm(estimated - true_cov) / np.linalg.norm(true_cov)))
    identity_amp = _gls_time_amplitude(X, template, np.eye(n_features))
    return {
        "experiment": "S7",
        "n_noise_traces": sample_counts,
        "sigma_over_oracle": sigma_ratio,
        "relative_covariance_error": covariance_error,
        "oracle_sigma": oracle_sigma,
        "identity_sigma_over_oracle": float(np.std(identity_amp - amplitudes, ddof=1) / oracle_sigma),
    }


def run_s8_residual_calibration(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {"seed": 8, "n_traces": 512, "n_features": 128} | config
    rng = np.random.default_rng(int(cfg["seed"]))
    n_features = int(cfg["n_features"])
    template = exponential_pulse(n_features, float(n_features), tau_rise=0.03, tau_decay=0.16)
    cov = _ar1_cov(n_features, rho=0.8, scale=0.02)
    metric = inverse_covariance(cov)
    amplitudes = rng.uniform(0.5, 1.5, size=int(cfg["n_traces"]))
    noise = rng.multivariate_normal(np.zeros(n_features), cov, size=amplitudes.size)
    X = amplitudes[:, None] * template[None, :] + noise
    amp = _gls_time_amplitude(X, template, metric)
    residuals = X - amp[:, None] * template[None, :]
    whitened = whiten_with_covariance(residuals, cov)
    whitened_cov = np.cov(whitened, rowvar=False)
    chi2_dof = weighted_residual(X, amp[:, None] * template[None, :], metric, normalize=False) / (n_features - 1)
    lag1 = float(np.mean(np.sum(whitened[:, :-1] * whitened[:, 1:], axis=1) / np.sum(whitened**2, axis=1)))
    return {
        "experiment": "S8",
        "mean_chi2_per_dof": float(np.mean(chi2_dof)),
        "std_chi2_per_dof": float(np.std(chi2_dof, ddof=1)),
        "whitened_covariance_relative_error": float(
            np.linalg.norm(whitened_cov - np.eye(n_features)) / np.linalg.norm(np.eye(n_features))
        ),
        "whitened_lag1_autocorrelation": lag1,
    }


def run_s9_multichannel(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "seed": 9,
        "n_channels": 3,
        "n_features": 32,
        "n_traces": 1200,
        "correlation_strength": 0.4,
    } | config
    rng = np.random.default_rng(int(cfg["seed"]))
    channels = int(cfg["n_channels"])
    n_features = int(cfg["n_features"])
    rho = float(cfg["correlation_strength"])
    channel_cov = np.full((channels, channels), rho)
    np.fill_diagonal(channel_cov, 1.0)
    time_cov = _ar1_cov(n_features, rho=0.65, scale=0.02)
    true_cov = block_covariance(channel_cov, time_cov)
    diagonal_cov = block_covariance(np.eye(channels), time_cov)
    true_metric = inverse_covariance(true_cov)
    diagonal_metric = inverse_covariance(diagonal_cov)

    pulse = exponential_pulse(n_features, float(n_features), tau_rise=0.03, tau_decay=0.16)
    gains = np.linspace(1.0, 0.6, channels)
    template = np.concatenate([gain * pulse for gain in gains])
    amplitudes = rng.uniform(0.5, 1.5, size=int(cfg["n_traces"]))
    noise = rng.multivariate_normal(np.zeros(template.size), true_cov, size=amplitudes.size)
    X = amplitudes[:, None] * template[None, :] + noise

    full_amp = _gls_time_amplitude(X, template, true_metric)
    diagonal_amp = _gls_time_amplitude(X, template, diagonal_metric)
    full_recon = full_amp[:, None] * template[None, :]
    diagonal_recon = diagonal_amp[:, None] * template[None, :]
    full_sigma = float(np.std(full_amp - amplitudes, ddof=1))
    diagonal_sigma = float(np.std(diagonal_amp - amplitudes, ddof=1))
    return {
        "experiment": "S9",
        "correlation_strength": rho,
        "full_covariance_sigma": full_sigma,
        "diagonal_covariance_sigma": diagonal_sigma,
        "diagonal_over_full_sigma": float(diagonal_sigma / full_sigma),
        "full_true_weighted_residual": float(np.mean(weighted_residual(X, full_recon, true_metric))),
        "diagonal_true_weighted_residual": float(np.mean(weighted_residual(X, diagonal_recon, true_metric))),
    }


_RUNNERS = {
    "S0": run_s0_smoke,
    "s0_smoke": run_s0_smoke,
    "S1": run_s1_of_crb,
    "s1_of_crb": run_s1_of_crb,
    "S2": run_s2_of_empca,
    "s2_of_empca": run_s2_of_empca,
    "of_empca_equivalence": run_s2_of_empca,
    "S3": run_s3_ae_bridge,
    "s3_ae_bridge": run_s3_ae_bridge,
    "S4": run_s4_white_control,
    "s4_white_control": run_s4_white_control,
    "S5": run_s5_metric_reversal,
    "s5_metric_reversal": run_s5_metric_reversal,
    "metric_reversal": run_s5_metric_reversal,
    "S6": run_s6_timing_rank_sweep,
    "s6_timing_rank_sweep": run_s6_timing_rank_sweep,
    "S7": run_s7_covariance_robustness,
    "s7_covariance_robustness": run_s7_covariance_robustness,
    "S8": run_s8_residual_calibration,
    "s8_residual_calibration": run_s8_residual_calibration,
    "S9": run_s9_multichannel,
    "s9_multichannel_covariance": run_s9_multichannel,
}


def run_synthetic_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a synthetic experiment by ``experiment_id``."""
    experiment_id = str(config.get("experiment_id", config.get("id", "")))
    if not experiment_id:
        raise ValueError("synthetic config requires experiment_id")
    try:
        return _RUNNERS[experiment_id](config)
    except KeyError as exc:
        raise ValueError(f"unknown synthetic experiment_id: {experiment_id}") from exc
