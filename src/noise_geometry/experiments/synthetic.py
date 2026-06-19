"""Synthetic Paper 1 validation experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..autoencoders import (
    empca_optimal_loss,
    train_weighted_linear_ae,
    train_whitened_linear_ae,
)
from ..filters import psd_amplitude_variance
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
from ..validation import train_test_split_indices

try:  # central rFFT<->real primitive (repo-root on path, e.g. pytest)
    from ...canonical.empca_equivalence_utils import (
        complex_to_real_whitened,
        real_weight_vector,
        rfft_to_real,
    )
except ImportError:  # src on path (scripts/run_experiment.py)
    from canonical.empca_equivalence_utils import (
        complex_to_real_whitened,
        real_weight_vector,
        rfft_to_real,
    )


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
    # OF/GLS amplitude as the rank-1 projection in the central whitened-real
    # representation: a_hat = <whitened x, whitened s> / <whitened s, whitened s>.
    # Identical to gls_amplitude and OptimumFilter.fit (pinned by tests), but
    # routed through the one shared primitive instead of a separate inner product.
    feat_X = complex_to_real_whitened(X_f, dataset.weights)
    feat_s = complex_to_real_whitened(s_f, dataset.weights)
    amp = (feat_X @ feat_s) / float(feat_s @ feat_s)
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
    cfg = {
        "seed": 7,
        "n_traces": 256,
        "n_samples": 1024,
        "noise_kind": "pink",
        "amplitude_model": "real",
    } | config
    amplitude_model = str(cfg["amplitude_model"])
    dataset = make_rank1_pulse_dataset(
        n_traces=int(cfg["n_traces"]),
        n_samples=int(cfg["n_samples"]),
        noise_kind=str(cfg["noise_kind"]),
        seed=int(cfg["seed"]),
        amplitude_model=amplitude_model,
    )
    summary = run_of_empca_equivalence(dataset, amplitude_model=amplitude_model)
    summary.update({"experiment": "S2", "noise_kind": str(cfg["noise_kind"])})
    return summary


def make_s3_dataset(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the S3 synthetic dataset deterministically from the config seed.

    Returns ``(X, weights, rank)`` where ``X`` is rank-``k`` signal plus diagonal
    colored Gaussian noise and ``weights`` is the diagonal inverse-noise metric.
    Shared by :func:`run_s3_ae_bridge` and ``scripts/train_s3_ae.py`` so the
    verification and any saved model train on identical data.
    """
    cfg = {"seed": 3, "n_traces": 512, "n_features": 128, "rank": 3} | config
    rng = np.random.default_rng(int(cfg["seed"]))
    rank = int(cfg["rank"])
    clean, _ = _rankk_signal(rng, int(cfg["n_traces"]), int(cfg["n_features"]), rank)
    variance = np.geomspace(0.02, 0.5, int(cfg["n_features"]))
    weights = 1.0 / variance
    noise = rng.normal(scale=np.sqrt(variance)[None, :], size=clean.shape)
    return clean + noise, weights, rank


def run_s3_ae_bridge(config: dict[str, Any]) -> dict[str, Any]:
    """S3: verify the EMPCA bridge with *independently trained* tied linear AEs.

    Rather than delegating to ``fit_weighted_pca`` (agreement by construction),
    this trains the tied weighted linear AE by gradient optimisation and checks
    that it converges to the EMPCA global optimum. Evidence per trained model:
    the optimality gap against ``L*``, the principal angle to EMPCA in the noise
    metric, and the M-orthonormality of the learned basis.

    Config keys: ``methods`` (subset of {"direct", "whitened"}), ``optimizers``
    (subset of {"lbfgs", "adam"}), and ``max_iter``. Defaults run both methods
    with L-BFGS plus a reported Adam pass on the direct model.
    """
    cfg = {
        "seed": 3,
        "n_traces": 512,
        "n_features": 128,
        "rank": 3,
        "methods": ["direct", "whitened"],
        "optimizers": ["lbfgs", "adam"],
        "max_iter": 5000,
    } | config
    X, weights, rank = make_s3_dataset(cfg)

    # EMPCA closed form provides the reference subspace and the global optimum L*.
    lstar, empca = empca_optimal_loss(X, weights, rank)
    empca_recon = project_onto_basis(X, empca.components, weights=weights, mean=empca.mean)

    trainers = {"direct": train_weighted_linear_ae, "whitened": train_whitened_linear_ae}
    runs: dict[str, Any] = {}
    for method in cfg["methods"]:
        for opt in cfg["optimizers"]:
            res = trainers[method](
                X, weights, rank, optimizer=opt, seed=int(cfg["seed"]), max_iter=int(cfg["max_iter"])
            )
            ae_recon = project_onto_basis(X, res.components, weights=weights, mean=res.mean)
            runs[f"{method}_{opt}"] = {
                "optimizer": res.optimizer,
                "method": res.method,
                "final_loss": res.final_loss,
                "optimality_gap": res.optimality_gap,
                "relative_gap": res.relative_gap,
                "max_principal_angle_deg": res.max_principal_angle_deg,
                "m_orthonormality_error": res.m_orthonormality_error,
                "reconstruction_rmse_vs_empca": float(np.sqrt(mse(empca_recon, ae_recon))),
                "n_iter": res.n_iter,
            }

    # Headline = independently trained direct L-BFGS model; keep legacy keys.
    headline = runs.get("direct_lbfgs") or next(iter(runs.values()))
    return {
        "experiment": "S3",
        "rank": rank,
        "optimal_loss": lstar,
        "runs": runs,
        "max_principal_angle_deg": headline["max_principal_angle_deg"],
        "mean_principal_angle_deg": headline["max_principal_angle_deg"],
        "reconstruction_rmse": headline["reconstruction_rmse_vs_empca"],
        "relative_gap": headline["relative_gap"],
        "m_orthonormality_error": headline["m_orthonormality_error"],
    }


def run_s4_white_control(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "seed": 4,
        "n_traces": 512,
        "n_features": 128,
        "rank": 3,
        "noise_level": 0.05,
        "test_frac": 0.0,
        "split_seed": 0,
    } | config
    rng = np.random.default_rng(int(cfg["seed"]))
    rank = int(cfg["rank"])
    clean, _ = _rankk_signal(rng, int(cfg["n_traces"]), int(cfg["n_features"]), rank)
    X = clean + rng.normal(scale=float(cfg["noise_level"]), size=clean.shape)
    weights = np.ones(X.shape[1], dtype=np.float64)

    test_frac = float(cfg["test_frac"])
    if test_frac > 0.0:
        train_idx, test_idx = train_test_split_indices(X.shape[0], test_frac, int(cfg["split_seed"]))
    else:
        train_idx = test_idx = np.arange(X.shape[0])
    X_tr, X_te = X[train_idx], X[test_idx]

    pca = fit_pca(X_tr, rank)
    empca = fit_weighted_pca(X_tr, weights, rank)
    pca_recon = project_onto_basis(X_te, pca.components, mean=pca.mean)
    empca_recon = project_onto_basis(X_te, empca.components, weights=weights, mean=empca.mean)
    angles = principal_angles(pca.components, empca.components)
    return {
        "experiment": "S4",
        "held_out": bool(test_frac > 0.0),
        "n_test": int(test_idx.size),
        "pca_mse": float(mse(X_te, pca_recon)),
        "empca_mse": float(mse(X_te, empca_recon)),
        "pca_weighted_residual": float(np.mean(weighted_residual(X_te, pca_recon, weights))),
        "empca_weighted_residual": float(np.mean(weighted_residual(X_te, empca_recon, weights))),
        "max_principal_angle_deg": float(np.max(angles)),
    }


def run_s5_metric_reversal(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "seed": 19,
        "n_traces": 512,
        "n_features": 256,
        "rank": 2,
        "test_frac": 0.0,   # > 0 enables a leakage-free held-out split
        "split_seed": 0,
    } | config
    X, clean, weights_f = _metric_reversal_dataset(cfg)
    rank = int(cfg["rank"])

    # Likelihood-preserving complex representation: stack Re and Im of the rFFT
    # (the previous version kept only the real part, discarding half the
    # information and breaking the Sigma^{-1} geometry). Route through the
    # central primitive so DC/Nyquist/weight conventions live in one place.
    feat_all = rfft_to_real(np.fft.rfft(X, axis=1))
    clean_all = rfft_to_real(np.fft.rfft(clean, axis=1))
    w_real = real_weight_vector(weights_f)

    # Held-out evaluation: fit the subspaces on the train split, score residuals
    # on unseen test traces (test_frac=0 reproduces the in-sample behaviour).
    test_frac = float(cfg["test_frac"])
    if test_frac > 0.0:
        train_idx, test_idx = train_test_split_indices(feat_all.shape[0], test_frac, int(cfg["split_seed"]))
    else:
        train_idx = test_idx = np.arange(feat_all.shape[0])
    feat_tr, feat_te, clean_te = feat_all[train_idx], feat_all[test_idx], clean_all[test_idx]

    pca = fit_pca(feat_tr, rank)
    wpca = fit_weighted_pca(feat_tr, w_real, rank)
    pca_recon = project_onto_basis(feat_te, pca.components, mean=pca.mean)
    wpca_recon = project_onto_basis(feat_te, wpca.components, weights=w_real, mean=wpca.mean)
    pca_wr = weighted_residual(feat_te, pca_recon, w_real)
    wpca_wr = weighted_residual(feat_te, wpca_recon, w_real)
    return {
        "experiment": "S5",
        "rank": rank,
        "representation": "complex_rfft_real_imag_stacked",
        "held_out": bool(test_frac > 0.0),
        "n_test": int(test_idx.size),
        "pca_raw_residual_to_observed": float(mse(feat_te, pca_recon)),
        "weighted_pca_raw_residual_to_observed": float(mse(feat_te, wpca_recon)),
        "pca_weighted_residual_to_observed": float(np.mean(pca_wr)),
        "weighted_pca_weighted_residual_to_observed": float(np.mean(wpca_wr)),
        "pca_nll_mean": float(np.mean(gaussian_nll(feat_te, pca_recon, w_real))),
        "weighted_pca_nll_mean": float(np.mean(gaussian_nll(feat_te, wpca_recon, w_real))),
        "pca_clean_mse_diagnostic": float(mse(clean_te, pca_recon)),
        "weighted_pca_clean_mse_diagnostic": float(mse(clean_te, wpca_recon)),
        "subspace_angle_deg_max": float(np.max(principal_angles(pca.components, wpca.components, weights=w_real))),
    }


def run_s6_timing_rank_sweep(config: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "seed": 6,
        "n_traces": 400,
        "n_features": 128,
        "rank_max": 6,
        "timing_jitter_samples": 3.0,
        "test_frac": 0.0,
        "split_seed": 0,
    } | config
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

    # Held-out rank sweep: fit each rank on the train split, score on test.
    test_frac = float(cfg["test_frac"])
    if test_frac > 0.0:
        train_idx, test_idx = train_test_split_indices(n_traces, test_frac, int(cfg["split_seed"]))
    else:
        train_idx = test_idx = np.arange(n_traces)
    X_tr, X_te, clean_te = X[train_idx], X[test_idx], clean[test_idx]

    amp_of = _gls_time_amplitude(X_te, template, metric)
    of_recon = amp_of[:, None] * template[None, :]
    ranks = list(range(1, int(cfg["rank_max"]) + 1))
    residuals = []
    clean_errors = []
    for rank in ranks:
        fit = fit_weighted_pca(X_tr, metric, rank)
        recon = project_onto_basis(X_te, fit.components, weights=metric, mean=fit.mean)
        residuals.append(float(np.mean(weighted_residual(X_te, recon, metric))))
        clean_errors.append(float(mse(clean_te, recon)))
    return {
        "experiment": "S6",
        "held_out": bool(test_frac > 0.0),
        "n_test": int(test_idx.size),
        "ranks": ranks,
        "weighted_residual_by_rank": residuals,
        "clean_mse_by_rank": clean_errors,
        "of_weighted_residual": float(np.mean(weighted_residual(X_te, of_recon, metric))),
        "of_amplitude_bias": float(np.mean(amp_of - amplitudes[test_idx])),
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
