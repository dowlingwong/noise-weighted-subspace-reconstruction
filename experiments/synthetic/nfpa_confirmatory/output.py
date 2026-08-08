"""Output contract (protocol section 13): one tidy row per (seed, method, split).

Raw seed-level results are written to disk before any aggregation, with the
frozen configuration hash embedded in every row. Aggregation reads only these
files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import PRIMARY, PRIMARY_HASH, PrimaryConfig
from .generator import Dataset
from .methods import FIT_FACTORY, MethodFit, fit_cw_pca
from .metrics import evaluate_fit
from .semantic import evaluate_semantics

COLUMNS = [
    "config_hash",
    "seed",
    "regime",
    "method",
    "metric",
    "rep_class",
    "split",
    "C",
    "T",
    "k_c",
    "k_t",
    "total_rank",
    "N_train",
    "N_validation",
    "N_test",
    "target_snr",
    "realized_snr_mean",
    "realized_snr_std",
    "noise_kind",
    "nonseparability_alpha",
    "train_raw_mse",
    "test_raw_mse",
    "train_normalized_chi2",
    "test_normalized_chi2",
    "angle_to_cw_pca_mean_deg",
    "angle_to_cw_pca_max_deg",
    "amplitude_bias",
    "amplitude_rmse",
    "amplitude_correlation",
    "rise_time_rmse",
    "decay_time_rmse",
    "channel_projector_error",
    "temporal_projector_error",
    "selected_restart",
    "converged",
    "runtime_seconds",
    "covariance_symmetry_error",
    "covariance_whitening_error",
    "covariance_mc_error",
]


def build_row(ds: Dataset, fit: MethodFit, cw_pca: MethodFit, *, runtime: float) -> dict:
    cfg: PrimaryConfig = ds.cfg
    test_metrics = evaluate_fit(ds, fit, cw_pca, ds.X_test)
    train_metrics = evaluate_fit(ds, fit, cw_pca, ds.X_train)
    semantics = evaluate_semantics(fit, ds)
    diag = fit.diag
    cr = ds.covariance_report
    return {
        "config_hash": PRIMARY_HASH,
        "seed": ds.seed,
        "regime": ds.regime,
        "method": fit.name,
        "metric": fit.metric,
        "rep_class": fit.rep_class,
        "split": "test",
        "C": cfg.C,
        "T": cfg.T,
        "k_c": cfg.k_c,
        "k_t": cfg.k_t,
        "total_rank": cfg.k_c * cfg.k_t,
        "N_train": cfg.N_train,
        "N_validation": cfg.N_validation,
        "N_test": cfg.N_test,
        "target_snr": cfg.target_snr,
        "realized_snr_mean": ds.snr_stats["realized_snr_mean"],
        "realized_snr_std": ds.snr_stats["realized_snr_std"],
        "noise_kind": ds.regime,
        "nonseparability_alpha": cfg.nonseparability_alpha,
        "train_raw_mse": train_metrics["raw_mse"],
        "test_raw_mse": test_metrics["raw_mse"],
        "train_normalized_chi2": train_metrics["normalized_chi2"],
        "test_normalized_chi2": test_metrics["normalized_chi2"],
        "angle_to_cw_pca_mean_deg": test_metrics["angle_to_cw_pca_mean_deg"],
        "angle_to_cw_pca_max_deg": test_metrics["angle_to_cw_pca_max_deg"],
        "amplitude_bias": semantics["amplitude_bias"],
        "amplitude_rmse": semantics["amplitude_rmse"],
        "amplitude_correlation": semantics["amplitude_correlation"],
        "rise_time_rmse": semantics["rise_time_rmse"],
        "decay_time_rmse": semantics["decay_time_rmse"],
        "channel_projector_error": semantics["channel_projector_error"],
        "temporal_projector_error": semantics["temporal_projector_error"],
        "selected_restart": diag.get("selected_restart", -1),
        "converged": diag.get("converged", True),
        "runtime_seconds": runtime,
        "covariance_symmetry_error": cr.get("symmetry_error", float("nan")),
        "covariance_whitening_error": cr.get("whitening_identity_error", float("nan")),
        "covariance_mc_error": cr.get("monte_carlo_covariance_relative_error", float("nan")),
    }


def write_raw_rows(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
