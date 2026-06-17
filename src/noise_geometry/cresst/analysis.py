"""CRESST pulse reconstruction with empirical noise geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..autoencoders import tied_linear_ae_closed_form
from ..metrics import mse, weighted_residual
from ..noise import estimate_covariance, inverse_covariance, regularize_covariance
from ..subspace import fit_pca, fit_weighted_pca, principal_angles, project_onto_basis
from ..utils.paths import dataset_root
from .loader import load_cresst_traces


def _select_input(config: dict[str, Any], root: Path) -> Path:
    configured = config.get("input_file")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else root / "raw" / path
    candidates = sorted((root / "raw").glob("**/*.npz")) + sorted((root / "raw").glob("**/*.h5"))
    candidates += sorted((root / "raw").glob("**/*.hdf5"))
    if not candidates:
        raise FileNotFoundError(
            f"No CRESST NPZ/HDF5 files under {root / 'raw'}. "
            "Run scripts/download/download_cresst.py and place the released files there."
        )
    return candidates[0]


def run_cresst_experiment(config: dict[str, Any], data_root_path: str | Path) -> dict[str, Any]:
    """Run a first OF/PCA/EMPCA/linear-AE comparison on released traces."""
    root = dataset_root("cresst", data_root_path)
    source = _select_input(config, root)
    traces, labels, metadata = load_cresst_traces(
        source,
        trace_key=config.get("trace_key"),
        label_key=config.get("label_key"),
    )
    rng = np.random.default_rng(int(config.get("seed", 22)))
    max_traces = min(int(config.get("max_traces", 2000)), traces.shape[0])
    chosen = rng.choice(traces.shape[0], size=max_traces, replace=False)
    traces = traces[chosen]
    labels = labels[chosen] if labels is not None else None

    baseline_samples = max(1, int(traces.shape[1] * float(config.get("baseline_fraction", 0.2))))
    traces = traces - traces[:, :baseline_samples].mean(axis=1, keepdims=True)
    max_features = int(config.get("max_features", 256))
    step = max(1, int(np.ceil(traces.shape[1] / max_features)))
    traces = traces[:, ::step][:, :max_features]

    permutation = rng.permutation(traces.shape[0])
    split = max(2, int(0.7 * traces.shape[0]))
    train = traces[permutation[:split]]
    test = traces[permutation[split:]]
    train_labels = labels[permutation[:split]] if labels is not None else None

    noise_label = config.get("noise_label")
    if train_labels is not None and noise_label is not None:
        noise_train = train[train_labels == noise_label]
    else:
        peak = np.max(np.abs(train), axis=1)
        threshold = np.quantile(peak, float(config.get("noise_quantile", 0.2)))
        noise_train = train[peak <= threshold]
    if noise_train.shape[0] < 2:
        raise ValueError("not enough noise-like traces to estimate covariance")

    covariance = regularize_covariance(
        estimate_covariance(noise_train),
        shrinkage=float(config.get("covariance_shrinkage", 0.1)),
    )
    metric = inverse_covariance(covariance)
    rank = int(config.get("rank", 4))
    pca = fit_pca(train, rank)
    empca = fit_weighted_pca(train, metric, rank)
    ae = tied_linear_ae_closed_form(train, rank, weights=metric)
    pca_recon = project_onto_basis(test, pca.components, mean=pca.mean)
    empca_recon = project_onto_basis(test, empca.components, weights=metric, mean=empca.mean)
    angles = principal_angles(empca.components, ae.components, weights=metric)

    pulse_strength = np.max(np.abs(train), axis=1)
    pulse_train = train[pulse_strength >= np.quantile(pulse_strength, 0.5)]
    template = pulse_train.mean(axis=0)
    template /= max(np.max(np.abs(template)), np.finfo(float).eps)
    denom = float(template @ metric @ template)
    amplitude = (test @ metric @ template) / denom
    of_recon = amplitude[:, None] * template[None, :]

    return {
        "experiment": str(config.get("experiment_id", "P2_CRESST")),
        "source": metadata,
        "n_traces": int(traces.shape[0]),
        "n_noise_traces": int(noise_train.shape[0]),
        "n_features": int(traces.shape[1]),
        "rank": rank,
        "pca_raw_mse": float(mse(test, pca_recon)),
        "empca_raw_mse": float(mse(test, empca_recon)),
        "pca_weighted_residual": float(np.mean(weighted_residual(test, pca_recon, metric))),
        "empca_weighted_residual": float(np.mean(weighted_residual(test, empca_recon, metric))),
        "of_weighted_residual": float(np.mean(weighted_residual(test, of_recon, metric))),
        "ae_empca_max_principal_angle_deg": float(np.max(angles)),
    }
