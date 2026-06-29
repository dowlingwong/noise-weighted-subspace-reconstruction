"""CRESST pulse reconstruction with empirical noise geometry.

Two input paths feed the same OF/PCA/EMPCA/linear-AE comparison:

- the generic NPZ/HDF5 path (``input_file`` + ``trace_key``/``label_key``);
- the public DMDC release path (``release_format: true``), which uses the
  explicit ``noise``/``clean`` flags to separate noise-only baselines from
  accepted pulses. See arXiv:2508.03078.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..autoencoders import tied_linear_ae_closed_form
from ..metrics import mse, weighted_residual
from ..noise import estimate_covariance, inverse_covariance, regularize_covariance
from ..subspace import fit_pca, fit_weighted_pca, principal_angles, project_onto_basis
from ..utils.paths import dataset_root
from .loader import load_cresst_release, load_cresst_traces, select_cresst_subsets


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


def _prep_traces(traces: np.ndarray, *, baseline_fraction: float, max_features: int) -> np.ndarray:
    """Baseline-subtract (per-trace) and downsample by striding."""
    traces = np.asarray(traces, dtype=np.float64)
    baseline_samples = max(1, int(traces.shape[1] * float(baseline_fraction)))
    traces = traces - traces[:, :baseline_samples].mean(axis=1, keepdims=True)
    step = max(1, int(np.ceil(traces.shape[1] / int(max_features))))
    return traces[:, ::step][:, : int(max_features)]


def _model_and_metrics(
    train: np.ndarray,
    test: np.ndarray,
    noise_train: np.ndarray,
    config: dict[str, Any],
    source_meta: Any,
    *,
    n_total: int,
) -> dict[str, Any]:
    """Fit OF/PCA/EMPCA/linear-AE under the empirical noise metric and score."""
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
        "source": source_meta,
        "n_traces": int(n_total),
        "n_noise_traces": int(noise_train.shape[0]),
        "n_features": int(train.shape[1]),
        "rank": rank,
        "pca_raw_mse": float(mse(test, pca_recon)),
        "empca_raw_mse": float(mse(test, empca_recon)),
        "pca_weighted_residual": float(np.mean(weighted_residual(test, pca_recon, metric))),
        "empca_weighted_residual": float(np.mean(weighted_residual(test, empca_recon, metric))),
        "of_weighted_residual": float(np.mean(weighted_residual(test, of_recon, metric))),
        "ae_empca_max_principal_angle_deg": float(np.max(angles)),
    }


def _use_release(config: dict[str, Any], root: Path) -> bool:
    if "release_format" in config:
        return bool(config["release_format"])
    split = str(config.get("split", "test"))
    return (root / "raw" / f"X_{split}.npy").exists()


def _run_release(config: dict[str, Any], root: Path) -> dict[str, Any]:
    """Run the comparison on the public DMDC release using noise/clean flags."""
    split = str(config.get("split", "test"))
    traces, features, meta = load_cresst_release(
        root / "raw",
        split=split,
        max_traces=config.get("max_traces"),
        seed=int(config.get("seed", 22)),
    )
    subsets = select_cresst_subsets(
        traces,
        features,
        noise_column=str(config.get("noise_column", "noise")),
        clean_column=str(config.get("clean_column", "clean")),
    )
    signal = subsets["pulse_traces"]
    noise = subsets["noise_traces"]
    if signal.shape[0] < 4:
        raise ValueError(f"too few clean pulses in split '{split}' ({signal.shape[0]})")
    if noise.shape[0] < 2:
        raise ValueError(f"too few noise traces in split '{split}' ({noise.shape[0]})")

    baseline_fraction = float(config.get("baseline_fraction", 0.2))
    max_features = int(config.get("max_features", 256))
    signal = _prep_traces(signal, baseline_fraction=baseline_fraction, max_features=max_features)
    noise_train = _prep_traces(noise, baseline_fraction=baseline_fraction, max_features=max_features)

    rng = np.random.default_rng(int(config.get("seed", 22)))
    permutation = rng.permutation(signal.shape[0])
    split_at = max(2, int(0.7 * signal.shape[0]))
    train = signal[permutation[:split_at]]
    test = signal[permutation[split_at:]]

    meta = dict(meta)
    meta.update({"n_pulse": subsets["n_pulse"], "n_noise": subsets["n_noise"]})
    return _model_and_metrics(train, test, noise_train, config, meta, n_total=signal.shape[0])


def run_cresst_experiment(config: dict[str, Any], data_root_path: str | Path) -> dict[str, Any]:
    """Run a first OF/PCA/EMPCA/linear-AE comparison on released traces."""
    root = dataset_root("cresst", data_root_path)
    if _use_release(config, root):
        return _run_release(config, root)

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

    traces = _prep_traces(
        traces,
        baseline_fraction=float(config.get("baseline_fraction", 0.2)),
        max_features=int(config.get("max_features", 256)),
    )

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

    return _model_and_metrics(train, test, noise_train, config, metadata, n_total=traces.shape[0])
