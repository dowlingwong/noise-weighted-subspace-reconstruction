"""Real artifact-based analysis for Paper 2 reconstruction runs.

This module closes the current gap between:

- training outputs under ``paper2/results/<experiment_name>/``
- talk/paper figures that should summarize actual AE/transformer runs

The analysis flow is:

1. discover concrete run folders with ``config.yaml`` and/or ``checkpoint_best.pt``;
2. optionally re-evaluate checkpoints on the test split and save
   ``predictions_test.h5`` plus ``analysis_metrics.json``;
3. aggregate metrics from run artifacts into CSV/JSON tables;
4. render comparison figures from the real run outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from paper2._torch import require_torch, torch
from paper2.data.datasets import ReconstructionBatch
from paper2.data.whitening import load_one_sided_psd
from paper2.losses.metrics import summarize_reconstruction_metrics
from paper2.trainers.train_reconstruction import (
    build_criterion,
    build_dataloaders,
    build_model,
    build_whitener,
    load_experiment_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class RunRecord:
    """One Paper 2 run directory plus parsed metadata."""

    run_dir: Path
    experiment_name: str
    config_path: Path | None
    checkpoint_path: Path | None
    metrics_path: Path | None
    analysis_metrics_path: Path | None
    curves_path: Path | None
    predictions_path: Path | None
    config: dict[str, Any]
    train_metrics: dict[str, Any]
    analysis_metrics: dict[str, Any]

    @property
    def model_family(self) -> str:
        return str(self.config.get("model", {}).get("family", "unknown"))

    @property
    def input_mode(self) -> str:
        return str(self.config.get("preprocessing", {}).get("input_mode", "unknown"))

    @property
    def loss_mode(self) -> str:
        return str(self.config.get("loss", {}).get("mode", "unknown"))

    @property
    def optimizer_name(self) -> str:
        return str(self.config.get("optimizer", {}).get("name", "unknown"))

    @property
    def latent_dim(self) -> int | None:
        value = self.config.get("model", {}).get("latent_dim")
        return None if value is None else int(value)

    @property
    def phase(self) -> str:
        name = self.experiment_name
        if name.startswith("experiment_d_"):
            return "architecture"
        if self.model_family in {"ae"}:
            return "ae_2x2"
        if self.model_family in {"transformer"}:
            return "transformer_2x2"
        return "other"

    @property
    def metrics(self) -> dict[str, Any]:
        return self.analysis_metrics if self.analysis_metrics else self.train_metrics


@dataclass(slots=True)
class AnalysisPaths:
    """Output paths created by the analysis runner."""

    output_dir: Path
    figures_dir: Path
    tables_dir: Path
    summary_csv: Path
    best_csv: Path
    manifest_json: Path


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_curves(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_relative(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _family_display_name(family: str) -> str:
    mapping = {
        "ae": "AE",
        "cnn_ae": "CNN AE",
        "linear_ae": "Linear AE",
        "transformer": "Transformer",
    }
    return mapping.get(family, family)


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def discover_runs(results_dir: str | Path) -> list[RunRecord]:
    """Discover concrete Paper 2 run folders under a results tree."""

    root = Path(results_dir)
    if not root.exists():
        return []

    runs: list[RunRecord] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if run_dir.name.startswith("_"):
            continue
        config_path = run_dir / "config.yaml"
        checkpoint_path = run_dir / "checkpoint_best.pt"
        metrics_path = run_dir / "metrics.json"
        analysis_metrics_path = run_dir / "analysis_metrics.json"
        curves_path = run_dir / "curves.csv"
        predictions_path = run_dir / "predictions_test.h5"
        if not any(path.exists() for path in (config_path, checkpoint_path, metrics_path, analysis_metrics_path)):
            continue
        config = _load_yaml(config_path)
        experiment_name = str(config.get("experiment", {}).get("name", run_dir.name))
        runs.append(
            RunRecord(
                run_dir=run_dir,
                experiment_name=experiment_name,
                config_path=config_path if config_path.exists() else None,
                checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
                metrics_path=metrics_path if metrics_path.exists() else None,
                analysis_metrics_path=analysis_metrics_path if analysis_metrics_path.exists() else None,
                curves_path=curves_path if curves_path.exists() else None,
                predictions_path=predictions_path if predictions_path.exists() else None,
                config=config,
                train_metrics=_load_json(metrics_path if metrics_path.exists() else None),
                analysis_metrics=_load_json(analysis_metrics_path if analysis_metrics_path.exists() else None),
            )
        )
    return runs


def _baseline_correct_template(template: np.ndarray, pretrigger: int) -> np.ndarray:
    template = np.asarray(template, dtype=np.float64).reshape(-1)
    window = max(1, min(int(pretrigger), template.shape[0]))
    baseline = float(np.mean(template[:window]))
    return template - baseline


def _load_optimum_filter_class():
    src_dir = REPO_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    module = importlib.import_module("OptimumFilter")
    return module.OptimumFilter


def _convert_meta_to_numpy(meta: dict[str, Any]) -> dict[str, np.ndarray]:
    converted: dict[str, np.ndarray] = {}
    for key, value in meta.items():
        if isinstance(value, np.ndarray):
            converted[key] = value
        elif hasattr(value, "detach"):
            converted[key] = value.detach().cpu().numpy()
        else:
            converted[key] = np.asarray(value)
    return converted


def _stack_meta_values(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.asarray([])
    first = np.asarray(values[0])
    if first.ndim == 0:
        return np.asarray([np.asarray(value).item() for value in values])
    return np.concatenate(values, axis=0)


def _prepare_batch(batch: ReconstructionBatch, device) -> tuple[ReconstructionBatch, dict[str, np.ndarray]]:
    meta_cpu = _convert_meta_to_numpy(batch.meta)
    meta_device = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.meta.items()
    }
    batch_device = ReconstructionBatch(x=batch.x.to(device), meta=meta_device)
    return batch_device, meta_cpu


def _run_optimum_filter(
    traces: np.ndarray,
    cfg_raw: dict[str, Any],
    template_path: Path | None,
    sampling_frequency: float,
    of_mode: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if template_path is None or not template_path.exists():
        return {}, {"of_metrics_status": "skipped_missing_template"}
    if traces.ndim != 3 or traces.shape[1] != 1:
        return {}, {"of_metrics_status": "skipped_multichannel"}

    template = np.load(template_path)
    pretrigger = int(cfg_raw["data"].get("pretrigger", 4000))
    template = _baseline_correct_template(template, pretrigger=pretrigger)
    if template.shape[0] != traces.shape[-1]:
        return {}, {
            "of_metrics_status": "skipped_template_length_mismatch",
            "template_len": int(template.shape[0]),
            "trace_len": int(traces.shape[-1]),
        }

    psd_path = Path(cfg_raw["preprocessing"]["psd_path"])
    if not psd_path.is_absolute():
        psd_path = REPO_ROOT / psd_path
    psd = load_one_sided_psd(psd_path)
    OptimumFilter = _load_optimum_filter_class()
    optimum_filter = OptimumFilter(template, psd, sampling_frequency)

    amplitude = np.zeros(traces.shape[0], dtype=np.float32)
    timing = np.zeros(traces.shape[0], dtype=np.float32)
    chi2 = np.zeros(traces.shape[0], dtype=np.float32)

    for idx in range(traces.shape[0]):
        trace = np.asarray(traces[idx, 0], dtype=np.float64)
        if of_mode == "shifted":
            amp, chisq, t0 = optimum_filter.fit_with_shift(trace)
            timing[idx] = float(t0)
        elif of_mode == "fixed":
            amp, chisq = optimum_filter.fit(trace)
            timing[idx] = 0.0
        else:
            raise ValueError(f"Unsupported of_mode: {of_mode}")
        amplitude[idx] = float(amp)
        chi2[idx] = float(chisq)

    return {
        "of_amplitude_pred": amplitude,
        "of_time_pred": timing,
        "of_chi2_pred": chi2,
    }, {"of_metrics_status": "ok", "of_mode": of_mode}


def _rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)))


def _corrcoef(lhs: np.ndarray, rhs: np.ndarray) -> float | None:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.size < 2 or rhs.size < 2:
        return None
    if np.std(lhs) == 0.0 or np.std(rhs) == 0.0:
        return None
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _derive_auxiliary_metrics(
    arrays: dict[str, np.ndarray],
    meta_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    amp_pred = arrays.get("of_amplitude_pred")
    time_pred = arrays.get("of_time_pred")
    amp_ref = meta_arrays.get("amplitude")
    time_ref = meta_arrays.get("t0")

    if amp_pred is not None and amp_ref is not None and amp_pred.shape == amp_ref.shape:
        metrics["amplitude_rmse"] = _rmse(amp_pred, amp_ref)
        metrics["amplitude_bias"] = float(np.mean(np.asarray(amp_pred) - np.asarray(amp_ref)))
        corr = _corrcoef(amp_pred, amp_ref)
        if corr is not None:
            metrics["amplitude_corr"] = corr

    if time_pred is not None and time_ref is not None and time_pred.shape == time_ref.shape:
        metrics["timing_rmse"] = _rmse(time_pred, time_ref)
        metrics["timing_bias"] = float(np.mean(np.asarray(time_pred) - np.asarray(time_ref)))
        corr = _corrcoef(time_pred, time_ref)
        if corr is not None:
            metrics["timing_corr"] = corr

    of_amp_ref = meta_arrays.get("of_amplitude")
    of_time_ref = meta_arrays.get("of_time")
    if amp_pred is not None and of_amp_ref is not None and amp_pred.shape == of_amp_ref.shape:
        metrics["of_reference_amplitude_rmse"] = _rmse(amp_pred, of_amp_ref)
    if time_pred is not None and of_time_ref is not None and time_pred.shape == of_time_ref.shape:
        metrics["of_reference_timing_rmse"] = _rmse(time_pred, of_time_ref)

    return metrics


def _write_predictions_file(
    output_path: Path,
    cfg_raw: dict[str, Any],
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for key, value in arrays.items():
            if value is None:
                continue
            handle.create_dataset(key, data=value)
        handle.attrs["experiment_name"] = str(cfg_raw["experiment"]["name"])
        handle.attrs["model_family"] = str(cfg_raw["model"]["family"])
        handle.attrs["input_mode"] = str(cfg_raw["preprocessing"]["input_mode"])
        handle.attrs["loss_mode"] = str(cfg_raw["loss"]["mode"])
        handle.attrs["metrics_json"] = json.dumps(metrics, sort_keys=True)


def _load_compatible_state_dict(model, state_dict: dict[str, Any]) -> None:
    """Load checkpoints across small architecture bookkeeping changes."""

    patch_norm_keys = {
        "encoder.patch_norm.weight",
        "encoder.patch_norm.bias",
    }
    model_state = model.state_dict()
    if patch_norm_keys.issubset(model_state):
        missing_patch_norm = patch_norm_keys - set(state_dict)
    else:
        missing_patch_norm = set()
    if missing_patch_norm:
        if missing_patch_norm == {
            "encoder.patch_norm.weight",
            "encoder.patch_norm.bias",
        }:
            state_dict = dict(state_dict)
            state_dict["encoder.patch_norm.weight"] = model_state["encoder.patch_norm.weight"]
            state_dict["encoder.patch_norm.bias"] = model_state["encoder.patch_norm.bias"]
        else:
            raise RuntimeError(f"Partial patch_norm checkpoint state is not supported: {missing_patch_norm}")

    model.load_state_dict(state_dict)


def evaluate_checkpoint_run(
    run: RunRecord,
    *,
    template_path: str | Path | None,
    sampling_frequency: float,
    of_mode: str,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate one trained checkpoint and persist concrete test predictions."""

    if run.config_path is None:
        raise FileNotFoundError(f"Run has no config.yaml: {run.run_dir}")
    if run.checkpoint_path is None:
        raise FileNotFoundError(f"Run has no checkpoint_best.pt: {run.run_dir}")

    predictions_path = run.run_dir / "predictions_test.h5"
    analysis_metrics_path = run.run_dir / "analysis_metrics.json"
    if (
        not force
        and predictions_path.exists()
        and analysis_metrics_path.exists()
    ):
        return _load_json(analysis_metrics_path)

    require_torch()
    cfg = load_experiment_config(run.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whitener = build_whitener(cfg).to(device)
    model = build_model(cfg, whitener).to(device)
    state_dict = torch.load(run.checkpoint_path, map_location=device)
    _load_compatible_state_dict(model, state_dict)
    model.eval()
    criterion = build_criterion(cfg)
    loaders = build_dataloaders(cfg)

    x_true_batches: list[np.ndarray] = []
    x_hat_batches: list[np.ndarray] = []
    z_batches: list[np.ndarray] = []
    meta_batches: dict[str, list[np.ndarray]] = {}

    loss_total = 0.0
    weighted_total = 0.0
    mse_total = 0.0
    n_batches = 0
    n_examples = 0

    with torch.no_grad():
        for batch in loaders["test"]:
            batch_device, meta_cpu = _prepare_batch(batch, device)
            output = model(batch_device.x)
            loss_out = criterion(output, batch_device, whitener)
            metrics = summarize_reconstruction_metrics(batch_device.x, output.x_hat, whitener)

            batch_n = int(batch_device.x.shape[0])
            loss_total += float(loss_out.total.detach().cpu()) * batch_n
            weighted_total += metrics.weighted_residual_mean * batch_n
            mse_total += metrics.reconstruction_mse * batch_n
            n_batches += 1
            n_examples += batch_n

            x_true_batches.append(batch_device.x.detach().cpu().numpy())
            x_hat_batches.append(output.x_hat.detach().cpu().numpy())
            if output.z is not None:
                z_batches.append(output.z.detach().cpu().numpy())
            for key, value in meta_cpu.items():
                meta_batches.setdefault(key, []).append(np.asarray(value))

    arrays: dict[str, np.ndarray] = {
        "x_true": np.concatenate(x_true_batches, axis=0),
        "x_hat": np.concatenate(x_hat_batches, axis=0),
    }
    if z_batches:
        arrays["z"] = np.concatenate(z_batches, axis=0)

    meta_arrays = {key: _stack_meta_values(values) for key, values in meta_batches.items()}
    arrays.update(meta_arrays)

    if of_mode != "none":
        of_arrays, of_meta = _run_optimum_filter(
            arrays["x_hat"],
            cfg.raw,
            template_path=Path(template_path) if template_path is not None else None,
            sampling_frequency=sampling_frequency,
            of_mode=of_mode,
        )
        arrays.update(of_arrays)
    else:
        of_meta = {"of_metrics_status": "disabled"}

    denom = max(n_examples, 1)
    analysis_metrics: dict[str, Any] = {
        "eval_loss": loss_total / denom,
        "weighted_residual_mean": weighted_total / denom,
        "reconstruction_mse": mse_total / denom,
        "n_test_batches": int(n_batches),
        "n_test_examples": int(arrays["x_true"].shape[0]),
        **of_meta,
    }
    analysis_metrics.update(_derive_auxiliary_metrics(arrays, meta_arrays))

    _write_predictions_file(predictions_path, cfg.raw, arrays, analysis_metrics)
    with analysis_metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis_metrics, handle, indent=2, sort_keys=True)
    return analysis_metrics


def _logical_group_key(run: RunRecord) -> str:
    return f"{run.phase}|{run.model_family}|{run.input_mode}|{run.loss_mode}"


def build_summary_frame(runs: Iterable[RunRecord], *, results_dir: str | Path) -> pd.DataFrame:
    """Turn run records into one flat summary table."""

    root = Path(results_dir)
    columns = [
        "experiment_name",
        "phase",
        "model_family",
        "model_family_display",
        "input_mode",
        "loss_mode",
        "optimizer",
        "latent_dim",
        "run_dir",
        "config_path",
        "checkpoint_path",
        "metrics_path",
        "analysis_metrics_path",
        "curves_path",
        "predictions_path",
        "has_checkpoint",
        "has_metrics_json",
        "has_analysis_metrics",
        "logical_group",
        "eval_loss",
        "weighted_residual_mean",
        "reconstruction_mse",
        "amplitude_rmse",
        "timing_rmse",
        "of_reference_amplitude_rmse",
        "of_reference_timing_rmse",
        "best_epoch",
    ]
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = run.metrics
        row = {
            "experiment_name": run.experiment_name,
            "phase": run.phase,
            "model_family": run.model_family,
            "model_family_display": _family_display_name(run.model_family),
            "input_mode": run.input_mode,
            "loss_mode": run.loss_mode,
            "optimizer": run.optimizer_name,
            "latent_dim": run.latent_dim,
            "run_dir": _safe_relative(run.run_dir, root),
            "config_path": _safe_relative(run.config_path, root),
            "checkpoint_path": _safe_relative(run.checkpoint_path, root),
            "metrics_path": _safe_relative(run.metrics_path, root),
            "analysis_metrics_path": _safe_relative(run.analysis_metrics_path, root),
            "curves_path": _safe_relative(run.curves_path, root),
            "predictions_path": _safe_relative(run.predictions_path, root),
            "has_checkpoint": run.checkpoint_path is not None,
            "has_metrics_json": run.metrics_path is not None,
            "has_analysis_metrics": run.analysis_metrics_path is not None,
            "logical_group": _logical_group_key(run),
            "eval_loss": _metric_value(metrics, "eval_loss"),
            "weighted_residual_mean": _metric_value(metrics, "weighted_residual_mean"),
            "reconstruction_mse": _metric_value(metrics, "reconstruction_mse"),
            "amplitude_rmse": _metric_value(metrics, "amplitude_rmse"),
            "timing_rmse": _metric_value(metrics, "timing_rmse"),
            "of_reference_amplitude_rmse": _metric_value(metrics, "of_reference_amplitude_rmse"),
            "of_reference_timing_rmse": _metric_value(metrics, "of_reference_timing_rmse"),
            "best_epoch": _metric_value(metrics, "epoch"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def select_best_runs(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Pick one best row per logical configuration using actual eval loss when available."""

    if summary_df.empty:
        return summary_df.copy()

    sortable = summary_df.copy()
    sortable["_eval_sort"] = sortable["eval_loss"].fillna(np.inf)
    sortable["_recon_sort"] = sortable["reconstruction_mse"].fillna(np.inf)
    sortable = sortable.sort_values(
        by=["logical_group", "_eval_sort", "_recon_sort", "experiment_name"],
        ascending=[True, True, True, True],
    )
    best = sortable.groupby("logical_group", as_index=False).first()
    return best.drop(columns=["_eval_sort", "_recon_sort"])


def _annotated_heatmap(
    ax,
    matrix: pd.DataFrame,
    *,
    title: str,
    cmap: str = "viridis",
) -> None:
    if matrix.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(no data)")
        return

    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = values[row_idx, col_idx]
            label = "NA" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_two_by_two_family(best_df: pd.DataFrame, family: str, output_path: Path) -> bool:
    subset = best_df[best_df["model_family"] == family].copy()
    subset = subset[subset["phase"].isin(["ae_2x2", "transformer_2x2"])]
    if subset.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    metric_titles = [
        ("weighted_residual_mean", "Weighted residual"),
        ("reconstruction_mse", "Reconstruction MSE"),
    ]
    for ax, (metric, title) in zip(axes, metric_titles):
        matrix = subset.pivot(index="input_mode", columns="loss_mode", values=metric)
        matrix = matrix.reindex(index=["raw", "prewhitened"], columns=["mse", "mahalanobis"])
        _annotated_heatmap(ax, matrix, title=title, cmap="magma_r")
    fig.suptitle(f"{_family_display_name(family)} 2x2 from actual run outputs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_architecture_comparison(best_df: pd.DataFrame, output_path: Path) -> bool:
    subset = best_df[best_df["phase"] == "architecture"].copy()
    if subset.empty:
        return False

    subset = subset.sort_values(by=["model_family_display", "experiment_name"])
    labels = subset["model_family_display"].tolist()
    metrics = [
        ("weighted_residual_mean", "Weighted residual"),
        ("reconstruction_mse", "Reconstruction MSE"),
        ("amplitude_rmse", "Amplitude RMSE"),
        ("timing_rmse", "Timing RMSE"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        values = subset[metric].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if not mask.any():
            ax.axis("off")
            ax.set_title(f"{title}\n(no data)")
            continue
        shown_labels = [label for label, keep in zip(labels, mask) if keep]
        shown_values = values[mask]
        bars = ax.bar(shown_labels, shown_values, color=["#4263eb", "#2b8a3e", "#d9480f"][: len(shown_values)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, shown_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Experiment D architecture comparison from actual checkpoints")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_learning_curves(runs: Iterable[RunRecord], output_path: Path) -> bool:
    plotted = False
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    metric_specs = [
        ("eval_loss", "Validation loss"),
        ("reconstruction_mse", "Validation reconstruction MSE"),
    ]

    for run in runs:
        curves = _load_curves(run.curves_path)
        if curves.empty or "epoch" not in curves.columns:
            continue
        label = run.experiment_name
        for ax, (metric, title) in zip(axes, metric_specs):
            if metric not in curves.columns:
                continue
            ax.plot(curves["epoch"], curves[metric], label=label, linewidth=1.8)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            plotted = True

    if not plotted:
        plt.close(fig)
        return False

    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle("Learning curves from actual Paper 2 runs")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_analysis_outputs(
    runs: list[RunRecord],
    *,
    results_dir: str | Path,
    output_dir: str | Path,
) -> AnalysisPaths:
    """Write summary tables and figures for the discovered runs."""

    out_dir = Path(output_dir)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_frame(runs, results_dir=results_dir)
    best_df = select_best_runs(summary_df)

    summary_csv = tables_dir / "run_summary.csv"
    best_csv = tables_dir / "best_by_logical_group.csv"
    manifest_json = out_dir / "manifest.json"
    summary_df.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    written_figures: list[str] = []
    figure_specs = [
        ("ae_2x2_actual.png", plot_two_by_two_family, {"family": "ae"}),
        ("transformer_2x2_actual.png", plot_two_by_two_family, {"family": "transformer"}),
        ("architecture_actual.png", plot_architecture_comparison, {}),
        ("learning_curves_actual.png", plot_learning_curves, {}),
    ]
    for filename, fn, kwargs in figure_specs:
        target = figures_dir / filename
        if fn is plot_two_by_two_family:
            written = fn(best_df, output_path=target, **kwargs)
        elif fn is plot_architecture_comparison:
            written = fn(best_df, output_path=target)
        else:
            written = fn(runs, output_path=target)
        if written:
            written_figures.append(filename)

    manifest = {
        "results_dir": str(Path(results_dir)),
        "n_runs_discovered": len(runs),
        "n_runs_with_analysis_metrics": int(sum(bool(run.analysis_metrics) for run in runs)),
        "summary_csv": str(summary_csv),
        "best_csv": str(best_csv),
        "figures": written_figures,
    }
    with manifest_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    return AnalysisPaths(
        output_dir=out_dir,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        summary_csv=summary_csv,
        best_csv=best_csv,
        manifest_json=manifest_json,
    )


def analyze_results_tree(
    *,
    results_dir: str | Path,
    output_dir: str | Path,
    evaluate_mode: str = "auto",
    force_eval: bool = False,
    template_path: str | Path | None = None,
    sampling_frequency: float = 2.5e5,
    of_mode: str = "shifted",
    only: Iterable[str] | None = None,
) -> tuple[list[RunRecord], AnalysisPaths]:
    """Main orchestration entry point for the real analysis path."""

    runs = discover_runs(results_dir)
    if only:
        wanted = {item.strip() for item in only if item and item.strip()}
        runs = [run for run in runs if run.experiment_name in wanted or run.run_dir.name in wanted]

    if evaluate_mode not in {"auto", "metrics-only", "checkpoints"}:
        raise ValueError(f"Unsupported evaluate_mode: {evaluate_mode}")

    if evaluate_mode != "metrics-only":
        for run in runs:
            if run.checkpoint_path is None or run.config_path is None:
                continue
            if evaluate_mode == "auto" and run.analysis_metrics_path is not None and not force_eval:
                continue
            evaluate_checkpoint_run(
                run,
                template_path=template_path,
                sampling_frequency=sampling_frequency,
                of_mode=of_mode,
                force=force_eval,
            )

    # Refresh records after potential writes.
    refreshed = {run.run_dir: run for run in discover_runs(results_dir)}
    runs = [refreshed.get(run.run_dir, run) for run in runs]
    paths = generate_analysis_outputs(runs, results_dir=results_dir, output_dir=output_dir)
    return runs, paths
