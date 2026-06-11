"""Real-data metric x coverage matrix for NPML slide regeneration.

This script turns the conceptual backup matrix into an actual four-run AE
experiment on the available K-alpha dataset:

- loss metric: `mahalanobis` vs `mse`
- train coverage: `full` vs `restricted`

Restricted coverage is implemented as a train-only filter that keeps samples
inside the central amplitude/timing quantile box, while validation and test
remain on the full held-out distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper2.analysis.reporting import RunRecord, discover_runs, evaluate_checkpoint_run
from paper2.trainers.train_reconstruction import load_experiment_config, run_experiment


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "paper2" / "results"
OUTPUT_DIR = RESULTS_DIR / "_analysis" / "real_metric_coverage_matrix"
TEMPLATE_PATH = REPO_ROOT / "data" / "k_alpha" / "template_K_alpha_tight.npy"

SUITE = [
    {
        "experiment_name": "metric_coverage_mahalanobis_full",
        "config_path": REPO_ROOT / "paper2" / "configs" / "metric_coverage_mahalanobis_full.yaml",
        "metric_mode": "mahalanobis",
        "coverage_mode": "full",
    },
    {
        "experiment_name": "metric_coverage_mahalanobis_restricted",
        "config_path": REPO_ROOT / "paper2" / "configs" / "metric_coverage_mahalanobis_restricted.yaml",
        "metric_mode": "mahalanobis",
        "coverage_mode": "restricted",
    },
    {
        "experiment_name": "metric_coverage_mse_full",
        "config_path": REPO_ROOT / "paper2" / "configs" / "metric_coverage_mse_full.yaml",
        "metric_mode": "mse",
        "coverage_mode": "full",
    },
    {
        "experiment_name": "metric_coverage_mse_restricted",
        "config_path": REPO_ROOT / "paper2" / "configs" / "metric_coverage_mse_restricted.yaml",
        "metric_mode": "mse",
        "coverage_mode": "restricted",
    },
]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _suite_run_records() -> list[RunRecord]:
    wanted = {spec["experiment_name"] for spec in SUITE}
    return [run for run in discover_runs(RESULTS_DIR) if run.experiment_name in wanted]


def run_training_suite(force: bool = False) -> None:
    for spec in SUITE:
        run_dir = RESULTS_DIR / spec["experiment_name"]
        checkpoint_path = run_dir / "checkpoint_best.pt"
        if checkpoint_path.exists() and not force:
            print(f"[metric-coverage] skip train existing {spec['experiment_name']}", flush=True)
            continue
        print(f"[metric-coverage] train {spec['experiment_name']}", flush=True)
        run_experiment(spec["config_path"])


def run_analysis_suite(force: bool = False, sampling_frequency: float = 2.5e5) -> None:
    runs = _suite_run_records()
    for run in runs:
        if run.checkpoint_path is None or run.config_path is None:
            continue
        analysis_path = run.run_dir / "analysis_metrics.json"
        if analysis_path.exists() and not force:
            print(f"[metric-coverage] skip analysis existing {run.experiment_name}", flush=True)
            continue
        print(f"[metric-coverage] analyze {run.experiment_name}", flush=True)
        evaluate_checkpoint_run(
            run,
            template_path=TEMPLATE_PATH,
            sampling_frequency=sampling_frequency,
            of_mode="shifted",
            force=force,
        )


def build_summary_frame() -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    run_map = {run.experiment_name: run for run in _suite_run_records()}
    for spec in SUITE:
        run = run_map.get(spec["experiment_name"])
        cfg = load_experiment_config(spec["config_path"]).raw
        metrics = {}
        coverage_summary = {}
        if run is not None:
            metrics = run.analysis_metrics if run.analysis_metrics else run.train_metrics
            coverage_summary = _load_json(run.run_dir / "coverage_summary.json")
        records.append(
            {
                "experiment_name": spec["experiment_name"],
                "metric_mode": spec["metric_mode"],
                "coverage_mode": spec["coverage_mode"],
                "config_path": str(spec["config_path"].relative_to(REPO_ROOT)),
                "train_n_before": coverage_summary.get("train_n_before"),
                "train_n_after": coverage_summary.get("train_n_after"),
                "train_fraction_retained": coverage_summary.get("train_fraction_retained"),
                "coverage_fields": ",".join(coverage_summary.get("coverage_fields", [])),
                "coverage_quantile_low": coverage_summary.get("coverage_quantile_low"),
                "coverage_quantile_high": coverage_summary.get("coverage_quantile_high"),
                "weighted_residual_mean": metrics.get("weighted_residual_mean"),
                "reconstruction_mse": metrics.get("reconstruction_mse"),
                "amplitude_rmse": metrics.get("amplitude_rmse"),
                "timing_rmse": metrics.get("timing_rmse"),
                "best_epoch": metrics.get("epoch"),
                "input_mode": cfg["preprocessing"]["input_mode"],
                "model_family": cfg["model"]["family"],
            }
        )
    return pd.DataFrame.from_records(records)


def ensure_complete_suite(frame: pd.DataFrame) -> None:
    required_metrics = ["weighted_residual_mean", "amplitude_rmse"]
    missing_rows = frame[frame[required_metrics].isna().any(axis=1)]
    if missing_rows.empty:
        return
    descriptions = [
        f"{row.experiment_name} (metric={row.metric_mode}, coverage={row.coverage_mode})"
        for row in missing_rows.itertuples(index=False)
    ]
    raise RuntimeError(
        "Real metric-coverage matrix is incomplete. Missing analyzed outputs for: "
        + ", ".join(descriptions)
        + ". Run this script with `--train --analyze` on a machine with torch."
    )


def _matrix_from_frame(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    matrix = frame.pivot(index="metric_mode", columns="coverage_mode", values=metric_name)
    return matrix.reindex(index=["mahalanobis", "mse"], columns=["full", "restricted"])


def _plot_matrix(
    matrix: pd.DataFrame,
    *,
    metric_name: str,
    title: str,
    output_path: Path,
) -> None:
    if matrix.isnull().any().any():
        missing = matrix.isnull()
        raise RuntimeError(
            f"Cannot render {metric_name}: missing cells at "
            f"{[(matrix.index[i], matrix.columns[j]) for i, j in zip(*np.where(missing.to_numpy()))]}"
        )

    baseline = float(matrix.loc["mahalanobis", "full"])
    ratios = matrix / max(baseline, 1e-12)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(["Full coverage", "Restricted coverage"])
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(["Mahalanobis", "MSE"])
    ax.set_title(title)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = float(matrix.iat[row_idx, col_idx])
            ratio = float(ratios.iat[row_idx, col_idx])
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3g}\n({ratio:.2f}x)",
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax, shrink=0.84, label=metric_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_outputs(frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    tables_dir = OUTPUT_DIR / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    frame.to_csv(tables_dir / "metric_coverage_run_summary.csv", index=False)

    metric_specs = [
        (
            "amplitude_rmse",
            "Real NPML Matrix: AE amplitude RMSE\n(actual K-alpha runs; lower is better)",
            figures_dir / "real_metric_coverage_matrix_amplitude_rmse.png",
        ),
        (
            "weighted_residual_mean",
            "Real NPML Matrix: AE Mahalanobis residual\n(actual K-alpha runs; lower is better)",
            figures_dir / "real_metric_coverage_matrix_weighted_residual.png",
        ),
    ]
    manifest: dict[str, Any] = {
        "results_dir": str(RESULTS_DIR),
        "suite_experiments": [spec["experiment_name"] for spec in SUITE],
        "figures": [],
    }
    for metric_name, title, output_path in metric_specs:
        matrix = _matrix_from_frame(frame, metric_name=metric_name)
        matrix.to_csv(tables_dir / f"{metric_name}_matrix.csv")
        _plot_matrix(matrix, metric_name=metric_name, title=title, output_path=output_path)
        manifest["figures"].append(output_path.name)

    with (OUTPUT_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the four training jobs before regenerating the matrix.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Recompute analysis_metrics.json from checkpoints before plotting.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoints/analysis outputs where applicable.",
    )
    parser.add_argument(
        "--sampling-frequency",
        type=float,
        default=2.5e5,
        help="Sampling frequency passed to the Optimum Filter analysis stage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.train:
        run_training_suite(force=args.force)
    if args.analyze:
        run_analysis_suite(force=args.force, sampling_frequency=args.sampling_frequency)

    frame = build_summary_frame()
    ensure_complete_suite(frame)
    render_outputs(frame)
    print(f"[metric-coverage] wrote outputs to {OUTPUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
