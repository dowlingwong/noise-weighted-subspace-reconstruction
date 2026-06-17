"""Create lightweight diagnostic figures from generated metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "results/metrics")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/figures")
    args = parser.parse_args()

    records = {}
    for path in sorted(args.metrics_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record.get("experiment_id")] = record.get("metrics", {})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    metrics = records.get("S5", {})
    if "pca_weighted_residual_to_observed" in metrics:
        labels = ("PCA", "EMPCA")
        values = (
            metrics["pca_weighted_residual_to_observed"],
            metrics["weighted_pca_weighted_residual_to_observed"],
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(labels, values)
        ax.set_ylabel("mean weighted residual")
        ax.set_title("Synthetic metric reversal")
        fig.tight_layout()
        out = args.output_dir / "synthetic_metric_reversal.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(out)

    metrics = records.get("S6", {})
    if "ranks" in metrics:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.plot(metrics["ranks"], metrics["weighted_residual_by_rank"], marker="o", label="weighted residual")
        ax.axhline(metrics["of_weighted_residual"], color="black", linestyle="--", label="fixed OF")
        ax.set_xlabel("rank")
        ax.set_ylabel("mean residual per feature")
        ax.set_title("Timing-jitter rank sweep")
        ax.legend()
        fig.tight_layout()
        out = args.output_dir / "synthetic_timing_rank_sweep.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(out)

    metrics = records.get("S7", {})
    if "n_noise_traces" in metrics:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ax.semilogx(metrics["n_noise_traces"], metrics["sigma_over_oracle"], marker="o")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set_xlabel("noise traces for covariance estimate")
        ax.set_ylabel("amplitude sigma / oracle")
        ax.set_title("Covariance estimation robustness")
        fig.tight_layout()
        out = args.output_dir / "synthetic_covariance_robustness.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(out)

    metrics = records.get("S9", {})
    if "full_covariance_sigma" in metrics:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(
            ("full", "diagonal"),
            (metrics["full_covariance_sigma"], metrics["diagonal_covariance_sigma"]),
        )
        ax.set_ylabel("amplitude error sigma")
        ax.set_title("Multichannel covariance")
        fig.tight_layout()
        out = args.output_dir / "synthetic_multichannel_covariance.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        generated.append(out)

    if generated:
        for out in generated:
            print(out)
    else:
        print("No metric-reversal metrics found; run scripts/run_all_core.py first.")


if __name__ == "__main__":
    main()
