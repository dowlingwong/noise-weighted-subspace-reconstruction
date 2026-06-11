"""Generate slide-ready figures from the real metric x coverage experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO_ROOT / "paper2" / "results" / "_analysis" / "real_metric_coverage_matrix"
TABLES_DIR = ANALYSIS_DIR / "tables"
TALK_DIR = REPO_ROOT / "paper2" / "results" / "_analysis" / "latest" / "figures" / "talk_deck_selected"


def _load_matrix(name: str) -> pd.DataFrame:
    frame = pd.read_csv(TABLES_DIR / f"{name}_matrix.csv")
    frame = frame.rename(columns={frame.columns[0]: "metric_mode"})
    return frame.set_index("metric_mode").reindex(index=["mahalanobis", "mse"], columns=["full", "restricted"])


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "metric_coverage_run_summary.csv")


def _draw_matrix(
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    value_fmt: str,
    ratio_mode: str,
    coverage_note: str,
    output_path: Path,
) -> None:
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.6, 6.7))
    im = ax.imshow(values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["Full coverage", "Restricted coverage"], fontsize=13)
    ax.set_yticks(range(2))
    ax.set_yticklabels(["Mahalanobis", "MSE"], fontsize=13)
    ax.set_title(title, fontsize=20, pad=10)

    if ratio_mode == "global":
        baseline = float(matrix.loc["mahalanobis", "full"])
        ratios = values / max(baseline, 1e-12)
    elif ratio_mode == "row":
        row_baselines = values[:, [0]]
        ratios = values / np.maximum(row_baselines, 1e-12)
    else:
        raise ValueError(f"Unsupported ratio_mode: {ratio_mode}")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            val = values[row_idx, col_idx]
            ratio = ratios[row_idx, col_idx]
            text = value_fmt.format(val) + f"\n({ratio:.2f}x)"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=15, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(colorbar_label, fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    fig.text(0.5, 0.035, coverage_note, ha="center", va="bottom", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _coverage_note(summary: pd.DataFrame) -> str:
    restricted = summary[summary["coverage_mode"] == "restricted"]
    if restricted.empty:
        return "Restricted coverage: train subset only; validation/test full"
    row = restricted.iloc[0]
    retained = int(row["train_n_after"])
    total = int(row["train_n_before"])
    frac = retained / max(total, 1)
    q_low = float(row["coverage_quantile_low"])
    q_high = float(row["coverage_quantile_high"])
    return (
        f"Restricted coverage = train only on central amplitude/t0 range "
        f"(q{q_low:.1f}-q{q_high:.1f}; {retained}/{total}, {frac:.1%})\n"
        "validation/test full"
    )


def _write_summary(summary: pd.DataFrame, output_path: Path) -> None:
    wr = _load_matrix("weighted_residual_mean")
    mse = _load_matrix("reconstruction_mse")
    full_metric_ratio = wr.loc["mse", "full"] / max(wr.loc["mahalanobis", "full"], 1e-12)

    lines = [
        "Real metric x coverage experiment",
        "",
        "Definition",
        "- Model family: AE",
        "- Input mode: raw",
        "- Metric axis: Mahalanobis vs MSE loss",
        "- Coverage axis: full vs restricted",
        "- Restricted means training only on the central amplitude/t0 quantile box",
        "- Validation/test remain on the full held-out distribution",
        "",
        "Training coverage summary",
    ]
    for row in summary.itertuples(index=False):
        retained = row.train_fraction_retained
        retained_str = "1.000" if pd.isna(retained) else f"{retained:.3f}"
        lines.append(
            f"- {row.experiment_name}: train {int(row.train_n_after)}/{int(row.train_n_before)} retained ({retained_str})"
        )

    lines.extend(
        [
            "",
            "Weighted residual matrix",
            wr.to_string(),
            "",
            "Reconstruction MSE matrix",
            mse.to_string(),
            "",
            "Interpretation",
            "- Coverage effect is real in both metrics: restricted training worsens held-out performance.",
            f"- Weighted residual degrades by {wr.loc['mahalanobis','restricted']/wr.loc['mahalanobis','full']:.2f}x under Mahalanobis and {wr.loc['mse','restricted']/wr.loc['mse','full']:.2f}x under MSE.",
            f"- Reconstruction MSE degrades by {mse.loc['mahalanobis','restricted']/mse.loc['mahalanobis','full']:.2f}x under Mahalanobis and {mse.loc['mse','restricted']/mse.loc['mse','full']:.2f}x under MSE.",
            f"- Full-coverage weighted residual is essentially tied across losses: MSE / Mahalanobis = {full_metric_ratio:.3f}x.",
            "- The clean slide claim is therefore the coverage claim: restricting the latent training range hurts held-out reconstruction.",
            "- The metric axis should be described as the scoring/optimization geometry, not as a standalone win claim from this matrix.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    summary = _load_summary()
    wr = _load_matrix("weighted_residual_mean")
    recon = _load_matrix("reconstruction_mse")
    coverage_note = _coverage_note(summary)

    _draw_matrix(
        wr,
        title="Real Metric × Coverage Matrix\nMahalanobis residual on held-out data",
        colorbar_label="mean weighted residual",
        value_fmt="{:.3g}",
        ratio_mode="row",
        coverage_note=coverage_note,
        output_path=TALK_DIR / "s16_real_metric_coverage_weighted_residual.png",
    )
    _draw_matrix(
        recon,
        title="Real Metric × Coverage Matrix\nReconstruction MSE on held-out data",
        colorbar_label="mean reconstruction MSE",
        value_fmt="{:.3g}",
        ratio_mode="row",
        coverage_note=coverage_note,
        output_path=TALK_DIR / "backup_real_metric_coverage_reconstruction_mse.png",
    )
    _write_summary(summary, TALK_DIR / "s16_real_metric_coverage_summary.txt")
    print(f"wrote figures to {TALK_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
