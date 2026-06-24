"""Loading and publication-style plotting helpers for Paper 1 notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_ROOT = SCRIPT_DIR.parent
DATA_ROOT = TRANSFER_ROOT / "data"
DERIVED_ROOT = DATA_ROOT / "derived"
FIGURE_ROOT = TRANSFER_ROOT / "figures"
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#777777",
}


def apply_publication_style() -> None:
    """Apply a compact, colorblind-safe publication plotting style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save one figure as vector PDF and 300-dpi PNG."""
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / f"{stem}.pdf"
    png = FIGURE_ROOT / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    return pdf, png


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def current_gwosc_record() -> dict[str, Any]:
    return load_json(DATA_ROOT / "gwosc" / "current" / "gwosc" / "experiment.json")


def followup_record(name: str) -> dict[str, Any] | None:
    path = DATA_ROOT / "gwosc" / "followup" / name
    return load_json(path) if path.is_file() else None


def plot_synthetic_validation_overview() -> plt.Figure:
    """Plot four representative interval-backed synthetic gate outcomes."""
    apply_publication_style()
    s1 = pd.read_csv(DATA_ROOT / "synthetic" / "S1_sweep_10seeds.csv")
    s2 = pd.read_csv(DATA_ROOT / "synthetic" / "S2_sweep_10seeds.csv")
    s5 = pd.read_csv(DATA_ROOT / "synthetic" / "S5_sweep_10seeds.csv")
    s8 = pd.read_csv(DATA_ROOT / "synthetic" / "S8_sweep_10seeds.csv")
    residual_difference = (
        s5["pca_weighted_residual_to_observed"]
        - s5["weighted_pca_weighted_residual_to_observed"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    panels = [
        (
            axes[0, 0],
            s1["seed"],
            s1["sigma_over_crb"],
            "S1: OF uncertainty / CRB",
            1.0,
        ),
        (
            axes[0, 1],
            s2["seed"],
            s2["weighted_angle_deg"],
            "S2: EMPCA–OF angle",
            None,
        ),
        (
            axes[1, 0],
            s5["seed"],
            residual_difference,
            "S5: PCA − weighted-PCA residual",
            0.0,
        ),
        (
            axes[1, 1],
            s8["seed"],
            s8["mean_chi2_per_dof"],
            "S8: residual χ² / dof",
            1.0,
        ),
    ]
    for index, (axis, x, y, title, target) in enumerate(panels):
        axis.plot(
            x,
            y,
            color=OKABE_ITO["blue"],
            marker="o",
            markersize=3.5,
            linewidth=1,
        )
        if target is not None:
            axis.axhline(
                target,
                color=OKABE_ITO["black"],
                linestyle="--",
                linewidth=0.9,
            )
        axis.set_xlabel("Seed")
        axis.set_title(title)
        axis.text(
            -0.14,
            1.05,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    axes[0, 0].set_ylabel("Ratio")
    axes[0, 1].set_ylabel("Angle (degrees)")
    axes[1, 0].set_ylabel("Weighted residual difference")
    axes[1, 1].set_ylabel("Ratio")
    fig.tight_layout()
    return fig


def plot_gwosc_null_calibration() -> plt.Figure:
    """Plot random and chronological held-out calibration ratios."""
    apply_publication_style()
    table = pd.read_csv(DERIVED_ROOT / "gwosc_null_calibration.csv")
    kinds = ["random", "chronological_block"]
    titles = ["Random held-out splits", "Chronological held-out blocks"]
    colors = {"H1": OKABE_ITO["blue"], "L1": OKABE_ITO["orange"]}
    markers = {"H1": "o", "L1": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for panel, (axis, kind, title) in enumerate(zip(axes, kinds, titles)):
        subset = table[table["split_kind"] == kind]
        axis.axhspan(0.5, 1.5, color=OKABE_ITO["green"], alpha=0.12)
        axis.axhspan(0.8, 1.2, color=OKABE_ITO["green"], alpha=0.18)
        axis.axhline(1.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
        for detector, offset in (("H1", -0.08), ("L1", 0.08)):
            detector_rows = subset[subset["detector"] == detector].reset_index(
                drop=True
            )
            x = np.arange(len(detector_rows), dtype=float) + offset
            axis.plot(
                x,
                detector_rows["null_sigma_over_predicted"],
                color=colors[detector],
                marker=markers[detector],
                markersize=4,
                linewidth=1,
                label=detector,
            )
        axis.set_xticks(np.arange(5))
        if kind == "random":
            axis.set_xticklabels(["150914", "150915", "150916", "150917", "150918"])
            axis.set_xlabel("Split seed")
        else:
            axis.set_xticklabels(["0", "1", "2", "3", "4"])
            axis.set_xlabel("Chronological block")
        axis.set_title(title)
        axis.text(
            -0.12,
            1.05,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    axes[0].set_ylabel("Observed / predicted null σ")
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, max(11.0, table["null_sigma_over_predicted"].max() * 1.05))
    fig.tight_layout()
    return fig


def plot_gwosc_reference_comparison() -> plt.Figure:
    """Plot PSD equality and diagnostic score-path correlations."""
    apply_publication_style()
    table = pd.read_csv(DERIVED_ROOT / "gwosc_reference_summary.csv")
    detectors = table["detector"].tolist()
    x = np.arange(len(detectors))
    width = 0.23

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].bar(
        x,
        table["psd_ratio_median"],
        color=[OKABE_ITO["blue"], OKABE_ITO["orange"]],
        width=0.55,
    )
    axes[0].axhline(1.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x, detectors)
    axes[0].set_ylim(0.98, 1.02)
    axes[0].set_ylabel("Repository / GWpy PSD")
    axes[0].set_title("Identical median-Welch PSD")

    correlation_columns = [
        ("corr_gls_vs_repository_whitened", "GLS vs direct whitened"),
        ("corr_gls_vs_gwpy_fir", "GLS vs GWpy FIR"),
        (
            "corr_repository_whitened_vs_gwpy_fir",
            "Direct whitened vs GWpy FIR",
        ),
    ]
    colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["purple"],
    ]
    for index, ((column, label), color) in enumerate(
        zip(correlation_columns, colors)
    ):
        axes[1].bar(
            x + (index - 1) * width,
            table[column],
            width=width,
            label=label,
            color=color,
        )
    axes[1].set_xticks(x, detectors)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Score correlation")
    axes[1].set_title("Earlier non-identical score paths")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
    )
    for index, axis in enumerate(axes):
        axis.text(
            -0.12,
            1.05,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.tight_layout()
    return fig


def plot_gwosc_run_history() -> plt.Figure:
    """Plot the archived remote run audit trail."""
    apply_publication_style()
    table = pd.read_csv(DERIVED_ROOT / "gwosc_run_history.csv")
    columns = ["Stage 0", "GWOSC gate", "Filter", "Local PSD"]
    status_text: list[list[str]] = []
    status_code: list[list[int]] = []
    for row in table.itertuples():
        stage = "pass" if bool(row.stage0_accepted) else "fail"
        if row.gwosc_status == "failed_acceptance":
            gwosc = "negative"
        elif row.gwosc_status in {"accepted", "passed"}:
            gwosc = "pass"
        elif row.gwosc_status == "not_run":
            gwosc = "not run"
        else:
            gwosc = str(row.gwosc_status)
        filter_status = "present" if bool(row.has_filter_equivalence) else "pending"
        local_status = "present" if bool(row.has_time_local_noise) else "pending"
        row_text = [stage, gwosc, filter_status, local_status]
        status_text.append(row_text)
        code_row = []
        for value in row_text:
            if value in {"pass", "present"}:
                code_row.append(2)
            elif value in {"negative", "fail"}:
                code_row.append(1)
            else:
                code_row.append(0)
        status_code.append(code_row)

    colors = [
        (0.88, 0.88, 0.88, 1.0),
        (0.93, 0.52, 0.30, 1.0),
        (0.37, 0.70, 0.48, 1.0),
    ]
    cmap = ListedColormap(colors)
    fig, axis = plt.subplots(figsize=(7.2, 2.8))
    axis.imshow(np.asarray(status_code), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axis.set_xticks(np.arange(len(columns)), columns)
    axis.set_yticks(
        np.arange(len(table)),
        [str(run_id).replace("20260622T", "") for run_id in table["run_id"]],
    )
    axis.set_xlabel("Archived evidence component")
    axis.set_ylabel("Run ID")
    axis.set_title("GWOSC evidence audit trail")
    for i, row in enumerate(status_text):
        for j, value in enumerate(row):
            axis.text(j, i, value, ha="center", va="center", fontsize=7)
    fig.tight_layout()
    return fig


def plot_paper_claim_support_matrix() -> plt.Figure:
    """Plot writing-constraint status for each paper claim area."""
    apply_publication_style()
    table = pd.read_csv(DERIVED_ROOT / "claim_status.csv")
    state_order = [
        "verified",
        "verified_negative",
        "implemented_pending_remote",
        "not_validated",
    ]
    colors = {
        "verified": OKABE_ITO["green"],
        "verified_negative": OKABE_ITO["orange"],
        "implemented_pending_remote": OKABE_ITO["sky"],
        "not_validated": OKABE_ITO["gray"],
    }
    y = np.arange(len(table))
    fig, axis = plt.subplots(figsize=(7.2, 3.6))
    for state_index, state in enumerate(state_order):
        subset = table["evidence_state"] == state
        axis.scatter(
            np.full(subset.sum(), state_index),
            y[subset.to_numpy()],
            s=90,
            color=colors[state],
            edgecolor=OKABE_ITO["black"],
            linewidth=0.4,
            label=state.replace("_", " "),
        )
    axis.set_xticks(np.arange(len(state_order)), [s.replace("_", "\n") for s in state_order])
    axis.set_yticks(y, table["topic"])
    axis.invert_yaxis()
    axis.set_xlim(-0.5, len(state_order) - 0.5)
    axis.set_title("Paper claim support state")
    axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    fig.tight_layout()
    return fig


def _metrics(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("metrics", record)


def plot_filter_equivalence() -> plt.Figure | None:
    """Plot identity errors and GLS-to-FIR correlations when evidence exists."""
    record = followup_record("filter_equivalence.json")
    if record is None:
        return None
    apply_publication_style()
    metrics = _metrics(record)
    groups = {
        "Synthetic": metrics["synthetic_control"]["sweep"],
        **{
            detector: detector_metrics["sweep"]
            for detector, detector_metrics in metrics["real_data"].items()
        },
    }
    fig, axes = plt.subplots(
        len(groups),
        2,
        figsize=(7.2, 2.35 * len(groups)),
        squeeze=False,
    )
    for row, (label, sweep) in enumerate(groups.items()):
        durations = sorted({item["fduration_seconds"] for item in sweep})
        trims = sorted({item["edge_trim_seconds"] for item in sweep})
        identity = np.full((len(durations), len(trims)), np.nan)
        correlation = np.full_like(identity, np.nan)
        for item in sweep:
            i = durations.index(item["fduration_seconds"])
            j = trims.index(item["edge_trim_seconds"])
            identity[i, j] = max(
                item["identity"]["max_abs_score_difference"],
                np.finfo(float).tiny,
            )
            correlation[i, j] = item["original_gls_vs_shared_fir"][
                "correlation"
            ]
        image0 = axes[row, 0].imshow(
            np.log10(identity),
            aspect="auto",
            cmap="cividis",
        )
        image1 = axes[row, 1].imshow(
            correlation,
            aspect="auto",
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )
        for axis in axes[row]:
            axis.set_xticks(np.arange(len(trims)), trims)
            axis.set_yticks(np.arange(len(durations)), durations)
            axis.set_xlabel("Edge trim per side (s)")
            axis.set_ylabel("FIR duration (s)")
        axes[row, 0].set_title(f"{label}: log10 identity error")
        axes[row, 1].set_title(f"{label}: original GLS/FIR correlation")
        fig.colorbar(image0, ax=axes[row, 0], fraction=0.046)
        fig.colorbar(image1, ax=axes[row, 1], fraction=0.046)
    fig.tight_layout()
    return fig


def plot_time_local_psd() -> plt.Figure | None:
    """Plot global/local score spread and primary chronological blocks."""
    record = followup_record("time_local_noise.json")
    if record is None:
        return None
    apply_publication_style()
    metrics = _metrics(record)
    groups = {
        "Synthetic": metrics["synthetic_control"],
        **metrics["real_data"],
    }
    model_order = [
        "global_leave_one_out",
        "local_radius_32s",
        "local_radius_64s",
        "local_radius_96s",
    ]
    labels = ["Global", "Local 32 s", "Local 64 s", "Local 96 s"]
    colors = [
        OKABE_ITO["gray"],
        OKABE_ITO["sky"],
        OKABE_ITO["blue"],
        OKABE_ITO["purple"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    x = np.arange(len(groups))
    width = 0.18
    for model_index, (model, label, color) in enumerate(
        zip(model_order, labels, colors)
    ):
        values = [
            group["summaries"][model]["scores"]["std"]
            for group in groups.values()
        ]
        axes[0].bar(
            x + (model_index - 1.5) * width,
            values,
            width=width,
            label=label,
            color=color,
        )
    axes[0].axhspan(0.8, 1.2, color=OKABE_ITO["green"], alpha=0.16)
    axes[0].axhline(1.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
    axes[0].set_xticks(x, list(groups))
    axes[0].set_ylabel("Held-out normalized-score std")
    axes[0].set_title("Global versus local PSD calibration")
    axes[0].legend(frameon=False, ncol=2)

    primary = "local_radius_64s"
    for label, group in groups.items():
        blocks = group["summaries"][primary]["chronological_blocks"]
        axes[1].plot(
            [block["block_id"] for block in blocks],
            [block["score_std"] for block in blocks],
            marker="o",
            linewidth=1,
            label=label,
        )
    axes[1].axhspan(0.8, 1.2, color=OKABE_ITO["green"], alpha=0.16)
    axes[1].axhline(1.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("Chronological block")
    axes[1].set_ylabel("Primary local score std")
    axes[1].set_title("64-second primary model by time block")
    axes[1].legend(frameon=False)
    for index, axis in enumerate(axes):
        axis.text(
            -0.12,
            1.05,
            chr(ord("A") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
        )
    fig.tight_layout()
    return fig
