from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf
from nbclient import NotebookClient

from npml_support import NPML_DIR, ensure_npml_dirs


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


COMMON_SETUP = """
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npml_support import *

dirs = ensure_npml_dirs()
"""


def build_notebooks() -> dict[str, nbf.NotebookNode]:
    notebooks: dict[str, nbf.NotebookNode] = {}

    notebooks["00_model_inventory_and_gap_map.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # NPML Model Inventory and Gap Map

                This notebook maps the five NPML talk experiments from the PDF to the code currently present in `src/`
                and to the minimum additional work needed to run the architecture-heavy experiments honestly.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                plan_df = pdf_experiment_plan()
                inventory_df = audit_src_models()
                needs_df = experiment_need_table()

                display(plan_df)
                display(inventory_df)
                display(needs_df)

                save_dataframe(plan_df, "00_pdf_experiment_plan.csv")
                save_dataframe(inventory_df, "00_src_model_inventory.csv")
                save_dataframe(needs_df, "00_needed_work.csv")
                """
            ),
            code(
                """
                fig = plot_readiness(inventory_df, "NPML experiment readiness from current src/ inventory")
                save_figure(fig, "00_readiness_overview.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["01_experiment_a_metric_ablation.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Experiment A: Metric Ablation

                `EMPCA` here is the weighted optimum from the existing EMPCA / weighted-SVD pipeline.
                `PCA` is the isotropic baseline. The notebook uses the actual repo simulator helpers and saves slide-ready figures.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                out = run_metric_ablation()
                metric_df = out["metric_df"]
                angle_df = out["angle_df"]

                display(metric_df)
                display(angle_df)

                save_dataframe(metric_df, "01_metric_ablation_metrics.csv")
                save_dataframe(angle_df, "01_metric_ablation_angles.csv")
                """
            ),
            code(
                """
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

                resid = metric_df.pivot(index="noise_type", columns="method", values="weighted_residual_mean")
                resid.plot(kind="bar", ax=axes[0], color=["#2b8a3e", "#c92a2a"])
                axes[0].set_title("Weighted residual")
                axes[0].set_ylabel("mean held-out chi2 proxy")
                axes[0].legend(title="")

                mse = metric_df.pivot(index="noise_type", columns="method", values="reconstruction_mse_clean")
                mse.plot(kind="bar", ax=axes[1], color=["#2b8a3e", "#c92a2a"])
                axes[1].set_title("Reconstruction MSE")
                axes[1].set_ylabel("MSE to clean trace")
                axes[1].legend(title="")

                amp = metric_df.pivot(index="noise_type", columns="method", values="amplitude_rmse")
                amp.plot(kind="bar", ax=axes[2], color=["#2b8a3e", "#c92a2a"])
                axes[2].set_title("Amplitude RMSE")
                axes[2].set_ylabel("RMSE")
                axes[2].legend(title="")

                save_figure(fig, "01_metric_ablation_triptych.png")
                plt.show()
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6.2, 4.0))
                ax.plot(angle_df["noise_type"], angle_df["mean_principal_angle_deg"], marker="o", lw=2.0, color="#1c7ed6")
                ax.set_title("PCA vs EMPCA subspace angle")
                ax.set_ylabel("mean principal angle [deg]")
                ax.set_xlabel("noise type")
                save_figure(fig, "01_metric_ablation_subspace_angle.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["02_experiment_b_coverage_ablation.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Experiment B: Coverage Ablation

                This notebook uses a talk-oriented synthetic proxy family with four latent factors:
                amplitude, timing shift, position-like distortion, and shape-like distortion.
                The position latent is synthetic because the current repo simulator does not expose detector position directly.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                out = run_coverage_ablation()
                coverage_df = out["coverage_df"]
                matrix_df = out["matrix_df"]

                display(coverage_df)
                display(matrix_df)

                save_dataframe(coverage_df, "02_coverage_ablation_metrics.csv")
                save_dataframe(matrix_df, "02_coverage_ablation_matrix.csv")
                """
            ),
            code(
                """
                metric_cols = ["amplitude_rmse", "timing_rmse", "position_rmse", "shape_rmse"]
                ratio_df = coverage_df.copy()
                full_empca = (
                    coverage_df[(coverage_df["training_condition"] == "full") & (coverage_df["method"] == "EMPCA")]
                    .iloc[0]
                )
                for col in metric_cols:
                    ratio_df[col] = ratio_df[col] / max(full_empca[col], 1e-12)

                empca_only = ratio_df[ratio_df["method"] == "EMPCA"].set_index("training_condition")[metric_cols]

                fig, ax = plt.subplots(figsize=(7.5, 3.8))
                im = ax.imshow(empca_only.to_numpy(), cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(len(metric_cols)))
                ax.set_xticklabels(metric_cols, rotation=20, ha="right")
                ax.set_yticks(range(len(empca_only.index)))
                ax.set_yticklabels(empca_only.index)
                ax.set_title("Coverage failure when latent directions are omitted")
                for i in range(empca_only.shape[0]):
                    for j in range(empca_only.shape[1]):
                        ax.text(j, i, f"{empca_only.iat[i, j]:.2f}x", ha="center", va="center", color="black")
                fig.colorbar(im, ax=ax, shrink=0.8, label="relative error vs full-coverage EMPCA")
                save_figure(fig, "02_coverage_ablation_heatmap.png")
                plt.show()
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(7.5, 4.2))
                resid_plot = coverage_df.pivot(index="training_condition", columns="method", values="weighted_residual_mean")
                resid_plot.plot(kind="bar", ax=ax, color=["#2b8a3e", "#c92a2a"])
                ax.set_title("Weighted residual under full vs restricted coverage")
                ax.set_ylabel("mean held-out chi2 proxy")
                ax.legend(title="")
                save_figure(fig, "02_coverage_ablation_weighted_residual.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["03_experiment_c_nfpa_vs_empca.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Experiment C: NFPA vs EMPCA

                This is a callable, notebook-safe version of the ideas in `src/NFPA/nfpa_demo.py`.
                It sweeps separable to non-separable signal regimes and measures how gracefully NFPA degrades relative to exact EMPCA.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                out = run_nfpa_regime_sweep()
                nfpa_df = out["nfpa_df"]

                display(nfpa_df)
                save_dataframe(nfpa_df, "03_nfpa_vs_empca_metrics.csv")
                """
            ),
            code(
                """
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))

                nfpa_df.plot(x="regime", y=["empca_weighted_residual", "nfpa_weighted_residual"], kind="bar", ax=axes[0], color=["#495057", "#1c7ed6"])
                axes[0].set_title("Weighted residual")
                axes[0].set_ylabel("mean whitened residual")
                axes[0].legend(["EMPCA", "NFPA"], title="")

                nfpa_df.plot(x="regime", y=["empca_reconstruction_mse", "nfpa_reconstruction_mse"], kind="bar", ax=axes[1], color=["#495057", "#1c7ed6"])
                axes[1].set_title("Reconstruction error")
                axes[1].set_ylabel("MSE to clean trace")
                axes[1].legend(["EMPCA", "NFPA"], title="")

                axes[2].plot(nfpa_df["regime"], nfpa_df["mean_principal_angle_deg"], marker="o", lw=2.0, color="#d9480f")
                axes[2].set_title("NFPA vs EMPCA subspace angle")
                axes[2].set_ylabel("mean principal angle [deg]")
                axes[2].set_xlabel("regime")

                save_figure(fig, "03_nfpa_vs_empca_triptych.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["04_experiment_d_architecture_bias_readiness.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Experiment D: Architecture Bias Readiness

                The current repository contains relevant backbones, but not a runnable reconstruction training stack for this experiment.
                This notebook makes that explicit so the talk can separate completed evidence from code still needed.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                status_df = experiment_d_status_table()
                inventory_df = audit_src_models()
                display(status_df)
                display(inventory_df[inventory_df["experiment"].isin(["D", "D/E"])])

                save_dataframe(status_df, "04_architecture_bias_status.csv")
                """
            ),
            code(
                """
                plot_df = inventory_df[inventory_df["experiment"].isin(["D", "D/E"])].copy()
                fig = plot_readiness(plot_df, "Experiment D/E code readiness")
                save_figure(fig, "04_architecture_bias_readiness.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["05_experiment_e_prewhitened_transformer_readiness.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Experiment E: Prewhitened Transformer Readiness

                This notebook audits the transformer files under `src/transformer/` and maps them to the four configurations requested in the NPML PDF.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                status_df = experiment_e_status_table()
                import_df = parse_transformer_imports()

                display(status_df)
                display(import_df)

                save_dataframe(status_df, "05_prewhitened_transformer_status.csv")
                save_dataframe(import_df, "05_transformer_import_audit.csv")
                """
            ),
            code(
                """
                fig = plot_experiment_e_coverage(status_df)
                save_figure(fig, "05_prewhitened_transformer_readiness.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    notebooks["06_final_2x2_matrix.ipynb"] = nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Final 2x2 Figure

                This notebook turns the metric and coverage results into the central matrix proposed in the NPML PDF:
                correct vs wrong metric crossed with full vs restricted coverage.
                """
            ),
            code(COMMON_SETUP),
            code(
                """
                matrix_df = pd.read_csv(TABLE_DIR / "02_coverage_ablation_matrix.csv")
                display(matrix_df)
                """
            ),
            code(
                """
                cols = [
                    "correct_metric_full_coverage",
                    "wrong_metric_full_coverage",
                    "correct_metric_restricted_coverage",
                    "wrong_metric_restricted_coverage",
                ]
                summary = matrix_df[cols].mean(axis=0)
                grid = pd.DataFrame(
                    [
                        [summary["correct_metric_full_coverage"], summary["correct_metric_restricted_coverage"]],
                        [summary["wrong_metric_full_coverage"], summary["wrong_metric_restricted_coverage"]],
                    ],
                    index=["Correct metric", "Wrong metric"],
                    columns=["Full coverage", "Restricted coverage"],
                )

                fig, ax = plt.subplots(figsize=(6.2, 5.2))
                im = ax.imshow(grid.to_numpy(), cmap="YlOrRd", aspect="auto")
                ax.set_xticks(range(2))
                ax.set_xticklabels(grid.columns)
                ax.set_yticks(range(2))
                ax.set_yticklabels(grid.index)
                ax.set_title("Central NPML matrix\\n(relative error score; lower is better)")
                labels = {
                    (0, 0): "Best case",
                    (1, 0): "Metric failure",
                    (0, 1): "Coverage failure",
                    (1, 1): "Worst case",
                }
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, f"{grid.iat[i, j]:.2f}x\\n{labels[(i, j)]}", ha="center", va="center", color="black")
                fig.colorbar(im, ax=ax, shrink=0.82, label="mean relative score")
                save_figure(fig, "06_final_2x2_matrix.png")
                plt.show()
                plt.close(fig)
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )

    return notebooks


def execute_notebook(path: Path) -> None:
    nb = nbf.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NPML_DIR)}})
    client.execute()
    nbf.write(nb, path)


def main() -> None:
    ensure_npml_dirs()
    notebooks = build_notebooks()
    for name, nb in notebooks.items():
        path = NPML_DIR / name
        nbf.write(nb, path)
    for name in notebooks:
        execute_notebook(NPML_DIR / name)
        print(f"executed {name}")


if __name__ == "__main__":
    main()
