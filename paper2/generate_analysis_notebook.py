#!/usr/bin/env python3
"""Generate a notebook for artifact-backed Paper 2 analysis.

This notebook is meant to be the human-facing layer on top of
``paper2.analysis.reporting``:

- choose a concrete ``paper2/results`` tree
- optionally re-evaluate checkpoints
- inspect run inventory and best-run summaries
- render paper/talk figures from actual model outputs
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "paper2" / "analysis" / "real_run_analysis.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


COMMON_SETUP = """
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
while not (ROOT / "paper2" / "__init__.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for module_name in list(sys.modules):
    if module_name == "paper2" or module_name.startswith("paper2."):
        del sys.modules[module_name]

from paper2.analysis.reporting import analyze_results_tree, discover_runs
"""


PARAMETERS = """
# Edit these paths before running on the server if needed.
RESULTS_DIR = Path("/ceph/dwong/noise-weighted-subspace-reconstruction/paper2/results")
OUTPUT_DIR = RESULTS_DIR / "_analysis" / "latest"
TEMPLATE_PATH = ROOT / "data" / "k_alpha" / "template_K_alpha_tight.npy"

# Use "checkpoints" to regenerate predictions/metrics from checkpoint_best.pt.
# Use "metrics-only" if you only want to reuse existing metrics.json / analysis_metrics.json.
EVALUATE_MODE = "checkpoints"
FORCE_EVAL = False
OF_MODE = "shifted"
SAMPLING_FREQUENCY = 2.5e5

# Optional filter: None or a list like ["transformer_prewhite_mahalanobis_l40s_muon01"]
ONLY = None
"""


def build_notebook() -> nbf.NotebookNode:
    return nbf.v4.new_notebook(
        cells=[
            md(
                """
                # Paper 2 Real-Run Analysis

                This notebook analyzes actual `paper2/results/<run>/` artifacts from AE / transformer training.

                It is designed to replace the old readiness-only workflow with a checkpoint-backed path:

                1. discover real run folders;
                2. optionally evaluate `checkpoint_best.pt` on the held-out test split;
                3. write `predictions_test.h5` and `analysis_metrics.json` into each run folder;
                4. aggregate the runs into comparison tables and figures for the paper and talk.
                """
            ),
            code(COMMON_SETUP),
            code(PARAMETERS),
            md(
                """
                ## Run Inventory

                This first pass only inspects the results tree. It tells you which runs have checkpoints,
                configs, metrics, and analysis artifacts available.
                """
            ),
            code(
                """
                runs = discover_runs(RESULTS_DIR)
                inventory_df = pd.DataFrame(
                    [
                        {
                            "experiment_name": run.experiment_name,
                            "model_family": run.model_family,
                            "input_mode": run.input_mode,
                            "loss_mode": run.loss_mode,
                            "has_checkpoint": run.checkpoint_path is not None,
                            "has_metrics_json": run.metrics_path is not None,
                            "has_analysis_metrics": run.analysis_metrics_path is not None,
                            "run_dir": str(run.run_dir),
                        }
                        for run in runs
                    ]
                )
                display(inventory_df.sort_values(["model_family", "experiment_name"]).reset_index(drop=True))
                print(f"discovered {len(runs)} run(s)")
                """
            ),
            md(
                """
                ## Execute Analysis

                This cell is the real entry point. When `EVALUATE_MODE="checkpoints"`, it loads
                `checkpoint_best.pt`, runs evaluation on the test split, computes actual reconstruction
                metrics, derives OF-based amplitude/time metrics from reconstructed traces, and writes
                aggregate paper/talk outputs under `OUTPUT_DIR`.
                """
            ),
            code(
                """
                analyzed_runs, analysis_paths = analyze_results_tree(
                    results_dir=RESULTS_DIR,
                    output_dir=OUTPUT_DIR,
                    evaluate_mode=EVALUATE_MODE,
                    force_eval=FORCE_EVAL,
                    template_path=TEMPLATE_PATH,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    of_mode=OF_MODE,
                    only=ONLY,
                )

                print(f"summary table: {analysis_paths.summary_csv}")
                print(f"best-run table: {analysis_paths.best_csv}")
                print(f"figures dir: {analysis_paths.figures_dir}")
                print(f"manifest: {analysis_paths.manifest_json}")
                """
            ),
            md(
                """
                ## Summary Tables

                `run_summary.csv` includes every discovered run. `best_by_logical_group.csv` picks the best
                available run per logical configuration, which is usually what you want for paper figures.
                """
            ),
            code(
                """
                summary_df = pd.read_csv(analysis_paths.summary_csv)
                best_df = pd.read_csv(analysis_paths.best_csv)

                display(summary_df.sort_values(["phase", "model_family", "experiment_name"]).reset_index(drop=True))
                display(best_df.sort_values(["phase", "model_family", "experiment_name"]).reset_index(drop=True))
                """
            ),
            md(
                """
                ## 2x2 and Architecture Figures

                These figures come from actual run outputs, not from placeholder readiness tables.
                """
            ),
            code(
                """
                figure_names = [
                    "ae_2x2_actual.png",
                    "transformer_2x2_actual.png",
                    "architecture_actual.png",
                    "learning_curves_actual.png",
                ]

                for name in figure_names:
                    path = analysis_paths.figures_dir / name
                    print(path)
                    if path.exists():
                        display(Image(filename=str(path)))
                    else:
                        print("missing")
                """
            ),
            md(
                """
                ## Best-Run Metrics for Slides

                This view is optimized for quickly copying headline numbers into slides or a draft.
                """
            ),
            code(
                """
                slide_cols = [
                    "experiment_name",
                    "phase",
                    "model_family",
                    "input_mode",
                    "loss_mode",
                    "optimizer",
                    "eval_loss",
                    "weighted_residual_mean",
                    "reconstruction_mse",
                    "amplitude_rmse",
                    "timing_rmse",
                ]
                display(
                    best_df[slide_cols]
                    .sort_values(["phase", "model_family", "input_mode", "loss_mode"])
                    .reset_index(drop=True)
                )
                """
            ),
            md(
                """
                ## Focused Comparisons

                These slices are useful for talk framing:

                - geometry effect inside the transformer 2x2;
                - best AE vs best transformer;
                - architecture-bias comparison at fixed prewhitened + Mahalanobis geometry.
                """
            ),
            code(
                """
                transformer_2x2 = best_df[best_df["model_family"] == "transformer"].copy()
                ae_2x2 = best_df[best_df["model_family"] == "ae"].copy()
                architecture = best_df[best_df["phase"] == "architecture"].copy()

                print("Transformer 2x2")
                display(transformer_2x2.sort_values(["input_mode", "loss_mode"]).reset_index(drop=True))

                print("AE 2x2")
                display(ae_2x2.sort_values(["input_mode", "loss_mode"]).reset_index(drop=True))

                print("Architecture comparison")
                display(architecture.sort_values(["model_family", "experiment_name"]).reset_index(drop=True))
                """
            ),
            md(
                """
                ## Notes

                - `metrics.json` is training-time best-run summary.
                - `analysis_metrics.json` is the new checkpoint-backed offline evaluation summary.
                - `predictions_test.h5` stores concrete held-out outputs such as `x_true`, `x_hat`, `z`,
                  metadata, and OF-derived amplitude/time predictions when enabled.
                """
            ),
        ],
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def main() -> int:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"[paper2-analysis-notebook] wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
