"""Generate and optionally execute the implementation notebook suite."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


COMMON_SETUP = dedent(
    """
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd

    def find_repo_root(start: Path) -> Path:
        for candidate in [start, *start.parents]:
            if (candidate / "plan" / "experiment_checklist.md").exists() and (candidate / "implementation").exists():
                return candidate
        raise RuntimeError("Could not locate repo root for notebook execution.")

    REPO_ROOT = find_repo_root(Path.cwd().resolve())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from implementation.notebook_support import *

    cfg = CanonicalConfig().validate()
    dirs = ensure_results_dirs(cfg)
    plt.style.use("seaborn-v0_8-whitegrid")
    pd.set_option("display.max_colwidth", 120)
    """
).strip()


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(spec: dict) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md(
            f"""
            # {spec['title']}

            **Goal**  
            {spec['goal']}

            **Checklist alignment**  
            {spec['checklist']}

            **Outputs**  
            - tables under `results/tables/`
            - figures under `results/figures/`
            - manifests under `results/manifests/`
            - executed notebook copy under `results/notebooks/` when this suite is run with `--execute`
            """
        ),
        code(COMMON_SETUP),
        *spec["cells"],
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    return nb


NOTEBOOK_SPECS = [
    {
        "name": "block_01_claim_mapping_and_execution_rules.ipynb",
        "title": "Block 01: Claim Mapping and Execution Rules",
        "goal": "Freeze the experiment-to-claim map against the current paper checklist before numerical work.",
        "checklist": "Maps E1-E12 to regimes, metrics, and artifact destinations.",
        "cells": [
            code(
                """
                claim_map = dataframe_from_claim_map()
                naming_rules = pd.DataFrame(
                    [
                        ("preprocessing", "baseline_mean__rfft__one_sided_psd"),
                        ("simulator_rank1", "QPSimulator aligned arrivals + external stationary noise"),
                        ("simulator_family", "QPSimulator controlled tau/t0/n_QP family"),
                        ("rank_label", "k{integer}"),
                        ("seed_label", "seed_{integer}"),
                        ("result_notebook", "results/notebooks/block_XX_*.ipynb"),
                    ],
                    columns=["name", "value"],
                )
                display(claim_map)
                display(naming_rules)
                """
            ),
            code(
                """
                claim_map_path = dirs["tables"] / "block_01_claim_map.csv"
                rules_path = dirs["manifests"] / "block_01_execution_rules.json"
                save_dataframe(claim_map, claim_map_path)
                save_json(
                    {
                        "canonical_config": config_as_frame(cfg).to_dict(orient="records"),
                        "naming_rules": naming_rules.to_dict(orient="records"),
                    },
                    rules_path,
                )
                pd.DataFrame(
                    [
                        manifest_row("block_01", "governance", str(claim_map_path.relative_to(REPO_ROOT)), cfg),
                        manifest_row("block_01", "governance", str(rules_path.relative_to(REPO_ROOT)), cfg),
                    ]
                )
                """
            ),
        ],
    },
    {
        "name": "block_02_data_preprocessing_and_psd_audit.ipynb",
        "title": "Block 02: Data, Preprocessing, and PSD Audit",
        "goal": "Freeze the real-data analysis contract used by later K-alpha and mixed real/simulation studies.",
        "checklist": "Supports the shared preprocessing assumptions behind E5 and E12.",
        "cells": [
            code(
                """
                out = run_block02_audit(cfg)
                bundle = out["bundle"]
                audit_df = out["audit_df"]
                reports_df = out["reports_df"]
                display(audit_df)
                display(reports_df)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                empirical = bundle.metadata["empirical_psd"]
                freqs = bundle.metadata["empirical_psd_freqs"]
                ax.loglog(freqs[1:], bundle.psd_one_sided[1:], label="canonical PSD")
                ax.loglog(freqs[1:], empirical[1:], label="empirical pretrigger PSD")
                ax.set_xlabel("frequency [Hz]")
                ax.set_ylabel("PSD")
                ax.set_title("Real-data PSD audit")
                ax.legend()
                save_figure(fig, dirs["figures"] / "block_02_psd_audit.png")
                plt.close(fig)
                """
            ),
            code(
                """
                audit_path = dirs["tables"] / "block_02_psd_audit.csv"
                reports_path = dirs["tables"] / "block_02_data_sources.csv"
                manifest_path = dirs["manifests"] / "block_02_split_manifest.json"
                save_dataframe(audit_df, audit_path)
                save_dataframe(reports_df, reports_path)
                save_json(out["split_manifest"], manifest_path)
                pd.DataFrame(
                    [
                        manifest_row("block_02", "real-support", str(audit_path.relative_to(REPO_ROOT)), cfg, bundle),
                        manifest_row("block_02", "real-support", str(manifest_path.relative_to(REPO_ROOT)), cfg, bundle),
                    ]
                )
                """
            ),
        ],
    },
    {
        "name": "block_03_of_baseline_crb_and_real_rank1_verification.ipynb",
        "title": "Block 03: OF Baseline, CRB, and Real Rank-1 Verification",
        "goal": "Use the real K-alpha dataset to anchor rank-1 equivalence, bridge diagnostics, and observed-vs-predicted amplitude spread.",
        "checklist": "Implements the real-data side of E5 and the full real verification target E12.",
        "cells": [
            code(
                """
                out = run_block03_real_rank1(cfg)
                summary = pd.Series(out["summary"], name="value").to_frame()
                display(summary)
                display(out["bridge_df"])
                """
            ),
            code(
                """
                amp_df = out["amplitude_df"]
                fig, ax = plt.subplots(figsize=(5.5, 5.5))
                ax.scatter(amp_df["of_gls"], amp_df["empca_rank1"], s=10, alpha=0.5)
                lo = min(amp_df["of_gls"].min(), amp_df["empca_rank1"].min())
                hi = max(amp_df["of_gls"].max(), amp_df["empca_rank1"].max())
                ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls="--")
                ax.set_xlabel("OF amplitude (GLS)")
                ax.set_ylabel("EMPCA rank-1 amplitude")
                ax.set_title("Real K-alpha: OF vs rank-1 EMPCA")
                save_figure(fig, dirs["figures"] / "block_03_real_rank1_scatter.png")
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                vals = amp_df["of_gls"].to_numpy()
                ax.hist(vals, bins=30, density=True, alpha=0.7, label="OF amplitudes")
                mu = vals.mean()
                sigma = vals.std(ddof=1)
                xs = np.linspace(vals.min(), vals.max(), 300)
                ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), color="black", lw=1.5, label="Gaussian fit")
                ax.set_xlabel("amplitude")
                ax.set_ylabel("density")
                ax.set_title("Real K-alpha amplitude histogram")
                ax.legend()
                save_figure(fig, dirs["figures"] / "block_03_real_amplitude_histogram.png")
                plt.close(fig)
                """
            ),
            code(
                """
                summary_path = dirs["tables"] / "block_03_e12_real_summary.csv"
                amp_path = dirs["tables"] / "block_03_real_rank1_amplitudes.csv"
                bridge_path = dirs["tables"] / "block_03_real_bridge.csv"
                manifest_path = dirs["manifests"] / "block_03_real_summary.json"
                save_dataframe(pd.DataFrame([out["summary"]]), summary_path)
                save_dataframe(out["amplitude_df"], amp_path)
                save_dataframe(out["bridge_df"], bridge_path)
                save_json({"summary": out["summary"], "of_stats": out["of_stats"]}, manifest_path)
                pd.DataFrame(
                    [
                        manifest_row("block_03", "real-support", str(summary_path.relative_to(REPO_ROOT)), cfg, out["bundle"]),
                        manifest_row("block_03", "real-support", str(bridge_path.relative_to(REPO_ROOT)), cfg, out["bundle"]),
                    ]
                )
                """
            ),
        ],
    },
    {
        "name": "block_04_synthetic_theorem_regime.ipynb",
        "title": "Block 04: Synthetic Theorem-Regime Verification",
        "goal": "Use controlled QPSimulator families to verify the rank-1 theorem regime, CRB variance law, and energy-resolution scaling cleanly.",
        "checklist": "Implements E1, E4, and the simulation side of E5.",
        "cells": [
            code(
                """
                out = run_block04_theorem_suite(cfg)
                display(out["rank1_summary_df"])
                display(out["crb_df"])
                display(out["resolution_df"])
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                df = out["resolution_df"]
                ax.loglog(df["noise_power"], df["sigma_emp"], marker="o", label="empirical")
                ax.loglog(df["noise_power"], df["sigma_pred"], marker="s", label="predicted")
                ax.set_xlabel("noise power")
                ax.set_ylabel("sigma_E proxy")
                ax.set_title("Simulation energy-resolution scaling")
                ax.legend()
                save_figure(fig, dirs["figures"] / "block_04_e5_resolution_scaling.png")
                plt.close(fig)
                """
            ),
            code(
                """
                rank1_path = dirs["tables"] / "block_04_e1_rank1_summary.csv"
                crb_path = dirs["tables"] / "block_04_e4_crb.csv"
                resolution_path = dirs["tables"] / "block_04_e5_resolution.csv"
                manifest_path = dirs["manifests"] / "block_04_theorem_summary.json"
                save_dataframe(out["rank1_summary_df"], rank1_path)
                save_dataframe(out["crb_df"], crb_path)
                save_dataframe(out["resolution_df"], resolution_path)
                save_json(out["resolution_summary"], manifest_path)
                pd.DataFrame(
                    [
                        manifest_row("block_04", "theorem-support", str(rank1_path.relative_to(REPO_ROOT)), cfg),
                        manifest_row("block_04", "theorem-support", str(crb_path.relative_to(REPO_ROOT)), cfg),
                    ]
                )
                """
            ),
        ],
    },
    {
        "name": "block_05_linear_ae_bridge_and_gauge.ipynb",
        "title": "Block 05: Linear-AE Bridge and Gauge Handling",
        "goal": "Compare EMPCA and the exact weighted-SVD bridge baseline on the same controlled rank-k family.",
        "checklist": "Implements E2.",
        "cells": [
            code(
                """
                out = run_block05_bridge_suite(cfg)
                display(out["bridge_df"])
                """
            ),
            code(
                """
                bridge = out["bridge_df"].copy()
                bridge["min_principal_cosine"] = bridge["principal_angle_cosines"].map(min)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(bridge["k"], bridge["min_principal_cosine"], marker="o")
                ax.set_xlabel("rank k")
                ax.set_ylabel("minimum principal-angle cosine")
                ax.set_ylim(0.0, 1.01)
                ax.set_title("EMPCA vs exact weighted bridge")
                save_figure(fig, dirs["figures"] / "block_05_bridge_cosines.png")
                plt.close(fig)
                """
            ),
            code(
                """
                bridge_path = dirs["tables"] / "block_05_e2_bridge.csv"
                manifest_path = dirs["manifests"] / "block_05_bridge_summary.json"
                save_dataframe(out["bridge_df"], bridge_path)
                save_json({"rows": out["bridge_df"].to_dict(orient="records")}, manifest_path)
                pd.DataFrame([manifest_row("block_05", "theorem-support", str(bridge_path.relative_to(REPO_ROOT)), cfg, out["bundle"])])
                """
            ),
        ],
    },
    {
        "name": "block_06_convergence_initialization_and_rank_selection.ipynb",
        "title": "Block 06: Convergence, Initialization, and Rank Selection",
        "goal": "Quantify rank saturation and EMPCA convergence traces on controlled QPSimulator families.",
        "checklist": "Implements E3 and E9.",
        "cells": [
            code(
                """
                out = run_block06_convergence_suite(cfg)
                display(out["rank_summary_df"])
                display(out["convergence_summary_df"])
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                rank_df = out["rank_summary_df"]
                ax.plot(rank_df["k"], rank_df["chi2_proxy_mean"], marker="o")
                ax.set_xlabel("rank k")
                ax.set_ylabel("held-out weighted residual")
                ax.set_title("Rank saturation on multi-dimensional family")
                save_figure(fig, dirs["figures"] / "block_06_rank_saturation.png")
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                conv = out["convergence_df"]
                for (k, init), group in conv.groupby(["k", "init"]):
                    if k in (1, 2):
                        ax.plot(group["iteration"], group["chi2"], label=f"k={k}, {init}")
                ax.set_xlabel("iteration")
                ax.set_ylabel("chi2 surrogate")
                ax.set_title("EMPCA convergence traces")
                ax.legend()
                save_figure(fig, dirs["figures"] / "block_06_convergence.png")
                plt.close(fig)
                """
            ),
            code(
                """
                mono_path = dirs["tables"] / "block_06_e3_monotonicity.csv"
                conv_path = dirs["tables"] / "block_06_e9_convergence.csv"
                rank_path = dirs["tables"] / "block_06_rank_selection.csv"
                manifest_path = dirs["manifests"] / "block_06_rank_selection_summary.json"
                save_dataframe(out["monotonicity_df"], mono_path)
                save_dataframe(out["convergence_df"], conv_path)
                save_dataframe(out["rank_summary_df"], rank_path)
                save_json({"summary_rows": out["convergence_summary_df"].to_dict(orient="records")}, manifest_path)
                pd.DataFrame([manifest_row("block_06", "theorem-support", str(rank_path.relative_to(REPO_ROOT)), cfg)])
                """
            ),
        ],
    },
    {
        "name": "block_07_rankk_quality_and_noiseaware_ablation.ipynb",
        "title": "Block 07: Rank-k Quality and Noise-Aware Ablation",
        "goal": "Show when rank helps, when isotropic loss fails under colored noise, and how mismatch/jitter change the baseline.",
        "checklist": "Implements E6, E7, and E8.",
        "cells": [
            code(
                """
                out = run_block07_ablation_suite(cfg)
                display(out["ablation_df"])
                display(out["mismatch_df"])
                display(out["shift_df"])
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                df = out["ablation_df"]
                ax.bar(df["noise_type"], df["relative_improvement"])
                ax.set_ylabel("relative improvement")
                ax.set_title("Noise-aware vs isotropic loss")
                save_figure(fig, dirs["figures"] / "block_07_e6_ablation.png")
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                curve = out["mismatch_curve_df"]
                ax.plot(curve["cosine_squared"], curve["of_estimate_for_unit_shape"], marker="o")
                ax.set_xlabel("cos^2(theta_w)")
                ax.set_ylabel("OF estimate for unit-normalized shape")
                ax.set_title("Template mismatch bias curve")
                save_figure(fig, dirs["figures"] / "block_07_e7_mismatch_curve.png")
                plt.close(fig)
                """
            ),
            code(
                """
                ablation_path = dirs["tables"] / "block_07_e6_ablation.csv"
                mismatch_path = dirs["tables"] / "block_07_e7_mismatch.csv"
                mismatch_curve_path = dirs["tables"] / "block_07_e7_bias_curve.csv"
                shift_path = dirs["tables"] / "block_07_e8_time_shift.csv"
                manifest_path = dirs["manifests"] / "block_07_ablation_summary.json"
                save_dataframe(out["ablation_df"], ablation_path)
                save_dataframe(out["mismatch_df"], mismatch_path)
                save_dataframe(out["mismatch_curve_df"], mismatch_curve_path)
                save_dataframe(out["shift_df"], shift_path)
                save_json({"saved": [str(p.relative_to(REPO_ROOT)) for p in [ablation_path, mismatch_path, mismatch_curve_path, shift_path]]}, manifest_path)
                pd.DataFrame([manifest_row("block_07", "theorem-support", str(ablation_path.relative_to(REPO_ROOT)), cfg)])
                """
            ),
        ],
    },
    {
        "name": "block_08_pc_interpretation_and_centering.ipynb",
        "title": "Block 08: PC Interpretation and Centering",
        "goal": "Interpret leading weighted components conservatively and quantify how centering changes those directions.",
        "checklist": "Uses the real K-alpha dataset to support cautious discussion-level interpretation work.",
        "cells": [
            code(
                """
                out = run_block08_pc_suite(cfg)
                display(out["overlap_df"].pivot_table(index=["fit_variant", "component"], columns="proxy", values="weighted_cosine"))
                display(out["corr_df"].pivot_table(index="component", columns="proxy", values="pearson_r"))
                """
            ),
            code(
                """
                overlap_pivot = out["overlap_df"].pivot_table(index=["fit_variant", "component"], columns="proxy", values="weighted_cosine")
                fig, ax = plt.subplots(figsize=(7, 4))
                im = ax.imshow(overlap_pivot.to_numpy(), aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(overlap_pivot.columns)), overlap_pivot.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(overlap_pivot.index)), [f"{a} / {b}" for a, b in overlap_pivot.index])
                ax.set_title("PC overlap matrix")
                fig.colorbar(im, ax=ax, label="weighted cosine")
                save_figure(fig, dirs["figures"] / "block_08_pc_overlap_matrix.png")
                plt.close(fig)
                """
            ),
            code(
                """
                overlap_path = dirs["tables"] / "block_08_pc_metrics.csv"
                corr_path = dirs["tables"] / "block_08_pc_correlations.csv"
                manifest_path = dirs["manifests"] / "block_08_pc_summary.json"
                save_dataframe(out["overlap_df"], overlap_path)
                save_dataframe(out["corr_df"], corr_path)
                save_json(out["summary"], manifest_path)
                pd.DataFrame([manifest_row("block_08", "real-support", str(overlap_path.relative_to(REPO_ROOT)), cfg)])
                """
            ),
        ],
    },
    {
        "name": "block_09_structured_noise_and_multichannel_robustness.ipynb",
        "title": "Block 09: Structured-Noise and Multichannel Robustness",
        "goal": "Stress-test the weighted methods outside the theorem regime without mixing these results into the equivalence claims.",
        "checklist": "Implements E10 and E11, plus a synthetic multichannel characterization note.",
        "cells": [
            code(
                """
                out = run_block09_robustness_suite(cfg)
                display(out["nonstationary_df"])
                display(out["artifact_df"])
                display(out["multichannel_df"])
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                df = out["nonstationary_df"]
                ax.bar(df["case"], df["amplitude_rmse"])
                ax.set_ylabel("amplitude RMSE")
                ax.set_title("Non-stationary PSD handling")
                ax.tick_params(axis="x", rotation=20)
                save_figure(fig, dirs["figures"] / "block_09_e10_nonstationary.png")
                plt.close(fig)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(5.5, 4))
                df = out["artifact_df"]
                ax.bar(df["pass"], df["amplitude_rmse"])
                ax.set_ylabel("amplitude RMSE")
                ax.set_title("Artifact flagging and refit")
                save_figure(fig, dirs["figures"] / "block_09_e11_artifacts.png")
                plt.close(fig)
                """
            ),
            code(
                """
                ns_path = dirs["tables"] / "block_09_e10_nonstationary.csv"
                art_path = dirs["tables"] / "block_09_e11_artifacts.csv"
                multi_path = dirs["tables"] / "block_09_multichannel_summary.csv"
                manifest_path = dirs["manifests"] / "block_09_robustness_summary.json"
                save_dataframe(out["nonstationary_df"], ns_path)
                save_dataframe(out["artifact_df"], art_path)
                save_dataframe(out["multichannel_df"], multi_path)
                save_json({"saved": [str(p.relative_to(REPO_ROOT)) for p in [ns_path, art_path, multi_path]]}, manifest_path)
                pd.DataFrame([manifest_row("block_09", "robustness-support", str(ns_path.relative_to(REPO_ROOT)), cfg)])
                """
            ),
        ],
    },
    {
        "name": "block_10_tables_figures_appendix_and_reproducibility.ipynb",
        "title": "Block 10: Tables, Figures, Appendix, and Reproducibility",
        "goal": "Inventory the generated artifacts and centralize a reproducibility manifest for the notebook suite.",
        "checklist": "Wraps the outputs of Blocks 01-09 into a paper-facing inventory.",
        "cells": [
            code(
                """
                expected = pd.DataFrame(
                    [
                        ("block_01", "results/tables/block_01_claim_map.csv"),
                        ("block_02", "results/manifests/block_02_split_manifest.json"),
                        ("block_03", "results/tables/block_03_e12_real_summary.csv"),
                        ("block_04", "results/tables/block_04_e4_crb.csv"),
                        ("block_05", "results/tables/block_05_e2_bridge.csv"),
                        ("block_06", "results/tables/block_06_e9_convergence.csv"),
                        ("block_07", "results/tables/block_07_e6_ablation.csv"),
                        ("block_08", "results/tables/block_08_pc_metrics.csv"),
                        ("block_09", "results/tables/block_09_e10_nonstationary.csv"),
                    ],
                    columns=["block_id", "artifact_relpath"],
                )
                expected["exists"] = expected["artifact_relpath"].map(lambda p: (REPO_ROOT / p).exists())
                display(expected)
                """
            ),
            code(
                """
                manifest_files = sorted(dirs["manifests"].glob("block_*.json"))
                table_files = sorted(dirs["tables"].glob("block_*.*"))
                figure_files = sorted(dirs["figures"].glob("block_*.*"))
                inventory = pd.DataFrame(
                    [{"kind": "manifest", "path": str(path.relative_to(REPO_ROOT))} for path in manifest_files]
                    + [{"kind": "table", "path": str(path.relative_to(REPO_ROOT))} for path in table_files]
                    + [{"kind": "figure", "path": str(path.relative_to(REPO_ROOT))} for path in figure_files]
                )
                display(inventory)
                """
            ),
            code(
                """
                manifest_path = dirs["manifests"] / "block_10_reproducibility_manifest.json"
                save_json(
                    {
                        "canonical_config": config_as_frame(cfg).to_dict(orient="records"),
                        "expected_artifacts": expected.to_dict(orient="records"),
                        "present_inventory": inventory.to_dict(orient="records"),
                    },
                    manifest_path,
                )
                pd.DataFrame([manifest_row("block_10", "packaging", str(manifest_path.relative_to(REPO_ROOT)), cfg)])
                """
            ),
        ],
    },
]


def write_notebooks() -> list[Path]:
    paths: list[Path] = []
    for spec in NOTEBOOK_SPECS:
        path = ROOT / spec["name"]
        nbf.write(notebook(spec), path)
        paths.append(path)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return paths


def execute_notebooks(paths: list[Path]) -> None:
    from nbclient import NotebookClient

    out_dir = REPO_ROOT / "results/notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        nb = nbf.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        client.execute()
        out_path = out_dir / path.name
        nbf.write(nb, out_path)
        print(f"Executed {path.relative_to(REPO_ROOT)} -> {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute notebooks after generation and save executed copies under results/notebooks/.")
    args = parser.parse_args()

    paths = write_notebooks()
    if args.execute:
        execute_notebooks(paths)


if __name__ == "__main__":
    main()
