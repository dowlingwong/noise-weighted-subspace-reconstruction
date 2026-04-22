"""Generate the 10 Apr 20 implementation notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
SUPPORT_SOURCE = (ROOT / "notebook_support.py").read_text(encoding="utf-8").strip()


COMMON_SETUP = dedent(
    """
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    from IPython.display import display

    def find_repo_root(start: Path) -> Path:
        for candidate in [start, *start.parents]:
            if (candidate / "implementation_blocks_apr20" / "notebook_support.py").exists():
                return candidate
        raise RuntimeError("Could not locate repo root from notebook working directory.")

    REPO_ROOT = find_repo_root(Path.cwd())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    """
).strip()

POST_SUPPORT_SETUP = dedent(
    """
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

            **Support regime**  
            `{spec['regime']}`

            **What this notebook does**
            - {spec['actions'][0]}
            - {spec['actions'][1]}
            - {spec['actions'][2]}

            **Execution note**  
            The cells are written to use local real traces when they exist, and otherwise fall back to small synthetic smoke data so the workflow remains runnable inside this repo.
            """
        ),
        code(COMMON_SETUP),
        code(SUPPORT_SOURCE),
        code(POST_SUPPORT_SETUP),
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
        "goal": "Freeze the claim-to-metric map and the cross-notebook execution rules before any later analysis drifts into inconsistent conventions.",
        "regime": "governance / mixed",
        "actions": [
            "materialize an internal claim map as a dataframe with explicit regimes, metrics, and artifact targets",
            "freeze naming rules, acceptance language, and block ownership",
            "save machine-readable metadata that later notebooks can point back to",
        ],
        "cells": [
            md(
                """
                ## Shared conventions

                This notebook is intentionally lightweight. Its main job is to codify the things that should not be silently redefined later:

                - preprocessing names and defaults;
                - support-regime labels;
                - artifact destinations;
                - acceptance language.
                """
            ),
            code(
                """
                claim_map = dataframe_from_claim_map()
                naming_rules = pd.DataFrame(
                    [
                        ("preprocessing_variant", "baseline_mean__rfft__psd_weighted"),
                        ("weight_variant", "one_sided_of_weight"),
                        ("rank_label", "k{integer}"),
                        ("seed_label", "seed_{integer}"),
                        ("split_label", "train_val_test__seed_fixed"),
                        ("gauge_label", "rank1_phase_sign_aligned_to_reference"),
                        ("smoothing_label", "empca_no_smoothing or empca_savgol"),
                    ],
                    columns=["name", "frozen_value"],
                )
                regime_rules = pd.DataFrame(
                    [
                        ("theorem-support", "Only matched assumptions, stationary Gaussian noise, no out-of-model artifacts."),
                        ("real-support", "Measured or local real traces under the same preprocessing for all compared methods."),
                        ("robustness-support", "Out-of-model perturbations used to quantify degradation, not to prove equivalence."),
                    ],
                    columns=["support_regime", "rule"],
                )
                display(claim_map)
                display(naming_rules)
                display(regime_rules)
                """
            ),
            code(
                """
                assert set(claim_map["support_regime"]).issubset(
                    {"theorem-support", "real-support", "robustness-support"}
                )
                assert claim_map["artifact_path"].is_unique

                claim_map_path = dirs["tables"] / "block_01_claim_map.csv"
                execution_rules_path = dirs["manifests"] / "block_01_execution_rules.json"
                save_dataframe(claim_map, claim_map_path)
                save_json(
                    {
                        "canonical_config": config_as_frame(cfg).to_dict(orient="records"),
                        "naming_rules": naming_rules.to_dict(orient="records"),
                        "regime_rules": regime_rules.to_dict(orient="records"),
                    },
                    execution_rules_path,
                )

                pd.DataFrame(
                    [
                        manifest_row(
                            block_id="block_01",
                            regime="governance",
                            output_path=str(claim_map_path.relative_to(REPO_ROOT)),
                            cfg=cfg,
                            extra={"artifact_type": "claim_map"},
                        ),
                        manifest_row(
                            block_id="block_01",
                            regime="governance",
                            output_path=str(execution_rules_path.relative_to(REPO_ROOT)),
                            cfg=cfg,
                            extra={"artifact_type": "execution_rules"},
                        ),
                    ]
                )
                """
            ),
        ],
    },
    {
        "name": "block_02_data_preprocessing_and_psd_audit.ipynb",
        "title": "Block 02: Data, Preprocessing, and PSD Audit",
        "goal": "Freeze one canonical data interface for traces, PSDs, weights, whitening, and split manifests.",
        "regime": "real-support with synthetic fallback",
        "actions": [
            "load the canonical template, PSD, and either local real traces or synthetic fallback traces",
            "apply the frozen baseline convention and deterministic train/validation/test split",
            "audit PSD/weight consistency and save a split manifest",
        ],
        "cells": [
            md(
                """
                ## Canonical input bundle

                The `DatasetBundle` created below is the shared contract for later notebooks. It separates:

                - raw time traces;
                - baseline-corrected time traces;
                - rFFT-domain traces;
                - one-sided PSD;
                - one-sided OF weights;
                - deterministic split indices.
                """
            ),
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=384)
                config_table = config_as_frame(cfg)
                display(config_table)
                pd.Series({k: v for k, v in bundle.metadata.items() if k != "trace_candidate_reports"}, name="value").to_frame()
                """
            ),
            code(
                """
                trace_candidate_reports = pd.DataFrame(bundle.metadata.get("trace_candidate_reports", []))
                display(trace_candidate_reports if len(trace_candidate_reports) else pd.DataFrame([{"note": "no trace candidate reports"}]))
                """
            ),
            code(
                """
                baseline_before = np.mean(bundle.traces_raw[:, : cfg.pretrigger], axis=1)
                baseline_after = np.mean(bundle.traces_baseline[:, : cfg.pretrigger], axis=1)
                split_manifest = build_split_manifest(bundle, cfg)
                psd_audit = audit_psd(bundle.psd_one_sided, cfg.trace_len, cfg.default_psd_floor_quantile)

                f_emp, psd_emp = calculate_psd(
                    bundle.traces_baseline[bundle.split_indices["train"], : cfg.pretrigger],
                    sampling_frequency=cfg.sampling_frequency,
                )
                audit_df = pd.DataFrame(
                    [
                        ("baseline_before_mean", float(np.mean(baseline_before))),
                        ("baseline_before_std", float(np.std(baseline_before))),
                        ("baseline_after_mean", float(np.mean(baseline_after))),
                        ("baseline_after_std", float(np.std(baseline_after))),
                        ("empirical_pretrigger_psd_mean", float(np.mean(psd_emp))),
                        ("canonical_psd_mean", float(np.mean(bundle.psd_one_sided))),
                        ("weight_zero_bin", float(bundle.weights_one_sided[0])),
                    ],
                    columns=["metric", "value"],
                )
                display(audit_df)
                display(pd.Series(psd_audit, name="value").to_frame())
                """
            ),
            code(
                """
                split_manifest_path = dirs["manifests"] / "block_02_split_manifest.json"
                psd_audit_path = dirs["tables"] / "block_02_psd_audit.csv"
                save_json(split_manifest | {"psd_audit": psd_audit}, split_manifest_path)
                save_dataframe(audit_df, psd_audit_path)

                manifest_df = pd.DataFrame(
                    [
                        manifest_row("block_02", "real-support", str(split_manifest_path.relative_to(REPO_ROOT)), cfg, bundle, {"artifact_type": "split_manifest"}),
                        manifest_row("block_02", "real-support", str(psd_audit_path.relative_to(REPO_ROOT)), cfg, bundle, {"artifact_type": "psd_audit"}),
                    ]
                )
                display(manifest_df)
                """
            ),
        ],
    },
    {
        "name": "block_03_of_baseline_crb_and_real_rank1_verification.ipynb",
        "title": "Block 03: OF Baseline, CRB, and Real Rank-1 Verification",
        "goal": "Establish OF as the rank-1 baseline statistically and compare it to rank-1 EMPCA under matched preprocessing.",
        "regime": "real-support with synthetic fallback",
        "actions": [
            "compute OF normalization, Fisher information, and CRB-style amplitude variance",
            "fit strict rank-1 EMPCA without smoothing on the canonical training split",
            "compare OF and EMPCA amplitudes, residual energies, and rank-1 direction overlap on held-out traces",
        ],
        "cells": [
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=512)
                train_idx = bundle.split_indices["train"]
                test_idx = bundle.split_indices["test"]

                of_stats = compute_of_statistics(bundle.template_freq, bundle.weights_one_sided)
                of_stats
                """
            ),
            code(
                """
                emp_fit = fit_weighted_empca(
                    bundle.traces_freq[train_idx],
                    bundle.weights_one_sided,
                    k=1,
                    n_iter=cfg.default_empca_iter,
                    patience=cfg.default_empca_patience,
                    smoothing=False,
                    init="template",
                    template_f=bundle.template_freq,
                    seed=cfg.seed,
                )
                align = weighted_rank1_alignment(emp_fit["basis"][0], bundle.template_freq, bundle.weights_one_sided)
                u_emp = align["basis_aligned"]
                cosine = align["weighted_cosine"]
                cosine
                """
            ),
            code(
                """
                of_amp_train = rankk_gls_coefficients(
                    bundle.traces_freq[train_idx], bundle.template_freq, bundle.weights_one_sided, return_complex=False
                ).reshape(-1)
                emp_amp_train = rankk_gls_coefficients(
                    bundle.traces_freq[train_idx], u_emp, bundle.weights_one_sided, return_complex=False
                ).reshape(-1)
                scale = float(np.dot(of_amp_train, emp_amp_train) / np.dot(emp_amp_train, emp_amp_train))

                of_amp_test_gls = rankk_gls_coefficients(
                    bundle.traces_freq[test_idx], bundle.template_freq, bundle.weights_one_sided, return_complex=False
                ).reshape(-1)
                of_amp_test_td = compute_of_amplitudes(
                    bundle.traces_baseline[test_idx], bundle.template_time, bundle.psd_one_sided, cfg.sampling_frequency
                )
                emp_amp_test = rankk_gls_coefficients(
                    bundle.traces_freq[test_idx], u_emp, bundle.weights_one_sided, return_complex=False
                ).reshape(-1)
                emp_amp_test_scaled = scale * emp_amp_test

                resid_of = residual_energy_per_trace(
                    bundle.traces_freq[test_idx], bundle.template_freq, of_amp_test_gls, bundle.weights_one_sided
                )
                resid_emp = residual_energy_per_trace(
                    bundle.traces_freq[test_idx], u_emp, emp_amp_test_scaled, bundle.weights_one_sided
                )

                summary = {
                    "source": bundle.metadata["source"],
                    "trace_candidate_reports": bundle.metadata.get("trace_candidate_reports", []),
                    "weighted_cosine_rank1": float(cosine),
                    "median_abs_amplitude_gap": float(np.median(np.abs(of_amp_test_gls - emp_amp_test_scaled))),
                    "median_relative_amplitude_gap": float(
                        np.median(np.abs(of_amp_test_gls - emp_amp_test_scaled) / np.maximum(np.abs(of_amp_test_gls), 1e-12))
                    ),
                    "of_time_vs_gls_max_abs_diff": float(np.max(np.abs(of_amp_test_td - of_amp_test_gls))),
                    "residual_ks": ks_compare(resid_of, resid_emp),
                    "residual_of_mean": float(np.mean(resid_of)),
                    "residual_emp_mean": float(np.mean(resid_emp)),
                    "empca_iterations_used": int(emp_fit["n_iter_used"]),
                }
                pd.Series(summary, name="value").to_frame()
                """
            ),
            code(
                """
                summary_path = dirs["manifests"] / "block_03_of_summary.json"
                table_path = dirs["tables"] / "block_03_rank1_comparison.csv"
                comparison_df = pd.DataFrame(
                    {
                        "of_amp_gls": of_amp_test_gls,
                        "of_amp_td": of_amp_test_td,
                        "emp_amp_scaled": emp_amp_test_scaled,
                        "resid_of": resid_of,
                        "resid_emp": resid_emp,
                    }
                )
                save_json(summary | {"of_statistics": of_stats}, summary_path)
                save_dataframe(comparison_df, table_path)
                display(comparison_df.head())
                """
            ),
        ],
    },
    {
        "name": "block_04_synthetic_theorem_regime.ipynb",
        "title": "Block 04: Synthetic Theorem-Regime Verification",
        "goal": "Run the cleanest matched-assumption synthetic study: planted signal plus stationary Gaussian noise only.",
        "regime": "theorem-support",
        "actions": [
            "generate synthetic rank-1 traces from the local template and measured PSD",
            "sweep event count and SNR-like amplitude scale to study finite-sample recovery",
            "compare planted truth, OF amplitudes, and rank-1 EMPCA under matched weighting",
        ],
        "cells": [
            code(
                """
                template_time, _ = load_template(cfg)
                psd_one_sided, _ = load_psd(cfg)
                sweep_rows = []
                sample_sizes = [64, 128, 256, 512]
                amplitude_ranges = [(0.9, 1.1), (0.7, 1.3)]

                for n_events in sample_sizes:
                    for amp_range in amplitude_ranges:
                        traces, meta, truth = generate_synthetic_rank1_dataset(
                            cfg,
                            n_events=n_events,
                            amplitude_range=amp_range,
                            timing_jitter_std=0.0,
                            width_sigma_range=(0.0, 0.0),
                            psd_one_sided=psd_one_sided,
                            template_time=template_time,
                            return_truth=True,
                        )
                        bundle = prepare_dataset(traces, cfg, template_time=template_time, psd_one_sided=psd_one_sided, metadata=meta)
                        train_idx = bundle.split_indices["train"]
                        test_idx = bundle.split_indices["test"]
                        fit = fit_weighted_empca(
                            bundle.traces_freq[train_idx],
                            bundle.weights_one_sided,
                            k=1,
                            n_iter=60,
                            patience=10,
                            smoothing=False,
                            init="template",
                            template_f=bundle.template_freq,
                            seed=cfg.seed,
                        )
                        aligned = weighted_rank1_alignment(fit["basis"][0], bundle.template_freq, bundle.weights_one_sided)
                        of_amp = rankk_gls_coefficients(bundle.traces_freq[test_idx], bundle.template_freq, bundle.weights_one_sided, return_complex=False).reshape(-1)
                        emp_amp = rankk_gls_coefficients(bundle.traces_freq[test_idx], aligned["basis_aligned"], bundle.weights_one_sided, return_complex=False).reshape(-1)
                        scale = float(np.dot(of_amp, emp_amp) / np.dot(emp_amp, emp_amp))
                        truth_amp = truth["amplitudes"][test_idx]
                        sweep_rows.append(
                            {
                                "n_events": int(n_events),
                                "amp_min": float(amp_range[0]),
                                "amp_max": float(amp_range[1]),
                                "weighted_cosine_template_vs_empca": float(aligned["weighted_cosine"]),
                                "of_truth_bias": float(np.mean(of_amp - truth_amp)),
                                "emp_truth_bias": float(np.mean(scale * emp_amp - truth_amp)),
                                "of_truth_rmse": float(np.sqrt(np.mean((of_amp - truth_amp) ** 2))),
                                "emp_truth_rmse": float(np.sqrt(np.mean((scale * emp_amp - truth_amp) ** 2))),
                            }
                        )

                sweep_df = pd.DataFrame(sweep_rows)
                display(sweep_df)
                """
            ),
            code(
                """
                theorem_path = dirs["tables"] / "block_04_synthetic_grid.csv"
                summary_path = dirs["manifests"] / "block_04_synthetic_summary.json"
                save_dataframe(sweep_df, theorem_path)
                save_json(
                    {
                        "sample_sizes": sample_sizes,
                        "amplitude_ranges": amplitude_ranges,
                        "best_weighted_cosine": float(sweep_df["weighted_cosine_template_vs_empca"].max()),
                        "median_emp_truth_rmse": float(sweep_df["emp_truth_rmse"].median()),
                    },
                    summary_path,
                )

                fig, ax = plt.subplots(figsize=(6, 4))
                for amp_max, group in sweep_df.groupby("amp_max"):
                    ax.plot(group["n_events"], group["weighted_cosine_template_vs_empca"], marker="o", label=f"amp_max={amp_max:.1f}")
                ax.set_xlabel("number of events")
                ax.set_ylabel("weighted cosine(empca, planted template)")
                ax.set_title("Synthetic theorem-regime recovery")
                ax.legend()
                plt.show()
                """
            ),
        ],
    },
    {
        "name": "block_05_linear_ae_bridge_and_gauge.ipynb",
        "title": "Block 05: Linear-AE Bridge and Gauge Handling",
        "goal": "Turn the EMPCA/linear-AE equivalence into a stable weighted-subspace computation with explicit gauge handling.",
        "regime": "theorem-support on canonical split",
        "actions": [
            "compare EMPCA against the exact whitened weighted-SVD baseline at fixed rank",
            "separate subspace agreement from coordinate agreement",
            "save a compact bridge summary that later sections can quote directly",
        ],
        "cells": [
            md(
                """
                ## Why this notebook uses a synthetic rank-k fallback

                The repo does not ship a real local rank-`k` calibration dataset. When that file is absent, this notebook uses a synthetic rank-`k` mixture so the bridge logic remains executable and the exact weighted SVD baseline still has a meaningful job.
                """
            ),
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=512, synthetic_rankk=True)
                train_idx = bundle.split_indices["train"]
                test_idx = bundle.split_indices["test"]
                k = 3

                exact = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=k)
                emp = fit_weighted_empca(
                    bundle.traces_freq[train_idx],
                    bundle.weights_one_sided,
                    k=k,
                    n_iter=cfg.default_empca_iter,
                    patience=cfg.default_empca_patience,
                    smoothing=False,
                    init="svd",
                    seed=cfg.seed,
                )
                cosines, angles_deg = principal_angles_weighted(emp["basis"], exact["basis_native"], bundle.weights_one_sided)
                bridge_summary = compute_residual_summary(
                    bundle.traces_freq[test_idx],
                    emp["basis"],
                    exact["basis_native"],
                    bundle.weights_one_sided,
                )
                bridge_summary["principal_angle_cosines"] = [float(x) for x in cosines]
                bridge_summary["principal_angles_deg"] = [float(x) for x in angles_deg]
                pd.Series(bridge_summary, name="value").to_frame()
                """
            ),
            code(
                """
                coeff_emp = rankk_gls_coefficients(bundle.traces_freq[test_idx], emp["basis"], bundle.weights_one_sided, return_complex=True)
                coeff_exact = rankk_gls_coefficients(bundle.traces_freq[test_idx], exact["basis_native"], bundle.weights_one_sided, return_complex=True)

                coordinate_df = pd.DataFrame(
                    {
                        "empca_coeff_norm": np.linalg.norm(coeff_emp, axis=1),
                        "exact_coeff_norm": np.linalg.norm(coeff_exact, axis=1),
                    }
                )
                display(coordinate_df.describe().T)
                """
            ),
            code(
                """
                summary_path = dirs["manifests"] / "block_05_bridge_summary.json"
                coord_path = dirs["tables"] / "block_05_coordinate_norms.csv"
                save_json(
                    {
                        "source": bundle.metadata["source"],
                        "rank": k,
                        "bridge_summary": bridge_summary,
                        "coordinate_note": "Subspace agreement is the primary diagnostic; latent coordinates are basis-dependent.",
                    },
                    summary_path,
                )
                save_dataframe(coordinate_df, coord_path)
                display(pd.DataFrame([manifest_row("block_05", "theorem-support", str(summary_path.relative_to(REPO_ROOT)), cfg, bundle)]))
                """
            ),
        ],
    },
    {
        "name": "block_06_convergence_initialization_and_rank_selection.ipynb",
        "title": "Block 06: Convergence, Initialization, and Rank Selection",
        "goal": "Provide a practical recipe for fitting weighted subspace models: convergence traces, initialization choices, and rank selection.",
        "regime": "real-support with synthetic fallback",
        "actions": [
            "compare unsmoothed and smoothed EMPCA objective traces",
            "compare random and SVD/template-informed initialization",
            "build a held-out weighted-residual saturation table across rank and seed",
        ],
        "cells": [
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=512, synthetic_rankk=True)
                train_idx = bundle.split_indices["train"]
                val_idx = bundle.split_indices["val"]

                fit_random = fit_weighted_empca(
                    bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3, n_iter=60, patience=8, smoothing=False, init="random", seed=cfg.seed
                )
                fit_svd = fit_weighted_empca(
                    bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3, n_iter=60, patience=8, smoothing=False, init="svd", seed=cfg.seed
                )
                fit_smooth = fit_weighted_empca(
                    bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3, n_iter=60, patience=8, smoothing=True, init="svd", seed=cfg.seed
                )
                """
            ),
            code(
                """
                convergence_df = pd.DataFrame(
                    {
                        "iter": np.arange(max(len(fit_random["chi2_trace"]), len(fit_svd["chi2_trace"]), len(fit_smooth["chi2_trace"]))),
                        "random_no_smooth": pd.Series(fit_random["chi2_trace"]),
                        "svd_no_smooth": pd.Series(fit_svd["chi2_trace"]),
                        "svd_smooth": pd.Series(fit_smooth["chi2_trace"]),
                    }
                )
                display(convergence_df.head())

                fig, ax = plt.subplots(figsize=(6, 4))
                for col in ["random_no_smooth", "svd_no_smooth", "svd_smooth"]:
                    ax.plot(convergence_df["iter"], convergence_df[col], marker="o", label=col)
                ax.set_xlabel("iteration")
                ax.set_ylabel("chi2 surrogate")
                ax.set_title("EMPCA convergence traces")
                ax.legend()
                plt.show()
                """
            ),
            code(
                """
                rank_rows = []
                for k in range(1, cfg.empirical_rank_max + 1):
                    for seed in [cfg.seed, cfg.seed + 1, cfg.seed + 2]:
                        fit = fit_weighted_empca(
                            bundle.traces_freq[train_idx],
                            bundle.weights_one_sided,
                            k=k,
                            n_iter=40,
                            patience=6,
                            smoothing=False,
                            init="svd",
                            seed=seed,
                        )
                        coeff_val = rankk_gls_coefficients(bundle.traces_freq[val_idx], fit["basis"], bundle.weights_one_sided, return_complex=True)
                        resid_val = residual_energy_per_trace(bundle.traces_freq[val_idx], fit["basis"], coeff_val, bundle.weights_one_sided)
                        rank_rows.append(
                            {
                                "k": k,
                                "seed": seed,
                                "val_residual_mean": float(np.mean(resid_val)),
                                "val_residual_std": float(np.std(resid_val)),
                                "n_iter_used": int(fit["n_iter_used"]),
                            }
                        )
                rank_df = pd.DataFrame(rank_rows)
                rank_summary = rank_df.groupby("k", as_index=False).agg(
                    val_residual_mean=("val_residual_mean", "mean"),
                    val_residual_std=("val_residual_mean", "std"),
                    avg_iterations=("n_iter_used", "mean"),
                )
                display(rank_summary)
                """
            ),
            code(
                """
                summary_path = dirs["manifests"] / "block_06_rank_selection_summary.json"
                table_path = dirs["tables"] / "block_06_rank_selection.csv"
                save_dataframe(rank_df, table_path)
                save_json(
                    {
                        "source": bundle.metadata["source"],
                        "recommended_primary_rule": "Choose k at the held-out weighted-residual saturation knee.",
                        "best_k_by_mean_val_residual": int(rank_summary.sort_values("val_residual_mean").iloc[0]["k"]),
                    },
                    summary_path,
                )
                display(rank_summary.sort_values("val_residual_mean"))
                """
            ),
        ],
    },
    {
        "name": "block_07_rankk_quality_and_noiseaware_ablation.ipynb",
        "title": "Block 07: Rank-k Quality and Noise-Aware Ablation",
        "goal": "Show when higher rank helps and whether noise-aware weighting matters relative to isotropic subspace fitting.",
        "regime": "real-support with colored-noise synthetic fallback",
        "actions": [
            "sweep rank k and evaluate held-out weighted residual saturation",
            "compare weighted versus isotropic rank-k baselines at matched capacity",
            "save one table that can drive the paper’s rank-k centerpiece figure",
        ],
        "cells": [
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=512, synthetic_rankk=True)
                train_idx = bundle.split_indices["train"]
                test_idx = bundle.split_indices["test"]
                rows = []

                for k in range(1, cfg.empirical_rank_max + 1):
                    weighted_basis = exact_weighted_subspace(bundle.traces_freq[train_idx], bundle.weights_one_sided, k=k)["basis_native"]
                    isotropic_basis = exact_isotropic_subspace(bundle.traces_freq[train_idx], k=k)

                    coeff_w = rankk_gls_coefficients(bundle.traces_freq[test_idx], weighted_basis, bundle.weights_one_sided, return_complex=True)
                    resid_w = residual_energy_per_trace(bundle.traces_freq[test_idx], weighted_basis, coeff_w, bundle.weights_one_sided)

                    coeff_i = rankk_gls_coefficients(bundle.traces_freq[test_idx], isotropic_basis, bundle.weights_one_sided, return_complex=True)
                    resid_i = residual_energy_per_trace(bundle.traces_freq[test_idx], isotropic_basis, coeff_i, bundle.weights_one_sided)

                    rows.append(
                        {
                            "k": k,
                            "weighted_residual_mean": float(np.mean(resid_w)),
                            "weighted_residual_std": float(np.std(resid_w)),
                            "isotropic_residual_mean": float(np.mean(resid_i)),
                            "isotropic_residual_std": float(np.std(resid_i)),
                            "relative_gain_weighted_vs_isotropic": float((np.mean(resid_i) - np.mean(resid_w)) / np.mean(resid_i)),
                        }
                    )

                ablation_df = pd.DataFrame(rows)
                display(ablation_df)
                """
            ),
            code(
                """
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(ablation_df["k"], ablation_df["weighted_residual_mean"], marker="o", label="weighted")
                ax.plot(ablation_df["k"], ablation_df["isotropic_residual_mean"], marker="s", label="isotropic")
                ax.set_xlabel("rank k")
                ax.set_ylabel("held-out weighted residual mean")
                ax.set_title("Rank-k saturation and loss-geometry ablation")
                ax.legend()
                plt.show()
                """
            ),
            code(
                """
                table_path = dirs["tables"] / "block_07_ablation.csv"
                summary_path = dirs["manifests"] / "block_07_ablation_summary.json"
                save_dataframe(ablation_df, table_path)
                save_json(
                    {
                        "source": bundle.metadata["source"],
                        "best_k_weighted": int(ablation_df.sort_values("weighted_residual_mean").iloc[0]["k"]),
                        "gain_at_k1": float(ablation_df.loc[ablation_df["k"] == 1, "relative_gain_weighted_vs_isotropic"].iloc[0]),
                        "max_gain": float(ablation_df["relative_gain_weighted_vs_isotropic"].max()),
                    },
                    summary_path,
                )
                """
            ),
        ],
    },
    {
        "name": "block_08_pc_interpretation_and_centering.ipynb",
        "title": "Block 08: PC Interpretation and Centering",
        "goal": "Turn the existing PC interpretation work into a conservative, reproducible overlap-and-correlation analysis.",
        "regime": "real-support with synthetic fallback",
        "actions": [
            "fit uncentered and centered weighted rank-3 subspaces",
            "compare leading components against template-like, timing-like, and width-like proxies",
            "save overlap and coefficient-correlation tables with explicit centering labels",
        ],
        "cells": [
            code(
                """
                bundle = load_or_make_dataset(cfg, prefer_real=True, synthetic_events=512, synthetic_rankk=True)
                train_idx = bundle.split_indices["train"]
                test_idx = bundle.split_indices["test"]
                proxy_time = build_proxy_family(bundle.template_time, cfg)
                proxy_freq = {name: np.fft.rfft(value) for name, value in proxy_time.items()}

                fit_uncentered = fit_weighted_empca(
                    bundle.traces_freq[train_idx], bundle.weights_one_sided, k=3, n_iter=60, patience=8, smoothing=False, init="svd", seed=cfg.seed
                )

                mean_train = np.mean(bundle.traces_baseline[train_idx], axis=0, keepdims=True)
                centered_freq = rfft_traces(bundle.traces_baseline - mean_train)
                fit_centered = fit_weighted_empca(
                    centered_freq[train_idx], bundle.weights_one_sided, k=3, n_iter=60, patience=8, smoothing=False, init="svd", seed=cfg.seed
                )
                """
            ),
            code(
                """
                rows = []
                for label, basis in [("uncentered", fit_uncentered["basis"]), ("centered", fit_centered["basis"])]:
                    for comp_idx in range(3):
                        component = basis[comp_idx]
                        for proxy_name, proxy in proxy_freq.items():
                            rows.append(
                                {
                                    "fit_variant": label,
                                    "component": f"PC{comp_idx + 1}",
                                    "proxy": proxy_name,
                                    "weighted_cosine": float(weighted_cosine(component, proxy, bundle.weights_one_sided)),
                                }
                            )
                overlap_df = pd.DataFrame(rows)
                display(overlap_df.pivot_table(index=["fit_variant", "component"], columns="proxy", values="weighted_cosine"))
                """
            ),
            code(
                """
                coeff_unc = rankk_gls_coefficients(bundle.traces_freq[test_idx], fit_uncentered["basis"], bundle.weights_one_sided, return_complex=False)
                amp_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["template_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)
                timing_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["timing_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)
                width_proxy = rankk_gls_coefficients(bundle.traces_freq[test_idx], proxy_freq["width_like"], bundle.weights_one_sided, return_complex=False).reshape(-1)

                corr_rows = []
                proxy_map = {"amplitude_proxy": amp_proxy, "timing_proxy": timing_proxy, "width_proxy": width_proxy}
                for comp_idx in range(coeff_unc.shape[1]):
                    for proxy_name, proxy_values in proxy_map.items():
                        corr_rows.append(
                            {
                                "component": f"PC{comp_idx + 1}",
                                "proxy": proxy_name,
                                "pearson_r": float(np.corrcoef(coeff_unc[:, comp_idx], proxy_values)[0, 1]),
                            }
                        )
                corr_df = pd.DataFrame(corr_rows)
                display(corr_df.pivot_table(index="component", columns="proxy", values="pearson_r"))
                """
            ),
            code(
                """
                overlap_path = dirs["tables"] / "block_08_pc_metrics.csv"
                corr_path = dirs["tables"] / "block_08_pc_correlations.csv"
                summary_path = dirs["manifests"] / "block_08_pc_summary.json"
                save_dataframe(overlap_df, overlap_path)
                save_dataframe(corr_df, corr_path)
                save_json(
                    {
                        "source": bundle.metadata["source"],
                        "note": "Interpretation is descriptive: weighted overlap and coefficient correlation, not proof of physical identity.",
                    },
                    summary_path,
                )
                """
            ),
        ],
    },
    {
        "name": "block_09_structured_noise_and_multichannel_robustness.ipynb",
        "title": "Block 09: Structured-Noise and Multichannel Robustness",
        "goal": "Stress the weighted methods under out-of-model perturbations without contaminating the theorem-support story.",
        "regime": "robustness-support",
        "actions": [
            "evaluate clean-trained methods on structured-noise perturbation suites",
            "compare rank-1, rank-k, and isotropic baselines by degradation",
            "summarize synthetic multichannel statistics and mark joint correlated OF as deferred",
        ],
        "cells": [
            code(
                """
                base_bundle = load_or_make_dataset(cfg, prefer_real=False, synthetic_events=384, synthetic_rankk=True)
                train_idx = base_bundle.split_indices["train"]
                test_idx = base_bundle.split_indices["test"]

                method_basis = {
                    "of_rank1": base_bundle.template_freq,
                    "weighted_rank1": fit_weighted_empca(
                        base_bundle.traces_freq[train_idx], base_bundle.weights_one_sided, k=1, n_iter=50, patience=8, smoothing=False, init="template", template_f=base_bundle.template_freq, seed=cfg.seed
                    )["basis"][0],
                    "weighted_rank3": fit_weighted_empca(
                        base_bundle.traces_freq[train_idx], base_bundle.weights_one_sided, k=3, n_iter=50, patience=8, smoothing=False, init="svd", seed=cfg.seed
                    )["basis"],
                    "isotropic_rank3": exact_isotropic_subspace(base_bundle.traces_freq[train_idx], k=3),
                }

                clean_scores = {}
                for method, basis in method_basis.items():
                    coeff = rankk_gls_coefficients(base_bundle.traces_freq[test_idx], basis, base_bundle.weights_one_sided, return_complex=True)
                    clean_scores[method] = float(np.mean(residual_energy_per_trace(base_bundle.traces_freq[test_idx], basis, coeff, base_bundle.weights_one_sided)))

                scenarios = apply_structured_noise_suite(base_bundle.traces_baseline, cfg, base_bundle.psd_one_sided)
                rows = []
                for scenario_name, payload in scenarios.items():
                    pert_bundle = prepare_dataset(
                        payload["traces"],
                        cfg,
                        template_time=base_bundle.template_time,
                        psd_one_sided=base_bundle.psd_one_sided,
                        metadata={"source": scenario_name, "support_regime": payload["regime"]},
                    )
                    for method, basis in method_basis.items():
                        coeff = rankk_gls_coefficients(pert_bundle.traces_freq[test_idx], basis, pert_bundle.weights_one_sided, return_complex=True)
                        resid = residual_energy_per_trace(pert_bundle.traces_freq[test_idx], basis, coeff, pert_bundle.weights_one_sided)
                        rows.append(
                            {
                                "scenario": scenario_name,
                                "method": method,
                                "residual_mean": float(np.mean(resid)),
                                "relative_degradation_vs_clean": float((np.mean(resid) - clean_scores[method]) / clean_scores[method]),
                            }
                        )
                robustness_df = pd.DataFrame(rows)
                display(robustness_df)
                """
            ),
            code(
                """
                multichannel = generate_multichannel_synthetic(cfg, n_events=128, n_channels=4, corr_strength=0.35)
                channel_cov = np.corrcoef(multichannel["traces_multichannel"][0])
                pd.DataFrame(channel_cov)
                """
            ),
            code(
                """
                table_path = dirs["tables"] / "block_09_robustness.csv"
                summary_path = dirs["manifests"] / "block_09_robustness_summary.json"
                save_dataframe(robustness_df, table_path)
                save_json(
                    {
                        "structured_noise_source": "synthetic noise_module-based perturbations",
                        "multichannel_note": "Synthetic multichannel characterization is included; joint correlated OF remains deferred until a dedicated implementation lands.",
                    },
                    summary_path,
                )
                """
            ),
        ],
    },
    {
        "name": "block_10_tables_figures_appendix_and_reproducibility.ipynb",
        "title": "Block 10: Tables, Figures, Appendix, and Reproducibility",
        "goal": "Turn the outputs of Blocks 01–09 into a reproducibility-aware artifact inventory for the paper.",
        "regime": "packaging / mixed",
        "actions": [
            "inventory the generated tables and manifests",
            "mark which expected paper artifacts exist and which are still pending",
            "save one reproducibility manifest that centralizes the notebook-facing defaults",
        ],
        "cells": [
            code(
                """
                expected = pd.DataFrame(
                    [
                        ("block_01", "results/tables/block_01_claim_map.csv"),
                        ("block_02", "results/manifests/block_02_split_manifest.json"),
                        ("block_03", "results/manifests/block_03_of_summary.json"),
                        ("block_04", "results/tables/block_04_synthetic_grid.csv"),
                        ("block_05", "results/manifests/block_05_bridge_summary.json"),
                        ("block_06", "results/tables/block_06_rank_selection.csv"),
                        ("block_07", "results/tables/block_07_ablation.csv"),
                        ("block_08", "results/tables/block_08_pc_metrics.csv"),
                        ("block_09", "results/tables/block_09_robustness.csv"),
                    ],
                    columns=["block_id", "artifact_relpath"],
                )
                expected["exists"] = expected["artifact_relpath"].map(lambda p: (REPO_ROOT / p).exists())
                display(expected)
                """
            ),
            code(
                """
                manifest_files = sorted((dirs["manifests"]).glob("block_*.json"))
                table_files = sorted((dirs["tables"]).glob("block_*.*"))
                inventory = pd.DataFrame(
                    [{"kind": "manifest", "path": str(path.relative_to(REPO_ROOT))} for path in manifest_files]
                    + [{"kind": "table", "path": str(path.relative_to(REPO_ROOT))} for path in table_files]
                )
                display(inventory)
                """
            ),
            code(
                """
                reproducibility_manifest = {
                    "canonical_config": config_as_frame(cfg).to_dict(orient="records"),
                    "expected_artifacts": expected.to_dict(orient="records"),
                    "present_inventory": inventory.to_dict(orient="records"),
                }
                manifest_path = dirs["manifests"] / "block_10_reproducibility_manifest.json"
                save_json(reproducibility_manifest, manifest_path)
                pd.DataFrame([{"saved_manifest": str(manifest_path.relative_to(REPO_ROOT))}])
                """
            ),
        ],
    },
]


def main() -> None:
    for spec in NOTEBOOK_SPECS:
        path = ROOT / spec["name"]
        nb = notebook(spec)
        nbf.write(nb, path)
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
