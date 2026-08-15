#!/usr/bin/env python3
"""Generate the Round 3 NFPA summary from archived seed-level results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, PRIMARY_HASH


def paired_summary(
    frame: pd.DataFrame, method_a: str, method_b: str, metric: str
) -> dict[str, float | list[float]]:
    values = frame.pivot(index="seed", columns="method", values=metric)
    differences = (values[method_a] - values[method_b]).to_numpy()
    rng = np.random.default_rng(PRIMARY.bootstrap_seed)
    indices = rng.integers(
        0,
        differences.size,
        size=(PRIMARY.bootstrap_resamples, differences.size),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    standard_deviation = float(differences.std(ddof=1))
    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "n_seeds": int(differences.size),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "percentile_95_ci": [
            float(value) for value in np.percentile(bootstrap_means, [2.5, 97.5])
        ],
        "standardized_paired_effect": (
            float(differences.mean() / standard_deviation)
            if standard_deviation > 0
            else (0.0 if differences.mean() == 0 else None)
        ),
    }


def validate_frame(frame: pd.DataFrame, expected_seeds: int) -> dict[str, object]:
    expected_methods = {"CW-PCA", "NFPA", "Iso-MPCA", "IsoPCA", "OF"}
    seeds = sorted(int(seed) for seed in frame["seed"].unique())
    methods = set(frame["method"].unique())
    return {
        "rows": int(len(frame)),
        "seeds": seeds,
        "seed_count": len(seeds),
        "methods": sorted(methods),
        "all_converged": bool(frame["converged"].all()),
        "config_hashes": sorted(frame["config_hash"].astype(str).unique()),
        "expected_shape": bool(
            len(seeds) == expected_seeds
            and methods == expected_methods
            and len(frame) == expected_seeds * len(expected_methods)
        ),
        "C": sorted(int(value) for value in frame["C"].unique()),
        "k_c": sorted(int(value) for value in frame["k_c"].unique()),
        "k_t": sorted(int(value) for value in frame["k_t"].unique()),
        "max_covariance_whitening_error": float(
            frame["covariance_whitening_error"].max()
        ),
        "max_covariance_mc_error_whitened": float(
            frame["covariance_mc_error_whitened"].max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    block_a = pd.read_csv(args.block_a)
    block_b = pd.read_csv(args.block_b)
    a_validation = validate_frame(block_a, PRIMARY.n_seeds)
    b_validation = validate_frame(block_b, 20)

    a_contrasts = {
        "nfpa_minus_iso_mpca_chi2": paired_summary(
            block_a, "NFPA", "Iso-MPCA", "test_normalized_chi2"
        ),
        "nfpa_minus_iso_mpca_mse": paired_summary(
            block_a, "NFPA", "Iso-MPCA", "test_raw_mse"
        ),
        "nfpa_minus_cw_pca_chi2": paired_summary(
            block_a, "NFPA", "CW-PCA", "test_normalized_chi2"
        ),
    }
    b_contrasts = {
        "nfpa_minus_iso_mpca_chi2": paired_summary(
            block_b, "NFPA", "Iso-MPCA", "test_normalized_chi2"
        ),
        "nfpa_minus_iso_mpca_mse": paired_summary(
            block_b, "NFPA", "Iso-MPCA", "test_raw_mse"
        ),
    }

    full_pair = block_b[block_b["method"].isin(["CW-PCA", "IsoPCA"])].pivot(
        index="seed", columns="method", values="angle_to_cw_pca_max_deg"
    )
    max_full_projector_angle = float(full_pair.to_numpy().max())
    white_chi2_ci = b_contrasts["nfpa_minus_iso_mpca_chi2"][
        "percentile_95_ci"
    ]
    summary = {
        "config_hash": PRIMARY_HASH,
        "bootstrap": {
            "resamples": PRIMARY.bootstrap_resamples,
            "seed": PRIMARY.bootstrap_seed,
            "interval": "seed-level paired percentile 95% CI",
        },
        "block_a": {
            "source": str(args.block_a),
            "validation": a_validation,
            "contrasts": a_contrasts,
            "gate": {
                "C_greater_than_k_c": a_validation["C"][0]
                > a_validation["k_c"][0],
                "predicted_chi2_interval_below_zero": a_contrasts[
                    "nfpa_minus_iso_mpca_chi2"
                ]["percentile_95_ci"][1]
                < 0,
                "predicted_mse_interval_above_zero": a_contrasts[
                    "nfpa_minus_iso_mpca_mse"
                ]["percentile_95_ci"][0]
                > 0,
            },
        },
        "block_b": {
            "source": str(args.block_b),
            "validation": b_validation,
            "contrasts": b_contrasts,
            "gate": {
                "predeclared_chi2_margin": PRIMARY.white_noise_chi2_margin,
                "predeclared_projector_angle_margin_deg": (
                    PRIMARY.white_noise_angle_margin_deg
                ),
                "max_cw_pca_iso_pca_projector_angle_deg": (
                    max_full_projector_angle
                ),
                "full_projectors_agree": (
                    max_full_projector_angle
                    <= PRIMARY.white_noise_angle_margin_deg
                ),
                "fixed_architecture_chi2_interval_inside_margin": bool(
                    white_chi2_ci[0] >= -PRIMARY.white_noise_chi2_margin
                    and white_chi2_ci[1] <= PRIMARY.white_noise_chi2_margin
                ),
            },
        },
    }
    summary["block_a"]["gate"]["pass"] = bool(
        a_validation["expected_shape"]
        and a_validation["all_converged"]
        and a_validation["config_hashes"] == [PRIMARY_HASH]
        and all(summary["block_a"]["gate"].values())
    )
    summary["block_b"]["gate"]["pass"] = bool(
        b_validation["expected_shape"]
        and b_validation["all_converged"]
        and b_validation["config_hashes"] == [PRIMARY_HASH]
        and summary["block_b"]["gate"]["full_projectors_agree"]
        and summary["block_b"]["gate"][
            "fixed_architecture_chi2_interval_inside_margin"
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
