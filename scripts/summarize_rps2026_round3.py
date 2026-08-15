#!/usr/bin/env python3
"""Generate the Round 3 summary from archived seed-level and rung-4 results."""

from __future__ import annotations

import argparse
import json
import re
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


def method_means(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = [
        "train_normalized_chi2",
        "test_normalized_chi2",
        "test_raw_mse",
        "angle_to_cw_pca_mean_deg",
    ]
    return {
        str(method): {metric: float(group[metric].mean()) for metric in metrics}
        for method, group in frame.groupby("method")
    }


def nested_differences(left: object, right: object, path: str = "") -> list[dict]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            differences.extend(
                nested_differences(left.get(key), right.get(key), f"{path}/{key}")
            )
        return differences
    if left == right:
        return []
    return [{"path": path, "left": left, "right": right}]


def benchmark_summary(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    match = re.search(
        r"Coarse Denoising Score for Model \S+: ([^\s]+)", record["stdout"]
    )
    if match is None:
        raise ValueError(f"score missing from {path}")
    return {
        "source": str(path),
        "returncode": int(record["returncode"]),
        "succeeded": bool(record["succeeded"]),
        "file_set": record["file_set"],
        "scored_files": record.get("scored_files"),
        "excluded_files": record.get("excluded_files", []),
        "score": float(match.group(1)),
        "command": record["command"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--t-mse-config", type=Path, required=True)
    parser.add_argument("--t-chi2-config", type=Path, required=True)
    parser.add_argument("--t-mse-inband", type=Path, required=True)
    parser.add_argument("--t-mse-all20", type=Path, required=True)
    parser.add_argument("--t-chi2-inband", type=Path, required=True)
    parser.add_argument("--t-chi2-all20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    block_a = pd.read_csv(args.block_a)
    block_b = pd.read_csv(args.block_b)
    smoke = pd.read_csv(args.smoke)
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
        "smoke": {
            "source": str(args.smoke),
            "validation": validate_frame(smoke, 2),
            "method_means": method_means(smoke),
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

    mse_config = json.loads(args.t_mse_config.read_text(encoding="utf-8"))
    chi2_config = json.loads(args.t_chi2_config.read_text(encoding="utf-8"))
    rung4 = {
        "config_sources": [str(args.t_mse_config), str(args.t_chi2_config)],
        "config_differences": nested_differences(mse_config, chi2_config),
        "t_mse": {
            "in_band": benchmark_summary(args.t_mse_inband),
            "all_20": benchmark_summary(args.t_mse_all20),
        },
        "t_chi2": {
            "in_band": benchmark_summary(args.t_chi2_inband),
            "all_20": benchmark_summary(args.t_chi2_all20),
        },
    }
    rung4["in_band_contrast_t_chi2_minus_t_mse"] = (
        rung4["t_chi2"]["in_band"]["score"]
        - rung4["t_mse"]["in_band"]["score"]
    )
    rung4["all_20_contrast_t_chi2_minus_t_mse"] = (
        rung4["t_chi2"]["all_20"]["score"]
        - rung4["t_mse"]["all_20"]["score"]
    )
    rung4["config_gate_pass"] = rung4["config_differences"] == [
        {"path": "/run_config/train/loss", "left": "mse", "right": "chi2"}
    ]
    summary["rung4"] = rung4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
