"""EXP-04: predeclared >=30-seed S5 confirmatory sweep, plus an S4 negative
control under the identical held-out protocol.

The frozen 10-seed S5 result (``transfer_paper/data/synthetic/S5_sweep_10seeds.*``)
is PRESERVED AS-IS and never overwritten by this script -- see the evidence
rule in docs/plans/REVISION_PLAN_2026-07-26.md. This script adds a new,
independent, predeclared run under ``results/sweeps_v2/<run_id>/``.

Predeclaration (fixed before this script is run, not tuned afterward):

- seeds: 30 independent seeds, 1000-1029 (disjoint from the frozen 0-9 block
  and from every other seed range already used elsewhere in this repo);
- test_frac=0.5, split_seed=0 (fixed split construction; only the
  data-generating seed varies across the sweep), rank=2 -- matching
  ``configs/synthetic/s5_metric_reversal_v2.yaml``;
- smallest practically relevant effect: an EMPCA weighted-residual
  improvement of at least 0.5% relative to the PCA weighted residual (about
  half of the ~1.1% effect observed in the frozen 10-seed result) -- a
  statistically resolvable but not merely numerical-noise threshold;
- S4 (white-noise) is run as a negative control under the identical
  protocol: PCA and EMPCA must remain statistically indistinguishable.

JSON schema guardrail: the output JSON's ``ratio`` block is the ONLY field
a prose/report generator may cite when describing an effect as a "x times"
multiple. A scalar difference or a fractional-difference percentage must
never be reported as a ratio unless it is read from this block.

    uv run python scripts/run_s5_confirmatory_sweep.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.experiments import run_synthetic_experiment  # noqa: E402
from noise_geometry.validation import (  # noqa: E402
    paired_difference,
    run_seed_sweep,
    summarize,
    write_json,
    write_rows_csv,
)

SEEDS = list(range(1000, 1030))  # predeclared: >=30 independent seeds
assert len(SEEDS) >= 30

S5_BASE_CONFIG = {"experiment_id": "S5", "test_frac": 0.5, "split_seed": 0, "rank": 2}
S4_BASE_CONFIG = {"experiment_id": "S4", "test_frac": 0.5, "split_seed": 0}

SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT = 0.005  # predeclared, 0.5%


def _add_ratio_fields(rows: list[dict[str, float]]) -> None:
    """Mutate rows in place, adding the only fields a 'x times' claim may cite."""
    for row in rows:
        if "pca_weighted_residual_to_observed" in row and "weighted_pca_weighted_residual_to_observed" in row:
            denom = row["weighted_pca_weighted_residual_to_observed"]
            row["pca_over_empca_weighted_residual_ratio"] = (
                row["pca_weighted_residual_to_observed"] / denom if denom != 0 else float("nan")
            )
        if "pca_raw_residual_to_observed" in row and "weighted_pca_raw_residual_to_observed" in row:
            denom = row["weighted_pca_raw_residual_to_observed"]
            row["pca_over_empca_raw_residual_ratio"] = (
                row["pca_raw_residual_to_observed"] / denom if denom != 0 else float("nan")
            )


def _mean(rows: list[dict[str, float]], key: str) -> float:
    vals = [r[key] for r in rows if key in r]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def run_s5_confirmatory() -> dict[str, Any]:
    rows = run_seed_sweep(run_synthetic_experiment, S5_BASE_CONFIG, SEEDS)
    _add_ratio_fields(rows)
    summary = summarize(rows, n_boot=2000, seed=0)

    weighted_diff = paired_difference(
        rows, "pca_weighted_residual_to_observed", "weighted_pca_weighted_residual_to_observed", n_boot=2000, seed=0
    )
    raw_diff = paired_difference(
        rows, "pca_raw_residual_to_observed", "weighted_pca_raw_residual_to_observed", n_boot=2000, seed=0
    )
    nll_diff = paired_difference(rows, "pca_nll_mean", "weighted_pca_nll_mean", n_boot=2000, seed=0)

    pca_weighted_mean = _mean(rows, "pca_weighted_residual_to_observed")
    empca_weighted_mean = _mean(rows, "weighted_pca_weighted_residual_to_observed")
    pca_raw_mean = _mean(rows, "pca_raw_residual_to_observed")
    empca_raw_mean = _mean(rows, "weighted_pca_raw_residual_to_observed")

    fractional_improvement = weighted_diff["mean_difference"] / pca_weighted_mean if pca_weighted_mean else float("nan")
    ratio_summary = summary.get("pca_over_empca_weighted_residual_ratio", {})
    raw_ratio_summary = summary.get("pca_over_empca_raw_residual_ratio", {})

    return {
        "experiment": "S5",
        "run_kind": "confirmatory_sweep_v2",
        "predeclaration": {
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "config": S5_BASE_CONFIG,
            "smallest_relevant_effect": {
                "metric": "weighted_residual_fractional_improvement_of_empca_over_pca",
                "threshold": SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT,
            },
        },
        "raw_value": {
            "pca_weighted_residual_mean": pca_weighted_mean,
            "empca_weighted_residual_mean": empca_weighted_mean,
            "pca_raw_residual_mean": pca_raw_mean,
            "empca_raw_residual_mean": empca_raw_mean,
        },
        "absolute_difference": {
            "weighted_residual_pca_minus_empca": weighted_diff,
            "raw_residual_pca_minus_empca": raw_diff,
            "nll_pca_minus_empca": nll_diff,
        },
        "fractional_difference": {
            "weighted_residual_improvement_of_empca_over_pca": fractional_improvement,
        },
        "ratio": {
            "weighted_residual_pca_over_empca_mean": ratio_summary.get("mean"),
            "weighted_residual_pca_over_empca_ci95": ratio_summary.get("ci95"),
            "raw_residual_pca_over_empca_mean": raw_ratio_summary.get("mean"),
            "raw_residual_pca_over_empca_ci95": raw_ratio_summary.get("ci95"),
            "_note": (
                "This is the ONLY block a manuscript may cite for a 'x times' "
                "claim. weighted_residual_pca_over_empca_mean is the likelihood "
                "metric ratio; raw_residual_pca_over_empca_mean is the "
                "raw-MSE-metric ratio, expected on the OPPOSITE side of 1 "
                "(the metric-reversal effect)."
            ),
        },
        "smallest_relevant_effect_check": {
            "observed_fractional_improvement": fractional_improvement,
            "threshold": SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT,
            "exceeds_threshold": bool(fractional_improvement > SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT),
        },
        "direction_check": {
            "weighted_residual_favors_empca": bool(weighted_diff["mean_difference"] > 0),
            "raw_residual_favors_pca": bool(raw_diff["mean_difference"] < 0),
            "orderings_reversed": bool(
                (weighted_diff["mean_difference"] > 0) != (raw_diff["mean_difference"] > 0)
            ),
        },
        "fraction_of_seeds_with_expected_sign": {
            "weighted_residual_empca_better": weighted_diff["fraction_positive"],
            "raw_residual_pca_better": 1.0 - raw_diff["fraction_positive"],
        },
        "summary": summary,
        "rows": rows,
    }


def run_s4_negative_control() -> dict[str, Any]:
    rows = run_seed_sweep(run_synthetic_experiment, S4_BASE_CONFIG, SEEDS)
    summary = summarize(rows, n_boot=2000, seed=0)
    weighted_diff = paired_difference(
        rows, "pca_weighted_residual", "empca_weighted_residual", n_boot=2000, seed=0
    )
    angle_summary = summary.get("max_principal_angle_deg", {})
    return {
        "experiment": "S4",
        "run_kind": "confirmatory_sweep_v2_negative_control",
        "predeclaration": {"seeds": SEEDS, "n_seeds": len(SEEDS), "config": S4_BASE_CONFIG},
        "absolute_difference": {"weighted_residual_pca_minus_empca": weighted_diff},
        "principal_angle_deg": angle_summary,
        "remains_negative_control": bool(not weighted_diff["ci95_excludes_zero"]),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    run_id = f"sweeps_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_root = REPO_ROOT / "results" / "sweeps_v2" / run_id

    s5 = run_s5_confirmatory()
    s5_dir = out_root / "S5"
    write_json(s5, s5_dir / "S5_confirmatory_30seeds.json")
    write_rows_csv(s5["rows"], s5_dir / "S5_confirmatory_30seeds.csv")

    s4 = run_s4_negative_control()
    s4_dir = out_root / "S4"
    write_json(s4, s4_dir / "S4_confirmatory_30seeds.json")
    write_rows_csv(s4["rows"], s4_dir / "S4_confirmatory_30seeds.csv")

    print(f"S5: n_seeds={s5['predeclaration']['n_seeds']}")
    print(f"  weighted-residual ratio (PCA/EMPCA): {s5['ratio']['weighted_residual_pca_over_empca_mean']:.4f}  "
          f"CI95={s5['ratio']['weighted_residual_pca_over_empca_ci95']}")
    print(f"  raw-residual ratio (PCA/EMPCA):      {s5['ratio']['raw_residual_pca_over_empca_mean']:.4f}  "
          f"CI95={s5['ratio']['raw_residual_pca_over_empca_ci95']}")
    print(f"  orderings reversed: {s5['direction_check']['orderings_reversed']}")
    print(f"  smallest-relevant-effect check: observed={s5['smallest_relevant_effect_check']['observed_fractional_improvement']:.4%} "
          f"threshold={s5['smallest_relevant_effect_check']['threshold']:.2%} "
          f"exceeds={s5['smallest_relevant_effect_check']['exceeds_threshold']}")
    print(f"S4 negative control: remains_negative_control={s4['remains_negative_control']}")
    print(f"\nwrote {out_root.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
