"""EXP-04: fast, no-sweep regression tests for the confirmatory-sweep script.

Locks in the JSON-schema guardrail (a 'x times' claim may only cite the
``ratio`` block) and the seed-independence predeclaration, without paying
the cost of the full >=30-seed sweep (~1 minute) in the default test suite.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_s5_confirmatory_sweep.py"

spec = importlib.util.spec_from_file_location("run_s5_confirmatory_sweep", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(REPO_ROOT / "src"))
spec.loader.exec_module(module)


def test_seeds_are_at_least_30_and_disjoint_from_frozen_block():
    assert len(module.SEEDS) >= 30
    assert len(set(module.SEEDS)) == len(module.SEEDS), "seeds must be unique"
    frozen_block = set(range(10))  # the frozen 10-seed S5/S4 results use seeds 0-9
    assert set(module.SEEDS).isdisjoint(frozen_block)


def test_add_ratio_fields_computes_expected_schema_and_values():
    rows = [
        {
            "pca_weighted_residual_to_observed": 48.49,
            "weighted_pca_weighted_residual_to_observed": 47.945,
            "pca_raw_residual_to_observed": 100.0,
            "weighted_pca_raw_residual_to_observed": 300.0,
        }
    ]
    module._add_ratio_fields(rows)
    row = rows[0]
    assert "pca_over_empca_weighted_residual_ratio" in row
    assert "pca_over_empca_raw_residual_ratio" in row
    assert abs(row["pca_over_empca_weighted_residual_ratio"] - (48.49 / 47.945)) < 1e-9
    assert abs(row["pca_over_empca_raw_residual_ratio"] - (100.0 / 300.0)) < 1e-9


def test_add_ratio_fields_handles_zero_denominator_without_raising():
    rows = [
        {
            "pca_weighted_residual_to_observed": 1.0,
            "weighted_pca_weighted_residual_to_observed": 0.0,
        }
    ]
    module._add_ratio_fields(rows)
    import math

    assert math.isnan(rows[0]["pca_over_empca_weighted_residual_ratio"])


def test_smallest_relevant_effect_threshold_is_predeclared_and_positive():
    assert module.SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT > 0.0
    assert module.SMALLEST_RELEVANT_WEIGHTED_RESIDUAL_FRACTIONAL_IMPROVEMENT < 0.05
