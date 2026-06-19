"""Tests for the seed-sweep + bootstrap-CI harness and S5 held-out evaluation."""

import numpy as np
import pytest

from src.noise_geometry.experiments.synthetic import run_s5_metric_reversal
from src.noise_geometry.validation import (
    flatten_metrics,
    paired_difference,
    run_seed_sweep,
    summarize,
    train_test_split_indices,
)


# --------------------------------------------------------------------------- #
# Core harness behaviour
# --------------------------------------------------------------------------- #
def test_flatten_metrics_drops_nonnumeric_recurses_and_expands_lists():
    flat = flatten_metrics(
        {"a": 1.0, "name": "x", "ok": True, "runs": {"r1": {"gap": 2.0}}, "vec": [3, 4]}
    )
    # strings/bools dropped; nested dicts dotted; numeric lists expanded by index
    assert flat == {"a": 1.0, "runs.r1.gap": 2.0, "vec.0": 3.0, "vec.1": 4.0}


def test_summarize_is_deterministic_and_covers_the_mean():
    rng = np.random.default_rng(0)
    rows = [{"x": float(v)} for v in rng.normal(5.0, 1.0, size=200)]
    s1 = summarize(rows, seed=42)["x"]
    s2 = summarize(rows, seed=42)["x"]
    assert s1 == s2  # deterministic for a fixed bootstrap seed
    assert s1["ci95"][0] < s1["mean"] < s1["ci95"][1]
    assert s1["ci68"][0] >= s1["ci95"][0] and s1["ci68"][1] <= s1["ci95"][1]


def test_paired_difference_detects_a_clear_positive_shift():
    rows = [{"a": 10.0 + i * 0.0, "b": 9.0} for i in range(15)]
    pd = paired_difference(rows, "a", "b", seed=1)
    assert pd["mean_difference"] == pytest.approx(1.0)
    assert pd["fraction_positive"] == 1.0
    assert pd["ci95_excludes_zero"] is True


def test_paired_difference_includes_zero_for_noise():
    rng = np.random.default_rng(3)
    rows = [{"a": float(v), "b": float(w)} for v, w in zip(rng.normal(size=40), rng.normal(size=40))]
    pd = paired_difference(rows, "a", "b", seed=7)
    assert pd["ci95_excludes_zero"] is False


def test_train_test_split_is_disjoint_and_covers_all():
    train, test = train_test_split_indices(100, 0.3, seed=5)
    assert set(train).isdisjoint(test)
    assert sorted([*train, *test]) == list(range(100))
    assert 25 <= test.size <= 35


# --------------------------------------------------------------------------- #
# Seed sweep over a real experiment
# --------------------------------------------------------------------------- #
def test_seed_sweep_produces_one_row_per_seed():
    seeds = [0, 1, 2, 3, 4]
    rows = run_seed_sweep(run_s5_metric_reversal, {"n_traces": 256, "test_frac": 0.5}, seeds)
    assert len(rows) == len(seeds)
    assert [int(r["seed"]) for r in rows] == seeds
    assert all("pca_weighted_residual_to_observed" in r for r in rows)


def test_s5_heldout_reversal_ci_excludes_zero():
    """The flagship payoff: paired weighted-residual reversal with a 95% CI.

    Across independent seeds and held-out evaluation, weighted PCA must beat PCA
    on the weighted (likelihood) residual with a paired 95% interval excluding
    zero -- the roadmap's S5 acceptance form.
    """
    seeds = list(range(12))
    rows = run_seed_sweep(
        run_s5_metric_reversal, {"n_traces": 512, "test_frac": 0.5, "split_seed": 0}, seeds
    )
    # held-out actually used: ~half the 512 traces are test
    assert all(r["n_test"] == 256.0 for r in rows)

    # PCA wins raw MSE (paired, excludes zero)
    raw = paired_difference(
        rows, "weighted_pca_raw_residual_to_observed", "pca_raw_residual_to_observed", seed=0
    )
    assert raw["ci95_excludes_zero"] and raw["mean_difference"] > 0

    # Weighted PCA wins the weighted residual (paired, excludes zero) -> reversal
    weighted = paired_difference(
        rows, "pca_weighted_residual_to_observed", "weighted_pca_weighted_residual_to_observed", seed=0
    )
    assert weighted["mean_difference"] > 0
    assert weighted["ci95_excludes_zero"]
