"""Synthetic-only tests for the post-freeze CRESST sensitivity analysis."""

from __future__ import annotations

import numpy as np

from src.noise_geometry.cresst.detector_metrics import paired_bootstrap_stats
from src.noise_geometry.cresst.negative_controls import CONTROL_NOMINAL
from src.noise_geometry.cresst.sensitivity import (
    build_estimands,
    event_table_as_structured,
    hierarchical_channel_bootstrap,
    sensitivity_summary,
    structured_as_event_table,
    validate_event_table,
    verify_frozen_reproduction,
)


def _events(n_per_channel: int = 6) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    labels = np.repeat([f"run_channel_{i}" for i in range(7)], n_per_channel)
    n = labels.size
    empca = rng.uniform(1.0, 3.0, n)
    pca = empca + np.repeat(np.linspace(-0.2, 0.5, 7), n_per_channel) + rng.normal(0, 0.02, n)
    of = empca + np.repeat(np.linspace(0.4, 1.2, 7), n_per_channel) + rng.normal(0, 0.03, n)
    multi = empca + np.repeat(np.linspace(-0.1, 0.3, 7), n_per_channel) + rng.normal(0, 0.02, n)
    events: dict[str, np.ndarray] = {
        "event_id": np.asarray([f"test:row:{i}" for i in range(n)]),
        "release_row_index": np.arange(n, dtype=np.int64),
        "trace_sha256": np.asarray([f"{i:064x}" for i in range(n)]),
        "run": np.repeat([str(i) for i in range(7)], n_per_channel),
        "channel": np.repeat([str(i + 10) for i in range(7)], n_per_channel),
        "channel_label": labels,
        "chronological_time": np.arange(n, dtype=float),
        "split_label": np.full(n, "evaluation"),
        "event_weight": np.ones(n),
        "of_weighted_residual": of,
        "multi_template_of_weighted_residual": multi,
        "pca_weighted_residual": pca,
        "empca_weighted_residual": empca,
    }
    for ci, name in enumerate(CONTROL_NOMINAL):
        events[f"control_{name}_weighted_residual"] = empca + 1.5 + 0.1 * ci + rng.normal(0, 0.02, n)
    return events


def test_event_table_validation_and_lossless_roundtrip():
    events = _events()
    report = validate_event_table(events, expected_events=42)
    assert report["n_channels"] == 7
    assert report["duplicate_event_ids"] == 0
    table = event_table_as_structured(events)
    restored = structured_as_event_table(table)
    assert table.shape == (42,)
    for name in events:
        np.testing.assert_array_equal(restored[name], events[name])


def test_estimands_use_empca_for_all_sensitivity_controls():
    events = _events()
    estimands = build_estimands(events, competitiveness_tolerance=1.1)
    np.testing.assert_allclose(
        estimands["d_control_mismatched_of_template"],
        events["control_mismatched_of_template_weighted_residual"] - events["empca_weighted_residual"],
    )
    np.testing.assert_allclose(
        estimands["d_comp"],
        1.1 * events["multi_template_of_weighted_residual"] - events["empca_weighted_residual"],
    )


def test_hierarchical_bootstrap_is_reproducible_and_resamples_duplicate_occurrences_independently():
    events = _events()
    estimands = build_estimands(events)
    args = (estimands, events["channel_label"], events["event_weight"])
    names_a, draws_a = hierarchical_channel_bootstrap(*args, replicates=80, seed=99)
    names_b, draws_b = hierarchical_channel_bootstrap(*args, replicates=80, seed=99)
    assert names_a == names_b
    np.testing.assert_array_equal(draws_a, draws_b)
    assert draws_a.shape == (80, len(estimands))
    assert np.all(np.std(draws_a, axis=0) > 0)


def test_sensitivity_summary_reports_channel_balance_normalization_and_cluster_warning():
    events = _events()
    summary, names, draws = sensitivity_summary(
        events,
        competitiveness_tolerance=1.1,
        bootstrap_replicates=100,
        bootstrap_seed=20260713,
    )
    assert summary["n_channels"] == 7
    assert summary["hierarchical_bootstrap"]["replicates"] == 100
    assert "coarse and potentially unstable" in summary["cluster_caution"]
    assert summary["positive_channel_counts_out_of_seven"]["d_OF"] == 7
    assert set(summary["normalized_effects"]) == {
        "of_minus_empca_fraction",
        "multi_template_of_minus_empca_fraction",
        "pca_minus_empca_fraction",
    }
    assert draws.shape == (100, len(names))


def _archive_for(events: dict[str, np.ndarray]) -> tuple[dict, dict]:
    labels = events["channel_label"].astype(str)
    channels = np.unique(labels)
    metrics = {
        "rank": 6,
        "covariance_model": "psd_stationary",
        "n_duplicate_traces": 0,
        "n_traces": int(len(labels) * 2),
        "n_features": 32,
        "n_channels_total": 7,
        "n_channels_used": 7,
        "n_noise_traces": 70,
        "ae_empca_max_principal_angle_deg": 1e-8,
        "of_weighted_residual": float(np.mean(events["of_weighted_residual"])),
        "multi_template_of_weighted_residual": float(np.mean(events["multi_template_of_weighted_residual"])),
        "pca_weighted_residual": float(np.mean(events["pca_weighted_residual"])),
        "empca_weighted_residual": float(np.mean(events["empca_weighted_residual"])),
        "per_channel": {},
        "pooled": {"negative_controls": {}},
    }
    for ci, label in enumerate(channels):
        mask = labels == label
        entry = {
            "n_train_pulses": 10,
            "n_test_pulses": int(mask.sum()),
            "n_train_noise": 10,
            "n_background_noise": 5,
            "sigma_of": 0.1 + ci,
            "pca_raw_mse": 0.2 + ci,
            "empca_raw_mse": 0.1 + ci,
            "ae_empca_max_principal_angle_deg": 1e-9 + ci * 1e-10,
            "of_weighted_residual": float(np.mean(events["of_weighted_residual"][mask])),
            "pca_weighted_residual": float(np.mean(events["pca_weighted_residual"][mask])),
            "empca_weighted_residual": float(np.mean(events["empca_weighted_residual"][mask])),
            "multi_template_of": {
                "weighted_residual_mean": float(np.mean(events["multi_template_of_weighted_residual"][mask])),
                "n_templates": 3,
            },
            "of_vs_empca": paired_bootstrap_stats(
                events["of_weighted_residual"][mask], events["empca_weighted_residual"][mask],
                bootstrap_reps=300, seed=22 + ci,
            ),
            "empca_competitive_with_multi_template_of": paired_bootstrap_stats(
                1.1 * events["multi_template_of_weighted_residual"][mask],
                events["empca_weighted_residual"][mask], bootstrap_reps=300, seed=23 + ci,
            ),
            "negative_controls": {},
        }
        for name, nominal in CONTROL_NOMINAL.items():
            nominal_column = "empca_weighted_residual" if nominal == "empca" else "of_weighted_residual"
            entry["negative_controls"][name] = paired_bootstrap_stats(
                events[f"control_{name}_weighted_residual"][mask], events[nominal_column][mask],
                bootstrap_reps=300, seed=24 + ci,
            )
        metrics["per_channel"][label] = entry
    metrics["pooled"]["of_vs_empca"] = paired_bootstrap_stats(
        events["of_weighted_residual"], events["empca_weighted_residual"], bootstrap_reps=300, seed=22,
    )
    metrics["pooled"]["empca_competitive_with_multi_template_of"] = paired_bootstrap_stats(
        1.1 * events["multi_template_of_weighted_residual"], events["empca_weighted_residual"],
        bootstrap_reps=300, seed=23,
    )
    for name, nominal in CONTROL_NOMINAL.items():
        nominal_column = "empca_weighted_residual" if nominal == "empca" else "of_weighted_residual"
        metrics["pooled"]["negative_controls"][name] = paired_bootstrap_stats(
            events[f"control_{name}_weighted_residual"], events[nominal_column], bootstrap_reps=300, seed=24,
        )
    archive = {
        "git_commit": "9c9f696ff6bca6e6b9729cdb90e41ee92d576787",
        "metrics": metrics,
    }
    return metrics, archive


def test_frozen_reproduction_verifier_recomputes_archived_bootstraps_from_events():
    events = _events(n_per_channel=5)
    metrics, archive = _archive_for(events)
    report = verify_frozen_reproduction(events, metrics, archive, expected_events=35)
    assert report["passed"], report["failures"]
    assert report["n_failures"] == 0
    assert report["n_checks"] > 100
