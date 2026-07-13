"""Post-freeze CRESST channel-robustness sensitivity analysis.

The frozen primary result remains the pooled 300-replicate event bootstrap.
This module only verifies an event-level regeneration against that archive and
then computes explicitly labelled post-freeze, equal-channel summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .detector_metrics import paired_bootstrap_stats
from .negative_controls import CONTROL_NOMINAL


CORE_RESIDUAL_COLUMNS = (
    "of_weighted_residual",
    "multi_template_of_weighted_residual",
    "pca_weighted_residual",
    "empca_weighted_residual",
)


def control_columns(events: Mapping[str, np.ndarray]) -> list[str]:
    return sorted(
        name
        for name in events
        if name.startswith("control_") and name.endswith("_weighted_residual")
    )


def _control_name(column: str) -> str:
    return column.removeprefix("control_").removesuffix("_weighted_residual")


def validate_event_table(events: Mapping[str, np.ndarray], *, expected_events: int | None = None) -> dict[str, Any]:
    """Fail-closed structural checks for the regenerated event table."""
    required = {
        "event_id",
        "release_row_index",
        "trace_sha256",
        "run",
        "channel",
        "channel_label",
        "chronological_time",
        "split_label",
        "event_weight",
        *CORE_RESIDUAL_COLUMNS,
    }
    missing = sorted(required - set(events))
    if missing:
        raise ValueError(f"event table is missing required columns: {missing}")
    controls = control_columns(events)
    expected_controls = {f"control_{name}_weighted_residual" for name in CONTROL_NOMINAL}
    if set(controls) != expected_controls:
        raise ValueError(
            "event table negative-control columns differ from the frozen five: "
            f"found={controls}, expected={sorted(expected_controls)}"
        )
    lengths = {name: len(np.asarray(values)) for name, values in events.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"event columns have inconsistent lengths: {lengths}")
    n_events = next(iter(lengths.values()))
    if expected_events is not None and n_events != int(expected_events):
        raise ValueError(f"expected exactly {expected_events} evaluation pulses, found {n_events}")
    event_ids = np.asarray(events["event_id"]).astype(str)
    trace_hashes = np.asarray(events["trace_sha256"]).astype(str)
    duplicate_event_ids = int(n_events - np.unique(event_ids).size)
    duplicate_trace_hashes = int(n_events - np.unique(trace_hashes).size)
    if duplicate_event_ids or duplicate_trace_hashes:
        raise ValueError(
            "event table contains duplicates: "
            f"event_ids={duplicate_event_ids}, trace_hashes={duplicate_trace_hashes}"
        )
    if set(np.asarray(events["split_label"]).astype(str)) != {"evaluation"}:
        raise ValueError("event table must contain evaluation pulses only")
    weights = np.asarray(events["event_weight"], dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("event weights must be finite and strictly positive")
    numeric = ["chronological_time", "event_weight", *CORE_RESIDUAL_COLUMNS, *controls]
    nonfinite = {name: int(np.sum(~np.isfinite(np.asarray(events[name], dtype=np.float64)))) for name in numeric}
    nonfinite = {name: count for name, count in nonfinite.items() if count}
    if nonfinite:
        raise ValueError(f"event table contains non-finite numerical values: {nonfinite}")
    return {
        "n_events": n_events,
        "n_channels": int(np.unique(np.asarray(events["channel_label"]).astype(str)).size),
        "duplicate_event_ids": duplicate_event_ids,
        "duplicate_trace_hashes": duplicate_trace_hashes,
        "control_columns": controls,
        "all_event_weights_one": bool(np.all(weights == 1.0)),
    }


def event_table_as_structured(events: Mapping[str, np.ndarray]) -> np.ndarray:
    """Convert column arrays to a lossless one-row-per-event structured array."""
    columns = list(events)
    n_events = len(np.asarray(events[columns[0]]))
    dtype: list[tuple[str, Any]] = []
    converted: dict[str, np.ndarray] = {}
    for name in columns:
        values = np.asarray(events[name])
        if values.dtype.kind in {"U", "S", "O"}:
            text = values.astype(str)
            width = max(1, max((len(value) for value in text), default=1))
            converted[name] = text.astype(f"<U{width}")
            dtype.append((name, f"<U{width}"))
        elif values.dtype.kind in {"i", "u"}:
            converted[name] = values.astype("<i8", copy=False)
            dtype.append((name, "<i8"))
        else:
            converted[name] = values.astype("<f8", copy=False)
            dtype.append((name, "<f8"))
    table = np.empty(n_events, dtype=dtype)
    for name in columns:
        table[name] = converted[name]
    return table


def structured_as_event_table(table: np.ndarray) -> dict[str, np.ndarray]:
    if table.dtype.names is None:
        raise ValueError("lossless event array must be a structured NumPy array")
    return {name: np.asarray(table[name]) for name in table.dtype.names}


def _close(actual: float, expected: float, *, rtol: float, atol: float) -> bool:
    return bool(np.isclose(float(actual), float(expected), rtol=rtol, atol=atol, equal_nan=True))


def _record_check(
    checks: list[dict[str, Any]], path: str, actual: Any, expected: Any, *, rtol: float, atol: float
) -> None:
    if isinstance(expected, (list, tuple)):
        passed = len(actual) == len(expected) and all(
            _close(a, e, rtol=rtol, atol=atol) for a, e in zip(actual, expected)
        )
    elif isinstance(expected, (int, float, np.integer, np.floating)) and not isinstance(expected, bool):
        passed = _close(actual, expected, rtol=rtol, atol=atol)
    else:
        passed = actual == expected
    checks.append({"path": path, "actual": actual, "expected": expected, "passed": bool(passed)})


def verify_frozen_reproduction(
    events: Mapping[str, np.ndarray],
    regenerated_metrics: Mapping[str, Any],
    archived_record: Mapping[str, Any],
    *,
    expected_events: int = 19_335,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Verify event arrays and regenerated summaries against the frozen JSON."""
    structure = validate_event_table(events, expected_events=expected_events)
    archived = archived_record["metrics"]
    checks: list[dict[str, Any]] = []
    _record_check(checks, "archive.git_commit", archived_record.get("git_commit"),
                  "9c9f696ff6bca6e6b9729cdb90e41ee92d576787", rtol=rtol, atol=atol)
    _record_check(checks, "metrics.rank", regenerated_metrics["rank"], archived["rank"], rtol=rtol, atol=atol)
    _record_check(
        checks, "metrics.covariance_model", regenerated_metrics["covariance_model"],
        archived["covariance_model"], rtol=rtol, atol=atol,
    )
    _record_check(
        checks, "metrics.n_duplicate_traces", regenerated_metrics["n_duplicate_traces"],
        archived["n_duplicate_traces"], rtol=rtol, atol=atol,
    )
    for key in ("n_traces", "n_features", "n_channels_total", "n_channels_used", "n_noise_traces"):
        _record_check(
            checks, f"metrics.{key}", regenerated_metrics[key], archived[key], rtol=rtol, atol=atol
        )
    _record_check(
        checks, "metrics.ae_empca_max_principal_angle_deg",
        regenerated_metrics["ae_empca_max_principal_angle_deg"],
        archived["ae_empca_max_principal_angle_deg"], rtol=rtol, atol=atol,
    )
    _record_check(checks, "event_table.n_events", structure["n_events"], expected_events, rtol=rtol, atol=atol)
    _record_check(checks, "event_table.n_channels", structure["n_channels"], 7, rtol=rtol, atol=atol)
    _record_check(checks, "event_table.duplicate_event_ids", structure["duplicate_event_ids"], 0, rtol=rtol, atol=atol)
    _record_check(checks, "event_table.duplicate_trace_hashes", structure["duplicate_trace_hashes"], 0, rtol=rtol, atol=atol)

    weights = np.asarray(events["event_weight"], dtype=np.float64)
    channel_labels = np.asarray(events["channel_label"]).astype(str)
    for column in CORE_RESIDUAL_COLUMNS:
        actual = float(np.average(np.asarray(events[column], dtype=np.float64), weights=weights))
        _record_check(checks, f"event_table.pooled.{column}", actual, archived[column], rtol=rtol, atol=atol)
        _record_check(
            checks, f"regenerated_metrics.{column}", regenerated_metrics[column], archived[column],
            rtol=rtol, atol=atol,
        )

    control_cols = control_columns(events)
    for ci, (label, channel_archive) in enumerate(archived["per_channel"].items()):
        mask = channel_labels == label
        regenerated_channel = regenerated_metrics["per_channel"][label]
        _record_check(
            checks, f"per_channel.{label}.n_test_pulses", int(mask.sum()),
            channel_archive["n_test_pulses"], rtol=rtol, atol=atol,
        )
        for key in ("n_train_pulses", "n_test_pulses", "n_train_noise", "n_background_noise"):
            _record_check(
                checks, f"regenerated.per_channel.{label}.{key}", regenerated_channel[key],
                channel_archive[key], rtol=rtol, atol=atol,
            )
        for key in ("sigma_of", "pca_raw_mse", "empca_raw_mse", "ae_empca_max_principal_angle_deg"):
            _record_check(
                checks, f"regenerated.per_channel.{label}.{key}", regenerated_channel[key],
                channel_archive[key], rtol=rtol, atol=atol,
            )
        _record_check(
            checks, f"regenerated.per_channel.{label}.multi_template_of.n_templates",
            regenerated_channel["multi_template_of"]["n_templates"],
            channel_archive["multi_template_of"]["n_templates"], rtol=rtol, atol=atol,
        )
        for column in CORE_RESIDUAL_COLUMNS:
            archived_value = (
                channel_archive["multi_template_of"]["weighted_residual_mean"]
                if column == "multi_template_of_weighted_residual"
                else channel_archive[column]
            )
            actual = float(np.average(np.asarray(events[column], dtype=np.float64)[mask], weights=weights[mask]))
            _record_check(checks, f"per_channel.{label}.{column}", actual, archived_value, rtol=rtol, atol=atol)

        of_stats = paired_bootstrap_stats(
            np.asarray(events["of_weighted_residual"], dtype=np.float64)[mask],
            np.asarray(events["empca_weighted_residual"], dtype=np.float64)[mask],
            bootstrap_reps=300,
            seed=22 + ci,
        )
        comp_stats = paired_bootstrap_stats(
            1.1 * np.asarray(events["multi_template_of_weighted_residual"], dtype=np.float64)[mask],
            np.asarray(events["empca_weighted_residual"], dtype=np.float64)[mask],
            bootstrap_reps=300,
            seed=23 + ci,
        )
        for key, actual_stats, expected_stats in (
            ("of_vs_empca", of_stats, channel_archive["of_vs_empca"]),
            ("empca_competitive_with_multi_template_of", comp_stats,
             channel_archive["empca_competitive_with_multi_template_of"]),
        ):
            _record_check(checks, f"per_channel.{label}.{key}.mean_difference",
                          actual_stats["mean_difference"], expected_stats["mean_difference"], rtol=rtol, atol=atol)
            _record_check(checks, f"per_channel.{label}.{key}.ci95_difference",
                          actual_stats["ci95_difference"], expected_stats["ci95_difference"], rtol=rtol, atol=atol)

        for column in control_cols:
            name = _control_name(column)
            nominal_column = (
                "empca_weighted_residual" if CONTROL_NOMINAL[name] == "empca" else "of_weighted_residual"
            )
            stats = paired_bootstrap_stats(
                np.asarray(events[column], dtype=np.float64)[mask],
                np.asarray(events[nominal_column], dtype=np.float64)[mask],
                bootstrap_reps=300,
                seed=24 + ci,
            )
            expected_stats = channel_archive["negative_controls"][name]
            _record_check(checks, f"per_channel.{label}.negative_controls.{name}.mean_difference",
                          stats["mean_difference"], expected_stats["mean_difference"], rtol=rtol, atol=atol)
            _record_check(checks, f"per_channel.{label}.negative_controls.{name}.ci95_difference",
                          stats["ci95_difference"], expected_stats["ci95_difference"], rtol=rtol, atol=atol)

    pooled_comparisons = {
        "of_vs_empca": paired_bootstrap_stats(
            events["of_weighted_residual"], events["empca_weighted_residual"],
            bootstrap_reps=300, seed=22,
        ),
        "empca_competitive_with_multi_template_of": paired_bootstrap_stats(
            1.1 * np.asarray(events["multi_template_of_weighted_residual"], dtype=np.float64),
            events["empca_weighted_residual"], bootstrap_reps=300, seed=23,
        ),
    }
    for key, stats in pooled_comparisons.items():
        expected_stats = archived["pooled"][key]
        _record_check(checks, f"pooled.{key}.mean_difference", stats["mean_difference"],
                      expected_stats["mean_difference"], rtol=rtol, atol=atol)
        _record_check(checks, f"pooled.{key}.ci95_difference", stats["ci95_difference"],
                      expected_stats["ci95_difference"], rtol=rtol, atol=atol)
    for column in control_cols:
        name = _control_name(column)
        nominal_column = "empca_weighted_residual" if CONTROL_NOMINAL[name] == "empca" else "of_weighted_residual"
        stats = paired_bootstrap_stats(
            events[column], events[nominal_column], bootstrap_reps=300, seed=24,
        )
        expected_stats = archived["pooled"]["negative_controls"][name]
        _record_check(checks, f"pooled.negative_controls.{name}.mean_difference",
                      stats["mean_difference"], expected_stats["mean_difference"], rtol=rtol, atol=atol)
        _record_check(checks, f"pooled.negative_controls.{name}.ci95_difference",
                      stats["ci95_difference"], expected_stats["ci95_difference"], rtol=rtol, atol=atol)

    failures = [check for check in checks if not check["passed"]]
    return {
        "analysis_label": "post-freeze sensitivity analysis",
        "passed": not failures,
        "rtol": rtol,
        "atol": atol,
        "structure": structure,
        "n_checks": len(checks),
        "n_failures": len(failures),
        "failures": failures,
        "checks": checks,
    }


def build_estimands(events: Mapping[str, np.ndarray], *, competitiveness_tolerance: float = 1.1) -> dict[str, np.ndarray]:
    empca = np.asarray(events["empca_weighted_residual"], dtype=np.float64)
    estimands = {
        "d_OF": np.asarray(events["of_weighted_residual"], dtype=np.float64) - empca,
        "d_comp": competitiveness_tolerance
        * np.asarray(events["multi_template_of_weighted_residual"], dtype=np.float64)
        - empca,
        "d_multi_raw": np.asarray(events["multi_template_of_weighted_residual"], dtype=np.float64) - empca,
        "d_PCA": np.asarray(events["pca_weighted_residual"], dtype=np.float64) - empca,
    }
    for column in control_columns(events):
        estimands[f"d_control_{_control_name(column)}"] = np.asarray(events[column], dtype=np.float64) - empca
    return estimands


def hierarchical_channel_bootstrap(
    estimands: Mapping[str, np.ndarray],
    channel_labels: np.ndarray,
    event_weights: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[list[str], np.ndarray]:
    """Two-level bootstrap with independent event resamples per channel occurrence."""
    names = list(estimands)
    values = np.column_stack([np.asarray(estimands[name], dtype=np.float64) for name in names])
    labels = np.asarray(channel_labels).astype(str)
    weights = np.asarray(event_weights, dtype=np.float64)
    channels = np.unique(labels)
    if channels.size < 2:
        raise ValueError("hierarchical bootstrap requires at least two observed channels")
    rows_by_channel = [np.flatnonzero(labels == channel) for channel in channels]
    rng = np.random.default_rng(int(seed))
    draws = np.empty((int(replicates), len(names)), dtype=np.float64)
    for bootstrap_index in range(int(replicates)):
        sampled_channels = rng.integers(0, channels.size, size=channels.size)
        occurrence_means = np.empty((channels.size, len(names)), dtype=np.float64)
        for occurrence, channel_index in enumerate(sampled_channels):
            rows = rows_by_channel[int(channel_index)]
            sampled_rows = rows[rng.integers(0, rows.size, size=rows.size)]
            occurrence_means[occurrence] = np.average(
                values[sampled_rows], axis=0, weights=weights[sampled_rows]
            )
        draws[bootstrap_index] = occurrence_means.mean(axis=0)
    return names, draws


def sensitivity_summary(
    events: Mapping[str, np.ndarray],
    *,
    competitiveness_tolerance: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[str], np.ndarray]:
    estimands = build_estimands(events, competitiveness_tolerance=competitiveness_tolerance)
    labels = np.asarray(events["channel_label"]).astype(str)
    weights = np.asarray(events["event_weight"], dtype=np.float64)
    channels = np.unique(labels)
    channel_means: dict[str, dict[str, float]] = {}
    normalized: dict[str, dict[str, float]] = {}
    for channel in channels:
        mask = labels == channel
        channel_means[channel] = {
            name: float(np.average(values[mask], weights=weights[mask]))
            for name, values in estimands.items()
        }
        method_means = {
            column: float(np.average(np.asarray(events[column], dtype=np.float64)[mask], weights=weights[mask]))
            for column in CORE_RESIDUAL_COLUMNS
        }
        empca = method_means["empca_weighted_residual"]
        normalized[channel] = {
            "of_minus_empca_fraction": 1.0 - empca / method_means["of_weighted_residual"],
            "multi_template_of_minus_empca_fraction": (
                1.0 - empca / method_means["multi_template_of_weighted_residual"]
            ),
            "pca_minus_empca_fraction": 1.0 - empca / method_means["pca_weighted_residual"],
        }
    bootstrap_names, draws = hierarchical_channel_bootstrap(
        estimands,
        labels,
        weights,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    intervals = np.percentile(draws, [2.5, 97.5], axis=0)
    estimand_summary: dict[str, Any] = {}
    for index, name in enumerate(bootstrap_names):
        per_channel = [channel_means[channel][name] for channel in channels]
        estimand_summary[name] = {
            "pooled_event_weighted_mean": float(np.average(estimands[name], weights=weights)),
            "equal_channel_mean": float(np.mean(per_channel)),
            "hierarchical_ci95": [float(intervals[0, index]), float(intervals[1, index])],
            "channel_means": {channel: channel_means[channel][name] for channel in channels},
        }
    normalized_summary: dict[str, Any] = {}
    for effect_name in next(iter(normalized.values())):
        values = np.asarray([normalized[channel][effect_name] for channel in channels])
        normalized_summary[effect_name] = {
            "channel_values": {channel: normalized[channel][effect_name] for channel in channels},
            "range": [float(np.min(values)), float(np.max(values))],
            "median": float(np.median(values)),
            "equal_channel_mean": float(np.mean(values)),
        }
    positive_counts = {
        name: int(sum(channel_means[channel][name] > 0 for channel in channels))
        for name in ("d_OF", "d_comp", "d_PCA")
    }
    summary = {
        "analysis_label": "post-freeze sensitivity analysis",
        "primary_analysis_status": (
            "The archived pooled 300-replicate event bootstrap remains the predeclared primary analysis."
        ),
        "cluster_caution": (
            "Only seven observed run/channel clusters are available; cluster-level percentile intervals "
            "are coarse and potentially unstable."
        ),
        "n_events": int(len(labels)),
        "n_channels": int(len(channels)),
        "channels": channels.tolist(),
        "competitiveness_tolerance": float(competitiveness_tolerance),
        "hierarchical_bootstrap": {
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "interval": "percentile 95%",
            "channel_sampling": "seven channels with replacement",
            "event_sampling": "observed event count with replacement independently per channel occurrence",
        },
        "estimands": estimand_summary,
        "normalized_effects": normalized_summary,
        "positive_channel_counts_out_of_seven": positive_counts,
        "sign_test": "not performed",
    }
    return summary, bootstrap_names, draws
