#!/usr/bin/env python3
"""Regenerate and audit the frozen CRESST event residuals, then run sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noise_geometry.cresst import run_cresst_experiment_with_event_data
from noise_geometry.cresst.sensitivity import (
    event_table_as_structured,
    sensitivity_summary,
    validate_event_table,
    verify_frozen_reproduction,
)
from noise_geometry.utils import load_config, resolve_data_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        path.write_text("", encoding="utf-8")
        return
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        line = str(message).rstrip()
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _environment_record() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "pandas", "matplotlib", "h5py", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = _git(["rev-parse", "HEAD"])
        status = _git(["status", "--short"])
    except Exception as exc:
        commit, status = None, f"unavailable: {exc}"
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "execution_git_commit": commit,
        "git_status_short": status.splitlines(),
        "cwd": os.getcwd(),
    }


def _require_frozen_contract(config: dict[str, Any], source: dict[str, Any], archived: dict[str, Any]) -> None:
    expected = {
        "experiment_id": "P2_CRESST_VALIDATION",
        "seed": 22,
        "rank": 6,
        "covariance_model": "psd_stationary",
        "covariance_shrinkage": 0.1,
        "split": "test",
        "channel_calibration_fraction": 0.7,
        "channel_columns": ["run", "channel"],
        "channel_time_column": "hours",
        "multi_template_of_competitive_tolerance": 1.1,
        "bootstrap_reps": 300,
        "max_traces": None,
        "max_features": 512,
        "baseline_fraction": 0.2,
    }
    discrepancies = {
        key: {"actual": source.get(key), "expected": value}
        for key, value in expected.items()
        if source.get(key) != value
    }
    if discrepancies:
        raise RuntimeError(f"frozen source configuration changed: {discrepancies}")
    expected_commit = config["frozen_source_commit"]
    if archived.get("git_commit") != expected_commit:
        raise RuntimeError(
            f"archived run commit {archived.get('git_commit')} != configured frozen source {expected_commit}"
        )
    if archived.get("experiment_id") != "P2_CRESST_VALIDATION":
        raise RuntimeError(f"unexpected archived experiment_id: {archived.get('experiment_id')}")


def _write_event_csv(path: Path, events: dict[str, np.ndarray]) -> None:
    columns = list(events)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(len(events[columns[0]])):
            row = []
            for column in columns:
                value = events[column][index]
                if isinstance(value, (float, np.floating)):
                    row.append(format(float(value), ".10g"))
                else:
                    row.append(value.item() if isinstance(value, np.generic) else value)
            writer.writerow(row)


def _write_summary_csvs(output: Path, summary: dict[str, Any]) -> None:
    channel_rows = []
    normalized = summary["normalized_effects"]
    for channel in summary["channels"]:
        row = {"channel": channel}
        for name, stats in normalized.items():
            row[name] = stats["channel_values"][channel]
        for name, stats in summary["estimands"].items():
            row[name] = stats["channel_means"][channel]
        channel_rows.append(row)
    _write_csv(output / "channel_summary.csv", channel_rows)

    estimand_rows = []
    for name, stats in summary["estimands"].items():
        estimand_rows.append(
            {
                "estimand": name,
                "pooled_event_weighted_mean": stats["pooled_event_weighted_mean"],
                "equal_channel_mean": stats["equal_channel_mean"],
                "hierarchical_ci95_lo": stats["hierarchical_ci95"][0],
                "hierarchical_ci95_hi": stats["hierarchical_ci95"][1],
            }
        )
    _write_csv(output / "estimand_summary.csv", estimand_rows)

    paper_rows = []
    for row in channel_rows:
        paper_rows.append(
            {
                "row_type": "channel_normalized_effect",
                "label": row["channel"],
                "normalized_OF_minus_EMPCA": row["of_minus_empca_fraction"],
                "normalized_PCA_minus_EMPCA": row["pca_minus_empca_fraction"],
                "pooled_event_weighted_mean": "",
                "equal_channel_mean": "",
                "hierarchical_ci95_lo": "",
                "hierarchical_ci95_hi": "",
            }
        )
    for row in estimand_rows:
        paper_rows.append(
            {
                "row_type": "estimand_summary",
                "label": row["estimand"],
                "normalized_OF_minus_EMPCA": "",
                "normalized_PCA_minus_EMPCA": "",
                "pooled_event_weighted_mean": row["pooled_event_weighted_mean"],
                "equal_channel_mean": row["equal_channel_mean"],
                "hierarchical_ci95_lo": row["hierarchical_ci95_lo"],
                "hierarchical_ci95_hi": row["hierarchical_ci95_hi"],
            }
        )
    _write_csv(output / "paper_facing_table.csv", paper_rows)


def _write_figure(output: Path, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    channels = summary["channels"]
    normalized = summary["normalized_effects"]
    of_values = np.asarray(
        [normalized["of_minus_empca_fraction"]["channel_values"][channel] * 100 for channel in channels]
    )
    pca_values = np.asarray(
        [normalized["pca_minus_empca_fraction"]["channel_values"][channel] * 100 for channel in channels]
    )
    y = np.arange(len(channels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].axvline(0, color="0.55", linewidth=0.8)
    axes[0].scatter(of_values, y - 0.13, label="OF − EMPCA", marker="o")
    axes[0].scatter(pca_values, y + 0.13, label="PCA − EMPCA", marker="s")
    axes[0].set_yticks(y, channels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("EMPCA residual reduction (%)")
    axes[0].set_title("Normalized effects by channel")
    axes[0].legend(frameon=False)

    core = ["d_OF", "d_PCA"]
    yy = np.arange(len(core))
    for index, name in enumerate(core):
        stats = summary["estimands"][name]
        lo, hi = stats["hierarchical_ci95"]
        equal = stats["equal_channel_mean"]
        pooled = stats["pooled_event_weighted_mean"]
        axes[1].plot([lo, hi], [index, index], color="C0", linewidth=2)
        axes[1].scatter([equal], [index], color="C0", marker="o", label="equal-channel + hierarchical CI" if index == 0 else None)
        axes[1].scatter([pooled], [index], color="C1", marker="D", label="frozen event-pooled point" if index == 0 else None)
    axes[1].axvline(0, color="0.55", linewidth=0.8)
    axes[1].set_yticks(yy, core)
    axes[1].invert_yaxis()
    axes[1].set_xscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("Paired weighted-residual difference")
    axes[1].set_title("Pooled vs channel-balanced sensitivity")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("CRESST post-freeze robustness (seven channels)")
    fig.savefig(output / "paper_figure.png", dpi=220)
    fig.savefig(output / "paper_figure.pdf")
    plt.close(fig)


def _leakage_markdown(audit: dict[str, Any], reproduction: dict[str, Any]) -> str:
    channel_lines = []
    for channel in audit["channels"]:
        channel_lines.append(
            f"| {channel['channel_label']} | {channel['pulse_calibration_count']} | "
            f"{channel['pulse_evaluation_count']} | {channel['noise_calibration_count']} | "
            f"{channel['noise_evaluation_count']} | {channel['calibration_evaluation_row_overlap_count']} | "
            f"{channel['calibration_evaluation_trace_hash_overlap_count']} |"
        )
    return "\n".join(
        [
            "# CRESST post-freeze reproduction and leakage audit",
            "",
            "This document reports a post-freeze sensitivity analysis. The archived pooled event bootstrap remains primary.",
            "",
            f"- Frozen reproduction passed: **{reproduction['passed']}** ({reproduction['n_checks']} checks; {reproduction['n_failures']} failures).",
            f"- Evaluation pulses: **{audit['n_evaluation_pulses']}** across **{audit['n_channels']}** channels.",
            f"- Full-release duplicate traces: **{audit['n_duplicate_full_release_traces']}**.",
            f"- All within-channel pulse/noise splits chronological: **{audit['all_channel_splits_chronological']}**.",
            f"- Calibration/evaluation source-row overlap: **{audit['total_calibration_evaluation_row_overlap_count']}**.",
            f"- Calibration/evaluation release-trace-hash overlap: **{audit['total_calibration_evaluation_trace_hash_overlap_count']}**.",
            "- PSD and inverse-noise metric: calibration noise only.",
            "- PCA, EMPCA, and tied linear-AE fits: calibration pulses only.",
            "- Single-OF template and multi-template bank: calibration pulses only.",
            "- All comparators receive the same baseline subtraction, retained samples, and residual normalization.",
            f"- Residual units/normalization: {audit['weighted_residual_definition']}",
            "- Frozen rank, shrinkage, templates, shifts, exclusions, thresholds, and split fraction were not changed.",
            "",
            "| channel | calibration pulses | evaluation pulses | calibration noise | evaluation noise | row overlap | trace-hash overlap |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *channel_lines,
            "",
        ]
    )


def _claim_handoff(output: Path, summary: dict[str, Any], checksums: dict[str, str]) -> str:
    d_of = summary["estimands"]["d_OF"]
    d_pca = summary["estimands"]["d_PCA"]
    pca_positive = summary["positive_channel_counts_out_of_seven"]["d_PCA"]
    if d_of["hierarchical_ci95"][0] > 0:
        capacity_text = "The channel-balanced sensitivity analysis supports lower EMPCA residual than fixed-template OF."
    else:
        capacity_text = "The channel-balanced sensitivity analysis does not establish lower EMPCA residual than fixed-template OF."
    if d_pca["hierarchical_ci95"][0] > 0:
        geometry_text = "The matched-rank PCA contrast also favors EMPCA."
    elif d_pca["hierarchical_ci95"][1] < 0:
        geometry_text = "The matched-rank PCA contrast favors ordinary PCA."
    else:
        geometry_text = f"The matched-rank PCA contrast is inconclusive/heterogeneous ({pca_positive}/7 positive channel effects)."
    checksum_lines = [f"- `{name}` — `{digest}`" for name, digest in sorted(checksums.items())]
    estimand_lines = []
    for name, stats in summary["estimands"].items():
        lo, hi = stats["hierarchical_ci95"]
        estimand_lines.append(
            f"| {name} | {stats['pooled_event_weighted_mean']:.10g} | "
            f"{stats['equal_channel_mean']:.10g} | [{lo:.10g}, {hi:.10g}] |"
        )
    normalized = summary["normalized_effects"]
    normalized_lines = []
    for channel in summary["channels"]:
        normalized_lines.append(
            f"| {channel} | {normalized['of_minus_empca_fraction']['channel_values'][channel]:.10g} | "
            f"{normalized['multi_template_of_minus_empca_fraction']['channel_values'][channel]:.10g} | "
            f"{normalized['pca_minus_empca_fraction']['channel_values'][channel]:.10g} |"
        )
    return "\n".join(
        [
            "# Writing-agent handoff: CRESST post-freeze sensitivity",
            "",
            "## Claim recommendation",
            "",
            f"{capacity_text} {geometry_text} State that only seven clusters make the hierarchical interval coarse and potentially unstable. Keep the archived pooled 300-replicate event bootstrap as the predeclared primary result.",
            "",
            "The conclusion is limited to held-out within-channel linear reconstruction on the seven released channels. Do not claim cross-channel/detector transfer, detector sensitivity or threshold improvement, 19,335 independent detector-level replications, nonlinear-autoencoder evidence, a universal covariance-weighting advantage when the PCA contrast is heterogeneous, or replacement of the frozen primary gate.",
            "",
            "## Pooled-primary points, channel-balanced points, and hierarchical intervals",
            "",
            "| estimand | pooled event-weighted mean | equal-channel mean | hierarchical 95% interval |",
            "|---|---:|---:|---:|",
            *estimand_lines,
            "",
            "## Normalized channel effects",
            "",
            "Values are residual-reduction fractions, not percentages.",
            "",
            "| channel | OF − EMPCA | multi-template OF − EMPCA | PCA − EMPCA |",
            "|---|---:|---:|---:|",
            *normalized_lines,
            "",
            "## Reproduction status",
            "",
            "The event export reproduced the frozen archive within the configured tolerance; no reproduction discrepancy remained when this handoff was written.",
            "",
            "## Key output checksums",
            "",
            *checksum_lines,
            "",
            "See `summary.json`, `channel_summary.csv`, and `estimand_summary.csv` for exact numerical values; see `SHA256SUMS` for the complete output manifest.",
            "",
        ]
    )


def _finalize_manifest(output: Path, status: str, config: dict[str, Any]) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"})
    entries = {
        str(path.relative_to(output)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in files
    }
    _write_json(
        output / "manifest.json",
        {
            "status": status,
            "analysis_label": "post-freeze sensitivity analysis",
            "frozen_source_commit": config.get("frozen_source_commit"),
            "files": entries,
        },
    )
    checksummed = [*files, output / "manifest.json"]
    (output / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(path)}  {path.relative_to(output)}" for path in checksummed) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path, data_root_arg: Path | None, output_arg: Path | None) -> Path:
    config = load_config(config_path)
    output = output_arg or REPO_ROOT / str(config["output_directory"])
    output = output if output.is_absolute() else REPO_ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite versioned sensitivity directory: {output}")
    output.mkdir(parents=True)
    log = RunLog(output / "console.log")
    shutil.copy2(config_path, output / "analysis_config.yaml")
    shutil.copy2(Path(__file__), output / "analysis_script.py")
    _write_json(output / "environment.json", _environment_record())
    status = "failed"
    try:
        source_config_path = REPO_ROOT / str(config["frozen_config"])
        archive_path = REPO_ROOT / str(config["archived_primary_json"])
        for path in (source_config_path, archive_path):
            if not path.is_file():
                raise FileNotFoundError(f"required frozen input is missing: {path}")
        expected_archive_sha = str(config["archived_primary_sha256"])
        if _sha256(archive_path) != expected_archive_sha:
            raise RuntimeError(
                f"archived JSON checksum mismatch: actual={_sha256(archive_path)}, expected={expected_archive_sha}"
            )
        expected_config_sha = str(config["frozen_config_sha256"])
        if _sha256(source_config_path) != expected_config_sha:
            raise RuntimeError(
                f"frozen config checksum mismatch: actual={_sha256(source_config_path)}, expected={expected_config_sha}"
            )
        source_config = load_config(source_config_path)
        archived = json.loads(archive_path.read_text(encoding="utf-8"))
        _require_frozen_contract(config, source_config, archived)
        data_root = resolve_data_root(data_root_arg, source_config)
        log(f"analysis_label=post-freeze sensitivity analysis")
        log(f"frozen_source_commit={config['frozen_source_commit']}")
        log(f"data_root={data_root}")
        log("regenerating frozen metrics and event residuals through one shared numerical path")
        regenerated, events, leakage = run_cresst_experiment_with_event_data(source_config, data_root)
        _write_json(output / "regenerated_primary_metrics.json", regenerated)
        structure = validate_event_table(events, expected_events=int(config["expected_evaluation_pulses"]))
        log(f"event_table={structure}")
        event_table = event_table_as_structured(events)
        np.savez_compressed(output / "event_residuals_lossless.npz", events=event_table)
        _write_event_csv(output / "event_residuals_compact.csv", events)
        _write_json(
            output / "event_residuals_metadata.json",
            {
                "format": "NumPy compressed archive containing one structured array named 'events'",
                "rows": len(event_table),
                "columns": list(event_table.dtype.names or ()),
                "lossless": "All floats are little-endian IEEE-754 float64; strings use fixed-width Unicode fields.",
                "compact_csv": "Human-readable view rounded to 10 significant decimal digits; use NPZ for analysis.",
                "trace_sha256": "SHA-256 of the unprocessed release trace represented as canonical little-endian float64 values.",
            },
        )
        reproduction = verify_frozen_reproduction(
            events,
            regenerated,
            archived,
            expected_events=int(config["expected_evaluation_pulses"]),
            rtol=float(config["reproduction_rtol"]),
            atol=float(config["reproduction_atol"]),
        )
        _write_json(output / "reproduction_audit.json", reproduction)
        _write_json(output / "leakage_audit.json", leakage)
        (output / "reproduction_and_leakage_audit.md").write_text(
            _leakage_markdown(leakage, reproduction), encoding="utf-8"
        )
        if not reproduction["passed"]:
            raise RuntimeError(
                "frozen event-level reproduction failed; sensitivity analysis stopped. "
                f"See {output / 'reproduction_audit.json'}"
            )
        leakage_failures = {
            "n_channels": leakage["n_channels"] != 7,
            "n_evaluation_pulses": leakage["n_evaluation_pulses"] != int(config["expected_evaluation_pulses"]),
            "duplicate_full_release_traces": leakage["n_duplicate_full_release_traces"] != 0,
            "nonchronological_split": not leakage["all_channel_splits_chronological"],
            "row_overlap": leakage["total_calibration_evaluation_row_overlap_count"] != 0,
            "trace_hash_overlap": leakage["total_calibration_evaluation_trace_hash_overlap_count"] != 0,
        }
        leakage_failures = {name: failed for name, failed in leakage_failures.items() if failed}
        if leakage_failures:
            raise RuntimeError(f"leakage audit failed; sensitivity analysis stopped: {leakage_failures}")
        log("frozen reproduction and leakage audits passed")
        summary, bootstrap_names, draws = sensitivity_summary(
            events,
            competitiveness_tolerance=float(config["competitiveness_tolerance"]),
            bootstrap_replicates=int(config["hierarchical_bootstrap_replicates"]),
            bootstrap_seed=int(config["hierarchical_bootstrap_seed"]),
        )
        summary["frozen_source_commit"] = config["frozen_source_commit"]
        summary["archived_primary_json"] = str(config["archived_primary_json"])
        summary["archived_primary_sha256"] = expected_archive_sha
        _write_json(output / "summary.json", summary)
        np.savez_compressed(
            output / "hierarchical_bootstrap_draws.npz",
            estimand_names=np.asarray(bootstrap_names),
            draws=draws,
        )
        _write_summary_csvs(output, summary)
        _write_figure(output, summary)
        key_names = [
            "event_residuals_lossless.npz",
            "summary.json",
            "channel_summary.csv",
            "estimand_summary.csv",
            "reproduction_audit.json",
            "reproduction_and_leakage_audit.md",
            "paper_facing_table.csv",
            "paper_figure.pdf",
        ]
        key_checksums = {name: _sha256(output / name) for name in key_names}
        (output / "WRITING_AGENT_HANDOFF.md").write_text(
            _claim_handoff(output, summary, key_checksums), encoding="utf-8"
        )
        status = "complete"
        log("sensitivity analysis complete")
        log(f"output={output}")
    except Exception:
        log("ERROR: sensitivity run failed closed")
        log(traceback.format_exc())
        _write_json(output / "run_status.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        _finalize_manifest(output, status, config)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/cresst/cresst_postfreeze_sensitivity.yaml",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    run(args.config.resolve(), args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
