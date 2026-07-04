"""Refresh the lightweight CRESST writing/plotting data bundle.

Sibling to refresh_bundle.py (GWOSC-only); CRESST gets its own script rather
than being folded into that file. Copies the named run's JSON/config/log
into transfer_paper/data/cresst/current/ and writes derived per-channel and
pooled CSV summaries under transfer_paper/data/derived/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TRANSFER_ROOT.parent
DATA_ROOT = TRANSFER_ROOT / "data"
CRESST_ROOT = DATA_ROOT / "cresst"
DERIVED_ROOT = DATA_ROOT / "derived"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_files(run_stub: str) -> dict[str, Path]:
    metrics_dir = REPO_ROOT / "results" / "metrics"
    files = {
        "json": metrics_dir / f"{run_stub}.json",
        "config": metrics_dir / f"{run_stub}.config.yaml",
        "log": metrics_dir / f"{run_stub}.log",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise SystemExit(f"missing run files: {missing}")
    return files


def _copy_run_files(files: dict[str, Path], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for _, source in files.items():
        target = destination / source.name
        shutil.copyfile(source, target)
        checksums[source.name] = _sha256(target)
    return checksums


def _per_channel_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for label, e in metrics["per_channel"].items():
        if e.get("skipped"):
            rows.append({"channel": label, "skipped": True, "reason": e.get("reason", "")})
            continue
        row: dict[str, Any] = {
            "channel": label,
            "skipped": False,
            "n_train_pulses": e["n_train_pulses"],
            "n_test_pulses": e["n_test_pulses"],
            "n_train_noise": e["n_train_noise"],
            "n_background_noise": e["n_background_noise"],
            "sigma_of": e["sigma_of"],
            "sigma_of_source": e["sigma_of_source"],
            "pca_weighted_residual": e["pca_weighted_residual"],
            "empca_weighted_residual": e["empca_weighted_residual"],
            "of_weighted_residual": e["of_weighted_residual"],
            "multi_template_of_weighted_residual": e["multi_template_of"]["weighted_residual_mean"],
            "of_vs_empca_mean_difference": e["of_vs_empca"]["mean_difference"],
            "of_vs_empca_ci95_lo": e["of_vs_empca"]["ci95_difference"][0],
            "of_vs_empca_ci95_hi": e["of_vs_empca"]["ci95_difference"][1],
            "of_vs_empca_exceeds_zero": e["of_vs_empca"]["exceeds_zero"],
            "competitive_mean_difference": e["empca_competitive_with_multi_template_of"]["mean_difference"],
            "competitive_exceeds_zero": e["empca_competitive_with_multi_template_of"]["exceeds_zero"],
        }
        for name, stats in e["negative_controls"].items():
            row[f"control_{name}_mean_difference"] = stats["mean_difference"]
            row[f"control_{name}_exceeds_zero"] = stats["exceeds_zero"]
        rows.append(row)
    return rows


def _pooled_row(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    acc = metrics["acceptance"]
    pooled = metrics["pooled"]
    row: dict[str, Any] = {
        "experiment": metrics.get("experiment"),
        "rank": metrics.get("rank"),
        "covariance_model": metrics.get("covariance_model"),
        "n_channels_total": metrics["n_channels_total"],
        "n_channels_used": metrics["n_channels_used"],
        "n_duplicate_traces": metrics["n_duplicate_traces"],
        "empca_beats_single_template_of": acc["empca_beats_single_template_of"],
        "empca_competitive_with_multi_template_of": acc["empca_competitive_with_multi_template_of"],
        "all_negative_controls_degraded": acc["all_negative_controls_degraded"],
        "passed": acc["passed"],
        "of_vs_empca_mean_difference": pooled["of_vs_empca"]["mean_difference"],
        "of_vs_empca_ci95_lo": pooled["of_vs_empca"]["ci95_difference"][0],
        "of_vs_empca_ci95_hi": pooled["of_vs_empca"]["ci95_difference"][1],
        "of_vs_empca_sign_reversed_channels": ";".join(pooled["of_vs_empca_sign_reversed_channels"]),
        "competitive_mean_difference": pooled["empca_competitive_with_multi_template_of"]["mean_difference"],
        "competitive_ci95_lo": pooled["empca_competitive_with_multi_template_of"]["ci95_difference"][0],
        "competitive_ci95_hi": pooled["empca_competitive_with_multi_template_of"]["ci95_difference"][1],
        "competitive_sign_reversed_channels": ";".join(pooled["empca_competitive_sign_reversed_channels"]),
    }
    for name, stats in pooled["negative_controls"].items():
        row[f"control_{name}_mean_difference"] = stats["mean_difference"]
        row[f"control_{name}_exceeds_zero"] = stats["exceeds_zero"]
        row[f"control_{name}_sign_reversed_channels"] = ";".join(
            pooled["negative_controls_sign_reversed_channels"].get(name, [])
        )
    return [row]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run stub under results/metrics/, e.g. p2_cresst_validation_seed22")
    args = parser.parse_args()

    files = _run_files(args.run)
    metrics = json.loads(files["json"].read_text(encoding="utf-8"))["metrics"]

    current_dir = CRESST_ROOT / "current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    checksums = _copy_run_files(files, current_dir / "cresst")
    manifest = {
        "run_stub": args.run,
        "base_commit": json.loads(files["json"].read_text(encoding="utf-8")).get("git_commit"),
        "sha256": checksums,
    }
    (current_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (current_dir / "SHA256SUMS").write_text(
        "\n".join(f"{v}  {k}" for k, v in sorted(checksums.items())) + "\n", encoding="utf-8"
    )

    _write_csv(DERIVED_ROOT / "cresst_per_channel_summary.csv", _per_channel_rows(metrics))
    _write_csv(DERIVED_ROOT / "cresst_pooled_summary.csv", _pooled_row(metrics))
    print(f"Refreshed {current_dir} and derived CRESST CSVs under {DERIVED_ROOT}")


if __name__ == "__main__":
    main()
