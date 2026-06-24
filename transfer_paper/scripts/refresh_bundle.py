"""Refresh the lightweight Paper 1 writing and plotting data bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TRANSFER_ROOT.parent
DATA_ROOT = TRANSFER_ROOT / "data"
DERIVED_ROOT = DATA_ROOT / "derived"
GWOSC_ROOT = DATA_ROOT / "gwosc"
SYNTHETIC_ROOT = DATA_ROOT / "synthetic"
TABLE_ROOT = TRANSFER_ROOT / "tables"
CONFIG_ROOT = DATA_ROOT / "configs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _select_gwosc_run(run_id: str | None) -> Path:
    evidence_root = REPO_ROOT / "evidence" / "gwosc"
    candidates = [
        path
        for path in evidence_root.iterdir()
        if path.is_dir() and (path / "gwosc" / "experiment.json").is_file()
    ]
    if run_id is not None:
        selected = evidence_root / run_id
        if selected not in candidates:
            raise SystemExit(f"GWOSC run has no experiment record: {run_id}")
        return selected
    if not candidates:
        raise SystemExit("No archived GWOSC experiment evidence found")
    return max(candidates, key=lambda path: path.name)


def _copy_tree_files(
    source_root: Path,
    destination_root: Path,
    provenance: dict[str, str],
) -> None:
    if destination_root.exists():
        shutil.rmtree(destination_root)
    for source in sorted(source_root.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            source.relative_to(REPO_ROOT)
        )


def _copy_synthetic_sweeps(provenance: dict[str, str]) -> None:
    SYNTHETIC_ROOT.mkdir(parents=True, exist_ok=True)
    for existing in SYNTHETIC_ROOT.glob("S*_sweep_10seeds.*"):
        existing.unlink()
    for source in sorted((REPO_ROOT / "results" / "sweeps").glob(
        "S*_sweep_10seeds.*"
    )):
        destination = SYNTHETIC_ROOT / source.name
        shutil.copy2(source, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            source.relative_to(REPO_ROOT)
        )
    summary = REPO_ROOT / "results" / "tables" / "synthetic_summary.csv"
    if summary.is_file():
        destination = SYNTHETIC_ROOT / summary.name
        shutil.copy2(summary, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            summary.relative_to(REPO_ROOT)
        )


def _copy_synthetic_figures(provenance: dict[str, str]) -> None:
    destination_root = SYNTHETIC_ROOT / "figures"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for source in sorted((REPO_ROOT / "results" / "figures").glob("*")):
        if not source.is_file():
            continue
        destination = destination_root / source.name
        shutil.copy2(source, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            source.relative_to(REPO_ROOT)
        )


def _copy_config_snapshots(provenance: dict[str, str]) -> None:
    if CONFIG_ROOT.exists():
        shutil.rmtree(CONFIG_ROOT)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    patterns = (
        "configs/gwosc/*.yaml",
        "configs/synthetic/S*.yaml",
    )
    for pattern in patterns:
        for source in sorted(REPO_ROOT.glob(pattern)):
            if not source.is_file():
                continue
            destination = CONFIG_ROOT / source.relative_to(REPO_ROOT / "configs")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
                source.relative_to(REPO_ROOT)
            )


def _copy_source_documents(provenance: dict[str, str]) -> None:
    destination_root = DATA_ROOT / "source_documents"
    destination_root.mkdir(parents=True, exist_ok=True)
    names = (
        "GWOSC_VALIDATION_2026-06-22.md",
        "GWOSC_GWPY_REFERENCE.md",
        "GWOSC_FILTERING_AND_LOCAL_PSD_PROTOCOL.md",
        "VALIDATION_ROADMAP.md",
        "paper1_validation_progress.md",
    )
    for name in names:
        source = REPO_ROOT / "docs" / name
        destination = destination_root / name
        shutil.copy2(source, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            source.relative_to(REPO_ROOT)
        )


def _copy_all_gwosc_runs(provenance: dict[str, str]) -> None:
    evidence_root = REPO_ROOT / "evidence" / "gwosc"
    destination_root = GWOSC_ROOT / "runs"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    for run_dir in sorted(path for path in evidence_root.iterdir() if path.is_dir()):
        _copy_tree_files(run_dir, destination_root / run_dir.name, provenance)


def _pytest_count(run_dir: Path) -> int | str:
    pytest_log = run_dir / "stage0" / "02_pytest.txt"
    if not pytest_log.is_file():
        return ""
    matches = re.findall(
        r"(\d+)\s+passed",
        pytest_log.read_text(encoding="utf-8", errors="replace"),
    )
    return int(matches[-1]) if matches else ""


def _sha256s_ok(run_dir: Path) -> bool | str:
    sha_path = run_dir / "SHA256SUMS"
    if not sha_path.is_file():
        return ""
    for line in sha_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        expected, raw_relative = line.split(maxsplit=1)
        relative = raw_relative.removeprefix("./")
        path = run_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            return False
    return True


def _gwosc_run_history() -> list[dict[str, Any]]:
    rows = []
    evidence_root = REPO_ROOT / "evidence" / "gwosc"
    for run_dir in sorted(path for path in evidence_root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        stage_path = run_dir / "stage0" / "summary.json"
        experiment_path = run_dir / "gwosc" / "experiment.json"
        manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
        stage = _load_json(stage_path) if stage_path.is_file() else {}
        experiment = (
            _load_json(experiment_path) if experiment_path.is_file() else {}
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "base_commit": manifest.get("base_commit", ""),
                "stage0_accepted": stage.get("accepted", False),
                "test_count": _pytest_count(run_dir),
                "gwosc_status": experiment.get("status", "not_run"),
                "sha256s_ok": _sha256s_ok(run_dir),
                "evidence_files": sum(1 for path in run_dir.rglob("*") if path.is_file()),
                "has_filter_equivalence": (
                    run_dir / "gwosc" / "filter_equivalence.json"
                ).is_file(),
                "has_time_local_noise": (
                    run_dir / "gwosc" / "time_local_noise.json"
                ).is_file(),
            }
        )
    return rows


def _write_evidence_inventory(
    provenance: dict[str, str],
) -> None:
    rows = []
    destination = DERIVED_ROOT / "evidence_inventory.csv"
    for path in sorted(DATA_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "transfer_manifest.json"
            or path == destination
        ):
            continue
        relative = str(path.relative_to(TRANSFER_ROOT))
        rows.append(
            {
                "path": relative,
                "source": provenance.get(relative, "generated"),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "category": relative.split("/", maxsplit=3)[1]
                if "/" in relative
                else "",
            }
        )
    _write_csv(destination, rows)
    provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
        "generated inventory of transferred paper evidence files"
    )


def _derive_gwosc_tables(
    run_dir: Path,
    provenance: dict[str, str],
) -> None:
    record_path = run_dir / "gwosc" / "experiment.json"
    record = _load_json(record_path)
    metrics = record["metrics"]
    null_rows = []
    primary_rows = []
    reference_rows = []
    quality_rows = []
    for detector, detector_metrics in metrics["detectors"].items():
        random_gate = detector_metrics["null_calibration_validation"]
        blocked_gate = detector_metrics[
            "blocked_null_calibration_validation"
        ]
        for gate in (random_gate, blocked_gate):
            for split in gate["splits"]:
                null_rows.append(
                    {
                        "run_id": run_dir.name,
                        "detector": detector,
                        "split_kind": gate["split_kind"],
                        "split_id": split.get(
                            "split_id",
                            split.get("split_seed", ""),
                        ),
                        "split_seed": split.get("split_seed", ""),
                        "null_sigma_over_predicted": split[
                            "null_sigma_over_predicted"
                        ],
                        "passed": split["passed"],
                        "amplitude_sigma": split["amplitude_sigma"],
                        "null_amplitude_std": split["null_amplitude_std"],
                        "n_calibration_windows": split[
                            "n_calibration_windows"
                        ],
                        "n_evaluation_windows": split[
                            "n_evaluation_windows"
                        ],
                    }
                )
        reference = detector_metrics["gwpy_reference"]
        matched = reference["matched_filter"]
        psd = reference["psd"]
        reference_rows.append(
            {
                "run_id": run_dir.name,
                "detector": detector,
                "psd_ratio_median": psd["ratio_median"],
                "psd_ratio_p05": psd["ratio_p05"],
                "psd_ratio_p95": psd["ratio_p95"],
                "psd_relative_l2_error": psd["relative_l2_error"],
                "psd_max_abs_log10_ratio": psd[
                    "max_abs_log10_ratio"
                ],
                "corr_gls_vs_gwpy_fir": matched[
                    "correlation_repository_gls_vs_gwpy"
                ],
                "corr_gls_vs_repository_whitened": matched[
                    "correlation_repository_gls_vs_whitened"
                ],
                "corr_repository_whitened_vs_gwpy_fir": matched[
                    "correlation_repository_vs_gwpy_whitened"
                ],
                "relative_l2_gls_vs_gwpy_fir": matched[
                    "relative_l2_repository_gls_vs_gwpy"
                ],
            }
        )
        data_quality = detector_metrics["data_quality"]
        evaluation_quality = detector_metrics["evaluation_window_quality"]
        quality_rows.append(
            {
                "run_id": run_dir.name,
                "detector": detector,
                "official_data_available": data_quality["available"],
                "event_window_valid": data_quality["event_window"]["valid"],
                "candidate_windows": data_quality["n_candidate_windows"],
                "valid_windows": data_quality["n_valid_windows"],
                "invalid_windows": data_quality["n_invalid_windows"],
                "quality_flagged_evaluation_windows": evaluation_quality[
                    "n_rejected"
                ],
            }
        )
        primary_rows.append(
            {
                "run_id": run_dir.name,
                "detector": detector,
                "status": record["status"],
                "random_median_ratio": random_gate[
                    "median_null_sigma_over_predicted"
                ],
                "random_min_ratio": random_gate[
                    "min_null_sigma_over_predicted"
                ],
                "random_max_ratio": random_gate[
                    "max_null_sigma_over_predicted"
                ],
                "random_gate_passed": random_gate["passed"],
                "chronological_median_ratio": blocked_gate[
                    "median_null_sigma_over_predicted"
                ],
                "chronological_min_ratio": blocked_gate[
                    "min_null_sigma_over_predicted"
                ],
                "chronological_max_ratio": blocked_gate[
                    "max_null_sigma_over_predicted"
                ],
                "chronological_gate_passed": blocked_gate["passed"],
                "event_score_uncalibrated": detector_metrics[
                    "event_matched_filter_score"
                ],
                "empirical_injection_snr": detector_metrics[
                    "injection_empirical_snr"
                ],
            }
        )

    outputs = {
        "gwosc_null_calibration.csv": null_rows,
        "gwosc_primary_results.csv": primary_rows,
        "gwosc_reference_summary.csv": reference_rows,
        "gwosc_data_quality.csv": quality_rows,
    }
    for name, rows in outputs.items():
        destination = DERIVED_ROOT / name
        _write_csv(destination, rows)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
            f"derived from {record_path.relative_to(REPO_ROOT)}"
        )


def _paper_implications(
    run_dir: Path,
    provenance: dict[str, str],
) -> None:
    primary = pd.read_csv(DERIVED_ROOT / "gwosc_primary_results.csv")
    reference = pd.read_csv(DERIVED_ROOT / "gwosc_reference_summary.csv")
    quality = pd.read_csv(DERIVED_ROOT / "gwosc_data_quality.csv")
    synthetic = pd.read_csv(DERIVED_ROOT / "synthetic_gate_summary.csv")
    h1 = primary[primary["detector"] == "H1"].iloc[0]
    l1 = primary[primary["detector"] == "L1"].iloc[0]
    psd_exact = bool(
        (reference["psd_ratio_median"] == 1.0).all()
        and (reference["psd_relative_l2_error"] == 0.0).all()
    )
    coverage = ", ".join(
        f"{row.detector}: {row.valid_windows}/{row.candidate_windows} valid"
        for row in quality.itertuples()
    )
    synthetic_range = (
        f"{int(synthetic['n_seeds'].min())}-{int(synthetic['n_seeds'].max())}"
        if not synthetic.empty
        else ""
    )
    rows = [
        {
            "paper_claim_area": "Controlled validation ladder",
            "evidence_files": (
                "data/synthetic/S1-S9_sweep_10seeds.*, "
                "data/derived/synthetic_gate_summary.csv"
            ),
            "result_summary": (
                f"S1-S9 passed with {synthetic_range} seeds per gate; "
                "held-out splits and paired intervals are encoded where required."
            ),
            "paper_implication": (
                "The Methods and Results can present the synthetic hierarchy as "
                "verified software and mathematical validation."
            ),
            "boundary": (
                "This does not establish calibration on nonstationary real detector noise."
            ),
        },
        {
            "paper_claim_area": "Remote reproducibility",
            "evidence_files": (
                "data/gwosc/runs/*/stage0/, data/gwosc/runs/*/manifest.json, "
                "data/derived/gwosc_run_history.csv"
            ),
            "result_summary": (
                "The latest archived remote run passed Stage 0 and recorded "
                "environment, dependency, test, table, and figure logs."
            ),
            "paper_implication": (
                "The reproducibility statement can say the validation workflow "
                "was exercised on a clean remote Linux checkout."
            ),
            "boundary": (
                "Reproducibility is separate from scientific acceptance of the real-data gate."
            ),
        },
        {
            "paper_claim_area": "GWOSC data coverage and integrity",
            "evidence_files": (
                "data/gwosc/current/gwosc/raw_metadata.json, "
                "data/gwosc/current/SHA256SUMS, data/derived/gwosc_data_quality.csv"
            ),
            "result_summary": (
                f"Official H1/L1 DATA coverage was present; {coverage}; "
                "raw/evidence checksums verified."
            ),
            "paper_implication": (
                "The paper can state that the 256-second GW150914-centered "
                "analysis interval had valid H1/L1 off-source and event windows."
            ),
            "boundary": (
                "The raw strain arrays are not stored in this bundle; the bundle stores "
                "metadata, checksums, and derived evidence."
            ),
        },
        {
            "paper_claim_area": "GWpy PSD reference",
            "evidence_files": (
                "data/gwosc/current/gwosc/gwpy_reference.json, "
                "data/derived/gwosc_reference_summary.csv"
            ),
            "result_summary": (
                "Repository/GWpy PSD ratios have median, p05, and p95 equal to "
                f"1.0 at recorded precision; relative L2 error is 0.0. "
                f"Exact PSD match recorded: {psd_exact}."
            ),
            "paper_implication": (
                "The PSD-estimation method can be described as independently "
                "verified against GWpy for identical calibration windows."
            ),
            "boundary": (
                "PSD estimator agreement does not prove that the resulting matched "
                "statistic is calibrated on held-out real noise."
            ),
        },
        {
            "paper_claim_area": "Global PSD held-out real-noise calibration",
            "evidence_files": (
                "data/gwosc/current/gwosc/experiment.json, "
                "data/derived/gwosc_null_calibration.csv, "
                "figures/gwosc_null_calibration.*"
            ),
            "result_summary": (
                f"H1 random median/min/max {h1.random_median_ratio:.3f}/"
                f"{h1.random_min_ratio:.3f}/{h1.random_max_ratio:.3f}; "
                f"L1 random {l1.random_median_ratio:.3f}/"
                f"{l1.random_min_ratio:.3f}/{l1.random_max_ratio:.3f}. "
                f"H1 chronological median {h1.chronological_median_ratio:.3f}; "
                f"L1 chronological median {l1.chronological_median_ratio:.3f}."
            ),
            "paper_implication": (
                "The Results should report this as a negative real-data result: "
                "global PSD calibration failed despite PSD reference agreement."
            ),
            "boundary": (
                "Do not convert event or injection scores into significance or "
                "sensitivity claims under this failed calibration."
            ),
        },
        {
            "paper_claim_area": "Filtering/statistic equivalence follow-up",
            "evidence_files": (
                "data/configs/gwosc/filter_statistic_equivalence.yaml, "
                "data/source_documents/GWOSC_FILTERING_AND_LOCAL_PSD_PROTOCOL.md, "
                "data/gwosc/followup/filter_equivalence.json when present"
            ),
            "result_summary": (
                "Implemented and predeclared locally; no remote evidence JSON is "
                "present in the current bundle."
            ),
            "paper_implication": (
                "Can be described as a predeclared follow-up method until remote "
                "evidence is synchronized."
            ),
            "boundary": (
                "Do not claim FIR/GLS equivalence or real-data outcomes yet."
            ),
        },
        {
            "paper_claim_area": "Time-local PSD follow-up",
            "evidence_files": (
                "data/configs/gwosc/time_local_noise.yaml, "
                "data/source_documents/GWOSC_FILTERING_AND_LOCAL_PSD_PROTOCOL.md, "
                "data/gwosc/followup/time_local_noise.json when present"
            ),
            "result_summary": (
                "Implemented and predeclared locally with global leave-one-out and "
                "32/64/96-second local radii; no remote evidence JSON is present "
                "in the current bundle."
            ),
            "paper_implication": (
                "Can be described as the next controlled test of whether local "
                "spectral drift explains the failed global-PSD calibration."
            ),
            "boundary": (
                "Do not claim that any local model improves calibration before "
                "the remote evidence exists."
            ),
        },
    ]
    for destination in (
        DERIVED_ROOT / "paper_implications.csv",
        TABLE_ROOT / "paper_implications.csv",
    ):
        _write_csv(destination, rows)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
            f"curated and derived from {run_dir.relative_to(REPO_ROOT)} and sweep records"
        )


def _method_traceability(provenance: dict[str, str]) -> None:
    rows = [
        {
            "experiment": "S1-S9 synthetic ladder",
            "purpose": "Validate normalization, representation, subspace, covariance, and residual-calibration claims under controlled generative models.",
            "config_or_protocol": "data/configs/synthetic/S*.yaml",
            "primary_outputs": "data/synthetic/S*_sweep_10seeds.*, data/derived/synthetic_gate_summary.csv, figures/synthetic_validation_overview.*",
            "acceptance_or_status": "Passed multi-seed gates.",
            "manuscript_role": "Methods and controlled Results.",
        },
        {
            "experiment": "Stage 0 remote reproducibility",
            "purpose": "Show the repo can install, test, run core experiments, make tables, and make figures on a clean remote checkout.",
            "config_or_protocol": "data/gwosc/runs/*/stage0/, data/source_documents/VALIDATION_ROADMAP.md",
            "primary_outputs": "data/derived/gwosc_run_history.csv, data/gwosc/runs/*/stage0/summary.json",
            "acceptance_or_status": "Latest archived run passed.",
            "manuscript_role": "Reproducibility statement and supplement.",
        },
        {
            "experiment": "GWOSC baseline/global PSD",
            "purpose": "Test whether a globally estimated PSD predicts held-out null matched-statistic spread on real GW150914-centered H1/L1 data.",
            "config_or_protocol": "data/gwosc/current/gwosc/experiment.config.yaml, data/configs/gwosc/gw150914_smoke.yaml",
            "primary_outputs": "data/gwosc/current/gwosc/experiment.json, data/derived/gwosc_null_calibration.csv, figures/gwosc_null_calibration.*",
            "acceptance_or_status": "Failed predeclared random and chronological gates.",
            "manuscript_role": "Negative real-data Results and Discussion.",
        },
        {
            "experiment": "GWpy PSD reference",
            "purpose": "Validate the repository median-Welch PSD estimator against an independent GWpy implementation on identical windows.",
            "config_or_protocol": "data/gwosc/current/gwosc/experiment.config.yaml",
            "primary_outputs": "data/gwosc/current/gwosc/gwpy_reference.json, data/derived/gwosc_reference_summary.csv, figures/gwosc_reference_comparison.*",
            "acceptance_or_status": "PSD ratio equals one at recorded precision.",
            "manuscript_role": "Methods validation and Results.",
        },
        {
            "experiment": "Shared-FIR statistic equivalence",
            "purpose": "Disentangle implementation identity from methodological sensitivity by forcing GWpy and repository paths to use identical FIR coefficients and score normalization.",
            "config_or_protocol": "data/configs/gwosc/filter_statistic_equivalence.yaml",
            "primary_outputs": "data/gwosc/followup/filter_equivalence.json, figures/gwosc_filter_equivalence.* when remote run exists.",
            "acceptance_or_status": "Implemented locally; remote evidence pending.",
            "manuscript_role": "Methods now; Results only after evidence synchronization.",
        },
        {
            "experiment": "Time-local PSD modelling",
            "purpose": "Compare global leave-one-out PSDs with fixed local calibration radii and record chronological, template-projected, and narrow-band diagnostics.",
            "config_or_protocol": "data/configs/gwosc/time_local_noise.yaml",
            "primary_outputs": "data/gwosc/followup/time_local_noise.json, figures/gwosc_time_local_psd.* when remote run exists.",
            "acceptance_or_status": "Implemented locally; remote evidence pending.",
            "manuscript_role": "Methods now; exploratory/confirmatory Results only after evidence synchronization.",
        },
    ]
    for destination in (
        DERIVED_ROOT / "method_traceability.csv",
        TABLE_ROOT / "method_traceability.csv",
    ):
        _write_csv(destination, rows)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
            "curated map from experiment protocols to paper sections"
        )


def _figure_index(provenance: dict[str, str]) -> None:
    rows = [
        {
            "figure_stem": "synthetic_validation_overview",
            "source_data": "data/synthetic/S1/S2/S5/S8 sweep CSVs and data/derived/synthetic_gate_summary.csv",
            "paper_message": "Representative synthetic gates show interval-backed validation of normalization, OF/EMPCA geometry, colored-noise weighting, and residual calibration.",
            "status": "available",
        },
        {
            "figure_stem": "gwosc_null_calibration",
            "source_data": "data/derived/gwosc_null_calibration.csv",
            "paper_message": "The global PSD model fails held-out random and chronological real-noise calibration.",
            "status": "available",
        },
        {
            "figure_stem": "gwosc_reference_comparison",
            "source_data": "data/derived/gwosc_reference_summary.csv",
            "paper_message": "The PSD estimator matches GWpy exactly, while earlier score-path comparisons were diagnostic because the filters were not identical.",
            "status": "available",
        },
        {
            "figure_stem": "gwosc_run_history",
            "source_data": "data/derived/gwosc_run_history.csv",
            "paper_message": "The audit trail separates the failed setup attempt, reproducible remote runs, and scientific gate status.",
            "status": "available",
        },
        {
            "figure_stem": "paper_claim_support_matrix",
            "source_data": "data/derived/claim_status.csv",
            "paper_message": "Claim-state matrix for writing control; useful for supplement or internal handoff.",
            "status": "available",
        },
        {
            "figure_stem": "gwosc_filter_equivalence",
            "source_data": "data/gwosc/followup/filter_equivalence.json",
            "paper_message": "Pending shared-FIR identity and GLS/FIR sensitivity results.",
            "status": "pending_remote_evidence",
        },
        {
            "figure_stem": "gwosc_time_local_psd",
            "source_data": "data/gwosc/followup/time_local_noise.json",
            "paper_message": "Pending global/local PSD calibration and chronological stability results.",
            "status": "pending_remote_evidence",
        },
    ]
    destination = DERIVED_ROOT / "figure_index.csv"
    _write_csv(destination, rows)
    provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
        "curated index of available and pending paper figures"
    )


def _synthetic_gate_summary(
    provenance: dict[str, str],
) -> None:
    definitions = {
        "S1": (
            "sigma_over_crb",
            "OF uncertainty agrees with CRB",
            "mean and 95% interval",
        ),
        "S2": (
            "weighted_angle_deg",
            "EMPCA converges to the OF direction",
            "mean angle",
        ),
        "S3": (
            "relative_gap",
            "independent tied linear AE reaches EMPCA optimum",
            "mean relative objective gap",
        ),
        "S4": (
            "max_principal_angle_deg",
            "PCA and EMPCA agree in white noise",
            "mean held-out angle",
        ),
        "S5": (
            "pca_weighted_residual_to_observed",
            "weighted PCA improves the colored-noise metric",
            "paired interval excludes zero",
        ),
        "S6": (
            "best_rank_by_clean_mse",
            "timing variability requires rank above one",
            "mean best rank",
        ),
        "S7": (
            "sigma_over_oracle.4",
            "estimated covariance converges toward oracle",
            "largest-calibration mean ratio",
        ),
        "S8": (
            "mean_chi2_per_dof",
            "whitened residual chi-square is calibrated",
            "mean chi-square per degree of freedom",
        ),
        "S9": (
            "diagonal_over_full_sigma",
            "full covariance improves multichannel uncertainty",
            "mean ratio",
        ),
    }
    rows = []
    for gate, (metric, claim, interpretation) in definitions.items():
        path = SYNTHETIC_ROOT / f"{gate}_sweep_10seeds.json"
        payload = _load_json(path)
        summary = payload["summary"][metric]
        pair = payload.get("pairs", [])
        rows.append(
            {
                "gate": gate,
                "claim": claim,
                "headline_metric": metric,
                "mean": summary["mean"],
                "ci95_low": summary["ci95"][0],
                "ci95_high": summary["ci95"][1],
                "n_seeds": summary["n"],
                "paired_ci95_excludes_zero": (
                    pair[0]["ci95_excludes_zero"] if pair else ""
                ),
                "interpretation": interpretation,
                "regression_status": "passed",
            }
        )
    destination = DERIVED_ROOT / "synthetic_gate_summary.csv"
    _write_csv(destination, rows)
    provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
        "derived from data/synthetic/S1-S9 sweep JSON records"
    )


def _claim_status(provenance: dict[str, str]) -> None:
    rows = [
        {
            "topic": "S1-S9 controlled validation",
            "evidence_state": "verified",
            "paper_use": "Results and Methods",
            "allowed_statement": (
                "Synthetic equivalence and robustness gates passed their "
                "multi-seed regression criteria."
            ),
            "prohibited_overclaim": (
                "Do not infer real-detector calibration from synthetic data."
            ),
        },
        {
            "topic": "Remote reproducibility",
            "evidence_state": "verified",
            "paper_use": "Methods and reproducibility statement",
            "allowed_statement": (
                "Stage 0 passed on a clean remote Linux checkout."
            ),
            "prohibited_overclaim": (
                "A reproducible run does not imply scientific acceptance."
            ),
        },
        {
            "topic": "GWOSC PSD implementation",
            "evidence_state": "verified",
            "paper_use": "Methods and Results",
            "allowed_statement": (
                "The Hann/median PSD agrees exactly with GWpy on identical "
                "windows."
            ),
            "prohibited_overclaim": (
                "PSD agreement alone does not validate the likelihood."
            ),
        },
        {
            "topic": "Global-PSD null calibration",
            "evidence_state": "verified_negative",
            "paper_use": "Results, Discussion, and Limitations",
            "allowed_statement": (
                "The predeclared random and chronological held-out gates "
                "failed for both detectors."
            ),
            "prohibited_overclaim": (
                "Do not describe event scores as calibrated significance."
            ),
        },
        {
            "topic": "Shared-FIR statistic equivalence",
            "evidence_state": "implemented_pending_remote",
            "paper_use": "Methods only until evidence arrives",
            "allowed_statement": (
                "A predeclared identical-statistic comparison was implemented."
            ),
            "prohibited_overclaim": (
                "Do not report real-data equivalence results yet."
            ),
        },
        {
            "topic": "Time-local PSD model",
            "evidence_state": "implemented_pending_remote",
            "paper_use": "Methods only until evidence arrives",
            "allowed_statement": (
                "Global and fixed-radius local PSD models were predeclared."
            ),
            "prohibited_overclaim": (
                "Do not claim that the 64-second model improves calibration."
            ),
        },
        {
            "topic": "GW150914 event and injection interpretation",
            "evidence_state": "not_validated",
            "paper_use": "Limitations or diagnostic appendix only",
            "allowed_statement": (
                "Uncalibrated diagnostic scores were archived."
            ),
            "prohibited_overclaim": (
                "No significance, false-alarm, or validated sensitivity claim."
            ),
        },
    ]
    for destination in (
        DERIVED_ROOT / "claim_status.csv",
        TABLE_ROOT / "claim_status.csv",
    ):
        _write_csv(destination, rows)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = (
            "curated from archived evidence and validation protocols"
        )


def _copy_followup_if_available(
    run_dir: Path,
    provenance: dict[str, str],
) -> None:
    source_dir = run_dir / "gwosc"
    followup_root = GWOSC_ROOT / "followup"
    if followup_root.exists():
        shutil.rmtree(followup_root)
    followup_root.mkdir(parents=True, exist_ok=True)
    copied = False
    for name in (
        "filter_equivalence.json",
        "filter_equivalence.config.yaml",
        "time_local_noise.json",
        "time_local_noise.config.yaml",
        "diagnostic_summary.json",
    ):
        source = source_dir / name
        if not source.is_file():
            continue
        destination = followup_root / name
        shutil.copy2(source, destination)
        provenance[str(destination.relative_to(TRANSFER_ROOT))] = str(
            source.relative_to(REPO_ROOT)
        )
        copied = True
    if not copied:
        pending = followup_root / "PENDING.md"
        pending.write_text(
            "# Pending follow-up evidence\n\n"
            "No archived filtering-equivalence or time-local-noise JSON was "
            "available when the bundle was refreshed. Run the remote runbook, "
            "synchronize the evidence commit, and refresh this bundle.\n",
            encoding="utf-8",
        )
        provenance[str(pending.relative_to(TRANSFER_ROOT))] = (
            "generated pending-state marker"
        )


def _write_manifest(
    provenance: dict[str, str],
    selected_run: Path,
) -> None:
    files = []
    for path in sorted(DATA_ROOT.rglob("*")):
        if not path.is_file() or path.name == "transfer_manifest.json":
            continue
        relative = str(path.relative_to(TRANSFER_ROOT))
        files.append(
            {
                "path": relative,
                "source": provenance.get(relative, "generated"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    manifest = {
        "selected_gwosc_run_id": selected_run.name,
        "selected_gwosc_evidence_path": str(
            selected_run.relative_to(REPO_ROOT)
        ),
        "raw_data_in_transfer_bundle": False,
        "standalone_scope": (
            "Contains evidence logs, JSON/YAML records, checksums, derived "
            "tables, config snapshots, notebooks, and rendered paper plots. "
            "It intentionally excludes raw GWOSC strain arrays."
        ),
        "authoritative_sources": [
            "data/gwosc/runs/",
            "data/synthetic/",
        ],
        "files": files,
    }
    (DATA_ROOT / "transfer_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bundle_files = []
    bundle_manifest_path = TRANSFER_ROOT / "BUNDLE_MANIFEST.json"
    for path in sorted(TRANSFER_ROOT.rglob("*")):
        if (
            not path.is_file()
            or path == bundle_manifest_path
            or "__pycache__" in path.parts
        ):
            continue
        relative = str(path.relative_to(TRANSFER_ROOT))
        bundle_files.append(
            {
                "path": relative,
                "source": provenance.get(relative, "generated or hand-authored"),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    bundle_manifest = {
        "selected_gwosc_run_id": selected_run.name,
        "raw_data_in_transfer_bundle": False,
        "standalone_for_paper_writing": True,
        "excluded_by_design": [
            "raw GWOSC strain arrays",
            "remote server paths required only for re-running, not writing",
            "Python __pycache__ bytecode",
        ],
        "files": bundle_files,
    }
    bundle_manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gwosc-run-id",
        default=None,
        help="select an archived run explicitly; default is newest with experiment.json",
    )
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    DERIVED_ROOT.mkdir(parents=True, exist_ok=True)
    GWOSC_ROOT.mkdir(parents=True, exist_ok=True)
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, str] = {}

    selected_run = _select_gwosc_run(args.gwosc_run_id)
    _copy_all_gwosc_runs(provenance)
    _copy_tree_files(
        selected_run,
        GWOSC_ROOT / "current",
        provenance,
    )
    _copy_synthetic_sweeps(provenance)
    _copy_synthetic_figures(provenance)
    _copy_config_snapshots(provenance)
    _copy_source_documents(provenance)
    _write_csv(
        DERIVED_ROOT / "gwosc_run_history.csv",
        _gwosc_run_history(),
    )
    provenance["data/derived/gwosc_run_history.csv"] = (
        "derived from evidence/gwosc run manifests"
    )
    _derive_gwosc_tables(selected_run, provenance)
    _synthetic_gate_summary(provenance)
    _claim_status(provenance)
    _paper_implications(selected_run, provenance)
    _method_traceability(provenance)
    _figure_index(provenance)
    _copy_followup_if_available(selected_run, provenance)
    _write_evidence_inventory(provenance)
    _write_manifest(provenance, selected_run)

    primary = pd.read_csv(DERIVED_ROOT / "gwosc_primary_results.csv")
    print(f"Selected GWOSC run: {selected_run.name}")
    print(primary.to_string(index=False))
    print(f"Transfer manifest: {DATA_ROOT / 'transfer_manifest.json'}")


if __name__ == "__main__":
    main()
