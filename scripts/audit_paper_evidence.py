"""EXP-00: read-only provenance audit of paper-facing evidence.

Inventories ``results/metrics/``, ``results/sweeps/`` (if present),
``evidence/``, ``data/noise_sample/Study_A/``, and ``transfer_paper/``; records
generator lineage for each synthetic experiment (S0-S9); and cross-checks that
every evidence path cited in ``transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md``
resolves to a real file.

This script never writes outside ``results/audits/`` and never modifies any
existing evidence, results, or transfer_paper file.

    uv run python scripts/audit_paper_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "audits"

# --------------------------------------------------------------------------
# Generator lineage: hand-traced from source, not inferred from manuscript
# prose. See docs/REVISION_PLAN_2026-07-26.md EXP-00: "Do not infer that S8
# is contaminated merely because Appendix G documents a fixed-modulus
# sampler... its lineage must be demonstrated rather than assumed."
#
# Traced 2026-07-25 by reading src/noise_geometry/experiments/synthetic.py,
# src/noise_geometry/synthetic/benchmarks.py, and src/noise_geometry/noise/
# synthetic.py end to end (see results/audits/noise_generator_provenance.md
# for the full narrative and the confirmed DC-bin scaling defect this trace
# uncovered in ``generate_colored_noise``).
# --------------------------------------------------------------------------

FOURIER_SAMPLER = {
    "generator_module": "src/noise_geometry/noise/synthetic.py",
    "generator_function": "generate_colored_noise",
    "sampler_class": "fourier_gaussian_real_imag_bins",
    "fft_based": True,
    "obsolete_fixed_modulus": False,
    "notes": (
        "Interior rFFT bins are independent circular complex Gaussians "
        "((re+1j*im)/sqrt(2)*scale); Nyquist bin (even N) is correctly "
        "rescaled by sqrt(2) to the full endpoint variance J*fs*N. The DC "
        "bin is NOT given the matching sqrt(2) correction: it is assigned "
        "variance J*fs*N/2 (the interior-bin formula) instead of the "
        "declared endpoint formula J*fs*N used by Nyquist and by the "
        "target paper convention (EXP-01). Confirmed empirically (200k-draw "
        "flat-PSD Monte Carlo): DC empirical E|X0|^2 ~= 128.2 against an "
        "interior-formula prediction of 128.0 and an endpoint-formula "
        "prediction of 256.0; Nyquist empirical ~= 255.3 against the "
        "endpoint prediction of 256.0. This is a real, previously "
        "untested inconsistency in this smaller core generator -- NOT "
        "present in the separate 'upgraded modular generator' "
        "noise_module/noise_module/NoiseGenerator.py, whose "
        "psd_density_to_rfft_power() halves ONLY the interior bins "
        "(power[1:upper] *= 0.5), leaving DC and Nyquist both at the full "
        "endpoint scale -- i.e. NoiseGenerator.py implements the DC/Nyquist "
        "convention correctly; src/noise_geometry/noise/synthetic.py does "
        "not. Practical impact on S0/S1/S2/S5 results is negligible (one "
        "bin out of 65-129 rFFT bins, and the S1/S2 pulse templates and the "
        "S5 sin/cos signal basis carry ~zero DC power), but it is a genuine "
        "defect that a rigorous audit must not silently pass over, and it "
        "should be fixed under EXP-01 with a regression test added first."
    ),
}

COVARIANCE_SAMPLER = {
    "generator_module": "numpy.random.Generator",
    "generator_function": "rng.multivariate_normal",
    "sampler_class": "exact_multivariate_gaussian_from_covariance",
    "fft_based": False,
    "obsolete_fixed_modulus": False,
    "notes": (
        "Direct draw from an explicit covariance matrix (AR(1) or block "
        "covariance built in-experiment), not routed through any FFT/PSD "
        "sampler at all. This is a correction to the plan's assumption "
        "('current core S8 path already uses Gaussian real and imaginary "
        "Fourier coefficients'): S8 (and S6, S7, S9) do not use Fourier "
        "bins in any form -- they sample multivariate Gaussian noise "
        "directly from a covariance matrix via numpy's Cholesky-based "
        "multivariate_normal. This is fully Gaussian and unaffected by any "
        "question of Fourier-bin convention, fixed-modulus sampling, or "
        "circulant-vs-Toeplitz structure -- it is exact-covariance sampling "
        "on a finite grid, which sidesteps that debate entirely."
    ),
}

DIAGONAL_SAMPLER = {
    "generator_module": "numpy.random.Generator",
    "generator_function": "rng.normal",
    "sampler_class": "diagonal_gaussian_real_domain",
    "fft_based": False,
    "obsolete_fixed_modulus": False,
    "notes": (
        "Per-sample independent (diagonal-covariance) Gaussian noise added "
        "directly in the time domain; no FFT or covariance matrix involved."
    ),
}

# experiment_id -> (dataset builder, sampler dict, held_out_default)
S_LINEAGE: dict[str, dict[str, Any]] = {
    "S0": {"builder": "make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py)", "sampler": FOURIER_SAMPLER},
    "S1": {"builder": "make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py)", "sampler": FOURIER_SAMPLER},
    "S2": {"builder": "make_rank1_pulse_dataset (src/noise_geometry/synthetic/benchmarks.py)", "sampler": FOURIER_SAMPLER},
    "S3": {"builder": "make_s3_dataset (src/noise_geometry/experiments/synthetic.py)", "sampler": DIAGONAL_SAMPLER},
    "S4": {"builder": "_rankk_signal + rng.normal (src/noise_geometry/experiments/synthetic.py)", "sampler": DIAGONAL_SAMPLER},
    "S5": {"builder": "_metric_reversal_dataset (src/noise_geometry/experiments/synthetic.py)", "sampler": FOURIER_SAMPLER},
    "S6": {"builder": "run_s6_timing_rank_sweep, AR(1) covariance (src/noise_geometry/experiments/synthetic.py)", "sampler": COVARIANCE_SAMPLER},
    "S7": {"builder": "run_s7_covariance_robustness, AR(1) covariance (src/noise_geometry/experiments/synthetic.py)", "sampler": COVARIANCE_SAMPLER},
    "S8": {"builder": "run_s8_residual_calibration, AR(1) covariance (src/noise_geometry/experiments/synthetic.py)", "sampler": COVARIANCE_SAMPLER},
    "S9": {"builder": "run_s9_multichannel, block covariance (src/noise_geometry/experiments/synthetic.py)", "sampler": COVARIANCE_SAMPLER},
}

STATUS_FROZEN_PRIMARY = "frozen_primary"
STATUS_POST_FREEZE = "post_freeze_sensitivity"
STATUS_DIAGNOSTIC = "diagnostic"
STATUS_SUPERSEDED = "superseded"
STATUS_UNKNOWN = "unknown"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit_for(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(REPO_ROOT))],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        commit = out.stdout.strip()
        return commit or None
    except Exception:
        return None


def load_json_safe(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_yaml_safe(path: Path) -> dict[str, Any] | None:
    try:
        return yaml.safe_load(path.read_text())
    except Exception:
        return None


def classify_metric_kind(key: str) -> str:
    lower = key.lower()
    if "ratio" in lower or lower.endswith("_over_oracle") or "_over_" in lower:
        return "ratio"
    if "diff" in lower or "difference" in lower or "delta" in lower:
        return "difference"
    if "sigma" in lower or "std" in lower or "ci95" in lower or "ci68" in lower:
        return "uncertainty"
    if "angle" in lower or "chi2" in lower or "nll" in lower or "residual" in lower or "mse" in lower:
        return "test_statistic_or_raw"
    return "raw_value"


def audit_results_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_dir = REPO_ROOT / "results" / "metrics"
    for json_path in sorted(metrics_dir.glob("*.json")):
        stem = json_path.stem
        cfg_path = json_path.with_suffix("").with_suffix(".config.yaml")
        data = load_json_safe(json_path) or {}
        cfg = load_yaml_safe(cfg_path) if cfg_path.exists() else data.get("config")
        experiment_id = str(data.get("experiment_id") or (cfg or {}).get("experiment_id") or "")
        seed = (cfg or {}).get("seed", data.get("config", {}).get("seed"))
        lineage = S_LINEAGE.get(experiment_id)
        # Detect by filename stem, not by experiment_id: these JSONs already
        # carry their own non-empty experiment_id (e.g. "P1_GWOSC_GW150914"),
        # so gating on experiment_id == "" here previously misclassified
        # every GWOSC/CRESST single-run record as sampler_class "unknown".
        is_public_data = "gwosc" in stem or "cresst" in stem
        status = STATUS_FROZEN_PRIMARY
        rows.append(
            {
                "artifact": str(json_path.relative_to(REPO_ROOT)),
                "artifact_type": "single_run_metrics_json",
                "experiment_id": experiment_id or ("gwosc" if "gwosc" in stem else "cresst" if "cresst" in stem else "unknown"),
                "seed": seed,
                "config_path": str(cfg_path.relative_to(REPO_ROOT)) if cfg_path.exists() else None,
                "sha256": sha256_of(json_path),
                "recorded_git_commit": data.get("git_commit"),
                "git_log_commit": git_commit_for(json_path),
                "held_out": data.get("metrics", {}).get("held_out"),
                "generator_module": (lineage or {}).get("builder") if lineage else ("public_data" if is_public_data else "unknown"),
                "sampler_class": (lineage or {}).get("sampler", {}).get("sampler_class") if lineage else ("real_detector_data" if is_public_data else "unknown"),
                "fft_based_sampler": (lineage or {}).get("sampler", {}).get("fft_based") if lineage else None,
                "status": status,
            }
        )
    return rows


def audit_transfer_paper_synthetic() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    syn_dir = REPO_ROOT / "transfer_paper" / "data" / "synthetic"
    if not syn_dir.exists():
        return rows
    for json_path in sorted(syn_dir.glob("*_sweep_10seeds.json")):
        data = load_json_safe(json_path) or {}
        experiment_id = str(data.get("experiment_id", ""))
        lineage = S_LINEAGE.get(experiment_id)
        rows.append(
            {
                "artifact": str(json_path.relative_to(REPO_ROOT)),
                "artifact_type": "seed_sweep_json_paper_facing",
                "experiment_id": experiment_id,
                "seed": data.get("seeds"),
                "config_path": None,
                "sha256": sha256_of(json_path),
                "recorded_git_commit": None,
                "git_log_commit": git_commit_for(json_path),
                "held_out": (data.get("config") or {}).get("test_frac", 0) not in (0, 0.0, None),
                "generator_module": (lineage or {}).get("builder"),
                "sampler_class": (lineage or {}).get("sampler", {}).get("sampler_class"),
                "fft_based_sampler": (lineage or {}).get("sampler", {}).get("fft_based"),
                "status": STATUS_FROZEN_PRIMARY,
                "provenance_gap": (
                    "This 10-seed sweep is the artifact actually cited by "
                    "transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md, but no matching "
                    "archived copy exists under results/sweeps/ in the main repo "
                    "tree (results/sweeps/ does not exist as of this audit). The "
                    "command and exact repo commit that produced this file "
                    "cannot be independently re-derived from the main tree alone."
                ),
            }
        )
    return rows


def audit_study_a_exports() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    study_a_dir = REPO_ROOT / "data" / "noise_sample" / "Study_A"
    if not study_a_dir.exists():
        return rows
    for path in sorted(study_a_dir.glob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "artifact": str(path.relative_to(REPO_ROOT)),
                "artifact_type": "study_a_derived_export",
                "experiment_id": "study_a",
                "seed": None,
                "config_path": None,
                "sha256": sha256_of(path),
                "recorded_git_commit": None,
                "git_log_commit": git_commit_for(path),
                "held_out": None,
                "generator_module": "unknown (notebook: data/noise_sample/read_dat_1.ipynb)",
                "sampler_class": "deterministic_psd_arithmetic",
                "fft_based_sampler": False,
                "status": STATUS_SUPERSEDED,
                "supersede_reason": (
                    "Per docs/ACADEMIC_EXPERIMENT_AND_NOISE_MODULE_EVALUATION.md "
                    "finding #1: this export's physical labels ('1/f phonon', "
                    "'vibrational', 'TES Johnson') and reported crossovers do not "
                    "match the checked-in component files "
                    "(noise_module/Al2O3_Al_athermal/{SQUID,Johnson,TD,Er}_noise.dat), "
                    "which the file's own numbers otherwise agree with (SQUID "
                    "dominant everywhere, no SQUID crossings). Superseded by "
                    "EXP-02 (results/noise_budget/<run_id>/study_a/)."
                ),
            }
        )
    return rows


def audit_evidence_dir() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence_dir = REPO_ROOT / "evidence"
    if not evidence_dir.exists():
        return rows
    for manifest_path in sorted(evidence_dir.glob("*/*/manifest.json")):
        run_dir = manifest_path.parent
        domain = run_dir.parent.name  # "cresst" or "gwosc"
        status = STATUS_FROZEN_PRIMARY if domain == "cresst" else STATUS_FROZEN_PRIMARY
        rows.append(
            {
                "artifact": str(run_dir.relative_to(REPO_ROOT)),
                "artifact_type": f"{domain}_evidence_run",
                "experiment_id": domain,
                "seed": None,
                "config_path": None,
                "sha256": sha256_of(manifest_path),
                "recorded_git_commit": None,
                "git_log_commit": git_commit_for(manifest_path),
                "held_out": True,
                "generator_module": "public_data",
                "sampler_class": "real_detector_data",
                "fft_based_sampler": None,
                "status": status,
                "note": "Frozen public-data evidence; no sampler rerun applies per EXP-06 (packaging only).",
            }
        )
    return rows


# --------------------------------------------------------------------------
# Manuscript evidence-map cross-check
# --------------------------------------------------------------------------

_BACKTICK_PATH_RE = re.compile(r"`([^`]+)`")


def parse_evidence_map_table(md_path: Path) -> list[dict[str, Any]]:
    text = md_path.read_text()
    lines = text.splitlines()
    rows: list[dict[str, Any]] = []
    in_table = False
    header_seen = False
    for line in lines:
        if line.startswith("| paper claim area"):
            in_table = True
            header_seen = True
            continue
        if in_table and line.startswith("| ---"):
            continue
        if in_table and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5:
                continue
            claim, evidence_cell, support, implication, boundary = cells[:5]
            paths = _BACKTICK_PATH_RE.findall(evidence_cell)
            rows.append(
                {
                    "claim_area": claim,
                    "cited_paths": paths,
                    "support_level": support,
                }
            )
        elif in_table and header_seen and not line.startswith("|"):
            break
    return rows


def resolve_cited_path(raw: str) -> Path | None:
    # Paths in the map are relative to transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md
    # and may contain glob wildcards (e.g. "S1_sweep_10seeds.*", "runs/*/stage0/").
    for base in (REPO_ROOT / "transfer_paper", REPO_ROOT):
        if "*" in raw:
            matches = sorted(base.glob(raw))
            if matches:
                return matches[0]
        else:
            candidate = (base / raw).resolve()
            if candidate.exists():
                return candidate
    return None


def cross_check_manuscript_map() -> dict[str, Any]:
    md_path = REPO_ROOT / "transfer_paper" / "MANUSCRIPT_EVIDENCE_MAP.md"
    claims = parse_evidence_map_table(md_path)
    resolved_claims = []
    unresolved: list[dict[str, Any]] = []
    for claim in claims:
        resolutions = []
        for raw in claim["cited_paths"]:
            found = resolve_cited_path(raw)
            resolutions.append({"cited": raw, "resolved": str(found.relative_to(REPO_ROOT)) if found else None})
            if found is None:
                unresolved.append({"claim_area": claim["claim_area"], "cited_path": raw})
        resolved_claims.append({**claim, "resolutions": resolutions})
    return {
        "source_map": str(md_path.relative_to(REPO_ROOT)),
        "n_claims": len(claims),
        "n_unresolved_paths": len(unresolved),
        "unresolved": unresolved,
        "claims": resolved_claims,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    rows = (
        audit_results_metrics()
        + audit_transfer_paper_synthetic()
        + audit_study_a_exports()
        + audit_evidence_dir()
    )
    manifest_check = cross_check_manuscript_map()

    provenance = {
        "generated_at_utc": generated_at,
        "repo_commit": git_commit_for(REPO_ROOT / "docs" / "REVISION_PLAN_2026-07-26.md"),
        "generator_lineage_by_experiment": {
            exp_id: {
                "builder": info["builder"],
                **info["sampler"],
            }
            for exp_id, info in S_LINEAGE.items()
        },
        "artifacts": rows,
        "manuscript_evidence_map_cross_check": manifest_check,
    }

    json_path = OUT_DIR / "noise_generator_provenance.json"
    json_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")

    csv_path = OUT_DIR / "synthetic_config_provenance.csv"
    csv_fields = [
        "artifact", "artifact_type", "experiment_id", "seed", "config_path",
        "sha256", "recorded_git_commit", "git_log_commit", "held_out",
        "generator_module", "sampler_class", "fft_based_sampler", "status",
    ]
    import csv as csv_mod

    with csv_path.open("w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        "# Noise generator and evidence provenance audit (EXP-00)",
        "",
        f"Generated: {generated_at}",
        "",
        "## Generator lineage per synthetic experiment",
        "",
        "| Experiment | Builder | Sampler class | FFT-based | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for exp_id, info in S_LINEAGE.items():
        s = info["sampler"]
        md_lines.append(
            f"| {exp_id} | {info['builder']} | {s['sampler_class']} | {s['fft_based']} | "
            f"{s['notes'][:140].replace(chr(10), ' ')}... |"
        )
    md_lines += [
        "",
        "See the JSON companion for the full untruncated notes per sampler, "
        "including the confirmed DC-bin scaling defect in "
        "`src/noise_geometry/noise/synthetic.py:generate_colored_noise`.",
        "",
        "## Manuscript evidence-map cross-check",
        "",
        f"- Claims parsed from `transfer_paper/MANUSCRIPT_EVIDENCE_MAP.md`: {manifest_check['n_claims']}",
        f"- Cited paths that failed to resolve: {manifest_check['n_unresolved_paths']}",
    ]
    if manifest_check["unresolved"]:
        md_lines.append("")
        md_lines.append("| Claim area | Unresolved cited path |")
        md_lines.append("| --- | --- |")
        for u in manifest_check["unresolved"]:
            md_lines.append(f"| {u['claim_area']} | `{u['cited_path']}` |")
    md_lines += [
        "",
        "## Artifact inventory summary",
        "",
        f"- Total artifacts inventoried: {len(rows)}",
        f"- Superseded (must not be cited as evidence going forward): "
        f"{sum(1 for r in rows if r['status'] == STATUS_SUPERSEDED)}",
        f"- Frozen primary: {sum(1 for r in rows if r['status'] == STATUS_FROZEN_PRIMARY)}",
        "",
        "Full per-artifact rows are in `synthetic_config_provenance.csv` and "
        "`noise_generator_provenance.json`.",
    ]
    md_path = OUT_DIR / "noise_generator_provenance.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")
    print(f"wrote {csv_path.relative_to(REPO_ROOT)}")
    print(f"artifacts inventoried: {len(rows)}")
    print(f"unresolved manuscript-evidence-map paths: {manifest_check['n_unresolved_paths']}")


if __name__ == "__main__":
    main()
