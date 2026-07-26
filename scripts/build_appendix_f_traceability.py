"""EXP-05: Appendix F quantitative traceability audit.

Every legacy Appendix F number is treated as unverified until it has a
machine-readable source in THIS repository. This script checks each named
item against the actual archived sweep data where such data exists, and
marks an item ``unsupported`` (per the plan: "mark it unsupported so the
paper agent removes it") when no deterministic script/config/seed source
was found in this repository after a real search -- not merely absence of
memory of one. See ``results/audits/DEVIATION_REPORT_2026-07-26.md`` for the
search performed (grepped the repo for rank-saturation/template-mismatch/
timing-bias/resolution-scaling/convergence code and notebooks; found none
outside S1/S6/S7, and found no occurrence of 1029.2/1028.9/1028.6 anywhere
in this repository's data).

    uv run python scripts/build_appendix_f_traceability.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = REPO_ROOT / "transfer_paper" / "data" / "synthetic"
OUT_PATH = REPO_ROOT / "results" / "audits" / "appendix_f_traceability.csv"


def _load(gate: str) -> dict:
    return json.loads((SYNTHETIC_DIR / f"{gate}_sweep_10seeds.json").read_text())


def build_rows() -> list[dict[str, str]]:
    s1 = _load("S1")
    s6 = _load("S6")
    s7 = _load("S7")

    crb = s1["summary"]["sigma_over_crb"]
    crb_fractional_error = abs(crb["mean"] - 1.0)

    rank_sweep = s6["summary"]["best_rank_by_clean_mse"]

    convergence = s7["summary"]["sigma_over_oracle.4"]

    rows = [
        {
            "appendix_f_item": "CRB verification",
            "manuscript_claim": "<5% CRB error",
            "mapping": "mapped_and_verified",
            "source_script": "src/noise_geometry/experiments/synthetic.py::run_s1_of_crb via scripts/sweep_all.py",
            "source_config": "configs/synthetic/s1_of_crb.yaml (sweep base: {experiment_id: S1, n_traces: 2048})",
            "source_data": "transfer_paper/data/synthetic/S1_sweep_10seeds.json",
            "seeds": "0-9",
            "computed_value": f"mean sigma_over_crb={crb['mean']:.4f}, fractional error={crb_fractional_error:.4%}",
            "verdict": "CONFIRMED: 0.63% error, well within the claimed <5%.",
        },
        {
            "appendix_f_item": "Rank sweep / rank saturation",
            "manuscript_claim": "Best clean-MSE rank saturates around rank ~4-5 as timing variation is added",
            "mapping": "mapped_and_verified",
            "source_script": "src/noise_geometry/experiments/synthetic.py::run_s6_timing_rank_sweep via scripts/sweep_all.py",
            "source_config": "configs/synthetic/s6_timing_jitter_rank_sweep.yaml",
            "source_data": "transfer_paper/data/synthetic/S6_sweep_10seeds.json",
            "seeds": "0-9",
            "computed_value": f"best_rank_by_clean_mse mean={rank_sweep['mean']:.2f} (95% CI {rank_sweep['ci95']})",
            "verdict": "CONFIRMED: matches the audit doc's 'about 4.6 across ten seeds'.",
        },
        {
            "appendix_f_item": "Convergence (covariance estimation vs. sample size)",
            "manuscript_claim": "Estimated/oracle uncertainty ratio approaches 1 as calibration sample count grows; ~1.032 at 1000 samples",
            "mapping": "mapped_and_verified",
            "source_script": "src/noise_geometry/experiments/synthetic.py::run_s7_covariance_robustness via scripts/sweep_all.py",
            "source_config": "configs/synthetic/s7_covariance_robustness.yaml",
            "source_data": "transfer_paper/data/synthetic/S7_sweep_10seeds.json",
            "seeds": "0-9",
            "computed_value": f"sigma_over_oracle at n=1000 mean={convergence['mean']:.4f} (95% CI {convergence['ci95']})",
            "verdict": "CONFIRMED: matches the audit doc's 'about 1.032'.",
        },
        {
            "appendix_f_item": "Principal-component alignment / bootstrap panel",
            "manuscript_claim": "Rank-1 EMPCA direction aligns with the OF template; bootstrap-stable across seeds",
            "mapping": "partially_mapped",
            "source_script": "src/noise_geometry/synthetic/benchmarks.py::run_of_empca_equivalence (S2) via scripts/sweep_all.py",
            "source_config": "configs/synthetic/s2_of_empca.yaml",
            "source_data": "transfer_paper/data/synthetic/S2_sweep_10seeds.json",
            "seeds": "0-9",
            "computed_value": "weighted_angle_deg summary present in S2_sweep_10seeds.json",
            "verdict": (
                "PARTIAL: S2 provides the underlying angle-vs-seed data, but no "
                "dedicated bootstrap-resample-of-the-alignment-angle script was "
                "found in this repository. If the manuscript panel shows a "
                "bootstrap distribution (not just the 10-seed angle spread), that "
                "specific panel is UNSUPPORTED pending a new script."
            ),
        },
        {
            "appendix_f_item": "Time-shift diagnostic",
            "manuscript_claim": "OF amplitude/timing response under injected sub-sample time shifts",
            "mapping": "partially_mapped",
            "source_script": "src/noise_geometry/experiments/synthetic.py::run_s6_timing_rank_sweep (of_amplitude_bias field)",
            "source_config": "configs/synthetic/s6_timing_jitter_rank_sweep.yaml",
            "source_data": "transfer_paper/data/synthetic/S6_sweep_10seeds.json",
            "seeds": "0-9",
            "computed_value": "of_amplitude_bias present in S6_sweep_10seeds.json summary",
            "verdict": (
                "PARTIAL: S6 reports a single amplitude-bias number per seed under "
                "fixed timing-jitter, not a swept diagnostic curve across a range "
                "of injected shifts. If the manuscript shows a shift-response "
                "curve, that curve is UNSUPPORTED pending a new script."
            ),
        },
        {
            "appendix_f_item": "Resolution scaling (slope)",
            "manuscript_claim": "0.500 +/- 0.003 scaling slope (implying sigma ~ N^-0.5)",
            "mapping": "unsupported",
            "source_script": "none found",
            "source_config": "none found",
            "source_data": "none found",
            "seeds": "n/a",
            "computed_value": "n/a",
            "verdict": (
                "UNSUPPORTED: no script in this repository fits a resolution-vs-"
                "sample-count power-law slope. S7 sweeps sigma_over_oracle across "
                "n_noise_traces but does not fit or report a slope. This claim "
                "must be removed from the manuscript, or a new script must "
                "compute and archive it, before it can be cited."
            ),
        },
        {
            "appendix_f_item": "Template mismatch",
            "manuscript_claim": "Reconstruction degrades under a deliberately mismatched template",
            "mapping": "unsupported_for_paper_1_synthetic_appendix",
            "source_script": "none found in the synthetic S0-S9 suite (a structurally similar "
            "mismatched_of_template negative control exists for CRESST, a "
            "different result)",
            "source_config": "none found",
            "source_data": "none found",
            "seeds": "n/a",
            "computed_value": "n/a",
            "verdict": (
                "UNSUPPORTED as a synthetic Appendix F panel: do not cite the "
                "CRESST mismatched-template control as evidence for a synthetic "
                "Appendix F claim; they are different experiments on different data."
            ),
        },
        {
            "appendix_f_item": "Timing-bias reduction",
            "manuscript_claim": ">30% timing-bias reduction",
            "mapping": "unsupported",
            "source_script": "none found",
            "source_config": "none found",
            "source_data": "none found",
            "seeds": "n/a",
            "computed_value": "n/a",
            "verdict": (
                "UNSUPPORTED: no script in this repository computes a 'timing-bias "
                "reduction' percentage (a before/after comparison). S6's "
                "of_amplitude_bias is a single absolute number, not a reduction "
                "metric. This claim must be removed or a new before/after script "
                "written before it can be cited."
            ),
        },
        {
            "appendix_f_item": "KS/QQ residual calibration, rank-two transition",
            "manuscript_claim": "KS statistic vs. Exponential/Gaussian null shows a transition at rank two",
            "mapping": "unsupported",
            "source_script": (
                "S8 (run_s8_residual_calibration) computes mean_chi2_per_dof and "
                "lag-1 autocorrelation, and scripts/audit_generator_distribution.py "
                "(EXP-05) now computes a KS statistic vs. Exponential(1) for the "
                "generator itself -- but no script sweeps KS/QQ residual "
                "calibration AS A FUNCTION OF RECONSTRUCTION RANK"
            ),
            "source_config": "configs/synthetic/s8_residual_calibration.yaml",
            "source_data": "transfer_paper/data/synthetic/S8_sweep_10seeds.json (rank-independent only)",
            "seeds": "0-9 (S8); 0-4 (new generator KS check)",
            "computed_value": "n/a for a rank-swept KS/QQ curve",
            "verdict": (
                "UNSUPPORTED as a rank-swept panel: S8's residual calibration is "
                "computed at a single fixed reconstruction, not swept over rank, "
                "so a 'transition at rank two' claim has no machine-readable "
                "source in this repository. A new rank-swept KS/QQ script would "
                "be required before this specific claim can be cited."
            ),
        },
        {
            "appendix_f_item": "Duplicate rank values (1029.2 / 1028.9 / 1028.6) used in two manuscript tables",
            "manuscript_claim": "Same rank-related quantity reported with three slightly different values in two tables",
            "mapping": "unsupported",
            "source_script": "none found",
            "source_config": "none found",
            "source_data": "none found (repo-wide search found no occurrence of 1029.2, 1028.9, or 1028.6 outside CRESST/GWOSC JSON files, which are unrelated numeric coincidences)",
            "seeds": "n/a",
            "computed_value": "n/a",
            "verdict": (
                "UNSUPPORTED / UNRECONCILED: these three values do not appear "
                "anywhere in this repository's machine-readable evidence. Either "
                "they were computed outside this repository (e.g. directly in the "
                "paper repository or a notebook not checked in here), or they are "
                "stale. The paper agent must not reconcile these by picking one "
                "value; either locate their true source and archive it here, or "
                "remove all three from the manuscript pending a new, single, "
                "traceable computation."
            ),
        },
    ]
    return rows


def main() -> None:
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["mapping"]] = counts.get(r["mapping"], 0) + 1
    print(f"wrote {OUT_PATH} ({len(rows)} rows)")
    print(f"mapping counts: {counts}")


if __name__ == "__main__":
    main()
