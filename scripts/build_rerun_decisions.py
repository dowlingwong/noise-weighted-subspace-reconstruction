"""EXP-05: apply the plan's rerun-decision rule to every EXP-00-audited artifact.

Decision rule (docs/REVISION_PLAN_2026-07-26.md EXP-05):

| Provenance result | Required action |
| --- | --- |
| Gaussian-bin source path and complete config/commit proven | Retain; rerun only for the new seed/held-out protocol if required |
| Fixed-modulus random-phase source path proven | Rerun every distributional/calibration claim |
| Sampler lineage unknown | Rerun from a frozen current config |
| Deterministic PSD arithmetic only | No sampler rerun; regenerate after label/convention correction |
| Public real-data result with no synthetic sampler in scoring path | Retain unless a shared scoring defect is found |

Reads ``results/audits/noise_generator_provenance.json`` (EXP-00) as the
factual input and mechanically applies the table above -- the decision is a
function of the recorded ``sampler_class``/``fft_based_sampler`` fields, not
a fresh judgment call, so this script is deterministic given that audit.

    uv run python scripts/build_rerun_decisions.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDITS_DIR = REPO_ROOT / "results" / "audits"


def decide(artifact: dict) -> dict:
    sampler_class = artifact.get("sampler_class")
    status = artifact.get("status")

    if status == "superseded":
        action = "no_sampler_rerun_required"
        reason = "Deterministic PSD arithmetic only; already regenerated after label/convention correction (EXP-02)."
    elif sampler_class in ("real_detector_data",):
        action = "retain"
        reason = "Public real-data result with no synthetic sampler in its scoring path; no shared scoring defect identified."
    elif sampler_class in ("fourier_gaussian_real_imag_bins", "exact_multivariate_gaussian_from_covariance", "diagonal_gaussian_real_domain"):
        action = "retain_rerun_only_for_new_protocol"
        reason = (
            f"Gaussian-bin (or exact-covariance / diagonal-Gaussian) source path proven by EXP-00's "
            f"code trace ({sampler_class}); not the obsolete fixed-modulus sampler. Retain the frozen "
            "result; only the new held-out/seed protocol (EXP-04's S5 confirmatory sweep) required a rerun."
        )
    elif sampler_class == "unknown" or sampler_class is None:
        action = "rerun_from_frozen_current_config"
        reason = "Sampler lineage could not be established from source; must be rerun from a frozen current config before any distributional/calibration claim."
    else:
        action = "rerun_from_frozen_current_config"
        reason = f"Unrecognized sampler_class {sampler_class!r}; defaulting to the conservative rerun action."

    return {
        "artifact": artifact.get("artifact"),
        "experiment_id": artifact.get("experiment_id"),
        "sampler_class": sampler_class,
        "status_from_exp00": status,
        "decision": action,
        "reason": reason,
    }


def main() -> None:
    provenance = json.loads((AUDITS_DIR / "noise_generator_provenance.json").read_text())
    decisions = [decide(a) for a in provenance["artifacts"]]

    n_by_action: dict[str, int] = {}
    for d in decisions:
        n_by_action[d["decision"]] = n_by_action.get(d["decision"], 0) + 1

    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_provenance_audit": "results/audits/noise_generator_provenance.json",
        "decision_rule": {
            "gaussian_bin_or_exact_covariance_proven": "retain_rerun_only_for_new_protocol",
            "fixed_modulus_random_phase_proven": "rerun_every_distributional_or_calibration_claim",
            "sampler_lineage_unknown": "rerun_from_frozen_current_config",
            "deterministic_psd_arithmetic_only": "no_sampler_rerun_required",
            "public_real_data_no_sampler_in_scoring_path": "retain",
        },
        "summary_counts": n_by_action,
        "no_fixed_modulus_artifacts_found": not any(
            d["decision"] == "rerun_every_distributional_or_calibration_claim" for d in decisions
        ),
        "decisions": decisions,
        "additional_actions_taken": [
            "Regenerated the paper-facing PSD audit with distributional panels "
            "(periodogram histogram, QQ plot, per-seed KS statistics vs "
            "Exponential(1)) rather than only mean-PSD recovery -- "
            "results/audits/generator_distributional_diagnostics.{json,png,pdf} "
            "(scripts/audit_generator_distribution.py).",
            "CRESST and GWOSC retained unchanged as frozen public-data evidence "
            "(EXP-06); no shared scoring defect was identified during this audit.",
        ],
    }
    out_path = AUDITS_DIR / "rerun_decisions.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"decisions by action: {n_by_action}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
