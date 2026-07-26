"""EXP-06: package frozen CRESST/GWOSC public-data evidence for the paper bundle.

No reruns, no threshold/window/radius changes. This script only reads
existing frozen evidence under ``transfer_paper/data/`` and ``evidence/`` and
writes a stable, paper-ready summary table for each dataset.

CRESST: exports pooled effects, equal-channel effects, hierarchical
intervals, all channel means, cluster count, run count, and the wrong-PSD
exception on channel ``35_58.0``, captioned to state that trigger efficiency
was inconclusive and no detector-sensitivity claim is supported.

GWOSC: exports implementation-identity metrics (shared-FIR statistic
equivalence) and calibration metrics (null calibration, time-local PSD) as
SEPARATE files with explicit status fields, so a paper-writing agent cannot
combine an implementation-identity "accepted: True" with a calibration
"accepted: False" into one misleading "passed" summary.

    uv run python scripts/package_public_data_evidence.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSFER = REPO_ROOT / "transfer_paper" / "data"
OUT_ROOT = REPO_ROOT / "results" / "public_data_package"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# CRESST
# --------------------------------------------------------------------------


def package_cresst() -> Path:
    pooled_rows = read_csv_rows(TRANSFER / "derived" / "cresst_pooled_summary.csv")
    per_channel_rows = read_csv_rows(TRANSFER / "derived" / "cresst_per_channel_summary.csv")
    sensitivity_dir = TRANSFER / "cresst" / "sensitivity" / "20260713_channel_robustness_v1"
    estimand_rows = read_csv_rows(sensitivity_dir / "estimand_summary.csv")
    channel_rows = read_csv_rows(sensitivity_dir / "channel_summary.csv")
    sensitivity_summary = json.loads((sensitivity_dir / "summary.json").read_text())

    channels = [r["channel"] for r in per_channel_rows]
    run_prefixes = sorted({c.split("_")[0] for c in channels})
    cluster_count = len(channels)
    run_count = len(run_prefixes)

    pooled = pooled_rows[0]
    wrong_psd_exception = {
        "channel": "35_58.0",
        "pooled_control_wrong_psd_mean_difference": float(pooled["control_wrong_psd_mean_difference"]),
        "pooled_control_wrong_psd_sign_reversed_channels": pooled["control_wrong_psd_sign_reversed_channels"],
        "per_channel_mean_difference": next(
            float(r["control_wrong_psd_mean_difference"]) for r in per_channel_rows if r["channel"] == "35_58.0"
        ),
        "per_channel_exceeds_zero": next(
            r["control_wrong_psd_exceeds_zero"] for r in per_channel_rows if r["channel"] == "35_58.0"
        ),
        "explanation": (
            "wrong_psd (calibrate with an identity metric instead of the true "
            "covariance) did not degrade the result on channel 35_58.0 alone "
            "-- the only sign reversal across all 5 controls x 7 channels, and "
            "the weakest-margin control pooled. Consistent with that channel's "
            "true noise being closer to white/uncorrelated than the others; "
            "reported as a limitation, not smoothed over. See docs/results/CRESST_RESULT.md."
        ),
    }

    estimands: dict[str, Any] = {}
    for row in estimand_rows:
        name = row["estimand"]
        estimands[name] = {
            "pooled_event_weighted_mean": float(row["pooled_event_weighted_mean"]),
            "equal_channel_mean": float(row["equal_channel_mean"]),
            "hierarchical_ci95": [float(row["hierarchical_ci95_lo"]), float(row["hierarchical_ci95_hi"])],
            "channel_means": {c["channel"]: float(c.get(name, "nan")) for c in channel_rows if name in c},
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CRESST-A",
        "status": "frozen_primary_plus_post_freeze_sensitivity",
        "no_rerun_policy": "EXP-06 packaging only; no threshold, window, or grid was changed to produce this file.",
        "primary_evidence": "results/metrics/p2_cresst_validation_seed22.json (evidence/cresst/20260704T013758Z_9c9f696ff6bc/)",
        "post_freeze_sensitivity_evidence": str((sensitivity_dir).relative_to(REPO_ROOT)),
        "cluster_count": cluster_count,
        "cluster_definition": "run/channel clusters (each of the 7 channels is its own cluster)",
        "run_count": run_count,
        "run_ids": run_prefixes,
        "channels": channels,
        "pooled_effects": {k: v for k, v in pooled.items()},
        "equal_channel_and_hierarchical_effects": estimands,
        "wrong_psd_exception": wrong_psd_exception,
        "cluster_caution": sensitivity_summary.get("cluster_caution"),
        "trigger_efficiency_status": (
            "Inconclusive within the predeclared amplitude grid for most channels "
            "(no 50%/90% crossing for 6 of 7 channels; NaN thresholds). Does not "
            "contradict the primary linear-hierarchy gate, but must not be "
            "described as validated detection sensitivity. See docs/results/CRESST_RESULT.md."
        ),
        "no_detection_sensitivity_claim_supported": True,
        "caption": (
            "CRESST-A public real-data validation: pooled event-weighted paired-"
            "bootstrap effects are primary; equal-channel and channel-then-event "
            "hierarchical-bootstrap effects are post-freeze sensitivity, reported "
            "alongside rather than in place of the primary gate. One channel-level "
            "sign reversal exists in the weakest-margin control (wrong_psd, "
            "channel 35_58.0). Trigger efficiency was inconclusive at the "
            "predeclared amplitude grid; no detector-sensitivity claim is supported."
        ),
    }

    out_dir = OUT_ROOT / "cresst"
    write_json(summary, out_dir / "cresst_stable_summary.json")

    flat_rows = []
    for name, est in estimands.items():
        for channel, mean in est["channel_means"].items():
            flat_rows.append(
                {
                    "estimand": name,
                    "channel": channel,
                    "channel_mean": mean,
                    "pooled_event_weighted_mean": est["pooled_event_weighted_mean"],
                    "equal_channel_mean": est["equal_channel_mean"],
                    "hierarchical_ci95_lo": est["hierarchical_ci95"][0],
                    "hierarchical_ci95_hi": est["hierarchical_ci95"][1],
                }
            )
    write_csv(flat_rows, out_dir / "cresst_stable_summary.csv")

    md = [
        "# CRESST-A stable summary (EXP-06)",
        "",
        summary["caption"],
        "",
        f"- Cluster count: {cluster_count} (channels: {', '.join(channels)})",
        f"- Run count: {run_count} (run prefixes: {', '.join(run_prefixes)})",
        f"- wrong_psd exception: channel {wrong_psd_exception['channel']}, "
        f"pooled mean difference {wrong_psd_exception['pooled_control_wrong_psd_mean_difference']:.4f} "
        f"(weakest-margin control; only sign reversal across 5 controls x 7 channels)",
        "",
        "Trigger efficiency: " + summary["trigger_efficiency_status"],
    ]
    (out_dir / "cresst_stable_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}")
    return out_dir


# --------------------------------------------------------------------------
# GWOSC
# --------------------------------------------------------------------------


def package_gwosc() -> Path:
    out_dir = OUT_ROOT / "gwosc"

    # -- Implementation-identity metrics (shared-FIR statistic equivalence) --
    fir_rows = read_csv_rows(TRANSFER / "derived" / "gwosc_filter_equivalence_summary.csv")
    identity_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GWOSC",
        "metric_class": "implementation_identity",
        "status": "verified_positive_for_implementation_identity_only",
        "scope": (
            "Confirms two independent code paths (explicit FFT convolution and "
            "GWpy FIR convolution) compute the SAME shared FIR statistic "
            "identically, for synthetic and real H1/L1 windows. This is an "
            "implementation-correctness check, NOT a calibration or detection-"
            "sensitivity result, and must never be reported alongside the "
            "calibration metrics below as a single combined 'passed' status."
        ),
        "rows": fir_rows,
        "all_accepted": all(r["accepted"] == "True" for r in fir_rows),
    }
    write_json(identity_summary, out_dir / "gwosc_implementation_identity_summary.json")
    write_csv(fir_rows, out_dir / "gwosc_implementation_identity_summary.csv")

    # -- Calibration metrics (global + time-local null calibration) --
    primary_rows = read_csv_rows(TRANSFER / "derived" / "gwosc_primary_results.csv")
    local_rows = read_csv_rows(TRANSFER / "derived" / "gwosc_time_local_psd_summary.csv")
    calibration_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "GWOSC",
        "metric_class": "real_noise_calibration",
        "status": "verified_negative_boundary_result",
        "no_retune_policy": (
            "Preserved as a negative boundary result. No threshold, PSD radius, "
            "window, or quality cut was changed after inspecting this result "
            "(EXP-06 is packaging only)."
        ),
        "global_psd_null_calibration": {
            "rows": primary_rows,
            "summary": (
                "H1 random median ratio 1.986, chronological 3.183; L1 random "
                "1.263, chronological 2.964 -- all exceed the acceptance gate "
                "(ratio near 1 expected under correct calibration)."
            ),
        },
        "time_local_psd_calibration": {
            "rows": local_rows,
            "summary": (
                "Synthetic control (local_radius_64s) passed (accepted=True); "
                "the same predeclared primary local model failed on real H1 and "
                "L1 (accepted=False)."
            ),
        },
        "scope": (
            "Real-noise calibration requires more than a correct PSD "
            "implementation and a predeclared local PSD model. Do not state "
            "that this method is calibrated on GWOSC data, and do not combine "
            "this status with the implementation-identity file above."
        ),
    }
    write_json(calibration_summary, out_dir / "gwosc_calibration_summary.json")
    write_csv(primary_rows, out_dir / "gwosc_calibration_primary_results.csv")
    write_csv(local_rows, out_dir / "gwosc_calibration_time_local_psd.csv")

    md = [
        "# GWOSC stable summary (EXP-06)",
        "",
        "Two SEPARATE metric classes -- do not merge into one status.",
        "",
        "## Implementation identity (verified positive)",
        identity_summary["scope"],
        f"All settings accepted: {identity_summary['all_accepted']}",
        "",
        "## Real-noise calibration (verified negative boundary result)",
        calibration_summary["scope"],
        "- Global PSD: " + calibration_summary["global_psd_null_calibration"]["summary"],
        "- Time-local PSD: " + calibration_summary["time_local_psd_calibration"]["summary"],
    ]
    (out_dir / "gwosc_stable_summary.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out_dir.relative_to(REPO_ROOT)}")
    return out_dir


def main() -> None:
    package_cresst()
    package_gwosc()


if __name__ == "__main__":
    main()
