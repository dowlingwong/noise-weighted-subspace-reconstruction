"""Round 12 calibration-size sweep: conditions and fail-closed validation.

E12 varies exactly one frozen quantity, ``N_calibration``, and holds every
other scientific choice at ``PRIMARY``. It is **post-hoc exploratory** by
construction: it was designed after the confirmatory result existed, so it can
support a limitation with a curve, never convert that limitation into a
robustness claim.

Two properties make the sweep checkable rather than merely plausible.

**The anchor condition is E10b.** ``config_hash`` is a hash of the whole frozen
configuration, so the ``N_cal = 2048`` condition necessarily has E10b's hash,
``98001a62855cb9a5``. That is not a coincidence to be admired; it is a gate.
:func:`validate_anchor_reproduces_e10b` requires the anchor's NFPA rows to
reproduce the archived E10b values to ``REPRODUCTION_RTOL``. If the sweep code
has drifted from the runner that produced the archived evidence, the anchor
fails and nothing is written.

**The control is covariance-independent.** Iso-MPCA never touches the estimated
covariance, so its held-out metrics must be identical at *every* calibration
size and must equal archived E1. A control that moves with ``N_cal`` means the
estimated covariance is leaking into the arm that is supposed to be blind to
it, which would invalidate every contrast in the sweep.

Nothing here makes a scientific choice: no threshold on degradation, no
predicted direction, no success criterion. An unfavourable curve is a result.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import r10_common as rc
from . import r10_validation as rv
from .config import PRIMARY, PrimaryConfig, config_hash
from .r9_common import IntegrityError

EXPERIMENT = "E12_calibration_size_sweep"

#: Post-hoc and exploratory. This string is written into every row, every
#: manifest and the sweep summary, so no downstream reader can mistake the
#: curve for confirmatory evidence.
ANALYSIS_LABEL = "post_hoc_exploratory_calibration_size_sweep"
EVIDENCE_CLASS = "post_hoc_exploratory_not_confirmatory"

#: The frozen calibration sizes. 2048 is the anchor and must stay in the list:
#: removing it removes the only gate that ties this sweep to archived evidence.
CALIBRATION_SIZES: tuple[int, ...] = (256, 512, 1024, 2048, 8192)

#: ``N_cal`` of the archived E10b condition, and its archived config hash.
ANCHOR_N_CAL = 2048
ANCHOR_CONFIG_HASH = "98001a62855cb9a5"

ARCHIVED_E10B_DIR = rc.RESULTS_ROOT / "E10b_calibrated_estimated_covariance"

#: Seeds are the frozen confirmatory set. The sweep varies calibration size,
#: not the seed budget; a different seed set would confound the two.
SEEDS: tuple[int, ...] = tuple(range(10))


def condition_label(n_cal: int) -> str:
    return f"n_cal_{int(n_cal)}"


def condition_config(n_cal: int) -> PrimaryConfig:
    """``PRIMARY`` with one field changed. ``PRIMARY`` itself is never mutated."""
    if int(n_cal) <= 0:
        raise IntegrityError(f"calibration size must be positive, got {n_cal!r}")
    return replace(PRIMARY, N_calibration=int(n_cal))


def validate_condition_set(sizes: Iterable[int]) -> list[int]:
    """The sweep must contain the anchor and no duplicates."""
    got = [int(n) for n in sizes]
    if len(set(got)) != len(got):
        raise IntegrityError(f"duplicate calibration sizes in {got}")
    if ANCHOR_N_CAL not in got:
        raise IntegrityError(
            f"the sweep must include the anchor N_cal={ANCHOR_N_CAL}; without it the "
            "sweep is untethered from archived E10b evidence"
        )
    return sorted(got)


def assert_anchor_hash_matches_archive() -> str:
    """The anchor condition must hash to the archived E10b configuration.

    Checked before any computation. If ``PRIMARY`` has been edited since the
    freeze, this fails here rather than after four hours of compute.
    """
    got = config_hash(condition_config(ANCHOR_N_CAL))
    if got != ANCHOR_CONFIG_HASH:
        raise IntegrityError(
            f"anchor config_hash {got} != archived E10b hash {ANCHOR_CONFIG_HASH}. "
            "PRIMARY changed since the Round 10 freeze; the sweep cannot be "
            "compared with archived evidence until that is resolved."
        )
    return got


def load_archived_e10b() -> dict[tuple[int, str], dict[str, float]]:
    """``(seed, method) -> archived E10b held-out metrics``, digest-checked."""
    rv.require_authoritative_archive_unchanged()
    table: dict[tuple[int, str], dict[str, float]] = {}
    for row in rc.read_rows_csv(ARCHIVED_E10B_DIR / "raw.csv"):
        table[(int(row["seed"]), row["method"])] = {
            "test_raw_mse": float(row["test_raw_mse"]),
            "test_normalized_chi2": float(row["test_normalized_chi2"]),
        }
    if not table:
        raise IntegrityError("archived E10b raw.csv produced no rows")
    return table


def validate_anchor_reproduces_e10b(
    rows: Sequence[dict], *, tolerance: float = rc.REPRODUCTION_RTOL
) -> dict[str, Any]:
    """The N_cal=2048 condition must reproduce archived E10b, recomputed here.

    Every comparison is against the archived file, never against a stored
    column in the new rows: a runner that wrote a convenient number into its
    own output could otherwise certify itself.
    """
    archived = load_archived_e10b()
    worst = 0.0
    checked = 0
    for row in rows:
        key = (int(row["seed"]), row["method"])
        if key not in archived:
            raise IntegrityError(
                f"{EXPERIMENT}: anchor row {key} has no archived E10b counterpart"
            )
        for metric in ("test_raw_mse", "test_normalized_chi2"):
            observed = float(row[metric])
            reference = archived[key][metric]
            err = abs(observed - reference) / max(abs(reference), 1e-15)
            worst = max(worst, err)
            checked += 1
            if err > tolerance:
                raise IntegrityError(
                    f"{EXPERIMENT}: anchor N_cal={ANCHOR_N_CAL} does not reproduce "
                    f"archived E10b at seed {key[0]} method {key[1]} metric {metric}: "
                    f"relative error {err:.3e} > {tolerance:.1e}. The sweep code has "
                    "drifted from the runner that produced the archived evidence."
                )
    return {
        "anchor_n_cal": ANCHOR_N_CAL,
        "comparisons": checked,
        "worst_relative_error": worst,
        "tolerance": tolerance,
        "passed": True,
    }


def validate_control_is_covariance_independent(
    by_condition: dict[int, Sequence[dict]], *, tolerance: float = rc.REPRODUCTION_RTOL
) -> dict[str, Any]:
    """Iso-MPCA must be identical across calibration sizes, and equal archived E1.

    The control never sees the estimated covariance. If it moves with
    ``N_cal``, estimation is leaking into the arm that defines the contrast,
    and every number in the sweep is suspect.
    """
    authoritative = rv.load_authoritative_metrics()
    reference: dict[int, dict[str, float]] = {}
    worst_across_conditions = 0.0
    worst_vs_archive = 0.0

    for n_cal in sorted(by_condition):
        for row in by_condition[n_cal]:
            if row["method"] != "Iso-MPCA":
                continue
            seed = int(row["seed"])
            values = {m: float(row[m]) for m in ("test_raw_mse", "test_normalized_chi2")}

            archived = authoritative.get((seed, "Iso-MPCA"))
            if archived is None:
                raise IntegrityError(
                    f"{EXPERIMENT}: no authoritative Iso-MPCA record for seed {seed}"
                )
            for metric, observed in values.items():
                err = abs(observed - float(archived[metric])) / max(
                    abs(float(archived[metric])), 1e-15
                )
                worst_vs_archive = max(worst_vs_archive, err)
                if err > tolerance:
                    raise IntegrityError(
                        f"{EXPERIMENT}: covariance-independent control drifted from "
                        f"archived E1 at N_cal={n_cal} seed {seed} {metric}: "
                        f"relative error {err:.3e} > {tolerance:.1e}"
                    )

            if seed not in reference:
                reference[seed] = values
                continue
            for metric, observed in values.items():
                ref = reference[seed][metric]
                err = abs(observed - ref) / max(abs(ref), 1e-15)
                worst_across_conditions = max(worst_across_conditions, err)
                if err > tolerance:
                    raise IntegrityError(
                        f"{EXPERIMENT}: control changed with calibration size at seed "
                        f"{seed} {metric} (N_cal={n_cal}): relative error {err:.3e}. "
                        "The estimated covariance is leaking into the control arm."
                    )
    return {
        "worst_relative_error_across_conditions": worst_across_conditions,
        "worst_relative_error_vs_archived_e1": worst_vs_archive,
        "tolerance": tolerance,
        "passed": True,
    }


def validate_condition_rows(
    rows: Sequence[dict],
    *,
    n_cal: int,
    columns: Sequence[str],
    seeds: Iterable[int],
) -> dict[str, Any]:
    """Structural gates, re-derived from the final rows of one condition."""
    cfg_hash = config_hash(condition_config(n_cal))
    rv._check_common(
        rows,
        experiment=f"{EXPERIMENT}[{condition_label(n_cal)}]",
        seeds=seeds,
        methods=("Iso-MPCA", "NFPA"),
        columns=columns,
        config_hash=cfg_hash,
        regime="colored",
        analysis_label=ANALYSIS_LABEL,
    )
    for row in rows:
        if int(row["n_calibration_events"]) != int(n_cal):
            raise IntegrityError(
                f"{EXPERIMENT}: seed {row['seed']} records n_calibration_events "
                f"{row['n_calibration_events']} in the N_cal={n_cal} condition"
            )
        for metric in ("test_raw_mse", "test_normalized_chi2"):
            if not rv._finite(row[metric]):
                raise IntegrityError(
                    f"{EXPERIMENT}: non-finite {metric} at seed {row['seed']}"
                )
    return {"condition": condition_label(n_cal), "config_hash": cfg_hash, "validated": True}


def sweep_summary(by_condition: dict[int, Sequence[dict]]) -> dict[str, Any]:
    """Per-condition degradation and contrast, plus the monotonicity diagnostic.

    No threshold and no predicted direction: the trend is *described*, and
    whether it is monotone is recorded as an observation, not a gate.
    """
    out: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "analysis_label": ANALYSIS_LABEL,
        "evidence_class": EVIDENCE_CLASS,
        "anchor_n_cal": ANCHOR_N_CAL,
        "calibration_sizes": sorted(by_condition),
        "seeds": sorted({int(r["seed"]) for rows in by_condition.values() for r in rows}),
        "scope_note": (
            "Post-hoc exploratory sweep over calibration size only. Every other "
            "frozen setting is unchanged. This supports the estimated-covariance "
            "limitation with a curve; it does not establish a robustness boundary, "
            "and it is not confirmatory evidence for any hypothesis."
        ),
        "interpretation_note": (
            "NFPA's fit is invariant to a global rescaling of both whitening "
            "factors, so degradation reflects covariance SHAPE error rather than "
            "scale error. Reading a shape-accuracy curve as a general robustness "
            "statement would overstate it."
        ),
    }
    per_condition: dict[str, Any] = {}
    for n_cal in sorted(by_condition):
        rows = list(by_condition[n_cal])
        nfpa = [r for r in rows if r["method"] == "NFPA"]
        block: dict[str, Any] = {
            "n_calibration_events": int(n_cal),
            "config_hash": config_hash(condition_config(n_cal)),
            "n_seeds": len(nfpa),
            "is_anchor": int(n_cal) == ANCHOR_N_CAL,
        }
        for col in (
            "delta_normalized_chi2_estimated_minus_oracle",
            "delta_raw_mse_estimated_minus_oracle",
        ):
            block[f"degradation.{col}"] = rc.bootstrap_paired([float(r[col]) for r in nfpa])
        for metric in ("test_normalized_chi2", "test_raw_mse"):
            block[f"contrast_nfpa_minus_iso.{metric}"] = rc.paired_contrast(
                rows, metric=metric, method_a="NFPA", method_b="Iso-MPCA"
            )
        for col in (
            "product_scale_ratio_vs_oracle",
            "channel_shape_error",
            "temporal_shape_error",
            "kronecker_relative_frobenius_error",
        ):
            vals = [float(r[col]) for r in nfpa]
            block[f"diagnostics.{col}"] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        per_condition[condition_label(n_cal)] = block
    out["per_condition"] = per_condition

    sizes = sorted(by_condition)
    for col, name in (
        ("channel_shape_error", "channel_shape_error"),
        ("temporal_shape_error", "temporal_shape_error"),
    ):
        means = [
            per_condition[condition_label(n)][f"diagnostics.{col}"]["mean"] for n in sizes
        ]
        out[f"trend.{name}"] = {
            "calibration_sizes": sizes,
            "means": means,
            "monotone_decreasing": all(a >= b for a, b in zip(means, means[1:])),
            "note": "observation, not a gate; no direction was predeclared",
        }
    reversal = {
        condition_label(n): bool(
            per_condition[condition_label(n)][
                "contrast_nfpa_minus_iso.test_normalized_chi2"
            ]["mean_difference"]
            < 0
            and per_condition[condition_label(n)][
                "contrast_nfpa_minus_iso.test_raw_mse"
            ]["mean_difference"]
            > 0
        )
        for n in sizes
    }
    out["reversal_sign_holds_per_condition"] = reversal
    out["reversal_sign_holds_everywhere"] = all(reversal.values())
    return out


def write_sweep_summary(out_root: Path, by_condition: dict[int, Sequence[dict]]) -> dict[str, Any]:
    summary = sweep_summary(by_condition)
    rc.write_json(out_root / "sweep_summary.json", summary)
    return summary


def refuse_archived_round10_dirs(out_root: Path) -> None:
    """E10b and E11 are archived evidence and are not in Round 10's refusal list.

    ``r10_common.ARCHIVED_DIRS`` was frozen before E10b and E11 existed, so it
    protects E1, E2, E8, E9 and E10 but not the two directories this sweep is
    compared against. Guard them here rather than editing a pinned Round 10
    file.
    """
    protected = (
        ARCHIVED_E10B_DIR,
        rc.RESULTS_ROOT / "E11_corrected_angles",
    )
    resolved = Path(out_root).expanduser().resolve()
    for archived in protected:
        try:
            resolved.relative_to(archived.resolve())
        except (ValueError, FileNotFoundError):
            continue
        raise IntegrityError(
            f"output directory {resolved} resolves into archived evidence "
            f"{archived}; E10b and E11 are read-only"
        )


def assert_sweep_summary_regenerates(
    out_root: Path, sizes: Sequence[int], written: dict[str, Any]
) -> dict[str, Any]:
    """The summary must be a pure function of the per-condition raw files.

    Re-read every ``raw.csv`` from disk, recompute, and compare canonical
    digests. A summary that cannot be re-derived from the rows it claims to
    describe is not evidence.
    """
    reread: dict[int, list[dict]] = {}
    for n_cal in sorted(sizes):
        raw = out_root / condition_label(n_cal) / "raw.csv"
        if not raw.exists():
            raise IntegrityError(f"{EXPERIMENT}: missing {raw} during regeneration audit")
        reread[int(n_cal)] = list(rc.read_rows_csv(raw))
    recomputed = sweep_summary(reread)
    want = rv.canonical_json_digest(written)
    got = rv.canonical_json_digest(recomputed)
    if want != got:
        raise IntegrityError(
            f"{EXPERIMENT}: sweep summary does not regenerate from the archived "
            f"rows ({got} != {want})"
        )
    return {"passed": True, "digest": got, "conditions": sorted(int(n) for n in sizes)}
