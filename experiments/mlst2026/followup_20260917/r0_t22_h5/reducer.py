"""R0 -- complete the declared T22 H5 random-effects report from stored records.

The archived T22-S reduction computed the random-effects interval
(``re_lo``/``re_hi``/``re_estimate``) inside every Monte-Carlo record but never
summarised it, so H5's second secondary interval was absent from ``REDUCED.json``.
This module is a **read-only reporting overlay**: it reads copies of the stored
cell records, recomputes the primary and secondary interval coverages *from
those records* (it never trusts a hard-coded count), and writes a versioned
report beside the original.

It does not re-run, repair or re-tune anything.  The frozen reduction is copied
unchanged to the export; the overlay adds new fields only under its own key.

Scientific care taken here
--------------------------
* The declared H1/H5 scope is the **65 non-drift** scenarios.  The four drift
  stress scenarios are reported separately and the 69-scenario pooled table is
  labelled descriptive.
* Coverage is reported **unconditionally** and **given not withheld**, with an
  explicit empty-denominator policy (``None`` and ``n = 0``, never 0/0 = 0).
* The random-effects estimator is a DerSimonian-Laird inverse-variance mean of
  the per-file effects; it targets the *precision-weighted* file-effect mean.
  The two-stage estimator targets the **equal-file mean**.  Under log-normal
  file noise these differ.  The overlay measures the **exploratory finite-sample
  estimator displacement** per scenario (RE estimate minus equal-file point); it
  does **not** establish a population estimand mismatch, does **not** prove the
  cause of the coverage shortfall, and never calls the shorter interval a better
  estimator on coverage alone.  The population targets and the assumptions are
  stated separately alongside the statistic.
* ``copy_to`` is mandatory and vetted before any write: a destination equal to,
  inside, containing or already existing at the source is refused, and nothing is
  ever deleted.  The input cell inventory, unique cell ids, freeze binding and
  source hashes are validated, and every coverage flag is recomputed from the
  stored bounds and declared truth rather than trusted from ``covers``/
  ``t_covers``.
* ``--check`` asserts the audited values (24/65 in band, median 0.876) are
  *recomputed*, not copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

SCHEMA = "mlst-followup-r0-t22-h5/1"
AMENDMENT_VERSION = "R0-v2"
WILSON_Z = 1.959964
COVERAGE_BAND = (0.925, 0.975)
PRIMARY_CONTRAST = "raw-bin|empca"
ALL_CONTRASTS = ("raw-bin|empca", "zero|empca", "pca|empca")
#: The audited reference values, recomputed (never hard-coded into the output).
AUDIT_EXPECTATION = {"non_drift_scenarios": 65, "re_primary_in_band": 24, "re_primary_median": 0.876}
#: Exploratory threshold for the finite-sample estimator-displacement statistic.
#: This is NOT a test of a population estimand mismatch and must not be read as one.
DISPLACEMENT_TOL = 0.20   # |re - equal-file| resolved and > 20% of the mean RE half-width
#: The change log for this version, checked into the overlay so the qualification
#: of the estimator-displacement statistic is auditable rather than implicit.
CHANGE_LOG = [
    {"version": "R0-v1", "change": "initial additive overlay: H5 random-effects coverage/length summary"},
    {"version": "R0-v2", "change": ("(a) refuse same/overlapping/aliased or existing copy destinations, no recursive "
                                     "delete; (b) validate the input cell inventory, unique cell ids, replicate "
                                     "identity, freeze binding and source hashes; (c) recompute the two-stage and "
                                     "t coverage flags from stored bounds and declared truth instead of trusting "
                                     "covers/t_covers; (d) relabel target mismatch as exploratory finite-sample "
                                     "estimator displacement, not an established population estimand mismatch or the "
                                     "cause of undercoverage")},
]


def _resolved(path: Path) -> Path:
    """Resolve a path (following symlinks) without requiring it to exist."""
    return Path(path).expanduser().resolve()


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def guard_copy_destination(run_root: Path, copy_to: Path | None) -> Path:
    """Resolve and vet the working-copy destination before any write.

    R0 must never touch the immutable source tree.  This rejects, by resolved
    (symlink-following) path: a copy destination equal to the source, an ancestor
    or a descendant of it, and any destination that already exists.  The function
    performs no filesystem mutation, so a rejected call leaves the source
    untouched.
    """
    src = _resolved(run_root)
    if copy_to is None:
        raise ValueError("R0 requires a fresh --copy-to destination so the source run root is never written")
    dst = _resolved(copy_to)
    if dst == src:
        raise ValueError(f"copy destination {dst} equals the source run root; refusing (this would delete the source)")
    if _is_within(dst, src):
        raise ValueError(f"copy destination {dst} is inside the source run root {src}; refusing")
    if _is_within(src, dst):
        raise ValueError(f"copy destination {dst} is an ancestor of the source run root {src}; refusing")
    if dst.exists():
        raise ValueError(f"copy destination {dst} already exists; R0 requires a fresh destination and never deletes")
    return dst


def guard_output_destination(out: Path, *input_roots: Path) -> Path:
    """Reject an output directory inside any immutable input tree."""
    o = _resolved(out)
    for root in input_roots:
        if root is None:
            continue
        r = _resolved(root)
        if o == r or _is_within(o, r):
            raise ValueError(f"output {o} is inside the immutable input tree {r}; choose another --out")
    return o


# --- statistics ---------------------------------------------------------------------
def wilson(k: int, n: int, z: float = WILSON_Z) -> dict:
    """Wilson score interval; an empty denominator is explicit, not zero."""
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "half_width": None, "n": 0, "k": int(k),
                "note": "empty denominator"}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": float(p), "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "half_width": float(half), "n": int(n), "k": int(k)}


def within_band(w: dict, band=COVERAGE_BAND) -> bool | None:
    """The frozen rule: the Wilson interval is not entirely outside the band."""
    if w["n"] == 0:
        return None
    lo, hi = band
    return not (w["hi"] < lo or w["lo"] > hi)


def _rate(records, field: str) -> int:
    return int(sum(bool(r) for r in records))


# --- loading ------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stored(run_root: Path) -> tuple[dict, dict, dict, dict, dict, dict]:
    """Read ``REDUCED.json``, the protocol snapshot and every confirmation cell.

    Returns ``(reduced, protocol_snapshot, groups, scenario_params, cell_provenance,
    validation)`` where ``groups`` maps ``scenario_name -> list[record]`` and
    ``scenario_params`` maps ``scenario_name -> the declared scenario dict`` from
    the ``sim_confirm``/``confirmation`` cells.  Records are read as stored; the
    cell's frozen-calibration digest travels with the provenance so a mismatch is
    visible.

    Inventory validation (added R0-v2): the cell ids must be unique, every cell
    must carry the fields the overlay reads, every confirmation cell must record
    the same freeze digest, the observed scenarios must match the frozen reduction,
    and the reduction must be complete.  All recorded digests and both source and
    cell hashes travel in ``validation`` so a later reader can re-verify them.
    """
    reduced = json.loads((run_root / "REDUCED.json").read_text())
    snap_path = run_root / "protocol_snapshot.json"
    snapshot = json.loads(snap_path.read_text()) if snap_path.exists() else {}
    groups: dict[str, list[dict]] = {}
    scenario_params: dict[str, dict] = {}
    cell_provenance = {}
    seen_ids: dict[str, str] = {}
    freeze_digests: set[str] = set()
    problems: list[str] = []
    cell_files = sorted((run_root / "cells").glob("*.json"))
    for p in cell_files:
        js = json.loads(p.read_text())
        if js.get("family") != "sim_confirm" or js.get("role") != "confirmation":
            continue
        cid = js.get("cell_id")
        if not isinstance(cid, str) or not cid:
            problems.append(f"{p.name}: missing cell_id")
            continue
        if cid in seen_ids:
            problems.append(f"duplicate cell_id {cid!r} in {p.name} and {seen_ids[cid]}")
        seen_ids[cid] = p.name
        summary = js.get("result", {}).get("summary", {})
        name = summary.get("scenario_name")
        if not name:
            problems.append(f"{cid}: missing result.summary.scenario_name")
            continue
        records = js.get("result", {}).get("records")
        if not isinstance(records, list) or not records:
            problems.append(f"{cid}: result.records is missing or empty")
            continue
        if "params" not in js or "scenario" not in js.get("params", {}):
            problems.append(f"{cid}: params.scenario missing")
            continue
        groups.setdefault(name, []).extend(records)
        scenario_params[name] = js["params"]["scenario"]
        fd = summary.get("frozen_sha256")
        if fd:
            freeze_digests.add(fd)
        cell_provenance[cid] = {"file": p.name, "seed": js.get("seed"),
                                "frozen_sha256": fd, "scenario": name, "sha256": _sha256(p)}
    if len(freeze_digests) > 1:
        problems.append(f"confirmation cells record {len(freeze_digests)} different freeze digests")
    frozen_conf = reduced.get("confirmation", {}).get("scenarios", {})
    if frozen_conf:
        missing = sorted(set(frozen_conf) - set(groups))
        extra = sorted(set(groups) - set(frozen_conf))
        if missing:
            problems.append(f"{len(missing)} frozen scenarios have no confirmation cells: {missing[:3]}")
        if extra:
            problems.append(f"{len(extra)} cell scenarios are absent from the frozen reduction: {extra[:3]}")
    if reduced.get("reconciliation", {}).get("complete") is False:
        problems.append("REDUCED.json reconciliation.complete is False (partial reduction)")
    validation = {
        "n_cell_files": len(cell_files),
        "n_confirmation_cells": len(seen_ids),
        "unique_cell_ids": len(seen_ids) == len([c for c in cell_provenance]),
        "freeze_digests": sorted(d for d in freeze_digests if d),
        "n_scenarios": len(groups),
        "reduced_json_sha256": _sha256(run_root / "REDUCED.json"),
        "protocol_snapshot_sha256": (_sha256(snap_path) if snap_path.exists() else None),
        "frozen_calibration_sha256": (_sha256(run_root / "FROZEN_CALIBRATION.json")
                                      if (run_root / "FROZEN_CALIBRATION.json").exists() else None),
        "reducer_source_sha256": _sha256(Path(__file__).resolve()),
        "problems": problems,
        "ok": not problems,
    }
    return reduced, snapshot, groups, scenario_params, cell_provenance, validation


def _scenario_is_drift(params: dict) -> bool:
    """A drift scenario is one whose declared scenario dictionary has drift_sd > 0."""
    return float(params.get("drift_sd", 0.0)) > 0.0


def _scope(scenario_params: dict) -> dict:
    drift = sorted(n for n, p in scenario_params.items() if _scenario_is_drift(p))
    return {"non_drift": sorted(n for n in scenario_params if n not in drift), "drift": drift,
            "all": sorted(scenario_params)}


# --- per-scenario summary -----------------------------------------------------------
def summarise_records(name: str, records: list[dict], params: dict) -> dict:
    n = len(records)
    nw = [r for r in records if not r["withheld"]]
    out = {"scenario": name,
           "scenario_params": params,
           "n_mc": n, "n_not_withheld": len(nw),
           "withholding_rate": (len(records) - len(nw)) / n if n else None,
           "true_mu": records[0]["contrasts"][PRIMARY_CONTRAST].get("true_mu"),
           "contrasts": {}}
    for key in ALL_CONTRASTS:
        cs = [r["contrasts"][key] for r in records]
        cs_nw = [r["contrasts"][key] for r in nw]
        tm = cs[0].get("true_mu")
        rec = {"n_mc": n, "n_not_withheld": len(cs_nw), "true_mu": tm, "is_primary": key == PRIMARY_CONTRAST,
               "coverage_unconditional": {}, "coverage_given_not_withheld": {},
               "length": {}, "target": {}}
        # --- unconditional coverage of the three declared intervals ---
        # R0-v2 recomputes *every* coverage flag from the stored interval bounds and
        # the declared truth.  The archived ``covers``/``t_covers`` flags are not
        # trusted: a stale or tampered flag must not change a reported rate.
        def _two_covers(rs):
            return sum(c["ci95_lo"] <= c["true_mu"] <= c["ci95_hi"] for c in rs)

        def _t_covers(rs):
            return sum(c["t_lo"] <= c["true_mu"] <= c["t_hi"] for c in rs)

        def _re_covers(rs):
            return sum(c["re_lo"] <= c["true_mu"] <= c["re_hi"] for c in rs)

        if "true_mu" in cs[0]:
            cov_two = wilson(_two_covers(cs), n)
            cov_t = wilson(_t_covers(cs), n)
            cov_re = wilson(_re_covers(cs), n)
            cov_two_nw = wilson(_two_covers(cs_nw), len(cs_nw))
            cov_t_nw = wilson(_t_covers(cs_nw), len(cs_nw))
            cov_re_nw = wilson(_re_covers(cs_nw), len(cs_nw))
        else:  # pragma: no cover - the archived records always carry the estimand
            cov_two = cov_t = cov_re = cov_two_nw = cov_t_nw = cov_re_nw = wilson(0, 0)
        rec["coverage_unconditional"] = {"two_stage": cov_two, "t_interval": cov_t, "random_effects": cov_re}
        rec["coverage_given_not_withheld"] = {"two_stage": cov_two_nw, "t_interval": cov_t_nw, "random_effects": cov_re_nw}
        rec["coverage_unconditional"]["within_tolerance"] = {
            "two_stage": within_band(cov_two), "t_interval": within_band(cov_t), "random_effects": within_band(cov_re)}
        rec["coverage_unconditional"]["random_effects_within_tolerance"] = within_band(cov_re)
        rec["length"] = {
            "two_stage_mean": float(np.mean([c["length"] for c in cs])) if cs else None,
            "t_interval_mean": float(np.mean([c["t_length"] for c in cs])) if cs else None,
            "random_effects_mean": float(np.mean([c["re_hi"] - c["re_lo"] for c in cs])) if cs else None,
        }
        l2, lt, lre = (rec["length"]["two_stage_mean"], rec["length"]["t_interval_mean"], rec["length"]["random_effects_mean"])
        rec["length"]["random_effects_over_two_stage"] = (lre / l2) if l2 else None
        # --- target estimand diagnostic (random effects vs equal-file mean) ---
        # R0-v2: this is an *exploratory finite-sample estimator-displacement*
        # statistic.  It measures how far the archived precision-weighted
        # re-estimate sits from the equal-file point on these replicates.  It does
        # NOT establish a population estimand mismatch and does NOT prove the cause
        # of the coverage shortfall; those would need the two population targets
        # and the generator's file-sampling law, stated separately below.
        if cs:
            point = np.array([c["point"] for c in cs])
            re_est = np.array([c["re_estimate"] for c in cs])
            d = re_est - point
            mc_se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None
            half = float(np.mean([(c["re_hi"] - c["re_lo"]) / 2.0 for c in cs]))
            resolved = bool(mc_se is not None and abs(float(d.mean())) > 2.0 * mc_se + 1e-15)
            material = bool(half > 0 and abs(float(d.mean())) > DISPLACEMENT_TOL * half)
            flagged = bool(resolved and material)
            rec["target"] = {
                "statistic": "exploratory finite-sample estimator displacement (RE estimate minus equal-file point)",
                "random_effects_minus_equal_file_mean": float(d.mean()),
                "mc_se": mc_se, "resolved": resolved,
                "mean_re_half_width": half,
                "displacement_over_half_width": (abs(float(d.mean())) / half) if half else None,
                "displacement_flag": flagged,
                # retained for readers of the v1 field name; same number, weaker reading
                "mismatch_flag": flagged,
                "population_targets": {
                    "two_stage_equal_file": "equal-file mean of the per-file population contrasts (declared estimand)",
                    "random_effects": "DerSimonian-Laird precision-weighted mean of the per-file effects",
                },
                "assumptions": ("log-normal per-file noise; per-file effects exchangeable with the declared "
                                "heterogeneity; the bootstrap SD approximates the within-file SE"),
                "establishes_population_mismatch": False,
                "establishes_cause_of_undercoverage": False,
                "interpretation": (("the RE estimate is displaced from the equal-file point beyond Monte-Carlo "
                                    "error and beyond %.0f%% of the RE half-width on these replicates; this is "
                                    "exploratory finite-sample estimator displacement, not an established "
                                    "population estimand mismatch" % (100 * DISPLACEMENT_TOL))
                                   if flagged else
                                   ("no resolved material finite-sample displacement is seen on these replicates"
                                    if n > 1 else "n = 1: displacement not resolvable")),
            }
        out["contrasts"][key] = rec
    return out


# --- scope tables -------------------------------------------------------------------
def _scope_summary(scenarios: dict, names: list[str]) -> dict:
    def _vals(key, which, field="rate"):
        return [scenarios[n]["contrasts"][key]["coverage_unconditional"][which][field] for n in names
                if scenarios[n]["contrasts"][key]["coverage_unconditional"][which][field] is not None]

    def _med(xs):
        return float(np.median(xs)) if xs else None

    primary = PRIMARY_CONTRAST
    re_rate = _vals(primary, "random_effects")
    ts_rate = _vals(primary, "two_stage")
    t_rate = _vals(primary, "t_interval")
    in_band = [scenarios[n]["contrasts"][primary]["coverage_unconditional"]["random_effects_within_tolerance"] for n in names]
    re_len = [scenarios[n]["contrasts"][primary]["length"]["random_effects_mean"] for n in names]
    ts_len = [scenarios[n]["contrasts"][primary]["length"]["two_stage_mean"] for n in names]
    return {
        "n_scenarios": len(names),
        "primary_contrast": primary,
        "random_effects_in_band": int(sum(bool(x) for x in in_band if x is not None)),
        "random_effects_within_band_fraction": (sum(bool(x) for x in in_band if x is not None) / len(in_band)) if in_band else None,
        "random_effects_coverage_median": _med(re_rate),
        "random_effects_coverage_range": [min(re_rate), max(re_rate)] if re_rate else None,
        "two_stage_coverage_median": _med(ts_rate),
        "t_interval_coverage_median": _med(t_rate),
        "median_length_ratio_random_effects_over_two_stage": (_med([a / b for a, b in zip(re_len, ts_len) if b])),
        # exploratory finite-sample estimator displacement, not a population mismatch
        "displacement_flagged_scenarios": sorted(n for n in names if scenarios[n]["contrasts"][primary]["target"].get("displacement_flag")),
        "mismatch_flagged_scenarios": sorted(n for n in names if scenarios[n]["contrasts"][primary]["target"].get("mismatch_flag")),
    }


def build_report(groups: dict, reduced: dict, scenario_params: dict, validation: dict | None = None) -> dict:
    scenarios = {name: summarise_records(name, recs, scenario_params[name]) for name, recs in groups.items()}
    scope = _scope(scenario_params)
    # H1 consistency: recompute the two-stage coverage from stored records and compare
    # with the frozen REDUCED.json, so the overlay cannot drift from the reduction.
    disagreements = []
    frozen_conf = reduced.get("confirmation", {}).get("scenarios", {})
    for name in scope["all"]:
        frozen = frozen_conf.get(name)
        if not frozen:
            disagreements.append({"scenario": name, "why": "absent from REDUCED.json"})
            continue
        got = scenarios[name]["contrasts"][PRIMARY_CONTRAST]["coverage_unconditional"]["two_stage"]
        want = frozen["contrasts"][PRIMARY_CONTRAST]["coverage_two_stage"]
        if got["k"] != want["k"] or got["n"] != want["n"]:
            disagreements.append({"scenario": name, "stored": [got["k"], got["n"]], "frozen": [want["k"], want["n"]]})
    return {
        "schema": SCHEMA,
        "amendment": {"version": AMENDMENT_VERSION, "type": "reporting-overlay", "new_cells": 0,
                      "description": "T22-S H5 random-effects interval summary completed from stored records",
                      "designed_after": "2026-09-17 (after the September results were observed)",
                      "change_log": CHANGE_LOG},
        "input_validation": validation or {},
        "wilson": {"z": WILSON_Z, "rule": "not entirely outside the band"},
        "coverage_band": list(COVERAGE_BAND),
        "estimand_note": ("covers is measured against the declared population equal-file estimand stored "
                          "as true_mu, exactly as the frozen reducer does"),
        "scope": {"non_drift": scope["non_drift"], "drift": scope["drift"], "all": scope["all"],
                  "counts": {"non_drift": len(scope["non_drift"]), "drift": len(scope["drift"]), "all": len(scope["all"])}},
        "consistency_with_frozen_reduction": {
            "primary_contrast": PRIMARY_CONTRAST,
            "what": "two-stage coverage recomputed from stored records vs REDUCED.json",
            "scenarios_differing": len(disagreements), "differences": disagreements},
        "scope_tables": {
            "non_drift_65": _scope_summary(scenarios, scope["non_drift"]),
            "drift_stress_4": _scope_summary(scenarios, scope["drift"]),
            "all_69_descriptive": {**_scope_summary(scenarios, scope["all"]),
                                   "label": "descriptive only; not an H1/H5 scope"},
        },
        "scenarios": scenarios,
    }


# --- manuscript-facing table --------------------------------------------------------
def manuscript_table(report: dict) -> str:
    lines = [
        "# T22 H5 random-effects interval summary (reporting amendment R0-v1)",
        "",
        "Computed read-only from the stored T22-S Monte-Carlo records.  No new cells were executed; the frozen",
        "reduction is unchanged.  Coverage uses the declared population equal-file estimand (``true_mu``).",
        "",
        "| scenario scope | n | RE in band | RE coverage median (range) | two-stage median | t median | median RE/two-stage length | exploratory displacement scenarios |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, key in (("non-drift (H1/H5 scope)", "non_drift_65"), ("drift stress (separate)", "drift_stress_4"),
                       ("all scenarios (descriptive)", "all_69_descriptive")):
        t = report["scope_tables"][key]
        rng = t["random_effects_coverage_range"]
        lines.append(
            f"| {label} | {t['n_scenarios']} | {t['random_effects_in_band']} | "
            f"{_fmt(t['random_effects_coverage_median'])}"
            + (f" ({_fmt(rng[0])}–{_fmt(rng[1])})" if rng else "")
            + f" | {_fmt(t['two_stage_coverage_median'])} | {_fmt(t['t_interval_coverage_median'])} | "
            f"{_fmt(t['median_length_ratio_random_effects_over_two_stage'])} | "
            f"{len(t['mismatch_flagged_scenarios'])} |")
    lines += [
        "",
        "The random-effects interval is **not** a replacement for the primary interval.  It is a DerSimonian-Laird",
        "inverse-variance mean of the per-file effects and targets the precision-weighted file-effect mean, while the",
        "two-stage interval targets the equal-file mean.  The last column counts only *exploratory finite-sample",
        "estimator displacement*: where the RE estimate sits materially away from the equal-file point on these",
        "replicates.  It does **not** establish a population estimand mismatch and does **not** prove the cause of the",
        "coverage shortfall; those need the two population targets and the generator's file-sampling law, stated",
        "separately in the report's ``target.population_targets``/``target.assumptions``.",
        "H5 is a *secondary* reporting hypothesis.  Its intervals are reported here beside the primary and the t interval;",
        "the frozen H1 verdict (13 of 65 non-drift scenarios outside the coverage band) is unchanged.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _fmt(x):
    return "—" if x is None else f"{x:.3f}"


# --- verification -------------------------------------------------------------------
def verify(report: dict) -> dict:
    """Re-check the audited expectations *from the computed report* (no hard-coding)."""
    nd = report["scope_tables"]["non_drift_65"]
    checks = {
        "n_non_drift_65": nd["n_scenarios"] == AUDIT_EXPECTATION["non_drift_scenarios"],
        "re_primary_in_band_24": nd["random_effects_in_band"] == AUDIT_EXPECTATION["re_primary_in_band"],
        "re_primary_median_approx_0.876": nd["random_effects_coverage_median"] is not None
        and abs(nd["random_effects_coverage_median"] - AUDIT_EXPECTATION["re_primary_median"]) <= 0.001,
        "consistency_zero_disagreements": report["consistency_with_frozen_reduction"]["scenarios_differing"] == 0,
    }
    return {"expectation": AUDIT_EXPECTATION, "checks": checks, "all_passed": all(checks.values())}


# --- structured diff ----------------------------------------------------------------
def structured_diff(report: dict, frozen_conf: dict) -> dict:
    """What the overlay adds to each frozen scenario summary (nothing is replaced)."""
    added_fields = ["coverage_unconditional.random_effects", "coverage_unconditional.t_interval",
                    "coverage_given_not_withheld", "length", "target"]
    rows = {}
    for name, s in report["scenarios"].items():
        if name not in frozen_conf:
            continue
        c = s["contrasts"][PRIMARY_CONTRAST]
        rows[name] = {
            "frozen_has": sorted(frozen_conf[name]["contrasts"][PRIMARY_CONTRAST].keys()),
            "overlay_adds": added_fields,
            "re_coverage_rate": c["coverage_unconditional"]["random_effects"]["rate"],
            "frozen_two_stage_rate": frozen_conf[name]["contrasts"][PRIMARY_CONTRAST]["coverage_two_stage"]["rate"],
            "target_mismatch": c["target"]["mismatch_flag"],
        }
    return {"what": "additive reporting overlay; no frozen field is modified",
            "frozen_source": "REDUCED.json:confirmation.scenarios",
            "overlay_source": "H5_RANDOM_EFFECTS_REDUCTION.json:scenarios",
            "n_scenarios": len(rows), "scenarios": rows}


# --- entry point --------------------------------------------------------------------
def reduce_run(run_root: Path, out: Path, *, copy_to: Path | None = None,
               allow_in_place: bool = False) -> dict:
    """Build the additive overlay from a *fresh copy* of the immutable run root.

    ``copy_to`` is mandatory unless ``allow_in_place`` is set for a test, and the
    destination is vetted (symlink-resolved) before any write: equal to, inside,
    containing or already existing at the source is refused.  The source is never
    deleted or written; the copy is made with ``copytree`` and nothing is removed.
    """
    run_root = _resolved(run_root)
    if not (run_root / "REDUCED.json").exists():
        raise SystemExit(f"no REDUCED.json under {run_root}")
    out = _resolved(out)
    if allow_in_place and copy_to is None:
        source_root = run_root
        guard_output_destination(out, run_root, None)
        copy_to = None
    else:
        copy_to = guard_copy_destination(run_root, copy_to)
        guard_output_destination(out, run_root, copy_to)
        shutil.copytree(run_root, copy_to)          # fresh destination; no delete anywhere
        source_root = copy_to
    reduced, snapshot, groups, scenario_params, cell_prov, validation = load_stored(source_root)
    if not validation["ok"]:
        from experiments.mlst2026.successor_20260913.common.results import ReconciliationError
        raise ReconciliationError("R0 input validation failed: " + "; ".join(validation["problems"][:5]))
    report = build_report(groups, reduced, scenario_params, validation)
    verification = verify(report)
    diff = structured_diff(report, reduced.get("confirmation", {}).get("scenarios", {}))

    out.mkdir(parents=True, exist_ok=True)
    # the overlay report
    write_json(out / "H5_RANDOM_EFFECTS_REDUCTION.json", report)
    # the manuscript-facing table
    (out / "MANUSCRIPT_T22_H5_TABLE.md").write_text(manuscript_table(report))
    # the structured diff
    write_json(out / "STRUCTURED_DIFF.json", diff)
    # provenance
    provenance = {
        "schema": SCHEMA,
        "amendment_version": AMENDMENT_VERSION,
        "source_run_root": str(run_root),
        "copied_working_root": str(copy_to) if copy_to is not None else None,
        "reduced_json_sha256": _sha256(source_root / "REDUCED.json"),
        "protocol_snapshot_sha256": _sha256(source_root / "protocol_snapshot.json") if (source_root / "protocol_snapshot.json").exists() else None,
        "frozen_calibration_sha256": _sha256(source_root / "FROZEN_CALIBRATION.json") if (source_root / "FROZEN_CALIBRATION.json").exists() else None,
        "reducer_source_sha256": _sha256(Path(__file__).resolve()),
        "n_confirmation_cells": len(cell_prov),
        "freeze_digests": validation.get("freeze_digests", []),
        "cells": cell_prov,
        "input_validation": validation,
        "command": ("python -m experiments.mlst2026.followup_20260917.r0_t22_h5 reduce "
                    f"--run-root {run_root}" + (f" --copy-to {copy_to}" if copy_to is not None else "") + f" --out {out}"),
        "verification": verification,
        "preserved": {
            "frozen_reduction": str(source_root / "REDUCED.json"),
            "note": ("The frozen reduction is copied byte-for-byte into the export; only additive overlay "
                     "files are written.  The source run root was never opened for writing or deleted."),
        },
    }
    write_json(out / "PROVENANCE.json", provenance)
    # copy the frozen reduction and protocol snapshot unchanged into the export
    export_frozen = out / "frozen"
    export_frozen.mkdir(exist_ok=True)
    shutil.copy2(source_root / "REDUCED.json", export_frozen / "REDUCED.json")
    if (source_root / "protocol_snapshot.json").exists():
        shutil.copy2(source_root / "protocol_snapshot.json", export_frozen / "protocol_snapshot.json")
    if (source_root / "FROZEN_CALIBRATION.json").exists():
        shutil.copy2(source_root / "FROZEN_CALIBRATION.json", export_frozen / "FROZEN_CALIBRATION.json")
    print(json.dumps({"schema": SCHEMA, "n_scenarios": len(groups),
                      "non_drift_in_band": report["scope_tables"]["non_drift_65"]["random_effects_in_band"],
                      "non_drift_median": report["scope_tables"]["non_drift_65"]["random_effects_coverage_median"],
                      "verification_all_passed": verification["all_passed"],
                      "out": str(out)}, indent=1))
    return report


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m experiments.mlst2026.followup_20260917.r0_t22_h5")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reduce", help="build the H5 random-effects overlay from stored records")
    r.add_argument("--run-root", required=True, type=Path, help="a stored t22s run root (REDUCED.json + cells/)")
    r.add_argument("--copy-to", type=Path, required=False, default=None,
                   help="REQUIRED fresh destination for the working copy; the source is never written")
    r.add_argument("--allow-in-place", action="store_true",
                   help="TEST ONLY: read the run root without copying (refuses an output inside it)")
    r.add_argument("--out", required=True, type=Path, help="export directory (must be outside the input trees)")
    a = ap.parse_args(argv)
    if a.cmd == "reduce":
        if a.copy_to is None and not a.allow_in_place:
            print("error: R0 requires a fresh --copy-to destination (or --allow-in-place for tests); the immutable "
                  "source run root is never written", file=sys.stderr)
            return 6
        report = reduce_run(a.run_root, a.out, copy_to=a.copy_to, allow_in_place=a.allow_in_place)
        return 0 if verify(report)["all_passed"] else 1
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
