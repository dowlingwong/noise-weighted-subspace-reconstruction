"""R1 runner: T22-S v3 small-file interval calibration (new protocol, fresh seeds).

Reuses the archived T22-S simulator and inference pipeline unchanged
(``experiments.mlst2026.successor_20260913.t22s``) and adds the frozen candidate
interval family from :mod:`candidates`.  The archived two-stage procedure stays in
the record as the named baseline.

Flow (as the framework requires):

    pilot cells  ->  candidate adequacy on development/pilot seeds  ->  freeze at
    most one primary replacement  ->  confirmation cells scored only against the
    frozen candidate.

A cell refuses to run a confirmation-type scenario without the freeze; the freeze
digest is checked at reduce.  This module is a *new protocol version*: it never
edits, re-runs or re-scores T22-S v2.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from experiments.mlst2026.successor_20260913.common import cli  # noqa: E402
from experiments.mlst2026.successor_20260913.common.matrix import RunMatrix, make_cell  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import (ReconciliationError, SchemaError,  # noqa: E402
                                                                     write_json_atomic)
from experiments.mlst2026.successor_20260913.t22s import pipeline as pl  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import simulator as sim  # noqa: E402

from . import candidates as cand  # noqa: E402

PACKAGE = "r1_t22_calibration"
#: v2 carries the exact archived baseline, the primary-eligibility split, the
#: effective-df fix and the unified type-I predicate.  v1 is retained unchanged.
PROTOCOL_DEFAULT = _HERE.parent / "PROTOCOL_R1_T22_CALIBRATION_v2.json"
FROZEN_NAME = "FROZEN_CANDIDATE.json"
FROZEN_SCHEMA = "mlst-r1-frozen-candidate/1"
SOURCE_FILES = [_HERE, _HERE.parent / "candidates.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t22s" / "pipeline.py",
                _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t22s" / "simulator.py",
                _ROOT / "experiments" / "mlst2026" / "t22_tidmad_narrowband.py"]
FAMILIES = ("sim_pilot", "sim_confirm", "control")
result_required = {"kind": str, "passed": bool, "checks": dict, "summary": dict}
PRIMARY = "raw-bin|empca"
NULL_CLAIM_KEYS = {name: keys for name, keys in pl.CLAIMS.items()}


# --- statistics ---------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> dict:
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "half_width": None, "n": 0, "k": int(k)}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": float(p), "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "half_width": float(half), "n": int(n), "k": int(k)}


def _within(w: dict, band) -> bool | None:
    if w is None or w["n"] == 0:
        return None
    lo, hi = band
    return not (w["hi"] < lo or w["lo"] > hi)


# --- matrix --------------------------------------------------------------------------
def _scenario_cells(scenarios, *, role, family, seeds, mc, chunks) -> list[dict]:
    cells = []
    for sc in scenarios:
        for chunk in range(chunks):
            params = {"scenario": sc.as_dict(), "mc_replicates": int(mc), "chunk": chunk}
            cells.append(make_cell(role=role, seed=int(seeds[chunk]), params=params, n_fits=0, family=family))
    return cells


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    M = P["matrix"]
    prod = M["production"]
    _check_catalogue(P)
    cells = []
    pilot_lo = P["seed_namespaces"]["pilot"][0]
    conf_lo = P["seed_namespaces"]["confirmation"][0]
    n_p = prod["pilot_mc_per_scenario"] // prod["mc_per_cell"]
    n_c = prod["mc_replicates_per_scenario"] // prod["mc_per_cell"]
    cells += _scenario_cells(sim.pilot_scenarios(), role="pilot", family="sim_pilot",
                             seeds=[pilot_lo + i for i in range(n_p)], mc=prod["mc_per_cell"], chunks=n_p)
    cells += _scenario_cells(sim.confirmation_scenarios(), role="confirmation", family="sim_confirm",
                             seeds=[conf_lo + i for i in range(n_c)], mc=prod["mc_per_cell"], chunks=n_c)
    for name, cfg in M["production"]["controls"].items():
        cells.append(make_cell(role="control", seed=None, params={"control": name, **cfg}, n_fits=0, family="control"))
    if fixture:
        fx = M["fixture"]
        dev_lo = P["seed_namespaces"]["development"][0]
        cells += _scenario_cells(sim.pilot_scenarios()[: fx["n_pilot_scenarios"]], role="development", family="sim_pilot",
                                 seeds=[dev_lo + i for i in range(fx["chunks"])], mc=fx["mc_per_cell"], chunks=fx["chunks"])
        cells += _scenario_cells(sim.development_scenarios(), role="development", family="sim_confirm",
                                 seeds=[dev_lo + 100 + i for i in range(fx["chunks"])], mc=fx["mc_per_cell"], chunks=fx["chunks"])
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def _check_catalogue(P: dict) -> None:
    decl = P["matrix"]["scenario_catalogue"]
    got = {"pilot": [s.name for s in sim.pilot_scenarios()],
           "confirmation": [s.name for s in sim.confirmation_scenarios()],
           "development": [s.name for s in sim.development_scenarios()]}
    for k, names in got.items():
        if decl.get(k) != names:
            raise ProtocolError(f"R1 scenario catalogue {k!r} differs from the code's catalogue")
    if tuple(P["candidate_family"]) != cand.CANDIDATES:
        raise ProtocolError("protocol candidate_family differs from the code's frozen candidate set")
    if tuple(P["inference"]["block_grid_evaluation"]) != pl.BLOCKS_EVAL:
        raise ProtocolError("R1 must reuse the archived block grids")


# --- cells ---------------------------------------------------------------------------
def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    if cell["family"] in ("sim_pilot", "sim_confirm"):
        return _run_sim_cell(P, cell, fixture, ctx), 0, 0
    if cell["family"] == "control":
        return _run_control_cell(P, cell, fixture, ctx), 0, 0
    raise RuntimeError(f"unknown family {cell['family']!r}")


def _augment(rec: dict, reps: dict) -> dict:
    out = {"candidates": {}}
    for key in rec["contrasts"]:
        c = rec["contrasts"][key]
        pf = cand.PerFile(y=np.array([v["mean"] for v in c["per_file"].values()]),
                          s=np.array([v["bootstrap_sd"] for v in c["per_file"].values()]),
                          n_bootstrap=int(reps["file"]))
        # carry the STORED archived endpoints and bootstrap p through exactly:
        # the baseline is not rebuilt from point +/- z*SE.
        ivs = cand.candidate_intervals(pf, c["point"], c["bootstrap_sd"],
                                       two_stage_lo=c.get("ci95_lo"), two_stage_hi=c.get("ci95_hi"),
                                       two_stage_p=c.get("p_two_sided"))
        tm = c.get("true_mu")
        out["candidates"][key] = {
            name: {**ivs[name],
                   "covers": (bool(ivs[name]["lo"] <= tm <= ivs[name]["hi"]) if tm is not None else None),
                   "p_coherent": cand.coherent_pvalue(name, ivs[name])}
            for name in cand.CANDIDATES}
    return out


def _run_sim_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    params = cell["params"]
    sc = sim.Scenario(**params["scenario"])
    frozen_sha = ""
    if cell["family"] == "sim_confirm":
        fz, frozen_sha = _read_frozen(ctx["out"], P, fixture=fixture)
        chosen = fz["chosen_candidate"]
    else:
        chosen = None
    reps = dict(P["matrix"]["fixture" if fixture else "production"]["replicates"])
    records = []
    for j in range(int(params["mc_replicates"])):
        seed = int(cell["seed"]) * 10_000 + j
        rec = pl.run_replicate(seed, sc, file_replicates=reps["file"], hier_replicates=reps["hier"],
                               curve_replicates=reps["curve"])
        rec.update(_augment(rec, reps))
        rec["problems"] = pl.check_record(rec, sc)
        records.append(rec)
    summary = {"scenario_name": sc.name, "n_mc": len(records), "chosen_candidate": chosen,
               "frozen_sha256": frozen_sha, "replicates": reps}
    problems = [p for r in records for p in r["problems"]]
    return {"kind": "simulation", "passed": not problems,
            "checks": {"invariant_violations": len(problems), "first": problems[:3]},
            "summary": summary, "records": records}


def _run_control_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    name = cell["params"]["control"]
    rng = np.random.default_rng(1717)
    checks = {}
    if name == "t_equal_file_matches_archived_t":
        sc = sim.make_scenario("alt-medium", 8, 0.3, tau=0.01)
        rec = pl.run_replicate(9_000_123, sc, file_replicates=120, hier_replicates=400, curve_replicates=60)
        rec.update(_augment(rec, {"file": 120}))
        c = rec["contrasts"][PRIMARY]
        iv = rec["candidates"][PRIMARY]["t_equal_file"]
        y = np.array([v["mean"] for v in c["per_file"].values()])
        arch = t22.t_interval(y)
        # The archived t interval uses a 3-decimal lookup table (T975); the candidate
        # uses the exact scipy quantile.  The check is that the pivot is the same and
        # the difference is confined to the quantile rounding, not a different formula.
        d_lo = abs(arch["ci95"][0] - iv["lo"]); d_hi = abs(arch["ci95"][1] - iv["hi"])
        scale = max(1e-30, abs(arch["ci95"][1] - arch["ci95"][0]))
        checks["point"] = {"archived": arch.get("mean"), "candidate": iv["point"],
                           "passed": abs(arch.get("mean", 0.0) - iv["point"]) <= 1e-12}
        checks["interval"] = {"archived": arch.get("ci95"), "candidate": [iv["lo"], iv["hi"]],
                              "abs_diff": [d_lo, d_hi], "relative_to_width": [d_lo / scale, d_hi / scale],
                              "archived_critical": arch.get("t975"), "candidate_critical": iv["critical"],
                              "note": "difference is the archived 3-decimal T975 lookup vs the exact scipy quantile",
                              "passed": max(d_lo, d_hi) <= 1e-4 * scale}
    elif name == "coherent_pvalue_matches_pivot":
        interval = {"point": 1.0, "se": 1.0, "dof": 7}
        p = cand.coherent_pvalue("t_equal_file", interval)
        from scipy import stats as st
        expect = 2.0 * (1.0 - st.t.cdf(1.0, 7))
        checks["p"] = {"got": p, "expected": float(expect), "passed": abs(p - expect) <= 1e-12}
    elif name == "interval_ordering":
        intervals = cand.candidate_intervals(cand.PerFile(y=np.array([1.0, 1.1, 0.9, 1.05]),
                                                          s=np.array([0.2, 0.3, 0.25, 0.22]), n_bootstrap=200),
                                             two_stage_point=1.0, two_stage_se=0.1)
        checks["ordering"] = {"passed": all(v["lo"] <= v["point"] <= v["hi"] for v in intervals.values()),
                              "detail": {k: [v["lo"], v["point"], v["hi"]] for k, v in intervals.items()}}
    elif name == "targets_documented":
        checks["docs"] = {"passed": cand.all_targets_documented(), "targets": cand.TARGET, "assumptions": cand.ASSUMPTIONS}
    else:
        raise RuntimeError(f"unknown control {name!r}")
    return {"kind": "control", "passed": all(c["passed"] for c in checks.values()), "checks": checks,
            "summary": {"control": name}}


# --- frozen candidate -----------------------------------------------------------------
def frozen_path(out: Path) -> Path:
    return Path(out) / FROZEN_NAME


def _read_frozen(out: Path, P: dict, *, fixture: bool) -> tuple[dict, str]:
    p = frozen_path(out)
    if not p.exists():
        raise RuntimeError(f"validation refused: {p} does not exist; run and reduce the pilot cells first")
    raw = p.read_bytes()
    fz = json.loads(raw)
    if fz.get("schema") != FROZEN_SCHEMA:
        raise RuntimeError(f"{p}: unexpected schema {fz.get('schema')!r}")
    if fz.get("protocol_sha256") != P["_sha256"]:
        raise RuntimeError(f"{p} was frozen under protocol {fz.get('protocol_sha256')}, not {P['_sha256']}")
    if bool(fz.get("development_only")) != bool(fixture):
        raise RuntimeError(f"{p} development_only does not match this run")
    # full pilot-content binding: the freeze must name the candidate family, the
    # eligibility split, the tolerances, the rule and the pilot cells it came from.
    if tuple(fz.get("candidate_family", ())) != cand.CANDIDATES:
        raise RuntimeError(f"{p} candidate_family differs from the code's frozen candidate set")
    if tuple(fz.get("primary_eligible", ())) != cand.PRIMARY_ELIGIBLE:
        raise RuntimeError(f"{p} primary_eligible differs from the code")
    if tuple(fz.get("secondary_diagnostic", ())) != cand.SECONDARY_DIAGNOSTIC:
        raise RuntimeError(f"{p} secondary_diagnostic differs from the code")
    if not fz.get("frozen_from_pilot_cells"):
        raise RuntimeError(f"{p} names no pilot cells; the freeze is not bound to pilot content")
    if fz.get("adequacy_rule") != cand.selection_rule_doc():
        raise RuntimeError(f"{p} adequacy rule differs from the code's current rule")
    return fz, hashlib.sha256(raw).hexdigest()


# --- reduce ---------------------------------------------------------------------------
def _pool(cells_in_family):
    groups = {}
    for js in cells_in_family:
        key = js["result"]["summary"]["scenario_name"]
        g = groups.setdefault(key, {"records": [], "cells": [], "scenario": sim.Scenario(**js["params"]["scenario"])})
        g["records"].extend(js["result"]["records"])
        g["cells"].append(js["cell_id"])
    return groups


def _claim_decisions(rec: dict, candidate: str) -> dict:
    """The decision family, with withholding enforced.

    A withheld replicate can never be 'supported': the dependence gate is a
    precondition of the whole decision, not a side note.  Returning True for a
    withheld record would break the unconditional-error and bound interpretation.
    """
    if bool(rec.get("withheld")):
        return {name: False for name in NULL_CLAIM_KEYS}
    p_raw = {}
    for name, keys in NULL_CLAIM_KEYS.items():
        ps = [rec["candidates"][k][candidate]["p_coherent"] for k in keys]
        p_raw[name] = float(max(ps))
    adj = t22.holm(p_raw)
    out = {}
    for name, keys in NULL_CLAIM_KEYS.items():
        intervals = [rec["candidates"][k][candidate] for k in keys]
        points = [rec["candidates"][k][candidate]["point"] for k in keys]
        favours = all(v > 0 for v in points)
        excludes = all(v["lo"] > 0 or v["hi"] < 0 for v in intervals)
        out[name] = bool(favours and excludes and adj[name] < 0.05)
    return out


def summarise_candidate(sc: sim.Scenario, records: list[dict], candidate: str, tol: dict) -> dict:
    n = len(records)
    nw = [r for r in records if not r["withheld"]]
    cov = {k: wilson(sum(r["candidates"][k][candidate]["covers"] for r in records), n) for k in sim.EXACT_MU_CONTRASTS}
    cov_nw = {k: wilson(sum(r["candidates"][k][candidate]["covers"] for r in nw), len(nw)) for k in sim.EXACT_MU_CONTRASTS}
    lengths = [r["candidates"][PRIMARY][candidate]["hi"] - r["candidates"][PRIMARY][candidate]["lo"] for r in records]
    mu = sc.true_mu()
    null_claims = {name: all(mu.get(k, mu["zero|empca"]) == 0.0 for k in keys) for name, keys in NULL_CLAIM_KEYS.items()}
    decisions = [_claim_decisions(r, candidate) for r in records]
    types = {}
    for name in NULL_CLAIM_KEYS:
        if null_claims[name]:
            types[name] = wilson(sum(d[name] for d in decisions), n)
    any_null = [any(d[c] for c in NULL_CLAIM_KEYS if null_claims[c]) for d in decisions]
    # per-scenario estimator diagnostics, with Monte-Carlo uncertainty
    pts = np.array([r["candidates"][PRIMARY][candidate]["point"] for r in records], dtype=float)
    ses = np.array([r["candidates"][PRIMARY][candidate].get("se", np.nan) for r in records], dtype=float)
    tm = mu.get(PRIMARY)
    biases = (pts - tm) if tm is not None else np.full(n, np.nan)
    claimed = {}
    for name in NULL_CLAIM_KEYS:
        claimed[name] = {"supported": wilson(sum(d[name] for d in decisions), n),
                         "is_null": bool(null_claims[name])}
    out = {"scenario": sc.name, "n_mc": n, "candidate": candidate, "n_files": sc.n_files,
           "is_drift": sc.drift_sd > 0, "true_mu": mu,
           "coverage_primary": cov[PRIMARY], "coverage_primary_given_not_withheld": cov_nw[PRIMARY],
           "coverage_all_contrasts": cov, "mean_length_primary": float(np.mean(lengths)),
           "coverage_within_tolerance": _within(cov[PRIMARY], tol["coverage"]),
           "estimator_summary": {
               "point_mean": float(np.mean(pts)) if n else None,
               "bias_mean": float(np.nanmean(biases)) if n and np.any(np.isfinite(biases)) else None,
               "bias_mc_se": (float(np.nanstd(biases, ddof=1) / np.sqrt(np.sum(np.isfinite(biases))))
                              if np.sum(np.isfinite(biases)) > 1 else None),
               "empirical_sd_of_point": float(np.std(pts, ddof=1)) if n > 1 else None,
               "reported_se_mean": float(np.nanmean(ses)) if n else None,
               "width_mean": float(np.mean(lengths)) if n else None,
               "width_sd": float(np.std(lengths, ddof=1)) if n > 1 else None,
               "unit": "readout physical-amplitude squared contrast"},
           "claims": claimed,
           "type1_per_claim": types,
           "any_null_claim_supported": (wilson(sum(any_null), n) if any(null_claims.values()) else None),
           "withholding": wilson(n - len(nw), n)}
    if out["any_null_claim_supported"] is not None:
        a = out["any_null_claim_supported"]
        out["type1_within_tolerance"] = cand.type1_accepts_upper_bound(a, tol["type1"][1])
        # the archived overlap rule, recorded separately and never used to gate selection
        out["type1_historical_overlap"] = cand.historical_type1_overlap(a, tol["type1"][1])
    return out


def _adequate(rows: list[dict], tol: dict, expected_non_drift: int | None = None) -> bool:
    """The predeclared all-scenario adequacy rule (non-drift pilot scenarios).

    Coverage must be within tolerance on *every* non-drift scenario, and the
    any-null-claim error must satisfy the unified upper-Wilson predicate on every
    null scenario.  A candidate with a target other than the equal-file mean is
    never adequate by this rule (``PRIMARY_ELIGIBLE`` gates the caller).
    """
    core = [r for r in rows if not r["is_drift"]]
    if not core or (expected_non_drift is not None and len(core) != expected_non_drift):
        return False
    if not all(r["coverage_within_tolerance"] is True for r in core):
        return False
    if any(r.get("type1_within_tolerance") is False for r in core):
        return False
    return True


def _freeze(P: dict, groups: dict, out: Path, fixture: bool, matrix_sha256: str = "") -> dict:
    tol = P["calibration_tolerances"]
    expected_non_drift = len([s for s in sim.pilot_scenarios() if s.drift_sd == 0])
    per_cand = {}
    for name in cand.CANDIDATES:
        rows = [summarise_candidate(g["scenario"], g["records"], name, tol) for g in groups.values()]
        # rows repeat per candidate per scenario; deduplicate by scenario name
        rows = list({r["scenario"]: r for r in rows}.values())
        per_cand[name] = {"primary_eligible": name in cand.PRIMARY_ELIGIBLE,
                          "adequate": (name in cand.PRIMARY_ELIGIBLE and _adequate(rows, tol, expected_non_drift)),
                          "median_length_primary": float(np.median([r["mean_length_primary"] for r in rows])),
                          "median_coverage_primary": float(np.median([r["coverage_primary"]["rate"] for r in rows])),
                          "non_drift_in_band": int(sum(r["coverage_within_tolerance"] is True for r in rows if not r["is_drift"])),
                          "expected_non_drift_pilot_scenarios": expected_non_drift}
    adequate = [n for n in cand.PRIMARY_ELIGIBLE if per_cand[n]["adequate"]]
    if adequate:
        chosen = min(adequate, key=lambda n: (per_cand[n]["median_length_primary"], cand.CANDIDATES.index(n)))
        reason = "adequate on every non-drift pilot scenario; shortest median primary length"
    else:
        chosen = "none"
        reason = "no primary-eligible candidate met the predeclared all-scenario adequacy rule; the calibration is reported as a failure"
    fz = {"schema": FROZEN_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"],
          "matrix_sha256": matrix_sha256,
          "development_only": bool(fixture), "chosen_candidate": chosen, "reason": reason,
          "candidate_family": list(cand.CANDIDATES), "primary_eligible": list(cand.PRIMARY_ELIGIBLE),
          "secondary_diagnostic": list(cand.SECONDARY_DIAGNOSTIC),
          "adequacy_rule": cand.selection_rule_doc(),
          "tolerances": tol, "candidate_summary": per_cand,
          "frozen_from_pilot_cells": sorted(c for g in groups.values() for c in g["cells"]),
          "statement": ("Candidate frozen before any confirmation cell is drawn.  A development_only freeze is "
                        "never citable.")}
    p = frozen_path(out)
    if p.exists():
        prev = json.loads(p.read_text())
        if prev != json.loads(json.dumps(fz, default=_json_default)):
            raise ReconciliationError(f"{p} exists and differs from the deterministic freeze; refusing to overwrite")
    else:
        write_json_atomic(p, fz)
    return fz


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _acceptance_from_confirmation(conf: dict, fixture: bool, *, protocol_sha256: str = "",
                                  matrix_sha256: str = "") -> dict:
    """The R1 acceptance receipt consumed by R4's cancellation gate (A2).

    Every field R4 reads is an explicit Boolean or an explicit identity; there is
    no field whose absence could be read as a pass.  A chosen candidate other than
    ``"none"`` that is not primary-eligible is refused (it cannot replace the
    equal-file primary).
    """
    chosen = conf.get("chosen_candidate")
    if chosen in (None, "none"):
        return {"schema": "mlst-followup-r1-acceptance/2", "frozen_candidate": chosen,
                "candidate_primary_eligible": False,
                "coverage_ok": {}, "type1_ok": {}, "withholding_screen_all_infeasible": chosen == "none",
                "development_only": bool(fixture), "confirmation_complete": bool(conf.get("complete")),
                "protocol_sha256": protocol_sha256, "matrix_sha256": matrix_sha256,
                "decisions_explicit": True,
                "note": "no usable interval frozen; every large-file design is cancelled"}
    if chosen not in cand.PRIMARY_ELIGIBLE:
        raise ReconciliationError(f"frozen candidate {chosen!r} is not primary-eligible; it cannot be accepted")
    by_n = conf["candidates"][chosen].get("by_n_files", {})
    cov = {n: v["coverage_ok"] for n, v in by_n.items()}
    t1 = {n: v["type1_ok"] for n, v in by_n.items()}
    if not all(isinstance(v, bool) for v in list(cov.values()) + list(t1.values())):
        raise ReconciliationError("acceptance decisions are not all explicit Booleans")
    return {"schema": "mlst-followup-r1-acceptance/2", "frozen_candidate": chosen,
            "candidate_primary_eligible": True,
            "coverage_ok": cov, "type1_ok": t1,
            "withholding_screen_all_infeasible": False,
            "development_only": bool(fixture), "confirmation_complete": bool(conf.get("complete")),
            "protocol_sha256": protocol_sha256, "matrix_sha256": matrix_sha256,
            "decisions_explicit": True,
            "note": "acceptance is per candidate file count; any False cancels R4's large-file cells"}


def acceptance_command(argv) -> int:
    """Write R1_ACCEPTANCE.json from a reduced R1 run root (A2)."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m experiments.mlst2026.followup_20260917.r1_t22_calibration acceptance")
    ap.add_argument("--run-root", required=True, type=Path)
    ap.add_argument("--out-file", type=Path, default=None)
    a = ap.parse_args(argv)
    red = json.loads((a.run_root / "REDUCED.json").read_text())
    conf = red.get("confirmation", {})
    if not conf.get("complete") or red.get("partial"):
        raise SystemExit("refusing: the confirmation reduction is incomplete or partial; no acceptance receipt "
                         "can be issued from it")
    acc = conf.get("acceptance")
    if acc is None:
        raise SystemExit("no confirmation acceptance in REDUCED.json; run the complete confirmation reduce first")
    dest = a.out_file or (a.run_root / "R1_ACCEPTANCE.json")
    dest.write_text(json.dumps(acc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"frozen_candidate": acc["frozen_candidate"], "coverage_ok": acc["coverage_ok"],
                      "type1_ok": acc["type1_ok"], "development_only": acc["development_only"],
                      "out": str(dest)}, indent=1))
    return 0


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    tol = P["calibration_tolerances"]
    by_family = {f: [js for js in cells.values() if js["family"] == f] for f in FAMILIES}
    violations = []
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "simulation":
            sc = sim.Scenario(**js["params"]["scenario"])
            for rec in r["records"]:
                probs = pl.check_record(rec, sc)
                if probs:
                    violations.append({"cell_id": js["cell_id"], "problems": probs[:3]})
        elif not r["passed"]:
            violations.append({"cell_id": js["cell_id"], "problems": ["control failed"]})
    if violations:
        raise SchemaError("scientific check failed: " + str(violations[:3]))
    summary = {"controls": {js["result"]["summary"]["control"]: {"passed": js["result"]["passed"], "checks": js["result"]["checks"]}
                            for js in by_family["control"]},
               "candidate_family": list(cand.CANDIDATES), "tolerances": tol,
               "targets": cand.TARGET, "assumptions": cand.ASSUMPTIONS, "selection_rule": cand.selection_rule_doc()}
    pilot_expected = [c for c in matrix.cells if c["family"] == "sim_pilot" and c["role"] == ("development" if fixture else "pilot")]
    pilot_groups = _pool(by_family["sim_pilot"])
    have = {js["cell_id"] for js in by_family["sim_pilot"]}
    pilot_complete = bool(pilot_expected) and all(c["cell_id"] in have for c in pilot_expected)
    summary["pilot"] = {"complete": pilot_complete, "n_cells": len(by_family["sim_pilot"])}
    if pilot_complete:
        summary["frozen_candidate"] = _freeze(P, pilot_groups, out, fixture, matrix_sha256=matrix.matrix_hash())
    else:
        summary["frozen_candidate"] = None
    conf_expected = [c for c in matrix.cells if c["family"] == "sim_confirm" and c["role"] == ("development" if fixture else "confirmation")]
    conf_cells = by_family["sim_confirm"]
    have = {js["cell_id"] for js in conf_cells}
    conf_complete = bool(conf_expected) and all(c["cell_id"] in have for c in conf_expected)
    conf = {"complete": conf_complete, "n_cells": len(conf_cells), "candidates": {}, "scope_tables": {}}
    if conf_cells:
        fz, sha = _read_frozen(out, P, fixture=fixture)
        if fz.get("matrix_sha256") and fz["matrix_sha256"] != matrix.matrix_hash():
            raise ReconciliationError(
                f"the freeze was made under matrix {fz['matrix_sha256']}, this run has {matrix.matrix_hash()}")
        bad = [js["cell_id"] for js in conf_cells if js["result"]["summary"].get("frozen_sha256") != sha]
        if bad:
            raise ReconciliationError(f"{len(bad)} confirmation cells were scored under a different frozen candidate")
        conf["frozen_sha256"] = sha
        conf["chosen_candidate"] = fz["chosen_candidate"]
        groups = _pool(conf_cells)
        for name in cand.CANDIDATES:
            rows = [summarise_candidate(g["scenario"], g["records"], name, tol) for g in groups.values()]
            by_n = {}
            for r in rows:
                if r["is_drift"]:
                    continue
                by_n.setdefault(str(r["n_files"]), []).append(r)
            conf["candidates"][name] = {
                "non_drift_65_in_band": int(sum(r["coverage_within_tolerance"] is True for r in rows if not r["is_drift"])),
                "non_drift_65_median_coverage": float(np.median([r["coverage_primary"]["rate"] for r in rows if not r["is_drift"]])),
                "drift_4_in_band": int(sum(r["coverage_within_tolerance"] is True for r in rows if r["is_drift"])),
                "median_length": float(np.median([r["mean_length_primary"] for r in rows])),
                "any_null_claim_type1_max": max([r["any_null_claim_supported"]["rate"] for r in rows
                                                 if r["any_null_claim_supported"] is not None], default=None),
                "by_n_files": {n: {
                    "n_scenarios": len(rs),
                    "coverage_ok": bool(all(r["coverage_within_tolerance"] is True for r in rs)),
                    "type1_ok": bool(all(r.get("type1_within_tolerance") is not False for r in rs)),
                    "coverage_median": float(np.median([r["coverage_primary"]["rate"] for r in rs])),
                } for n, rs in sorted(by_n.items())}}
        conf["scope_tables"]["non_drift_65"] = {"candidate_rows": {n: conf["candidates"][n] for n in cand.CANDIDATES}}
        # An acceptance receipt may only be issued from a COMPLETE confirmation
        # reduction: a partial run cannot certify that an interval is accepted.
        if conf_complete and not partial:
            conf["acceptance"] = _acceptance_from_confirmation(conf, fixture, protocol_sha256=P["_sha256"],
                                                               matrix_sha256=matrix.matrix_hash())
        else:
            conf["acceptance"] = None
            conf["acceptance_refused"] = ("confirmation reduction is incomplete (or partial); no acceptance "
                                          "receipt is issued")
    summary["confirmation"] = conf
    summary["statement"] = (
        "T22 interval recalibration on fresh seeds with the frozen candidate family.  It measures whether a "
        "prespecified small-file interval achieves the archived coverage band and keeps the type-I rule coherent.  "
        "The archived T22 verdicts remain withheld and unchanged; no result here transfers to unexamined instrument "
        "data." + ("  DEVELOPMENT ONLY: fixture output, never citable." if fixture else ""))
    return summary


# --- entry point ---------------------------------------------------------------------
def _precheck_frozen(argv) -> str | None:
    """Refuse a production run that could reach confirmation cells without a freeze."""
    if not argv or argv[0] != "run":
        return None
    out, fams = None, None
    i = 0
    while i < len(argv):
        if argv[i] == "--out" and i + 1 < len(argv):
            out = Path(argv[i + 1])
        if argv[i] == "--families":
            fams = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                fams.append(argv[j]); j += 1
        i += 1
    if out is None or frozen_path(out).exists():
        return None
    if fams is not None and "sim_confirm" not in fams:
        return None
    return (f"validation refused: {frozen_path(out)} does not exist.  Run the pilot first, then reduce, which "
            f"freezes the candidate; confirmation cells are scored only against it.")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "acceptance":
        return acceptance_command(argv[1:])
    msg = _precheck_frozen(argv)
    if msg:
        print("error: " + msg, file=sys.stderr)
        return 6
    return cli.main_template(package=PACKAGE,
                             description="R1: T22 small-file interval calibration (new protocol; frozen candidates)",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                             result_required=result_required, source_files=SOURCE_FILES,
                             fixture_overrides=lambda P: {"scale": "fixture", **P["matrix"]["fixture"]}, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
