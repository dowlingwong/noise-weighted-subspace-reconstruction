"""T19-S v2 runner (work package L1).

    python -m experiments.mlst2026.successor_20260913.t19s validate  --protocol PROTOCOL_T19S_v2.json
    python -m experiments.mlst2026.successor_20260913.t19s enumerate --protocol ... --out DIR
    python -m experiments.mlst2026.successor_20260913.t19s run       --protocol ... --out DIR [--fixture] [--roles ...]
    python -m experiments.mlst2026.successor_20260913.t19s reduce    --protocol ... --out DIR [--allow-partial]
    python -m experiments.mlst2026.successor_20260913.t19s estimate  --protocol ... --out DIR

Built on ``common.cli.main_template``; this module only supplies the three
callbacks (matrix, cell, reduce), the result schema and the fixture hook.
The science is in :mod:`.observables` (deterministic) and :mod:`.synthetic`
(paired fits).  Nothing runs on import.

Exit codes are the common ones; in addition ``reduce`` exits 4 when a
deterministic cell violates one of the protocol's scientific checks (a
control displacement, the 2-D closed form, the exact projector identity or a
predeclared deterministic order) — a scientific-check failure is a defect of
the package, not an outcome.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from ..common import cli
from ..common.matrix import RunMatrix, make_cell
from ..common.results import ReconciliationError, SchemaError, stable_bootstrap_ci, write_json_atomic
from . import observables as O
from . import synthetic as S

PACKAGE = "t19s"
HERE = Path(__file__).resolve().parent
SOURCE_FILES = [HERE / "runner.py", HERE / "observables.py", HERE / "synthetic.py", HERE / "PROTOCOL_T19S_v2.json",
                HERE.parents[2] / "paper1" / "t19_coupling.py", HERE.parents[2] / "paper1" / "g1.py"]
SELECTION_FILE = "PILOT_SELECTION.json"
SELECTION_SCHEMA = "t19s-pilot-selection/1"
DIRECTION_SEED_BASE = 20260913
ANALYTIC_OPERATOR_SEED = 20260913
BAND = (0.5, 2.0)

# every per-cell ``result`` block carries these keys (see SCHEMA.md)
RESULT_REQUIRED = {"kind": str, "d": int, "k": int, "model": str, "records": list, "summary": dict, "checks": dict}


class ScientificCheckError(SchemaError):
    """A deterministic cell violates a declared mathematical invariant (exit 4)."""


# =============================================================================
# matrix
# =============================================================================
def design_of(P: dict, fixture: bool) -> dict:
    return P["design"]["fixture" if fixture else "production"]


def fixture_overrides(P: dict) -> dict:
    return {"design": "fixture", **P["design"]["fixture"]}


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    D = design_of(P, fixture)
    d, k = int(D["d"]), int(D["k"])
    cells = []
    det = D["deterministic"]
    operator = {"kind": "analytic", "d": d, "seed": ANALYTIC_OPERATOR_SEED} if fixture else {"kind": "g1_population", "d": d}
    idx = 0
    # deterministic scaling checks (covariance model) and controls
    for fam, n_dir in det["families"].items():
        for _ in range(int(n_dir)):
            if fam == "two_d_example":
                params = {"kind": "deterministic", "model": "covariance", "family": fam, "d": 2, "k": 1,
                          "operator": {"kind": "two_d_example"}, "eps_grid": det["two_d_example_eps_grid"],
                          "reference": det["two_d_reference"]}
                cells.append(make_cell(role="algebraic", seed=None, params=params, n_fits=0,
                                       n_solves=len(params["eps_grid"]), family="det_two_d"))
                continue
            params = {"kind": "deterministic", "model": "covariance", "family": fam, "d": d, "k": k, "operator": operator,
                      "eps_grid": det["eps_grid"], "direction_seed": DIRECTION_SEED_BASE + idx}
            idx += 1
            role = "control" if fam in O.EXACT_FAMILIES else "algebraic"
            prefix = "ctrl" if fam in O.EXACT_FAMILIES else "det"
            cells.append(make_cell(role=role, seed=None, params=params, n_fits=0, n_solves=len(det["eps_grid"]), family=f"{prefix}_{fam}"))
    for fam, n_dir in det["additive_control_families"].items():
        for _ in range(int(n_dir)):
            params = {"kind": "deterministic", "model": "additive", "family": fam, "d": d, "k": k, "operator": operator,
                      "eps_grid": det["eps_grid"], "direction_seed": DIRECTION_SEED_BASE + idx}
            idx += 1
            cells.append(make_cell(role="control", seed=None, params=params, n_fits=0, n_solves=len(det["eps_grid"]),
                                   family=f"ctrl_additive_{fam}", notes="CONTROL model A + eps G, not the paper's model"))
    # paired synthetic fits
    pp = D["paired"]
    base = {"kind": "paired", "d": d, "k": k, "n_train": int(D["n_train"]), "eps_grid": pp["eps_grid"],
            "m_over_d_grid": pp["m_over_d_grid"], "n_train_ratio_grid": pp["n_train_ratio_grid"], "rng_offsets": pp["rng_offsets"]}
    n_fits, n_solves = S.plan_counts(base)
    if fixture:
        lo, hi = pp["development_seeds"]
        for s in range(int(lo), int(hi) + 1):
            cells.append(make_cell(role="development", seed=s, params={**base, "stage": "development"}, n_fits=n_fits, n_solves=n_solves, family="paired"))
    else:
        lo, hi = pp["pilot_seeds"]
        for s in range(int(lo), int(hi) + 1):
            cells.append(make_cell(role="pilot", seed=s, params={**base, "stage": "pilot"}, n_fits=n_fits, n_solves=n_solves, family="paired"))
        lo, hi = pp["confirmation_seeds"]
        for s in range(int(lo), int(hi) + 1):
            cells.append(make_cell(role="confirmation", seed=s, params={**base, "stage": "confirmation"}, n_fits=n_fits, n_solves=n_solves, family="paired"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


# =============================================================================
# cells
# =============================================================================
def analytic_operator(d: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lam = np.concatenate([np.array([20.0, 6.0]), np.ones(d - k)])
    assert k == 2
    rng = np.random.default_rng(seed)
    Q = O.orth(rng.standard_normal((d, d)))
    A = O.sym(Q @ np.diag(lam) @ Q.T)
    return A, lam, Q


def load_operator(params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    op = params["operator"]
    if op["kind"] == "g1_population":
        g = S.make_generator(int(op["d"]), int(params["k"]))
        return g.A_pop, g.lam, g.Q, g.label
    if op["kind"] == "analytic":
        A, lam, Q = analytic_operator(int(op["d"]), int(params["k"]), int(op["seed"]))
        return A, lam, Q, f"analytic_d{op['d']}"
    if op["kind"] == "two_d_example":
        return O.TWO_D_A.copy(), np.array([2.0, 1.0]), np.eye(2), "two_d_example"
    raise ValueError(op)


def run_deterministic_cell(params: dict) -> tuple[dict, int, int]:
    """One fixed direction, one eps path, one model.  n_solves = len(eps_grid)."""
    A, lam, Q, op_label = load_operator(params)
    d, k, fam, model = A.shape[0], int(params["k"]), params["family"], params["model"]
    if fam == "two_d_example":
        E0 = O.TWO_D_E0.copy()
    else:
        rng = np.random.default_rng(int(params["direction_seed"]))
        E0 = O.make_direction(fam, Q, k, rng, lam_b=lam)
    E0q = Q.T @ E0 @ Q
    spec = O.spectral_summary(lam, k)
    eps_op0 = float(np.linalg.norm(E0, 2))
    solve = O.MODELS[model]
    records = []
    solves = 0
    for eps in [float(e) for e in params["eps_grid"]]:
        E = eps * E0
        sol = solve(A, E, k); solves += 1
        X1, ident = O.first_order_graph(lam, eps * E0q, k, model)
        eps_op = eps * eps_op0
        eta = O.eta_of(eps_op, lam, k)
        rec = {"eps": eps, "eps_op": float(eps_op), "eta": eta, "eta_basis": "population (fixed operator)",
               "lambda_min_IplusE": sol["lambda_min_IplusE"], "valid": bool(sol["valid"]),
               "first_order_identifiable": bool(ident), "eigensolver_residual": sol["eigensolver_residual"],
               "perturbed_separation": sol["separation"], "in_domain": bool(eta <= O.ETA_IN_DOMAIN)}
        if not sol["valid"]:
            rec.update({"observables": None, "bound": None})
            records.append(rec)
            continue
        obs = O.observables(sol["V"], Q, lam, k, X1)
        rec["observables"] = obs
        rec["bound"] = O.bound_record(obs["projector_matrix"]["residual_op"], eta, applies=(model == "covariance"))
        if fam == "two_d_example":
            cf = O.two_d_closed_form(eps)
            X = O.graph_of(sol["V"], Q, k)
            rec["two_d"] = {"t_measured": float(X[0, 0]), "t_closed_form": cf["t"], "abs_error_t": float(abs(X[0, 0] - cf["t"])),
                            "residual_over_eps2_measured": float(obs["projector_matrix"]["residual_fro"] / eps ** 2),
                            "residual_over_eps2_closed_form": cf["projector_matrix_residual_over_eps2"],
                            "reference": params["reference"].get(f"{eps:g}")}
        records.append(rec)
    valid = [r for r in records if r["valid"]]
    # slopes along this fixed path, window = in-domain points
    slopes = {}
    for name in O.OBSERVABLES:
        win = [r for r in valid if r["in_domain"]]
        slopes[name] = {**O.slope_loglog([r["eps"] for r in win], [O.residual_of(r["observables"], name) for r in win]),
                        "window": "eta <= 0.1", "eps_used": [r["eps"] for r in win]}
    exact = fam in O.EXACT_FAMILIES
    predeclared = None if exact else O.PREDECLARED_ORDERS[fam]
    tol = 0.25
    checks = {"all_points_valid": bool(len(valid) == len(records)),
              "projector_identity_lt_1e-10": bool(all(r["observables"]["identity_abs_error"] < 1e-10 for r in valid))}
    if exact:
        thr = 1e-12 if fam == "zero" else 1e-9
        mx = max(r["observables"]["projector_matrix"]["measured_fro"] for r in valid)
        gx = max(r["observables"]["graph"]["measured_fro"] for r in valid)
        checks.update({"absolute_branch_all_points": bool(all(r["observables"]["absolute_branch"] for r in valid)),
                       "ratio_recorded_null": bool(all(r["observables"]["graph"]["ratio"] is None for r in valid)),
                       f"max_displacement_lt_{thr:g}": bool(max(mx, gx) < thr),
                       "bound_satisfied_all_points": bool(all(r["bound"]["satisfied"] for r in valid)) if model == "covariance" else None,
                       "max_projector_displacement_fro": float(mx), "max_graph_displacement_fro": float(gx)})
    else:
        for name in O.OBSERVABLES:
            s = slopes[name]["slope"]
            checks[f"order_{name}"] = bool(s is not None and abs(s - predeclared[name]) <= tol)
        checks["orders_within_tolerance"] = bool(all(checks[f"order_{n}"] for n in O.OBSERVABLES))
        if fam == "two_d_example":
            checks["two_d_graph_closed_form_lt_1e-12"] = bool(all(r["two_d"]["abs_error_t"] < 1e-12 for r in valid))
            refs = [r for r in valid if r["two_d"]["reference"] is not None]
            checks["two_d_projector_reference_lt_1e-5"] = bool(refs and all(abs(r["two_d"]["residual_over_eps2_measured"] - r["two_d"]["reference"]) < 1e-5 for r in refs))
            checks["two_d_projector_over_eps2_closed_form_lt_1e-9"] = bool(all(abs(r["two_d"]["residual_over_eps2_measured"] - r["two_d"]["residual_over_eps2_closed_form"]) < 1e-9 for r in valid))
        if model == "covariance":
            checks["bound_satisfied_in_domain"] = bool(all(r["bound"]["satisfied"] for r in valid if r["in_domain"]))
    result = {"kind": "deterministic", "d": int(d), "k": k, "model": model, "family": fam, "operator": op_label,
              "model_label": "the paper's model A(I+E)^{-1}" if model == "covariance" else "CONTROL: additive A + eps G (not the paper's model)",
              "direction_seed": params.get("direction_seed"), "direction_op_norm": eps_op0, "spectrum": spec,
              "E0_blocks": {"11": float(np.linalg.norm(E0q[:k, :k], 2)), "21": float(np.linalg.norm(E0q[k:, :k], 2)), "22": float(np.linalg.norm(E0q[k:, k:], 2))},
              "predeclared_orders": predeclared, "slope_tolerance": tol, "records": records, "slopes": slopes,
              "summary": {"n_points": len(records), "n_valid": len(valid), "n_in_domain": int(sum(r["in_domain"] for r in valid)),
                          "max_bound_violation_in_domain": float(max([r["bound"]["violation"] for r in valid if r["in_domain"]] or [0.0])) if model == "covariance" else None},
              "checks": checks}
    return result, 0, solves


def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> tuple[dict, int, int]:
    params = cell["params"]
    if params["kind"] == "deterministic":
        return run_deterministic_cell(params)
    sel_sha = None
    if cell["role"] == "confirmation":
        sel = Path(ctx["out"]) / SELECTION_FILE
        if not sel.exists():
            raise RuntimeError(f"confirmation cell refused: {sel} is absent; reduce the pilot cells first "
                               "(the frozen selection must exist before any confirmation cell is drawn)")
        sel_sha = hashlib.sha256(sel.read_bytes()).hexdigest()
    return S.run_paired_cell(params, int(cell["seed"]), selection_sha256=sel_sha)


# =============================================================================
# reduce
# =============================================================================
def _median(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def _selection_from(paired: list[dict], P: dict, *, development_only: bool) -> dict:
    """The predeclared pilot selection rule, applied to the given paired cells."""
    eps_grid = [float(e) for e in paired[0]["params"]["eps_grid"]]
    m_grid = sorted({r["m"] for c in paired for r in c["result"]["records"] if r["source"] == "estimated"})
    recs = [r for c in paired for r in c["result"]["records"]]
    med_eps = {f"{e:g}": _median([r["eta_sample"] for r in recs if r["source"] == "engineered" and r["eps_nominal"] == e]) for e in eps_grid}
    in_dom = [e for e in eps_grid if med_eps[f"{e:g}"] is not None and med_eps[f"{e:g}"] <= O.ETA_IN_DOMAIN]
    stress = [e for e in eps_grid if med_eps[f"{e:g}"] is not None and O.ETA_STRESS[0] <= med_eps[f"{e:g}"] <= O.ETA_STRESS[1]]
    eng = {"eps_in_domain": sorted(in_dom)[-3:], "eps_stress": sorted({stress[0], stress[-1]}) if stress else [], "median_eta_sample_by_eps": med_eps}
    est = {}
    for fam in S.ESTIMATED_FAMILIES:
        med_m = {str(m): _median([r["eta_sample"] for r in recs if r["family"] == fam and r["m"] == m]) for m in m_grid}
        est[fam] = {"m_in_domain": [m for m in m_grid if med_m[str(m)] is not None and med_m[str(m)] <= O.ETA_IN_DOMAIN],
                    "m_stress": [m for m in m_grid if med_m[str(m)] is not None and O.ETA_STRESS[0] <= med_m[str(m)] <= O.ETA_STRESS[1]],
                    "median_eta_sample_by_m": med_m,
                    "theorem_domain_reachable": bool(any(med_m[str(m)] is not None and med_m[str(m)] <= O.ETA_IN_DOMAIN for m in m_grid))}
    digest = hashlib.sha256(json.dumps(sorted((c["cell_id"], c["attempt_uuid"]) for c in paired)).encode()).hexdigest()
    return {"schema": SELECTION_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"], "development_only": bool(development_only),
            "rule": P["pilot_selection_rule"], "source_cells_sha256": digest, "n_source_cells": len(paired),
            "source_role": paired[0]["role"], "engineered": eng, "estimated": est}


def _freeze_selection(out: Path, selection: dict) -> tuple[dict, str, bool]:
    """Write the selection once; refuse if a different one is already frozen."""
    path = out / SELECTION_FILE
    body = json.dumps(selection, indent=1, sort_keys=True).encode("utf-8")
    if path.exists():
        prev = json.loads(path.read_text())
        if {k: v for k, v in prev.items() if k != "frozen_utc"} != selection:
            raise ReconciliationError(f"{path} is already frozen and differs from the selection the pilot cells now imply; "
                                      "a frozen selection is never overwritten")
        return prev, hashlib.sha256(path.read_bytes()).hexdigest(), False
    write_json_atomic(path, selection)
    return selection, hashlib.sha256(path.read_bytes()).hexdigest(), True


def _ci(values, seed, P):
    b = P["inference"]["bootstrap"]
    return stable_bootstrap_ci(values, seed=seed, replicates=int(b["replicates"]), alpha=float(b["alpha"]))


def _score_confirmation(cells: list[dict], selection: dict, P: dict) -> dict:
    """Estimands (a)-(d) on the given paired cells with the frozen selection."""
    recs = [dict(r, seed=c["seed"]) for c in cells for r in c["result"]["records"]]
    seeds = sorted({c["seed"] for c in cells})
    bseed = int(P["inference"]["bootstrap"]["seed"])
    eps_sel = set(selection["engineered"]["eps_in_domain"])
    m_sel = {fam: set(selection["estimated"][fam]["m_in_domain"]) for fam in S.ESTIMATED_FAMILIES}

    def selected(r):
        if r["source"] == "engineered":
            return r["eps_nominal"] in eps_sel
        if r["source"] == "estimated":
            return r["m"] in m_sel[r["family"]]
        return False

    # (a) theorem-domain check
    a = {"definition": P["estimands"]["a_theorem_domain_check"], "by_source": {}}
    for src, fams in (("engineered", ("C", "F")), ("estimated", S.ESTIMATED_FAMILIES)):
        for fam in fams:
            sel_fits = [r for r in recs if r["family"] == fam and selected(r)]
            scored = [r for r in sel_fits if r["in_domain"]]
            fell_out = [r for r in sel_fits if not r["in_domain"]]
            per_seed = []
            for s in seeds:
                rs = [r for r in scored if r["seed"] == s]
                if rs:
                    per_seed.append(float(np.mean([r["bound"]["satisfied"] for r in rs])))
            a["by_source"][fam] = {
                "n_selected_arm_fits": len(sel_fits), "n_scored_in_domain": len(scored), "n_selected_but_out_of_domain": len(fell_out),
                "out_of_domain_classes": {c: int(sum(r["domain_class"] == c for r in fell_out)) for c in ("intermediate", "stress", "beyond", "indefinite")},
                "fraction_bound_satisfied": float(np.mean([r["bound"]["satisfied"] for r in scored])) if scored else None,
                "n_violations": int(sum(not r["bound"]["satisfied"] for r in scored)),
                "max_remainder_over_eta2": max([r["bound"]["remainder_over_eta2"] for r in scored if r["bound"]["remainder_over_eta2"] is not None] or [None]) if scored else None,
                "violations": [{"seed": r["seed"], "arm": r["arm"], "eta": r["eta_used_for_classification"], "lambda_min": r["lambda_min_IplusE"],
                                "residual_op": r["bound"]["residual_op"], "violation": r["bound"]["violation"]} for r in scored if not r["bound"]["satisfied"]][:50],
                "per_seed_fraction_ci": _ci(per_seed, bseed + 1, P) if per_seed else None,
                "undefined_reason": None if scored else ("no selected in-domain arm" if not sel_fits else "no selected-arm fit realised in-domain")}
    pooled = [r for r in recs if selected(r) and r["in_domain"]]
    per_seed = [float(np.mean([r["bound"]["satisfied"] for r in pooled if r["seed"] == s])) for s in seeds if any(r["seed"] == s for r in pooled)]
    a["pooled"] = {"n_scored": len(pooled), "fraction_bound_satisfied": float(np.mean([r["bound"]["satisfied"] for r in pooled])) if pooled else None,
                   "per_seed_fraction_ci": _ci(per_seed, bseed + 2, P) if per_seed else None,
                   "all_in_domain_fits_any_arm": {"n": int(sum(r["in_domain"] and r["bound"]["applies"] for r in recs)),
                                                  "fraction_bound_satisfied": float(np.mean([r["bound"]["satisfied"] for r in recs if r["in_domain"] and r["bound"]["applies"]])) if any(r["in_domain"] and r["bound"]["applies"] for r in recs) else None}}

    # (b) first-order agreement of the projector norm, by source and class (descriptive)
    b = {"definition": P["estimands"]["b_first_order_agreement_norm"], "band": list(BAND), "by_family_and_class": {}}
    for fam in ("C", "F", *S.ESTIMATED_FAMILIES, "D"):
        for cls in ("in_domain", "intermediate", "stress", "beyond", "indefinite"):
            rs = [r for r in recs if r["family"] == fam and r["domain_class"] == cls]
            if not rs:
                continue
            ratios = [r["observables"]["projector_norm"]["ratio"] for r in rs]
            defined = [x for x in ratios if x is not None]
            per_seed = [float(np.mean([BAND[0] <= r["observables"]["projector_norm"]["ratio"] <= BAND[1] for r in rs if r["seed"] == s and r["observables"]["projector_norm"]["ratio"] is not None]))
                        for s in seeds if any(r["seed"] == s and r["observables"]["projector_norm"]["ratio"] is not None for r in rs)]
            b["by_family_and_class"][f"{fam}/{cls}"] = {
                "n": len(rs), "n_absolute_branch": len(ratios) - len(defined), "label": "outside the theorem" if cls != "in_domain" else "theorem domain",
                "fraction_in_band": float(np.mean([BAND[0] <= x <= BAND[1] for x in defined])) if defined else None,
                "ratio_median": _median(defined), "ratio_min": float(min(defined)) if defined else None, "ratio_max": float(max(defined)) if defined else None,
                "graph_ratio_median": _median([r["observables"]["graph"]["ratio"] for r in rs]),
                "angle_ratio_median": _median([r["observables"]["angle"]["ratio"] for r in rs]),
                "per_seed_fraction_ci": _ci(per_seed, bseed + 3, P) if per_seed else None}

    # (c) residual orders along the fixed sample-basis direction, per family and observable
    c = {"definition": P["estimands"]["c_residual_orders"], "tolerance": float(P["predeclared_orders"]["slope_tolerance_paired"]), "by_family": {}}
    full_grid = [float(e) for e in cells[0]["params"]["eps_grid"]]
    use_eps = sorted(eps_sel) if len(eps_sel) >= 3 else full_grid
    window = "selected in-domain eps" if len(eps_sel) >= 3 else "full path (fewer than 3 in-domain eps selected)"
    for fam, kind in (("C", "pure_coupling"), ("F", "generic")):
        c["by_family"][fam] = {"window": window, "eps_used": use_eps, "by_observable": {}}
        for name in O.OBSERVABLES:
            per_seed = []
            for s in seeds:
                rs = [r for r in recs if r["family"] == fam and r["seed"] == s and r["eps_nominal"] in use_eps]
                sl = O.slope_loglog([r["eps_nominal"] for r in rs], [O.residual_of(r["observables"], name) for r in rs])
                if sl["slope"] is not None:
                    per_seed.append(sl["slope"])
            pred = O.PREDECLARED_ORDERS[kind][name]
            ci = _ci(per_seed, bseed + 4, P) if per_seed else None
            c["by_family"][fam]["by_observable"][name] = {
                "predeclared_order": pred, "n_seeds_with_slope": len(per_seed), "mean_slope_ci": ci,
                "slope_median": _median(per_seed),
                "order_consistent": (bool(abs(ci["mean"] - pred) <= c["tolerance"]) if ci else None), "is_outcome_not_assertion": True}

    # (d) training-sample arm, descriptive
    d = {"definition": P["estimands"]["d_training_sample_arm"], "by_N": {}}
    for N in sorted({r["N"] for r in recs if r["family"] == "D"}):
        rs = [r for r in recs if r["family"] == "D" and r["N"] == N]
        d["by_N"][str(N)] = {"n": len(rs), "eta_D_median": _median([r["eta_D"] for r in rs]),
                             "domain_counts": {k: int(sum(r["domain_class"] == k for r in rs)) for k in ("in_domain", "intermediate", "stress", "beyond")},
                             "graph_ratio_median": _median([r["observables"]["graph"]["ratio"] for r in rs]),
                             "projector_norm_ratio_median": _median([r["observables"]["projector_norm"]["ratio"] for r in rs]),
                             "angle_ratio_median": _median([r["observables"]["angle"]["ratio"] for r in rs]),
                             "remainder_over_eta_D2_median": _median([r["bound"]["remainder_over_eta2"] for r in rs]), "bound_asserted": False}

    # realised-class census: every attempt, by arm
    census = {}
    for arm in sorted({r["arm"] for r in recs}):
        rs = [r for r in recs if r["arm"] == arm]
        census[arm] = {"n": len(rs), "classes": {k: int(sum(r["domain_class"] == k for r in rs)) for k in ("in_domain", "intermediate", "stress", "beyond", "indefinite")},
                       "eta_median": _median([r["eta_used_for_classification"] for r in rs]), "eta_basis": rs[0]["eta_used_is"],
                       "floor_applied": int(sum(r["floor_applied"] for r in rs)), "selected_for_scoring": bool(selected(rs[0]))}
    return {"n_cells": len(cells), "n_seeds": len(seeds), "n_fits_recorded": sum(len(c["result"]["records"]) + 1 for c in cells),
            "a_theorem_domain_check": a, "b_first_order_agreement_norm": b, "c_residual_orders": c, "d_training_sample_arm": d,
            "census_by_arm": census,
            "stress_label": "eta in [0.2, 1.3] is OUTSIDE the theorem domain; nothing in that range validates or extends the 44 eta^2 bound"}


def _deterministic_summary(cells: list[dict]) -> tuple[dict, list[str]]:
    failures = []
    table = {"covariance": {}, "additive_control": {}}
    controls = {}
    for c in cells:
        r = c["result"]
        fam, model = r["family"], r["model"]
        key = "covariance" if model == "covariance" else "additive_control"
        for name, ok in r["checks"].items():
            if ok is False:
                failures.append(f"{c['cell_id']} ({model}/{fam}, {r['operator']}): check {name} failed")
        if fam in O.EXACT_FAMILIES:
            controls.setdefault(fam, []).append({"cell_id": c["cell_id"], "operator": r["operator"],
                                                 "max_projector_displacement_fro": r["checks"]["max_projector_displacement_fro"],
                                                 "max_graph_displacement_fro": r["checks"]["max_graph_displacement_fro"],
                                                 "absolute_branch": r["checks"]["absolute_branch_all_points"]})
            continue
        entry = table[key].setdefault(fam, {"predeclared": r["predeclared_orders"], "operator": r["operator"], "directions": []})
        entry["directions"].append({"cell_id": c["cell_id"], "direction_seed": r["direction_seed"],
                                    "slopes": {n: r["slopes"][n]["slope"] for n in O.OBSERVABLES},
                                    "n_points": {n: r["slopes"][n]["n_points"] for n in O.OBSERVABLES},
                                    "within_tolerance": r["checks"]["orders_within_tolerance"],
                                    "max_identity_error": max(p["observables"]["identity_abs_error"] for p in r["records"] if p["valid"]),
                                    "max_eigensolver_residual": max(p["eigensolver_residual"] for p in r["records"] if p["valid"]),
                                    "two_d": ([{"eps": p["eps"], **p["two_d"]} for p in r["records"]] if fam == "two_d_example" else None)})
    return {"order_table": table, "controls": controls, "n_cells": len(cells)}, failures


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    cs = list(cells.values())
    det = [c for c in cs if c["result"]["kind"] == "deterministic"]
    paired = [c for c in cs if c["result"]["kind"] == "paired"]
    det_summary, failures = _deterministic_summary(det)
    # paired-cell invariants are also scientific checks
    for c in paired:
        for name, ok in c["result"]["checks"].items():
            if ok is False:
                failures.append(f"{c['cell_id']} (paired seed {c['seed']}): check {name} failed")
    if failures:
        raise ScientificCheckError("scientific checks failed: " + "; ".join(failures[:8]) + (f" (+{len(failures) - 8} more)" if len(failures) > 8 else ""))

    dev = [c for c in paired if c["role"] == "development"]
    pilot = [c for c in paired if c["role"] == "pilot"]
    conf = [c for c in paired if c["role"] == "confirmation"]
    out = Path(out)
    summary = {"scientific_checks": {"passed": True, "n_deterministic_cells": len(det), "n_paired_cells": len(paired)},
               "deterministic": det_summary, "stage": None, "selection": None, "pilot": None, "confirmation": None}
    if dev:
        # fixture: the same code path as pilot -> confirmation, on development cells, labelled
        sel = _selection_from(dev, P, development_only=True)
        sel, sha, written = _freeze_selection(out, sel)
        summary.update({"stage": "development", "selection": {"file": SELECTION_FILE, "sha256": sha, "written_now": written, "development_only": True},
                        "pilot": {"n_cells": len(dev), "development_only": True, "engineered": sel["engineered"], "estimated": sel["estimated"]},
                        "confirmation": {**_score_confirmation(dev, sel, P), "development_only": True,
                                         "note": "development cells scored with the development selection to exercise the reducer; never citable"}})
    elif pilot and not conf:
        sel = _selection_from(pilot, P, development_only=False)
        sel, sha, written = _freeze_selection(out, sel)
        summary.update({"stage": "pilot", "selection": {"file": SELECTION_FILE, "sha256": sha, "written_now": written},
                        "pilot": {"n_cells": len(pilot), "engineered": sel["engineered"], "estimated": sel["estimated"]},
                        "confirmation": {"status": "not run", "note": "the pilot-stage reduce is partial by construction (confirmation cells absent)"}})
    elif conf:
        path = out / SELECTION_FILE
        if not path.exists():
            raise ReconciliationError(f"confirmation cells present but {path} is missing: the frozen selection is required to score")
        sel = json.loads(path.read_text())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sel.get("development_only"):
            raise ReconciliationError("the frozen selection is development-only; confirmation cannot be scored against it")
        bad = [c["cell_id"] for c in conf if c["result"].get("selection_sha256") != sha]
        if bad:
            raise ReconciliationError(f"{len(bad)} confirmation cells recorded a different PILOT_SELECTION.json digest than the one on disk (first: {bad[:3]})")
        if pilot:
            implied = _selection_from(pilot, P, development_only=False)
            if {k: v for k, v in sel.items() if k != "frozen_utc"} != implied:
                raise ReconciliationError("the frozen selection differs from the one the pilot cells imply")
        n_conf_expected = sum(1 for c in matrix.cells if c["role"] == "confirmation")
        summary.update({"stage": "confirmation", "selection": {"file": SELECTION_FILE, "sha256": sha},
                        "pilot": {"n_cells": len(pilot), "engineered": sel["engineered"], "estimated": sel["estimated"]},
                        "confirmation": {**_score_confirmation(conf, sel, P), "n_expected": n_conf_expected, "partial": bool(len(conf) < n_conf_expected)}})
    else:
        summary["stage"] = "deterministic_only"
    summary["status_vocabulary"] = "locally tested" if dev else "executed (not validated)"
    return summary


# =============================================================================
# production resource estimate from fixture timings
# =============================================================================
def production_estimate(P: dict, fixture_root: Path) -> dict:
    """Measured fixture timings (per family, from a --fixture run root) x the
    declared ``budget.fixture_to_production_scale`` x the PRODUCTION matrix.

    The common ``estimate`` subcommand rebuilds the matrix with the run root's
    own fixture flag, so pointed at a fixture root it prices the fixture matrix;
    this function prices the production matrix instead and is exposed as the
    package-local ``estimate-production`` command.  Absent timings are reported
    as absent, never guessed."""
    from ..common.results import read_cells
    fixture_root = Path(fixture_root)
    if not (fixture_root / "run_matrix_summary.json").exists():
        raise cli.UsageError(f"{fixture_root} has no run matrix")
    if not json.loads((fixture_root / "run_matrix_summary.json").read_text()).get("fixture"):
        raise cli.UsageError("estimate-production expects a --fixture run root for the timings")
    timings: dict[str, list[float]] = {}
    for files in read_cells(fixture_root).values():
        for _, js in files:
            timings.setdefault(js["family"] or "-", []).append(float(js.get("wall_s", 0.0)))
    prod = build_matrix(P, False)
    scale = P["budget"].get("fixture_to_production_scale", {})
    est = {"basis": "fixture wall seconds per cell x declared scale x production matrix", "fixture_root": str(fixture_root),
           "families": {}, "total_cpu_seconds_point": 0.0, "unmeasured_families": []}
    for fam, info in prod.counts()["by_family"].items():
        t = timings.get(fam)
        if not t:
            est["unmeasured_families"].append(fam)
            est["families"][fam] = {"cells": info["cells"], "measured": False}
            continue
        sc = float(scale.get(fam, scale.get("default", 1.0)))
        per = float(sum(t) / len(t)) * sc
        est["families"][fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": True,
                                "fixture_seconds_per_cell": float(sum(t) / len(t)), "fixture_cells_timed": len(t), "scale": sc,
                                "production_seconds_per_cell": per, "production_cpu_seconds": per * info["cells"]}
        est["total_cpu_seconds_point"] += per * info["cells"]
    est["total_cpu_hours_point"] = est["total_cpu_seconds_point"] / 3600.0
    est["counts"] = prod.counts()
    est["stages"] = {"pilot_cells": prod.counts()["by_role"].get("pilot", {}).get("cells"), "confirmation_cells": prod.counts()["by_role"].get("confirmation", {}).get("cells"),
                     "deterministic_cells": sum(prod.counts()["by_role"].get(r, {}).get("cells", 0) for r in ("algebraic", "control"))}
    est["retry_allowance"] = {"policy": P["budget"].get("retry_policy", "unspecified"),
                              "fits_headroom_under_cap": P["budget"]["estimator_fits_cap"] - prod.counts()["fits"],
                              "cpu_seconds_headroom_at_25_percent": 0.25 * est["total_cpu_seconds_point"]}
    est["memory_note"] = P["budget"].get("scale_basis", "")
    est["parallelism"] = "cells are independent; one ledger per run root (advisory flock; serial fallback on network filesystems); <WORKERS> unresolved"
    return est


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "estimate-production":
        import argparse
        ap = argparse.ArgumentParser(prog=f"python -m experiments.mlst2026.successor_20260913.{PACKAGE} estimate-production")
        ap.add_argument("--protocol", required=True, type=Path)
        ap.add_argument("--out", required=True, type=Path, help="a --fixture run root holding the timings; RESOURCE_ESTIMATE_PRODUCTION.json is written there")
        a = ap.parse_args(argv[1:])
        try:
            from ..common.protocol import load_protocol
            P = load_protocol(a.protocol)
            est = production_estimate(P, a.out)
        except (cli.UsageError, cli.ProtocolError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 6
        write_json_atomic(Path(a.out) / "RESOURCE_ESTIMATE_PRODUCTION.json", est)
        print(json.dumps(est, indent=1))
        return 0
    return cli.main_template(package=PACKAGE, description="T19-S v2: covariance-error observables, theorem-domain check and residual orders on G1",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run, result_required=RESULT_REQUIRED,
                             source_files=SOURCE_FILES, fixture_overrides=fixture_overrides, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
