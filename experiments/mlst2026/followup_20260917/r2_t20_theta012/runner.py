"""R2 runner: T20-S v3 theta=0.12 scaling and crossover on an extended N grid.

This is a NEW protocol (``PROTOCOL_R2_T20_THETA012_v1.json``) opened after the
September results were observed.  It imports the archived analytic generator and
analysis primitives unchanged (``successor_20260913.t20s.generator`` and
``.analysis``) and never edits, re-runs or re-scores a t20s artifact.

Flow (as the framework requires)::

    algebraic + pilot cells  ->  pilot reduce (writes FROZEN_MODEL.json)  ->
    confirmation cells       ->  confirmation reduce (scores the two endpoints)

Endpoint (a) re-evaluates the OLD t20s frozen d-model on fresh seeds with the
offset convention fixed (``a_ref`` applied exactly once); endpoint (b) selects a
new model on fresh pilot seeds only and freezes it before independent
confirmation.  A confirmation cell refuses to run without the freeze, and the
freeze digest is checked at reduce.

The offset convention is load bearing.  The t20s reducer stored
``prediction_at_grid`` already including ``a_ref``; R2 rebuilds every
d-prediction from the original ``(c, d)`` and the CURRENT algebraic ``a_ref``
and rejects (exit 4) a prediction that is not ``base + a_ref`` exactly once.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026.successor_20260913.common import cli, fitcount  # noqa: E402
from experiments.mlst2026.successor_20260913.common.matrix import RunMatrix, make_cell  # noqa: E402
from experiments.mlst2026.successor_20260913.common.protocol import ProtocolError, canonical_json, load_protocol  # noqa: E402
from experiments.mlst2026.successor_20260913.common.results import (ReconciliationError, SchemaError,  # noqa: E402
                                                                     read_cells, write_json_atomic)
from experiments.mlst2026.successor_20260913.t20s import generator as GEN  # noqa: E402
from . import analysis as AN  # noqa: E402

PACKAGE = "r2_t20_theta012"
DESCRIPTION = ("R2: T20-S v3 theta=0.12 scaling and crossover on an extended N grid "
               "(old-prediction endpoint + fresh pilot-frozen new model)")
HERE = _HERE.parent
PROTOCOL_DEFAULT = HERE / "PROTOCOL_R2_T20_THETA012_v1.json"
FROZEN_NAME = "FROZEN_MODEL.json"
FROZEN_SCHEMA = "r2-t20-frozen-model/1"
PHASES = ("pilot", "confirmation")
TOL = 1e-9
OLD_GRID_MAX = 3200
OPTIONAL_ENV = "R2_T20_INCLUDE_OPTIONAL"
SOURCE_FILES = [
    HERE / "runner.py", HERE / "analysis.py", PROTOCOL_DEFAULT,
    _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t20s" / "generator.py",
    _ROOT / "experiments" / "mlst2026" / "successor_20260913" / "t20s" / "analysis.py",
    _ROOT / "experiments" / "paper1" / "t20_population_accounting.py",
    _ROOT / "experiments" / "synthetic" / "nfpa_confirmatory" / "methods.py",
    _ROOT / "experiments" / "synthetic" / "nfpa_confirmatory" / "generator.py",
]
RESULT_REQUIRED = {"kind": str, "passed": bool, "checks": dict, "summary": dict}


class ScienceError(SchemaError):
    """A scientific self-check failed (exit 4)."""


# --------------------------------------------------------------------------------------
# design and matrix
# --------------------------------------------------------------------------------------
def _optional_from_env() -> bool:
    return os.environ.get(OPTIONAL_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def design(P: dict, fixture: bool, include_optional: bool = False) -> dict:
    S = dict(P["matrix"]["fixture"] if fixture else P["matrix"]["production"])
    S["fixture"] = bool(fixture)
    S["k"] = int(S["k_c"]) * int(S["k_t"])
    S["d"] = int(S["C"]) * int(S["T"])
    required = [float(t) for t in S["geometries"]["required"]]
    opt = S["geometries"].get("optional_costed_expansion")
    S["thetas_required"] = required
    S["optional_theta"] = float(opt["theta"]) if opt else None
    S["thetas"] = list(required)
    if include_optional and S["optional_theta"] is not None:
        S["thetas"] = list(required) + [S["optional_theta"]]
    S["N_grid"] = sorted(int(n) for n in list(S.get("N_core", [])) + list(S.get("N_anchor", [])))
    S["anchors"] = set(int(n) for n in S.get("N_anchor", []))
    S["seeds"] = {"pilot": list(range(int(S["pilot_seeds"][0]), int(S["pilot_seeds"][1]) + 1)),
                  "confirmation": list(range(int(S["confirmation_seeds"][0]), int(S["confirmation_seeds"][1]) + 1))}
    S["old_grid_max"] = int(P["grid_catalogue"]["old_grid_max"])
    return S


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    return _build_matrix(P, fixture, include_optional=_optional_from_env())


def _build_matrix(P: dict, fixture: bool, include_optional: bool = False) -> RunMatrix:
    D = design(P, fixture, include_optional=include_optional)
    thetas = list(D["thetas"])
    base = {k: D[k] for k in ("C", "T", "k_c", "k_t", "spectrum", "geometry_seed", "R_pop",
                              "pop_als_n_iter", "pop_als_tol")}
    base["fixture"] = bool(fixture)
    cells: list[dict] = []
    for ti, th in enumerate(thetas):
        cells.append(make_cell(
            role="algebraic", seed=None, family="algebraic", n_fits=0,
            n_solves=1 + (int(D["R_pop"]) + 1) + 1,
            params={**base, "kind": "algebraic", "theta": float(th), "theta_index": ti, "thetas_all": thetas,
                    "control": "theta0_a_zero" if float(th) == 0.0 else None},
            notes="invariants, partial-trace lower bound, population best-found a (solves only)"))
    for phase in PHASES:
        role = "development" if fixture else phase
        for seed in D["seeds"][phase]:
            for ti, th in enumerate(thetas):
                for ni, N in enumerate(D["N_grid"]):
                    params = {**base, "kind": "fit", "phase": phase, "theta": float(th), "theta_index": ti,
                              "N": int(N), "N_index": ni, "band": "anchor" if N in D["anchors"] else "core",
                              "R": int(D["R"]), "als_n_iter": int(D["als_n_iter"]), "als_tol": float(D["als_tol"]),
                              "control": "theta0_a_zero" if float(th) == 0.0 else None}
                    cells.append(make_cell(role=role, seed=int(seed), family=phase, n_fits=1 + int(D["R"]),
                                           n_solves=0, params=params))
    for name, cfg in P["matrix"]["controls"].items():
        cells.append(make_cell(role="control", seed=None, family="control", n_fits=0, n_solves=0,
                               params={"control": name, **dict(cfg)}))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def fixture_overrides(P: dict) -> dict:
    return {"scale": "fixture", "thetas": list(P["matrix"]["fixture"]["geometries"]["required"])}


# --------------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------------
def _geometry(D: dict) -> GEN.Geometry:
    return GEN.make_geometry(C=D["C"], T=D["T"], k_c=D["k_c"], k_t=D["k_t"], spec=D["spectrum"],
                             seed=D["geometry_seed"])


def _reference(D: dict, theta: float) -> dict:
    return GEN.population_reference(C=D["C"], T=D["T"], k_c=D["k_c"], k_t=D["k_t"], spec=D["spectrum"],
                                    seed=D["geometry_seed"], theta=theta, R_pop=D["R_pop"],
                                    n_iter=D["pop_als_n_iter"], tol=D["pop_als_tol"])


def _ref_public(ref: dict) -> dict:
    return {k: v for k, v in ref.items() if k not in ("B_U_star", "B_F_best", "K")}


def run_algebraic_cell(P: dict, cell: dict, D: dict) -> dict:
    p = cell["params"]
    thetas = [float(t) for t in p.get("thetas_all", D["thetas"])]
    GEN._population_reference_cached.cache_clear()   # the algebraic cell pays its own solves
    g = _geometry(D)
    inv = GEN.invariants(g, thetas, R_pop=D["R_pop"], n_iter=D["pop_als_n_iter"], tol=D["pop_als_tol"],
                         spec=D["spectrum"])
    ref = _ref_public(_reference(D, float(p["theta"])))
    th = float(p["theta"])
    excluded = th in [float(t) for t in inv["excluded_thetas"]]
    res = {"kind": "algebraic", "theta": th, "theta_index": int(p["theta_index"]),
           "eigenvalues_max_abs_dev": ref["eigenvalues_max_abs_dev"], "eig_tol": GEN.EIG_TOL,
           "trace_K": ref["trace_K"], "trace_mu": ref["trace_mu"], "numerical_rank": ref["numerical_rank"],
           "d": ref["d"], "k": ref["k"], "snr_per_coordinate": ref["snr_per_coordinate"], "Delta_k": ref["Delta_k"],
           "partial_trace": ref["partial_trace"], "k_minus_cap": ref["partial_trace"]["k_minus_cap"],
           "a_lower_bound": ref["a_lower_bound"], "a_best_found": ref["a_best_found"],
           "a_interval": [ref["a_lower_bound"], ref["a_best_found"]],
           "R_U_star": ref["R_U_star"], "R_F_best_found": ref["R_F_best_found"], "pop_als": ref["pop_als"],
           "is_theta0_control": bool(th == 0.0),
           "theta0_control_passed": (bool(abs(ref["a_lower_bound"]) <= TOL and ref["a_best_found"] <= 1e-8)
                                     if th == 0.0 else None),
           "spectrum_fixed": bool(ref["eigenvalues_max_abs_dev"] <= GEN.EIG_TOL),
           "nonseparability_moves_across_grid": bool(inv["nonseparability_moves"]),
           "k_minus_cap_by_theta": {str(t): inv["per_theta"][t]["partial_trace"]["k_minus_cap"] for t in thetas},
           "a_interval_by_theta": {str(t): [inv["per_theta"][t]["a_lower_bound"], inv["per_theta"][t]["a_best_found"]]
                                   for t in thetas},
           "invariant_problems": inv["problems"][str(th)], "excluded": bool(excluded),
           "excluded_thetas_across_grid": [float(t) for t in inv["excluded_thetas"]],
           "solves": {"eigendecomposition": 1, "population_als_inits": D["R_pop"] + 1, "bound": 1},
           "note": ("population reference for this theta; a_best_found is an UPPER bound on a, a_lower_bound is "
                    "the partial-trace bound; neither certifies P*_F")}
    res["checks"] = {"spectrum_fixed": res["spectrum_fixed"], "excluded": not res["excluded"]}
    check_algebraic_result(res)
    res["passed"] = True
    res["summary"] = {"theta": th, "theta_index": int(p["theta_index"]), "a_interval": res["a_interval"],
                      "k_minus_cap": res["k_minus_cap"]}
    return res


def select_restart(losses: list[float]) -> int:
    """Training-loss restart selection (first minimum), mirroring ``als_fit``."""
    return int(np.argmin(np.asarray(losses, float)))


def run_fit_cell(P: dict, cell: dict, D: dict, ctx: dict) -> tuple[dict, int]:
    from experiments.synthetic.nfpa_confirmatory.methods import als_fit  # audited, unchanged
    p = cell["params"]
    th, N, R = float(p["theta"]), int(p["N"]), int(p["R"])
    C, T, k_c, k_t, k = D["C"], D["T"], D["k_c"], D["k_t"], D["k"]
    if p["phase"] == "confirmation" and not (Path(ctx["out"]) / FROZEN_NAME).exists():
        raise cli.UsageError(f"confirmation cell {cell['cell_id']} refused: {FROZEN_NAME} is not in the run root")
    g = _geometry(D)
    ref = _reference(D, th)
    K = ref["K"]
    phase_code = PHASES.index(p["phase"])
    Z = GEN.draw_data(g, th, N, [int(cell["seed"]), int(p["theta_index"]), N, phase_code, int(D["geometry_seed"])])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    S = Zc.T @ Zc / N
    fitcount.count_fit("pca")
    w, U = np.linalg.eigh(S)
    order = np.argsort(w)[::-1]
    B_U = U[:, order[:k]]
    pca_train_residual = float(np.trace(S) - np.sum(w[order[:k]]))
    als_seed = int(cell["seed"]) * 1000 + int(p["theta_index"]) * 100 + int(p["N_index"])
    out = als_fit(Zc.reshape(N, C, T, order="F"), k_c, k_t, n_iter=int(p["als_n_iter"]), n_restarts=R,
                  seed=als_seed, tol=float(p["als_tol"]), collect_restarts=True)
    fitcount.count_completed("als_restart", n=len(out["restarts"]))
    losses = [float(r["loss"]) for r in out["restarts"]]
    risks = [float(GEN.risk(K, GEN.kron_basis(r["U_c"], r["U_t"]))) for r in out["restarts"]]
    sel = select_restart(losses)
    B_F = GEN.kron_basis(out["restarts"][sel]["U_c"], out["restarts"][sel]["U_t"])
    R_F_hat, R_U_hat = float(GEN.risk(K, B_F)), float(GEN.risk(K, B_U))
    d = R_F_hat - R_U_hat
    x_U = R_U_hat - ref["R_U_star"]
    x_F_ref = R_F_hat - ref["R_F_best_found"]
    a_best = ref["a_best_found"]
    pop_best = int(np.argmin(risks))
    spread_risk = float(max(risks) - min(risks))
    res = {"kind": "fit", "phase": p["phase"], "theta": th, "theta_index": int(p["theta_index"]), "N": N,
           "N_index": int(p["N_index"]), "band": p["band"], "R": R, "C": C, "T": T, "k_c": k_c, "k_t": k_t, "k": k,
           "R_U_hat": R_U_hat, "R_F_hat": R_F_hat, "d": float(d), "x_U": float(x_U), "x_F_ref": float(x_F_ref),
           "a_lower_bound": float(ref["a_lower_bound"]), "a_best_found": float(a_best), "a_ref": float(a_best),
           "a_interval": [float(ref["a_lower_bound"]), a_best],
           "identity_residual": float(d - (a_best + x_F_ref - x_U)),
           "R_U_star": float(ref["R_U_star"]), "R_F_best_found": float(ref["R_F_best_found"]),
           "Delta_k": float(ref["Delta_k"]), "eigenvalues_max_abs_dev": float(ref["eigenvalues_max_abs_dev"]),
           "restart_train_losses": losses, "restart_pop_risks": risks,
           "restart_converged": [bool(r["converged"]) for r in out["restarts"]],
           "selected_restart": sel, "als_selected_restart": int(out["restart"]), "selected_by": "training_loss",
           "population_best_restart": pop_best, "selection_matches_population_best": bool(sel == pop_best),
           "restart_spread_loss": float(max(losses) - min(losses)), "restart_spread_risk": spread_risk,
           "optimization_unresolved": bool(spread_risk >= abs(d)),
           "pca_train_residual": pca_train_residual, "als_train_loss": float(losses[sel]),
           "fits": {"pca": 1, "als_restarts": len(out["restarts"])}, "fits_observed": fitcount.observed()}
    check_fit_result(res)
    res["passed"] = True
    res["checks"] = {"identity": bool(abs(res["identity_residual"]) <= TOL),
                     "x_U_nonnegative": bool(res["x_U"] >= -TOL),
                     "training_loss_selection": bool(sel == int(np.argmin(losses)))}
    res["summary"] = {"phase": p["phase"], "theta": th, "theta_index": int(p["theta_index"]), "N": N,
                      "d": float(d), "x_U": float(x_U), "a_ref": float(a_best)}
    return res, fitcount.observed()


# --------------------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------------------
def _old_params_for(old_ref: dict, theta: float, *, tol: float = 1e-12) -> dict | None:
    for k, v in old_ref["per_theta"].items():
        if abs(float(k) - float(theta)) <= tol:
            return v
    return None


def run_control_cell(P: dict, cell: dict, fixture: bool, ctx: dict) -> dict:
    name = cell["params"]["control"]
    old = P["old_frozen_prediction"]
    checks: dict[str, dict] = {}
    if name == "offset_applied_once":
        params = _old_params_for(old, 0.12)
        if params is None:
            raise ScienceError("control offset_applied_once: the old frozen prediction has no theta=0.12 entry")
        off = float(params["offset_a_ref"])
        N = np.asarray(old["old_grid"], float)
        base, pred = AN.old_endpoint_prediction(params, N, off)
        double = base + 2.0 * off
        checks["base_plus_offset_once"] = {
            "passed": bool(np.max(np.abs(pred - (base + off))) <= 1e-12),
            "max_dev": float(np.max(np.abs(pred - (base + off))))}
        checks["double_application_is_distinct"] = {
            "passed": bool(abs(off) <= 1e-12 or float(np.min(np.abs(pred - double))) > 1e-6 * abs(off)),
            "min_abs_difference": float(np.min(np.abs(pred - double))), "offset": off}
        # the September trap: reuse a stored offset-bearing prediction and add a_ref again
        reuse_double = np.asarray(pred, float) + off
        checks["reused_stored_prediction_would_double"] = {
            "passed": bool(abs(off) <= 1e-12 or float(np.min(np.abs(reuse_double - pred))) > 1e-9),
            "note": "base+2a_ref differs from base+a_ref by |a_ref|; a stored prediction already contains a_ref"}
        a_fresh = 1.2345
        b2, p2 = AN.old_endpoint_prediction(params, N, a_fresh)
        checks["current_a_ref_used_in_fresh_evaluation"] = {
            "passed": bool(np.max(np.abs(p2 - (b2 + a_fresh))) <= 1e-12),
            "changes_with_a_ref": bool(float(np.max(np.abs(p2 - pred))) > 0.1)}
        guard_ok = False
        try:
            AN.check_offset_applied_once(params=params, N=N, offset=off, prediction=double, where="control")
        except ValueError:
            guard_ok = True
        checks["guard_rejects_double_application"] = {"passed": bool(guard_ok)}
    elif name == "old_selection_margins":
        cv = old["cv_errors"]
        m = AN.old_selection_margins(cv)
        checks["margin_over_none_matches_record"] = {
            "passed": bool(abs(m["margin_vs_none_rel"] - float(old["reported_margin_vs_none_relative"])) <= 1e-12),
            "computed": m["margin_vs_none_rel"], "recorded": float(old["reported_margin_vs_none_relative"])}
        checks["margin_over_c_over_N_alpha_matches_record"] = {
            "passed": bool(abs(m["margin_vs_c_over_N_alpha_rel"]
                               - float(old["reported_margin_vs_c_over_N_alpha_relative"])) <= 1e-12),
            "computed": m["margin_vs_c_over_N_alpha_rel"],
            "recorded": float(old["reported_margin_vs_c_over_N_alpha_relative"])}
        checks["the_two_margins_are_distinct"] = {
            "passed": bool(m["margin_vs_none_rel"] > 5.0 * m["margin_vs_c_over_N_alpha_rel"]),
            "margin_vs_none_rel": m["margin_vs_none_rel"],
            "margin_vs_c_over_N_alpha_rel": m["margin_vs_c_over_N_alpha_rel"],
            "note": "the reported 2.7% margin was over none; the margin over c/N^alpha is smaller"}
    elif name == "grid_and_offset_catalogue":
        gc = P["grid_catalogue"]
        D = design(P, fixture)
        checks["grid_is_old_plus_extended"] = {
            "passed": list(gc["N"]) == sorted(list(gc["old_N"]) + list(gc["extended_N"]))
                      and list(D["N_grid"]) == list(gc["N"])}
        checks["old_grid_max_declared"] = {"passed": int(gc["old_grid_max"]) == max(gc["old_N"])}
        checks["offset_convention_recorded"] = {"passed": bool(old.get("offset_convention"))}
        checks["old_params_cover_required_geometries"] = {
            "passed": all(_old_params_for(old, th) is not None
                          for th in P["matrix"]["production"]["geometries"]["required"])}
    else:
        raise ScienceError(f"unknown control {name!r}")
    passed = all(c["passed"] for c in checks.values())
    return {"kind": "control", "passed": bool(passed), "checks": checks, "summary": {"control": name}}


def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    D = design(P, fixture)
    if cell["role"] == "algebraic":
        return run_algebraic_cell(P, cell, D), 0, int(cell["n_solves"])
    if cell["family"] == "control":
        return run_control_cell(P, cell, fixture, ctx), 0, 0
    if cell["family"] in PHASES:
        res, fits = run_fit_cell(P, cell, D, ctx)
        return res, fits, 0
    raise ScienceError(f"unknown cell family {cell['family']!r}")


# --------------------------------------------------------------------------------------
# scientific self-checks
# --------------------------------------------------------------------------------------
def check_algebraic_result(r: dict) -> None:
    where = f"algebraic theta={r.get('theta')}"
    if r["eigenvalues_max_abs_dev"] > GEN.EIG_TOL:
        raise ScienceError(f"{where}: spectrum not fixed ({r['eigenvalues_max_abs_dev']:.2e} > {GEN.EIG_TOL})")
    if r["a_lower_bound"] < -TOL:
        raise ScienceError(f"{where}: negative lower bound {r['a_lower_bound']}")
    if r["a_lower_bound"] > r["a_best_found"] + TOL:
        raise ScienceError(f"{where}: lower bound {r['a_lower_bound']} exceeds best-found a {r['a_best_found']}")
    if r["a_best_found"] < -TOL:
        raise ScienceError(f"{where}: best-found a negative {r['a_best_found']}")
    if r["theta"] != 0.0 and abs(r["Delta_k"] * r["k_minus_cap"] - r["a_lower_bound"]) > TOL:
        raise ScienceError(f"{where}: lower bound is not Delta_k (k - cap)")
    if r["theta"] == 0.0 and not (abs(r["a_lower_bound"]) <= TOL and r["a_best_found"] <= 1e-8):
        raise ScienceError(f"{where}: theta = 0 control failed")
    if r["a_interval"] != [r["a_lower_bound"], r["a_best_found"]]:
        raise ScienceError(f"{where}: a_interval inconsistent with its bounds")


def check_fit_result(r: dict) -> None:
    where = f"fit theta={r.get('theta')} N={r.get('N')}"
    if r["selected_by"] != "training_loss":
        raise ScienceError(f"{where}: restart selection must be by training loss, got {r['selected_by']!r}")
    losses = r["restart_train_losses"]
    if len(losses) != r["R"] or len(r["restart_pop_risks"]) != r["R"]:
        raise ScienceError(f"{where}: restart records do not match R={r['R']}")
    if r["selected_restart"] != int(np.argmin(losses)) or r["als_selected_restart"] != r["selected_restart"]:
        raise ScienceError(f"{where}: selected restart is not the training-loss minimiser")
    if abs(r["identity_residual"]) > TOL:
        raise ScienceError(f"{where}: identity residual {r['identity_residual']:.2e} exceeds {TOL}")
    if abs(r["d"] - (r["R_F_hat"] - r["R_U_hat"])) > TOL:
        raise ScienceError(f"{where}: d is not R_F_hat - R_U_hat")
    if r["x_U"] < -TOL:
        raise ScienceError(f"{where}: x_U = {r['x_U']} < 0 is impossible")
    if abs(r["x_U"] - (r["R_U_hat"] - r["R_U_star"])) > TOL:
        raise ScienceError(f"{where}: x_U is not R_U_hat - R_U_star")
    if abs(r["x_F_ref"] - (r["R_F_hat"] - r["R_F_best_found"])) > TOL:
        raise ScienceError(f"{where}: x_F_ref is not R_F_hat - R_F_best_found")
    if r["a_lower_bound"] > r["a_best_found"] + TOL or r["a_lower_bound"] < -TOL or r["a_best_found"] < -TOL:
        raise ScienceError(f"{where}: a interval is not a valid interval")
    if abs(r["a_best_found"] - (r["R_F_best_found"] - r["R_U_star"])) > TOL:
        raise ScienceError(f"{where}: a_best_found is not R_F_best_found - R_U_star")
    if r["theta"] == 0.0 and not (abs(r["a_lower_bound"]) <= TOL and r["a_best_found"] <= 1e-8):
        raise ScienceError(f"{where}: theta = 0 control failed")
    if r["restart_spread_risk"] < -TOL or abs(r["restart_spread_risk"]
                                              - (max(r["restart_pop_risks"]) - min(r["restart_pop_risks"]))) > TOL:
        raise ScienceError(f"{where}: restart spread inconsistent with the recorded restart risks")
    if r["optimization_unresolved"] != bool(r["restart_spread_risk"] >= abs(r["d"])):
        raise ScienceError(f"{where}: optimization_unresolved flag inconsistent")
    if r["eigenvalues_max_abs_dev"] > GEN.EIG_TOL:
        raise ScienceError(f"{where}: spectrum not fixed")


def check_reference_consistency(fit: dict, alg: dict) -> None:
    for key in ("a_lower_bound", "a_best_found", "R_U_star", "R_F_best_found", "Delta_k"):
        if abs(float(fit[key]) - float(alg[key])) > TOL:
            raise ScienceError(f"fit cell theta={fit['theta']} N={fit['N']}: {key} differs from the algebraic cell")


def check_offset_convention(entry: dict, N, *, where: str) -> None:
    """Enforce ``prediction_at_grid = base_prediction_at_grid + offset_a_ref`` (once).

    The reduce path calls :func:`analysis.check_offset_applied_once` directly; this
    wrapper exists so a test or a future caller can enforce the same rule with the
    exit-4 :class:`ScienceError` type."""
    try:
        AN.check_offset_applied_once(params=entry["params"], N=N, offset=float(entry["offset_a_ref"]),
                                     prediction=entry["prediction_at_grid"], where=where)
    except ValueError as e:
        raise ScienceError(str(e)) from e


def _public_band(band: dict) -> dict:
    """A band without its bootstrap draws, for the stored reduction."""
    return {k: v for k, v in band.items() if k != "maxt_draws"}


# --------------------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------------------
def frozen_path(out) -> Path:
    return Path(out) / FROZEN_NAME


def _pilot_digest(cells: dict) -> str:
    items = sorted((cid, hashlib.sha256(canonical_json(js["result"]).encode()).hexdigest())
                   for cid, js in cells.items() if js["result"].get("kind") == "fit"
                   and js["result"]["phase"] == "pilot")
    return hashlib.sha256(canonical_json(items).encode()).hexdigest()


def _table(fits: list[dict], thetas: list[float], N_grid: list[int], key: str):
    out = {}
    for ti, th in enumerate(thetas):
        rows = [f for f in fits if f["theta_index"] == ti]
        seeds = sorted({f["_seed"] for f in rows})
        M = np.full((len(seeds), len(N_grid)), np.nan)
        for f in rows:
            M[seeds.index(f["_seed"]), N_grid.index(f["N"])] = f[key]
        out[ti] = (seeds, M)
    return out


def _bands_by_theta(tab, *, seed, replicates, alpha, min_n):
    out = {}
    for ti, (seeds, M) in tab.items():
        if M.size == 0 or np.any(np.isnan(M)):
            out[ti] = None
        else:
            out[ti] = AN.simultaneous_band(M, seed=seed + ti, replicates=replicates, alpha=alpha, min_n=min_n)
    return out


def _per_theta_summary(fits: list[dict], D: dict, alg_by_theta: dict, *, seed: int, replicates: int, alpha: float,
                       min_n: int) -> dict:
    thetas, N_grid = D["thetas"], D["N_grid"]
    per = {}
    tab_d = _table(fits, thetas, N_grid, "d")
    tab_xu = _table(fits, thetas, N_grid, "x_U")
    tab_xf = _table(fits, thetas, N_grid, "x_F_ref")
    for ti, th in enumerate(thetas):
        seeds, Md = tab_d[ti]
        _, Mxu = tab_xu[ti]
        rows = [f for f in fits if f["theta_index"] == ti]
        if not rows or np.any(np.isnan(Md)):
            per[str(th)] = {"complete": False, "n_seeds": len(seeds)}
            continue
        ci_d = AN.simultaneous_band(Md, seed=seed + ti, replicates=replicates, alpha=alpha, min_n=min_n)
        ci_xu = AN.simultaneous_band(Mxu, seed=seed + 100 + ti, replicates=replicates, alpha=alpha, min_n=min_n)
        prof = AN.archived_sign_profile(ci_d)
        ci_d_store, ci_xu_store = _public_band(ci_d), _public_band(ci_xu)
        by_N = {}
        for ni, N in enumerate(N_grid):
            rN = [f for f in rows if f["N"] == N]
            unres = float(np.mean([f["optimization_unresolved"] for f in rN]))
            by_N[str(N)] = {"n": len(rN), "d_mean": float(np.mean([f["d"] for f in rN])),
                            "d_sim_ci": [ci_d["lower"][ni], ci_d["upper"][ni]],
                            "d_sign": prof["signs"][ni] if prof["signs"] else None,
                            "x_U_mean": float(np.mean([f["x_U"] for f in rN])),
                            "x_F_ref_mean": float(np.mean([f["x_F_ref"] for f in rN])),
                            "restart_spread_risk_median": float(np.median([f["restart_spread_risk"] for f in rN])),
                            "selection_matches_population_best_frac": float(
                                np.mean([f["selection_matches_population_best"] for f in rN])),
                            "optimization_unresolved_frac": unres,
                            "sampling_vs_optimization_unresolved": bool(unres > 0.5)}
        alg = alg_by_theta[ti]
        per[str(th)] = {"complete": True, "theta_index": ti, "n_seeds": len(seeds), "seeds": seeds,
                        "excluded": bool(alg["excluded"]), "a_interval": alg["a_interval"],
                        "k_minus_cap": alg["k_minus_cap"], "Delta_k": alg["Delta_k"],
                        "d_simultaneous": ci_d_store, "x_U_simultaneous": ci_xu_store, "sign_profile": prof,
                        "by_N": by_N,
                        "adjacent_sign_brackets_N": [[N_grid[i], N_grid[i + 1]] for i in prof["adjacent_brackets"]],
                        "crossing_report": {
                            "brackets_N": [[N_grid[i], N_grid[i + 1]] for i in prof["adjacent_brackets"]],
                            "no_crossing_within_grid": bool(len(prof["adjacent_brackets"]) == 0),
                            "grid_bound_N": (int(N_grid[-1]) if len(prof["adjacent_brackets"]) == 0 else None),
                            "note": ("a crossing is reported only as an adjacent grid bracket of decided opposite "
                                     "signs; with no such pair the largest grid N is the bound and no crossing is "
                                     "claimed; a crossing is never interpolated")},
                        "max_identity_residual": float(max(abs(f["identity_residual"]) for f in rows)),
                        "min_x_U": float(min(f["x_U"] for f in rows))}
    return per


def _selection(fits: list[dict], D: dict, alg_by_theta: dict) -> dict:
    thetas, N_grid = D["thetas"], D["N_grid"]
    usable = [ti for ti, th in enumerate(thetas) if not alg_by_theta[ti]["excluded"]]
    seeds = sorted({f["_seed"] for f in fits})
    tab_xu = _table(fits, thetas, N_grid, "x_U")
    tab_d = _table(fits, thetas, N_grid, "d")
    targets = {"x_U": {ti: tab_xu[ti][1] for ti in usable},
               "d_minus_a_ref": {ti: tab_d[ti][1] - alg_by_theta[ti]["a_best_found"] for ti in usable}}
    out = {}
    for name, Y in targets.items():
        sel = AN.select_model(Y, N_grid, seeds)
        frozen = {}
        if sel["selected"] != "none":
            for ti in usable:
                params = AN.fit_model(sel["selected"], np.asarray(N_grid, float), Y[ti].mean(axis=0))
                base = AN.predict(params, np.asarray(N_grid, float))
                off = float(alg_by_theta[ti]["a_best_found"]) if name == "d_minus_a_ref" else 0.0
                frozen[str(thetas[ti])] = {"params": params, "offset_a_ref": off,
                                           "base_prediction_at_grid": base.tolist(),
                                           "prediction_at_grid": (base + off).tolist()}
        out[name] = {"selected": sel["selected"],
                     "cv": {k: v for k, v in sel.items() if k not in ("detail",)},
                     "cv_detail": sel.get("detail", {}), "frozen_per_theta": frozen,
                     "usable_theta_indices": usable}
    return out


def _freeze(P: dict, matrix: RunMatrix, cells: dict, out: Path, fixture: bool, D: dict, alg_by_theta: dict,
            sel: dict, per: dict) -> dict:
    thetas = D["thetas"]
    pilot_fits = [js for js in cells.values() if js["result"].get("kind") == "fit"
                  and js["result"]["phase"] == "pilot"]
    old = P["old_frozen_prediction"]
    old_ref = {str(th): {k: _old_params_for(old, th)[k] for k in ("c", "d", "offset_a_ref")}
               for th in thetas if _old_params_for(old, th) is not None}
    frozen = {"schema": FROZEN_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"],
              "matrix_sha256": matrix.matrix_hash(), "development_only": bool(fixture),
              "phase_frozen_from": "pilot", "pilot_cells": len(pilot_fits),
              "pilot_seeds": sorted({int(js["seed"]) for js in pilot_fits}),
              "pilot_digest": _pilot_digest(cells), "confirmation_cells_present_at_freeze": 0,
              "grid": {"theta": thetas, "N": D["N_grid"]},
              "old_prediction_reference": {"source": old["source"], "model": old["model"],
                                           "old_grid": old["old_grid"], "per_theta": old_ref},
              "new_selection": {name: {"selected": s["selected"], "cv_error": s["cv"]["cv_error"],
                                       "reason": s["cv"].get("reason"),
                                       "best_parametric": s["cv"].get("best_parametric"),
                                       "relative_improvement": s["cv"].get("relative_improvement"),
                                       "fit_half": s["cv"].get("fit_half"), "holdout_half": s["cv"].get("holdout_half"),
                                       "frozen_per_theta": s["frozen_per_theta"],
                                       "usable_theta_indices": s["usable_theta_indices"]}
                                for name, s in sel.items()},
              "offset_convention": ("prediction_at_grid = base_prediction_at_grid + offset_a_ref, applied exactly "
                                    "once; offset_a_ref is the CURRENT algebraic a_best_found. The t20s stored "
                                    "prediction_at_grid already contains the old a_ref and is never used as a base"),
              "candidate_family": list(AN.CANDIDATES), "min_relative_improvement": AN.MIN_RELATIVE_IMPROVEMENT,
              "statement": ("the new model was selected on fresh pilot seeds only and is frozen before any "
                            "confirmation cell; a development_only freeze is never citable")}
    p = frozen_path(out)
    if p.exists():
        prev = json.loads(p.read_text())
        if canonical_json({k: v for k, v in prev.items() if k != "frozen_utc"}) != canonical_json(
                json.loads(json.dumps(frozen, default=_js))):
            raise ReconciliationError(f"{p} exists and differs from the selection the pilot cells now imply; "
                                      f"a frozen model is never overwritten")
        frozen["frozen_utc"] = prev.get("frozen_utc")
    else:
        frozen["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json_atomic(p, frozen)
    return frozen


def _scoring(P: dict, frozen: dict, D: dict, alg_by_theta: dict, fits: list[dict], per: dict, *, inf: dict) -> dict:
    thetas, N_grid = D["thetas"], D["N_grid"]
    replicates, min_n = int(inf["replicates"]), int(inf["min_seeds_for_max_t_bootstrap"])
    tab_d = _table(fits, thetas, N_grid, "d")
    tab_xu = _table(fits, thetas, N_grid, "x_U")
    # predeclared claims: primary theta=0.12 and reference theta=0.25, theta=0 excluded (control)
    claim_thetas = [th for th in thetas if abs(float(th)) > 1e-12]
    old = P["old_frozen_prediction"]
    old_grid_max = int(P["grid_catalogue"]["old_grid_max"])
    # number of claims (old endpoint over claim_thetas with an old prediction, plus new endpoints)
    n_old_claims = sum(1 for th in claim_thetas if _old_params_for(old, th) is not None)
    n_new_claims = sum(1 for name, s in frozen["new_selection"].items() if s["selected"] != "none"
                       for th_s in s["frozen_per_theta"] if any(abs(float(th_s) - th) <= 1e-12 for th in claim_thetas))
    m_claims = max(1, n_old_claims + n_new_claims)
    alpha_claim = float(inf["alpha"]) / m_claims
    claim_bands = {"d": _bands_by_theta(tab_d, seed=int(inf["bootstrap_seed"]) + 5000, replicates=replicates,
                                        alpha=alpha_claim, min_n=min_n),
                   "x_U": _bands_by_theta(tab_xu, seed=int(inf["bootstrap_seed"]) + 6000, replicates=replicates,
                                          alpha=alpha_claim, min_n=min_n)}

    claims: dict[str, float] = {}
    # endpoint (a): OLD frozen prediction re-evaluated on fresh seeds
    endpoint_a = {}
    for ti, th in enumerate(thetas):
        p_old = _old_params_for(old, th)
        if p_old is None:
            endpoint_a[str(th)] = {"scored": False, "reason": "no old frozen prediction for this geometry"}
            continue
        band = per[str(th)]["d_simultaneous"] if per.get(str(th), {}).get("complete") else None
        if band is None:
            endpoint_a[str(th)] = {"scored": False, "reason": "confirmation cells incomplete for this geometry"}
            continue
        a_ref = float(alg_by_theta[ti]["a_best_found"])
        base, pred = AN.old_endpoint_prediction(p_old, N_grid, a_ref)
        try:
            AN.check_offset_applied_once(params=p_old, N=N_grid, offset=a_ref, prediction=pred,
                                         where=f"endpoint_a theta={th}")
        except ValueError as e:
            raise ScienceError(str(e)) from e
        score = AN.score_prediction(pred, band)
        extrap = [bool(int(N) > old_grid_max) for N in N_grid]
        inside_old = [bool(i) for i, e in zip(score["inside_band"], extrap) if not e]
        endpoint_a[str(th)] = {**score, "a_ref_current": a_ref, "base_prediction_at_grid": base.tolist(),
                               "offset_a_ref": a_ref, "extrapolated_flags": extrap,
                               "old_grid_only_supported": bool(all(inside_old)) if inside_old else None,
                               "old_grid_max": old_grid_max,
                               "note": ("endpoint (a): original t20s (c, d), a_ref_current added exactly once; "
                                        "N > 3200 are extrapolations and are excluded from the old-grid claim")}
        if band is not None and any(abs(float(th) - cth) <= 1e-12 for cth in claim_thetas):
            cb = claim_bands["d"][ti]
            if cb is not None:
                cs = AN.score_prediction(pred, cb)
                endpoint_a[str(th)]["claim_alpha"] = alpha_claim
                endpoint_a[str(th)]["claim_supported_at_grid"] = cs["supported_at_grid"]
                endpoint_a[str(th)]["claim_pvalue"] = cs["pvalue"]
                if cs["pvalue"] is not None:
                    claims[f"old_d|theta={th}"] = cs["pvalue"]

    # endpoint (b): the NEW frozen model
    endpoint_b = {}
    for name, s in frozen["new_selection"].items():
        if s["selected"] == "none":
            endpoint_b[name] = {"selected": "none", "scored": False,
                                "note": "no parametric model was frozen; bracket/profile only"}
            continue
        per_theta = {}
        for th_s, fz in s["frozen_per_theta"].items():
            th = float(th_s)
            if not per.get(str(th), {}).get("complete"):
                per_theta[str(th)] = {"scored": False, "reason": "confirmation cells incomplete for this geometry"}
                continue
            band = per[str(th)]["x_U_simultaneous"] if name == "x_U" else per[str(th)]["d_simultaneous"]
            try:
                AN.check_offset_applied_once(params=fz["params"], N=N_grid, offset=float(fz["offset_a_ref"]),
                                             prediction=fz["prediction_at_grid"], where=f"endpoint_b {name} theta={th}")
            except ValueError as e:
                raise ScienceError(str(e)) from e
            score = AN.score_prediction(fz["prediction_at_grid"], band)
            entry = {**score, "params": fz["params"], "offset_a_ref": float(fz["offset_a_ref"])}
            per_theta[str(th)] = entry
            ti = thetas.index(th) if th in thetas else None
            if ti is not None and any(abs(th - cth) <= 1e-12 for cth in claim_thetas):
                cb = claim_bands["x_U" if name == "x_U" else "d"][ti]
                if cb is not None:
                    cs = AN.score_prediction(fz["prediction_at_grid"], cb)
                    entry["claim_alpha"] = alpha_claim
                    entry["claim_supported_at_grid"] = cs["supported_at_grid"]
                    entry["claim_pvalue"] = cs["pvalue"]
                    if cs["pvalue"] is not None:
                        claims[f"new_{name}|theta={th}"] = cs["pvalue"]
        endpoint_b[name] = {"selected": s["selected"], "scored": True, "per_theta": per_theta}
    holm = AN.holm(claims)
    claim_supported: dict[str, bool] = {}
    for th in thetas:
        ea = endpoint_a.get(str(th), {})
        if "claim_supported_at_grid" in ea:
            claim_supported[f"old_d|theta={th}"] = bool(ea["claim_supported_at_grid"])
    for name, d in endpoint_b.items():
        if not d.get("scored"):
            continue
        for t, e in d["per_theta"].items():
            if "claim_supported_at_grid" in e:
                claim_supported[f"new_{name}|theta={t}"] = bool(e["claim_supported_at_grid"])
    return {"endpoint_a_old_prediction": endpoint_a, "endpoint_b_new_model": endpoint_b,
            "multiplicity": {"rule": "per-claim simultaneous band at alpha/m; Holm step-down reported beside it",
                             "m_claims": m_claims, "alpha": float(inf["alpha"]), "alpha_claim": alpha_claim,
                             "claims": list(claims.keys()), "holm": holm,
                             "all_of_rule": "a prediction is supported only if inside at every grid N"},
            "claim_supported": claim_supported}


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    D = design(P, fixture)
    alg_cells = [js for js in cells.values() if js["result"].get("kind") == "algebraic"]
    if alg_cells:
        D["thetas"] = [float(t) for t in alg_cells[0]["params"]["thetas_all"]]
    inf = P["inference"]
    # scientific re-checks on every cell
    alg_by_theta = {}
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "algebraic":
            check_algebraic_result(r)
            alg_by_theta[r["theta_index"]] = r
        elif r["kind"] == "control":
            if not r.get("passed"):
                raise ScienceError(f"control cell {js['cell_id']} failed: {r.get('checks')}")
    fits = []
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "fit":
            check_fit_result(r)
            if r["theta_index"] not in alg_by_theta:
                raise ReconciliationError(f"fit cell {js['cell_id']} has no algebraic reference for theta index "
                                          f"{r['theta_index']}")
            check_reference_consistency(r, alg_by_theta[r["theta_index"]])
            fits.append({**r, "_seed": int(js["seed"]), "_cell_id": js["cell_id"]})
    missing_alg = [ti for ti in range(len(D["thetas"])) if ti not in alg_by_theta]
    if missing_alg:
        raise ReconciliationError(f"algebraic cells missing for theta indices {missing_alg}; cannot reduce")
    phase = "confirmation" if any(f["phase"] == "confirmation" for f in fits) else "pilot"
    planned = {ph: [c["cell_id"] for c in matrix.cells if c["params"].get("phase") == ph] for ph in PHASES}
    present = {ph: sorted({f["_cell_id"] for f in fits if f["phase"] == ph}) for ph in PHASES}
    pilot_complete = set(planned["pilot"]) == set(present["pilot"])
    frozen_path_ = frozen_path(out)
    thetas = D["thetas"]
    summary = {"phase": phase, "fixture": fixture,
               "design": {k: v for k, v in D.items() if k not in ("anchors", "seeds")},
               "grid": {"theta": thetas, "N": D["N_grid"], "old_N": P["grid_catalogue"]["old_N"],
                        "extended_N": P["grid_catalogue"]["extended_N"]},
               "pilot_complete": pilot_complete,
               "confirmation_complete": set(planned["confirmation"]) == set(present["confirmation"]),
               "invariants": {str(thetas[ti]): {k: alg_by_theta[ti][k] for k in
                                                ("eigenvalues_max_abs_dev", "spectrum_fixed", "trace_K",
                                                 "numerical_rank", "snr_per_coordinate", "Delta_k", "k_minus_cap",
                                                 "a_interval", "excluded", "invariant_problems",
                                                 "theta0_control_passed")} for ti in sorted(alg_by_theta)},
               "nonseparability_moves": all(alg_by_theta[ti]["nonseparability_moves_across_grid"]
                                            for ti in alg_by_theta),
               "excluded_thetas": [thetas[ti] for ti in sorted(alg_by_theta) if alg_by_theta[ti]["excluded"]],
               "controls": {"theta0_a_zero": (alg_by_theta[thetas.index(0.0)]["theta0_control_passed"]
                                              if 0.0 in thetas else None),
                            "spectrum_fixed": all(alg_by_theta[ti]["spectrum_fixed"] for ti in alg_by_theta),
                            "bound_order": True, "identity": True, "x_U_nonnegative": True,
                            "training_loss_selection": True, "reference_consistency": True,
                            "offset_applied_once": True,
                            "note": "True means every cell passed the corresponding self-check; a failure raises "
                                    "ScienceError (exit 4) instead of reducing"},
               "old_selection_margins": AN.old_selection_margins(P["old_frozen_prediction"]["cv_errors"]),
               "offset_convention": P["old_frozen_prediction"]["offset_convention"],
               "vocabulary": {"a": "interval [partial-trace lower bound, best-found upper bound]; not certified",
                              "x_F_ref": "reference-relative excess risk; may be negative",
                              "x_U": "excess risk of PCA; exact, >= 0", "d": "paired class contrast"},
               "historical_note": ("new analytic Gaussian family; says nothing about the archived pulse-family "
                                   "proxy (c/N refuted; N* ~ 960 interpolation; 800/1600 reversal stand)")}
    if phase == "pilot":
        if not pilot_complete:
            raise ReconciliationError(f"pilot phase incomplete; refusing to freeze a selection")
        if present["confirmation"]:
            raise ReconciliationError("confirmation cells present while reducing the pilot phase")
        pilot_fits = [f for f in fits if f["phase"] == "pilot"]
        per = _per_theta_summary(pilot_fits, D, alg_by_theta, seed=int(inf["bootstrap_seed"]),
                                 replicates=int(inf["replicates"]), alpha=float(inf["alpha"]),
                                 min_n=int(inf["min_seeds_for_max_t_bootstrap"]))
        sel = _selection(pilot_fits, D, alg_by_theta)
        frozen = _freeze(P, matrix, cells, out, fixture, D, alg_by_theta, sel, per)
        summary.update({"pilot": {"per_theta": per,
                                  "selection": {n: {k: v for k, v in s.items() if k != "cv_detail"}
                                                for n, s in sel.items()},
                                  "selection_cv_detail": {n: s["cv_detail"] for n, s in sel.items()},
                                  "frozen_choice": {n: s["selected"] for n, s in sel.items()}},
                        "frozen_model_file": str(frozen_path_), "frozen_utc": frozen["frozen_utc"],
                        "frozen_choice": {n: s["selected"] for n, s in sel.items()},
                        "partial_note": ("a pilot-phase reduce precedes the confirmation cells by design; the pilot "
                                         "phase itself is complete (pilot_complete)")})
        return summary
    # confirmation
    if not frozen_path_.exists():
        raise ReconciliationError(f"confirmation cells present but {FROZEN_NAME} is missing; cannot score")
    frozen = json.loads(frozen_path_.read_text())
    if frozen.get("protocol_sha256") != P["_sha256"]:
        raise ReconciliationError(f"{FROZEN_NAME} was frozen under a different protocol")
    if not pilot_complete or frozen.get("pilot_digest") != _pilot_digest(cells):
        raise ReconciliationError("the pilot cells in the run root do not match the digest recorded at freeze time")
    conf_fits = [f for f in fits if f["phase"] == "confirmation"]
    per = _per_theta_summary(conf_fits, D, alg_by_theta, seed=int(inf["bootstrap_seed"]) + 1000,
                             replicates=int(inf["replicates"]), alpha=float(inf["alpha"]),
                             min_n=int(inf["min_seeds_for_max_t_bootstrap"]))
    scoring = _scoring(P, frozen, D, alg_by_theta, conf_fits, per, inf=inf)
    summary.update({"confirmation": {**scoring, "per_theta": per,
                                     "frozen_selection": {k: frozen[k] for k in
                                                          ("new_selection", "old_prediction_reference", "pilot_digest",
                                                           "frozen_utc", "offset_convention")},
                                     "partial": bool(partial or not summary["confirmation_complete"])},
                    "frozen_choice": {n: s["selected"] for n, s in frozen["new_selection"].items()}})
    return summary


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# --------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------
def _confirmation_precheck(argv) -> int | None:
    args = cli.build_parser(PACKAGE, DESCRIPTION).parse_args(argv)
    if args.cmd != "run":
        return None
    try:
        P = load_protocol(args.protocol)
    except (ProtocolError, OSError):
        return None
    if P["package"] != PACKAGE:
        return None
    fixture = bool(args.fixture)
    matrix = build_matrix(P, fixture)
    sub = matrix.subset(roles=["development", "control", "algebraic"]) if fixture else matrix
    if args.roles or args.cells or args.families:
        try:
            sub = sub.subset(roles=args.roles, cell_ids=args.cells, families=args.families)
        except ValueError:
            return None
    wants_conf = [c for c in sub.cells if c["params"].get("phase") == "confirmation"]
    if not wants_conf:
        return None
    frozen = Path(args.out) / FROZEN_NAME
    if not frozen.exists():
        print(f"error: {len(wants_conf)} confirmation cell(s) selected but {frozen} does not exist; run and reduce "
              f"the pilot phase first (nothing was charged)", file=sys.stderr)
        return 6
    fz = json.loads(frozen.read_text())
    if fz.get("protocol_sha256") != P["_sha256"]:
        print(f"error: {frozen} was frozen under a different protocol (nothing was charged)", file=sys.stderr)
        return 6
    if fz.get("matrix_sha256") != matrix.matrix_hash():
        print(f"error: {frozen} was frozen for a different run matrix (nothing was charged)", file=sys.stderr)
        return 6
    # the pilot-content binding is checked HERE, before any confirmation draw or
    # charge, not only at reduction.  The freeze must match the pilot cells on disk.
    try:
        observed = read_cells(Path(args.out))
        pilot_cells = {cid: js for cid, files in observed.items() for _, js in files}
    except Exception:
        pilot_cells = {}
    if not pilot_cells or not any(js["result"].get("phase") == "pilot" for js in pilot_cells.values()):
        print(f"error: {frozen} exists but no pilot cells were found in {args.out}; refusing to draw confirmation "
              f"(nothing was charged)", file=sys.stderr)
        return 6
    if fz.get("pilot_digest") != _pilot_digest(pilot_cells):
        print(f"error: {frozen} pilot digest does not match the pilot cells on disk (the pilot content changed "
              f"after the freeze); refusing (nothing was charged)", file=sys.stderr)
        return 6
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "truth-audit":
        import argparse
        from . import truth_audit as ta
        ap = argparse.ArgumentParser(prog="r2_t20_theta012 truth-audit")
        ap.add_argument("--protocol", required=True, type=Path)
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--production", action="store_true", help="audit the production design (slow)")
        a = ap.parse_args(argv[1:])
        P = load_protocol(a.protocol)
        rec = ta.write_receipt(P, a.out, fixture=not a.production)
        print(json.dumps({"all_passed": rec["all_passed"], "checks": {k: v["passed"] for k, v in rec["checks"].items()},
                          "out": str(a.out)}, indent=1))
        return 0 if rec["all_passed"] else 4
    rc = _confirmation_precheck(argv)
    if rc is not None:
        return rc
    return cli.main_template(package=PACKAGE, description=DESCRIPTION, build_matrix=build_matrix, run_cell=run_cell,
                             reduce_run=reduce_run, result_required=RESULT_REQUIRED, source_files=SOURCE_FILES,
                             fixture_overrides=fixture_overrides, argv=argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
