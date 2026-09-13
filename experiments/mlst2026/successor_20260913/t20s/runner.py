"""T20-S runner (work package L2) on the shared local-package skeleton.

    python -m experiments.mlst2026.successor_20260913.t20s validate  --protocol P
    python -m experiments.mlst2026.successor_20260913.t20s enumerate --protocol P --out RUN
    python -m experiments.mlst2026.successor_20260913.t20s run       --protocol P --out RUN --families pilot algebraic
    python -m experiments.mlst2026.successor_20260913.t20s reduce    --protocol P --out RUN --allow-partial   (pilot: writes FROZEN_SELECTION.json)
    python -m experiments.mlst2026.successor_20260913.t20s run       --protocol P --out RUN --families confirmation
    python -m experiments.mlst2026.successor_20260913.t20s reduce    --protocol P --out RUN                    (confirmation: scores the frozen choice)
    python -m experiments.mlst2026.successor_20260913.t20s estimate  --protocol P --out RUN

Cells
-----
* ``algebraic`` (one per theta, seed None): invariants of K(theta), the
  partial-trace lower bound and the population best-found a.  Solves only.
* fit cells (family ``pilot`` / ``confirmation``; role = family in production,
  ``development`` in a fixture): one paired dataset, top-k PCA (1 fit) and
  separable ALS with R restarts (R fits), every quantity of the risk identity.

Scientific self-checks (:func:`check_fit_result`, :func:`check_algebraic_result`)
run at completion and again at reduce; a violation is a :class:`ScienceError`
(exit 4), independent of the byte-digest check.

The confirmation phase refuses to draw any confirmation cell unless
``FROZEN_SELECTION.json`` exists in the same run root (checked BEFORE any
attempt is charged, and again inside the cell).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..common import cli
from ..common import fitcount
from ..common.matrix import RunMatrix, make_cell
from ..common.protocol import ProtocolError, canonical_json, load_protocol
from ..common.results import ReconciliationError, SchemaError, write_json_atomic
from . import analysis as AN
from . import generator as GEN

PACKAGE = "t20s"
DESCRIPTION = "T20-S v2: fixed-spectrum geometry knob, optimality-gap intervals, paired learner contrasts, pilot-frozen scaling selection"
HERE = Path(__file__).resolve().parent
SOURCE_FILES = [HERE / "runner.py", HERE / "generator.py", HERE / "analysis.py", HERE / "PROTOCOL_T20S_v2.json",
                cli.REPO_ROOT / "experiments" / "paper1" / "t20_population_accounting.py",
                cli.REPO_ROOT / "experiments" / "synthetic" / "nfpa_confirmatory" / "methods.py",
                cli.REPO_ROOT / "experiments" / "synthetic" / "nfpa_confirmatory" / "generator.py"]
FROZEN_NAME = "FROZEN_SELECTION.json"
TOL = 1e-9
PHASES = ("pilot", "confirmation")


class ScienceError(SchemaError):
    """A scientific self-check failed (exit 4)."""


# --------------------------------------------------------------------------------------
# design
# --------------------------------------------------------------------------------------
def design(P: dict, fixture: bool) -> dict:
    D = dict(P["fixture"] if fixture else P["design"]["production"])
    D["k"] = int(D["k_c"]) * int(D["k_t"])
    D["d"] = int(D["C"]) * int(D["T"])
    D["N_grid"] = sorted(int(n) for n in list(D.get("N_core", [])) + list(D.get("N_anchor", [])))
    D["anchors"] = set(int(n) for n in D.get("N_anchor", []))
    D["thetas"] = [float(t) for t in D["thetas"]]
    D["seeds"] = {"pilot": list(range(int(D["pilot_seeds"][0]), int(D["pilot_seeds"][1]) + 1)),
                  "confirmation": list(range(int(D["confirmation_seeds"][0]), int(D["confirmation_seeds"][1]) + 1))}
    D["fixture"] = bool(fixture)
    return D


def _geometry(D: dict) -> GEN.Geometry:
    return GEN.make_geometry(C=D["C"], T=D["T"], k_c=D["k_c"], k_t=D["k_t"], spec=D["spectrum"], seed=D["geometry_seed"])


def _reference(D: dict, theta: float) -> dict:
    return GEN.population_reference(C=D["C"], T=D["T"], k_c=D["k_c"], k_t=D["k_t"], spec=D["spectrum"], seed=D["geometry_seed"],
                                    theta=theta, R_pop=D["R_pop"], n_iter=D["pop_als_n_iter"], tol=D["pop_als_tol"])


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    D = design(P, fixture)
    cells = []
    base = {"C": D["C"], "T": D["T"], "k_c": D["k_c"], "k_t": D["k_t"], "spectrum": D["spectrum"], "geometry_seed": D["geometry_seed"],
            "R_pop": D["R_pop"], "pop_als_n_iter": D["pop_als_n_iter"], "pop_als_tol": D["pop_als_tol"], "fixture": bool(fixture)}
    for ti, th in enumerate(D["thetas"]):
        cells.append(make_cell(role="algebraic", seed=None, family="algebraic", n_fits=0, n_solves=1 + (D["R_pop"] + 1) + 1,
                               params={**base, "kind": "algebraic", "theta": th, "theta_index": ti, "thetas_all": D["thetas"],
                                       "control": "theta0_a_zero" if th == 0.0 else None},
                               notes="invariants, partial-trace lower bound, population best-found a (solves only)"))
    for phase in PHASES:
        role = "development" if fixture else phase
        for seed in D["seeds"][phase]:
            for ti, th in enumerate(D["thetas"]):
                for ni, N in enumerate(D["N_grid"]):
                    params = {**base, "kind": "fit", "phase": phase, "theta": th, "theta_index": ti, "N": int(N), "N_index": ni,
                              "band": "anchor" if N in D["anchors"] else "core", "R": int(D["R"]), "als_n_iter": int(D["als_n_iter"]),
                              "als_tol": float(D["als_tol"]), "control": "theta0_a_zero" if th == 0.0 else None}
                    cells.append(make_cell(role=role, seed=int(seed), family=phase, n_fits=1 + int(D["R"]), n_solves=0, params=params))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


def fixture_overrides(P: dict) -> dict:
    return {k: v for k, v in P["fixture"].items() if k != "note"}


# --------------------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------------------
def _ref_public(ref: dict) -> dict:
    return {k: v for k, v in ref.items() if k not in ("B_U_star", "B_F_best", "K")}


def run_algebraic_cell(P: dict, cell: dict, D: dict) -> dict:
    p = cell["params"]
    GEN._population_reference_cached.cache_clear()   # the algebraic cell pays for its own solves (honest wall time)
    g = _geometry(D)
    inv = GEN.invariants(g, D["thetas"], R_pop=D["R_pop"], n_iter=D["pop_als_n_iter"], tol=D["pop_als_tol"], spec=D["spectrum"])
    ref = _ref_public(_reference(D, p["theta"]))
    th = float(p["theta"])
    excluded = th in inv["excluded_thetas"]
    res = {"kind": "algebraic", "theta": th, "theta_index": int(p["theta_index"]),
           "eigenvalues_max_abs_dev": ref["eigenvalues_max_abs_dev"], "eig_tol": GEN.EIG_TOL,
           "trace_K": ref["trace_K"], "trace_mu": ref["trace_mu"], "numerical_rank": ref["numerical_rank"], "d": ref["d"], "k": ref["k"],
           "snr_per_coordinate": ref["snr_per_coordinate"], "Delta_k": ref["Delta_k"],
           "partial_trace": ref["partial_trace"], "k_minus_cap": ref["partial_trace"]["k_minus_cap"],
           "a_lower_bound": ref["a_lower_bound"], "a_best_found": ref["a_best_found"],
           "a_interval": [ref["a_lower_bound"], ref["a_best_found"]],
           "R_U_star": ref["R_U_star"], "R_F_best_found": ref["R_F_best_found"], "pop_als": ref["pop_als"],
           "is_theta0_control": bool(th == 0.0),
           "theta0_control_passed": (bool(abs(ref["a_lower_bound"]) <= TOL and ref["a_best_found"] <= 1e-8) if th == 0.0 else None),
           "spectrum_fixed": bool(ref["eigenvalues_max_abs_dev"] <= GEN.EIG_TOL),
           "nonseparability_moves_across_grid": bool(inv["nonseparability_moves"]),
           "k_minus_cap_by_theta": {str(t): inv["per_theta"][t]["partial_trace"]["k_minus_cap"] for t in D["thetas"]},
           "a_interval_by_theta": {str(t): [inv["per_theta"][t]["a_lower_bound"], inv["per_theta"][t]["a_best_found"]] for t in D["thetas"]},
           "invariant_problems": inv["problems"][str(th)], "excluded": bool(excluded),
           "excluded_thetas_across_grid": [float(t) for t in inv["excluded_thetas"]],
           "solves": {"eigendecomposition": 1, "population_als_inits": D["R_pop"] + 1, "bound": 1},
           "note": "population reference for this theta; a_best_found is an UPPER bound on a (best of restarts), a_lower_bound is the partial-trace bound; neither certifies P*_F"}
    res["checks_passed"] = True
    check_algebraic_result(res)
    return res


def select_restart(losses: list[float]) -> int:
    """Training-loss restart selection (first minimum).  Mirrors ``als_fit``;
    the cell records both and the self-check requires agreement."""
    return int(np.argmin(np.asarray(losses, float)))


def run_fit_cell(P: dict, cell: dict, D: dict, ctx: dict) -> tuple[dict, int]:
    from experiments.synthetic.nfpa_confirmatory.methods import als_fit  # audited, unchanged
    p = cell["params"]
    th, N, R = float(p["theta"]), int(p["N"]), int(p["R"])
    C, T, k_c, k_t, k = D["C"], D["T"], D["k_c"], D["k_t"], D["k"]
    if p["phase"] == "confirmation" and not (Path(ctx["out"]) / FROZEN_NAME).exists():
        raise cli.UsageError(f"confirmation cell {cell['cell_id']} refused: {FROZEN_NAME} is not in the run root {ctx['out']}")
    g = _geometry(D)
    ref = _reference(D, th)
    K = ref["K"]
    phase_code = PHASES.index(p["phase"])
    Z = GEN.draw_data(g, th, N, [int(cell["seed"]), int(p["theta_index"]), N, phase_code, int(D["geometry_seed"])])
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # learner U: top-k PCA of the centred sample second moment (closed form; 1 fit)
    S = Zc.T @ Zc / N
    fitcount.count_fit("pca")
    w, U = np.linalg.eigh(S)                 # one estimator invocation on drawn data
    order = np.argsort(w)[::-1]
    B_U = U[:, order[:k]]
    pca_train_residual = float(np.trace(S) - np.sum(w[order[:k]]))
    # learner F: separable ALS, R restarts, training-loss selection (R fits)
    als_seed = int(cell["seed"]) * 100 + int(p["theta_index"]) * 10 + int(p["N_index"])
    out = als_fit(Zc.reshape(N, C, T, order="F"), k_c, k_t, n_iter=int(p["als_n_iter"]), n_restarts=R, seed=als_seed,
                  tol=float(p["als_tol"]), collect_restarts=True)
    # ``als_fit`` is an audited historical module and is not edited; the restarts it
    # actually performed are counted from what it returned, one fit per restart.
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
    res = {"kind": "fit", "phase": p["phase"], "theta": th, "theta_index": int(p["theta_index"]), "N": N, "N_index": int(p["N_index"]),
           "band": p["band"], "R": R, "C": C, "T": T, "k_c": k_c, "k_t": k_t, "k": k,
           "R_U_hat": R_U_hat, "R_F_hat": R_F_hat, "d": float(d), "x_U": float(x_U), "x_F_ref": float(x_F_ref),
           "a_lower_bound": float(ref["a_lower_bound"]), "a_best_found": float(a_best), "a_ref": float(a_best),
           "a_interval": [float(ref["a_lower_bound"]), float(a_best)],
           "identity_residual": float(d - (a_best + x_F_ref - x_U)),
           "R_U_star": float(ref["R_U_star"]), "R_F_best_found": float(ref["R_F_best_found"]), "Delta_k": float(ref["Delta_k"]),
           "eigenvalues_max_abs_dev": float(ref["eigenvalues_max_abs_dev"]),
           "restart_train_losses": losses, "restart_pop_risks": risks, "restart_converged": [bool(r["converged"]) for r in out["restarts"]],
           "selected_restart": sel, "als_selected_restart": int(out["restart"]), "selected_by": "training_loss",
           "population_best_restart": pop_best, "selection_matches_population_best": bool(sel == pop_best),
           "restart_spread_loss": float(max(losses) - min(losses)), "restart_spread_risk": spread_risk,
           "optimization_unresolved": bool(spread_risk >= abs(d)),
           "pca_train_residual": pca_train_residual, "als_train_loss": float(losses[sel]),
           "fits": {"pca": 1, "als_restarts": len(out["restarts"])}, "fits_observed": fitcount.observed(),
           "checks_passed": True}
    check_fit_result(res)
    return res, fitcount.observed()


def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    D = design(P, fixture)
    if cell["role"] == "algebraic":
        res = run_algebraic_cell(P, cell, D)
        return res, 0, cell["n_solves"]
    res, fits = run_fit_cell(P, cell, D, ctx)
    return res, fits, 0


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
        raise ScienceError(f"{where}: lower bound {r['a_lower_bound']} exceeds best-found a {r['a_best_found']} (impossible for a valid bound)")
    if r["a_best_found"] < -TOL:
        raise ScienceError(f"{where}: best-found a negative {r['a_best_found']}")
    if abs(r["Delta_k"] * r["k_minus_cap"] - r["a_lower_bound"]) > TOL and r["theta"] != 0.0:
        raise ScienceError(f"{where}: lower bound is not Delta_k (k - cap)")
    if r["theta"] == 0.0 and not (abs(r["a_lower_bound"]) <= TOL and r["a_best_found"] <= 1e-8):
        raise ScienceError(f"{where}: theta = 0 control failed (a_lower={r['a_lower_bound']}, a_best={r['a_best_found']})")
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
        raise ScienceError(f"{where}: selected restart {r['selected_restart']} is not the training-loss minimiser {int(np.argmin(losses))}")
    if abs(r["identity_residual"]) > TOL:
        raise ScienceError(f"{where}: identity residual {r['identity_residual']:.2e} exceeds {TOL}")
    if abs(r["d"] - (r["R_F_hat"] - r["R_U_hat"])) > TOL:
        raise ScienceError(f"{where}: d is not R_F_hat - R_U_hat")
    if r["x_U"] < -TOL:
        raise ScienceError(f"{where}: x_U = {r['x_U']} < 0 is impossible (P*_U is the global minimiser)")
    if abs(r["x_U"] - (r["R_U_hat"] - r["R_U_star"])) > TOL:
        raise ScienceError(f"{where}: x_U is not R_U_hat - R_U_star")
    if abs(r["x_F_ref"] - (r["R_F_hat"] - r["R_F_best_found"])) > TOL:
        raise ScienceError(f"{where}: x_F_ref is not R_F_hat - R_F_best_found")
    if r["a_lower_bound"] > r["a_best_found"] + TOL or r["a_lower_bound"] < -TOL or r["a_best_found"] < -TOL:
        raise ScienceError(f"{where}: a interval [{r['a_lower_bound']}, {r['a_best_found']}] is not a valid interval")
    if abs(r["a_best_found"] - (r["R_F_best_found"] - r["R_U_star"])) > TOL:
        raise ScienceError(f"{where}: a_best_found is not R_F_best_found - R_U_star")
    if r["theta"] == 0.0 and not (abs(r["a_lower_bound"]) <= TOL and r["a_best_found"] <= 1e-8):
        raise ScienceError(f"{where}: theta = 0 control failed")
    sel_risk = r["restart_pop_risks"][r["selected_restart"]]
    if abs((sel_risk - r["R_U_star"]) - (r["a_best_found"] + r["x_F_ref"])) > 1e-6 * max(1.0, abs(sel_risk)):
        raise ScienceError(f"{where}: selected restart's recorded population risk is inconsistent with R_F_hat")
    if r["restart_spread_risk"] < -TOL or abs(r["restart_spread_risk"] - (max(r["restart_pop_risks"]) - min(r["restart_pop_risks"]))) > TOL:
        raise ScienceError(f"{where}: restart spread inconsistent with the recorded restart risks")
    if r["optimization_unresolved"] != bool(r["restart_spread_risk"] >= abs(r["d"])):
        raise ScienceError(f"{where}: optimization_unresolved flag inconsistent")
    if r["eigenvalues_max_abs_dev"] > GEN.EIG_TOL:
        raise ScienceError(f"{where}: spectrum not fixed")


def check_reference_consistency(fit: dict, alg: dict) -> None:
    for key in ("a_lower_bound", "a_best_found", "R_U_star", "R_F_best_found", "Delta_k"):
        if abs(float(fit[key]) - float(alg[key])) > TOL:
            raise ScienceError(f"fit cell theta={fit['theta']} N={fit['N']}: {key}={fit[key]} differs from the algebraic cell's {alg[key]}")


RESULT_REQUIRED = {"kind": str, "theta": float, "theta_index": int, "a_lower_bound": float, "a_best_found": float, "a_interval": list,
                   "Delta_k": float, "eigenvalues_max_abs_dev": float, "R_U_star": float, "R_F_best_found": float, "checks_passed": bool}


# --------------------------------------------------------------------------------------
# reduce
# --------------------------------------------------------------------------------------
def _pilot_digest(cells: dict) -> str:
    items = sorted((cid, hashlib.sha256(canonical_json(js["result"]).encode()).hexdigest())
                   for cid, js in cells.items() if js["result"].get("kind") == "fit" and js["result"]["phase"] == "pilot")
    return hashlib.sha256(canonical_json(items).encode()).hexdigest()


def _table(fits: list[dict], thetas: list[float], N_grid: list[int], key: str) -> dict[int, tuple[list[int], np.ndarray]]:
    """theta_index -> (seeds, array (n_seeds, n_N)) for result key ``key``."""
    out = {}
    for ti, th in enumerate(thetas):
        rows = [f for f in fits if f["theta_index"] == ti]
        seeds = sorted({f["_seed"] for f in rows})
        M = np.full((len(seeds), len(N_grid)), np.nan)
        for f in rows:
            M[seeds.index(f["_seed"]), N_grid.index(f["N"])] = f[key]
        out[ti] = (seeds, M)
    return out


def _per_theta_summary(fits: list[dict], D: dict, alg_by_theta: dict, *, seed: int, replicates: int, alpha: float, min_n: int) -> dict:
    thetas, N_grid = D["thetas"], D["N_grid"]
    per = {}
    tab_d = _table(fits, thetas, N_grid, "d")
    tab_xu = _table(fits, thetas, N_grid, "x_U")
    tab_xf = _table(fits, thetas, N_grid, "x_F_ref")
    for ti, th in enumerate(thetas):
        seeds, Md = tab_d[ti]
        _, Mxu = tab_xu[ti]
        _, Mxf = tab_xf[ti]
        rows = [f for f in fits if f["theta_index"] == ti]
        if not rows or np.any(np.isnan(Md)):
            per[str(th)] = {"complete": False, "n_seeds": len(seeds)}
            continue
        ci_d = AN.simultaneous_ci(Md, seed=seed + ti, replicates=replicates, alpha=alpha, min_n=min_n)
        ci_xu = AN.simultaneous_ci(Mxu, seed=seed + 100 + ti, replicates=replicates, alpha=alpha, min_n=min_n)
        prof = AN.sign_profile(ci_d)
        by_N = {}
        for ni, N in enumerate(N_grid):
            rN = [f for f in rows if f["N"] == N]
            unres = float(np.mean([f["optimization_unresolved"] for f in rN]))
            by_N[str(N)] = {"n": len(rN), "d_mean": float(np.mean([f["d"] for f in rN])), "d_sim_ci": [ci_d["lower"][ni], ci_d["upper"][ni]],
                            "d_sign": prof["signs"][ni] if prof["signs"] else None,
                            "x_U_mean": float(np.mean([f["x_U"] for f in rN])), "x_F_ref_mean": float(np.mean([f["x_F_ref"] for f in rN])),
                            "R_U_hat_mean": float(np.mean([f["R_U_hat"] for f in rN])), "R_F_hat_mean": float(np.mean([f["R_F_hat"] for f in rN])),
                            "restart_spread_risk_median": float(np.median([f["restart_spread_risk"] for f in rN])),
                            "restart_spread_loss_median": float(np.median([f["restart_spread_loss"] for f in rN])),
                            "selection_matches_population_best_frac": float(np.mean([f["selection_matches_population_best"] for f in rN])),
                            "optimization_unresolved_frac": unres, "sampling_vs_optimization_unresolved": bool(unres > 0.5),
                            "als_converged_frac": float(np.mean([np.mean(f["restart_converged"]) for f in rN]))}
        alg = alg_by_theta[ti]
        per[str(th)] = {"complete": True, "theta_index": ti, "n_seeds": len(seeds), "seeds": seeds, "excluded": bool(alg["excluded"]),
                        "a_interval": alg["a_interval"], "k_minus_cap": alg["k_minus_cap"], "Delta_k": alg["Delta_k"],
                        "d_simultaneous": ci_d, "x_U_simultaneous": ci_xu, "sign_profile": prof, "by_N": by_N,
                        "adjacent_sign_brackets_N": [[N_grid[i], N_grid[i + 1]] for i in prof["adjacent_brackets"]],
                        "max_identity_residual": float(max(abs(f["identity_residual"]) for f in rows)),
                        "min_x_U": float(min(f["x_U"] for f in rows)),
                        "control_theta0": ({"a_lower_bound": alg["a_lower_bound"], "a_best_found": alg["a_best_found"], "passed": alg["theta0_control_passed"],
                                            "d_profile": prof["profile"]} if th == 0.0 else None)}
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
        sel = AN.cv_select(Y, N_grid, seeds)
        frozen = {}
        if sel["selected"] != "none":
            for ti in usable:
                params = AN.fit_model(sel["selected"], np.asarray(N_grid, float), Y[ti].mean(axis=0))
                pred = AN.predict(params, np.asarray(N_grid, float))
                if name == "d_minus_a_ref":
                    pred = pred + alg_by_theta[ti]["a_best_found"]
                frozen[str(thetas[ti])] = {"params": params, "prediction_at_grid": pred.tolist(),
                                           "offset_a_ref": alg_by_theta[ti]["a_best_found"] if name == "d_minus_a_ref" else 0.0}
        out[name] = {"selected": sel["selected"], "cv": {k: v for k, v in sel.items() if k != "detail"}, "cv_detail": sel.get("detail", {}),
                     "frozen_per_theta": frozen, "usable_theta_indices": usable}
    return out


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    fixture = any(js.get("development_only") for js in cells.values())
    D = design(P, fixture)
    inf = P["inference"]
    # scientific re-checks on every cell (independent of the digest check)
    alg_by_theta = {}
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "algebraic":
            check_algebraic_result(r)
            alg_by_theta[r["theta_index"]] = r
    fits = []
    for js in cells.values():
        r = js["result"]
        if r["kind"] == "fit":
            check_fit_result(r)
            if r["theta_index"] not in alg_by_theta:
                raise ReconciliationError(f"fit cell {js['cell_id']} has no algebraic reference cell for theta index {r['theta_index']}")
            check_reference_consistency(r, alg_by_theta[r["theta_index"]])
            fits.append({**r, "_seed": int(js["seed"]), "_cell_id": js["cell_id"]})
    missing_alg = [ti for ti in range(len(D["thetas"])) if ti not in alg_by_theta]
    if missing_alg:
        raise ReconciliationError(f"algebraic cells missing for theta indices {missing_alg}; cannot reduce")
    phase = "confirmation" if any(f["phase"] == "confirmation" for f in fits) else "pilot"
    planned = {ph: [c["cell_id"] for c in matrix.cells if c["params"].get("phase") == ph] for ph in PHASES}
    present = {ph: sorted({f["_cell_id"] for f in fits if f["phase"] == ph}) for ph in PHASES}
    pilot_complete = set(planned["pilot"]) == set(present["pilot"])
    frozen_path = Path(out) / FROZEN_NAME
    thetas = D["thetas"]
    summary = {"phase": phase, "fixture": fixture, "design": {k: v for k, v in D.items() if k not in ("anchors", "seeds")},
               "grid": {"theta": thetas, "N": D["N_grid"]}, "pilot_complete": pilot_complete,
               "confirmation_complete": set(planned["confirmation"]) == set(present["confirmation"]),
               "invariants": {str(thetas[ti]): {k: alg_by_theta[ti][k] for k in ("eigenvalues_max_abs_dev", "spectrum_fixed", "trace_K", "numerical_rank",
                                                                              "snr_per_coordinate", "Delta_k", "k_minus_cap", "a_interval", "excluded",
                                                                              "invariant_problems", "theta0_control_passed")}
                              for ti in sorted(alg_by_theta)},
               "nonseparability_moves": all(alg_by_theta[ti]["nonseparability_moves_across_grid"] for ti in alg_by_theta),
               "excluded_thetas": [thetas[ti] for ti in sorted(alg_by_theta) if alg_by_theta[ti]["excluded"]],
               "controls": {"theta0_a_zero": alg_by_theta[thetas.index(0.0)]["theta0_control_passed"] if 0.0 in thetas else None,
                            "spectrum_fixed": all(alg_by_theta[ti]["spectrum_fixed"] for ti in alg_by_theta),
                            "bound_order": True, "identity": True, "x_U_nonnegative": True, "training_loss_selection": True,
                            "reference_consistency": True,
                            "note": "True means every cell passed the corresponding self-check; a failure raises ScienceError (exit 4) instead of reducing"},
               "vocabulary": {"a": "interval [partial-trace lower bound, best-found upper bound]; not certified",
                              "x_F_ref": "reference-relative excess risk (estimation + optimization); may be negative",
                              "x_U": "excess risk of PCA; exact, >= 0", "d": "paired class contrast"},
               "historical_note": "new analytic Gaussian family; says nothing about the archived pulse-family proxy (c/N refuted; N* ~ 960 interpolation; 800/1600 reversal stand)"}
    if phase == "pilot":
        if not pilot_complete:
            raise ReconciliationError(f"pilot phase incomplete ({len(present['pilot'])}/{len(planned['pilot'])} cells); refusing to freeze a selection")
        if present["confirmation"]:
            raise ReconciliationError("confirmation cells present while reducing the pilot phase")
        pilot_fits = [f for f in fits if f["phase"] == "pilot"]
        per = _per_theta_summary(pilot_fits, D, alg_by_theta, seed=int(inf["bootstrap_seed"]), replicates=int(inf["replicates"]), alpha=float(inf["alpha"]),
                                 min_n=int(inf["min_seeds_for_max_t_bootstrap"]))
        sel = _selection(pilot_fits, D, alg_by_theta)
        frozen = {"schema": "t20s-frozen-selection/1", "package": PACKAGE, "protocol_sha256": P["_sha256"], "matrix_sha256": matrix.matrix_hash(),
                  "development_only": fixture, "phase_frozen_from": "pilot", "pilot_cells": len(pilot_fits), "pilot_seeds": sorted({f["_seed"] for f in pilot_fits}),
                  "pilot_digest": _pilot_digest(cells), "confirmation_cells_present_at_freeze": 0,
                  "grid": {"theta": thetas, "N": D["N_grid"]}, "candidates": list(AN.CANDIDATES),
                  "algorithm": inf["scaling_selection"]["algorithm"],
                  "selection": {name: {"selected": s["selected"], "cv_error": s["cv"]["cv_error"], "reason": s["cv"].get("reason"),
                                       "fit_half": s["cv"].get("fit_half"), "holdout_half": s["cv"].get("holdout_half"),
                                       "frozen_per_theta": s["frozen_per_theta"], "usable_theta_indices": s["usable_theta_indices"]}
                                for name, s in sel.items()},
                  "excluded_thetas": summary["excluded_thetas"]}
        if frozen_path.exists():
            prev = json.loads(frozen_path.read_text())
            prev_cmp = {k: v for k, v in prev.items() if k != "frozen_utc"}
            if canonical_json(prev_cmp) != canonical_json(json.loads(json.dumps(frozen, default=_js))):
                raise ReconciliationError(f"{frozen_path} exists and differs from the selection the pilot cells now imply; a frozen selection is never overwritten")
            frozen_utc = prev.get("frozen_utc")
        else:
            frozen_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_json_atomic(frozen_path, {**frozen, "frozen_utc": frozen_utc})
        summary.update({"pilot": {"per_theta": per, "selection": {n: {k: v for k, v in s.items() if k != "cv_detail"} for n, s in sel.items()},
                                  "selection_cv_detail": {n: s["cv_detail"] for n, s in sel.items()}},
                        "frozen_selection_file": str(frozen_path), "frozen_utc": frozen_utc,
                        "frozen_choice": {n: s["selected"] for n, s in sel.items()},
                        "partial_note": "a pilot-phase reduce precedes the confirmation cells by design; 'partial' refers to the confirmation "
                                        "cells not yet drawn, while the pilot phase itself is complete (pilot_complete)"})
        return summary
    # confirmation phase
    if not frozen_path.exists():
        raise ReconciliationError(f"confirmation cells present but {FROZEN_NAME} is missing from the run root; confirmation cannot be scored")
    frozen = json.loads(frozen_path.read_text())
    if frozen.get("protocol_sha256") != P["_sha256"]:
        raise ReconciliationError("FROZEN_SELECTION.json was frozen under a different protocol")
    if not pilot_complete or frozen.get("pilot_digest") != _pilot_digest(cells):
        raise ReconciliationError("the pilot cells in the run root do not match the digest recorded at freeze time")
    conf_fits = [f for f in fits if f["phase"] == "confirmation"]
    per = _per_theta_summary(conf_fits, D, alg_by_theta, seed=int(inf["bootstrap_seed"]) + 1000, replicates=int(inf["replicates"]), alpha=float(inf["alpha"]),
                             min_n=int(inf["min_seeds_for_max_t_bootstrap"]))
    scoring = {}
    for name, s in frozen["selection"].items():
        if s["selected"] == "none":
            scoring[name] = {"selected": "none", "scored": False, "note": "no parametric model was frozen; bracket/profile only"}
            continue
        per_theta = {}
        for th_s, fz in s["frozen_per_theta"].items():
            p = per.get(th_s)
            if not p or not p.get("complete"):
                per_theta[th_s] = {"scored": False, "reason": "confirmation cells incomplete for this theta"}
                continue
            ci = p["d_simultaneous"] if name == "d_minus_a_ref" else p["x_U_simultaneous"]
            per_theta[th_s] = AN.score_frozen_prediction(np.asarray(fz["prediction_at_grid"]), ci)
        scoring[name] = {"selected": s["selected"], "scored": True, "per_theta": per_theta,
                         "supported_at_grid_all_thetas": all(v.get("supported_at_grid", False) for v in per_theta.values())}
    summary.update({"confirmation": {"per_theta": per, "frozen_selection": {k: frozen[k] for k in ("selection", "pilot_digest", "frozen_utc", "excluded_thetas")},
                                     "scoring": scoring, "partial": bool(partial or not summary["confirmation_complete"])},
                    "frozen_choice": {n: s["selected"] for n, s in frozen["selection"].items()}})
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
# main: confirmation refusal BEFORE any attempt is charged, then the common template
# --------------------------------------------------------------------------------------
def _confirmation_precheck(argv) -> int | None:
    args = cli.build_parser(PACKAGE, DESCRIPTION).parse_args(argv)
    if args.cmd != "run":
        return None
    try:
        P = load_protocol(args.protocol)
    except (ProtocolError, OSError):
        return None  # the template reports protocol problems
    if P["package"] != PACKAGE:
        return None  # the template reports the protocol/package mismatch
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
        print(f"error: {len(wants_conf)} confirmation cell(s) selected but {frozen} does not exist; run and reduce the pilot phase first "
              f"(nothing was charged)", file=sys.stderr)
        return 6
    fz = json.loads(frozen.read_text())
    if fz.get("protocol_sha256") != P["_sha256"]:
        print(f"error: {frozen} was frozen under a different protocol (nothing was charged)", file=sys.stderr)
        return 6
    if fz.get("matrix_sha256") != matrix.matrix_hash():
        print(f"error: {frozen} was frozen for a different run matrix (nothing was charged)", file=sys.stderr)
        return 6
    return None


def _project_estimate_to_production(argv) -> None:
    """The common ``estimate`` multiplies the scaled per-cell timings by the matrix
    of the run root; for a fixture root that is the fixture matrix.  Append the
    projection onto the FULL production matrix (README: 'measured fixture timings
    x declared scale x full matrix') without touching the common output."""
    args = cli.build_parser(PACKAGE, DESCRIPTION).parse_args(argv)
    path = Path(args.out) / "RESOURCE_ESTIMATE.json"
    if not path.exists():
        return
    P = load_protocol(args.protocol)
    est = json.loads(path.read_text())
    prod = cli._production_only(build_matrix(P, False)).counts()
    proj = {"families": {}, "total_cpu_seconds_point": 0.0, "unmeasured_families": [], "note":
            "production matrix counts x production_seconds_per_cell (fixture timing x declared scale); point estimate, no variance; "
            "single-threaded BLAS as measured; failures/retries add up to retry_allowance; role 'development' is excluded, so the "
            "total and 'production_counts' cover the same cells"}
    for fam, info in prod["by_family"].items():
        e = est["families"].get(fam)
        if not e or not e.get("measured"):
            proj["unmeasured_families"].append(fam)
            proj["families"][fam] = {"cells": info["cells"], "measured": False}
            continue
        per = float(e["production_seconds_per_cell"])
        proj["families"][fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "production_seconds_per_cell": per,
                                 "production_cpu_seconds": per * info["cells"]}
        proj["total_cpu_seconds_point"] += per * info["cells"]
    proj["total_cpu_hours_point"] = proj["total_cpu_seconds_point"] / 3600.0
    proj["production_counts"] = prod
    proj["retry_allowance"] = {"policy": P["budget"].get("retry_policy", "unspecified"),
                               "fits_headroom_under_cap": P["budget"]["estimator_fits_cap"] - prod["fits"]}
    est["production_projection"] = proj
    est["estimate_matrix_note"] = ("'families' and 'total_cpu_seconds_point' above refer to the matrix of this run root "
                                   f"({'fixture' if json.loads((Path(args.out) / 'run_matrix_summary.json').read_text()).get('fixture') else 'production'}); "
                                   "'production_projection' is the full production matrix")
    write_json_atomic(path, est)
    print(json.dumps({"production_projection": {k: proj[k] for k in ("total_cpu_seconds_point", "total_cpu_hours_point", "retry_allowance", "unmeasured_families")}}, indent=1))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rc = _confirmation_precheck(argv)
    if rc is not None:
        return rc
    rc = cli.main_template(package=PACKAGE, description=DESCRIPTION, build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                           result_required=RESULT_REQUIRED, source_files=SOURCE_FILES, fixture_overrides=fixture_overrides, argv=argv)
    if rc == 0 and argv and argv[0] == "estimate":
        _project_estimate_to_production(argv)
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
