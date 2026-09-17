"""R3 runner: T21-S finite-size recovery before a loadings claim.

Built on ``common.cli.main_template`` (validate / enumerate / run / reduce /
estimate).  The archived t21s geometry and predictor are imported unchanged; the
new science is the fixed-gamma ladder, the finite-size endpoint and the gated
interpretation in :mod:`scaling`.

Per data cell (one targeted configuration x one rung x one seed): draw
z_i = s_i + n_i with s_i ~ N(0, K), n_i ~ N(0, I) in whitened R^d, fit rank-k PCA
on the centred training data (ONE fit, charged), score every planted direction by
ordinary and unique-component recovery, and compute the archived unfitted
predictor from the exact K (ONE eigen solve, counted as a solve).

Phases: pilot cells feed the pilot-only selection of an alternative near-edge
band; the pilot reducer writes ``FROZEN_EDGE.json`` into the run root;
confirmation cells refuse to run without it (exit 6, nothing charged).  The
finite-size endpoint is fixed-design, so the freeze gates the candidate, not the
endpoint.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ...successor_20260913.common import cli
from ...successor_20260913.common import fitcount
from ...successor_20260913.common.cli import UsageError
from ...successor_20260913.common.matrix import RunMatrix, make_cell
from ...successor_20260913.common.protocol import ProtocolError, load_protocol
from ...successor_20260913.common.results import SchemaError, sha256_file, write_json_atomic
from ...successor_20260913.t21s import geometry as GEO
from ...successor_20260913.t21s import predictor as PRED
from ...successor_20260913.t21s import runner as T21RUN

from . import scaling as SC

PACKAGE = "r3_t21_finite"
HERE = Path(__file__).resolve().parent
PROTOCOL_DEFAULT = HERE / "PROTOCOL_R3_T21_FINITE_v1.json"
SOURCE_FILES = [HERE / "runner.py", HERE / "scaling.py", HERE / "benchmark.py",
                HERE.parents[1] / "successor_20260913" / "t21s" / "geometry.py",
                HERE.parents[1] / "successor_20260913" / "t21s" / "predictor.py",
                # the imported T21 runner supplies the data generation and the fitting
                HERE.parents[1] / "successor_20260913" / "t21s" / "runner.py",
                HERE.parents[1] / "successor_20260913" / "common" / "cli.py",
                HERE.parents[1] / "successor_20260913" / "common" / "matrix.py",
                HERE.parents[1] / "successor_20260913" / "common" / "results.py"]
FROZEN_NAME = "FROZEN_EDGE.json"
FROZEN_SCHEMA = "mlst-r3-frozen-edge/1"
PHASES = ("pilot", "confirmation")

RESULT_REQUIRED = {"kind": str, "scientific_checks_ok": bool}
DATA_RESULT_REQUIRED = {
    "d": int, "n_train": int, "k": int, "size_mult": int, "gamma": float, "sqrt_gamma": float,
    "chance_level": float, "theta": list, "theta_over_sqrt_gamma": list, "population_excess_eigenvalues": list,
    "c2_per_spike": list, "loadings_sq": list, "unique_loadings_sq": list, "predicted_recovery": list,
    "predicted_unique_recovery": list, "recovery": list, "unique_recovery": list, "unique_defined": list,
    "unique_norm_ratio": list, "abs_error": list, "abs_error_unique": list, "sum_recovery": float,
    "fitted_rank": int, "sample_top_eigenvalues": list, "gram_realised_max_abs_error": float, "edge_class": list,
}
ALGEBRAIC_RESULT_REQUIRED = {"check": str, "ok": bool, "details": dict}

OLD_OUTCOME_PRESERVED = (
    "T21-S (successor_20260913, immutable): H1 primary 296/298 passed the old 0.05 tolerance -- orthogonal 190/192, "
    "correlated 106/106 -- and H1 is NOT supported overall because it required both arms; under "
    "predictor_fails_on_orthogonal_only the loadings step was untested.  H1' near-edge: correlated 102/102, "
    "orthogonal 9/16.  H2 overlap confound: supported.  H0 all-zero: passed.  The five registered T21 predictions "
    "(archive, 11 Sep 2026) remain NOT met.  R3 restates these; it neither edits nor repairs them."
)


class ScientificCheckError(SchemaError):
    """A package-level scientific invariant failed (exit 4, like a schema error)."""


# --------------------------------------------------------------------------- design helpers
def design(P: dict, fixture: bool) -> dict:
    g = P["design"]
    if fixture:
        F = P["fixture"]
        seeds = {"pilot": list(range(int(F["development_seeds"]["pilot_phase"][0]), int(F["development_seeds"]["pilot_phase"][1]) + 1)),
                 "confirmation": list(range(int(F["development_seeds"]["confirmation_phase"][0]), int(F["development_seeds"]["confirmation_phase"][1]) + 1))}
    else:
        s = g["seeds"]
        seeds = {"pilot": list(range(int(s["pilot"][0]), int(s["pilot"][1]) + 1)),
                 "confirmation": list(range(int(s["confirmation"][0]), int(s["confirmation"][1]) + 1))}
    return {"k": int(g["k"]), "J": int(g["n_directions"]), "rungs": SC.ladder(P, fixture),
            "geometries": list(g["geometry"]["arms"]), "arms": sorted(g["gamma_arms"]), "seeds": seeds}


def fixture_overrides(P: dict) -> dict:
    eff = design(P, True)
    return {"sizes": eff["rungs"], "seeds": eff["seeds"]}


def _check_protocol_geometry(P: dict) -> None:
    g = P["design"]["geometry"]
    if int(g["frame_seed"]) != GEO.FRAME_SEED:
        raise ProtocolError(f"protocol frame_seed {g['frame_seed']} != geometry.FRAME_SEED {GEO.FRAME_SEED}")
    for arm in g["arms"]:
        declared = np.asarray(g["gram"][arm], dtype=float)
        if declared.shape != GEO.gram(arm).shape or np.max(np.abs(declared - GEO.gram(arm))) > 0:
            raise ProtocolError(f"protocol Gram for {arm!r} differs from the archived geometry")
    if P["design"]["predictor"]["formula_id"] != PRED.FORMULA_ID:
        raise ProtocolError(f"protocol predictor formula_id {P['design']['predictor']['formula_id']!r} != predictor.FORMULA_ID {PRED.FORMULA_ID!r}")


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    _check_protocol_geometry(P)
    eff = design(P, fixture)
    cells = []
    for phase in PHASES:
        role = "development" if fixture else phase
        for cfg in SC.all_configurations(P, fixture):
            for seed in eff["seeds"][phase]:
                params = {**cfg, "phase": phase}
                cells.append(make_cell(role=role, seed=int(seed), params=params, n_fits=1, n_solves=1,
                                       family=SC.family_name(phase, cfg)))
    for geo in eff["geometries"]:
        cells.append(make_cell(role="algebraic", seed=None, params={"check": "ladder_preservation", "geometry": geo},
                               n_fits=0, n_solves=1, family="algebraic"))
    cells.append(make_cell(role="algebraic", seed=None, params={"check": "predictor_sanity"}, n_fits=0, n_solves=0, family="algebraic"))
    cells.append(make_cell(role="algebraic", seed=None, params={"check": "legacy_classification_rule"}, n_fits=0, n_solves=0, family="algebraic"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


# --------------------------------------------------------------------------- per-cell science
def run_data_cell(P: dict, cell: dict) -> dict:
    params = cell["params"]
    d, n, k = int(params["d"]), int(params["n_train"]), int(params["k"])
    pred = SC.compute_predictions(P, params)
    U = GEO.directions(d, params["geometry"])
    Up, ratio = GEO.unique_components(U)
    theta = SC.theta_vector(P, params)
    Z = T21RUN.draw_data(cell["seed"], n, U, theta)
    Pb, s_top = T21RUN.fit_pca(Z, k)                                   # the one charged fit
    rec = [T21RUN._recovery(Pb, U[:, j]) for j in range(U.shape[1])]
    urec = [(T21RUN._recovery(Pb, Up[:, j]) if pred["unique_defined"][j] else None) for j in range(U.shape[1])]
    res = {"kind": params["kind"], "d": d, "n_train": n, "k": k, "size_mult": int(params["size_mult"]), **pred,
           "recovery": rec, "unique_recovery": urec,
           "abs_error": [abs(r - p) for r, p in zip(rec, pred["predicted_recovery"])],
           "abs_error_unique": [(abs(r - p) if r is not None else None) for r, p in zip(urec, pred["predicted_unique_recovery"])],
           "sum_recovery": float(np.sum(rec)), "fitted_rank": int(Pb.shape[1]),
           "sample_top_eigenvalues": [float(x) for x in s_top]}
    res["scientific_checks_ok"] = True
    check_data_result(P, cell, res)
    return res


def run_algebraic_cell(P: dict, cell: dict, fixture: bool) -> dict:
    params = cell["params"]
    if params["check"] == "ladder_preservation":
        full = SC.preservation_check(P, fixture)
        det = {"geometry": params["geometry"], **full["per_geometry"][params["geometry"]],
               "ladder": full["rungs"], "fixture": bool(fixture)}
        res = {"kind": "algebraic", "check": "ladder_preservation", "ok": bool(det["ok"]), "details": det}
    elif params["check"] == "predictor_sanity":
        det = {arm: PRED.sanity(gamma=g) for arm, g in sorted(P["design"]["gamma_arms"].items())}
        det["formula_id"] = PRED.FORMULA_ID
        res = {"kind": "algebraic", "check": "predictor_sanity", "ok": bool(all(v["ok"] for v in det.values() if isinstance(v, dict))), "details": det}
    elif params["check"] == "legacy_classification_rule":
        det = _legacy_rule_check(P, fixture)
        res = {"kind": "algebraic", "check": "legacy_classification_rule", "ok": bool(det["ok"]), "details": det}
    else:
        raise UsageError(f"unknown algebraic check {params['check']!r}")
    res["scientific_checks_ok"] = bool(res["ok"])
    if not res["ok"]:
        raise ScientificCheckError(f"{cell['cell_id']}: algebraic check {params['check']} failed: {json.dumps(det, sort_keys=True)[:400]}")
    return res


def _legacy_rule_check(P: dict, fixture: bool) -> dict:
    """The R3 legacy classifier must reproduce the archived one exactly for the
    targeted configurations at the base rung."""
    base = SC.ladder(P, fixture)[0]
    rows = []
    ok = True
    for geo in P["design"]["geometry"]["arms"]:
        for arm in sorted(P["design"]["gamma_arms"]):
            for cfg in SC.grid_configs(P) + [{"config_id": "all_zero", "varied_direction": None, "ratio": None},
                                              {"config_id": "all_strong", "varied_direction": None, "ratio": None}]:
                kind = cfg["config_id"] if cfg["config_id"] in ("all_zero", "all_strong") else "grid"
                params = {"geometry": geo, "gamma_arm": arm, "size_mult": base["mult"], "d": base["d"],
                          "n_train": base["n_train"][arm], "k": int(P["design"]["k"]), "kind": kind,
                          "config_id": cfg["config_id"], "varied_direction": cfg["varied_direction"], "ratio": cfg["ratio"]}
                pred, _U, _Up, _t = T21RUN.compute_predictions(P, params)
                mine = SC.legacy_edge_classes(P, pred)
                same = list(mine) == list(pred["edge_class"])
                ok = ok and same
                rows.append({"arm": arm, "geometry": geo, "config_id": cfg["config_id"], "edge_class": mine, "matches_archived": same})
    return {"ok": bool(ok), "matches_archived": {"ok": bool(ok)}, "rows": rows[:12], "n_rows": len(rows),
            "note": "the R3 legacy classifier is compared with t21s.runner.edge_classes on the targeted base-rung configurations"}


def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    if cell["role"] == "algebraic":
        return run_algebraic_cell(P, cell, fixture), 0, int(cell["n_solves"])
    if cell["params"]["phase"] == "confirmation":
        if not (Path(ctx["out"]) / FROZEN_NAME).exists():
            raise ScientificCheckError(f"{cell['cell_id']}: confirmation cell without {FROZEN_NAME} in the run root")
    res = run_data_cell(P, cell)
    obs = fitcount.require()
    res["fits_observed"] = obs.as_dict()
    return res, obs.total, 1


# --------------------------------------------------------------------------- scientific checks
def _close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def check_data_result(P: dict, cell: dict, res: dict, cache: dict | None = None) -> None:
    where = cell["cell_id"]
    for k_, t in DATA_RESULT_REQUIRED.items():
        if k_ not in res:
            raise ScientificCheckError(f"{where}: data result lacks {k_!r}")
        if t is float and not isinstance(res[k_], (int, float)):
            raise ScientificCheckError(f"{where}: {k_} must be a number")
        if t in (int, list, dict) and not isinstance(res[k_], t):
            raise ScientificCheckError(f"{where}: {k_} must be {t.__name__}")
    J_ = int(P["design"]["n_directions"])
    for k_ in ("recovery", "unique_recovery", "predicted_recovery", "predicted_unique_recovery", "theta",
               "abs_error", "edge_class", "unique_defined"):
        if len(res[k_]) != J_:
            raise ScientificCheckError(f"{where}: {k_} must have {J_} entries")
    params = cell["params"]
    if int(res["d"]) != int(params["d"]) or int(res["n_train"]) != int(params["n_train"]) or int(res["k"]) != int(params["k"]):
        raise ScientificCheckError(f"{where}: stored d/N/k disagree with the cell params")
    if int(res["size_mult"]) != int(params["size_mult"]):
        raise ScientificCheckError(f"{where}: stored size_mult disagrees with the cell params")
    if not _close(res["gamma"], res["d"] / res["n_train"]):
        raise ScientificCheckError(f"{where}: gamma is not d/N")
    ck = SC.config_key(params)
    if cache is not None and ck in cache:
        pred = cache[ck]
    else:
        pred = SC.compute_predictions(P, params)
        if cache is not None:
            cache[ck] = pred
    if res["gram_realised_max_abs_error"] > GEO.GRAM_TOLERANCE:
        raise ScientificCheckError(f"{where}: realised Gram error {res['gram_realised_max_abs_error']:.3e} > {GEO.GRAM_TOLERANCE}")
    for k_ in ("theta", "predicted_recovery", "population_excess_eigenvalues", "c2_per_spike", "unique_norm_ratio"):
        if any(not _close(a, b) for a, b in zip(res[k_], pred[k_])):
            raise ScientificCheckError(f"{where}: stored {k_} disagrees with the recomputation from the exact K")
    for a, b in zip(res["predicted_unique_recovery"], pred["predicted_unique_recovery"]):
        if (a is None) != (b is None) or (a is not None and not _close(a, b)):
            raise ScientificCheckError(f"{where}: predicted_unique_recovery disagrees with the recomputation")
    if list(res["unique_defined"]) != list(pred["unique_defined"]):
        raise ScientificCheckError(f"{where}: unique_defined disagrees with the design")
    # the legacy near-edge classification is recomputed from the protocol band; a
    # widened band silently applied to the old classification is refused here
    legacy = SC.legacy_edge_classes(P, pred)
    if list(res["edge_class"]) != legacy:
        raise ScientificCheckError(f"{where}: stored legacy edge_class {res['edge_class']} != the protocol-band "
                                   f"recomputation {legacy} (a widened band was applied to the old classification)")
    if not _close(res["gamma"], pred["gamma"]) or not _close(res["chance_level"], pred["chance_level"]):
        raise ScientificCheckError(f"{where}: gamma / chance_level disagree with d and N")
    for j in range(J_):
        r = res["recovery"][j]
        if not (-1e-9 <= r <= 1.0 + 1e-9):
            raise ScientificCheckError(f"{where}: recovery[{j}] = {r} outside [0, 1]")
        u = res["unique_recovery"][j]
        if res["unique_defined"][j]:
            if u is None or not (-1e-9 <= u <= 1.0 + 1e-9):
                raise ScientificCheckError(f"{where}: unique_recovery[{j}] = {u} undefined or outside [0, 1]")
        elif u is not None:
            raise ScientificCheckError(f"{where}: unique_recovery[{j}] must be null when the unique component is negligible")
        if not _close(res["abs_error"][j], abs(r - res["predicted_recovery"][j])):
            raise ScientificCheckError(f"{where}: abs_error[{j}] is not |recovery - predicted|")
    if params["geometry"] == "orthogonal":
        diff = max(abs(a - b) for a, b in zip(res["recovery"], res["unique_recovery"]))
        if diff > 1e-9:
            raise ScientificCheckError(f"{where}: orthogonal geometry but unique and ordinary recovery differ by {diff:.3e}")
        if res["sum_recovery"] > res["k"] + 1e-9:
            raise ScientificCheckError(f"{where}: sum_j ||P u_j||^2 = {res['sum_recovery']} exceeds the rank {res['k']}")
    if res["fitted_rank"] != int(res["k"]):
        raise ScientificCheckError(f"{where}: fitted rank {res['fitted_rank']} != k {res['k']}")
    if params["kind"] == "all_zero" and any(abs(x) > 1e-12 for x in res["predicted_recovery"]):
        raise ScientificCheckError(f"{where}: all-zero cell has a non-zero prediction")
    if not res.get("scientific_checks_ok", False):
        raise ScientificCheckError(f"{where}: scientific_checks_ok is not true")


def check_all_zero_control(P: dict, data_cells: list[dict]) -> dict:
    z = float(P["design"]["controls"]["all_zero"]["z_tolerance"])
    out = {}
    groups: dict[str, list[dict]] = {}
    for c in data_cells:
        if c["params"]["kind"] == "all_zero":
            groups.setdefault(f"{c['params']['geometry']}|{c['params']['gamma_arm']}|d{c['params']['d']}", []).append(c)
    for key, cs in sorted(groups.items()):
        per_seed = [float(np.mean(c["result"]["recovery"])) for c in sorted(cs, key=lambda c: c["seed"])]
        d, k = int(cs[0]["result"]["d"]), int(cs[0]["result"]["k"])
        chance = k / d
        se = T21RUN._beta_sd(k, d) / np.sqrt(len(per_seed))
        mean = float(np.mean(per_seed))
        out[key] = {"n_seeds": len(per_seed), "mean_recovery": mean, "chance_level": chance,
                    "sd_beta_single_direction": T21RUN._beta_sd(k, d), "tolerance_abs": float(z * se), "z": z,
                    "abs_deviation": abs(mean - chance), "pass": bool(abs(mean - chance) <= z * se),
                    "max_predicted_recovery": float(max(max(c["result"]["predicted_recovery"]) for c in cs))}
    return out


# --------------------------------------------------------------------------- reduction
def _group_by_config(data_cells: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for c in data_cells:
        g.setdefault(SC.config_key(c["params"]), []).append(c)
    return {k: sorted(v, key=lambda c: c["seed"]) for k, v in sorted(g.items())}


def summarise_configs(data_cells: list[dict]) -> dict:
    out = {}
    for key, cs in _group_by_config(data_cells).items():
        r0 = cs[0]["result"]
        J_ = len(r0["recovery"])
        ent = {"params": {k: v for k, v in cs[0]["params"].items() if k != "phase"},
               "phases": sorted({c["params"]["phase"] for c in cs}), "n_seeds": len(cs), "seeds": [c["seed"] for c in cs],
               "d": r0["d"], "size_mult": r0["size_mult"], "gamma": r0["gamma"], "chance_level": r0["chance_level"],
               "theta_over_sqrt_gamma": r0["theta_over_sqrt_gamma"], "directions": []}
        for j in range(J_):
            ent["directions"].append({
                "j": j, "predicted_recovery": r0["predicted_recovery"][j],
                "predicted_unique_recovery": r0["predicted_unique_recovery"][j], "unique_defined": r0["unique_defined"][j],
                "unique_norm_ratio": r0["unique_norm_ratio"][j], "edge_class": r0["edge_class"][j],
                "recovery": SC.seed_stats([c["result"]["recovery"][j] for c in cs], key + f"|rec{j}"),
                "unique_recovery": SC.seed_stats([c["result"]["unique_recovery"][j] for c in cs], key + f"|urec{j}")})
        out[key] = ent
    return out


def frozen_content(P: dict, matrix: RunMatrix, selection: dict, regime: dict, fixture: bool) -> dict:
    tol = P["design"]["predictor"]["tolerance"]
    confs: dict[str, dict] = {}
    for c in matrix.cells:
        if c["role"] == "algebraic" or c["params"]["phase"] != "confirmation":
            continue
        key = SC.config_key(c["params"])
        if key not in confs:
            pred = SC.compute_predictions(P, c["params"])
            selected_band = selection["selected"]["band"]
            confs[key] = {"params": {k: v for k, v in c["params"].items() if k != "phase"},
                          "d": int(c["params"]["d"]), "gamma": pred["gamma"],
                          "chance_level": pred["chance_level"], "theta": pred["theta"],
                          "theta_over_sqrt_gamma": pred["theta_over_sqrt_gamma"],
                          "population_excess_eigenvalues": pred["population_excess_eigenvalues"],
                          "predicted_recovery": pred["predicted_recovery"],
                          "predicted_unique_recovery": pred["predicted_unique_recovery"],
                          "unique_defined": pred["unique_defined"], "legacy_edge_class": pred["edge_class"],
                          "candidate_edge_class": (SC.candidate_edge_classes(P, pred, selected_band) if selected_band else None),
                          "confirmation_cell_ids": [], "confirmation_seeds": []}
        confs[key]["confirmation_cell_ids"].append(c["cell_id"])
        confs[key]["confirmation_seeds"].append(c["seed"])
    return {"schema": FROZEN_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"], "matrix_sha256": matrix.matrix_hash(),
            "development_only": bool(fixture),
            "legacy": {"formula_id": PRED.FORMULA_ID, "formula": PRED.FORMULA_TEXT,
                       "coefficients_fitted_on_any_data": False,
                       "source": P["design"]["predictor"]["source"], "tolerance": tol,
                       "scoring_rule": P["design"]["predictor"]["scoring_rule"]},
            "endpoint_spec": dict(P["design"]["finite_size_endpoint"]),
            "candidate_family": SC.candidate_family(P), "candidate_selection_rule": P["design"]["candidate_selection_rule"],
            "selected_candidate": selection["selected"], "selection_reason": selection["reason"],
            "pilot_candidate_evidence": selection["evidence"], "pilot_tolerance_abs": selection["tolerance_abs"],
            "pilot_regime_check": regime,
            "confirmation_permitted": True,
            "why_permitted": ("the finite-size endpoint is fixed-design and involves no selection; the freeze's role is to "
                              "record the pilot-only candidate before any confirmation cell is drawn, and confirmation is "
                              "refused unless this file exists (exit 6, nothing charged)"),
            "if_selected_none": "the candidate is reported as not adequate; the legacy classification is still reported",
            "n_confirmation_configurations": len(confs),
            "n_confirmation_cells": sum(len(v["confirmation_cell_ids"]) for v in confs.values()), "configurations": confs}


def _strip_time(js: dict) -> dict:
    return {k: v for k, v in js.items() if k != "frozen_utc"}


def load_frozen(out: Path, P: dict, matrix: RunMatrix) -> dict | None:
    p = Path(out) / FROZEN_NAME
    if not p.exists():
        return None
    js = json.loads(p.read_text())
    if js.get("schema") != FROZEN_SCHEMA or js.get("package") != PACKAGE:
        raise ScientificCheckError(f"{p}: not an R3 frozen edge file")
    if js.get("protocol_sha256") != P["_sha256"] or js.get("matrix_sha256") != matrix.matrix_hash():
        raise ScientificCheckError(f"{p}: frozen for a different protocol/matrix")
    if js["legacy"]["formula_id"] != PRED.FORMULA_ID:
        raise ScientificCheckError(f"{p}: frozen formula {js['legacy']['formula_id']!r} is not the implemented {PRED.FORMULA_ID!r}")
    return js


def _row(group_key: str, pred: dict, klass: list[str], cs: list[dict], tol: dict, *,
         expected_n_seeds: int) -> list[dict]:
    rows = []
    for j in range(len(klass)):
        t = float(tol["primary_abs"] if klass[j] == "primary" else tol["near_edge_abs"])
        st = SC.seed_stats([c["result"]["recovery"][j] for c in cs], group_key + f"|rec{j}")
        rows.append({"config": group_key, "geometry": cs[0]["params"]["geometry"], "gamma_arm": cs[0]["params"]["gamma_arm"],
                     "kind": cs[0]["params"]["kind"], "config_id": cs[0]["params"].get("config_id"),
                     "size_mult": int(cs[0]["params"]["size_mult"]), "d": int(cs[0]["params"]["d"]),
                     "varied_direction": cs[0]["params"].get("varied_direction"), "ratio": cs[0]["params"].get("ratio"),
                     "direction": j, "edge_class": klass[j], "n_seeds": st["n"], "expected_n_seeds": expected_n_seeds,
                     "predicted": pred["predicted_recovery"][j], "observed_mean": st["mean"], "observed_se": st["se"],
                     "observed_ci95": st["ci95"], "abs_error_of_mean": abs(st["mean"] - pred["predicted_recovery"][j]),
                     "tolerance_abs": t, "pass": bool(abs(st["mean"] - pred["predicted_recovery"][j]) <= t),
                     "ci95_covers_prediction": (bool(st["ci95"][0] <= pred["predicted_recovery"][j] <= st["ci95"][1]) if st["ci95"] else None)})
    return rows


def score_legacy(P: dict, frozen: dict, conf_cells: list[dict]) -> dict:
    """Score the legacy predictor against the frozen confirmation design.

    Expected seed counts come from the DECLARED matrix as written into the frozen
    file, never from the observed cell count.  An incomplete configuration
    therefore reports ``n_seeds < expected_n_seeds`` and cannot label itself
    complete; the observed seed set must equal the frozen seed set.
    """
    tol = frozen["legacy"]["tolerance"]
    rows = []
    incomplete = []
    for key, cs in _group_by_config(conf_cells).items():
        fz = frozen["configurations"].get(key)
        if fz is None:
            raise ScientificCheckError(f"confirmation configuration {key} is not in {FROZEN_NAME}")
        expected_seeds = set(fz.get("confirmation_seeds", []))
        observed_seeds = {c["seed"] for c in cs}
        if observed_seeds != expected_seeds:
            incomplete.append({"config": key, "missing": sorted(expected_seeds - observed_seeds),
                               "extra": sorted(observed_seeds - expected_seeds)})
        for c in cs:
            if c["cell_id"] not in fz["confirmation_cell_ids"]:
                raise ScientificCheckError(f"{c['cell_id']}: not listed in the frozen file for {key}")
            if list(c["result"]["edge_class"]) != list(fz["legacy_edge_class"]):
                raise ScientificCheckError(f"{c['cell_id']}: legacy classification differs from the frozen one")
        rows += _row(key, {"predicted_recovery": fz["predicted_recovery"]}, fz["legacy_edge_class"], cs, tol,
                     expected_n_seeds=len(expected_seeds))
    prim = [r for r in rows if r["edge_class"] == "primary"]
    near = [r for r in rows if r["edge_class"] == "near_edge"]
    return {"rows": rows,
            "overall": _agg(rows),
            "per_geometry": {g: _agg([r for r in rows if r["geometry"] == g]) for g in sorted({r["geometry"] for r in rows})},
            "incomplete_configurations": incomplete,
            "all_configurations_complete": (not incomplete and all(r["n_seeds"] == r["expected_n_seeds"] for r in rows)),
            "interpretation_rule": P["failure_interpretation"]}


def _agg(rows: list[dict]) -> dict:
    prim = [r for r in rows if r["edge_class"] == "primary"]
    near = [r for r in rows if r["edge_class"] == "near_edge"]
    return {"n_rows": len(rows), "n_primary": len(prim), "n_primary_pass": int(sum(r["pass"] for r in prim)),
            "primary_all_pass": bool(prim) and all(r["pass"] for r in prim),
            "n_near_edge": len(near), "n_near_edge_pass": int(sum(r["pass"] for r in near)),
            "near_edge_all_pass": bool(near) and all(r["pass"] for r in near),
            "max_primary_abs_error": (max(r["abs_error_of_mean"] for r in prim) if prim else None)}


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    out = Path(out)
    fixture = any(js.get("development_only") for js in cells.values())
    data_cells = [js for js in cells.values() if js["role"] != "algebraic"]
    alg_cells = [js for js in cells.values() if js["role"] == "algebraic"]
    by_id = {c["cell_id"]: c for c in matrix.cells}
    cache: dict = {}
    for js in data_cells:
        check_data_result(P, by_id[js["cell_id"]], js["result"], cache)
    for js in alg_cells:
        if not js["result"].get("ok"):
            raise ScientificCheckError(f"{js['cell_id']}: algebraic check {js['result'].get('check')} recorded as failed")
    # generator invariant, recomputed at reduce
    preservation = SC.preservation_check(P, fixture)
    if not preservation["ok"]:
        raise ScientificCheckError("ladder preservation failed at reduce: the realised Gram / population spectrum is not invariant across rungs")
    phases = {}
    for ph in PHASES:
        exp = [c["cell_id"] for c in matrix.cells if c["role"] != "algebraic" and c["params"]["phase"] == ph]
        obs = [k for k in exp if k in cells]
        phases[ph] = {"expected": len(exp), "observed": len(obs), "complete": len(obs) == len(exp)}
    exp_alg = [c["cell_id"] for c in matrix.cells if c["role"] == "algebraic"]
    phases["algebraic"] = {"expected": len(exp_alg), "observed": sum(k in cells for k in exp_alg), "complete": all(k in cells for k in exp_alg)}
    pilot_cells = [js for js in data_cells if js["params"]["phase"] == "pilot"]
    conf_cells = [js for js in data_cells if js["params"]["phase"] == "confirmation"]
    controls = {"all_zero": check_all_zero_control(P, data_cells)}
    bad = [k for k, v in controls["all_zero"].items() if not v["pass"]]
    if bad:
        raise ScientificCheckError(f"all-zero control not at chance k/d in {bad}: {json.dumps({k: controls['all_zero'][k] for k in bad})[:500]}")
    orth = [js for js in data_cells if js["params"]["geometry"] == "orthogonal"]
    corr = [js for js in data_cells if js["params"]["geometry"] == "correlated" and js["params"]["kind"] != "all_zero"]
    controls["orthogonal_unique_equals_ordinary"] = {"n_cells": len(orth), "max_abs_diff": (max(max(abs(a - b) for a, b in zip(js["result"]["recovery"], js["result"]["unique_recovery"])) for js in orth) if orth else None)}
    controls["correlated_unique_differs"] = {"n_cells": len(corr), "max_abs_diff": (max(max(abs(a - b) for a, b in zip(js["result"]["recovery"], js["result"]["unique_recovery"])) for js in corr) if corr else None),
                                             "note": "descriptive: the two recoveries are different observables in the correlated arm"}
    controls["all_strong"] = {k: {"n_seeds": v["n_seeds"], "mean_recovery_by_direction": [d_["recovery"]["mean"] for d_ in v["directions"]],
                                  "predicted_by_direction": [d_["predicted_recovery"] for d_ in v["directions"]]}
                              for k, v in summarise_configs([js for js in data_cells if js["params"]["kind"] == "all_strong"]).items()}
    summary = {"phase_completeness": phases, "controls": controls, "ladder_preservation": {"ok": preservation["ok"], "rungs": preservation["rungs"]},
               "algebraic": {js["result"]["check"]: js["result"]["ok"] for js in alg_cells},
               "configurations": summarise_configs(data_cells),
               "historical_outcomes_preserved": OLD_OUTCOME_PRESERVED,
               "status_vocabulary": "development_only fixture output (never citable)" if fixture else "executed (not yet validated)"}
    # overlap-confound evidence, kept separate from the endpoint: in the correlated
    # arm ordinary and unique-component recovery are different observables.
    summary["overlap_confound"] = {
        "note": ("descriptive only and kept separate from the finite-size endpoint: in the correlated arm "
                 "ordinary recovery mixes a direction's own excitation with its overlap with other excited "
                 "directions, while unique-component recovery does not; both and the chance level are reported"),
        "correlated_grid": {k: {"chance_level": v["chance_level"],
                                "directions": [{"j": d_["j"], "ordinary_mean": d_["recovery"]["mean"],
                                                "ordinary_predicted": d_["predicted_recovery"],
                                                "unique_mean": d_["unique_recovery"]["mean"],
                                                "unique_predicted": d_["predicted_unique_recovery"],
                                                "unique_defined": d_["unique_defined"]} for d_ in v["directions"]]}
                            for k, v in summary["configurations"].items()
                            if v["params"]["geometry"] == "correlated" and v["params"]["kind"] == "grid"}}
    # pilot: candidate selection and the freeze
    frozen = load_frozen(out, P, matrix)
    if conf_cells and frozen is None:
        raise ScientificCheckError(f"confirmation cells present but no {FROZEN_NAME} in the run root; the file that permitted them is gone")
    if pilot_cells:
        regime = _pilot_regime(P, pilot_cells)
        selection = SC.select_edge_candidate(P, pilot_cells)
        summary["pilot"] = {"n_cells": len(pilot_cells), "regime_check": regime, "complete": phases["pilot"]["complete"],
                            "candidate_selection": {"selected": selection["selected"], "reason": selection["reason"],
                                                    "evidence": selection["evidence"]}}
        if phases["pilot"]["complete"]:
            content = frozen_content(P, matrix, selection, regime, fixture)
            if frozen is None:
                content["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                write_json_atomic(out / FROZEN_NAME, content)
                frozen = json.loads((out / FROZEN_NAME).read_text())
                summary["frozen_edge_written_now"] = True
            elif _strip_time(frozen) != json.loads(json.dumps(content, sort_keys=True)):
                raise ScientificCheckError(f"{FROZEN_NAME} in the run root differs from what this pilot would freeze; it is not rewritten")
    summary["frozen_edge"] = ({"path": str(out / FROZEN_NAME), "sha256": sha256_file(out / FROZEN_NAME),
                               "selected_candidate": frozen["selected_candidate"], "confirmation_permitted": frozen["confirmation_permitted"],
                               "frozen_utc": frozen.get("frozen_utc")} if frozen else None)
    # confirmation: legacy classification and the frozen candidate, both reported
    if conf_cells:
        summary["confirmation"] = {"n_cells": len(conf_cells), "complete": phases["confirmation"]["complete"],
                                   **score_legacy(P, frozen, conf_cells)}
        sel = frozen["selected_candidate"]
        if sel and sel.get("band"):
            summary["candidate"] = {"selected": sel, "scored": True, **SC.score_candidate(P, sel["band"], conf_cells)}
        else:
            summary["candidate"] = {"selected": sel, "scored": False,
                                    "note": "no adequate candidate on pilot; the candidate is reported as 'none' and is not scored"}
        endpoint = SC.finite_size_endpoint(P, summary["confirmation"]["rows"])
        summary["finite_size_endpoint"] = endpoint
        gated = SC.gated_interpretation(P, endpoint, {"selected": sel})
        if fixture:
            gated["development_only_caveat"] = ("fixture (development-only) output, never citable: the loadings step "
                                                "cannot be read as supported from development data under any outcome")
        summary["gated_interpretation"] = gated
    return summary


def _pilot_regime(P: dict, pilot_cells: list[dict]) -> dict:
    R = P["design"]["pilot_regime_rule"]
    arms: dict[str, dict] = {}
    for c in pilot_cells:
        p = c["params"]
        if p["kind"] != "grid" or int(p["size_mult"]) != 1:
            continue
        a = arms.setdefault(f"{p['geometry']}|{p['gamma_arm']}", {"below": [], "above": [], "chance_level": c["result"]["chance_level"]})
        j = int(p["varied_direction"])
        r = c["result"]
        if p["config_id"] == "below_edge":
            a["below"].append(r["unique_recovery"][j] if r["unique_recovery"][j] is not None else r["recovery"][j])
        if p["config_id"] == "above_edge":
            a["above"].append(r["recovery"][j])
    out = {"rule": R, "arms": {}}
    for key, a in sorted(arms.items()):
        below_ok = bool(a["below"]) and float(np.mean(a["below"])) <= a["chance_level"] + float(R["below_max_excess_over_chance"])
        above_ok = bool(a["above"]) and float(np.mean(a["above"])) >= float(R["above_min_recovery"])
        out["arms"][key] = {"below_mean": (float(np.mean(a["below"])) if a["below"] else None), "n_below": len(a["below"]),
                            "above_mean": (float(np.mean(a["above"])) if a["above"] else None), "n_above": len(a["above"]),
                            "chance_level": a["chance_level"], "below_ok": below_ok, "above_ok": above_ok,
                            "populated": bool(below_ok and above_ok)}
    out["populated"] = bool(out["arms"]) and all(v["populated"] for v in out["arms"].values())
    out["gates_endpoint"] = False
    out["note"] = "descriptive in R3; the endpoint is fixed-design and confirmation is gated only by the presence of the frozen candidate"
    return out


# --------------------------------------------------------------------------- entry point
def _confirmation_guard(argv) -> int | None:
    ap = cli.build_parser(PACKAGE, "guard")
    args = ap.parse_args(argv)
    if args.cmd != "run" or not Path(args.protocol).exists():
        return None
    P = load_protocol(args.protocol)
    if P.get("package") != PACKAGE:
        return None
    fixture = bool(args.fixture)
    matrix = build_matrix(P, fixture)
    sub = matrix.subset(roles=["development", "control", "algebraic"]) if fixture else matrix
    try:
        sub = sub.subset(roles=args.roles, cell_ids=args.cells, families=args.families) if (args.roles or args.cells or args.families) else sub
    except ValueError as e:
        raise UsageError(str(e))
    wants_conf = [c for c in sub.cells if c["role"] != "algebraic" and c["params"]["phase"] == "confirmation"]
    if not wants_conf:
        return None
    frozen_path = Path(args.out) / FROZEN_NAME
    if not frozen_path.exists():
        print(f"refusing: {len(wants_conf)} confirmation-phase cell(s) selected but {frozen_path} does not exist; "
              f"run and reduce the pilot phase first (nothing was charged)", file=sys.stderr)
        return 6
    fz = load_frozen(Path(args.out), P, matrix)
    if not fz["confirmation_permitted"]:
        print(f"refusing: {frozen_path} does not permit confirmation (nothing was charged)", file=sys.stderr)
        return 6
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "preservation-receipt":
        import argparse
        ap = argparse.ArgumentParser(prog="r3_t21_finite preservation-receipt")
        ap.add_argument("--protocol", required=True, type=Path)
        ap.add_argument("--out", required=True, type=Path)
        ap.add_argument("--production", action="store_true", help="receipt for the production d-ladder")
        a = ap.parse_args(argv[1:])
        P = load_protocol(a.protocol)
        pc = SC.preservation_check(P, fixture=not a.production)
        receipt = {"schema": "mlst-followup-ladder-preservation/1", "package": PACKAGE,
                   "fixture": not a.production, "ok": bool(pc["ok"]),
                   "rung_receipts": pc.get("rung_receipts", []),
                   "rungs": pc.get("rungs", []),
                   "per_geometry": {g: {"ok": v["ok"], "configs": {k: {"ok": c["ok"], "rung_receipts": c["rung_receipts"]}
                                                                  for k, c in v["configs"].items()}}
                                    for g, v in pc["per_geometry"].items()},
                   "note": "executable pass/fail at every rung of the d-ladder; not protocol prose"}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"ok": receipt["ok"], "rungs": receipt["rung_receipts"], "out": str(a.out)}, indent=1))
        return 0 if receipt["ok"] else 4
    try:
        rc = _confirmation_guard(argv)
    except (ProtocolError, UsageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 6
    except SchemaError as e:
        print(f"schema error: {e}", file=sys.stderr)
        return 4
    if rc is not None:
        return rc
    return cli.main_template(package=PACKAGE, description="R3: T21-S finite-size recovery before a loadings claim",
                             build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                             result_required=RESULT_REQUIRED, source_files=SOURCE_FILES,
                             fixture_overrides=fixture_overrides, argv=argv)


if __name__ == "__main__":
    sys.exit(main())
