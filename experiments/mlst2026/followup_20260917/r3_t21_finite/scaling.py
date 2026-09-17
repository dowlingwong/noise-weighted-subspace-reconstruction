"""R3 scaling science: fixed-gamma ladder, generator preservation, the finite-size
endpoint, the pilot-selected alternative near-edge band and the gated interpretation.

Everything here is a pure function of the protocol and the drawn data summaries.
The archived t21s geometry and predictor modules are imported unchanged
(``successor_20260913.t21s``); this module adds no new formula to the legacy
predictor and never edits the archived tree.

Layout
------
* :func:`ladder` / :func:`grid_configs` / :func:`all_configurations` -- the
  targeted catalogue, instantiated at every rung of the fixed-gamma ladder.
* :func:`legacy_band`, :func:`candidate_family`, :func:`edge_classes_band` -- the
  legacy [0.7, 1.8] classifier and the separate, labeled alternative bands.
* :func:`preservation_check` -- the generator invariant that scaling d must not
  move the Gram or the population spike spectrum.
* :func:`select_edge_candidate` -- pilot-only selection of at most one candidate.
* :func:`finite_size_endpoint` -- decline of the error on the two old failure
  configurations plus largest-size legacy agreement, with across-seed uncertainty.
* :func:`gated_interpretation` -- the orthogonal prerequisite that gates any
  correlated-arm support for the added-loadings step.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ...successor_20260913.t21s import geometry as GEO
from ...successor_20260913.t21s import predictor as PRED
from ...successor_20260913.t21s import runner as T21RUN

KINDS = ("grid", "all_zero", "all_strong")
BOOTSTRAP_REPLICATES = 1000


# --------------------------------------------------------------------------- catalogue
def ladder(P: dict, fixture: bool) -> list[dict]:
    """The prespecified d/N ladder as a list of {mult, d, n_train} rungs."""
    block = P["fixture"] if fixture else P["design"]
    return [{"mult": int(s["mult"]), "d": int(s["d"]),
             "n_train": {k: int(v) for k, v in s["n_train"].items()}} for s in block["sizes"]]


def grid_configs(P: dict) -> list[dict]:
    return [{"config_id": str(c["config_id"]), "varied_direction": int(c["varied_direction"]), "ratio": float(c["ratio"])}
            for c in P["design"]["grid"]["configurations"]]


def all_configurations(P: dict, fixture: bool) -> list[dict]:
    """Every seed-free configuration: grid, all_zero and all_strong at every rung
    for both geometries and both gamma arms."""
    g = P["design"]
    out = []
    for geo in g["geometry"]["arms"]:
        for arm in sorted(g["gamma_arms"]):
            for rung in ladder(P, fixture):
                n = int(rung["n_train"][arm])
                base = {"geometry": geo, "gamma_arm": arm, "size_mult": int(rung["mult"]),
                        "d": int(rung["d"]), "n_train": n, "k": int(g["k"])}
                for cfg in grid_configs(P):
                    out.append({**base, "kind": "grid", "config_id": cfg["config_id"],
                                "varied_direction": cfg["varied_direction"], "ratio": cfg["ratio"]})
                out.append({**base, "kind": "all_zero", "config_id": None, "varied_direction": None, "ratio": None})
                out.append({**base, "kind": "all_strong", "config_id": None, "varied_direction": None, "ratio": None})
    return out


def config_key(p: dict) -> str:
    j = "-" if p.get("varied_direction") is None else f"j{p['varied_direction']}"
    r = "-" if p.get("ratio") is None else f"r{p['ratio']:g}"
    cid = p.get("config_id") or "-"
    return f"{p['geometry']}|{p['gamma_arm']}|d{p['d']}|{p['kind']}|{cid}|{j}|{r}"


def config_id_lookup(P: dict) -> dict[str, dict]:
    return {c["config_id"]: c for c in grid_configs(P)}


def family_name(phase: str, params: dict) -> str:
    return f"{phase}-{params['geometry']}-{params['gamma_arm']}-d{params['d']}"


# --------------------------------------------------------------------------- classification
def legacy_band(P: dict) -> tuple[float, float]:
    lo, hi = P["design"]["predictor"]["tolerance"]["near_edge_band_theta_over_sqrt_gamma"]
    return float(lo), float(hi)


def legacy_loading_min(P: dict) -> float:
    return float(P["design"]["predictor"]["tolerance"]["near_edge_loading_min"])


def candidate_family(P: dict) -> list[dict]:
    return [{"id": str(c["id"]), "band": [float(c["band"][0]), float(c["band"][1])]}
            for c in P["design"]["candidate_family"]]


def edge_classes_band(loadings_sq: list, population_excess_eigenvalues: list, sqrt_gamma: float,
                      band: tuple[float, float], loading_min: float) -> list[str]:
    """The archived classifier with a replaceable band: ``near_edge`` when some
    population spike carrying at least ``loading_min`` of the direction lies
    inside ``[lo, hi]`` (on the population eigenvalue over sqrt(gamma)), else
    ``primary``.  Called with the protocol's legacy band this is the legacy
    classification, unchanged."""
    lo, hi = band
    lam = np.asarray(population_excess_eigenvalues, dtype=float) / float(sqrt_gamma)
    out = []
    for row in loadings_sq:
        near = any(l2 >= loading_min and lo <= r <= hi for l2, r in zip(row, lam))
        out.append("near_edge" if near else "primary")
    return out


def legacy_edge_classes(P: dict, pred: dict) -> list[str]:
    return edge_classes_band(pred["loadings_sq"], pred["population_excess_eigenvalues"],
                             pred["sqrt_gamma"], legacy_band(P), legacy_loading_min(P))


def candidate_edge_classes(P: dict, pred: dict, band: list[float]) -> list[str]:
    return edge_classes_band(pred["loadings_sq"], pred["population_excess_eigenvalues"],
                             pred["sqrt_gamma"], (float(band[0]), float(band[1])), legacy_loading_min(P))


def compute_predictions(P: dict, params: dict) -> dict:
    """The archived design-only prediction block with the legacy classification
    made explicit: reuses ``t21s.runner.compute_predictions`` (same formula, same
    exact K) and overwrites ``edge_class`` with this package's own legacy
    classifier so a mutation of the classifier is independently detectable."""
    pred, _U, _Up, _theta = T21RUN.compute_predictions(P, params)
    pred["edge_class"] = legacy_edge_classes(P, pred)
    return pred


def theta_vector(P: dict, params: dict) -> np.ndarray:
    gamma = float(params["d"]) / float(params["n_train"])
    return GEO.theta_vector(gamma=gamma, kind=params["kind"], varied_direction=params.get("varied_direction"),
                            ratio=params.get("ratio"), strong_ratio=float(P["design"]["grid"]["strong_ratio"]))


# --------------------------------------------------------------------------- generator preservation
def preservation_check(P: dict, fixture: bool = False) -> dict:
    """Scaling d and N together at fixed gamma must not move the intended Gram or
    the population spike quantities.  For every geometry, gamma arm and
    configuration the exact-K population spectrum (a function of the Gram and
    theta only) is compared across rungs; every rung's realised Gram is compared
    with the declared one and with the Schur unique-norm identity."""
    rungs = ladder(P, fixture)
    g = P["design"]
    out: dict = {"rungs": [{"mult": r["mult"], "d": r["d"]} for r in rungs], "per_geometry": {}}
    global_ok = True
    for geo in g["geometry"]["arms"]:
        G = GEO.gram(geo)
        Ginv = np.linalg.inv(G)
        schur = 1.0 / np.diag(Ginv)
        per: dict = {"schur_unique_norm_ratio": [float(x) for x in schur], "configs": {}}
        for arm in sorted(g["gamma_arms"]):
            for cfg in grid_configs(P) + [{"config_id": "all_zero", "varied_direction": None, "ratio": None},
                                          {"config_id": "all_strong", "varied_direction": None, "ratio": None}]:
                kind = "grid" if cfg["config_id"] not in ("all_zero", "all_strong") else cfg["config_id"]
                key = f"{arm}|{cfg['config_id']}"
                spectra, gram_errs, ratios, c2_checks = [], [], [], []
                for r in rungs:
                    gamma = float(r["d"]) / float(r["n_train"][arm])
                    theta = GEO.theta_vector(gamma=gamma, kind=kind, varied_direction=cfg["varied_direction"],
                                             ratio=cfg["ratio"], strong_ratio=float(g["grid"]["strong_ratio"]))
                    spec = GEO.population_spectrum_exact(G, theta)
                    U = GEO.directions(int(r["d"]), geo)
                    _Up, ratio = GEO.unique_components(U)
                    spectra.append(spec)
                    gram_errs.append(float(np.max(np.abs(U.T @ U - G))))
                    ratios.append(np.asarray(ratio, dtype=float))
                    c2_checks.append([float(PRED.c2_isolated_spike(t, gamma)) for t in theta])
                spec_arr = np.asarray(spectra)
                spectrum_invariance = float(np.max(np.abs(spec_arr - spec_arr[0])))
                ratio_max_err = float(max(np.max(np.abs(rr - schur)) for rr in ratios))
                gram_max = float(max(gram_errs))
                # the scaled spike quantities c2(theta_i, gamma) must also be invariant across rungs
                c2_invariance = float(np.max(np.abs(np.asarray(c2_checks) - np.asarray(c2_checks)[0])))
                ok = (gram_max <= GEO.GRAM_TOLERANCE and spectrum_invariance <= 1e-8
                      and ratio_max_err <= 1e-9 and c2_invariance <= 1e-12)
                global_ok = global_ok and ok
                # A3.2: an explicit pass/fail receipt at EVERY rung of the d-ladder
                rung_ok = []
                for ri, r in enumerate(rungs):
                    rung_ok.append({
                        "mult": r["mult"], "d": int(r["d"]),
                        "gram_ok": bool(gram_errs[ri] <= GEO.GRAM_TOLERANCE),
                        "spectrum_ok": bool(np.max(np.abs(spec_arr[ri] - spec_arr[0])) <= 1e-8),
                        "unique_norm_ratio_ok": bool(np.max(np.abs(ratios[ri] - schur)) <= 1e-9),
                        "c2_ok": bool(np.max(np.abs(np.asarray(c2_checks)[ri] - np.asarray(c2_checks)[0])) <= 1e-12),
                    })
                    rung_ok[-1]["ok"] = all(rung_ok[-1][k] for k in
                                            ("gram_ok", "spectrum_ok", "unique_norm_ratio_ok", "c2_ok"))
                per["configs"][key] = {"ok": bool(ok), "gram_max_abs_error": gram_max,
                                       "population_spectrum_invariance_max_abs": spectrum_invariance,
                                       "unique_norm_ratio_schur_max_abs_error": ratio_max_err,
                                       "c2_invariance_max_abs": c2_invariance,
                                       "rung_receipts": rung_ok,
                                       "population_excess_eigenvalues": [float(x) for x in spectra[0]]}
        per["ok"] = all(v["ok"] for v in per["configs"].values())
        global_ok = global_ok and per["ok"]
        out["per_geometry"][geo] = per
    out["ok"] = bool(global_ok)
    # per-rung roll-up across every geometry/arm/config
    all_rung = {}
    for geo, per in out["per_geometry"].items():
        for key, cfg in per["configs"].items():
            for rr in cfg["rung_receipts"]:
                m = all_rung.setdefault(rr["mult"], {"mult": rr["mult"], "d": rr["d"], "n": 0, "n_ok": 0,
                                                     "gram_ok": True, "spectrum_ok": True,
                                                     "unique_norm_ratio_ok": True, "c2_ok": True})
                m["n"] += 1
                m["n_ok"] += int(rr["ok"])
                for flag in ("gram_ok", "spectrum_ok", "unique_norm_ratio_ok", "c2_ok"):
                    m[flag] = m[flag] and bool(rr[flag])
    out["rung_receipts"] = [
        {**v, "ok": bool(v["n"] > 0 and v["n_ok"] == v["n"])} for _, v in sorted(all_rung.items())]
    return out


# --------------------------------------------------------------------------- statistics
def _bootstrap_ci(x: np.ndarray, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES, alpha: float = 0.05) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(replicates, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def seed_stats(values: list, key: str) -> dict:
    """Mean/sd/se/CI95 over independent seeds; None values dropped (undefined
    unique components).  Bootstrap replicates are never treated as samples."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "se": None, "ci95": None, "undefined": True}
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    x = np.asarray(vals, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()),
            "sd": (float(x.std(ddof=1)) if x.size > 1 else None),
            "se": (float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None),
            "ci95": _bootstrap_ci(x, seed=seed), "replicates": BOOTSTRAP_REPLICATES, "undefined": False}


# --------------------------------------------------------------------------- candidate selection
def _pilot_grid_cells(pilot_cells: list[dict]) -> list[dict]:
    return [c for c in pilot_cells if c["params"]["kind"] == "grid"]


def select_edge_candidate(P: dict, pilot_cells: list[dict]) -> dict:
    """Pilot-only selection of at most one alternative near-edge band.

    A candidate band is *adequate* if, with that band reclassifying the pilot
    pairs, every candidate-primary pair has |seed-mean observed - predicted| <=
    0.05 and at least one pair is candidate-primary.  Among adequate bands the
    narrowest wins; ties follow the declared family order.  No confirmation cell
    enters this function."""
    tol = float(P["design"]["predictor"]["tolerance"]["primary_abs"])
    cells = _pilot_grid_cells(pilot_cells)
    groups: dict[str, list[dict]] = {}
    for c in cells:
        groups.setdefault(config_key(c["params"]), []).append(c)
    evidence = {}
    adequate: list[dict] = []
    for cand in candidate_family(P):
        n_primary = n_pass = 0
        max_err = None
        failures = []
        for key, cs in sorted(groups.items()):
            pred = compute_predictions(P, cs[0]["params"])
            klass = candidate_edge_classes(P, pred, cand["band"])
            for j in range(len(klass)):
                if klass[j] != "primary":
                    continue
                st = seed_stats([cc["result"]["recovery"][j] for cc in cs], key + f"|cand{j}")
                err = abs(st["mean"] - pred["predicted_recovery"][j])
                n_primary += 1
                max_err = err if max_err is None else max(max_err, err)
                if err <= tol:
                    n_pass += 1
                else:
                    failures.append({"config": key, "direction": j, "abs_error": err})
        ok = bool(n_primary) and n_pass == n_primary
        rec = {"id": cand["id"], "band": cand["band"], "width": float(cand["band"][1] - cand["band"][0]),
               "n_primary_scored": n_primary, "n_primary_pass": n_pass, "max_abs_error": max_err,
               "adequate": ok, "failures": failures[:5]}
        evidence[cand["id"]] = rec
        if ok:
            adequate.append(rec)
    if adequate:
        chosen = min(adequate, key=lambda r: (r["width"], [c["id"] for c in candidate_family(P)].index(r["id"])))
        selected = {"id": chosen["id"], "band": chosen["band"], "width": chosen["width"]}
        reason = "adequate on every pilot candidate-primary pair; narrowest band among adequate"
    else:
        selected = {"id": "none", "band": None, "width": None}
        reason = "no candidate met the predeclared pilot adequacy rule; the candidate is reported as 'none'"
    return {"selected": selected, "reason": reason, "tolerance_abs": tol, "n_pilot_grid_cells": len(cells),
            "family": candidate_family(P), "evidence": evidence,
            "rule": P["design"]["candidate_selection_rule"]}


def score_candidate(P: dict, band: list[float], confirmation_cells: list[dict]) -> dict:
    """Score a frozen candidate band on fresh confirmation cells: candidate-primary
    pairs against the legacy 0.05 tolerance, candidate-near-edge pairs against 0.10."""
    tol_p = float(P["design"]["predictor"]["tolerance"]["primary_abs"])
    tol_n = float(P["design"]["predictor"]["tolerance"]["near_edge_abs"])
    rows = []
    groups: dict[str, list[dict]] = {}
    for c in confirmation_cells:
        groups.setdefault(config_key(c["params"]), []).append(c)
    for key, cs in sorted(groups.items()):
        pred = compute_predictions(P, cs[0]["params"])
        klass = candidate_edge_classes(P, pred, band)
        for j in range(len(klass)):
            st = seed_stats([c["result"]["recovery"][j] for c in cs], key + f"|cand{j}")
            t = tol_p if klass[j] == "primary" else tol_n
            rows.append({"config": key, "direction": j, "edge_class": klass[j], "predicted": pred["predicted_recovery"][j],
                         "observed_mean": st["mean"], "observed_se": st["se"], "observed_ci95": st["ci95"], "n_seeds": st["n"],
                         "abs_error_of_mean": abs(st["mean"] - pred["predicted_recovery"][j]), "tolerance_abs": t,
                         "pass": bool(abs(st["mean"] - pred["predicted_recovery"][j]) <= t)})
    prim = [r for r in rows if r["edge_class"] == "primary"]
    near = [r for r in rows if r["edge_class"] == "near_edge"]
    return {"band": [float(band[0]), float(band[1])], "n_rows": len(rows), "n_primary": len(prim), "n_primary_pass": sum(r["pass"] for r in prim),
            "primary_all_pass": bool(prim) and all(r["pass"] for r in prim),
            "n_near_edge": len(near), "n_near_edge_pass": sum(r["pass"] for r in near),
            "max_primary_abs_error": (max(r["abs_error_of_mean"] for r in prim) if prim else None), "rows": rows}


# --------------------------------------------------------------------------- finite-size endpoint
def _endpoint_spec(P: dict) -> dict:
    return dict(P["design"]["finite_size_endpoint"])


def finite_size_endpoint(P: dict, rows: list[dict]) -> dict:
    """The predeclared finite-size endpoint over scored confirmation rows.

    ``rows`` carry the legacy classification and per-(configuration, direction)
    seed means.  The endpoint is the conjunction of (a) the decline of the
    seed-mean absolute error on the two old failure configuration types from the
    base rung to the largest rung, each largest-rung error at or below the
    original 0.05 tolerance, and (b) largest-size legacy agreement in the
    orthogonal arm (the prerequisite).  Correlated agreement is reported separately."""
    spec = _endpoint_spec(P)
    tol = float(spec["primary_tolerance"])
    present = sorted({int(r["size_mult"]) for r in rows})
    largest = int(spec["largest_size_mult"]) if int(spec["largest_size_mult"]) in present else (max(present) if present else int(spec["largest_size_mult"]))
    cfg_by_id = config_id_lookup(P)

    def _rung_rows(sel_rows, pair_cfg):
        out = []
        for r in sel_rows:
            if r["geometry"] != "orthogonal" or r["gamma_arm"] != "hi" or r["kind"] != "grid":
                continue
            if r["varied_direction"] != pair_cfg["varied_direction"] or r["ratio"] != pair_cfg["ratio"]:
                continue
            if r["direction"] != pair_cfg["varied_direction"]:
                continue
            out.append(r)
        return sorted(out, key=lambda r: r["size_mult"])

    decline = {}
    for cid in spec["decline_pair_config_ids"]:
        pair_cfg = cfg_by_id[cid]
        rr = _rung_rows(rows, pair_cfg)
        rungs = [{"size_mult": r["size_mult"], "d": r["d"], "abs_error_of_mean": r["abs_error_of_mean"],
                  "observed_se": r["observed_se"], "observed_ci95": r["observed_ci95"], "n_seeds": r["n_seeds"],
                  "pass": r["pass"], "predicted": r["predicted"], "observed_mean": r["observed_mean"]} for r in rr]
        base = next((x for x in rungs if x["size_mult"] == 1), None)
        top = next((x for x in rungs if x["size_mult"] == largest), None)
        decline_ok = bool(base is not None and top is not None and top["abs_error_of_mean"] < base["abs_error_of_mean"])
        largest_within = bool(top is not None and top["abs_error_of_mean"] <= tol)
        decline[cid] = {"varied_direction": pair_cfg["varied_direction"], "ratio": pair_cfg["ratio"], "rungs": rungs,
                        "base_abs_error": (base["abs_error_of_mean"] if base else None),
                        "largest_abs_error": (top["abs_error_of_mean"] if top else None),
                        "largest_within_tolerance": largest_within, "decline_ok": decline_ok,
                        "ok": bool(decline_ok and largest_within)}

    def _agreement(sel):
        prim = [r for r in sel if r["edge_class"] == "primary"]
        by_geo = {}
        for g in sorted({r["geometry"] for r in sel}):
            gp = [r for r in prim if r["geometry"] == g]
            by_geo[g] = {"n_primary": len(gp), "n_primary_pass": int(sum(r["pass"] for r in gp)),
                         "all_pass": bool(gp) and all(r["pass"] for r in gp),
                         "max_abs_error": (max(r["abs_error_of_mean"] for r in gp) if gp else None)}
        return by_geo

    largest_rows = [r for r in rows if r["size_mult"] == largest]
    by_size = {}
    for mult in sorted({r["size_mult"] for r in rows}):
        sr = [r for r in rows if r["size_mult"] == mult]
        by_size[str(mult)] = _agreement(sr)
    agreement_largest = {"mult": largest, "d": (largest_rows[0]["d"] if largest_rows else None),
                         "by_geometry": _agreement(largest_rows)}
    orth = agreement_largest["by_geometry"].get("orthogonal", {}).get("all_pass", False)
    corr = agreement_largest["by_geometry"].get("correlated", {}).get("all_pass", False)
    met = bool(all(v["ok"] for v in decline.values()) and orth)
    return {"spec": spec, "decline": decline, "agreement_by_size": by_size,
            "agreement_largest": agreement_largest, "orthogonal_prerequisite_pass": bool(orth),
            "correlated_largest_pass": bool(corr), "endpoint_met": met,
            "statement": ("the finite-size endpoint is met" if met else "the finite-size endpoint is NOT met")}


def gated_interpretation(P: dict, endpoint: dict, candidate: dict) -> dict:
    """The orthogonal prerequisite gates the added-loadings step.  A correlated-arm
    pass may support the loadings step only if the orthogonal isolated-spike
    prerequisite passed its prespecified criterion; otherwise the step is untested."""
    orth = bool(endpoint["orthogonal_prerequisite_pass"])
    corr = bool(endpoint["correlated_largest_pass"])
    if not orth:
        status = "untested"
        reason = P["failure_interpretation"]["orthogonal_prerequisite_fails"]
    elif corr:
        status = "supported on synthetic Gaussian spiked data only"
        reason = P["failure_interpretation"]["legacy_passes_both_at_largest_size"]
    else:
        status = "not supported"
        reason = P["failure_interpretation"]["correlated_fails_after_orthogonal_passes"]
    return {"orthogonal_prerequisite_pass": orth, "correlated_largest_pass": corr,
            "endpoint_met": bool(endpoint["endpoint_met"]),
            "loadings_step_status": status, "reason": reason,
            "candidate_selected": candidate["selected"]["id"],
            "candidate_is_separate": "the selected band is a labeled alternative; the legacy classification above is unchanged and is the classification of record"}
