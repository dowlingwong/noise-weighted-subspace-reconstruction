"""T21-S runner (work package L3): independent geometry and excitation.

Built on ``common.cli.main_template``.  Subcommands: validate / enumerate /
run / reduce / estimate (see ``python -m experiments.mlst2026.successor_20260913.t21s --help``).

Per data cell (one configuration x one seed): draw z_i = s_i + n_i with
s_i ~ N(0, K), n_i ~ N(0, I) in whitened R^d (N x d), fit rank-k PCA (k = J = 4)
on the centred training data (ONE fit, charged), score every planted direction
u_j by ordinary recovery ||P u_j||^2/||u_j||^2 and by unique-component recovery
||P u_j^perp||^2/||u_j^perp||^2 (undefined when ||u_j^perp||^2/||u_j||^2 < 1e-3),
and compute the frozen predictor from the exact K (ONE eigen solve, counted as a
solve, never as a fit).

Phases: pilot cells populate the regimes; the pilot reducer writes
``FROZEN_PREDICTOR.json`` into the run root; confirmation cells refuse to run
without it (exit 6, nothing charged) and are scored against it.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ..common import cli
from ..common.cli import UsageError
from ..common.matrix import RunMatrix, make_cell
from ..common.protocol import ProtocolError, load_protocol, namespace_of_seed
from ..common.results import SchemaError, sha256_file, write_json_atomic
from . import geometry as GEO
from . import predictor as PRED

PACKAGE = "t21s"
HERE = Path(__file__).resolve().parent
PROTOCOL_DEFAULT = HERE / "PROTOCOL_T21S_v2.json"
SOURCE_FILES = [HERE / "runner.py", HERE / "geometry.py", HERE / "predictor.py"]
FROZEN_NAME = "FROZEN_PREDICTOR.json"
FROZEN_SCHEMA = "mlst-local-package-frozen-predictor/1"
PHASES = ("pilot", "confirmation")
KINDS = ("grid", "all_zero", "all_strong")

# keys every cell result must carry (checked by common.cli via check_cell_schema)
RESULT_REQUIRED = {"kind": str, "scientific_checks_ok": bool}
# keys every DATA cell result must carry in addition (checked here, in run and in reduce)
DATA_RESULT_REQUIRED = {
    "d": int, "n_train": int, "k": int, "gamma": float, "sqrt_gamma": float, "chance_level": float,
    "theta": list, "theta_over_sqrt_gamma": list, "population_excess_eigenvalues": list, "c2_per_spike": list,
    "loadings_sq": list, "unique_loadings_sq": list, "predicted_recovery": list, "predicted_unique_recovery": list,
    "recovery": list, "unique_recovery": list, "unique_defined": list, "unique_norm_ratio": list,
    "abs_error": list, "abs_error_unique": list, "sum_recovery": float, "fitted_rank": int,
    "sample_top_eigenvalues": list, "gram_realised_max_abs_error": float, "edge_class": list,
}
ALGEBRAIC_RESULT_REQUIRED = {"check": str, "ok": bool, "details": dict}


class ScientificCheckError(SchemaError):
    """A package-level scientific invariant failed (exit 4, like a schema error)."""


# --------------------------------------------------------------------------- design helpers
def design(P: dict, fixture: bool) -> dict:
    """Effective design: production from the protocol, fixture from its ``fixture`` block."""
    D = P["design"]
    eff = {"d": int(D["d"]), "n_train": {k: int(v) for k, v in D["n_train"].items()}, "k": int(D["k"]),
           "J": int(D["n_directions"]), "grid_ratios": [float(x) for x in D["grid"]["theta_over_sqrt_gamma"]],
           "varied_directions": [int(x) for x in D["grid"]["varied_directions"]],
           "strong_ratio": float(D["grid"]["strong_ratio"]), "geometries": list(D["geometry"]["arms"]),
           "seeds": {"pilot": list(range(int(D["seeds"]["pilot"][0]), int(D["seeds"]["pilot"][1]) + 1)),
                     "confirmation": list(range(int(D["seeds"]["confirmation"][0]), int(D["seeds"]["confirmation"][1]) + 1))}}
    if fixture:
        F = P["fixture"]
        eff["d"] = int(F["d"])
        eff["n_train"] = {k: int(v) for k, v in F["n_train"].items()}
        eff["seeds"] = {"pilot": list(range(int(F["development_seeds"]["pilot_phase"][0]), int(F["development_seeds"]["pilot_phase"][1]) + 1)),
                        "confirmation": list(range(int(F["development_seeds"]["confirmation_phase"][0]), int(F["development_seeds"]["confirmation_phase"][1]) + 1))}
        if "grid_ratios" in F:
            eff["grid_ratios"] = [float(x) for x in F["grid_ratios"]]
        if "varied_directions" in F:
            eff["varied_directions"] = [int(x) for x in F["varied_directions"]]
    return eff


def fixture_overrides(P: dict) -> dict:
    eff = design(P, True)
    return {"d": eff["d"], "n_train": eff["n_train"], "seeds": eff["seeds"], "grid_ratios": eff["grid_ratios"],
            "varied_directions": eff["varied_directions"]}


def configurations(eff: dict) -> list[dict]:
    """Every configuration (geometry x gamma arm x kind x varied x ratio), seed-free."""
    out = []
    for geometry in eff["geometries"]:
        for arm, n in sorted(eff["n_train"].items()):
            base = {"geometry": geometry, "gamma_arm": arm, "d": eff["d"], "n_train": int(n), "k": eff["k"]}
            for j in eff["varied_directions"]:
                for r in eff["grid_ratios"]:
                    out.append({**base, "kind": "grid", "varied_direction": int(j), "ratio": float(r)})
            out.append({**base, "kind": "all_zero", "varied_direction": None, "ratio": None})
            out.append({**base, "kind": "all_strong", "varied_direction": None, "ratio": None})
    return out


def config_key(p: dict) -> str:
    j = "-" if p.get("varied_direction") is None else f"j{p['varied_direction']}"
    r = "-" if p.get("ratio") is None else f"r{p['ratio']:g}"
    return f"{p['geometry']}|{p['gamma_arm']}|N{p['n_train']}|{p['kind']}|{j}|{r}"


def _check_protocol_geometry(P: dict) -> None:
    """The protocol is the design of record; geometry.py must implement exactly it."""
    g = P["design"]["geometry"]
    if int(g["frame_seed"]) != GEO.FRAME_SEED:
        raise ProtocolError(f"protocol frame_seed {g['frame_seed']} != geometry.FRAME_SEED {GEO.FRAME_SEED}")
    for arm in g["arms"]:
        declared = np.asarray(g["gram"][arm], dtype=float)
        if declared.shape != GEO.gram(arm).shape or np.max(np.abs(declared - GEO.gram(arm))) > 0:
            raise ProtocolError(f"protocol Gram for {arm!r} differs from geometry.GRAM[{arm!r}]")
    if P["design"]["predictor"]["formula_id"] != PRED.FORMULA_ID:
        raise ProtocolError(f"protocol predictor formula_id {P['design']['predictor']['formula_id']!r} != predictor.FORMULA_ID {PRED.FORMULA_ID!r}")


def build_matrix(P: dict, fixture: bool) -> RunMatrix:
    _check_protocol_geometry(P)
    eff = design(P, fixture)
    cells = []
    for phase in PHASES:
        role = "development" if fixture else phase
        for cfg in configurations(eff):
            for seed in eff["seeds"][phase]:
                params = {**cfg, "phase": phase}
                cells.append(make_cell(role=role, seed=int(seed), params=params, n_fits=1, n_solves=1,
                                       family=f"{phase}-{cfg['geometry']}-{cfg['gamma_arm']}"))
    # algebraic cells: geometry invariants per arm (one eigen solve each) and the predictor's data-free sanity
    for geometry in eff["geometries"]:
        cells.append(make_cell(role="algebraic", seed=None, params={"check": "geometry_invariants", "geometry": geometry, "d": eff["d"], "k": eff["k"]},
                               n_fits=0, n_solves=1, family="algebraic"))
    cells.append(make_cell(role="algebraic", seed=None, params={"check": "predictor_sanity", "d": eff["d"], "k": eff["k"]}, n_fits=0, n_solves=0, family="algebraic"))
    return RunMatrix(PACKAGE, P["_sha256"], cells)


# --------------------------------------------------------------------------- per-cell science
def _recovery(P_basis: np.ndarray, w: np.ndarray) -> float:
    """||P w||^2 / ||w||^2 for the orthonormal d x k basis P_basis."""
    return float(np.sum((P_basis.T @ w) ** 2) / float(w @ w))


def fit_pca(Z: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Rank-k closed-form PCA on centred data: returns (d x k orthonormal basis, top-k sample eigenvalues of Z^T Z / N)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Zc, full_matrices=False)
    return vt[:k].T, (s[:k] ** 2) / Z.shape[0]


def draw_data(seed: int, n: int, U: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """z_i = s_i + n_i with s_i ~ N(0, K), K = U diag(theta) U^T, n_i ~ N(0, I).
    The stream order (latents first, then noise) is fixed so that the same
    seed gives the same underlying draw in every configuration with the same
    (n, d): configurations are paired by seed."""
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal((n, U.shape[1]))
    noise = rng.standard_normal((n, U.shape[0]))
    return (g * np.sqrt(theta)[None, :]) @ U.T + noise


def edge_classes(P: dict, pred_block: dict, sqrt_gamma: float) -> list[str]:
    """Data-free classification per direction: ``near_edge`` when a population
    spike carrying at least ``loading_min`` of the direction lies inside the
    declared BBP band, else ``primary``."""
    tol = P["design"]["predictor"]["tolerance"]
    lo, hi = tol["near_edge_band_theta_over_sqrt_gamma"]
    lmin = float(tol["near_edge_loading_min"])
    lam = np.asarray(pred_block["population_excess_eigenvalues"]) / sqrt_gamma
    out = []
    for row in pred_block["loadings_sq"]:
        near = any(l2 >= lmin and lo <= r <= hi for l2, r in zip(row, lam))
        out.append("near_edge" if near else "primary")
    return out


def compute_predictions(P: dict, params: dict) -> dict:
    """Everything that is a pure function of the design (no data): theta, exact K spectrum, loadings, predictions."""
    d, n, k = int(params["d"]), int(params["n_train"]), int(params["k"])
    gamma = d / n
    sg = float(np.sqrt(gamma))
    U = GEO.directions(d, params["geometry"])
    Up, ratio = GEO.unique_components(U)
    udef = [bool(r >= GEO.UNIQUE_UNDEFINED_BELOW) for r in ratio]
    theta = GEO.theta_vector(gamma=gamma, kind=params["kind"], varied_direction=params.get("varied_direction"),
                             ratio=params.get("ratio"), strong_ratio=float(P["design"]["grid"]["strong_ratio"]))
    K = GEO.signal_covariance(U, theta)
    pred = PRED.predict_all(U, Up, udef, K, gamma)
    pred.update({"theta": [float(x) for x in theta], "theta_over_sqrt_gamma": [float(x / sg) for x in theta],
                 "gamma": float(gamma), "sqrt_gamma": sg, "chance_level": k / d, "unique_defined": udef,
                 "unique_norm_ratio": [float(x) for x in ratio],
                 "gram_realised_max_abs_error": float(np.max(np.abs(U.T @ U - GEO.gram(params["geometry"]))))})
    pred["edge_class"] = edge_classes(P, pred, sg)
    return pred, U, Up, theta


def run_data_cell(P: dict, cell: dict) -> dict:
    params = cell["params"]
    d, n, k = int(params["d"]), int(params["n_train"]), int(params["k"])
    pred, U, Up, theta = compute_predictions(P, params)
    Z = draw_data(cell["seed"], n, U, theta)
    Pb, s_top = fit_pca(Z, k)                                   # the one charged fit
    rec = [_recovery(Pb, U[:, j]) for j in range(U.shape[1])]
    urec = [(_recovery(Pb, Up[:, j]) if pred["unique_defined"][j] else None) for j in range(U.shape[1])]
    res = {"kind": params["kind"], "d": d, "n_train": n, "k": k, **pred, "recovery": rec, "unique_recovery": urec,
           "abs_error": [abs(r - p) for r, p in zip(rec, pred["predicted_recovery"])],
           "abs_error_unique": [(abs(r - p) if r is not None else None) for r, p in zip(urec, pred["predicted_unique_recovery"])],
           "sum_recovery": float(np.sum(rec)), "fitted_rank": int(Pb.shape[1]),
           "sample_top_eigenvalues": [float(x) for x in s_top]}
    res["scientific_checks_ok"] = True
    check_data_result(P, cell, res)                              # raises ScientificCheckError -> charged failure
    return res


def run_algebraic_cell(P: dict, cell: dict) -> dict:
    params = cell["params"]
    if params["check"] == "geometry_invariants":
        # the spectrum identity is checked at gamma = 1 (theta_j = strong_ratio for every j); it is a property of the construction, not of N
        theta = GEO.theta_vector(gamma=1.0, kind="all_strong", varied_direction=None, ratio=None, strong_ratio=float(P["design"]["grid"]["strong_ratio"]))
        det = GEO.check_invariants(int(params["d"]), params["geometry"], theta)
        res = {"kind": "algebraic", "check": "geometry_invariants", "ok": bool(det["ok"]), "details": det}
    elif params["check"] == "predictor_sanity":
        det = {arm: PRED.sanity(gamma=params["d"] / n) for arm, n in sorted(P["design"]["n_train"].items())}
        det["formula_id"] = PRED.FORMULA_ID
        res = {"kind": "algebraic", "check": "predictor_sanity", "ok": bool(all(v["ok"] for v in det.values() if isinstance(v, dict))), "details": det}
    else:
        raise UsageError(f"unknown algebraic check {params['check']!r}")
    res["scientific_checks_ok"] = bool(res["ok"])
    if not res["ok"]:
        raise ScientificCheckError(f"{cell['cell_id']}: algebraic check {params['check']} failed: {json.dumps(det, sort_keys=True)[:400]}")
    return res


def run_cell(P: dict, cell: dict, fixture: bool, ctx: dict):
    if cell["role"] == "algebraic":
        return run_algebraic_cell(P, cell), 0, int(cell["n_solves"])
    if cell["params"]["phase"] == "confirmation":
        frozen = Path(ctx["out"]) / FROZEN_NAME
        if not frozen.exists():
            raise ScientificCheckError(f"{cell['cell_id']}: confirmation cell without {FROZEN_NAME} in the run root")
    return run_data_cell(P, cell), 1, 1


# --------------------------------------------------------------------------- scientific checks
def _close(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def check_data_result(P: dict, cell: dict, res: dict, cache: dict | None = None) -> None:
    """Invariants every data cell must satisfy; recomputes the design-only
    quantities from the cell's params and refuses disagreement.  Raises
    :class:`ScientificCheckError` (exit 4).  ``cache`` (optional, one per
    reduce call) memoises the recomputation per configuration; it is never
    shared between a run and a reduce."""
    where = cell["cell_id"]
    for k_, t in DATA_RESULT_REQUIRED.items():
        if k_ not in res:
            raise ScientificCheckError(f"{where}: data result lacks {k_!r}")
        if t is float and not isinstance(res[k_], (int, float)):
            raise ScientificCheckError(f"{where}: {k_} must be a number")
        if t in (int, list, dict) and not isinstance(res[k_], t):
            raise ScientificCheckError(f"{where}: {k_} must be {t.__name__}")
    J_ = int(P["design"]["n_directions"])
    for k_ in ("recovery", "unique_recovery", "predicted_recovery", "predicted_unique_recovery", "theta", "abs_error", "edge_class", "unique_defined"):
        if len(res[k_]) != J_:
            raise ScientificCheckError(f"{where}: {k_} must have {J_} entries")
    ck = config_key(cell["params"])
    if cache is not None and ck in cache:
        pred = cache[ck]
    else:
        pred, _, _, _ = compute_predictions(P, cell["params"])
        if cache is not None:
            cache[ck] = pred
    if res["gram_realised_max_abs_error"] > GEO.GRAM_TOLERANCE:
        raise ScientificCheckError(f"{where}: realised Gram error {res['gram_realised_max_abs_error']:.3e} > {GEO.GRAM_TOLERANCE}")
    for k_ in ("theta", "predicted_recovery", "population_excess_eigenvalues", "c2_per_spike", "unique_norm_ratio"):
        if any(not _close(a, b) for a, b in zip(res[k_], pred[k_])):
            raise ScientificCheckError(f"{where}: stored {k_} {res[k_]} disagrees with the recomputation from the exact K {pred[k_]}")
    for a, b in zip(res["predicted_unique_recovery"], pred["predicted_unique_recovery"]):
        if (a is None) != (b is None) or (a is not None and not _close(a, b)):
            raise ScientificCheckError(f"{where}: predicted_unique_recovery disagrees with the recomputation")
    if list(res["unique_defined"]) != list(pred["unique_defined"]) or list(res["edge_class"]) != list(pred["edge_class"]):
        raise ScientificCheckError(f"{where}: unique_defined / edge_class disagree with the design")
    if not _close(res["gamma"], pred["gamma"]) or not _close(res["chance_level"], pred["chance_level"]):
        raise ScientificCheckError(f"{where}: gamma / chance_level disagree with d and N")
    for j in range(J_):
        r = res["recovery"][j]
        if not (-1e-9 <= r <= 1.0 + 1e-9):
            raise ScientificCheckError(f"{where}: recovery[{j}] = {r} outside [0, 1] (unnormalised direction or wrong observable)")
        u = res["unique_recovery"][j]
        if res["unique_defined"][j]:
            if u is None or not (-1e-9 <= u <= 1.0 + 1e-9):
                raise ScientificCheckError(f"{where}: unique_recovery[{j}] = {u} undefined or outside [0, 1]")
        elif u is not None:
            raise ScientificCheckError(f"{where}: unique_recovery[{j}] must be null when the unique component is negligible")
        if not _close(res["abs_error"][j], abs(r - res["predicted_recovery"][j])):
            raise ScientificCheckError(f"{where}: abs_error[{j}] is not |recovery - predicted|")
    if cell["params"]["geometry"] == "orthogonal":
        diff = max(abs(a - b) for a, b in zip(res["recovery"], res["unique_recovery"]))
        if diff > 1e-9:
            raise ScientificCheckError(f"{where}: orthogonal geometry but unique and ordinary recovery differ by {diff:.3e}")
        if res["sum_recovery"] > res["k"] + 1e-9:
            raise ScientificCheckError(f"{where}: sum_j ||P u_j||^2 = {res['sum_recovery']} exceeds the rank {res['k']}")
    if res["fitted_rank"] != int(res["k"]):
        raise ScientificCheckError(f"{where}: fitted rank {res['fitted_rank']} != k {res['k']}")
    if not res.get("scientific_checks_ok", False):
        raise ScientificCheckError(f"{where}: scientific_checks_ok is not true")


def _beta_sd(k: int, d: int) -> float:
    """sd of ||P u||^2 for a fixed unit u and a Haar-random rank-k projector in R^d: Beta(k/2, (d-k)/2)."""
    return float(np.sqrt(2.0 * k * (d - k) / (d ** 2 * (d + 2))))


def check_all_zero_control(P: dict, data_cells: list[dict]) -> dict:
    """Pure-noise cells: PCA on N(0, I) gives a Haar-distributed subspace, so
    E||P u_j||^2 = k/d exactly.  The per-seed mean over directions must lie
    within z * sd_beta / sqrt(n_seeds) of k/d (z declared in the protocol; the
    direction average is not credited, which is conservative)."""
    z = float(P["design"]["controls"]["all_zero"]["z_tolerance"])
    out = {}
    groups: dict[str, list[dict]] = {}
    for c in data_cells:
        if c["params"]["kind"] == "all_zero":
            groups.setdefault(f"{c['params']['geometry']}|{c['params']['gamma_arm']}|N{c['params']['n_train']}", []).append(c)
    for key, cs in sorted(groups.items()):
        per_seed = [float(np.mean(c["result"]["recovery"])) for c in sorted(cs, key=lambda c: c["seed"])]
        d, k = int(cs[0]["result"]["d"]), int(cs[0]["result"]["k"])
        chance = k / d
        se = _beta_sd(k, d) / np.sqrt(len(per_seed))
        mean = float(np.mean(per_seed))
        out[key] = {"n_seeds": len(per_seed), "mean_recovery": mean, "chance_level": chance, "sd_beta_single_direction": _beta_sd(k, d),
                    "tolerance_abs": float(z * se), "z": z, "abs_deviation": abs(mean - chance), "pass": bool(abs(mean - chance) <= z * se),
                    "max_predicted_recovery": float(max(max(c["result"]["predicted_recovery"]) for c in cs))}
    return out


# --------------------------------------------------------------------------- reduction
def _group_by_config(data_cells: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for c in data_cells:
        g.setdefault(config_key(c["params"]), []).append(c)
    return {k: sorted(v, key=lambda c: c["seed"]) for k, v in sorted(g.items())}


BOOTSTRAP_REPLICATES = 1000


def _bootstrap_ci(x: np.ndarray, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES, alpha: float = 0.05) -> list[float]:
    """Fixed-seed percentile bootstrap of the mean (vectorised form of
    ``common.results.stable_bootstrap_ci``; deterministic for a given key)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(replicates, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def _seed_stats(values: list, key: str) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "se": None, "ci95": None, "undefined": True}
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    x = np.asarray(vals, dtype=float)
    return {"n": int(x.size), "mean": float(x.mean()), "sd": (float(x.std(ddof=1)) if x.size > 1 else None),
            "se": (float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else None), "ci95": _bootstrap_ci(x, seed=seed),
            "replicates": BOOTSTRAP_REPLICATES, "undefined": False}


def summarise_configs(data_cells: list[dict]) -> dict:
    out = {}
    for key, cs in _group_by_config(data_cells).items():
        r0 = cs[0]["result"]
        J_ = len(r0["recovery"])
        ent = {"params": {k: v for k, v in cs[0]["params"].items() if k != "phase"}, "phases": sorted({c["params"]["phase"] for c in cs}),
               "n_seeds": len(cs), "seeds": [c["seed"] for c in cs], "gamma": r0["gamma"], "chance_level": r0["chance_level"],
               "theta_over_sqrt_gamma": r0["theta_over_sqrt_gamma"], "population_excess_eigenvalues_over_sqrt_gamma": [x / r0["sqrt_gamma"] for x in r0["population_excess_eigenvalues"]],
               "directions": []}
        for j in range(J_):
            ent["directions"].append({
                "j": j, "predicted_recovery": r0["predicted_recovery"][j], "predicted_unique_recovery": r0["predicted_unique_recovery"][j],
                "unique_defined": r0["unique_defined"][j], "unique_norm_ratio": r0["unique_norm_ratio"][j], "edge_class": r0["edge_class"][j],
                "recovery": _seed_stats([c["result"]["recovery"][j] for c in cs], key + f"|rec{j}"),
                "unique_recovery": _seed_stats([c["result"]["unique_recovery"][j] for c in cs], key + f"|urec{j}")})
        out[key] = ent
    return out


def pilot_regime_check(P: dict, pilot_cells: list[dict]) -> dict:
    """Does the grid populate below / near / above the BBP edge?  Rule frozen in
    the protocol (``design.pilot_regime_rule``); evaluated per (geometry, gamma arm)
    on the varied direction of pilot grid cells."""
    R = P["design"]["pilot_regime_rule"]
    arms: dict[str, dict] = {}
    for c in pilot_cells:
        p = c["params"]
        if p["kind"] != "grid":
            continue
        a = arms.setdefault(f"{p['geometry']}|{p['gamma_arm']}|N{p['n_train']}", {"below": [], "near": [], "above": [], "chance_level": c["result"]["chance_level"]})
        j = int(p["varied_direction"])
        r = c["result"]
        if p["ratio"] in R["below_ratios"]:
            a["below"].append(r["unique_recovery"][j] if r["unique_recovery"][j] is not None else r["recovery"][j])
        if p["ratio"] in R["near_ratios"]:
            a["near"].append(r["recovery"][j])
        if p["ratio"] in R["above_ratios"]:
            a["above"].append(r["recovery"][j])
    out = {"rule": R, "arms": {}}
    for key, a in sorted(arms.items()):
        below_ok = bool(a["below"]) and float(np.mean(a["below"])) <= a["chance_level"] + float(R["below_max_excess_over_chance"])
        above_ok = bool(a["above"]) and float(np.mean(a["above"])) >= float(R["above_min_recovery"])
        near_ok = len(a["near"]) > 0
        out["arms"][key] = {"below_mean_unique_recovery": (float(np.mean(a["below"])) if a["below"] else None), "n_below": len(a["below"]),
                            "near_mean_recovery": (float(np.mean(a["near"])) if a["near"] else None), "n_near": len(a["near"]),
                            "above_mean_recovery": (float(np.mean(a["above"])) if a["above"] else None), "n_above": len(a["above"]),
                            "chance_level": a["chance_level"], "below_ok": below_ok, "near_ok": near_ok, "above_ok": above_ok,
                            "populated": bool(below_ok and near_ok and above_ok)}
    out["populated"] = bool(out["arms"]) and all(v["populated"] for v in out["arms"].values())
    return out


def frozen_content(P: dict, matrix: RunMatrix, regime: dict, fixture: bool) -> dict:
    """Design-only content of FROZEN_PREDICTOR.json (no confirmation data enters)."""
    tol = P["design"]["predictor"]["tolerance"]
    confs: dict[str, dict] = {}
    for c in matrix.cells:
        if c["role"] == "algebraic" or c["params"]["phase"] != "confirmation":
            continue
        key = config_key(c["params"])
        if key not in confs:
            pred, _, _, _ = compute_predictions(P, c["params"])
            confs[key] = {"params": {k: v for k, v in c["params"].items() if k != "phase"}, "theta": pred["theta"],
                          "theta_over_sqrt_gamma": pred["theta_over_sqrt_gamma"], "gamma": pred["gamma"], "chance_level": pred["chance_level"],
                          "population_excess_eigenvalues": pred["population_excess_eigenvalues"], "predicted_recovery": pred["predicted_recovery"],
                          "predicted_unique_recovery": pred["predicted_unique_recovery"], "unique_defined": pred["unique_defined"],
                          "edge_class": pred["edge_class"], "confirmation_cell_ids": [], "confirmation_seeds": []}
        confs[key]["confirmation_cell_ids"].append(c["cell_id"])
        confs[key]["confirmation_seeds"].append(c["seed"])
    return {"schema": FROZEN_SCHEMA, "package": PACKAGE, "protocol_sha256": P["_sha256"], "matrix_sha256": matrix.matrix_hash(),
            "development_only": bool(fixture),
            "predictor": {"formula_id": PRED.FORMULA_ID, "formula": PRED.FORMULA_TEXT, "coefficients_fitted_on_any_data": False,
                          "source": P["design"]["predictor"]["source"]},
            "tolerance": tol, "scoring_rule": P["design"]["predictor"]["scoring_rule"],
            "pilot_regime_check": regime, "confirmation_permitted": bool(regime["populated"]),
            "if_not_permitted": "the grid did not populate the declared regimes on pilot seeds; confirmation is refused; a changed grid is a new protocol version with provenance, never an edit",
            "n_confirmation_configurations": len(confs), "n_confirmation_cells": sum(len(v["confirmation_cell_ids"]) for v in confs.values()),
            "configurations": confs}


def _strip_time(js: dict) -> dict:
    return {k: v for k, v in js.items() if k != "frozen_utc"}


def load_frozen(out: Path, P: dict, matrix: RunMatrix) -> dict | None:
    p = Path(out) / FROZEN_NAME
    if not p.exists():
        return None
    js = json.loads(p.read_text())
    if js.get("schema") != FROZEN_SCHEMA or js.get("package") != PACKAGE:
        raise ScientificCheckError(f"{p}: not a T21-S frozen predictor file")
    if js.get("protocol_sha256") != P["_sha256"] or js.get("matrix_sha256") != matrix.matrix_hash():
        raise ScientificCheckError(f"{p}: frozen for a different protocol/matrix")
    if js["predictor"]["formula_id"] != PRED.FORMULA_ID:
        raise ScientificCheckError(f"{p}: frozen formula {js['predictor']['formula_id']!r} is not the implemented {PRED.FORMULA_ID!r}")
    return js


def score_confirmation(P: dict, frozen: dict, conf_cells: list[dict]) -> dict:
    """|seed-mean observed - frozen predicted| per (configuration, direction) against the frozen tolerance."""
    tol = frozen["tolerance"]
    rows = []
    for key, cs in _group_by_config(conf_cells).items():
        fz = frozen["configurations"].get(key)
        if fz is None:
            raise ScientificCheckError(f"confirmation configuration {key} is not in {FROZEN_NAME}")
        for c in cs:
            if c["cell_id"] not in fz["confirmation_cell_ids"]:
                raise ScientificCheckError(f"{c['cell_id']}: confirmation cell not listed in the frozen file for {key}")
            for a, b in zip(c["result"]["predicted_recovery"], fz["predicted_recovery"]):
                if not _close(a, b):
                    raise ScientificCheckError(f"{c['cell_id']}: cell prediction differs from the frozen prediction")
        J_ = len(fz["predicted_recovery"])
        for j in range(J_):
            klass = fz["edge_class"][j]
            t = float(tol["primary_abs"] if klass == "primary" else tol["near_edge_abs"])
            st = _seed_stats([c["result"]["recovery"][j] for c in cs], key + f"|rec{j}")
            row = {"config": key, "geometry": fz["params"]["geometry"], "gamma_arm": fz["params"]["gamma_arm"], "kind": fz["params"]["kind"],
                   "varied_direction": fz["params"]["varied_direction"], "ratio": fz["params"]["ratio"], "direction": j, "edge_class": klass,
                   "n_seeds": st["n"], "expected_n_seeds": len(fz["confirmation_cell_ids"]), "predicted": fz["predicted_recovery"][j],
                   "observed_mean": st["mean"], "observed_se": st["se"], "observed_ci95": st["ci95"], "abs_error_of_mean": abs(st["mean"] - fz["predicted_recovery"][j]),
                   "tolerance_abs": t, "pass": bool(abs(st["mean"] - fz["predicted_recovery"][j]) <= t),
                   "ci95_covers_prediction": (bool(st["ci95"][0] <= fz["predicted_recovery"][j] <= st["ci95"][1]) if st["ci95"] else None)}
            pu = fz["predicted_unique_recovery"][j]
            if fz["unique_defined"][j] and pu is not None:
                su = _seed_stats([c["result"]["unique_recovery"][j] for c in cs], key + f"|urec{j}")
                row.update({"unique_predicted": pu, "unique_observed_mean": su["mean"], "unique_observed_se": su["se"], "unique_observed_ci95": su["ci95"],
                            "unique_abs_error_of_mean": abs(su["mean"] - pu), "unique_pass": bool(abs(su["mean"] - pu) <= t)})
            else:
                row.update({"unique_predicted": None, "unique_observed_mean": None, "unique_abs_error_of_mean": None, "unique_pass": None, "unique_undefined": True})
            rows.append(row)
    def agg(sel):
        prim = [r for r in sel if r["edge_class"] == "primary"]
        near = [r for r in sel if r["edge_class"] == "near_edge"]
        uni = [r for r in sel if r.get("unique_pass") is not None]
        return {"n_rows": len(sel), "n_primary": len(prim), "n_primary_pass": sum(r["pass"] for r in prim),
                "primary_all_pass": bool(prim) and all(r["pass"] for r in prim),
                "n_near_edge": len(near), "n_near_edge_pass": sum(r["pass"] for r in near),
                "max_primary_abs_error": (max(r["abs_error_of_mean"] for r in prim) if prim else None),
                "n_unique_scored": len(uni), "n_unique_pass": sum(r["unique_pass"] for r in uni),
                "unique_all_pass": bool(uni) and all(r["unique_pass"] for r in uni),
                "ci95_coverage_fraction": (float(np.mean([r["ci95_covers_prediction"] for r in sel if r["ci95_covers_prediction"] is not None])) if sel else None)}
    complete = all(r["n_seeds"] == r["expected_n_seeds"] for r in rows)
    per_geo = {g: agg([r for r in rows if r["geometry"] == g]) for g in sorted({r["geometry"] for r in rows})}
    per_arm = {f"{r['geometry']}|{r['gamma_arm']}": None for r in rows}
    for k_ in per_arm:
        g, a = k_.split("|")
        per_arm[k_] = agg([r for r in rows if r["geometry"] == g and r["gamma_arm"] == a])
    verdict = {g: ("supported on synthetic Gaussian spiked data only" if v["primary_all_pass"] else "not supported") for g, v in per_geo.items()}
    # the overlap confound: unexcited direction (ratio 0) in the correlated arm, ordinary vs unique
    confound = [{"config": r["config"], "direction": r["direction"], "ordinary_observed_mean": r["observed_mean"], "ordinary_predicted": r["predicted"],
                 "unique_observed_mean": r.get("unique_observed_mean"), "unique_predicted": r.get("unique_predicted"), "chance_level": frozen["configurations"][r["config"]]["chance_level"]}
                for r in rows if r["kind"] == "grid" and r["ratio"] == 0.0 and r["varied_direction"] == r["direction"]]
    return {"rows": rows, "all_configurations_complete": complete, "overall": agg(rows), "per_geometry": per_geo, "per_arm": per_arm,
            "verdict_by_geometry": verdict, "unexcited_direction_overlap_confound": confound,
            "interpretation_rule": P["failure_interpretation"]}


def reduce_run(P: dict, matrix: RunMatrix, cells: dict, out: Path, partial: bool) -> dict:
    out = Path(out)
    fixture = any(js.get("development_only") for js in cells.values())
    data_cells = [js for js in cells.values() if js["role"] != "algebraic"]
    alg_cells = [js for js in cells.values() if js["role"] == "algebraic"]
    by_id = {c["cell_id"]: c for c in matrix.cells}
    # 1. per-cell scientific invariants (recomputed; digests were already checked by common)
    cache: dict = {}
    for js in data_cells:
        check_data_result(P, by_id[js["cell_id"]], js["result"], cache)
    for js in alg_cells:
        if not js["result"].get("ok"):
            raise ScientificCheckError(f"{js['cell_id']}: algebraic check {js['result'].get('check')} recorded as failed")
    # 2. phase completeness (the common 'partial' flag does not know about phases)
    phases = {}
    for ph in PHASES:
        exp = [c["cell_id"] for c in matrix.cells if c["role"] != "algebraic" and c["params"]["phase"] == ph]
        obs = [k for k in exp if k in cells]
        phases[ph] = {"expected": len(exp), "observed": len(obs), "complete": len(obs) == len(exp)}
    exp_alg = [c["cell_id"] for c in matrix.cells if c["role"] == "algebraic"]
    phases["algebraic"] = {"expected": len(exp_alg), "observed": sum(k in cells for k in exp_alg), "complete": all(k in cells for k in exp_alg)}
    pilot_cells = [js for js in data_cells if js["params"]["phase"] == "pilot"]
    conf_cells = [js for js in data_cells if js["params"]["phase"] == "confirmation"]
    # 3. controls (statistical control on the pure-noise cells; refuses when violated)
    controls = {"all_zero": check_all_zero_control(P, data_cells)}
    bad = [k for k, v in controls["all_zero"].items() if not v["pass"]]
    if bad:
        raise ScientificCheckError(f"all-zero control not at chance k/d in {bad}: {json.dumps({k: controls['all_zero'][k] for k in bad})[:600]}")
    orth = [js for js in data_cells if js["params"]["geometry"] == "orthogonal"]
    corr = [js for js in data_cells if js["params"]["geometry"] == "correlated" and js["params"]["kind"] != "all_zero"]
    controls["orthogonal_unique_equals_ordinary"] = {"n_cells": len(orth), "max_abs_diff": (max(max(abs(a - b) for a, b in zip(js["result"]["recovery"], js["result"]["unique_recovery"])) for js in orth) if orth else None)}
    controls["correlated_unique_differs"] = {"n_cells": len(corr), "max_abs_diff": (max(max(abs(a - b) for a, b in zip(js["result"]["recovery"], js["result"]["unique_recovery"])) for js in corr) if corr else None),
                                             "note": "descriptive: in the correlated arm the two recoveries are different observables; a difference is expected, not scored"}
    controls["all_strong"] = {k: {"n_seeds": v["n_seeds"], "mean_recovery_by_direction": [d_["recovery"]["mean"] for d_ in v["directions"]],
                                  "predicted_by_direction": [d_["predicted_recovery"] for d_ in v["directions"]]}
                              for k, v in summarise_configs([js for js in data_cells if js["params"]["kind"] == "all_strong"]).items()}
    summary = {"phase_completeness": phases, "controls": controls, "algebraic": {js["result"]["check"]: js["result"]["ok"] for js in alg_cells},
               "configurations": summarise_configs(data_cells),
               "historical_verdicts_preserved": "All five registered T21 predictions (archive, 11 Sep 2026) remain NOT met; this package tests the post-hoc spike/loadings explanation on independent geometry and cannot repair those verdicts.",
               "status_vocabulary": "development_only fixture output" if fixture else "executed (not yet validated)"}
    # 4. pilot: regime check and the freeze
    frozen = load_frozen(out, P, matrix)
    if conf_cells and frozen is None:
        # the run-time guard required the frozen file before any confirmation cell was drawn, so its absence now
        # means the run root was altered; it is not silently regenerated from the (complete) pilot
        raise ScientificCheckError(f"confirmation cells present but no {FROZEN_NAME} in the run root; the file that permitted them is gone")
    if pilot_cells:
        regime = pilot_regime_check(P, pilot_cells)
        summary["pilot"] = {"n_cells": len(pilot_cells), "regime_check": regime, "complete": phases["pilot"]["complete"]}
        if phases["pilot"]["complete"]:
            content = frozen_content(P, matrix, regime, fixture)
            if frozen is None:
                content["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                write_json_atomic(out / FROZEN_NAME, content)
                frozen = json.loads((out / FROZEN_NAME).read_text())
                summary["frozen_predictor_written_now"] = True
            elif _strip_time(frozen) != json.loads(json.dumps(content, sort_keys=True)):
                raise ScientificCheckError(f"{FROZEN_NAME} in the run root differs from what this pilot would freeze; it is not rewritten")
    summary["frozen_predictor"] = ({"path": str(out / FROZEN_NAME), "sha256": sha256_file(out / FROZEN_NAME), "confirmation_permitted": frozen["confirmation_permitted"],
                                    "formula_id": frozen["predictor"]["formula_id"], "frozen_utc": frozen.get("frozen_utc")} if frozen else None)
    # 5. confirmation: scored against the frozen file only
    if conf_cells:
        if not frozen["confirmation_permitted"]:
            raise ScientificCheckError("confirmation cells present although the frozen file does not permit confirmation")
        summary["confirmation"] = {"n_cells": len(conf_cells), "complete": phases["confirmation"]["complete"], **score_confirmation(P, frozen, conf_cells)}
    return summary


# --------------------------------------------------------------------------- entry point
def _confirmation_guard(argv) -> int | None:
    """Refuse (exit 6, nothing charged) a ``run`` that would draw any
    confirmation-phase cell while the run root has no permitting FROZEN_PREDICTOR.json."""
    ap = cli.build_parser(PACKAGE, "guard")
    args = ap.parse_args(argv)
    if args.cmd != "run" or not Path(args.protocol).exists():
        return None                                  # main_template reports a missing protocol
    P = load_protocol(args.protocol)
    if P.get("package") != PACKAGE:
        return None                                  # main_template reports the mismatch
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
        print(f"refusing: {frozen_path} records that the pilot did not populate the declared regimes; confirmation is not permitted "
              f"under this protocol version (nothing was charged)", file=sys.stderr)
        return 6
    return None


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
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
    rc = cli.main_template(package=PACKAGE, description="T21-S: independent geometry and excitation (L3)",
                           build_matrix=build_matrix, run_cell=run_cell, reduce_run=reduce_run,
                           result_required=RESULT_REQUIRED, source_files=SOURCE_FILES,
                           fixture_overrides=fixture_overrides, argv=argv)
    if rc == 0 and argv and argv[0] == "estimate":
        _project_estimate_to_production(argv)
    return rc


def _project_estimate_to_production(argv) -> None:
    """The common estimate multiplies measured per-cell times by the declared
    scale and by the cell counts of the matrix *of the run root*; on a fixture
    root those are fixture counts.  Add the projection onto the production
    matrix (same family names, production cell counts) to RESOURCE_ESTIMATE.json."""
    args = cli.build_parser(PACKAGE, "estimate").parse_args(argv)
    P = load_protocol(args.protocol)
    path = Path(args.out) / "RESOURCE_ESTIMATE.json"
    est = json.loads(path.read_text())
    prod = build_matrix(P, False)
    counts = prod.counts()
    fams = {}
    total = 0.0
    unmeasured = []
    for fam, info in counts["by_family"].items():
        m = est["families"].get(fam)
        if not m or not m.get("measured"):
            unmeasured.append(fam)
            fams[fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": False}
            continue
        per = float(m["production_seconds_per_cell"])
        fams[fam] = {"cells": info["cells"], "fits": info["fits"], "solves": info["solves"], "measured": True,
                     "production_seconds_per_cell": per, "production_cpu_seconds": per * info["cells"]}
        total += per * info["cells"]
    est["production_projection"] = {
        "matrix_sha256": prod.matrix_hash(), "counts": counts, "families": fams, "unmeasured_families": unmeasured,
        "total_cpu_seconds_point": total, "total_cpu_hours_point": total / 3600.0,
        "wall_seconds_at_W_workers": "total_cpu_seconds_point / <WORKERS> (embarrassingly parallel by cell; ledger appends serialised by flock)",
        "retry_headroom": {"fits_planned": counts["fits"], "cap": P["budget"]["estimator_fits_cap"],
                           "fits_headroom_under_cap": P["budget"]["estimator_fits_cap"] - counts["fits"],
                           "cpu_seconds_if_all_headroom_used": (total / counts["fits"]) * (P["budget"]["estimator_fits_cap"] - counts["fits"]) if counts["fits"] else None},
        "note": "per-cell seconds = measured fixture seconds x budget.fixture_to_production_scale (a flop-count ratio; fixture cells are overhead-dominated, so this projection is an upper bound); reduce time is not included (a reduce of the full production root recomputes 104 exact-K spectra and 2 x 416 bootstraps)",
    }
    write_json_atomic(path, est)
    print(json.dumps({"production_projection": {k: est["production_projection"][k] for k in ("total_cpu_seconds_point", "total_cpu_hours_point", "unmeasured_families", "retry_headroom")}}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
