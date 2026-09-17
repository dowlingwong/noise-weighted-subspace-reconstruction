"""R6 / TIDMAD readiness: the matched design, exposure gate and the two named contrasts.

This module prepares a matched whole-window versus narrowband comparison on the
SAME independent, unexposed files, with the SAME calibration and evaluation units,
rank, labels and scorer.  It reuses the archived T22 interface
(``successor_20260913.t22s.matched_stage``) unchanged for the exposure inventory,
the synthetic fixture and the canonical matched scoring; it adds only the
weighted-residual contrast that the archived stage never computed.

Honesty boundary
----------------
No independent TIDMAD file is present in this environment and no release file list
exists in the archive, so the matched design is **declared, not executed**: every
confirmatory entry point is DATA-BLOCKED.  The estimator wiring is exercised on a
synthetic HDF5 fixture and is labelled
``"synthetic-fixture, not real data, not validated on real inputs"``.  A weighted
residual identity is applied only where its assumptions hold (fixed equal-trace
affine maps, unit noise covariance, vanishing signal/noise cross moment); the
physical endpoint remains empirical and is never asserted from the identity.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026 import tidmad_band as tb  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402
from experiments.mlst2026.successor_20260913.common import fitcount  # noqa: E402
from experiments.mlst2026.successor_20260913.t22s import matched_stage as ms  # noqa: E402

NON_REAL_LABEL = "synthetic-fixture, not real data, not validated on real inputs"
CONFIRMATORY_BLOCKED = "DATA-BLOCKED"

#: the two P3 contrasts are scored on the same evaluation units; the methods being
#: ordered are the two loss geometries (classic PCA versus weighted EMPCA).
WEIGHTED_RESIDUAL_CONTRAST = ("pca", "empca")
TRUTH_REFERENCED_CONTRAST = ("pca", "empca")
#: usefulness against the named baselines is a THIRD question, reported separately.
USEFULNESS_BASELINES = (("zero", "empca"), ("calibration-mean", "empca"), ("raw-bin", "empca"))

#: The whitened-risk identity and the exact conditions under which it may be applied.
RISK_IDENTITY = {
    "identity": "R_clean,Sigma = R_obs + 2 tr(B) - d",
    "applies_when": ("fixed equal-trace affine maps, evaluation noise with E[ww^T]=I and E[tw^T]=0, and a "
                     "correct covariance; a population statement conditional on the fitted map"),
    "applied_to_learned_maps": False,
    "physical_endpoint": "remains empirical; the identity does not imply a physical-parameter ordering",
}


class ConfirmatoryBlocked(RuntimeError):
    """The confirmatory matched comparison is data-blocked; the reasons are attached."""

    def __init__(self, reasons, detail=None):
        self.reasons = list(reasons)
        self.detail = detail or {}
        super().__init__("DATA-BLOCKED: " + "; ".join(self.reasons))


# --- (i) matched design declaration ---------------------------------------------------------
def matched_design(P: dict) -> dict:
    """Declare the matched comparison without opening any instrument file."""
    d = P["tidmad"]["matched_design"]
    return {
        "instrument": "tidmad",
        "design_id": d["design_id"],
        "representations": list(d["representations"]),
        "same_units": ["cal_read", "cal_ref", "fit_read", "eval_read", "eval_ref"],
        "rank": int(d["rank"]),
        "band_width": int(d["band_width"]),
        "band_selected_here": False,
        "band_selection_note": ("the band is an input to matched scoring; selecting it on development data is a "
                                "separate step and is NOT part of matched scoring"),
        "calibration": "ONE transfer calibration from cal_read/cal_ref, shared by both representations",
        "labels": "ONE label array t_eval = cal.g * eval_ref[:, k], shared by both representations",
        "scoring": {"weighted_residual": "held-out whitened observation residual |q - eval_read|^2 / N_k",
                    "truth_referenced": "held-out squared error against the injected reference (te.squared_errors)",
                    "usefulness": "same units against zero, calibration-mean and raw-bin baselines"},
        "independent_unit": {
            "unit": "one release file the archive never opened",
            "requirement": ("both representations must use the SAME independent, unexposed files; a never-read "
                            "window of an already examined file is NOT an independent unit and no new window cut "
                            "from a previously examined file may be represented as a new independent file"),
            "resolved": False,
            "why_unresolved": ("the archive names the files it opened but does not contain the release file list, "
                               "so the set of release files not yet exposed cannot be derived here"),
        },
        "p3": {"hypothesis": "P3",
               "statement": ("where an injected reference is recorded, the reconstruction preferred by the "
                             "weighted residual also has the lower clean-signal error in the named metric"),
               "weighted_residual_contrast": {"a": WEIGHTED_RESIDUAL_CONTRAST[0], "b": WEIGHTED_RESIDUAL_CONTRAST[1]},
               "truth_referenced_contrast": {"a": TRUTH_REFERENCED_CONTRAST[0], "b": TRUTH_REFERENCED_CONTRAST[1]},
               "tested": False,
               "note": ("scored on the same evaluation units; usefulness against zero/calibration-mean/raw-bin is "
                        "a separate question and is not P3's antecedent")},
        "risk_identity": dict(RISK_IDENTITY),
        "controls": ["non-learned/B=1 identity", "covariance adequacy", "whitened-risk identity arithmetic"],
        "executable_here": False,
        "requires": ("independent unexposed release files, the release file list, a complete verified simulation "
                     "calibration and a non-inconclusive development dependence gate"),
        "label": NON_REAL_LABEL,
    }


# --- (ii) exposure inventory + gate ---------------------------------------------------------
def exposure_inventory(archive_roots=None) -> dict:
    """Reuse the archived exposure inventory; it opens no instrument data."""
    return ms.exposure_inventory() if archive_roots is None else ms.exposure_inventory(archive_roots)


def exposure_inventory_check(inv: dict) -> dict:
    """Gate the matched design on the exposure inventory.

    The load-bearing check is that the set of release files the archive never opened
    is *resolved*; the archive does not contain the release file list, so it is not,
    and a matched comparison cannot claim independent units.  A region of an already
    examined file is explicitly not an independent unit.
    """
    rfl = inv["release_file_list"]
    unresolved = rfl.get("files_not_yet_exposed") is None
    checks = {
        "release_file_list_derivable_from_archive": {"passed": bool(rfl.get("derivable_from_archive")),
                                                     "value": bool(rfl.get("derivable_from_archive"))},
        "independent_file_set_resolved": {"passed": not unresolved,
                                          "value": rfl.get("files_not_yet_exposed")},
        "files_known_to_archive": len(inv.get("files_known_to_archive", [])),
        "regions_recorded": len(inv.get("regions", [])),
    }
    blocked = unresolved
    return {
        "blocked": blocked,
        "status": CONFIRMATORY_BLOCKED if blocked else "ready",
        "checks": checks,
        "release_file_list_statement": rfl.get("statement"),
        "independent_units_note": inv.get("independent_units_note"),
        "note": ("no new windows cut from previously examined files may be represented as new independent "
                 "files; only files the archive never opened can count"),
    }


def confirmatory_gate(P: dict, *, inventory: dict, n_available_files=None, calibration: dict | None = None,
                      dependence_verdict: str | None = None, feasibility_rules: dict | None = None) -> dict:
    """Block confirmatory execution unless every declared gate is satisfied.

    Missing files, an incomplete/infeasible simulation calibration, and an
    inconclusive development gate each block.  Returns a DATA-BLOCKED status with the
    reasons; it never fabricates a file count or a calibration.
    """
    reasons: list[str] = []
    inv_check = exposure_inventory_check(inventory)
    if not inv_check["checks"]["independent_file_set_resolved"]["passed"]:
        reasons.append("independent unexposed file set unresolved: the archive does not contain the release "
                       "file list, so the files the archive never opened cannot be named here")
    if n_available_files is None:
        reasons.append("no independent file count is available (no release file list and no instrument files "
                       "present in this environment)")
        n_for_feasibility = 0
    else:
        n_for_feasibility = int(n_available_files)
    rules = feasibility_rules or P["tidmad"]["feasibility"]
    feasibility = ms.feasibility(calibration, n_for_feasibility, rules=rules,
                                 dependence_verdict=dependence_verdict)
    if not feasibility:
        reasons.extend(feasibility["reasons"])
    blocked = bool(reasons)
    return {
        "blocked": blocked,
        "status": CONFIRMATORY_BLOCKED if blocked else "ready",
        "reasons": reasons,
        "inventory_check": inv_check,
        "feasibility": dict(feasibility),
        "n_available_files": n_available_files,
        "calibration_complete": bool(calibration and calibration.get("complete")),
        "dependence_verdict": dependence_verdict,
        "statement": ("a matched rerun on already exposed data must be labelled accordingly; no outcome here "
                      "establishes the adequacy of the archived inference on unexamined instrument data"),
    }


# --- (iii) the two named contrasts on the same units ---------------------------------------
def weighted_observation_residual(q: np.ndarray, observed: np.ndarray, noise_power: float) -> np.ndarray:
    """Held-out whitened observation residual at the injection bin: ``|q - x|^2 / N_k``."""
    n = max(float(noise_power), np.finfo(float).tiny)
    return np.abs(np.asarray(q) - np.asarray(observed)) ** 2 / n


def _contrast(metric: dict, a: str, b: str) -> dict:
    d = metric[a] - metric[b]
    mean = float(np.mean(d))
    return {"a": a, "b": b, "mean_delta": mean, "sd_per_window": float(np.std(d, ddof=1)),
            "n_windows": int(d.size), "preferred": (b if mean > 0 else a),
            "direction": ("b lower" if mean > 0 else "a lower")}


def matched_contrasts(stretch: dict, *, k_inj: int, B: int, w: int, N_bins: np.ndarray,
                      exposure_status: str = "synthetic-fixture") -> dict:
    """Score the P3 contrasts and the usefulness baselines on identical evaluation units.

    Calls the archived :func:`matched_stage.matched_scores` for the canonical shared
    unit declaration and the truth-referenced contrasts, then computes the
    weighted-residual contrast by re-invoking the SAME estimators on the SAME inputs
    (the archived entry points do not return the per-window estimates, so this second
    pass is the only way to whitened-score the identical units).  Every output is
    labelled with ``exposure_status`` and the non-real label.
    """
    if exposure_status not in ms.EXPOSURE_LABELS:
        raise ValueError(f"exposure_status must be one of {ms.EXPOSURE_LABELS}, got {exposure_status!r}")
    for key in ("cal_read", "cal_ref", "fit_read", "eval_read", "eval_ref"):
        if key not in stretch:
            raise ValueError(f"stretch lacks {key!r}; pass the dict load_stretch returns")
    k = int(k_inj)
    N_bins = np.asarray(N_bins, dtype=float)

    stage = ms.matched_scores(stretch, k_inj=k, B=int(B), w=int(w), N_bins=N_bins,
                              exposure_status=exposure_status)

    cal = te.fit_transfer(stretch["cal_read"][:, k], stretch["cal_ref"][:, k], float(N_bins[k]))
    est_ww, _diag_ww = te.compute_estimates(stretch["eval_read"], stretch["fit_read"], k, int(w), N_bins, cal)
    fitcount.count_completed("whole-window-estimator", n=len(est_ww))
    est_b, _diag_b = tb.compute_band_estimates(stretch["eval_read"], stretch["fit_read"], k, int(B), int(w),
                                               N_bins, cal)
    fitcount.count_completed("band-estimator", n=len(est_b))
    reps = {"whole-window": est_ww, "band": est_b}
    observed = stretch["eval_read"][:, k]
    weighted = {rep: {m: weighted_observation_residual(q, observed, float(N_bins[k])) for m, q in est.items()}
                for rep, est in reps.items()}

    residual = {rep: _contrast(weighted[rep], *WEIGHTED_RESIDUAL_CONTRAST) for rep in reps}
    truth = {}
    for rep in reps:
        c = stage["per_representation"][rep]["contrasts"][
            f"{TRUTH_REFERENCED_CONTRAST[0]}|{TRUTH_REFERENCED_CONTRAST[1]}"]
        truth[rep] = {"a": TRUTH_REFERENCED_CONTRAST[0], "b": TRUTH_REFERENCED_CONTRAST[1],
                      "mean_delta": float(c["mean_delta_squared"]),
                      "sd_per_window": float(c["sd_per_window"]), "n_windows": int(c["n_windows"]),
                      "preferred": (TRUTH_REFERENCED_CONTRAST[1] if c["mean_delta_squared"] > 0
                                    else TRUTH_REFERENCED_CONTRAST[0])}
        residual[rep]["cross_check_archived_truth_mean_delta"] = float(c["mean_delta_squared"])
    agreement = {rep: bool(np.sign(residual[rep]["mean_delta"]) == np.sign(truth[rep]["mean_delta"])) for rep in reps}

    e2 = {rep: {m: te.squared_errors(q, cal.g * stretch["eval_ref"][:, k], int(w)) for m, q in est.items()}
          for rep, est in reps.items()}
    usefulness = {f"{a}|{b}": {rep: _contrast(e2[rep], a, b) for rep in reps}
                  for a, b in USEFULNESS_BASELINES}

    shared_index = stage["shared"]["evaluation_window_index"]
    return {
        "exposure_status": exposure_status,
        "label": NON_REAL_LABEL,
        "representation_is_the_only_difference": True,
        "band_width": int(B), "band_selected_here": False,
        "k_inj": k, "window_samples": int(w),
        "p3": {
            "hypothesis": "P3",
            "tested": True,
            "tested_on_real_data": False,
            "real_data_tested": False,
            "weighted_residual_contrast": residual,
            "truth_referenced_contrast": truth,
            "ordering_agrees": agreement,
            "note": ("tested on the synthetic fixture only; P3 is a named-metric ordering claim, not a "
                     "usefulness claim, and neither archived TIDMAD stage computed this matched contrast"),
        },
        "usefulness_against_baselines": usefulness,
        "whole_window_vs_band_same_method": stage["whole_window_vs_band_same_method"],
        "shared": {**stage["shared"], "same_units_for_both_representations": True,
                   "our_evaluation_window_index_len": len(shared_index)},
        "per_representation_units": {rep: {"evaluation_window_index": stage["per_representation"][rep]["evaluation_window_index"],
                                           "label_sha256": stage["per_representation"][rep]["label_sha256"]}
                                     for rep in reps},
        "risk_identity": dict(RISK_IDENTITY),
        "n_fits": int(stage["n_fits"]) + len(est_ww) + len(est_b),
        "counting_convention": ("the archived matched interface charges one fit per estimator per "
                                "representation; the weighted-residual pass re-invokes the same estimators on "
                                "the identical inputs and is charged separately and reported here"),
    }


# --- (iv) controls (synthetic / arithmetic only) --------------------------------------------
def control_risk_identity(*, seed: int = 20260917, d: int = 16, k: int = 4,
                          n: int = 200_000) -> dict:
    """Arithmetic check of ``R_clean = R_obs + 2 tr(B) - d`` for fixed equal-trace maps.

    Synthetic unit-noise draw in whitened coordinates; a fixed rank-k orthogonal
    projector ``B`` and the identity ``B=I``.  This is a methodological control on the
    identity's assumptions, NOT a real-data result.
    """
    rng = np.random.default_rng(int(seed))
    t = rng.normal(size=(int(n), int(d)))
    w = rng.normal(size=(int(n), int(d)))
    y = t + w
    Q, _ = np.linalg.qr(rng.normal(size=(int(d), int(k))))
    out = {}
    for label, B in (("rank_k_orthogonal", Q @ Q.T), ("identity", np.eye(int(d)))):
        yhat = y @ B
        r_obs = float(np.mean(np.sum((y - yhat) ** 2, axis=1)))
        r_clean = float(np.mean(np.sum((t - yhat) ** 2, axis=1)))
        correction = float(2.0 * np.trace(B) - int(d))
        pred = r_obs + correction
        out[label] = {"r_obs": r_obs, "r_clean": r_clean, "correction": correction,
                      "r_clean_predicted": pred, "abs_error": abs(r_clean - pred),
                      "passed": bool(abs(r_clean - pred) <= 0.05 * max(1.0, abs(correction)))}
    return {"d": int(d), "k": int(k), "n": int(n), "arms": out,
            "passed": all(v["passed"] for v in out.values()),
            "label": "methodological control on synthetic draws; not real data",
            "note": ("the identity is population-level, conditional on the fixed map, and does not imply a "
                     "physical-parameter ordering")}


def covariance_adequacy(stretch: dict, *, k_inj: int, N_bins: np.ndarray) -> dict:
    """Report whether the whitened calibration residual is consistent with unit noise.

    On a synthetic fixture this is wiring only; on real inputs it would be a
    necessary (not sufficient) adequacy diagnostic for the noise covariance.
    """
    k = int(k_inj)
    N_bins = np.asarray(N_bins, dtype=float)
    cal = te.fit_transfer(stretch["cal_read"][:, k], stretch["cal_ref"][:, k], float(N_bins[k]))
    resid = stretch["cal_read"][:, k] - cal.g * stretch["cal_ref"][:, k]
    whitened = np.abs(resid) ** 2 / max(float(N_bins[k]), np.finfo(float).tiny)
    finite = bool(np.all(np.isfinite(whitened)))
    return {
        "n_calibration_windows": int(whitened.size),
        "whitened_residual_mean_power": float(np.mean(whitened)),
        "whitened_residual_sd_power": float(np.std(whitened, ddof=1)),
        "finite": finite,
        "verdict": ("diagnostic only: a finite whitened residual mean is necessary, not sufficient, for "
                    "covariance adequacy; on a synthetic fixture this is wiring and nothing more"),
        "label": NON_REAL_LABEL,
    }


def assert_b1_identity(stretch: dict, *, k_inj: int, w: int, N_bins: np.ndarray) -> dict:
    """Non-learned B = 1 identity control: band-PCA = band-EMPCA = bin read exactly."""
    k = int(k_inj)
    N_bins = np.asarray(N_bins, dtype=float)
    cal = te.fit_transfer(stretch["cal_read"][:, k], stretch["cal_ref"][:, k], float(N_bins[k]))
    est_b1, diag1 = tb.compute_band_estimates(stretch["eval_read"], stretch["fit_read"], k, 1, int(w), N_bins, cal)
    fitcount.count_completed("b1-identity-estimator", n=len(est_b1))
    ident = tb.assert_single_bin_identity(diag1, "B=1 R6 fixture", tb.IDENTITY_RTOL)
    return {**ident, "label": NON_REAL_LABEL}
