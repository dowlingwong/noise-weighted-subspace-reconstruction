"""R2 analysis: the two endpoints, fresh-seed model selection, offset-safe
reporting, and the simultaneous band over the extended N grid.

This module is the NEW code of R2.  It imports the archived T20-S analysis
primitives unchanged (``successor_20260913.t20s.analysis``) where they are the
right tool -- the sign profile is exactly the archived
:func:`...analysis.sign_profile` -- and implements the parts the R2 protocol
changes:

* the two endpoints
  (a) fresh-seed evaluation of the OLD t20s frozen prediction with the offset
      applied exactly once, and
  (b) a model selected only on fresh pilot seeds and frozen before
      confirmation, with ``"none"`` retained as a valid outcome;
* the extended N grid and the explicit old-grid / extrapolation distinction;
* offset-safe reporting: a prediction is always rebuilt from the original
  ``(c, d)`` and the current algebraic ``a_ref``; a stored offset-bearing
  prediction is never used as a base.

Offset convention (the September diagnosis)
-------------------------------------------
The t20s reducer stored ``prediction_at_grid`` already including ``a_ref``
(``successor_20260913/t20s/runner.py:371: pred = pred + a_best_found``).  A
first draft of the September diagnosis then subtracted ``a_ref`` from the
observed contrast and produced spurious residuals.  R2 therefore:

* keeps the ORIGINAL ``(c, d)`` of the frozen model;
* computes ``base(N) = c / N + d / N**2`` (the model of ``d - a_ref``);
* adds the CURRENT algebraic ``a_ref`` exactly once: ``pred = base + a_ref``;
* never adds ``a_ref`` twice and never starts from a stored prediction.

:func:`old_base_prediction` and :func:`old_endpoint_prediction` are the only
places a d-prediction is formed; :func:`check_offset_applied_once` is the
numerical guard, called at reduce and in the tests.

Nothing here reads files or launches work on import.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_ROOT = Path(__file__).resolve().parents[4]
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.mlst2026.successor_20260913.t20s import analysis as OLD_AN  # noqa: E402  (archived, unchanged)

__all__ = [
    "CANDIDATES", "MIN_RELATIVE_IMPROVEMENT", "OLD_GRID", "EXTENDED_GRID", "OLD_GRID_MAX",
    "fit_model", "predict", "split_halves", "select_model", "old_selection_margins",
    "simultaneous_band", "band_pvalue", "score_prediction", "holm",
    "old_base_prediction", "old_endpoint_prediction", "archived_sign_profile",
]

#: The four archived scaling candidates, retained for comparability.
#: ``none`` is always a valid outcome and is never repaired on confirmation.
CANDIDATES = ("none", "c_over_N", "c_over_N_alpha", "c_over_N_d_over_N2")

#: A parametric candidate must beat ``none`` by at least this relative margin in
#: the fresh pilot cross-validation before it is frozen.  Fixed before any R2
#: data was drawn; NOT derived from the archived 2.7% margin.
MIN_RELATIVE_IMPROVEMENT = 0.01

OLD_GRID = (400, 800, 1000, 1250, 1600, 3200)
EXTENDED_GRID = (6400, 12800)
OLD_GRID_MAX = 3200


# --------------------------------------------------------------------------------------
# scaling models (new code; the archived candidate set is kept, the rule is R2's)
# --------------------------------------------------------------------------------------
def fit_model(model: str, N: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of ``model`` to per-N values ``y`` (one number per N)."""
    N = np.asarray(N, float)
    y = np.asarray(y, float)
    if model == "none":
        return {"model": model, "N": N.tolist(), "y": y.tolist()}
    if model == "c_over_N":
        x = 1.0 / N
        return {"model": model, "c": float(np.dot(x, y) / np.dot(x, x))}
    if model == "c_over_N_alpha":
        def c_of(alpha: float) -> float:
            x = N ** (-alpha)
            return float(np.dot(x, y) / np.dot(x, x))

        def sse(alpha: float) -> float:
            x = N ** (-alpha)
            return float(np.sum((y - c_of(alpha) * x) ** 2))

        from scipy.optimize import minimize_scalar
        res = minimize_scalar(sse, bounds=(0.0, 4.0), method="bounded", options={"xatol": 1e-8})
        alpha = float(res.x)
        return {"model": model, "c": c_of(alpha), "alpha": alpha}
    if model == "c_over_N_d_over_N2":
        X = np.column_stack([1.0 / N, 1.0 / N ** 2])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {"model": model, "c": float(coef[0]), "d": float(coef[1])}
    raise ValueError(f"unknown model {model!r}")


def predict(params: dict, N: np.ndarray) -> np.ndarray:
    N = np.asarray(N, float)
    m = params["model"]
    if m == "none":
        table = dict(zip(params["N"], params["y"]))
        return np.array([table[float(n)] if float(n) in table else np.nan for n in N])
    if m == "c_over_N":
        return params["c"] / N
    if m == "c_over_N_alpha":
        return params["c"] * N ** (-params["alpha"])
    if m == "c_over_N_d_over_N2":
        return params["c"] / N + params["d"] / N ** 2
    raise ValueError(m)


def split_halves(seeds) -> tuple[list[int], list[int]]:
    """Deterministic split of the sorted seeds into halves (first gets the odd seed)."""
    s = sorted(int(x) for x in seeds)
    h = (len(s) + 1) // 2
    return s[:h], s[h:]


def select_model(Y: dict, N_grid, seeds, *, min_relative_improvement: float = MIN_RELATIVE_IMPROVEMENT) -> dict:
    """Two-fold out-of-sample selection of the R2 NEW model.

    ``Y[theta_index]`` is an array ``(n_seeds, n_N)`` of per-seed targets ordered
    like ``seeds`` and ``N_grid``.  A parametric candidate beats ``none`` only if
    its summed holdout MSE is strictly smaller AND the relative improvement over
    ``none`` is at least ``min_relative_improvement`` (the declared complexity
    guard).  Otherwise ``"none"`` is selected.  This function is R2's own; the
    archived ``cv_select`` is deliberately not called.
    """
    A, B = split_halves(seeds)
    seeds_sorted = sorted(int(s) for s in seeds)
    idx = {s: i for i, s in enumerate(seeds_sorted)}
    N = np.asarray(N_grid, float)
    err = {m: 0.0 for m in CANDIDATES}
    n_terms = 0
    detail: dict[str, list] = {}
    for th_i, Yt in sorted(Y.items()):
        Yt = np.asarray(Yt, float)
        for fit_seeds, hold_seeds in ((A, B), (B, A)):
            if not fit_seeds or not hold_seeds:
                continue
            yfit = Yt[[idx[s] for s in fit_seeds]].mean(axis=0)
            Yhold = Yt[[idx[s] for s in hold_seeds]]
            for m in CANDIDATES:
                p = fit_model(m, N, yfit)
                pr = predict(p, N)
                e = float(np.mean((Yhold - pr[None, :]) ** 2))
                err[m] += e
                detail.setdefault(m, []).append({"theta_index": int(th_i), "fit_seeds": fit_seeds,
                                                 "holdout_seeds": hold_seeds, "mse": e})
            n_terms += 1
    if n_terms == 0:
        return {"selected": "none", "cv_error": err, "reason": "no holdout fold could be formed",
                "folds": 0, "best_parametric": None, "relative_improvement": None}
    param = [m for m in CANDIDATES if m != "none"]
    best = min(param, key=lambda m: (err[m], CANDIDATES.index(m)))
    denom = err["none"] if err["none"] > 0 else 0.0
    rel = (err["none"] - err[best]) / denom if denom > 0 else 0.0
    if err[best] < err["none"] and rel >= float(min_relative_improvement):
        selected = best
        reason = (f"best parametric candidate {best} beats none by {rel:.4f} (>= complexity guard "
                  f"{min_relative_improvement}) out of sample")
    else:
        selected = "none"
        reason = (f"best parametric candidate {best} did not beat none by the declared complexity guard "
                  f"{min_relative_improvement} (relative improvement {rel:.4f}); frozen 'none'")
    return {"selected": selected, "cv_error": {m: float(v) for m, v in err.items()}, "folds": n_terms,
            "fit_half": A, "holdout_half": B, "best_parametric": best,
            "relative_improvement": float(rel), "reason": reason, "detail": detail}


def old_selection_margins(cv_errors: dict) -> dict:
    """The two archived pilot selection margins, kept distinct.

    ``margin_vs_none`` is the margin that was reported as 2.7%; the margin over
    ``c/N^alpha`` is a DIFFERENT, smaller quantity.  Conflating them is exactly
    the reporting error the September diagnosis warns about.
    """
    none = float(cv_errors["none"])
    sel = float(cv_errors["c_over_N_d_over_N2"])
    alpha = float(cv_errors["c_over_N_alpha"])
    return {
        "selected": "c_over_N_d_over_N2",
        "margin_vs_none_abs": none - sel,
        "margin_vs_none_rel": (none - sel) / none if none else None,
        "margin_vs_c_over_N_alpha_abs": alpha - sel,
        "margin_vs_c_over_N_alpha_rel": (alpha - sel) / alpha if alpha else None,
        "none_minus_c_over_N_alpha_abs": none - alpha,
        "none_minus_c_over_N_alpha_rel": (none - alpha) / none if none else None,
        "note": ("margin_vs_none is the reported 2.7% figure; margin_vs_c_over_N_alpha is the smaller "
                 "0.2895-vs-0.2907 margin. They are different quantities and must not be interchanged"),
    }


# --------------------------------------------------------------------------------------
# simultaneous inference over the N grid
# --------------------------------------------------------------------------------------
def simultaneous_band(Y: np.ndarray, *, seed: int, replicates: int = 2000, alpha: float = 0.05,
                      min_n: int = 8) -> dict:
    """Max-|t| (studentized) bootstrap over seeds (rows) for the means of the
    columns (the N grid).  Returns the per-column means, standard errors, the
    simultaneous half-widths, and the bootstrap draws of the max-|t| statistic
    (so a per-claim p-value can be formed).  The Bonferroni-t band is the
    reported band when the studentized bootstrap is degenerate (n < min_n, or
    fewer than half of the replicates have a positive column SE)."""
    Y = np.asarray(Y, float)
    n, m = Y.shape
    mean = Y.mean(axis=0)
    if n < 2:
        return {"n_seeds": int(n), "mean": mean.tolist(), "se": [None] * m, "method": "none",
                "q_max_t": None, "half_width": [None] * m, "lower": [None] * m, "upper": [None] * m,
                "maxt_draws": [], "valid_replicates": 0, "replicates": int(replicates), "alpha": alpha,
                "bonferroni_t_crit": None, "t_df": 0, "fallback_reason": "n < 2"}
    se = Y.std(axis=0, ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(int(seed))
    maxt: list[float] = []
    for _ in range(int(replicates)):
        pick = rng.integers(0, n, size=n)
        Yb = Y[pick]
        mb = Yb.mean(axis=0)
        sb = Yb.std(axis=0, ddof=1) / np.sqrt(n)
        if np.any(sb <= 0):
            continue
        maxt.append(float(np.max(np.abs((mb - mean) / sb))))
    valid = len(maxt)
    t_bonf = float(stats.t.ppf(1 - alpha / (2 * m), df=n - 1))
    hw_bonf = (t_bonf * se)
    if n >= int(min_n) and valid >= max(1, int(replicates) // 2):
        q = float(np.percentile(maxt, 100 * (1 - alpha)))
        method = "max_abs_t_bootstrap"
        hw = q * se
    else:
        q = None
        method = "bonferroni_t_fallback"
        hw = hw_bonf
    return {"n_seeds": int(n), "mean": mean.tolist(), "se": se.tolist(), "method": method,
            "q_max_t": q, "half_width": hw.tolist(), "lower": (mean - hw).tolist(),
            "upper": (mean + hw).tolist(), "maxt_draws": maxt, "valid_replicates": int(valid),
            "replicates": int(replicates), "alpha": float(alpha), "bonferroni_t_crit": t_bonf,
            "t_df": int(n - 1), "fallback_reason": (None if method == "max_abs_t_bootstrap" else
                                                    (f"n = {n} < min_n = {min_n}" if n < int(min_n)
                                                     else f"only {valid}/{replicates} valid replicates"))}


def band_pvalue(band: dict, observed_maxt: float) -> float:
    """Bootstrap (or t) p-value for H0: the prediction equals the band mean.

    A SMALL p is evidence that the prediction is NOT adequate; the all-of
    support decision uses :func:`score_prediction`'s inside flags.
    """
    if band.get("method") == "max_abs_t_bootstrap" and band.get("maxt_draws"):
        d = np.asarray(band["maxt_draws"], float)
        return float((1.0 + float(np.sum(d >= observed_maxt))) / (1.0 + d.size))
    df = int(band.get("t_df") or max(1, int(band.get("n_seeds", 2)) - 1))
    return float(2.0 * (1.0 - stats.t.cdf(observed_maxt, df)))


def score_prediction(pred, band: dict) -> dict:
    """Score a frozen per-N prediction against a simultaneous band."""
    pred = np.asarray(pred, float)
    mean = np.asarray(band["mean"], float)
    se = np.asarray(band["se"], float) if band.get("se") and all(s is not None for s in band["se"]) else None
    if se is None:
        return {"scored": False, "reason": "no standard errors", "supported_at_grid": False,
                "inside_band": [None] * len(pred)}
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(pred - mean) / se
    finite = np.isfinite(t)
    obs = float(np.nanmax(t)) if np.any(finite) else None
    inside = [bool(l <= p <= u) for p, l, u in zip(pred, band["lower"], band["upper"])]
    crit = band["q_max_t"] if band["method"] == "max_abs_t_bootstrap" else band["bonferroni_t_crit"]
    return {"scored": True, "prediction": pred.tolist(), "inside_band": inside,
            "n_inside": int(sum(inside)), "n_grid": int(len(inside)),
            "studentized_abs_dev": [None if not np.isfinite(x) else float(x) for x in t],
            "max_studentized_abs_dev": obs, "critical_value": crit,
            "pvalue": (band_pvalue(band, obs) if obs is not None and np.isfinite(obs) else None),
            "supported_at_grid": bool(all(inside)),
            "rms_error": float(np.sqrt(np.mean((pred - mean) ** 2)))}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down; smaller p is stronger evidence of inadequacy."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1] if kv[1] is not None else 1.0))
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        pv = 1.0 if p is None else float(p)
        running = max(running, (m - i) * pv)
        adj[k] = float(min(1.0, running))
    return {"adjusted": adj, "reject_inadequacy_0.05": {k: adj[k] < 0.05 for k in adj},
            "order": [k for k, _ in items], "m": m,
            "note": "small p / reject_inadequacy means the frozen prediction is significantly far from the truth"}


# --------------------------------------------------------------------------------------
# the two endpoints
# --------------------------------------------------------------------------------------
def old_base_prediction(params: dict, N) -> np.ndarray:
    """Base prediction ``c/N + d/N^2`` of the OLD frozen model (the model of
    ``d - a_ref``).  It contains NO offset; the offset is added once, elsewhere."""
    N = np.asarray(N, float)
    return params["c"] / N + params["d"] / N ** 2


def old_endpoint_prediction(params: dict, N, a_ref_current: float) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint (a): the OLD frozen prediction on fresh seeds.

    Returns ``(base, prediction)`` with ``prediction = base + a_ref_current`` and
    the offset applied EXACTLY ONCE.  The stored t20s ``prediction_at_grid`` is
    NOT used as a base: it already contains the old a_ref.
    """
    base = old_base_prediction(params, N)
    return base, base + float(a_ref_current)


def check_offset_applied_once(*, params: dict, N, offset: float, prediction, tol: float = 1e-9,
                              where: str = "offset") -> None:
    """Reject a d-prediction that is not ``c/N + d/N^2 + a_ref`` (offset once).

    Raises :class:`ValueError` (the runner wraps it as ``ScienceError``, exit 4).
    The double-application trap is caught because for a nonzero offset
    ``base + 2 a_ref`` differs from ``base + a_ref`` by ``|a_ref|``.
    """
    base = (np.asarray(predict(params, N), float) if "model" in params
            else np.asarray(old_base_prediction(params, N), float))
    pred = np.asarray(prediction, float)
    if base.shape != pred.shape:
        raise ValueError(f"{where}: prediction shape {pred.shape} does not match base shape {base.shape}")
    dev = float(np.max(np.abs(pred - (base + float(offset)))))
    if dev > tol:
        raise ValueError(f"{where}: prediction is not base + a_ref applied exactly once "
                         f"(max deviation {dev:.3e} > {tol}); a stored offset-bearing prediction may have "
                         f"been reused as a base")
    if abs(float(offset)) > 1e-9:
        dbl = float(np.min(np.abs(pred - (base + 2.0 * float(offset)))))
        if dbl <= max(tol, 1e-6 * abs(float(offset))):
            raise ValueError(f"{where}: prediction equals base + 2 a_ref (offset applied twice)")


def archived_sign_profile(band: dict) -> dict:
    """The archived sign profile over the N grid, unchanged."""
    return OLD_AN.sign_profile(band)
