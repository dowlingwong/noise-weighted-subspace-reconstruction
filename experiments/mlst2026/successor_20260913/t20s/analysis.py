"""T20-S analysis primitives: scaling-model candidates, the pilot-only selection
algorithm, simultaneous (max-|t|) bootstrap intervals over the N grid, sign
profiles and adjacent sign brackets.  Pure functions of arrays; deterministic
given the bootstrap seed.  Nothing here reads files.

Scaling candidates (the four declared in the 12 September design input)::

    none          no parametric model: per-N mean of the fitting seeds (bracket only)
    c_over_N      y(N) = c / N
    c_over_N_alpha y(N) = c / N**alpha,  alpha in [0, 4]
    c_over_N_d_over_N2  y(N) = c / N + d / N**2

For the contrast d(N) = a + x_F - x_U the candidates are fitted to d - a_ref,
where a_ref is the population best-found approximation cost of that theta (a
constant of the algebraic cell, never a fitted intercept); see SCHEMA.md.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

CANDIDATES = ("none", "c_over_N", "c_over_N_alpha", "c_over_N_d_over_N2")


# --------------------------------------------------------------------------------------
# scaling models
# --------------------------------------------------------------------------------------
def fit_model(model: str, N: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares fit of ``model`` to per-N values ``y`` (already averaged over
    the fitting seeds).  Returns parameters; see :func:`predict`."""
    N = np.asarray(N, float)
    y = np.asarray(y, float)
    if model == "none":
        return {"model": model, "N": N.tolist(), "y": y.tolist()}
    if model == "c_over_N":
        x = 1.0 / N
        return {"model": model, "c": float(np.dot(x, y) / np.dot(x, x))}
    if model == "c_over_N_alpha":
        def c_of(alpha):
            x = N ** (-alpha)
            return float(np.dot(x, y) / np.dot(x, x))

        def sse(alpha):
            x = N ** (-alpha)
            return float(np.sum((y - c_of(alpha) * x) ** 2))
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


def split_halves(seeds: list[int]) -> tuple[list[int], list[int]]:
    """Deterministic split of the sorted pilot seeds into a first half and a
    second half (the first half gets the extra seed when the count is odd)."""
    s = sorted(int(x) for x in seeds)
    h = (len(s) + 1) // 2
    return s[:h], s[h:]


def cv_select(Y: dict[int, np.ndarray], N_grid: list[int], seeds: list[int]) -> dict:
    """Two-fold out-of-sample selection.  ``Y[theta_index]`` is an array
    (n_seeds, n_N) of per-seed targets ordered like ``seeds`` and ``N_grid``.
    Every candidate is fitted on one half of the seeds (per-N means) and scored
    on the per-seed values of the other half; the two folds are summed over
    theta.  The parametric candidate with the smallest error is selected only
    if it beats ``none`` strictly; otherwise the selection is ``none``."""
    A, B = split_halves(seeds)
    seeds_sorted = sorted(int(s) for s in seeds)
    idx = {s: i for i, s in enumerate(seeds_sorted)}
    N = np.asarray(N_grid, float)
    err = {m: 0.0 for m in CANDIDATES}
    n_terms = 0
    detail = {}
    for th_i, Yt in sorted(Y.items()):
        Yt = np.asarray(Yt, float)
        for fit_seeds, hold_seeds in ((A, B), (B, A)):
            if not fit_seeds or not hold_seeds:
                continue
            yfit = Yt[[idx[s] for s in fit_seeds]].mean(axis=0)
            Yhold = Yt[[idx[s] for s in hold_seeds]]
            for m in CANDIDATES:
                p = fit_model(m, N, yfit)
                pred = predict(p, N)
                e = float(np.mean((Yhold - pred[None, :]) ** 2))
                err[m] += e
                detail.setdefault(m, []).append({"theta_index": int(th_i), "fit_seeds": fit_seeds, "holdout_seeds": hold_seeds, "mse": e})
            n_terms += 1
    if n_terms == 0:
        return {"selected": "none", "cv_error": err, "reason": "no holdout fold could be formed", "folds": 0}
    param = [m for m in CANDIDATES if m != "none"]
    best = min(param, key=lambda m: (err[m], CANDIDATES.index(m)))
    selected = best if err[best] < err["none"] else "none"
    return {"selected": selected, "cv_error": {m: float(v) for m, v in err.items()}, "folds": n_terms,
            "fit_half": A, "holdout_half": B, "detail": detail,
            "reason": ("beats the non-parametric bracket out of sample" if selected != "none"
                       else "no parametric candidate beats the non-parametric bracket out of sample")}


# --------------------------------------------------------------------------------------
# simultaneous inference over the N grid
# --------------------------------------------------------------------------------------
def simultaneous_ci(Y: np.ndarray, *, seed: int, replicates: int = 2000, alpha: float = 0.05, min_n: int = 8) -> dict:
    """Max-|t| (studentized) bootstrap over seeds (rows) for the means of the
    columns (the N grid).  Returns per-column means, standard errors, the
    simultaneous half-widths ``q * SE``, and the Bonferroni-t fallback.

    The bootstrap is used only with at least ``min_n`` seeds (with very few
    rows the resampling distribution of max |t| is degenerate: with n = 2 the
    only non-degenerate resample is the original sample and q = 0).  A
    replicate whose resampled column standard error is zero is dropped; if
    fewer than half the replicates survive, or n < min_n, the Bonferroni-t
    band (alpha / (2 m), n - 1 df) is the reported simultaneous band and the
    method field says so."""
    Y = np.asarray(Y, float)
    n, m = Y.shape
    mean = Y.mean(axis=0)
    if n < 2:
        return {"n_seeds": int(n), "mean": mean.tolist(), "se": [None] * m, "method": "none",
                "q_max_t": None, "half_width": [None] * m, "lower": [None] * m, "upper": [None] * m,
                "bonferroni": {"t_crit": None, "half_width": [None] * m}, "replicates": 0, "valid_replicates": 0}
    se = Y.std(axis=0, ddof=1) / np.sqrt(n)
    se_safe = np.where(se > 0, se, np.nan)
    rng = np.random.default_rng(seed)
    maxt = []
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
    hw_bonf = (t_bonf * se).tolist()
    if n >= int(min_n) and valid >= max(1, int(replicates) // 2):
        q = float(np.percentile(maxt, 100 * (1 - alpha)))
        method = "max_abs_t_bootstrap"
        hw = (q * se).tolist()
    else:
        q = None
        method = "bonferroni_t_fallback"
        hw = hw_bonf
    lower = [float(mu - h) for mu, h in zip(mean, hw)]
    upper = [float(mu + h) for mu, h in zip(mean, hw)]
    return {"n_seeds": int(n), "mean": mean.tolist(), "se": se.tolist(), "method": method, "q_max_t": q,
            "half_width": hw, "lower": lower, "upper": upper, "alpha": alpha,
            "bonferroni": {"t_crit": t_bonf, "half_width": hw_bonf,
                           "lower": (mean - np.asarray(hw_bonf)).tolist(), "upper": (mean + np.asarray(hw_bonf)).tolist()},
            "replicates": int(replicates), "valid_replicates": int(valid), "se_zero_columns": int(np.sum(np.isnan(se_safe))),
            "min_seeds_for_bootstrap": int(min_n),
            "fallback_reason": (None if method == "max_abs_t_bootstrap" else
                                (f"n = {n} < min_n = {min_n}" if n < int(min_n) else f"only {valid}/{replicates} valid replicates"))}


def sign_profile(ci: dict) -> dict:
    """Per-N signs decided by the simultaneous band ('+', '-', '0' undetermined),
    the profile string, and the adjacent brackets with opposite decided signs.
    Nonmonotone profiles are reported as they are; nothing is interpolated."""
    lo, hi = ci.get("lower"), ci.get("upper")
    if lo is None or any(x is None for x in lo):
        return {"signs": None, "profile": None, "adjacent_brackets": [], "n_decided": 0}
    signs = []
    for l, u in zip(lo, hi):
        signs.append("+" if l > 0 else ("-" if u < 0 else "0"))
    brackets = [i for i in range(len(signs) - 1) if {signs[i], signs[i + 1]} == {"+", "-"}]
    return {"signs": signs, "profile": " ".join(signs), "adjacent_brackets": brackets,
            "n_decided": int(sum(s != "0" for s in signs)), "monotone_sign_change": len(brackets) <= 1}


def score_frozen_prediction(pred: np.ndarray, ci: dict) -> dict:
    """Score a frozen model's per-N predictions against the confirmation band:
    inside/outside per N, max studentized deviation versus the simultaneous
    quantile (or the Bonferroni critical value)."""
    pred = np.asarray(pred, float)
    mean = np.asarray(ci["mean"], float)
    se = ci.get("se")
    if se is None or any(s is None for s in se) or ci["method"] == "none":
        return {"scored": False, "reason": "no standard errors"}
    se = np.asarray(se, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(pred - mean) / se
    crit = ci["q_max_t"] if ci["method"] == "max_abs_t_bootstrap" else ci["bonferroni"]["t_crit"]
    inside = [bool(l <= p <= u) for p, l, u in zip(pred, ci["lower"], ci["upper"])]
    return {"scored": True, "prediction": pred.tolist(), "inside_band": inside, "n_inside": int(sum(inside)),
            "n_grid": int(len(inside)), "studentized_abs_dev": [None if not np.isfinite(x) else float(x) for x in t],
            "max_studentized_abs_dev": float(np.nanmax(t)) if np.any(np.isfinite(t)) else None,
            "critical_value": crit, "supported_at_grid": bool(all(inside)),
            "rms_error": float(np.sqrt(np.mean((pred - mean) ** 2)))}
