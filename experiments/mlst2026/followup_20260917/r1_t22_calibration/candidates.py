"""R1 candidate intervals for the T22 small-file calibration.

The frozen T22-S primary interval (a two-stage hierarchical file bootstrap with a
percentile z-interval) under-covers at 4 and 8 files and over-covers at 16.  The
audit's diagnosis is standard-error miscalibration: the ratio of the bootstrap SD
to the empirical SD of the point estimate tracks coverage (r = 0.79).  This module
declares a *small, theoretically justified* candidate set, all computed from the
same per-file summaries the archived scorer already stores, so paired comparisons
are exact on the same simulated replicate.

Every candidate reports, together, a point, an interval and a p-value derived from
the **same** pivot.  A candidate that repairs coverage but is scored with the
archived interval's p-value would be incoherent, so :func:`coherent_pvalue` is what
the claim rule uses when a candidate is selected.

Nothing here is fitted and nothing here is selected by looking at old
confirmation records: the candidates are frozen in the protocol before any new
production cell is drawn; the pilot picks at most one adequate candidate by a
predeclared all-scenario rule.

Candidates
----------
``two_stage``        the archived primary, named baseline.  Its endpoints, point
                     and bootstrap p-value are carried through **exactly** from
                     the stored record; it is never rebuilt from point +/- z*SE.
``t_equal_file``     Student-t interval on the per-file means (the archived
                     secondary t interval); equal-file target.
``two_stage_t``      the two-stage bootstrap SE with a Student-t critical value
                     at ``n_files - 1`` degrees of freedom; equal-file target.
``normal_z``         an explicitly named *new* candidate: point +/- z*SE, the
                     normal approximation that the baseline is **not**.
``welch_fe_t``       secondary diagnostic only: fixed-effect inverse-variance
                     mean.  Its target (inverse-variance/fixed-effect mean) is
                     not the equal-file estimand, so it is excluded from primary
                     selection.  Its dof is the file count, tied to the sampling
                     design, not the bootstrap draw count.
``re_t``             secondary diagnostic only: DerSimonian-Laird random effects.
                     Its target (precision-weighted mean) is not the equal-file
                     estimand, so it is excluded from primary selection.

Primary eligibility
-------------------
Only candidates whose estimand is the archived **equal-file mean** may replace
the primary estimator.  ``PRIMARY_ELIGIBLE`` names them; ``SECONDARY_DIAGNOSTIC``
candidates are reported beside the primary for description and cannot be selected
(the archived RE interval is reported unchanged by R0).  Selecting a shorter
interval for a different target using equal-file coverage is exactly the error
this split prevents.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # scipy is pinned in the validated environments; the fallback is exact for small df
    from scipy import stats as _st
    def _t975(dof: float) -> float:
        return float(_st.t.ppf(0.975, dof))
except Exception:  # pragma: no cover - exercised only without scipy
    _T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}

    def _t975(dof: float) -> float:
        return float(_T975.get(int(round(dof)), 1.96))


Z975 = 1.959964

#: Frozen candidate set, in tie-break order.
CANDIDATES = ("two_stage", "normal_z", "t_equal_file", "two_stage_t", "welch_fe_t", "re_t")

#: Candidates whose target is the archived equal-file mean: primary-eligible.
PRIMARY_ELIGIBLE = ("two_stage", "normal_z", "t_equal_file", "two_stage_t")
#: Target-mismatched candidates: reported beside the primary, never selected.
SECONDARY_DIAGNOSTIC = ("welch_fe_t", "re_t")

#: Which candidates carry a coherent p-value and may therefore enter the claim rule.
CLAIM_ELIGIBLE = PRIMARY_ELIGIBLE


@dataclass(frozen=True)
class PerFile:
    y: np.ndarray          # observed per-file contrasts
    s: np.ndarray          # per-file bootstrap SDs
    n_bootstrap: int       # within-file bootstrap replicates (recorded; not used as dof)


def _random_effects(y: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """DerSimonian-Laird estimate, SE and tau (identical to the archived helper)."""
    if y.size == 0:
        return float("nan"), float("nan"), float("nan")
    w = 1.0 / s ** 2
    fe = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (y.size - 1)) / C) if C > 0 else 0.0
    wr = 1.0 / (s ** 2 + tau2)
    re = float(np.sum(wr * y) / np.sum(wr))
    se = float(np.sqrt(1.0 / np.sum(wr)))
    return re, se, float(np.sqrt(tau2))


def candidate_intervals(pf: PerFile, two_stage_point: float, two_stage_se: float, *,
                        two_stage_lo: float | None = None, two_stage_hi: float | None = None,
                        two_stage_p: float | None = None) -> dict:
    """All candidate point/intervals from the same per-file summaries.

    ``two_stage_point`` and ``two_stage_se`` are the archived primary summary.
    ``two_stage_lo``/``two_stage_hi``/``two_stage_p`` are the **stored archived**
    endpoints and bootstrap p-value.  When they are supplied (as they always are
    by the runner) the ``two_stage`` baseline carries them through exactly; only
    if they are absent does it fall back to a normal reconstruction, and that
    fallback is labelled rather than silently presented as the archived interval.
    """
    y, s = pf.y, pf.s
    n = int(y.size)
    out: dict[str, dict] = {}

    def _base(name, **kw):
        row = {"primary_eligible": name in PRIMARY_ELIGIBLE,
               "estimand": ("equal_file" if name in PRIMARY_ELIGIBLE else
                            ("inverse_variance_fixed_effect" if name == "welch_fe_t" else
                             "precision_weighted_random_effects")),
               "assumptions": ASSUMPTIONS[name]}
        row.update(kw)
        return row

    # two_stage: the archived primary, carried through exactly when its own
    # endpoints/p-value are supplied; the normal rebuild is a labelled fallback.
    if two_stage_lo is not None and two_stage_hi is not None:
        out["two_stage"] = _base("two_stage", point=float(two_stage_point), lo=float(two_stage_lo),
                                 hi=float(two_stage_hi), se=float(two_stage_se), dof=None,
                                 critical=None, p_coherent=(None if two_stage_p is None else float(two_stage_p)),
                                 form="archived two-stage bootstrap percentile (stored endpoints)")
    else:
        lo = two_stage_point - Z975 * two_stage_se
        hi = two_stage_point + Z975 * two_stage_se
        out["two_stage"] = _base("two_stage", point=float(two_stage_point), lo=float(lo), hi=float(hi),
                                 se=float(two_stage_se), dof=None, critical=Z975, p_coherent=None,
                                 form="two-stage bootstrap SE rebuilt as point +/- z*SE (archived endpoints absent)")

    # normal_z: the explicitly named new normal-approximation candidate.
    out["normal_z"] = _base("normal_z", point=float(two_stage_point),
                            lo=float(two_stage_point - Z975 * two_stage_se),
                            hi=float(two_stage_point + Z975 * two_stage_se),
                            se=float(two_stage_se), dof=None, critical=Z975, p_coherent=None,
                            form="point +/- z*SE (normal approximation; a NAMED NEW candidate)")

    if n >= 2:
        mfile = float(np.mean(y)); sfile = float(np.std(y, ddof=1))
        half = _t975(n - 1) * sfile / np.sqrt(n)
        out["t_equal_file"] = _base("t_equal_file", point=mfile, lo=mfile - half, hi=mfile + half,
                                    se=sfile / np.sqrt(n), dof=n - 1, critical=_t975(n - 1),
                                    form="student-t on file means")
        crit = _t975(n - 1)
        out["two_stage_t"] = _base("two_stage_t", point=float(two_stage_point),
                                   lo=float(two_stage_point - crit * two_stage_se),
                                   hi=float(two_stage_point + crit * two_stage_se),
                                   se=float(two_stage_se), dof=n - 1, critical=crit,
                                   form="two-stage bootstrap SE, t(n-1) critical value")
        # welch_fe_t: fixed-effect inverse-variance mean.  dof = n-1 is tied to the
        # sampling design (the number of independent file estimates), NOT the
        # bootstrap draw count; the within-file bootstrap SDs are treated as known.
        w = 1.0 / s ** 2
        fe = float(np.sum(w * y) / np.sum(w))
        se_fe = float(np.sqrt(1.0 / np.sum(w)))
        dof = float(max(n - 1, 1))
        crit = _t975(dof)
        out["welch_fe_t"] = _base("welch_fe_t", point=fe, lo=fe - crit * se_fe, hi=fe + crit * se_fe,
                                  se=se_fe, dof=dof, critical=crit,
                                  form="inverse-variance file mean, t(n-1); SECONDARY (target differs)")
        # re_t: DerSimonian-Laird with a Student-t critical value
        re, se_re, tau = _random_effects(y, s)
        crit = _t975(n - 1)
        out["re_t"] = _base("re_t", point=re, lo=re - crit * se_re, hi=re + crit * se_re,
                            se=se_re, dof=n - 1, critical=crit, tau=tau,
                            form="DerSimonian-Laird random effects, t(n-1); SECONDARY (target differs)")
    else:
        for name in CANDIDATES:
            out.setdefault(name, _base(name, point=float(two_stage_point), lo=float(out["two_stage"]["lo"]),
                                       hi=float(out["two_stage"]["hi"]), se=float(two_stage_se), dof=None,
                                       critical=Z975, form="single file: falls back to the baseline"))
    return out


def candidate_coverage(intervals: dict, true_mu: float) -> dict:
    return {name: bool(v["lo"] <= true_mu <= v["hi"]) for name, v in intervals.items()}


def coherent_pvalue(name: str, interval: dict) -> float:
    """A two-sided p-value from the same pivot as the interval.

    This is what keeps a selected replacement's decision rule coherent: a
    candidate may not be scored for coverage on one pivot and for significance on
    another.
    """
    if interval.get("p_coherent") is not None:
        # the archived bootstrap p-value for the archived baseline: carried through
        return float(interval["p_coherent"])
    point, se = float(interval["point"]), float(interval["se"])
    if not np.isfinite(point) or not np.isfinite(se) or se <= 0:
        return 1.0
    z = abs(point) / se
    dof = interval.get("dof")
    if dof is None:
        # a normal pivot, used only for the explicitly named normal candidate
        return float(min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))))
    try:
        return float(2.0 * (1.0 - _st.t.cdf(z, dof)))
    except Exception:  # pragma: no cover
        return float(min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))))


# --- target estimands and assumptions, declared explicitly -------------------------
TARGET = {
    "two_stage": "equal-file mean of the per-file population contrasts (the archived estimand)",
    "normal_z": "equal-file mean; normal pivot (explicitly a new candidate, not the baseline)",
    "t_equal_file": "equal-file mean; Student-t pivot on the file means",
    "two_stage_t": "equal-file mean; archived two-stage SE with a t(n-1) critical value",
    "welch_fe_t": "SECONDARY: inverse-variance (fixed-effect) mean; not the equal-file target",
    "re_t": "SECONDARY: precision-weighted (random-effects) mean; not the equal-file target",
}
ASSUMPTIONS = {
    "two_stage": "the archived hierarchical bootstrap; percentile pivot; stored endpoints used exactly",
    "normal_z": "file means approximately normal with the archived bootstrap SE treated as known",
    "t_equal_file": "file means approximately normal and exchangeable around a common mean",
    "two_stage_t": "the two-stage bootstrap SD is an unbiased SE estimate with n-1 effective dof",
    "welch_fe_t": "independent file estimates, within-file bootstrap SDs treated as known; dof = n_files - 1",
    "re_t": "file effects exchangeable with a common random-effects mean (DerSimonian-Laird)",
}


def type1_accepts_upper_bound(w: dict, upper: float = 0.07) -> bool | None:
    """The single executable type-I acceptance predicate.

    The protocol's rule is: the upper Wilson bound of the any-null-claim error
    rate is at most ``upper``.  This is the predicate used by the protocol
    documentation, pilot selection, confirmation, receipts and tests.  It is
    deliberately stricter than the archived overlap test (which accepted any
    interval overlapping the band) and the two are not interchangeable.
    """
    if w is None or w.get("n", 0) == 0 or w.get("hi") is None:
        return None
    return bool(w["hi"] <= upper)


def historical_type1_overlap(w: dict, hi: float = 0.07) -> bool | None:
    """The archived T22-S overlap predicate, kept only for the baseline comparison."""
    if w is None or w.get("n", 0) == 0 or w.get("lo") is None:
        return None
    return bool(w["lo"] <= hi)


def selection_rule_doc() -> dict:
    return {
        "rule": ("A PRIMARY-ELIGIBLE candidate is ADEQUATE on the pilot set if, on every non-drift pilot "
                 "scenario, its 95% coverage Wilson interval is not entirely outside [0.925, 0.975], and on "
                 "every null pilot scenario the Holm decision family using this candidate's coherent p-value "
                 "has an any-null-claim type-I rate whose UPPER Wilson bound is <= 0.07. Among adequate "
                 "candidates the one with the shortest median interval length is the primary replacement. If "
                 "none is adequate the freeze is 'none' and the calibration is reported as a failure."),
        "type1_predicate": "upper Wilson bound of the any-null-claim rate <= 0.07 (candidates.type1_accepts_upper_bound)",
        "historical_baseline_predicate": "archived T22-S overlap rule (candidates.historical_type1_overlap), reported separately",
        "no_retuning": "Confirmation cells are scored against the frozen candidate only; they never reselect.",
        "tie_break": "shortest median interval length; ties broken by the declared candidate order",
        "family": list(CANDIDATES),
        "primary_eligible": list(PRIMARY_ELIGIBLE),
        "secondary_diagnostic": list(SECONDARY_DIAGNOSTIC),
        "eligibility": ("only equal-file-target candidates may replace the primary; target-mismatched candidates "
                        "are reported beside it and cannot be selected"),
    }


def all_targets_documented() -> bool:
    return set(TARGET) == set(CANDIDATES) and set(ASSUMPTIONS) == set(CANDIDATES)
