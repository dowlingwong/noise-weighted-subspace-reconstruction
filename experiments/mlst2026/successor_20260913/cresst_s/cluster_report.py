"""CRESST-S part B: cluster-aware *descriptive* reporting of the archive.

Every quantity in this module is a **descriptive sensitivity of the archive**.
None of it is population inference.  Seven run/channel pairs drawn from two runs
are seven correlated clusters; nothing computed here establishes coverage for a
new run or a new detector, and the wording of every output says so.

Contents
--------
``per_channel_effects``      the archive's per-channel paired event bootstrap (iid and
                             block-64), reproduced on the archived substream.
``leave_one_channel_out``    drop each channel, recompute theta and its hierarchical interval.
``leave_one_run_out``        drop each run.  With two runs this is *identical* to the
                             run-stratified estimate for the retained run, which is
                             checked explicitly.
``run_stratified``           run 33 alone (2 channels) and run 35 alone (5 channels).
``variance_decomposition``   within-channel (event) and between-channel components, and an
                             explicit statement that a between-*run* variance component is
                             not identifiable from two runs.

Reproduction note: ``per_channel_effects``, ``leave_one_channel_out`` and
``leave_one_run_out`` consume their archived substream *sequentially in the archived
order*, so each is one cell rather than one cell per channel.  Splitting them would
change the RNG consumption order and forfeit bit-identical reproduction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.mlst2026 import t13a_cresst_channel_bootstrap as t13a  # noqa: E402

from . import archive_recompute as AR  # noqa: E402

DESCRIPTIVE_LABEL = ("descriptive sensitivity of the archive - not population inference; "
                     "seven run/channel clusters from two runs cannot establish coverage for a "
                     "new run or a new detector")

BLOCK_MAIN = t13a.BLOCK_MAIN   # 64


def _pct(d: dict) -> dict:
    return {k: (100.0 * v if k in AR.ENDPOINT_KEYS else v) for k, v in d.items()}


# --- per-channel effects -------------------------------------------------------------------
def per_channel_effects(labels, runs, clusters, *, replicates: int, seed: int = AR.ARCHIVE_SEED) -> dict:
    """Paired event bootstrap of each channel's own fraction (archived stream order)."""
    rng = AR.stream("per_channel_effects", seed)
    fractions = t13a.channel_fractions(clusters)
    rows = []
    for i, lab in enumerate(labels):
        row = {"channel_label": lab, "run": runs[i], "n_evaluation_events": int(clusters[i][0].size),
               "fraction": float(fractions[i]), "fraction_pct": float(100 * fractions[i])}
        for b in (1, BLOCK_MAIN):
            mp, me = t13a._cluster_means(rng, clusters[i], int(replicates), b)
            d = t13a.percentile_ci(1.0 - me / mp, AR.ALPHA)
            key = "event_iid" if b == 1 else f"event_block{b}"
            row[key] = dict(d)
            row[key + "_pct"] = _pct(d)
        rows.append(row)
    return {"label": DESCRIPTIVE_LABEL, "replicates": int(replicates), "rows": rows,
            "resampling": "events within the channel only; the channel layer is not resampled"}


# --- leave-one-out --------------------------------------------------------------------------
def leave_one_channel_out(labels, runs, clusters, *, replicates: int, seed: int = AR.ARCHIVE_SEED) -> dict:
    rng = AR.stream("leave_one_channel_out", seed)
    theta = t13a.theta_point(clusters)
    rows = []
    for i, lab in enumerate(labels):
        keep = [clusters[j] for j in range(len(clusters)) if j != i]
        pt = t13a.theta_point(keep)
        draws = t13a.bootstrap(keep, rng, int(replicates), hierarchical=True,
                               estimand="mean_fraction", block=1)
        rows.append({"dropped_channel": lab, "dropped_run": runs[i],
                     "n_channels_retained": len(keep), "theta": float(pt), "theta_pct": float(100 * pt),
                     "shift_from_headline_pct": float(100 * (pt - theta)),
                     **_pct(t13a.percentile_ci(draws, AR.ALPHA))})
    return {"label": DESCRIPTIVE_LABEL, "replicates": int(replicates),
            "theta_all_channels": float(theta), "rows": rows}


def leave_one_run_out(labels, runs, clusters, *, replicates: int, seed: int = AR.ARCHIVE_SEED) -> dict:
    rng = AR.stream("leave_one_run_out", seed)
    theta = t13a.theta_point(clusters)
    rows = []
    for r in sorted(set(runs)):
        keep = [clusters[j] for j in range(len(clusters)) if runs[j] != r]
        pt = t13a.theta_point(keep)
        draws = t13a.bootstrap(keep, rng, int(replicates), hierarchical=True,
                               estimand="mean_fraction", block=1)
        rows.append({"dropped_run": r, "n_channels_retained": len(keep),
                     "retained_channels": [labels[j] for j in range(len(labels)) if runs[j] != r],
                     "retained_runs": sorted({runs[j] for j in range(len(runs)) if runs[j] != r}),
                     "theta": float(pt), "theta_pct": float(100 * pt),
                     "shift_from_headline_pct": float(100 * (pt - theta)),
                     **_pct(t13a.percentile_ci(draws, AR.ALPHA))})
    return {"label": DESCRIPTIVE_LABEL, "replicates": int(replicates),
            "theta_all_channels": float(theta), "n_runs": len(set(runs)), "rows": rows}


# --- run-stratified -------------------------------------------------------------------------
def run_stratified(labels, runs, clusters, *, replicates: int, seed: int = AR.ARCHIVE_SEED) -> dict:
    """Each run analysed alone: run 33 has 2 channels, run 35 has 5.

    Two intervals per stratum: the channel-fixed event bootstrap (events only) and the
    archive's hierarchical construction applied *within* the stratum (channels of that
    run resampled with replacement).  The second is reported with an explicit warning
    because a stratum of 2 clusters supports one degree of freedom at the cluster level.
    """
    rng = AR.stream("run_stratified", seed)
    theta_all = t13a.theta_point(clusters)
    strata = []
    for r in sorted(set(runs)):
        idx = [j for j in range(len(clusters)) if runs[j] == r]
        keep = [clusters[j] for j in idx]
        pt = t13a.theta_point(keep)
        row = {"run": r, "channels": [labels[j] for j in idx], "n_channels": len(idx),
               "n_evaluation_events": int(sum(int(clusters[j][0].size) for j in idx)),
               "theta": float(pt), "theta_pct": float(100 * pt),
               "shift_from_all_channels_pct": float(100 * (pt - theta_all)),
               "cluster_degrees_of_freedom": len(idx) - 1,
               "cluster_interval_is_coarse": bool(len(idx) <= 3)}
        for hier, key in ((False, "channel_fixed"), (True, "hierarchical_within_run")):
            draws = t13a.bootstrap(keep, rng, int(replicates), hierarchical=hier,
                                   estimand="mean_fraction", block=1)
            row[key] = _pct(t13a.percentile_ci(draws, AR.ALPHA))
        strata.append(row)
    out = {"label": DESCRIPTIVE_LABEL, "replicates": int(replicates),
           "theta_all_channels": float(theta_all), "theta_all_channels_pct": float(100 * theta_all),
           "strata": strata,
           "note": ("With exactly two runs the leave-one-run-out estimate for a dropped run is the "
                    "run-stratified estimate of the other run.  That identity is checked, and it is a "
                    "property of having two runs, not evidence about either run.")}
    if len(strata) == 2:
        a, b = strata[0], strata[1]
        out["between_run_contrast"] = {
            "runs": [a["run"], b["run"]],
            "theta_difference": float(a["theta"] - b["theta"]),
            "theta_difference_pct": float(100 * (a["theta"] - b["theta"])),
            "degrees_of_freedom": 1,
            "interpretation": ("a single one-degree-of-freedom contrast between two runs; it is a "
                               "description of the archive, not an estimate of run-to-run variability"),
        }
    return out


def check_loro_matches_stratified(loro: dict, strat: dict, *, tol: float = 1e-12) -> dict:
    """With two runs, dropping run g must give exactly the stratified estimate of the other."""
    by_run = {s["run"]: s for s in strat["strata"]}
    runs = sorted(by_run)
    rows, ok = {}, True
    for row in loro["rows"]:
        dropped = row["dropped_run"]
        others = [r for r in runs if r != dropped]
        if len(others) != 1:
            rows[dropped] = {"checked": False,
                             "reason": f"{len(others)} retained runs; the identity only holds for two runs"}
            continue
        dev = abs(float(row["theta"]) - float(by_run[others[0]]["theta"]))
        rows[dropped] = {"checked": True, "retained_run": others[0], "loro_theta": float(row["theta"]),
                         "stratified_theta": float(by_run[others[0]]["theta"]), "abs_deviation": dev,
                         "within_tolerance": bool(dev <= tol)}
        ok = ok and dev <= tol
    return {"tolerance": float(tol), "per_dropped_run": rows, "passed": bool(ok and rows)}


# --- variance decomposition -------------------------------------------------------------------
def _chi2_width_factor(df: int) -> float:
    """Ratio of the upper to the lower endpoint of the exact chi-square 95 % interval for a
    variance estimated with ``df`` degrees of freedom.  df = 1 gives ~5.1e3; df = 3 ~43."""
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(0.975, df) / chi2.ppf(0.025, df))
    except Exception:  # pragma: no cover - scipy is pinned in requirements.txt
        table = {1: 5115.9, 2: 176.4, 3: 43.3, 4: 21.7, 5: 14.0, 6: 10.6, 7: 8.7}
        return float(table.get(int(df), float("nan")))


def variance_decomposition(labels, runs, clusters, *, replicates: int, seed: int = AR.ARCHIVE_SEED,
                           min_runs_for_between_run: int = 4,
                           claim_between_run_identified: bool = False) -> dict:
    """What two runs can and cannot identify.

    ``within_channel``   mean over channels of the variance of the channel fraction under
                         the paired event bootstrap with the channel held fixed.  This is
                         the finite-event sampling component.
    ``between_channel``  ``max(0, s2_f - mean(within_channel))``: the dispersion of the seven
                         channel fractions net of that sampling component (a one-way
                         random-effects moment estimator).
    ``between_run``      **not identifiable here.**  With G = 2 runs the between-run mean
                         square carries one degree of freedom, the moment estimator is a
                         difference of two mean squares (and can be negative), and the exact
                         chi-square 95 % interval for a one-degree-of-freedom variance spans a
                         factor of about 5.1e3.  A between-run component is also completely
                         confounded with any channel-level heterogeneity that happens to align
                         with the run split.  The raw arithmetic is reported for inspection and
                         is explicitly flagged as not identified.

    ``claim_between_run_identified`` is a TEST-ONLY mutation switch: it makes the cell assert
    identification, which the package's own scientific check must reject.
    """
    rng = AR.stream("variance_decomposition", seed)
    f = np.asarray(t13a.channel_fractions(clusters), dtype=float)
    C = f.size
    run_of = list(runs)
    groups = sorted(set(run_of))
    G = len(groups)

    within = []
    for i in range(C):
        mp, me = t13a._cluster_means(rng, clusters[i], int(replicates), 1)
        d = 1.0 - me / mp
        within.append({"channel_label": labels[i], "run": run_of[i],
                       "n_evaluation_events": int(clusters[i][0].size),
                       "bootstrap_variance": float(np.var(d, ddof=1)),
                       "bootstrap_sd": float(np.std(d, ddof=1))})
    mean_within = float(np.mean([w["bootstrap_variance"] for w in within]))
    s2_total = float(np.var(f, ddof=1))
    between_channel = float(max(0.0, s2_total - mean_within))

    # one-way (runs) analysis of variance on the channel fractions
    n_g = np.array([sum(1 for r in run_of if r == g) for g in groups], dtype=float)
    m_g = np.array([float(f[[i for i in range(C) if run_of[i] == g]].mean()) for g in groups])
    N = float(n_g.sum())
    pooled = float((n_g * m_g).sum() / N)
    ss_b = float((n_g * (m_g - pooled) ** 2).sum())
    ss_w = float(sum(((f[[i for i in range(C) if run_of[i] == g]] - m_g[k]) ** 2).sum()
                     for k, g in enumerate(groups)))
    df_b, df_w = G - 1, int(N) - G
    ms_b = ss_b / df_b if df_b > 0 else float("nan")
    ms_w = ss_w / df_w if df_w > 0 else float("nan")
    c0 = float((N - (n_g ** 2).sum() / N) / df_b) if df_b > 0 else float("nan")
    naive = float((ms_b - ms_w) / c0) if df_b > 0 and c0 > 0 else float("nan")

    identifiable = bool(G >= int(min_runs_for_between_run))
    between_run = {
        "n_runs": G, "runs": groups, "channels_per_run": {g: int(n_g[k]) for k, g in enumerate(groups)},
        "degrees_of_freedom": df_b,
        "mean_square_between_runs": ms_b, "mean_square_within_runs": ms_w, "c0": c0,
        "naive_moment_estimate": naive,
        "naive_moment_estimate_can_be_negative": bool(naive < 0) if naive == naive else None,
        "chi2_95_width_factor_at_this_df": _chi2_width_factor(max(1, df_b)),
        "chi2_95_width_factor_at_min_required_df": _chi2_width_factor(max(1, int(min_runs_for_between_run) - 1)),
        "min_runs_required_for_a_reported_component": int(min_runs_for_between_run),
        "identified": bool(claim_between_run_identified) if not identifiable else True,
        "identifiable_from_this_design": identifiable,
        "statement": ("A between-run variance component is NOT identifiable from two runs: the "
                      "between-run mean square has one degree of freedom, the moment estimator is a "
                      "difference of mean squares that may be negative, its exact 95 % chi-square "
                      "interval spans a factor of about 5.1e3, and any channel-level heterogeneity "
                      "aligned with the run split is completely confounded with it.  The value above "
                      "is arithmetic, not an estimate, and this package refuses to report it as one."),
    }
    return {
        "label": DESCRIPTIVE_LABEL,
        "replicates": int(replicates),
        "n_channels": int(C), "n_runs": G,
        "channel_fractions": [float(x) for x in f],
        "total_observed_dispersion_of_channel_fractions": s2_total,
        "within_channel": {"per_channel": within, "mean_bootstrap_variance": mean_within,
                           "definition": "variance of 1 - mean(EMPCA)/mean(PCA) under the paired "
                                         "event bootstrap with the channel held fixed"},
        "between_channel": {"moment_estimate": between_channel,
                            "definition": "max(0, Var_c(fraction_c) - mean_c within_channel_variance_c)",
                            "share_of_total": float(between_channel / s2_total) if s2_total > 0 else None,
                            "degrees_of_freedom": int(C - 1)},
        "between_run": between_run,
        "nested_attempt": {
            "model": "fraction_{g,c} = theta + u_g + v_{g,c} + event noise",
            "separable_components": ["within_channel (event)", "between_channel (channels, pooled over runs)"],
            "non_separable_components": ["u_g (between run) and the part of v_{g,c} aligned with the run split"],
            "reason": "G = 2 gives one between-run degree of freedom",
        },
    }


def check_variance_decomposition(vd: dict, *, min_runs: int) -> list[str]:
    """Scientific gate.  A between-run variance component may only be reported as
    identified when the design has at least ``min_runs`` runs."""
    problems = []
    br = vd.get("between_run", {})
    n_runs = int(br.get("n_runs", vd.get("n_runs", 0)))
    if bool(br.get("identified")) and n_runs < int(min_runs):
        problems.append(f"variance decomposition claims an identified between-run component from "
                        f"{n_runs} run(s); at least {min_runs} are required "
                        f"(one between-run degree of freedom is not a variance estimate)")
    if bool(br.get("identifiable_from_this_design")) != (n_runs >= int(min_runs)):
        problems.append("between_run.identifiable_from_this_design disagrees with the number of runs")
    wc = vd.get("within_channel", {}).get("mean_bootstrap_variance")
    if wc is None or not np.isfinite(float(wc)) or float(wc) < 0:
        problems.append("within-channel variance component missing or not finite")
    bc = vd.get("between_channel", {}).get("moment_estimate")
    if bc is None or not np.isfinite(float(bc)) or float(bc) < 0:
        problems.append("between-channel variance component missing, negative or not finite")
    return problems


def compare_leave_one_out(recomputed_rows: list[dict], archived_rows: list[dict], key: str,
                          *, point_tol: float, endpoint_tol_fn) -> dict:
    """Deviation of the recomputed leave-one-out rows against the archived ones.

    ``theta`` and ``shift_from_headline_pct`` are exact arithmetic on the archived
    residuals and are compared at ``point_tol``; the interval endpoints are Monte-Carlo
    quantities compared at the tolerance produced by ``endpoint_tol_fn(archived_row)``.
    """
    arch = {str(r[key]): r for r in archived_rows}
    rows, ok, worst_pt, worst_ep = {}, True, 0.0, 0.0
    for r in recomputed_rows:
        a = arch.get(str(r[key]))
        if a is None:
            rows[str(r[key])] = {"present_in_archive": False}
            ok = False
            continue
        d_pt = abs(float(r["theta"]) - float(a["theta"]))
        d_sh = abs(float(r["shift_from_headline_pct"]) - float(a["shift_from_headline_pct"]))
        tol_ep = float(endpoint_tol_fn(a))
        eps = {}
        for k in ("ci95_lo", "ci95_hi", "bootstrap_mean", "bootstrap_sd"):
            dev = abs(float(r[k]) - float(a[k]))
            eps[k] = {"recomputed": float(r[k]), "archived": float(a[k]), "abs_deviation": dev,
                      "tolerance": tol_ep, "within_tolerance": bool(dev <= tol_ep)}
            worst_ep = max(worst_ep, dev)
        worst_pt = max(worst_pt, d_pt, d_sh)
        row_ok = bool(d_pt <= point_tol and d_sh <= point_tol * 100.0
                      and all(e["within_tolerance"] for e in eps.values())
                      and int(r["n_channels_retained"]) == int(a["n_channels_retained"]))
        rows[str(r[key])] = {"present_in_archive": True, "theta_abs_deviation": d_pt,
                             "shift_pct_abs_deviation": d_sh, "point_tolerance": float(point_tol),
                             "n_channels_retained_match": int(r["n_channels_retained"]) == int(a["n_channels_retained"]),
                             "endpoints": eps, "within_tolerance": row_ok}
        ok = ok and row_ok
    return {"key": key, "rows": rows, "max_point_abs_deviation": float(worst_pt),
            "max_endpoint_abs_deviation": float(worst_ep), "all_within_tolerance": bool(ok)}
