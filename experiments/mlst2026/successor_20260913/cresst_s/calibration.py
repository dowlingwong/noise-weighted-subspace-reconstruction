"""CRESST-S part D: calibration of the *procedure*, not evidence about CRESST.

Before the archive's hierarchical interval could be used to generalize beyond the
observed clusters, its coverage has to be calibrated under a stated dependence
model.  This module simulates channel/run structured data with a **declared**
between-run variance and measures the coverage of the archive's own construction
(resample the C channel occurrences with replacement, then resample events within
each drawn occurrence, percentile interval) as a function of the number of runs
and the number of channels per run.

Nothing here uses CRESST data.  Nothing here is evidence about CRESST.  It is a
property of the estimator applied to a declared generator.

Generator (declared in the protocol)
------------------------------------
For run g = 1..G and channel c = 1..C within the run::

    f_{g,c} = theta + u_g + v_{g,c},   u_g ~ N(0, sd_run^2),  v_{g,c} ~ N(0, sd_channel^2)

Each channel carries ``n_events`` paired events::

    p_i = exp(N(0, sigma_p^2) - sigma_p^2 / 2)                       (PCA residual)
    e_i = p_i * (1 - f_{g,c}) * exp(N(0, sigma_e^2) - sigma_e^2 / 2)  (EMPCA residual)

so that the channel statistic ``1 - mean(e)/mean(p)`` concentrates on ``f_{g,c}``.
The estimand is ``theta``: the mean over the *population* of runs and channels,
which is what a generalizing claim would need.

Why coverage falls with few runs
--------------------------------
The archive's construction treats the G*C observed channels as exchangeable clusters.
The true variance of the equal-channel mean is ``sd_run^2/G + sd_channel^2/(G C)``,
while the construction's implied variance is ``s^2/(G C)`` with
``E[s^2] = sd_channel^2 + sd_run^2 C (G-1)/(G C - 1)``.  The ratio of true to implied
variance therefore *falls* as G grows, and the number of clusters available to the
percentile bootstrap grows with G as well.  Both effects push coverage up with G.
This is a statement about the procedure; the archive's own design is G = 2.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.mlst2026 import t13a_cresst_channel_bootstrap as t13a  # noqa: E402

CALIBRATION_LABEL = ("calibration of a procedure under a declared dependence model; "
                     "NOT evidence about CRESST and NOT a coverage claim for any real detector")

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval for a Monte-Carlo coverage rate."""
    k, n = int(k), int(n)
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "half_width": None, "n": 0, "k": k}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": float(p), "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "half_width": float(half), "n": n, "k": k}


@dataclass(frozen=True)
class CalibrationDesign:
    """One point of the calibration grid.  Every field is declared in the protocol."""
    n_runs: int
    channels_per_run: int
    n_events: int
    sd_run: float
    sd_channel: float
    theta: float
    sigma_p: float
    sigma_e: float
    mc_replicates: int
    bootstrap_replicates: int
    alpha: float = 0.05

    def as_dict(self) -> dict:
        return {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool) else float(v)
                    if isinstance(v, float) else v) for k, v in asdict(self).items()}

    @property
    def n_clusters(self) -> int:
        return int(self.n_runs) * int(self.channels_per_run)

    def work_proxy(self) -> float:
        """Event index draws per cell: the cost driver."""
        return float(self.mc_replicates) * float(self.bootstrap_replicates) * float(self.n_clusters) * float(self.n_events)


def implied_variance_ratio(d: CalibrationDesign) -> dict:
    """Analytic diagnostic: true variance of the equal-channel mean divided by the
    variance the archive's construction implies in expectation."""
    G, C = float(d.n_runs), float(d.channels_per_run)
    tr, tc = float(d.sd_run) ** 2, float(d.sd_channel) ** 2
    n = G * C
    true_var = tr / G + tc / n
    e_s2 = tc + (tr * C * (G - 1.0) / (n - 1.0) if n > 1 else 0.0)
    implied = e_s2 / n
    ratio = float(true_var / implied) if implied > 0 else float("nan")
    nominal_z = Z95 / math.sqrt(ratio) if ratio and ratio == ratio and ratio > 0 else float("nan")
    return {"true_variance_of_equal_channel_mean": float(true_var),
            "expected_bootstrap_implied_variance": float(implied),
            "true_over_implied": ratio,
            "large_sample_coverage_normal_reference": float(math.erf(nominal_z / math.sqrt(2.0)))
            if nominal_z == nominal_z else None}


# --- generator -------------------------------------------------------------------------------
def simulate_clusters(rng, d: CalibrationDesign):
    """Return ``(clusters, run_of_cluster, true_channel_fractions)`` in the archived
    ``list[(pca, empca)]`` cluster format, so the archived resampling helpers apply."""
    u = rng.normal(0.0, d.sd_run, size=d.n_runs)
    clusters, run_of, truth = [], [], []
    for g in range(int(d.n_runs)):
        for _c in range(int(d.channels_per_run)):
            f = float(d.theta + u[g] + rng.normal(0.0, d.sd_channel))
            f = float(np.clip(f, -2.0, 0.95))
            p = np.exp(rng.normal(0.0, d.sigma_p, size=int(d.n_events)) - 0.5 * d.sigma_p ** 2)
            e = p * (1.0 - f) * np.exp(rng.normal(0.0, d.sigma_e, size=int(d.n_events)) - 0.5 * d.sigma_e ** 2)
            clusters.append((np.ascontiguousarray(p), np.ascontiguousarray(e)))
            run_of.append(g)
            truth.append(f)
    return clusters, run_of, truth


# --- the archive's own construction, generalised to C clusters ---------------------------------
def hierarchical_interval(clusters, rng, *, replicates: int, alpha: float = 0.05, block: int = 1) -> dict:
    """The archive's construction: resample the C channel occurrences with replacement,
    then resample events within each drawn occurrence (the same drawn events for both
    methods, so the pairing is preserved), average the C occurrence fractions, take the
    percentile interval.

    This mirrors ``t13a.bootstrap(..., hierarchical=True, estimand='mean_fraction')``,
    which hard-codes the archive's seven occurrences; here the number of occurrences
    drawn equals the number of observed clusters, which is the same rule at C = 7.
    """
    C = len(clusters)
    R = int(replicates)
    out = np.empty(R)
    done = 0
    chunk = 1000
    while done < R:
        m = min(chunk, R - done)
        choice = rng.integers(0, C, size=(m, C))
        mp = np.empty(choice.shape)
        me = np.empty(choice.shape)
        for c in range(C):
            mask = choice == c
            k = int(mask.sum())
            if k == 0:
                continue
            a, b = t13a._cluster_means(rng, clusters[c], k, block)
            mp[mask] = a
            me[mask] = b
        out[done:done + m] = (1.0 - me / mp).mean(axis=1)
        done += m
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_mean": float(out.mean()),
            "bootstrap_sd": float(out.std(ddof=1)), "replicates": R}


def channel_fixed_interval(clusters, rng, *, replicates: int, alpha: float = 0.05, block: int = 1) -> dict:
    """The same statistic with the channel layer held fixed (events only)."""
    C = len(clusters)
    R = int(replicates)
    mp = np.empty((R, C))
    me = np.empty((R, C))
    for c in range(C):
        a, b = t13a._cluster_means(rng, clusters[c], R, block)
        mp[:, c] = a
        me[:, c] = b
    draws = (1.0 - me / mp).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"ci95_lo": float(lo), "ci95_hi": float(hi), "bootstrap_mean": float(draws.mean()),
            "bootstrap_sd": float(draws.std(ddof=1)), "replicates": R}


# --- coverage ---------------------------------------------------------------------------------
def coverage(d: CalibrationDesign, *, seed: int, also_channel_fixed: bool = False) -> dict:
    """Monte-Carlo coverage of the archive's hierarchical construction for ``theta``.

    Also reports coverage of the *realised* equal-channel mean of the simulated true
    fractions (the estimand a channel bootstrap can plausibly target), which separates
    "the interval misses the population mean" from "the interval is too short".
    """
    rng = np.random.default_rng(int(seed))
    k_pop = k_real = k_cf = 0
    lengths, realised = [], []
    for _ in range(int(d.mc_replicates)):
        clusters, _run_of, truth = simulate_clusters(rng, d)
        iv = hierarchical_interval(clusters, rng, replicates=d.bootstrap_replicates, alpha=d.alpha)
        lo, hi = iv["ci95_lo"], iv["ci95_hi"]
        f_bar = float(np.mean(truth))
        k_pop += int(lo <= d.theta <= hi)
        k_real += int(lo <= f_bar <= hi)
        lengths.append(hi - lo)
        realised.append(f_bar)
        if also_channel_fixed:
            cf = channel_fixed_interval(clusters, rng, replicates=d.bootstrap_replicates, alpha=d.alpha)
            k_cf += int(cf["ci95_lo"] <= d.theta <= cf["ci95_hi"])
    n = int(d.mc_replicates)
    out = {
        "label": CALIBRATION_LABEL,
        "design": d.as_dict(), "seed": int(seed), "n_clusters": d.n_clusters,
        "nominal_level": float(1.0 - d.alpha),
        "coverage_of_population_theta": wilson(k_pop, n),
        "coverage_of_realised_channel_mean": wilson(k_real, n),
        "mean_interval_length": float(np.mean(lengths)),
        "sd_of_realised_channel_mean": float(np.std(realised, ddof=1)) if n > 1 else None,
        "analytic": implied_variance_ratio(d),
        "replicates": {"monte_carlo": n,
                       "bootstrap_per_monte_carlo_replicate": int(d.bootstrap_replicates),
                       "bootstrap_total": n * int(d.bootstrap_replicates)},
        "work_proxy_event_draws": d.work_proxy(),
    }
    if also_channel_fixed:
        out["coverage_of_population_theta_channel_fixed"] = wilson(k_cf, n)
        out["replicates"]["bootstrap_total"] *= 2
    return out


def coverage_curve(rows: list[dict]) -> dict:
    """Coverage of the population estimand against the number of runs, for one
    (channels_per_run, sd_run) group."""
    curve = {}
    for r in sorted(rows, key=lambda x: int(x["design"]["n_runs"])):
        curve[str(int(r["design"]["n_runs"]))] = {
            "coverage": r["coverage_of_population_theta"],
            "n_clusters": int(r["n_clusters"]),
            "mean_interval_length": r["mean_interval_length"],
            "analytic_true_over_implied": r["analytic"]["true_over_implied"],
        }
    ns = sorted(int(k) for k in curve)
    out = {"by_n_runs": curve, "n_runs_evaluated": ns}
    if len(ns) >= 2:
        lo, hi = str(ns[0]), str(ns[-1])
        a = curve[lo]["coverage"]["rate"]
        b = curve[hi]["coverage"]["rate"]
        out["coverage_increases_with_runs"] = bool(a < b)
        out["coverage_gain_fewest_to_most_runs"] = float(b - a)
        out["fewest_runs"] = int(ns[0])
        out["most_runs"] = int(ns[-1])
        out["coverage_at_fewest_runs"] = float(a)
        out["coverage_at_most_runs"] = float(b)
    return out


def null_dependence_control(*, n_runs: int, channels_per_run: int, n_events: int, sd_channel: float,
                            sd_run_alt: float, theta: float, sigma_p: float, sigma_e: float,
                            mc_replicates: int, bootstrap_replicates: int, seed: int,
                            coverage_band: list) -> dict:
    """Scientific control on the calibration itself.

    Arm A: ``sd_run = 0``.  Runs carry no extra variance, the observed channels really are
    exchangeable, and the archive's construction must cover close to nominal (within the
    declared band, which allows the small-cluster-count shortfall of a percentile interval).

    Arm B: ``sd_run = sd_run_alt > 0``.  The same construction must cover strictly less.
    If it did not, the calibration could not detect the failure mode it exists to measure.
    """
    common = dict(n_runs=int(n_runs), channels_per_run=int(channels_per_run), n_events=int(n_events),
                  sd_channel=float(sd_channel), theta=float(theta), sigma_p=float(sigma_p),
                  sigma_e=float(sigma_e), mc_replicates=int(mc_replicates),
                  bootstrap_replicates=int(bootstrap_replicates))
    a = coverage(CalibrationDesign(sd_run=0.0, **common), seed=int(seed))
    b = coverage(CalibrationDesign(sd_run=float(sd_run_alt), **common), seed=int(seed) + 1)
    ca = a["coverage_of_population_theta"]
    cb = b["coverage_of_population_theta"]
    lo, hi = float(coverage_band[0]), float(coverage_band[1])
    in_band = bool(ca["hi"] >= lo and ca["lo"] <= hi)
    degrades = bool(cb["rate"] < ca["rate"])
    return {
        "label": CALIBRATION_LABEL,
        "null_arm": {"sd_run": 0.0, "coverage": ca, "analytic": a["analytic"], "design": a["design"]},
        "dependent_arm": {"sd_run": float(sd_run_alt), "coverage": cb, "analytic": b["analytic"],
                          "design": b["design"]},
        "coverage_band_for_null_arm": [lo, hi],
        "null_arm_in_band": in_band,
        "dependence_degrades_coverage": degrades,
        "coverage_drop": float(ca["rate"] - cb["rate"]),
        "passed": bool(in_band and degrades),
        "replicates": {"monte_carlo": 2 * int(mc_replicates),
                       "bootstrap_total": a["replicates"]["bootstrap_total"] + b["replicates"]["bootstrap_total"]},
    }
