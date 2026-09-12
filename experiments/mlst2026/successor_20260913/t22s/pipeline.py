"""T22-S part A: the archived inference pipeline, run on simulated files.

Every inferential step below is the archived function, imported and called
with the archived semantics; nothing is re-implemented:

* ``te.fit_transfer``            transfer gain and the calibration-mean baseline
* ``te.squared_errors``          the endpoint, in physical-amplitude units
* ``te.block_variance_curve``    per-contrast block-variance curves
* ``te.aggregate_block_curves``  the declared *maximum* aggregation rule
* ``te.select_block_length``     stabilised / stabilised-at-endpoint / inconclusive
* ``te.joint_bootstrap``         joint calibration/evaluation circular-block bootstrap
* ``te.percentile_interval``     per-file 95 % percentile interval
* ``t22.combine_stage``          two-stage hierarchical file bootstrap of the
                                 equal-file mean, Holm across the three claims,
                                 the verdict rules, the secondary t-interval and
                                 random-effects summaries

Stage order, as archived: development files -> curves -> aggregate -> frozen
block lengths and dependence verdict; evaluation files scored at the frozen
block lengths; combined stage under the frozen dependence verdict.

:func:`check_record` holds the scientific invariants the reducer enforces on
every Monte-Carlo record (they are what a mutated pipeline fails).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.mlst2026 import t22_tidmad_narrowband as t22  # noqa: E402
from experiments.mlst2026 import tidmad_endpoint as te  # noqa: E402

from . import simulator as sim  # noqa: E402

BLOCKS_EVAL = (1, 2, 4, 8, 16, 32)     # archived T22 evaluation grid (amendment A2)
BLOCKS_CAL = (1, 2, 4, 8)              # archived calibration grid
CLAIMS = {"limited_recovery": ("zero|empca", "calibration-mean|empca"),
          "denoising_benefit": ("raw-bin|empca",),
          "metric_contribution": ("pca|empca",)}
ARCHIVED_REPLICATES = {"file": 2000, "hier": 10000, "curve": 400}
REDUCED_REPLICATES = {"file": 1000, "hier": 4000, "curve": 200}


def _key(a, b):
    return f"{a}|{b}"


def method_estimates(f: sim.SimFile, cal: te.TransferCalibration) -> dict:
    """The five method estimates the archived scorer would hold for this file."""
    n = f.eval_read.size
    return {"zero": np.zeros(n, dtype=np.complex128),
            "calibration-mean": np.full(n, cal.calibration_mean_coefficient, dtype=np.complex128),
            "raw-bin": f.eval_read.copy(),
            "pca": f.generated_estimates["pca"],
            "empca": f.generated_estimates["empca"]}


def file_curves(f: sim.SimFile, sc: sim.Scenario, *, curve_replicates: int, seed: int) -> tuple[list, dict]:
    """Per-contrast evaluation block-variance curves and the calibration curve."""
    cal = te.fit_transfer(f.cal_read, f.cal_ref, f.N)
    est = method_estimates(f, cal)
    t_eval = cal.g * f.eval_ref
    e2 = {m: te.squared_errors(q, t_eval, sc.w) for m, q in est.items()}
    curves = []
    for j, (a, b) in enumerate(sim.CONTRASTS):
        d = e2[a] - e2[b]
        curves.append(te.block_variance_curve(d, BLOCKS_EVAL, replicates=curve_replicates, seed=seed + 17 + 7 * j))
    cal_series = np.abs(f.cal_read - cal.g * f.cal_ref) ** 2
    cal_curve = te.block_variance_curve(cal_series, BLOCKS_CAL, replicates=curve_replicates, seed=seed + 29)
    return curves, cal_curve


def dependence_stage(dev_files, sc: sim.Scenario, *, curve_replicates: int, seed: int) -> dict:
    curves, cal_curves = [], []
    for i, f in enumerate(dev_files):
        cs, cc = file_curves(f, sc, curve_replicates=curve_replicates, seed=seed + 1000 * i)
        curves.extend(cs)
        cal_curves.append(cc)
    agg = te.aggregate_block_curves(curves, BLOCKS_EVAL)
    agg_cal = te.aggregate_block_curves(cal_curves, BLOCKS_CAL)
    sel = agg["selection"]
    return {"verdict": sel["verdict"], "block_length_evaluation": int(sel["block_length"]),
            "block_length_calibration": int(agg_cal["selection"]["block_length"]),
            "n_curves": int(agg.get("n_input_curves", 0)),
            "aggregate_curve": {str(b): (float(v["variance"]) if v.get("usable") else None) for b, v in agg["curve"].items()},
            "per_curve_verdict_counts": agg.get("per_curve_verdict_counts", {}),
            "endpoint_ratio": float(sel.get("endpoint_ratio", float("nan"))) if "endpoint_ratio" in sel else None}


def score_file(f: sim.SimFile, sc: sim.Scenario, *, replicates: int, seed: int, b_eval: int, b_cal: int) -> dict:
    """The archived per-file scoring: observed contrasts, joint bootstrap, intervals."""
    cal = te.fit_transfer(f.cal_read, f.cal_ref, f.N)
    est = method_estimates(f, cal)
    t_eval = cal.g * f.eval_ref
    draws = te.joint_bootstrap(est, f.eval_read, f.eval_ref, f.cal_read, f.cal_ref, f.N, sc.w, list(sim.CONTRASTS),
                               replicates=replicates, seed=seed, b_eval=b_eval, b_cal=b_cal)
    contrasts = {}
    for (a, b) in sim.CONTRASTS:
        obs = te.observed_contrast(est, f.eval_read, t_eval, sc.w, a, b)
        ci = te.percentile_interval(draws[(a, b)])
        contrasts[_key(a, b)] = {**obs, **ci, "ci_excludes_zero": ci["excludes_zero"]}
    return {"file": f.name, "in_combined_estimate": True, "contrasts": contrasts,
            "_draws": {_key(a, b): draws[(a, b)] for (a, b) in sim.CONTRASTS},
            "transfer_calibration": cal.as_dict()}


def combine_protocol(*, hier_replicates: int, seed: int) -> SimpleNamespace:
    """The Protocol-like object ``t22.combine_stage`` reads."""
    return SimpleNamespace(contrasts=list(sim.CONTRASTS), file_replicates=int(hier_replicates),
                           file_seed=int(seed), min_files_for_combined=4)


def run_replicate(seed: int, sc: sim.Scenario, *, file_replicates: int, hier_replicates: int,
                  curve_replicates: int, dependence_override: dict | None = None) -> dict:
    """One Monte-Carlo replicate of the whole archived pipeline; returns a record.

    ``dependence_override`` exists for tests only (it is recorded when used).
    """
    dev, ev, info = sim.simulate_scenario(seed, sc)
    dep = dependence_stage(dev, sc, curve_replicates=curve_replicates, seed=seed + 500_000)
    if dependence_override:
        dep = {**dep, **dependence_override, "overridden": True}
    per_file = {}
    for i, f in enumerate(ev):
        per_file[f.name] = score_file(f, sc, replicates=file_replicates, seed=seed + 10_000 * (i + 1),
                                      b_eval=dep["block_length_evaluation"], b_cal=dep["block_length_calibration"])
    P = combine_protocol(hier_replicates=hier_replicates, seed=seed + 900_000)
    comb = t22.combine_stage(per_file, P, dep["verdict"])
    mu = sc.true_mu()
    contrasts = {}
    for (a, b) in sim.CONTRASTS:
        key = _key(a, b)
        c = comb["combined"][key]
        pr = c["primary_equal_file"]
        ti = c["secondary_equal_file_t_interval"]
        re_ = c["secondary_random_effects"]
        rec = {"point": float(pr["equal_file_mean_of_point_estimates"]),
               "bootstrap_mean": float(pr["equal_file_mean_of_bootstrap_means"]),
               "ci95_lo": float(pr["ci95_lo"]), "ci95_hi": float(pr["ci95_hi"]),
               "bootstrap_sd": float(pr["bootstrap_sd"]), "p_two_sided": float(pr["p_two_sided"]),
               "excludes_zero": bool(pr["excludes_zero"]),
               "t_lo": float(ti["ci95"][0]), "t_hi": float(ti["ci95"][1]), "t_mean": float(ti["mean"]),
               "re_estimate": float(re_["estimate"]), "re_lo": float(re_["ci95"][0]), "re_hi": float(re_["ci95"][1]),
               "per_file": {name: {"mean": float(r["contrasts"][key]["mean_delta_squared"]),
                                   "bootstrap_mean": float(r["contrasts"][key]["bootstrap_mean"]),
                                   "bootstrap_sd": float(r["contrasts"][key]["bootstrap_sd"])}
                            for name, r in per_file.items()}}
        if key in mu:
            rec["true_mu"] = float(mu[key])
            rec["covers"] = bool(rec["ci95_lo"] <= mu[key] <= rec["ci95_hi"])
            rec["t_covers"] = bool(rec["t_lo"] <= mu[key] <= rec["t_hi"])
            rec["bias"] = rec["point"] - mu[key]
        rec["length"] = rec["ci95_hi"] - rec["ci95_lo"]
        rec["t_length"] = rec["t_hi"] - rec["t_lo"]
        contrasts[key] = rec
    verdicts = {name: {"verdict": v["verdict"], "p_raw": float(v["p_raw"]), "p_holm": float(v["p_holm_adjusted"])}
                for name, v in comb["verdicts"].items()}
    return {"seed": int(seed), "scenario": sc.name, "dependence": dep, "withheld": dep["verdict"] == "inconclusive",
            "contrasts": contrasts, "verdicts": verdicts, "n_files_entering": len(comb["files_entering"]),
            "replicates": {"file": int(file_replicates), "hier": int(hier_replicates), "curve": int(curve_replicates)},
            "info": info}


# --- scientific invariants -----------------------------------------------------------
def check_record(rec: dict, sc: sim.Scenario | None = None) -> list[str]:
    """Invariants of the declared procedure; a non-empty list is a rejection.

    1. dependence gate: an inconclusive dependence verdict withholds every claim;
    2. a 'supported' verdict needs a positive point estimate, an interval
       excluding zero and Holm-adjusted p < 0.05; Holm p >= raw p;
    3. coverage indicators agree with the stored interval and the true estimand
       (recomputed from the scenario when one is supplied);
    4. two-stage bootstrap variance equals the analytic two-stage decomposition
       (1/n) * mean_i[sigma_i^2 + (m_i - m_bar)^2] up to Monte-Carlo error, where
       m_i and sigma_i are the per-file bootstrap mean and sd -- a file-only
       resample (no within-file draw) violates it whenever within-file variance
       is a visible share.
    """
    problems = []
    dep = rec["dependence"]["verdict"]
    if rec.get("withheld") != (dep == "inconclusive"):
        problems.append("withheld flag disagrees with the dependence verdict")
    for name, v in rec["verdicts"].items():
        if dep == "inconclusive" and v["verdict"] != "inconclusive":
            problems.append(f"{name}: verdict {v['verdict']!r} under an inconclusive dependence assessment (gate dropped)")
        if v["p_holm"] + 1e-12 < v["p_raw"]:
            problems.append(f"{name}: Holm-adjusted p below the raw p")
        if v["verdict"] == "supported":
            for key in CLAIMS[name]:
                c = rec["contrasts"][key]
                if not (c["point"] > 0 and c["excludes_zero"] and v["p_holm"] < 0.05):
                    problems.append(f"{name}: 'supported' without point>0, interval excluding zero and Holm p<0.05 on {key}")
    mu = sc.true_mu() if sc is not None else {}
    for key, c in rec["contrasts"].items():
        if not (c["ci95_lo"] <= c["ci95_hi"]):
            problems.append(f"{key}: interval endpoints out of order")
        if c["excludes_zero"] != bool(c["ci95_lo"] > 0 or c["ci95_hi"] < 0):
            problems.append(f"{key}: excludes_zero disagrees with the interval")
        if "true_mu" in c:
            tm = mu.get(key, c["true_mu"])
            if abs(tm - c["true_mu"]) > 1e-12:
                problems.append(f"{key}: recorded true_mu differs from the scenario's estimand")
            if c["covers"] != bool(c["ci95_lo"] <= tm <= c["ci95_hi"]):
                problems.append(f"{key}: coverage indicator disagrees with the interval")
            if c["t_covers"] != bool(c["t_lo"] <= tm <= c["t_hi"]):
                problems.append(f"{key}: t-interval coverage indicator disagrees with the interval")
            if abs((c["point"] - tm) - c["bias"]) > 1e-9 * max(1.0, abs(tm)):
                problems.append(f"{key}: bias disagrees with point - true_mu")
        pf = c.get("per_file") or {}
        if pf:
            n = len(pf)
            m = np.array([v["bootstrap_mean"] for v in pf.values()])
            s2 = np.array([v["bootstrap_sd"] ** 2 for v in pf.values()])
            v_exact = (np.mean(s2) + np.mean((m - m.mean()) ** 2)) / n
            R = rec["replicates"]["hier"]
            tol = 6.0 * np.sqrt(2.0 / max(R, 2)) + 0.05
            if v_exact > 0 and abs(c["bootstrap_sd"] ** 2 / v_exact - 1.0) > tol:
                problems.append(f"{key}: two-stage bootstrap variance {c['bootstrap_sd']**2:.3g} is not the analytic "
                                f"two-stage value {v_exact:.3g} within {tol:.2f} relative (within-file draws missing?)")
            if abs(np.mean([v["mean"] for v in pf.values()]) - c["point"]) > 1e-9 * max(1.0, abs(c["point"])):
                problems.append(f"{key}: equal-file point is not the mean of the per-file points")
    return problems
