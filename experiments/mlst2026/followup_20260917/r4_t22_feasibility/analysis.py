"""R4 analysis: the unconditionally-bounded support probability and the two modules.

Module (a) evaluation-file power/coverage at a fixed development design.
Module (b) development-data adequacy at a fixed evaluation design.

The decisive screening fact is arithmetic and exact: because a withheld
replicate can never be 'supported', the unconditional support probability is at
most ``1 - P(withheld)``.  A design whose withholding cannot be pushed below 0.5
therefore **cannot** reach an unconditional medium-effect power of 0.5, no matter
how many evaluation files are added.  This bound is evaluated with the Wilson
uncertainty of the withholding estimate before any large-file-count extension is
priced.
"""
from __future__ import annotations

import numpy as np

Z = 1.959964


def wilson(k: int, n: int, z: float = Z) -> dict:
    if n <= 0:
        return {"rate": None, "lo": None, "hi": None, "half_width": None, "n": 0, "k": int(k)}
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": float(p), "lo": float(max(0.0, centre - half)), "hi": float(min(1.0, centre + half)),
            "half_width": float(half), "n": int(n), "k": int(k)}


def within(w: dict, band) -> bool | None:
    if w is None or w["n"] == 0:
        return None
    lo, hi = band
    return not (w["hi"] < lo or w["lo"] > hi)


def support_bound(records: list[dict]) -> dict:
    """The exact bound P(supported) <= 1 - P(withheld), plus its MC envelope.

    ``observed_max`` = 1 - observed withholding rate.
    ``plausible_max`` = 1 - withholding Wilson lower bound: the largest
    unconditional power a design with this withholding could have.
    """
    n = len(records)
    k_wh = sum(r["withheld"] for r in records)
    wh = wilson(k_wh, n)
    return {"n_mc": n, "withholding": wh,
            "observed_max_unconditional_support": None if wh["rate"] is None else 1.0 - wh["rate"],
            "plausible_max_unconditional_support": None if wh["lo"] is None else 1.0 - wh["lo"],
            "screen_power_0_5_possible": None if wh["lo"] is None else bool(1.0 - wh["lo"] >= 0.5),
            "note": "a withheld replicate can never be supported, so this bound is exact"}


def scenario_summary(sc, records: list[dict], tol: dict, *, claims: dict) -> dict:
    n = len(records)
    nw = [r for r in records if not r["withheld"]]
    mu = sc.true_mu()
    inference = (records[0].get("inference") if records else None)
    n_cal_dev = records[0].get("n_cal_dev") if records else None
    out = {"scenario": sc.name, "n_mc": n, "n_files": sc.n_files, "n_dev_files": sc.n_dev_files,
           "n_cal": (n_cal_dev if n_cal_dev is not None else sc.n_cal), "n_cal_eval": sc.n_cal,
           "is_drift": sc.drift_sd > 0, "true_mu": mu,
           "inference": inference,
           "withholding": wilson(n - len(nw), n), "support_bound": support_bound(records),
           "coverage_primary": {}, "claims": {}}
    for key in mu:
        cov = wilson(sum(r["contrasts"][key]["covers"] for r in records), n)
        out["coverage_primary"][key] = cov
        out["coverage_primary"][key]["within_tolerance"] = within(cov, tol["coverage"])
    null_claims = {name: all(mu.get(k, mu.get("zero|empca", 0.0)) == 0.0 for k in keys)
                   for name, keys in claims.items()}
    any_null = 0
    for name, keys in claims.items():
        k_sup = sum(r["verdicts"][name]["verdict"] == "supported" for r in records)
        k_nw = sum(r["verdicts"][name]["verdict"] == "supported" for r in nw)
        row = {"is_null": null_claims[name], "supported": wilson(k_sup, n),
               "supported_given_not_withheld": wilson(k_nw, len(nw))}
        if null_claims[name]:
            row["type1"] = row["supported"]
            row["type1_within_tolerance"] = within(row["supported"], tol["type1"])
        out["claims"][name] = row
    if any(null_claims.values()):
        any_null = sum(any(r["verdicts"][c]["verdict"] == "supported" for c in claims if null_claims[c])
                       for r in records)
        out["any_null_claim_supported"] = wilson(any_null, n)
        out["any_null_claim_supported"]["within_tolerance"] = within(out["any_null_claim_supported"], tol["type1"])
    return out


def feasibility_row(rows: list[dict], rules: dict, tol: dict, *,
                    expected_scenarios: int | None = None, complete: bool = True) -> dict:
    """The R4 feasibility decision for one evaluation-file count.

    ``complete`` is the reconciliation completeness of the prespecified scenario
    set at this file count; an incomplete reduction can never be feasible, only
    provisional.  The any-null-claim family error is included in the decision,
    not just the per-claim rates.
    """
    core = [r for r in rows if not r["is_drift"]]
    nulls = [r for r in rows if any(c.get("is_null") for c in r["claims"].values())]
    med = [r for r in core if abs(r["true_mu"].get("raw-bin|empca", 0.0) - 0.0001171875) < 1e-12]
    t1 = [c["type1_within_tolerance"] for r in nulls for c in r["claims"].values() if c.get("is_null")]
    fam = [r["any_null_claim_supported"]["within_tolerance"] for r in rows
           if r.get("any_null_claim_supported") is not None]
    cov = [r["coverage_primary"]["raw-bin|empca"]["within_tolerance"] for r in core]
    wh = [r["withholding"]["rate"] for r in core if r["withholding"]["rate"] is not None]
    pow_uncond = [r["claims"]["denoising_benefit"]["supported"]["rate"] for r in med]
    pow_cond = [r["claims"]["denoising_benefit"]["supported_given_not_withheld"]["rate"] for r in med]
    complete = bool(complete and (expected_scenarios is None or len(rows) >= expected_scenarios))
    row = {
        "n_scenarios": len(rows), "n_core": len(core), "n_null_scenarios": len(nulls),
        "expected_scenarios": expected_scenarios, "complete": complete,
        "type1_per_claim_within_tolerance": (all(t1) if t1 else None),
        "any_null_family_within_tolerance": (all(fam) if fam else None),
        "coverage_within_tolerance": (all(cov) if cov else None),
        "withholding_rate_max": max(wh) if wh else None,
        "power_medium_effect_min": min(pow_uncond) if pow_uncond else None,
        "power_medium_effect_min_given_not_withheld": min(pow_cond) if pow_cond else None,
        "max_unconditional_support_over_medium_scenarios": (
            min(r["support_bound"]["plausible_max_unconditional_support"] for r in med) if med else None),
        "screen_power_0_5_possible_min": (min(r["support_bound"]["screen_power_0_5_possible"] for r in med) if med else None),
        "rules": rules,
    }
    row["feasible_under_rules"] = bool(
        complete
        and row["type1_per_claim_within_tolerance"] and row["any_null_family_within_tolerance"]
        and row["coverage_within_tolerance"]
        and row["withholding_rate_max"] is not None and row["withholding_rate_max"] <= rules["max_withholding_rate"]
        and row["power_medium_effect_min"] is not None and row["power_medium_effect_min"] >= rules["min_power_medium_effect"])
    # the screening conclusion is independent of any file-count extension
    row["screened_infeasible_by_withholding_bound"] = bool(
        row["screen_power_0_5_possible_min"] is False
        and rules["min_power_medium_effect"] >= 0.5)
    return row
