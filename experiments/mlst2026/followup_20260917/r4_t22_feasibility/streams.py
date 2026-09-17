"""R4: explicit separate development/evaluation random streams.

Source change, recorded explicitly
----------------------------------
The archived simulator (``successor_20260913.t22s.simulator.simulate_scenario``)
draws development and evaluation files from **one** ``default_rng(seed)`` stream,
so changing ``n_files`` changes the evaluation realization only (development is
drawn first) but changing ``n_dev_files`` shifts the whole stream, and any
evaluation-file extension shares the stream's continued state.  R4 separates the
two streams under named sub-seeds so that:

* extending evaluation files to 32/64 does not alter the development realization
  (so module (a) is a clean evaluation-file comparison at a fixed development
  design);
* changing the development design does not alter the evaluation realization
  (so module (b) is a clean development-data comparison at fixed evaluation data).

The archived simulator is never edited; this wrapper is the recorded source
change and every R4 record stores the two sub-seeds.
"""
from __future__ import annotations

import hashlib

import numpy as np

from experiments.mlst2026.successor_20260913.t22s import simulator as sim
from experiments.mlst2026.followup_20260917.r1_t22_calibration import candidates as cand


def derive(seed: int, stream: str) -> int:
    """A stable sub-seed for a named stream (no global RNG state)."""
    h = hashlib.sha256(f"r4|{int(seed)}|{stream}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


def simulate_scenario_streams(seed: int, sc, *, n_cal_dev: int | None = None) -> tuple[list, list, dict]:
    """Same file model as the archived simulator, on two independent streams.

    ``n_cal_dev`` overrides the calibration length **for the development files
    only**.  The evaluation files always use ``sc.n_cal``.  Because development
    and evaluation are drawn from separate named streams and from separate
    scenario objects, changing the development design (``n_dev_files`` or
    ``n_cal_dev``) leaves every evaluation array and parameter unchanged, and
    changing the evaluation file count leaves the development arrays unchanged.
    """
    import dataclasses
    dev_seed, eval_seed = derive(seed, "development"), derive(seed, "evaluation")
    dev_rng = np.random.default_rng(dev_seed)
    eval_rng = np.random.default_rng(eval_seed)
    dev_sc = sc if (n_cal_dev is None or int(n_cal_dev) == int(sc.n_cal)) else \
        dataclasses.replace(sc, n_cal=int(n_cal_dev))
    lam_dev = sim.file_effects(dev_rng, dev_sc, dev_sc.n_dev_files)
    lam_eval = sim.file_effects(eval_rng, sc, sc.n_files)
    dev = [sim.simulate_file(dev_rng, dev_sc, f"dev{i:02d}", lam_dev[i]) for i in range(dev_sc.n_dev_files)]
    ev = [sim.simulate_file(eval_rng, sc, f"file{i:02d}", lam_eval[i]) for i in range(sc.n_files)]
    info = {"lam_emp_files": [float(x) for x in lam_eval], "N_files": [f.N for f in ev],
            "realised_equal_file_mu_raw-bin|empca": float(np.mean([f.lam_emp * f.N for f in ev]) * 4.0 / sc.w),
            "n_cal_dev": int(dev_sc.n_cal), "n_cal_eval": int(sc.n_cal),
            "stream_seeds": {"development": dev_seed, "evaluation": eval_seed},
            "stream_source_change": "separate development/evaluation streams and development/evaluation designs (R4)"}
    return dev, ev, info


def _r1_claim_verdicts(records_iv: dict, candidate: str, withheld: bool) -> dict:
    """The three-claim Holm family, scored on the SELECTED R1 candidate.

    Each claim's p-value is the selected candidate's coherent p for its
    constituent contrasts; the interval exclusion uses the same candidate's
    interval.  A withheld replicate yields ``inconclusive`` for every claim.
    """
    from experiments.mlst2026 import t22_tidmad_narrowband as t22
    from experiments.mlst2026.successor_20260913.t22s import pipeline as pl
    claims = pl.CLAIMS
    raw_p = {name: float(max(records_iv[k][candidate]["p_coherent"] for k in keys))
             for name, keys in claims.items()}
    adj = t22.holm(raw_p)
    out = {}
    for name, keys in claims.items():
        ivs = [records_iv[k][candidate] for k in keys]
        favours = all(v["point"] > 0 for v in ivs)
        excludes = all(v["lo"] > 0 or v["hi"] < 0 for v in ivs)
        if not records_iv:
            verdict = "not-evaluable"
        elif withheld:
            verdict = "inconclusive"
        elif favours and excludes and adj[name] < 0.05:
            verdict = "supported"
        elif favours:
            verdict = "ordering-only"
        else:
            verdict = "not-supported"
        out[name] = {"verdict": verdict, "p_raw": raw_p[name], "p_holm": float(adj[name]),
                     "candidate": candidate}
    return out


def run_replicate_streams(seed: int, sc, *, file_replicates: int, hier_replicates: int, curve_replicates: int,
                          dependence_override: dict | None = None,
                          inference: dict | None = None,
                          n_cal_dev: int | None = None) -> dict:
    """The archived pipeline stages on the separated streams.

    Mirrors ``successor_20260913.t22s.pipeline.run_replicate`` stage for stage
    (development curves -> dependence verdict -> evaluation scoring -> combined
    stage) and reuses every archived function; only the simulator call differs.

    When ``inference`` names a selected R1 candidate, that candidate's interval
    and coherent p-value replace the archived primary for coverage, exclusion and
    the Holm family, and every record identifies the inference version and the R1
    receipt used.  The archived baseline remains available under
    ``baseline_contrasts``/``baseline_verdicts`` for a separately labelled
    comparison, so a run that only tests the old interval is never described as
    testing the corrected one.
    """
    from experiments.mlst2026 import t22_tidmad_narrowband as t22
    from experiments.mlst2026.successor_20260913.t22s import pipeline as pl

    inference = inference or {"candidate": "two_stage", "version": "archived-t22s-baseline"}
    selected = inference.get("candidate", "two_stage")
    if selected not in cand.CANDIDATES:
        raise ValueError(f"R4 inference candidate {selected!r} is not in the frozen R1 candidate set")
    dev, ev, info = simulate_scenario_streams(seed, sc, n_cal_dev=n_cal_dev)
    dep = pl.dependence_stage(dev, sc, curve_replicates=curve_replicates, seed=seed + 500_000)
    if dependence_override:
        dep = {**dep, **dependence_override, "overridden": True}
    per_file = {}
    for i, f in enumerate(ev):
        per_file[f.name] = pl.score_file(f, sc, replicates=file_replicates, seed=seed + 10_000 * (i + 1),
                                         b_eval=dep["block_length_evaluation"], b_cal=dep["block_length_calibration"])
    P = pl.combine_protocol(hier_replicates=hier_replicates, seed=seed + 900_000)
    comb = t22.combine_stage(per_file, P, dep["verdict"])
    withheld = dep["verdict"] == "inconclusive"
    mu = sc.true_mu()
    contrasts, baseline_contrasts, cand_iv = {}, {}, {}
    for (a, b) in sim.CONTRASTS:
        key = f"{a}|{b}"
        c = comb["combined"][key]
        pr = c["primary_equal_file"]
        ti = c["secondary_equal_file_t_interval"]
        re_ = c["secondary_random_effects"]
        rec = {"point": float(pr["equal_file_mean_of_point_estimates"]),
               "bootstrap_mean": float(pr["equal_file_mean_of_bootstrap_means"]),
               "ci95_lo": float(pr["ci95_lo"]), "ci95_hi": float(pr["ci95_hi"]),
               "bootstrap_sd": float(pr["bootstrap_sd"]), "p_two_sided": float(pr["p_two_sided"]),
               "excludes_zero": bool(pr["excludes_zero"]),
               "t_lo": float(ti["ci95"][0]), "t_hi": float(ti["ci95"][1]),
               "re_estimate": float(re_["estimate"]), "re_lo": float(re_["ci95"][0]), "re_hi": float(re_["ci95"][1]),
               "per_file": {name: {"mean": float(r["contrasts"][key]["mean_delta_squared"]),
                                   "bootstrap_mean": float(r["contrasts"][key]["bootstrap_mean"]),
                                   "bootstrap_sd": float(r["contrasts"][key]["bootstrap_sd"])}
                            for name, r in per_file.items()}}
        # the frozen R1 candidate family, from the same per-file summaries
        pf = cand.PerFile(y=np.array([v["mean"] for v in rec["per_file"].values()]),
                          s=np.array([v["bootstrap_sd"] for v in rec["per_file"].values()]),
                          n_bootstrap=int(file_replicates))
        ivs = cand.candidate_intervals(pf, rec["point"], rec["bootstrap_sd"],
                                       two_stage_lo=rec["ci95_lo"], two_stage_hi=rec["ci95_hi"],
                                       two_stage_p=rec["p_two_sided"])
        for name in cand.CANDIDATES:
            ivs[name]["p_coherent"] = cand.coherent_pvalue(name, ivs[name])
        cand_iv[key] = ivs
        sel = ivs[selected]
        rec["selected_candidate"] = selected
        rec["selected_point"] = float(sel["point"])
        rec["selected_lo"] = float(sel["lo"])
        rec["selected_hi"] = float(sel["hi"])
        rec["selected_p_coherent"] = float(sel["p_coherent"])
        rec["selected_length"] = float(sel["hi"] - sel["lo"])
        rec["selected_excludes_zero"] = bool(sel["lo"] > 0 or sel["hi"] < 0)
        if key in mu:
            rec["true_mu"] = float(mu[key])
            rec["covers"] = bool(rec["selected_lo"] <= mu[key] <= rec["selected_hi"])
            rec["t_covers"] = bool(rec["t_lo"] <= mu[key] <= rec["t_hi"])
            rec["bias"] = rec["selected_point"] - mu[key]
        rec["length"] = rec["selected_length"]
        rec["t_length"] = rec["t_hi"] - rec["t_lo"]
        contrasts[key] = rec
        # the archived baseline, kept separately and never conflated with the selection
        baseline_contrasts[key] = {
            "point": rec["point"], "lo": rec["ci95_lo"], "hi": rec["ci95_hi"],
            "p_two_sided": rec["p_two_sided"], "length": rec["ci95_hi"] - rec["ci95_lo"],
            "covers": (bool(rec["ci95_lo"] <= mu[key] <= rec["ci95_hi"]) if key in mu else None),
        }
    verdicts = _r1_claim_verdicts(cand_iv, selected, withheld)
    baseline_verdicts = {name: {"verdict": v["verdict"], "p_raw": float(v["p_raw"]),
                                "p_holm": float(v["p_holm_adjusted"])}
                         for name, v in comb["verdicts"].items()}
    return {"seed": int(seed), "scenario": sc.name, "dependence": dep, "withheld": withheld,
            "contrasts": contrasts, "verdicts": verdicts,
            "baseline_contrasts": baseline_contrasts, "baseline_verdicts": baseline_verdicts,
            "inference": {"candidate": selected, "version": inference.get("version"),
                          "r1_receipt": inference.get("r1_receipt"),
                          "r1_receipt_sha256": inference.get("r1_receipt_sha256"),
                          "r1_protocol_sha256": inference.get("r1_protocol_sha256"),
                          "applies_r1_candidate": bool(inference.get("applies_r1_candidate",
                                                                     selected != "two_stage"))},
            "n_files_entering": len(comb["files_entering"]),
            "n_cal_dev": info["n_cal_dev"], "n_cal_eval": info["n_cal_eval"],
            "replicates": {"file": int(file_replicates), "hier": int(hier_replicates), "curve": int(curve_replicates)},
            "stream_seeds": info["stream_seeds"], "info": info}
