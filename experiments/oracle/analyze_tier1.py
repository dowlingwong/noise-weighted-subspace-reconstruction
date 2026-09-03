"""Endpoint analysis for the Tier-1 row tables (P3-E1/E3/E4/E5/E6).

Reads ``results/oracle/<ID>/<mode>/rows.csv`` and writes ``summary.json`` plus a short
``SUMMARY.md`` with, per claim:

* C1  detection power per (subject, stage, monitor, family) at the calibrated 1 % FAR threshold,
      and the realised held-out clean FAR;
* C2  N-vs-S attribution: macro-F1 of a logistic classifier on (a) the full layerwise signature
      (all monitors + tangent statistics), (b) the generic-only arm (input tests, MMD, C2ST,
      embedding moments), with leave-one-seed-out evaluation; conformal abstention on the
      undeclared U families (AUROC unknown-vs-known, risk–coverage AUC);
* C4  alarm–consequence: within each (family, severity) stratum, AUROC of the alarm for
      K >= kappa_m across windows and seeds, and the off-diagonal rates;
* mechanism signature: mean_disp_outside_T_S and resid_var_T_S for S vs N families, with a
      seed-level bootstrap interval on the difference;
* layerwise profile (mlp_ae): alignment drop per stage for N vs S.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .oracle_cov.abstain import auroc, conformal_threshold, risk_coverage
from .oracle_cov.protocol import RESULTS_ROOT, jsonable
from .oracle_cov.report import LogisticAttribution, macro_f1, nested_bootstrap

GENERIC = {"input_ks_bh", "mmd_rbf", "c2st_logistic_auc", "embedding_moments"}
TANGENT_KEYS = ["mean_disp_outside_T_S", "mean_disp_in_T_U", "resid_var_T_S", "event_disp_in_T_U_excess", "contrastive_in_dominant", "chi2_bar",
                "mean_disp_excess_z", "mean_disp_excess_z_T_S", "mean_disp_excess_z_T_U"]


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            if v in ("", "None"):
                r[k] = None
            elif v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def c1_power(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["subject"], r["stage"], r["monitor"], r["family"])].append(bool(r["alarm"]))
    return {"|".join(k): {"power": float(np.mean(v)), "n_windows": len(v)} for k, v in out.items()}


def c1_power_by_severity(rows):
    out = defaultdict(list)
    for r in rows:
        out[(r["subject"], r["stage"], r["monitor"], r["family"], r["severity"])].append(bool(r["alarm"]))
    res = defaultdict(dict)
    for (sub, st, mon, fam, sev), v in out.items():
        res[f"{sub}|{st}|{mon}|{fam}"][str(sev)] = float(np.mean(v))
    return dict(res)


def mean_K_by_severity(rows):
    seen = {}
    for r in rows:
        key = (r["subject"], r["family"], r["severity"], int(r["seed"]), int(r["window"]))
        seen[key] = float(r["K"])
    res = defaultdict(lambda: defaultdict(list))
    for (sub, fam, sev, _, _), K in seen.items():
        res[f"{sub}|{fam}"][str(sev)].append(K)
    return {k: {sev: float(np.mean(v)) for sev, v in d.items()} for k, d in res.items()}


def window_features(rows):
    """One feature vector per (seed, subject, stage, family, severity, window)."""
    feats = defaultdict(dict); label = {}; meta = {}
    for r in rows:
        key = (int(r["seed"]), r["subject"], r["stage"], r["family"], r["severity"], int(r["window"]))
        feats[key][r["monitor"]] = float(r["score"])
        for t in TANGENT_KEYS:
            if r.get(t) is not None:
                feats[key][t] = float(r[t])
        label[key] = r["contract"]; meta[key] = {"declared": bool(r["declared"]), "K": float(r["K"]), "high": bool(r["high_consequence"])}
    return feats, label, meta


def c2_attribution(rows):
    feats, label, meta = window_features(rows)
    keys = [k for k in feats if label[k] in ("N", "S")]
    ukeys = [k for k in feats if label[k] == "U"]
    res = {}
    for (subject, stage) in sorted({(k[1], k[2]) for k in keys}):
        ks = [k for k in keys if k[1] == subject and k[2] == stage]
        names_all = sorted({n for k in ks for n in feats[k]})
        names_gen = [n for n in names_all if n in GENERIC]
        # deployable on Tier 2: no forward-model tangent statistics, no covariance-requiring monitor
        names_dep = [n for n in names_all if n not in TANGENT_KEYS and n != "pullback_whitened"]
        seeds = sorted({k[0] for k in ks})
        arms = {}
        for arm, names in (("layerwise_full", names_all), ("layerwise_deployable", names_dep), ("generic_only", names_gen)):
            yt, yp, nonconf = [], [], []
            for s in seeds:                       # leave-one-seed-out
                tr = [k for k in ks if k[0] != s]; te = [k for k in ks if k[0] == s]
                if not tr or not te:
                    continue
                Xtr = np.array([[feats[k].get(n, 0.0) for n in names] for k in tr]); ytr = np.array([label[k] for k in tr])
                Xte = np.array([[feats[k].get(n, 0.0) for n in names] for k in te]); yte = np.array([label[k] for k in te])
                clf = LogisticAttribution().fit(Xtr, ytr)
                yp += list(clf.predict(Xte)); yt += list(yte)
                p = clf.proba_S(Xte); nonconf += list(1.0 - np.maximum(p, 1 - p))
            f1 = macro_f1(np.array(yt), np.array(yp)) if yt else float("nan")
            arms[arm] = {"macro_f1": f1, "n": len(yt), "features": names}
            if arm == "layerwise_full" and ukeys and nonconf:
                uk = [k for k in ukeys if k[1] == subject and k[2] == stage]
                Xall = np.array([[feats[k].get(n, 0.0) for n in names] for k in ks]); yall = np.array([label[k] for k in ks])
                clf = LogisticAttribution().fit(Xall, yall)
                Xu = np.array([[feats[k].get(n, 0.0) for n in names] for k in uk])
                pu = clf.proba_S(Xu); nu = 1.0 - np.maximum(pu, 1 - pu)
                rc = risk_coverage(np.array(nonconf), np.array(yt) == np.array(yp))
                arms[arm]["unknown_auroc"] = auroc(nu, np.array(nonconf))
                # second nonconformity: feature-space Mahalanobis distance to the declared (N+S) feature cloud
                mu, sd = Xall.mean(0), Xall.std(0) + 1e-9
                d_known = np.sqrt(np.mean(((Xall - mu) / sd) ** 2, axis=1)); d_unknown = np.sqrt(np.mean(((Xu - mu) / sd) ** 2, axis=1))
                arms[arm]["unknown_auroc_mahalanobis"] = auroc(d_unknown, d_known)
                arms[arm]["risk_coverage_auc"] = rc["risk_coverage_auc"]; arms[arm]["risk_at_80_coverage"] = rc["risk_at_80_coverage"]
                arms[arm]["conformal_threshold_alpha0.1"] = conformal_threshold(np.array(nonconf), 0.1)
        if "layerwise_full" in arms and "generic_only" in arms:
            arms["delta_f1_layerwise_minus_generic"] = arms["layerwise_full"]["macro_f1"] - arms["generic_only"]["macro_f1"]
        if "layerwise_deployable" in arms and "generic_only" in arms:
            arms["delta_f1_deployable_minus_generic"] = arms["layerwise_deployable"]["macro_f1"] - arms["generic_only"]["macro_f1"]
        res[f"{subject}|{stage}"] = arms
    return res


def c4_consequence(rows, kappa_m=0.10, K_key="K", high_key="high_consequence"):
    out = {}
    by = defaultdict(list)
    for r in rows:
        by[(r["subject"], r["stage"], r["monitor"])].append(r)
    for key, rs in by.items():
        strata = defaultdict(list)
        for r in rs:
            strata[(r["family"], r["severity"])].append(r)
        # within-stratum AUROC needs both classes; pool strata that have both
        aucs = []
        for st, srs in strata.items():
            hi = [float(r["score"]) for r in srs if r[high_key]]; lo = [float(r["score"]) for r in srs if not r[high_key]]
            if hi and lo:
                aucs.append(auroc(np.array(hi), np.array(lo)))
        A = np.array([float(r["score"]) for r in rs]); alarm = np.array([bool(r["alarm"]) for r in rs]); high = np.array([bool(r[high_key]) for r in rs])
        out["|".join(key)] = {
            "within_stratum_auroc_mean": float(np.mean(aucs)) if aucs else None, "n_strata_with_both": len(aucs),
            "pooled_auroc_high_consequence": auroc(A[high], A[~high]) if high.any() and (~high).any() else None,
            "missed_consequence_rate": float(np.mean(~alarm[high])) if high.any() else None,
            "harmless_alarm_rate": float(np.mean(alarm[~high])) if (~high).any() else None,
            "spearman_A_K": float(np.corrcoef(np.argsort(np.argsort(A)), np.argsort(np.argsort([float(r[K_key]) for r in rs])))[0, 1]) if len(rs) > 2 else None,
            "fraction_high_consequence": float(np.mean(high)),
        }
    return out


def mechanism_signature(rows):
    seen = set(); recs = []
    for r in rows:
        key = (int(r["seed"]), r["family"], r["severity"], int(r["window"]))
        if key in seen or r["contract"] not in ("N", "S"):
            continue
        seen.add(key); recs.append(r)
    out = {}
    for stat in ("mean_disp_excess_z", "mean_disp_excess_z_T_S", "mean_disp_excess_z_T_U", "mean_disp_excess_frac_in_T_S", "mean_disp_outside_T_S", "resid_var_T_S", "event_disp_in_T_U_excess"):
        S = [r for r in recs if r["contract"] == "S"]; N = [r for r in recs if r["contract"] == "N"]
        g = lambda rs: np.array([np.nan if r[stat] is None else float(r[stat]) for r in rs], dtype=float)
        out[stat] = {"S_mean": float(np.nanmean(g(S))), "N_mean": float(np.nanmean(g(N))),
                     "per_family": {f: float(np.nanmean(g([r for r in recs if r["family"] == f]))) for f in sorted({r["family"] for r in recs})}}
        vals = np.concatenate([g(S), g(N)])
        sd = np.array([int(r["seed"]) for r in S] + [int(r["seed"]) for r in N]); sign = np.array([1.0] * len(S) + [-1.0] * len(N))
        # paired-by-seed difference S - N via nested bootstrap on signed contributions
        diffs = []
        for s in sorted(set(sd)):
            m = sd == s
            diffs.append(np.nanmean(vals[m & (sign > 0)]) - np.nanmean(vals[m & (sign < 0)]))
        out[stat]["S_minus_N_per_seed"] = diffs
        out[stat]["S_minus_N_bootstrap"] = nested_bootstrap(np.array(diffs), np.arange(len(diffs)), np.arange(len(diffs)), resamples=1000)
    return out


def layerwise_profile(rows):
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["subject"] == "mlp_ae" and r["monitor"].startswith("layer_alignment") and r["contract"] in ("N", "S"):
            out[r["contract"]][r["stage"]].append(float(r["score"]))
    return {c: {st: {"mean_alignment_drop_score": float(np.mean(v)), "n": len(v)} for st, v in d.items()} for c, d in out.items()}


def far_summary(path: Path):
    if not path.exists():
        return None
    rows = load(path)
    by = defaultdict(list)
    for r in rows:
        by[(r["subject"], r["stage"], r["monitor"])].append(float(r["realised_far"]))
    return {"|".join(k): float(np.mean(v)) for k, v in by.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True); ap.add_argument("--mode", default="dev"); ap.add_argument("--kappa-m", type=float, default=0.10)
    a = ap.parse_args(argv)
    d = RESULTS_ROOT / a.experiment / a.mode
    rows = load(d / "rows.csv")
    summary = {"experiment": a.experiment, "mode": a.mode, "n_rows": len(rows), "seeds": sorted({int(r["seed"]) for r in rows}),
               "citable": a.mode == "confirmatory",
               "realised_clean_far": far_summary(d / "clean_far.csv"),
               "C1_power": c1_power(rows), "C1_power_by_severity": c1_power_by_severity(rows), "mean_K_by_severity": mean_K_by_severity(rows),
               "C2_attribution": c2_attribution(rows), "C4_consequence": c4_consequence(rows, a.kappa_m),
               "C4_consequence_recon": c4_consequence(rows, a.kappa_m, "K_recon", "high_consequence_recon") if "K_recon" in rows[0] else None,
               "mechanism_signature": mechanism_signature(rows), "layerwise_profile": layerwise_profile(rows)}
    (d / "summary.json").write_text(json.dumps(jsonable(summary), indent=2))
    write_md(d / "SUMMARY.md", summary)
    print(f"wrote {d / 'summary.json'}")


def write_md(path: Path, s: dict):
    L = [f"# {s['experiment']} ({s['mode']}) — {'citable' if s['citable'] else 'NOT citable (dev)'}", "",
         f"Rows: {s['n_rows']}, seeds: {s['seeds']}", ""]
    L += ["## Mechanism signature (S vs N)", "", "| statistic | S mean | N mean | S−N bootstrap 95 % |", "|---|---|---|---|"]
    for k, v in s["mechanism_signature"].items():
        b = v["S_minus_N_bootstrap"]; L.append(f"| {k} | {v['S_mean']:.3f} | {v['N_mean']:.3f} | {b['mean']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}] |")
    L += ["", "## C2 attribution (leave-one-seed-out)", "", "| subject/stage | layerwise full F1 | deployable F1 | generic F1 | ΔF1 full−generic | ΔF1 deployable−generic | unknown AUROC |", "|---|---|---|---|---|---|---|"]
    for k, v in s["C2_attribution"].items():
        lw, dp, ge = v.get("layerwise_full", {}), v.get("layerwise_deployable", {}), v.get("generic_only", {})
        ua = lw.get("unknown_auroc"); ua = float("nan") if ua is None else ua
        um = lw.get("unknown_auroc_mahalanobis"); um = float("nan") if um is None else um
        L.append(f"| {k} | {lw.get('macro_f1', float('nan')):.3f} | {dp.get('macro_f1', float('nan')):.3f} | {ge.get('macro_f1', float('nan')):.3f} | {v.get('delta_f1_layerwise_minus_generic', float('nan')):.3f} | {v.get('delta_f1_deployable_minus_generic', float('nan')):.3f} | {ua:.3f} (margin) / {um:.3f} (Mahalanobis) |")
    L += ["", "## C4 alarm–consequence (pooled over strata)", "", "| subject/stage/monitor | within-stratum AUROC | pooled AUROC | missed-consequence | harmless-alarm |", "|---|---|---|---|---|"]
    for k, v in s["C4_consequence"].items():
        f = lambda x: "—" if x is None else f"{x:.3f}"
        L.append(f"| {k} | {f(v['within_stratum_auroc_mean'])} | {f(v['pooled_auroc_high_consequence'])} | {f(v['missed_consequence_rate'])} | {f(v['harmless_alarm_rate'])} |")
    if s.get("C4_consequence_recon"):
        L += ["", "## C4 with the reconstruction consequence K_recon (whitened reconstruction error vs clean signal)", "", "| subject/stage/monitor | within-stratum AUROC | pooled AUROC | missed-consequence | harmless-alarm | frac. high |", "|---|---|---|---|---|---|"]
        for k, v in s["C4_consequence_recon"].items():
            f = lambda x: "—" if x is None else f"{x:.3f}"
            L.append(f"| {k} | {f(v['within_stratum_auroc_mean'])} | {f(v['pooled_auroc_high_consequence'])} | {f(v['missed_consequence_rate'])} | {f(v['harmless_alarm_rate'])} | {f(v['fraction_high_consequence'])} |")
    L += ["", "## C1 detection power (fraction of windows above the 1 % FAR threshold)", "", "| subject/stage/monitor/family | power |", "|---|---|"]
    for k, v in sorted(s["C1_power"].items()):
        L.append(f"| {k} | {v['power']:.2f} |")
    if s["layerwise_profile"]:
        L += ["", "## Layerwise alignment-drop score (mlp_ae)", "", "| contract | stage | mean score |", "|---|---|---|"]
        for c, d in s["layerwise_profile"].items():
            for st, v in d.items():
                L.append(f"| {c} | {st} | {v['mean_alignment_drop_score']:.2f} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
