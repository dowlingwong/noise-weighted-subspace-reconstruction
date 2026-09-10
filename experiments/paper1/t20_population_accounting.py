"""T20 — population risk accounting for the structural prior (Part II protocol).

Exact three-term accounting of the NFPA-vs-CW-PCA class contrast at matched rank,
against the generator's population covariance (no held-out Monte Carlo):

    R(P_F_hat) - R(P_U_hat) = [R(P_F*) - R(P_U*)]   (approximation cost a)
                            + [R(P_F_hat) - R(P_F*)] (restricted excess x_F)
                            - [R(P_U_hat) - R(P_U*)] (unrestricted excess x_U),

with R(P) = tr((I - P) C) the population whitened centred clean-signal risk and C the
population covariance of the whitened centred clean events, estimated once per alpha
from two independent 100 000-event halves (their difference is the resolution floor).

Configuration (protocol II.T20): C = 8, T = 32, (k_c, k_t) = (2, 3), rank 6, oracle
covariance, SNR 10, alpha in {0, 0.4, 0.8}.  Stage A computes the population
projectors (P_U* = top-6 eigenprojector; P_F* = NFPA ALS on the population covariance,
50 random restarts plus an HOSVD-initialised restart, uncertified if they disagree by
more than the restart spread) and applies Gate A.  Stage B fits both classes on
N in {50, ..., 1600} training events, seeds 0-9, 3 ALS restarts selected by training
loss, and reports a, x_F, x_U, d per cell with c_F, c_U from x ~ c/N and the implied
crossover N* = (c_U - c_F)/a.  Stage C (seeds 10-19) runs only if N* lies inside the
grid for some alpha.

Usage::

    python -m experiments.paper1.t20_population_accounting --stage A
    python -m experiments.paper1.t20_population_accounting --stage B
    python -m experiments.paper1.t20_population_accounting --stage C   # only if triggered
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import generator as gen
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.covariance import full_covariance, whiten_separable
from experiments.synthetic.nfpa_confirmatory.generator import kron_basis, make_dataset
from experiments.synthetic.nfpa_confirmatory.methods import _whiten_batch, als_fit, fit_cw_pca

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "mlst_2026-09-11" / "t20"

ALPHAS = [0.0, 0.4, 0.8]
N_GRID = [50, 100, 200, 400, 800, 1600]
SEEDS_B = list(range(10))
SEEDS_C = list(range(10, 20))
N_RESTARTS = 3
POP_HALF = 100_000
POP_SEED = 20260911
POP_RESTARTS = 50
FIT_CAP = 1200
GATE_FLOOR_MULT = 10.0


def cfg_for(alpha: float, N: int):
    return replace(PRIMARY, T=32, nonseparability_alpha=alpha, N_train=N, N_validation=20, N_test=100,
                   n_restarts=N_RESTARTS, target_snr=10.0)


# ----------------------------------------------------------------------------
# population covariance and population ALS
# ----------------------------------------------------------------------------

def population_cov(alpha: float, seed: int, n: int = POP_HALF) -> tuple[np.ndarray, dict]:
    cfg = cfg_for(alpha, n)
    rng = np.random.default_rng(seed)
    V = gen.make_channel_basis(cfg); Tb = gen.make_temporal_basis(cfg)
    cov = full_covariance(cfg, "colored")
    clean, _ = gen.generate_clean_events(rng, n, cfg, V, Tb)
    clean, snr = gen.snr_normalize(clean, cov, cfg)
    Z = whiten_separable(clean, cov["Sigma_c"], cov["Sigma_t"])
    Z = Z - Z.mean(axis=0, keepdims=True)
    flat = Z.reshape(n, -1, order="F")
    return flat.T @ flat / n, snr


def risk(Cpop: np.ndarray, B: np.ndarray) -> float:
    """R(P) = tr((I - B B^T) C) for an orthonormal basis B."""
    return float(np.trace(Cpop) - np.trace(B.T @ Cpop @ B))


def als_population(Cpop: np.ndarray, C: int, T: int, k_c: int, k_t: int, *, n_iter: int, n_restarts: int,
                   seed: int, tol: float) -> dict:
    """ALS for the separable class directly on a population covariance (Fortran vec)."""
    Kt = Cpop.reshape(C, T, C, T, order="F")            # K[c, t, c', t']
    # HOSVD initialisation from the marginals
    Mc0 = np.einsum("ctdt->cd", Kt); Mt0 = np.einsum("ctcs->ts", Kt)
    def top(M, k):
        w, v = np.linalg.eigh(0.5 * (M + M.T)); return v[:, np.argsort(w)[::-1][:k]]
    inits = [("hosvd", top(Mc0, k_c), top(Mt0, k_t))]
    for r in range(n_restarts):
        rng = np.random.default_rng(seed * 1000 + r)
        inits.append((f"random{r}", np.linalg.qr(rng.standard_normal((C, k_c)))[0], np.linalg.qr(rng.standard_normal((T, k_t)))[0]))
    results = []
    for name, U_c, U_t in inits:
        prev = None
        for _ in range(n_iter):
            M_t = np.einsum("ca,ctds,da->ts", U_c, Kt, U_c)
            U_t = top(M_t, k_t)
            M_c = np.einsum("tl,ctds,sl->cd", U_t, Kt, U_t)
            U_c = top(M_c, k_c)
            val = float(np.einsum("ca,tl,ctds,da,sl->", U_c, U_t, Kt, U_c, U_t))
            if prev is not None and abs(val - prev) < tol:
                break
            prev = val
        results.append({"init": name, "captured": val, "U_c": U_c, "U_t": U_t})
    best = max(results, key=lambda r: r["captured"])
    vals = np.array([r["captured"] for r in results])
    return {"best": best, "hosvd_captured": results[0]["captured"], "best_captured": float(vals.max()),
            "restart_spread": float(vals.max() - vals.min()), "n_inits": len(results),
            "best_init": best["init"]}


def stage_A() -> dict:
    out: dict = {"stage": "A", "alphas": {}}
    for alpha in ALPHAS:
        t0 = time.time()
        C1, snr1 = population_cov(alpha, POP_SEED)
        C2, snr2 = population_cov(alpha, POP_SEED + 1)
        Cpop = 0.5 * (C1 + C2)
        cfg = cfg_for(alpha, 100)
        w = np.sort(np.linalg.eigvalsh(Cpop))[::-1]
        k = cfg.k_c * cfg.k_t
        # unrestricted optimum
        vals, vecs = np.linalg.eigh(Cpop); order = np.argsort(vals)[::-1]
        B_U = vecs[:, order[:k]]
        R_U = risk(Cpop, B_U)
        # restricted optimum (population ALS) on the mean covariance and on each half (resolution floor)
        pop = als_population(Cpop, cfg.C, cfg.T, cfg.k_c, cfg.k_t, n_iter=400, n_restarts=POP_RESTARTS, seed=17, tol=1e-13)
        B_F = kron_basis(pop["best"]["U_c"], pop["best"]["U_t"])
        R_F = risk(Cpop, B_F)
        a = R_F - R_U
        # resolution floor: a computed on the two halves with the same projectors
        a_half = [risk(Ch, B_F) - risk(Ch, B_U) for Ch in (C1, C2)]
        floor = abs(a_half[0] - a_half[1])
        hosvd_gap = pop["best_captured"] - pop["hosvd_captured"]
        certified = hosvd_gap <= pop["restart_spread"] + 1e-12
        out["alphas"][str(alpha)] = {
            "trace_C": float(np.trace(Cpop)), "eigenvalues_top8": [float(x) for x in w[:8]],
            "gap_rank6": float(w[5] - w[6]), "R_U_star": R_U, "R_F_star": R_F, "a": a,
            "a_halves": a_half, "resolution_floor": floor,
            "population_als": {kk: v for kk, v in pop.items() if kk != "best"},
            "certified": bool(certified), "hosvd_vs_best_gap": hosvd_gap,
            "realized_snr": {"half1": snr1, "half2": snr2},
            "U_c_star": pop["best"]["U_c"].tolist(), "U_t_star": pop["best"]["U_t"].tolist(),
            "B_U_star": B_U.tolist(), "seconds": time.time() - t0,
        }
        print(f"alpha={alpha}: a={a:.3e} floor={floor:.2e} R_U*={R_U:.4f} R_F*={R_F:.4f} gap6={w[5]-w[6]:.4f} certified={certified} init={pop['best_init']} spread={pop['restart_spread']:.2e}", flush=True)
    # Gate A
    a0 = out["alphas"]["0.0"]; a8 = out["alphas"]["0.8"]
    out["gate_A"] = {"a_alpha0_is_zero": bool(abs(a0["a"]) <= GATE_FLOOR_MULT * max(a0["resolution_floor"], 1e-12)),
                     "a_alpha08_resolvable": bool(a8["a"] > GATE_FLOOR_MULT * max(a8["resolution_floor"], 1e-12)),
                     "floor_multiplier": GATE_FLOOR_MULT}
    out["gate_A"]["passed"] = out["gate_A"]["a_alpha0_is_zero"] and out["gate_A"]["a_alpha08_resolvable"]
    print("Gate A:", json.dumps(out["gate_A"]), flush=True)
    return out


# ----------------------------------------------------------------------------
# finite-sample fits against the population
# ----------------------------------------------------------------------------

def fit_cell(alpha: float, N: int, seed: int, popA: dict) -> dict:
    cfg = cfg_for(alpha, N)
    ds = make_dataset(seed, "colored", cfg)
    Cpop = np.array(popA["Cpop"])
    B_U = np.array(popA["B_U_star"]); B_F = kron_basis(np.array(popA["U_c_star"]), np.array(popA["U_t_star"]))
    R_U_star, R_F_star = risk(Cpop, B_U), risk(Cpop, B_F)
    t0 = time.time()
    cw = fit_cw_pca(ds)
    Z = _whiten_batch(ds, ds.X_train)
    out = als_fit(Z, cfg.k_c, cfg.k_t, n_iter=cfg.n_als_iter, n_restarts=cfg.n_restarts, seed=seed,
                  tol=cfg.als_convergence_tol, collect_restarts=True)
    B_Fhat = kron_basis(out["U_c"], out["U_t"])
    R_Fhat = risk(Cpop, B_Fhat); R_Uhat = risk(Cpop, cw.P)
    per_restart = [risk(Cpop, kron_basis(r["U_c"], r["U_t"])) for r in out["restarts"]]
    x_F = R_Fhat - R_F_star; x_U = R_Uhat - R_U_star; a = R_F_star - R_U_star
    d = R_Fhat - R_Uhat
    return {"alpha": alpha, "N": N, "seed": seed, "a": a, "x_F": x_F, "x_U": x_U, "d": d,
            "identity_residual": float(d - (a + x_F - x_U)),
            "R_F_hat": R_Fhat, "R_U_hat": R_Uhat,
            "restart_risks": per_restart, "restart_spread_risk": float(max(per_restart) - min(per_restart)),
            "selected_restart": out["restart"], "population_best_restart": int(np.argmin(per_restart)),
            "selection_matches_population_best": bool(out["restart"] == int(np.argmin(per_restart))),
            "als_converged": bool(out["converged"]), "train_loss": out["loss"],
            "walltime_s": time.time() - t0}


def fit_lines(rows: list[dict]) -> dict:
    """c_F, c_U from x ~ c / N (through the origin, seed means), with a bootstrap over seeds."""
    res: dict = {}
    for alpha in ALPHAS:
        ra = [r for r in rows if r["alpha"] == alpha]
        Ns = sorted({r["N"] for r in ra})
        def means(key, rs):
            return np.array([np.mean([r[key] for r in rs if r["N"] == N]) for N in Ns])
        invN = 1.0 / np.array(Ns, float)
        def fit_c(y):
            return float(np.sum(invN * y) / np.sum(invN * invN))
        cF = fit_c(means("x_F", ra)); cU = fit_c(means("x_U", ra))
        a = float(np.mean([r["a"] for r in ra]))
        Nstar = (cU - cF) / a if a > 0 else None
        # seed bootstrap for c_U - c_F and N*
        rng = np.random.default_rng(20260911); boots = []
        seeds = sorted({r["seed"] for r in ra})
        for _ in range(2000):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            rs = [r for s in pick for r in ra if r["seed"] == s]
            boots.append(fit_c(means("x_U", rs)) - fit_c(means("x_F", rs)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        per_N = {}
        for N in Ns:
            rn = [r for r in ra if r["N"] == N]
            ds = np.array([r["d"] for r in rn])
            per_N[str(N)] = {"d_mean": float(ds.mean()), "d_ci": [float(x) for x in np.percentile(
                [np.mean(rng.choice(ds, size=len(ds), replace=True)) for _ in range(2000)], [2.5, 97.5])],
                "x_F_mean": float(np.mean([r["x_F"] for r in rn])), "x_U_mean": float(np.mean([r["x_U"] for r in rn])),
                "restart_spread_median": float(np.median([r["restart_spread_risk"] for r in rn])),
                "selection_matches_frac": float(np.mean([r["selection_matches_population_best"] for r in rn]))}
        # observed sign change of d(N) on the grid (model-free), log-linear interpolation
        dm = [per_N[str(N)]["d_mean"] for N in Ns]
        N_obs = None
        for i in range(len(Ns) - 1):
            if dm[i] < 0 < dm[i + 1]:
                t = -dm[i] / (dm[i + 1] - dm[i])
                N_obs = float(np.exp(np.log(Ns[i]) + t * (np.log(Ns[i + 1]) - np.log(Ns[i]))))
                break
        # goodness of the c/N model: relative RMS misfit of x_U over the grid
        xU = means("x_U", ra); misfit = float(np.sqrt(np.mean((xU - cU * invN) ** 2)) / np.sqrt(np.mean(xU ** 2)))
        slope = float(np.polyfit(np.log(Ns), np.log(xU), 1)[0])
        res[str(alpha)] = {"a": a, "c_F": cF, "c_U": cU, "c_U_minus_c_F": cU - cF, "c_U_minus_c_F_ci": [float(lo), float(hi)],
                           "N_star": Nstar, "N_star_inside_grid": bool(Nstar is not None and min(Ns) <= Nstar <= max(Ns)),
                           "N_star_observed": N_obs, "N_star_observed_inside_grid": bool(N_obs is not None),
                           "c_over_N_model_relative_misfit_xU": misfit, "x_U_loglog_slope": slope,
                           "per_N": per_N}
    return res


def provenance(argv):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unavailable"
    return {"code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty": bool(git("status", "--porcelain")),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_config_hash": config_hash(PRIMARY), "cell_config_hash_example": config_hash(cfg_for(0.0, 50)),
            "python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.platform(),
            "hostname": socket.gethostname(), "command": argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "C"], required=True)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    prov = provenance(sys.argv)
    t0 = time.time()
    if args.stage == "A":
        res = stage_A()
        # store the population covariances for stage B
        for alpha in ALPHAS:
            C1, _ = population_cov(alpha, POP_SEED); C2, _ = population_cov(alpha, POP_SEED + 1)
            res["alphas"][str(alpha)]["Cpop"] = (0.5 * (C1 + C2)).tolist()
        res["provenance"] = prov
        (args.out / "stage_A.json").write_text(json.dumps(res))
        (args.out / "stage_A_summary.json").write_text(json.dumps({"gate_A": res["gate_A"], "alphas": {
            a: {k: v for k, v in d.items() if k not in ("Cpop", "U_c_star", "U_t_star", "B_U_star")} for a, d in res["alphas"].items()}}, indent=1))
        return 0 if res["gate_A"]["passed"] else 4
    popA = json.loads((args.out / "stage_A.json").read_text())
    assert popA["gate_A"]["passed"], "Gate A not passed; stage B/C refuse to run"
    if args.stage == "B":
        seeds, cells = SEEDS_B, [(a, N) for a in ALPHAS for N in N_GRID]
    else:
        summB = json.loads((args.out / "stage_B_summary.json").read_text())
        cells = []
        # Cell selection adapted after stage B: the registered c/N model for x_U was refuted on the
        # stage-B grid (relative misfit and log-log slope recorded in stage_B_summary), so the
        # location is taken from the OBSERVED sign change of d(N) rather than from (c_U-c_F)/a.
        # The confirmation criterion and the fresh seeds 10-19 are unchanged.
        for alpha, s in summB["lines"].items():
            if s.get("N_star_observed") is not None:
                Ns = N_GRID; i = int(np.argmin([abs(np.log(N) - np.log(s["N_star_observed"])) for N in Ns]))
                cells += [(float(alpha), Ns[j]) for j in range(max(0, i - 1), min(len(Ns), i + 2))]
        cells = cells[:6]
        if not cells:
            print("stage C not triggered: no N* inside the grid"); return 0
        seeds = SEEDS_C
    n_fits = len(cells) * len(seeds) * (N_RESTARTS + 1)
    prior = 0 if args.stage == "B" else json.loads((args.out / "stage_B_summary.json").read_text())["n_fits"]
    if n_fits + prior > FIT_CAP:
        print(f"refusing: {n_fits}+{prior} fits exceed cap {FIT_CAP}"); return 2
    rows = []
    with open(args.out / f"stage_{args.stage}_rows.jsonl", "w") as fh:
        for (alpha, N) in cells:
            for seed in seeds:
                r = fit_cell(alpha, N, seed, popA["alphas"][str(alpha)])
                fh.write(json.dumps(r) + "\n"); rows.append(r)
            print(f"alpha={alpha} N={N}: d={np.mean([r['d'] for r in rows if r['alpha']==alpha and r['N']==N]):+.3e}  ({time.time()-t0:.0f}s)", flush=True)
    summ = {"stage": args.stage, "n_fits": n_fits, "cells": cells, "seeds": seeds,
            "max_identity_residual": float(max(abs(r["identity_residual"]) for r in rows))}
    if args.stage == "B":
        summ["lines"] = fit_lines(rows)
        summ["stage_C_triggered"] = any(v["N_star_inside_grid"] for v in summ["lines"].values())
    else:
        conf = {}
        for alpha in sorted({c[0] for c in cells}):
            rr = [r for r in rows if r["alpha"] == alpha]
            Ns = sorted({r["N"] for r in rr}); rng = np.random.default_rng(1)
            ci = {}
            for N in Ns:
                ds = np.array([r["d"] for r in rr if r["N"] == N])
                ci[str(N)] = [float(x) for x in np.percentile([np.mean(rng.choice(ds, len(ds), True)) for _ in range(2000)], [2.5, 97.5])]
            lo, hi = ci[str(Ns[0])], ci[str(Ns[-1])]
            conf[str(alpha)] = {"d_ci_by_N": ci, "confirmed": bool((lo[1] < 0 and hi[0] > 0) or (lo[0] > 0 and hi[1] < 0))}
        summ["confirmation"] = conf
        summ["cell_selection"] = "observed sign change of d(N) in stage B (registered c/N-model N* refuted there); criterion and seeds as registered"
    prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); prov["wall_seconds"] = time.time() - t0
    summ["provenance"] = prov
    (args.out / f"stage_{args.stage}_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k in ("lines", "confirmation", "stage_C_triggered", "max_identity_residual")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
