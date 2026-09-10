"""T21 — continuous excitation and the recovery-variance curve (Part II protocol).

How does recovery of a planted physical direction depend on how strongly the
training data excite it, and where do "weakly excited" and "unexcited" separate?

Generator: the frozen planted two-axis family (PRIMARY: C = 8, T = 256, Brownian 1/f^2
temporal noise, AR(1) channel correlation, SNR 10) at alpha = 0 with its four physical
latents — amplitude (mean 1, sd 0.2), rise time (sd 0.15), decay time (sd 0.15) and
the secondary channel-mixing mode (sd 0.5).  Excitation strength s_j scales the
standard deviation of latent j in the TRAINING draw only (amplitude: A -> 1 + s(A-1),
which leaves the RNG stream untouched); the evaluation draw keeps every latent at s = 1.
Learner: rank-4 covariance-weighted PCA (closed form) on the whitened centred
training events.  Direction j is the whitened Jacobian column
u_j = Sigma^{-1/2} dS/dz_j at the reference point (A = 1, r = d = m = 0), which the
generator's coefficient map gives in closed form.

Per (cell, seed): recovery ||P u_j||^2 / ||u_j||^2 for every j; the population
eigenvalue of the whitened centred clean covariance along u_j (Rayleigh quotient, +1 for
the noise) and its gap to the fifth population eigenvalue, from 20 000 clean events at
the cell's excitation (no fits); the linear-probe RMSE of latent j on the evaluation set
relative to its prior sd; across seeds, the variance of recovery.

Cells (19): each direction alone at s in {0, 0.03, 0.1, 0.3} with the others at 1,
plus all-at-1 (control), all-at-0.1, and the control at N = 3200 (added at freeze after the
unit check; see cells()).  Seeds 0-9; 190 fits (cap 200).

Usage::

    python -m experiments.paper1.t21_excitation
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import generator as gen
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.covariance import _sqrt_inv, full_covariance, whiten_separable

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results" / "mlst_2026-09-11" / "t21"

def fast_whiten(X: np.ndarray, Sigma_c: np.ndarray, Sigma_t: np.ndarray) -> np.ndarray:
    """BLAS form of ``whiten_separable`` (same symmetric roots); pinned to agree with it."""
    Wc = _sqrt_inv(Sigma_c); Wt = _sqrt_inv(Sigma_t)
    return np.matmul(np.matmul(Wc, X), Wt)

DIRECTIONS = ["amplitude", "rise_time", "decay_time", "secondary"]
GRID = [0.0, 0.03, 0.1, 0.3]
SEEDS = list(range(10))
RANK = 4
N_TRAIN, N_EVAL, N_POP = 800, 2000, 20000
FIT_CAP = 200
PRIOR_SD = {"amplitude": 0.2, "rise_time": 0.15, "decay_time": 0.15, "secondary": 0.5}


def cells() -> list[dict]:
    out = [{"id": f"{d}-s{s}", "scales": {k: (s if k == d else 1.0) for k in DIRECTIONS}} for d in DIRECTIONS for s in GRID]
    out.append({"id": "all-1", "scales": {k: 1.0 for k in DIRECTIONS}})
    out.append({"id": "all-0.1", "scales": {k: 0.1 for k in DIRECTIONS}})
    # added at freeze after the unit check (before any run): the control at N = 3200 (d/N = 0.64),
    # because at N = 800 (d/N = 2.56) two of the four directions sit near the BBP threshold even
    # when fully excited, so the registered "strongly excited" regime needs a larger N to exist.
    out.append({"id": "all-1-N3200", "scales": {k: 1.0 for k in DIRECTIONS}, "n_train": 3200})
    assert len(out) == 19 and len({c["id"] for c in out}) == 19
    return out


@contextlib.contextmanager
def scaled_latents(scales: dict):
    original = gen.sample_latents

    def patched(rng, n):
        lat = original(rng, n)
        lat["amplitude"] = 1.0 + scales["amplitude"] * (lat["amplitude"] - 1.0)
        for k in ("rise_time", "decay_time", "secondary"):
            lat[k] = scales[k] * lat[k]
        return lat

    gen.sample_latents = patched
    try:
        yield
    finally:
        gen.sample_latents = original


def jacobian_directions(cfg, V, Tb, cov, snr_scale: float) -> dict:
    """Whitened dS/dz_j at (A=1, r=d=m=0) from build_coefficients: S = sum z_ab v_a t_b^T."""
    dirs = {}
    base = {"amplitude": (1.0, 0.0, 0.0, 0.0), "rise_time": (0, 1, 0, 0), "decay_time": (0, 0, 1, 0), "secondary": (0, 0, 0, 1)}
    for name, (dA, dr, dd, dm) in base.items():
        z = np.zeros((cfg.k_c, cfg.k_t))
        # derivative of the coefficient map at A=1, r=d=m=0
        z[0, 0] = dA; z[0, 1] = dr; z[0, 2] = dd; z[1, 0] = dm; z[1, 1] = 0.25 * dr; z[1, 2] = 0.25 * dd
        S = np.einsum("ab,ia,jb->ij", z, V, Tb)[None] * snr_scale
        u = whiten_separable(S, cov["Sigma_c"], cov["Sigma_t"])[0].reshape(-1, order="F")
        dirs[name] = u
    return dirs


def make_cell_data(cell: dict, seed: int):
    n_train = int(cell.get("n_train", N_TRAIN))
    cfg = replace(PRIMARY, nonseparability_alpha=0.0, N_train=n_train, N_test=N_EVAL)
    V = gen.make_channel_basis(cfg); Tb = gen.make_temporal_basis(cfg)
    cov = full_covariance(cfg, "colored")
    rng = np.random.default_rng(seed)
    with scaled_latents(cell["scales"]):
        clean_tr, lat_tr = gen.generate_clean_events(rng, n_train, cfg, V, Tb)
    clean_ev, lat_ev = gen.generate_clean_events(rng, N_EVAL, cfg, V, Tb)          # evaluation: full excitation
    # one SNR scale for the whole cell, set on a full-excitation reference so cells are comparable
    ref, _ = gen.generate_clean_events(np.random.default_rng(10_000 + seed), 2000, cfg, V, Tb)
    _, snr = gen.snr_normalize(ref, cov, cfg)
    Zref = fast_whiten(ref, cov["Sigma_c"], cov["Sigma_t"])
    scale = cfg.target_snr / np.sqrt(np.mean(np.sum(Zref.reshape(len(ref), -1) ** 2, axis=1)))
    clean_tr *= scale; clean_ev *= scale
    X_tr = clean_tr + gen.generate_noise(rng, n_train, cfg, cov)
    X_ev = clean_ev + gen.generate_noise(rng, N_EVAL, cfg, cov)
    dirs = jacobian_directions(cfg, V, Tb, cov, scale)
    return cfg, cov, X_tr, lat_tr, X_ev, lat_ev, dirs, float(scale)


def population_spectrum(cell: dict, dirs: dict, scale: float, seed: int = 20260911) -> dict:
    cfg = replace(PRIMARY, nonseparability_alpha=0.0)
    V = gen.make_channel_basis(cfg); Tb = gen.make_temporal_basis(cfg)
    cov = full_covariance(cfg, "colored")
    rng = np.random.default_rng(seed)
    with scaled_latents(cell["scales"]):
        clean, _ = gen.generate_clean_events(rng, N_POP, cfg, V, Tb)
    clean *= scale
    Z = fast_whiten(clean, cov["Sigma_c"], cov["Sigma_t"]).reshape(N_POP, -1, order="F")
    Z = Z - Z.mean(axis=0, keepdims=True)
    # top eigenvalues via the CT x CT covariance
    Cpop = Z.T @ Z / N_POP
    w = np.sort(np.linalg.eigvalsh(Cpop))[::-1] + 1.0
    out = {"eigenvalues_top6": [float(x) for x in w[:6]], "lambda_5": float(w[RANK]),
           "n_signal_eigenvalues_above_floor": int(np.sum(w[:RANK + 2] > 1.05))}
    # At alpha = 0 the discarded population eigenvalues sit exactly on the noise floor (1.0),
    # so "gap to lambda_5" is degenerate.  The registered ratio is therefore defined as the
    # excess over the floor relative to the BBP detection threshold sqrt(d/N) of the fit:
    # excess/threshold < 1 means the direction is undetectable at that N even when excited.
    for name, u in dirs.items():
        un = u / np.linalg.norm(u)
        lam = float(un @ Cpop @ un) + 1.0
        out[name] = {"lambda_along_direction": lam, "excess_over_floor": lam - 1.0,
                     "gap_to_lambda5": lam - float(w[RANK]),
                     "ratio_excess_over_bbp_N800": (lam - 1.0) / np.sqrt(Cpop.shape[0] / N_TRAIN),
                     "ratio_excess_over_bbp_N3200": (lam - 1.0) / np.sqrt(Cpop.shape[0] / 3200)}
    return out


def fit_and_score(cfg, cov, X_tr, lat_tr, X_ev, lat_ev, dirs) -> dict:
    n_train = X_tr.shape[0]
    mu = X_tr.mean(axis=0, keepdims=True)
    Z_tr = fast_whiten(X_tr - mu, cov["Sigma_c"], cov["Sigma_t"]).reshape(n_train, -1, order="F")
    Z_ev = fast_whiten(X_ev - mu, cov["Sigma_c"], cov["Sigma_t"]).reshape(N_EVAL, -1, order="F")
    _, _, vh = np.linalg.svd(Z_tr, full_matrices=False)
    P = vh[:RANK].T                                                # (CT, 4) orthonormal
    rec = {name: float(np.sum((P.T @ u) ** 2) / np.sum(u ** 2)) for name, u in dirs.items()}
    rec_meta = {"d_over_N": P.shape[0] / n_train, "bbp_threshold": float(np.sqrt(P.shape[0] / n_train))}
    C_tr, C_ev = Z_tr @ P, Z_ev @ P
    probes = {}
    Xd = np.column_stack([C_tr, np.ones(n_train)]); Xe = np.column_stack([C_ev, np.ones(N_EVAL)])
    for name in DIRECTIONS:
        y_tr, y_ev = lat_tr[name], lat_ev[name]
        beta, *_ = np.linalg.lstsq(Xd, y_tr, rcond=None)
        pred = Xe @ beta
        rmse = float(np.sqrt(np.mean((y_ev - pred) ** 2)))
        probes[name] = {"rmse": rmse, "rmse_over_prior_sd": rmse / float(np.std(y_ev)),
                        "r2": float(1 - np.sum((y_ev - pred) ** 2) / np.sum((y_ev - y_ev.mean()) ** 2))}
    return {"recovery": rec, "probe": probes, "chance_level": RANK / P.shape[0], **rec_meta}


def provenance(argv):
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO_ROOT, text=True).strip()
        except Exception:
            return "unavailable"
    return {"code_commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "tree_dirty": bool(git("status", "--porcelain")),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_config_hash": config_hash(PRIMARY), "python": sys.version.split()[0], "numpy": np.__version__,
            "platform": platform.platform(), "hostname": socket.gethostname(), "command": argv,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def summarise(rows: list[dict], pops: dict) -> dict:
    out: dict = {"cells": {}}
    for cid in sorted({r["cell"] for r in rows}):
        rc = [r for r in rows if r["cell"] == cid]
        entry = {"n": len(rc), "directions": {}}
        for name in DIRECTIONS:
            rec = np.array([r["recovery"][name] for r in rc])
            entry["directions"][name] = {
                "recovery_mean": float(rec.mean()), "recovery_var": float(rec.var(ddof=1)), "recovery_min": float(rec.min()),
                "probe_rmse_over_prior_median": float(np.median([r["probe"][name]["rmse_over_prior_sd"] for r in rc])),
                "probe_r2_median": float(np.median([r["probe"][name]["r2"] for r in rc])),
                "excitation": rc[0]["scales"][name],
                "population": pops[cid][name],
            }
        out["cells"][cid] = entry
    # prediction (i): recovery as a function of the BBP-normalised excess only -> (ratio, recovery) pairs across cells
    pairs = []
    for cid, e in out["cells"].items():
        for name, dd in e["directions"].items():
            key = "ratio_excess_over_bbp_N3200" if cid == "all-1-N3200" else "ratio_excess_over_bbp_N800"
            pairs.append({"cell": cid, "direction": name, "excitation": dd["excitation"],
                          "excess_over_floor": dd["population"]["excess_over_floor"],
                          "bbp_ratio": dd["population"][key], "recovery_mean": dd["recovery_mean"], "recovery_var": dd["recovery_var"]})
    out["ratio_recovery_pairs"] = pairs
    # prediction (ii): regimes
    ctrl = out["cells"]["all-1"]["directions"]
    strong_var = float(np.mean([v["recovery_var"] for v in ctrl.values()]))
    regimes = {}
    for cid, e in out["cells"].items():
        for name, dd in e["directions"].items():
            s = dd["excitation"]
            if s == 0.0:
                regimes[f"{cid}:{name}"] = {"regime": "unexcited", "recovery_mean": dd["recovery_mean"],
                                            "at_chance": bool(dd["recovery_mean"] < 20 * rows[0]["chance_level"]),
                                            "probe_rmse_over_prior": dd["probe_rmse_over_prior_median"]}
            elif s < 1.0 and cid not in ("all-0.1",):
                regimes[f"{cid}:{name}"] = {"regime": "weakly-excited-candidate", "recovery_mean": dd["recovery_mean"],
                                            "var_over_strong": dd["recovery_var"] / max(strong_var, 1e-16)}
    out["regimes"] = regimes
    out["strong_excitation_recovery_var"] = strong_var
    ctrl_big = out["cells"].get("all-1-N3200", {"directions": {}})["directions"]
    out["prediction_ii"] = {
        "unexcited_at_chance_all": all(v["at_chance"] for v in regimes.values() if v["regime"] == "unexcited"),
        "strongly_excited_recovery_gt_0.95_all_N800": all(v["recovery_mean"] > 0.95 for v in ctrl.values()),
        "strongly_excited_recovery_gt_0.95_all_N3200": all(v["recovery_mean"] > 0.95 for v in ctrl_big.values()) if ctrl_big else None,
        "control_recovery_N800": {k: v["recovery_mean"] for k, v in ctrl.items()},
        "control_recovery_N3200": {k: v["recovery_mean"] for k, v in ctrl_big.items()},
        "weak_cells_var_gt_10x_strong": {k: bool(v["var_over_strong"] > 10) for k, v in regimes.items() if v["regime"] == "weakly-excited-candidate"},
    }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cs = cells()
    n_fits = len(cs) * len(SEEDS)
    if n_fits > FIT_CAP:
        print(f"refusing: {n_fits} > cap {FIT_CAP}"); return 2
    prov = provenance(sys.argv); t0 = time.time()
    rows: list[dict] = []; pops: dict = {}
    with open(OUT_DIR / "rows.jsonl", "w") as fh:
        for cell in cs:
            for seed in SEEDS:
                cfg, cov, X_tr, lat_tr, X_ev, lat_ev, dirs, scale = make_cell_data(cell, seed)
                if cell["id"] not in pops:
                    pops[cell["id"]] = population_spectrum(cell, dirs, scale)
                r = fit_and_score(cfg, cov, X_tr, lat_tr, X_ev, lat_ev, dirs)
                r.update({"cell": cell["id"], "scales": cell["scales"], "seed": seed, "snr_scale": scale})
                fh.write(json.dumps(r) + "\n"); rows.append(r)
            print(f"{cell['id']:>18s}: recovery " + " ".join(f"{k[:3]}={np.mean([r['recovery'][k] for r in rows if r['cell']==cell['id']]):.3f}" for k in DIRECTIONS) + f"  ({time.time()-t0:.0f}s)", flush=True)
    summ = summarise(rows, pops)
    summ["population_spectra"] = pops
    prov["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); prov["wall_seconds"] = time.time() - t0
    prov["n_fits"] = n_fits
    (OUT_DIR / "summary.json").write_text(json.dumps(summ, indent=1))
    outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT_DIR.iterdir() if p.name != "manifest.json"}
    (OUT_DIR / "manifest.json").write_text(json.dumps({"task": "T21", "provenance": prov, "cells": [c["id"] for c in cs], "seeds": SEEDS, "outputs": outputs}, indent=1))
    print(json.dumps(summ["prediction_ii"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
