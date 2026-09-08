"""P1-E6 — rank-one corollary check at C = 8 (Paper 1, Corollary 6.2).

Separable whitened template s = v t^T, isotropic whitened noise, k_c = k_t = 1.
Population NFPA must learn v/|v| and t/|t| up to sign and its scalar code must
be proportional to the joint-channel optimal-filter amplitude.

Task card E6* runs the population point (n = 20 000) at seeds 0-9 and, in
addition, an n-curve at n in {1000, 3000, 10000, 20000} on three seeds, so the
finite-n approach of |cos theta_t| to 1 is a curve rather than a single point.
If the temporal cosine plateaus below 0.999 the exponent beta of
``1 - |cos theta_t| ~ n^{-beta}`` is fitted and reported: that, not the raw
threshold, is the finite-sample statement Cor. 6.3 needs.

The run is resumable: one JSON line per (seed, n).

Usage::

    python -m experiments.paper1.rank_one_check --seeds 0 1 2 3 4 5 6 7 8 9 --n 20000
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY
from experiments.synthetic.nfpa_confirmatory.methods import als_fit

from . import common

EXPERIMENT_ID = "P1-E6"

COLUMNS = [
    "seed", "n", "C", "T", "snr",
    "abs_cos_channel", "abs_cos_temporal",
    # task-card names
    "amp_corr_code_vs_of", "amp_corr_code_vs_truth",
    # pre-existing names kept (ground rule 4: add columns, never rename)
    "amplitude_correlation_code_vs_of", "amplitude_correlation_code_vs_truth",
    "converged", "arm", "walltime_s",
]


def one_seed(seed: int, n: int, C: int, T: int, snr: float, *, chunk: int = 5000) -> dict:
    """One (seed, n) point. The code/OF projections are chunked over events."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(C); v /= np.linalg.norm(v)
    t = np.exp(-np.arange(T) / (0.1 * T)) * (1 - np.exp(-np.arange(T) / (0.02 * T))); t /= np.linalg.norm(t)
    A = rng.uniform(0.5, 1.5, size=n) * snr
    Z = A[:, None, None] * np.outer(v, t)[None] + rng.standard_normal((n, C, T))
    Z -= Z.mean(axis=0)
    out = als_fit(Z, 1, 1, n_iter=100, n_restarts=5, seed=seed)
    u_c, u_t = out["U_c"][:, 0], out["U_t"][:, 0]
    code = np.empty(n); of = np.empty(n)
    for i in range(0, n, chunk):
        block = Z[i:i + chunk]
        code[i:i + chunk] = np.einsum("nct,c,t->n", block, u_c, u_t)
        of[i:i + chunk] = np.einsum("nct,c,t->n", block, v, t)
    r_of = float(abs(np.corrcoef(code, of)[0, 1]))
    r_truth = float(abs(np.corrcoef(code, A - A.mean())[0, 1]))
    return {
        "seed": int(seed), "n": int(n), "C": int(C), "T": int(T), "snr": float(snr),
        "abs_cos_channel": float(abs(u_c @ v)), "abs_cos_temporal": float(abs(u_t @ t)),
        "amp_corr_code_vs_of": r_of, "amp_corr_code_vs_truth": r_truth,
        "amplitude_correlation_code_vs_of": r_of,
        "amplitude_correlation_code_vs_truth": r_truth,
        "converged": bool(out["converged"]),
        "walltime_s": time.time() - t0,
    }


def _fit_beta(ns: np.ndarray, gaps: np.ndarray) -> dict:
    """Least-squares fit of ``log(1 - |cos|) = a - beta log n``."""
    ok = np.isfinite(gaps) & (gaps > 0)
    if int(ok.sum()) < 2:
        return {"beta": None, "intercept": None, "n_points": int(ok.sum()),
                "note": "not enough positive gaps to fit an exponent"}
    A = np.column_stack([np.ones(int(ok.sum())), np.log(ns[ok])])
    coef, *_ = np.linalg.lstsq(A, np.log(gaps[ok]), rcond=None)
    return {"beta": float(-coef[1]), "intercept": float(coef[0]), "n_points": int(ok.sum())}


def _load_done(out: Path) -> dict:
    done: dict = {}
    for path in sorted(out.glob("points*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(int(rec["seed"]), int(rec["n"]))] = rec
    return done


def run(seeds: list[int], n: int, snr: float, curve_ns: list[int], curve_seeds: list[int]) -> Path:
    out = common.out_dir_for(EXPERIMENT_ID)
    jsonl = out / "points.jsonl"
    done = _load_done(out)
    t_start = time.time()

    plan = [(s, n, "population") for s in seeds]
    plan += [(s, m, "n_curve") for m in curve_ns for s in curve_seeds]

    for seed, n_i, arm in plan:
        if (seed, n_i) in done:
            done[(seed, n_i)].setdefault("arm", arm)
            continue
        row = one_seed(seed, n_i, PRIMARY.C, PRIMARY.T, snr)
        row["arm"] = arm
        done[(seed, n_i)] = row
        with jsonl.open("a") as fh:
            fh.write(json.dumps(common.as_jsonable(row)) + "\n")
        print(f"[{EXPERIMENT_ID}] seed={seed} n={n_i} ({arm}) "
              f"cos_t={row['abs_cos_temporal']:.6f} in {row['walltime_s']:.1f}s", flush=True)

    rows = [done[k] for k in sorted(done)]
    rc.write_rows_csv(out / "rank_one_check.csv", rows, COLUMNS)

    pop = [r for r in rows if r["n"] == n and r["seed"] in seeds]
    summary: dict = {
        "n_population": n, "seeds": seeds, "snr": snr,
        "curve_ns": curve_ns, "curve_seeds": curve_seeds,
    }
    for key in ("abs_cos_channel", "abs_cos_temporal",
                "amp_corr_code_vs_of", "amp_corr_code_vs_truth"):
        vals = [r[key] for r in pop]
        summary[key] = {"min": float(np.min(vals)), "mean": float(np.mean(vals)),
                        "max": float(np.max(vals))}
    summary["accept"] = bool(summary["abs_cos_channel"]["min"] >= 0.999
                             and summary["abs_cos_temporal"]["min"] >= 0.999
                             and summary["amp_corr_code_vs_of"]["min"] >= 0.999)
    summary["predictions"] = {
        "abs_cos_channel_min_ge_0.999": {"value": summary["abs_cos_channel"]["min"],
                                         "ok": bool(summary["abs_cos_channel"]["min"] >= 0.999)},
        "abs_cos_temporal_min_ge_0.999": {"value": summary["abs_cos_temporal"]["min"],
                                          "ok": bool(summary["abs_cos_temporal"]["min"] >= 0.999)},
        "amp_corr_code_vs_of_min_ge_0.999": {"value": summary["amp_corr_code_vs_of"]["min"],
                                             "ok": bool(summary["amp_corr_code_vs_of"]["min"] >= 0.999)},
    }

    curve = {}
    ns = np.array(sorted({r["n"] for r in rows if r["seed"] in curve_seeds}), dtype=float)
    for key in ("abs_cos_temporal", "abs_cos_channel"):
        means = np.array([np.mean([r[key] for r in rows
                                   if r["n"] == int(m) and r["seed"] in curve_seeds]) for m in ns])
        curve[key] = {"n": ns.tolist(), "mean": means.tolist(),
                      "gap_mean": (1.0 - means).tolist(),
                      "power_law_fit": _fit_beta(ns, 1.0 - means)}
    summary["finite_n_curve"] = curve
    summary["temporal_plateau_below_0.999"] = bool(summary["abs_cos_temporal"]["min"] < 0.999)

    rc.write_json(out / "summary.json", common.as_jsonable(summary))
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=[PRIMARY], seeds=seeds,
                          extra={"n_events": n, "snr": snr, "curve_ns": curve_ns,
                                 "curve_seeds": curve_seeds,
                                 "walltime_s": time.time() - t_start,
                                 "host": platform.node(), "platform": platform.platform()})
    print(json.dumps(common.as_jsonable(summary["predictions"]), indent=1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--curve-ns", type=int, nargs="+", default=[1000, 3000, 10000, 20000])
    ap.add_argument("--curve-seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(argv)
    print(f"wrote {run(a.seeds, a.n, a.snr, a.curve_ns, a.curve_seeds)}")


if __name__ == "__main__":
    main()
