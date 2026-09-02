"""P1-E6 — rank-one corollary check at C = 8 (Paper 1, Corollary 6.2).

Separable whitened template s = v t^T, isotropic whitened noise, k_c = k_t = 1.
Population NFPA must learn v/|v| and t/|t| up to sign and its scalar code must
be proportional to the joint-channel optimal-filter amplitude.

Usage::  python -m experiments.paper1.rank_one_check --seeds 0 1 2 3 4 5 6 7 8 9 [--n 20000]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.synthetic.nfpa_confirmatory import r9_common as rc
from experiments.synthetic.nfpa_confirmatory.config import PRIMARY, config_hash
from experiments.synthetic.nfpa_confirmatory.methods import als_fit

from . import common

EXPERIMENT_ID = "P1-E6"


def one_seed(seed: int, n: int, C: int, T: int, snr: float) -> dict:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(C); v /= np.linalg.norm(v)
    t = np.exp(-np.arange(T) / (0.1 * T)) * (1 - np.exp(-np.arange(T) / (0.02 * T))); t /= np.linalg.norm(t)
    A = rng.uniform(0.5, 1.5, size=n) * snr
    Z = A[:, None, None] * np.outer(v, t)[None] + rng.standard_normal((n, C, T))
    Z -= Z.mean(axis=0)
    out = als_fit(Z, 1, 1, n_iter=100, n_restarts=5, seed=seed)
    u_c, u_t = out["U_c"][:, 0], out["U_t"][:, 0]
    code = np.einsum("nct,c,t->n", Z, u_c, u_t)
    of = np.einsum("nct,c,t->n", Z, v, t)  # whitened matched filter (OF amplitude up to scale)
    return {
        "seed": seed, "n": n, "C": C, "T": T,
        "abs_cos_channel": float(abs(u_c @ v)), "abs_cos_temporal": float(abs(u_t @ t)),
        "amplitude_correlation_code_vs_of": float(abs(np.corrcoef(code, of)[0, 1])),
        "amplitude_correlation_code_vs_truth": float(abs(np.corrcoef(code, A - A.mean())[0, 1])),
        "converged": bool(out["converged"]),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--snr", type=float, default=10.0)
    a = ap.parse_args(argv)
    out = common.out_dir_for(EXPERIMENT_ID)
    rows = [one_seed(s, a.n, PRIMARY.C, PRIMARY.T, a.snr) for s in a.seeds]
    rc.write_rows_csv(out / "rank_one_check.csv", rows, list(rows[0].keys()))
    summary = {k: {"min": float(min(r[k] for r in rows)), "mean": float(np.mean([r[k] for r in rows]))}
               for k in ("abs_cos_channel", "abs_cos_temporal", "amplitude_correlation_code_vs_of", "amplitude_correlation_code_vs_truth")}
    summary["accept"] = bool(summary["abs_cos_channel"]["min"] >= 0.999 and summary["abs_cos_temporal"]["min"] >= 0.999
                             and summary["amplitude_correlation_code_vs_of"]["min"] >= 0.999)
    rc.write_json(out / "summary.json", summary)
    common.write_manifest(out, experiment_id=EXPERIMENT_ID, cfgs=[PRIMARY], seeds=a.seeds, extra={"n_events": a.n, "snr": a.snr})
    print(summary)


if __name__ == "__main__":
    main()
